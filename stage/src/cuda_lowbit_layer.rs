//! One complete resident CUDA HC -> GDN -> HC -> original512 MoE -> residual.
//! A diagnostic gate, not a model generator. No CPU route/activation handoff.
use crate::{
    cuda::{Buffer, Device, Graph, Module, PreparedCall},
    cuda_affine4, cuda_affine4_split, cuda_gdn, cuda_hc, cuda_linear, cuda_lowbit_moe,
    cuda_mcg_moe, cuda_mcg_tensor, cuda_router, cuda_shared, gdn, hc,
    lowbit_layer::Weights,
    lowbit_router::Scores,
    lowbit_shared::Arithmetic,
    mcg_moe::Bank,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, time::Instant};
const GUARD: usize = 256;
pub struct Modules<'d> {
    pub(crate) affine: Module<'d>,
    pub(crate) split: Module<'d>,
    pub(crate) hc: Module<'d>,
    gdn: Module<'d>,
    pub(crate) router: Module<'d>,
    pub(crate) linear: Module<'d>,
    pub(crate) project: Module<'d>,
    pub(crate) moe: Module<'d>,
    pub(crate) shared: Module<'d>,
    pub(crate) join: Module<'d>,
    pub(crate) tensor: Module<'d>,
    /// R60 S4: the K4 tensor/decode module (mcg4_tensor: grouped prefill, load-ahead mcg_project4v).
    pub(crate) k4: Module<'d>,
    /// R58: the multi-row MMA projection module (batched prefill), when built.
    pub(crate) rows: Option<Module<'d>>,
}
impl<'d> Modules<'d> {
    pub fn load(d: &'d Device) -> Result<Self> {
        let (project, moe) = cuda_mcg_moe::modules(d)?;
        Ok(Self {
            affine: cuda_affine4::module(d)?,
            split: cuda_affine4_split::module(d)?,
            hc: cuda_hc::module(d)?,
            gdn: cuda_gdn::module(d)?,
            router: cuda_router::module(d)?,
            linear: cuda_linear::load_module(d)?,
            project,
            moe,
            shared: cuda_shared::module(d)?,
            join: cuda_lowbit_moe::module(d)?,
            tensor: cuda_mcg_tensor::module(d)?,
            k4: cuda_mcg_tensor::module_k4(d)?,
            rows: cuda_affine4::rows_module(d)?,
        })
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Expert {
    Scalar,
    Mma,
    Hybrid,
}
impl Expert {
    pub(crate) fn arm(self) -> Option<cuda_mcg_tensor::Arm> {
        match self {
            Self::Scalar => None,
            Self::Mma => Some(cuda_mcg_tensor::Arm::Mma),
            Self::Hybrid => Some(cuda_mcg_tensor::Arm::Hybrid),
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub hc_split: bool,
    pub separate_write: bool,
    pub gdn_fused: bool,
    pub all_states: bool,
    pub wide: bool,
    pub expert: Expert,
}
impl Policy {
    /// R60: the canonical layers' baseline kernels (affine8 has no split-K or MMA variants; the
    /// K4 experts take the scalar MCG projection path).
    pub fn canonical() -> Self {
        Self { hc_split: true, separate_write: false, gdn_fused: true, all_states: false, wide: false, expert: Expert::Scalar }
    }
    pub fn hc(self) -> cuda_hc::Policy {
        cuda_hc::Policy {
            split_down: self.hc_split,
            separate_write: self.separate_write,
        }
    }
    pub fn gdn(self) -> cuda_gdn::Policy {
        cuda_gdn::Policy {
            fused: self.gdn_fused,
            all_states: self.all_states,
            projection: if cuda_affine4::rows_arm() { 4 } else if self.wide { 3 } else { 0 },
        }
    }
    pub fn shared(self) -> cuda_shared::Policy {
        cuda_shared::Policy {
            combined: true,
            projection: cuda_shared::Projection::Split {
                width: if self.wide { 4 } else { 1 },
                splits: 4,
            },
            down: cuda_shared::Projection::Original(if self.wide { 3 } else { 0 }),
            arithmetic: Arithmetic::StagedBf16,
        }
    }
    fn router(self) -> cuda_router::Projection {
        if self.wide {
            cuda_router::Projection::Tensor(cuda_linear::Arm::Split(16))
        } else {
            cuda_router::Projection::Qmv
        }
    }
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn read(b: &Buffer<'_>, n: usize) -> Result<Vec<u16>> {
    ensure!(n * 2 + GUARD <= b.byte_len(), "layer read bound");
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(raw[n * 2..].iter().all(|&v| v == 255), "layer tail canary");
    let v: Vec<_> = raw[..n * 2]
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        v.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "layer finite/fully written"
    );
    Ok(v)
}
pub struct Layer<'m, 'd> {
    d: &'d Device,
    hc: [cuda_hc::Hc<'m, 'd>; 2],
    gdn: cuda_gdn::Gdn<'m, 'd>,
    mlp: cuda_lowbit_moe::Engine<'m, 'd>,
    input: Buffer<'d>,
    output: Buffer<'d>,
    rows: Cell<usize>,
    epoch: Cell<u64>,
    valid: Cell<Option<(usize, bool)>>,
    layer: usize,
}
impl<'m, 'd> Layer<'m, 'd> {
    pub fn new(d: &'d Device, m: &'m Modules<'d>, w: &Weights, bank: &Bank) -> Result<Self> {
        Self::new_with_prefix_capacity(d, m, w, bank, gdn::MAX_T)
    }
    pub fn new_with_prefix_capacity(
        d: &'d Device,
        m: &'m Modules<'d>,
        w: &Weights,
        bank: &Bank,
        prefix_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            w.layer < 48 && w.layer % 4 != 3 && w.layer != 1,
            "non-PLE recurrent layer"
        );
        Self::build(d, m, w, bank, prefix_capacity)
    }
    /// R60 pipeline split: one canonical GDN layer (target weights: affine8 trunk, K4 experts,
    /// BF16 router) on the RTX 5090. Record with `Policy::canonical()` (baseline kernels).
    pub fn new_canonical(
        d: &'d Device,
        m: &'m Modules<'d>,
        m8: &'m Module<'d>,
        w: &crate::target_layer::Weights,
        bank: &Bank,
        prefix_capacity: usize,
    ) -> Result<Self> {
        ensure!(w.layer < 48 && w.layer % 4 != 3 && w.layer != 1, "canonical non-PLE recurrent layer");
        ensure!((1..=gdn::MAX_T).contains(&prefix_capacity), "layer GDN prefix capacity");
        let mlp = cuda_lowbit_moe::Engine::new_canonical(
            d, &m.router, &m.linear, &m.project, &m.moe, m8, &m.shared, &m.join, &m.k4, &w.mlp, bank,
        )?;
        let gdn = cuda_gdn::Gdn::new_canonical(d, m8, &m.gdn, &w.gdn, prefix_capacity)?;
        Ok(Self {
            d,
            hc: [
                cuda_hc::Hc::new_canonical(d, m8, &m.hc, &w.hc[0])?,
                cuda_hc::Hc::new_canonical(d, m8, &m.hc, &w.hc[1])?,
            ],
            gdn,
            mlp,
            input: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            output: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            rows: Cell::new(0),
            epoch: Cell::new(0),
            valid: Cell::new(None),
            layer: w.layer,
        })
    }
    /// Only the complete PLE wrapper may construct layer1's recurrent body.
    pub(crate) fn after_ple(
        d: &'d Device,
        m: &'m Modules<'d>,
        w: &Weights,
        bank: &Bank,
        prefix_capacity: usize,
    ) -> Result<Self> {
        ensure!(w.layer == 1, "PLE recurrent body layer");
        Self::build(d, m, w, bank, prefix_capacity)
    }
    fn build(
        d: &'d Device,
        m: &'m Modules<'d>,
        w: &Weights,
        bank: &Bank,
        prefix_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=gdn::MAX_T).contains(&prefix_capacity),
            "layer GDN prefix capacity"
        );
        let mut mlp = cuda_lowbit_moe::Engine::new(
            d, &m.router, &m.linear, &m.project, &m.moe, &m.affine, &m.split, &m.shared, &m.join,
            &w.router, bank, &w.shared,
        )?;
        mlp.enable_tensor(&m.tensor)?;
        let pump = mlp.router_weight_region();
        let mut gdn = cuda_gdn::Gdn::new_with_prefix_capacity(d, &m.affine, &m.gdn, &w.gdn, prefix_capacity)?;
        if let Some(r) = &m.rows {
            gdn.enable_rows(r)?;
        }
        let layer = Self {
            d,
            hc: [
                cuda_hc::Hc::new(d, &m.affine, &m.split, &m.hc, &w.hc[0])?,
                cuda_hc::Hc::new(d, &m.affine, &m.split, &m.hc, &w.hc[1])?,
            ],
            gdn,
            mlp,
            input: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            output: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            rows: Cell::new(0),
            epoch: Cell::new(0),
            valid: Cell::new(None),
            layer: w.layer,
        };
        // R52 pumps (candidate arm, QWEN_R52_PUMP=1 loads the pump kernel): the MLP-site
        // front warms the router (QWEN_R52_PUMP_ROUTER=1), the mixer-site front warms the
        // packed qkv weights of this layer's multi4 (QWEN_R52_PUMP_M4=1).
        if std::env::var("QWEN_R52_PUMP_ROUTER").as_deref() == Ok("1") { layer.hc[1].set_pump(pump); }
        if std::env::var("QWEN_R52_PUMP_M4").as_deref() == Ok("1") {
            if let Some(&qkv) = layer.gdn.multi4_regions().first() { layer.hc[0].set_pump(qkv); }
        }
        Ok(layer)
    }
    pub fn description(&self) -> Value {
        json!({"layer":self.layer,"hc":[self.hc[0].description(),self.hc[1].description()],"gdn":self.gdn.description(),"mlp":self.mlp.description(),"boundary_bytes":self.input.byte_len()+self.output.byte_len(),"host_intermediate_transfers":0,"scope":"one non-PLE recurrent layer; all512 original expert weights; no embedding/PLE/QSA/head/token sampling"})
    }
    pub fn prepare(&self, x: &[u16], initial: &gdn::Initial) -> Result<()> {
        self.prepare_initial(initial)?;
        self.prepare_input(x)
    }
    fn invalidate(&self) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.gdn.invalidate_output();
    }
    pub(crate) fn invalidate_output(&self) {
        self.valid.set(None);
        self.gdn.invalidate_output();
    }
    pub(crate) fn prepare_initial(&self, initial: &gdn::Initial) -> Result<()> {
        self.invalidate();
        self.gdn.prepare_initial(initial)
    }
    pub(crate) fn prepare_initial_lean(&self, initial: &gdn::Initial) -> Result<()> {
        self.invalidate();
        self.gdn.prepare_initial_lean(initial)
    }
    pub(crate) fn prepare_resident(&self, rows: usize) -> Result<()> {
        self.invalidate();
        self.gdn.prepare_resident_input(rows)?;
        self.rows.set(rows);
        Ok(())
    }
    pub fn prepare_input(&self, x: &[u16]) -> Result<()> {
        self.invalidate();
        ensure!(
            !x.is_empty()
                && x.len() % hc::F == 0
                && x.len() / hc::F <= hc::MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "layer input shape/finite"
        );
        self.input.upload(&vec![255; self.input.byte_len()])?;
        self.input.upload(bytes(x))?;
        ensure!(read(&self.input, x.len())? == x, "layer input readback");
        let rows = x.len() / hc::F;
        self.gdn.prepare_resident_input(rows)?;
        self.rows.set(rows);
        Ok(())
    }
    pub fn initial(&self) -> Result<gdn::Initial> {
        self.gdn.initial()
    }
    /// R60 S4: the committed state read exactly (see Gdn::initial_lean).
    pub fn initial_lean(&self) -> Result<gdn::Initial> {
        self.gdn.initial_lean()
    }
    /// R52 pump: weights of this layer's first big launch (GDN multi4).
    pub(crate) fn first_projection_regions(&self) -> Vec<(u64, usize)> {
        self.gdn.multi4_regions()
    }
    /// R52 pump: let this layer's MoE warm the given constant regions into L2.
    pub(crate) fn set_mlp_pump(&self, regions: &[(u64, usize)]) -> Result<()> {
        self.mlp.set_pump_table(regions)
    }
    /// R58 sibling screen: this layer's last routed expert ids.
    pub(crate) fn route_ids(&self, rows: usize) -> Result<Vec<u32>> {
        self.mlp.route_ids(rows)
    }
    /// R58 S3: zero this layer's MLP output (before skipping its MLP).
    pub(crate) fn zero_mlp_output(&self) -> Result<()> {
        self.mlp.zero_output()
    }
    /// R53: this layer's MoE trace words (traced variants only).
    pub(crate) fn moe_trace(&self) -> Result<Option<(u32, Vec<u64>)>> {
        self.mlp.r52_trace()
    }
    pub(crate) fn moe_status(&self) -> Result<Option<u32>> {
        self.mlp.r52_status()
    }
    pub(crate) fn output(&self) -> &Buffer<'d> {
        &self.output
    }
    /// R56 lean drafting commit (see cuda_gdn::Gdn::record_commit).
    pub(crate) fn record_commit(&self, accepted: usize, p: Policy) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.gdn.record_commit(accepted, self.rows.get(), p.all_states)
    }
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        self.gdn.state_buffers()
    }
    pub(crate) fn gdn_ref(&self) -> &cuda_gdn::Gdn<'m, 'd> {
        &self.gdn
    }
    /// R56: after a state restore, no forward is pending.
    pub(crate) fn invalidate_pending(&self) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.gdn.finish_commit();
    }
    pub(crate) fn finish_commit(&self) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.gdn.finish_commit();
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let (rows, all) = self
            .valid
            .get()
            .ok_or_else(|| anyhow::anyhow!("layer commit requires all stages checked"))?;
        ensure!(
            accepted <= rows && (all || accepted == 0 || accepted == rows),
            "layer unavailable prefix checkpoint"
        );
        // Validation errors leave the last completed result usable. Once the
        // state mutation starts, any launch/sync failure invalidates old plans.
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        self.gdn.commit(accepted)
    }
    fn record(&self, rows: usize, p: Policy) -> Result<Vec<Vec<PreparedCall<'_, 'm, 'd>>>> {
        self.record_from(&self.input, rows, p)
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<Vec<PreparedCall<'a, 'm, 'd>>>> {
        ensure!(
            rows == self.rows.get() && rows > 0 && !input.aliases(&self.output),
            "layer resident readiness/alias"
        );
        let pre = &self.hc[0];
        let post = &self.hc[1];
        Ok(vec![
            pre.record_from(input, None, rows, p.hc())?,
            self.gdn.record_from(pre.mixed(), rows, p.gdn())?,
            post.record_from(
                input,
                Some((self.gdn.output(), pre.injection())),
                rows,
                p.hc(),
            )?,
            // R58 S3: a skipped MLP records nothing (its zeroed output feeds the HC write)
            if crate::cuda_model::skip_mlp(self.layer) {
                vec![]
            } else {
                self.mlp.record_from(
                    post.mixed(),
                    post.mixed(),
                    rows,
                    p.shared(),
                    p.router(),
                    Scores::StagedBf16,
                    4,
                    p.expert.arm(),
                )?
            },
            vec![post.record_write_from(
                post.updated_stream(),
                self.mlp.output(),
                post.injection(),
                &self.output,
                rows,
                self.mlp.r53_counter_reset(),
            )?],
        ])
    }
    pub(crate) fn poison(&self, p: Policy) -> Result<()> {
        self.invalidate_output();
        self.hc[0].poison(p.hc())?;
        self.gdn.poison()?;
        self.hc[1].poison(p.hc())?;
        self.mlp.poison(p.shared())?;
        self.output.upload(&vec![255; self.output.byte_len()])
    }
    pub(crate) fn collect_from(&self, input: &Buffer<'d>, rows: usize, p: Policy) -> Result<Run> {
        ensure!(
            rows == self.rows.get() && rows > 0,
            "layer collect readiness"
        );
        let pre = self.hc[0].collect_from(input, rows, false, p.hc())?;
        let attention = self.gdn.collect(rows, p.gdn())?;
        let post = self.hc[1].collect_from(input, rows, true, p.hc())?;
        let mlp = self.mlp.collect(rows, p.shared(), 4, p.expert.arm())?;
        let output = read(&self.output, rows * hc::F)?;
        ensure!(
            post.stream == hc::write(&pre.stream, &attention.output, &pre.stages.inject)?,
            "conditional attention residual"
        );
        ensure!(
            output == hc::write(&post.stream, &mlp.output, &post.stages.inject)?,
            "conditional final residual"
        );
        self.valid.set(Some((rows, p.all_states)));
        Ok(Run {
            pre,
            attention,
            post,
            mlp,
            output,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
            phase_us: vec![],
        })
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        let rows = self.rows.get();
        ensure!(
            rows > 0 && (1..=3).contains(&repetitions),
            "layer plan readiness/bounds"
        );
        let c = self.record(rows, p)?;
        let calls = c.iter().map(Vec::len).sum::<usize>();
        ensure!(calls * repetitions <= 128, "layer graph call budget");
        let exec = if graph {
            let mut all = c.into_iter().flatten().collect::<Vec<_>>();
            for _ in 1..repetitions {
                all.extend(self.record(rows, p)?.into_iter().flatten());
            }
            Executor::Graph(self.d.capture(all)?)
        } else {
            Executor::Groups(c)
        };
        Ok(Plan {
            owner: self,
            epoch: self.epoch.get(),
            exec,
            rows,
            p,
            repetitions,
            calls,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Groups(Vec<Vec<PreparedCall<'a, 'm, 'd>>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Layer<'m, 'd>,
    epoch: u64,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    p: Policy,
    pub repetitions: usize,
    pub calls: usize,
}
pub struct Run {
    pub pre: cuda_hc::Run,
    pub attention: cuda_gdn::Run,
    pub post: cuda_hc::Run,
    pub mlp: cuda_lowbit_moe::Run,
    pub output: Vec<u16>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
    pub phase_us: Vec<f64>,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, wait_phase: bool) -> Result<Run> {
        let a = self.owner;
        let p = self.p;
        ensure!(
            a.rows.get() == self.rows && a.epoch.get() == self.epoch,
            "layer plan invalidated"
        );
        ensure!(
            !wait_phase || matches!(self.exec, Executor::Groups(_)),
            "phase waits need prepared calls"
        );
        a.poison(p)?;
        a.d.sync()?;
        let start = Instant::now();
        let mut wait_us = 0.;
        let mut enqueue_us = 0.;
        let mut phase_us = vec![];
        match &self.exec {
            Executor::Graph(g) => {
                let t = Instant::now();
                g.launch()?;
                enqueue_us += t.elapsed().as_secs_f64() * 1e6;
            }
            Executor::Groups(c) => {
                for _ in 0..self.repetitions {
                    for group in c {
                        let phase = Instant::now();
                        let t = Instant::now();
                        for call in group {
                            call.enqueue()?;
                        }
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_phase {
                            let t = Instant::now();
                            a.d.sync()?;
                            wait_us += t.elapsed().as_secs_f64() * 1e6;
                            phase_us.push(phase.elapsed().as_secs_f64() * 1e6);
                        }
                    }
                }
            }
        }
        let t = Instant::now();
        a.d.sync()?;
        wait_us += t.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut out = a.collect_from(&a.input, self.rows, p)?;
        out.total_us = total_us;
        out.enqueue_us = enqueue_us;
        out.wait_us = wait_us;
        out.phase_us = phase_us;
        Ok(out)
    }
}

/// R60 S4 (prefill v2): every chunk-sized intermediate of a stage forward, shared by its layers.
pub struct PrefillScratch<'m, 'd> {
    pub rows: usize,
    pub hc: cuda_hc::HcScratch<'d>,
    pub gdn: cuda_gdn::GdnScratch<'d>,
    pub attn: crate::cuda_lowbit_attention::AttnScratch<'d>,
    pub moe: cuda_lowbit_moe::MoeScratch<'d>,
    pub k64: crate::cuda_mcg_moe::K64<'m, 'd>,
}
impl<'m, 'd> Layer<'m, 'd> {
    /// R60 S4 (prefill v2): this recurrent layer over chunk rows: pre HC, GDN, post HC (the GDN
    /// residual folded into its norm), MoE, the final residual write into `output`, then the GDN
    /// commit (before the next layer reuses the shared scratch).
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a PrefillScratch<'m, 'd>,
        input: &'a Buffer<'d>,
        output: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && !input.aliases(output), "recurrent layer prefill");
        let (pre, post) = (&self.hc[0], &self.hc[1]);
        let mut calls = pre.record_prefill(&s.hc, input, None, rows)?;
        calls.extend(self.gdn.record_prefill(&s.gdn, s.hc.mixed(), rows)?);
        calls.extend(post.record_prefill(&s.hc, input, Some((s.gdn.output(), s.hc.injection())), rows)?);
        calls.extend(self.mlp.record_prefill(&s.moe, &s.k64, s.hc.mixed(), rows)?);
        calls.push(post.record_write_prefill(s.hc.updated_stream(), s.moe.output(), s.hc.injection(), output, rows)?);
        calls.extend(self.gdn.record_commit_prefill(&s.gdn, rows)?);
        Ok(calls)
    }
}
