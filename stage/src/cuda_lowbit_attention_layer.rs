//! One resident CUDA HC -> full QSA attention -> HC -> original512 MoE layer.
//! Diagnostic composition; packed weights stay resident, with no host handoffs.
pub use crate::cuda_lowbit_layer::{Expert, Modules};
use crate::{
    cuda::{Buffer, Device, Graph, Module, PreparedCall},
    cuda_hc, cuda_linear, cuda_lowbit_attention, cuda_lowbit_moe, cuda_router, cuda_shared, hc,
    lowbit_attention_layer::Weights,
    lowbit_router::Scores,
    lowbit_shared::Arithmetic,
    mcg_moe::Bank,
    qsa, target_attention as a,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, time::Instant};
const GUARD: usize = 256;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub hc_split: bool,
    pub separate_write: bool,
    pub attention_parts: usize,
    pub wide: bool,
    pub expert: Expert,
}
impl Policy {
    pub fn hc(self) -> cuda_hc::Policy {
        cuda_hc::Policy {
            split_down: self.hc_split,
            separate_write: self.separate_write,
        }
    }
    pub fn attention(self) -> cuda_lowbit_attention::Policy {
        cuda_lowbit_attention::Policy {
            splits: self.attention_parts,
            projection: if crate::cuda_affine4::rows_arm() { 4 } else if self.wide { 3 } else { 0 },
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
/// R69: a slot's forward bookkeeping.
#[derive(Default)]
struct Fwd {
    rows: Cell<usize>,
    valid: Cell<Option<(u64, usize)>>,
    epoch: Cell<u64>,
}
pub struct Layer<'m, 'd> {
    d: &'d Device,
    hc: [cuda_hc::Hc<'m, 'd>; 2],
    attention: cuda_lowbit_attention::Attention<'m, 'd>,
    mlp: cuda_lowbit_moe::Engine<'m, 'd>,
    input: Buffer<'d>,
    output: Buffer<'d>,
    /// R69: per conversation slot: the last forward's rows, recorded epoch/rows and invalidation epoch
    fwd: Vec<Fwd>,
    active: Cell<usize>,
    layer: usize,
}
impl<'m, 'd> Layer<'m, 'd> {
    fn f(&self) -> &Fwd {
        &self.fwd[self.active.get()]
    }
    /// R69: add a conversation slot whose KV views the top `rows` rows of slot 0's buffers.
    pub(crate) fn add_slot_view(&mut self, rows: usize) -> Result<usize> {
        let slot = self.attention.add_slot_view(rows)?;
        self.fwd.push(Fwd::default());
        ensure!(slot + 1 == self.fwd.len(), "attention layer slot bookkeeping");
        Ok(slot)
    }
    /// R69: the slot every following call works on.
    pub(crate) fn select_slot(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.fwd.len(), "attention layer slot {slot}");
        self.attention.select_slot(slot)?;
        self.active.set(slot);
        Ok(())
    }
    /// R71: re-point slot `slot` (1..) to rows [base, base + rows) of slot 0's buffers (see Attention::set_slot_view).
    pub(crate) fn set_slot_view(&self, slot: usize, base: usize, rows: usize, keep: bool) -> Result<()> {
        self.attention.set_slot_view(slot, base, rows, keep)
    }
    /// R71: rows slot `slot` holds.
    pub(crate) fn live_rows(&self, slot: usize) -> usize {
        self.attention.live_rows(slot)
    }
    /// R71: (base, rows) of slot `slot`'s view.
    pub(crate) fn slot_view(&self, slot: usize) -> (usize, usize) {
        self.attention.slot_view(slot)
    }
    /// R69: the active slot holds nothing.
    pub(crate) fn drop_slot_state(&self) {
        self.attention.drop_slot_state();
        let f = self.f();
        f.rows.set(0);
        f.valid.set(None);
        f.epoch.set(f.epoch.get() + 1);
    }
    /// R53: this layer's MoE trace words (traced variants only).
    pub(crate) fn moe_trace(&self) -> Result<Option<(u32, Vec<u64>)>> {
        self.mlp.r52_trace()
    }
    pub(crate) fn moe_status(&self) -> Result<Option<u32>> {
        self.mlp.r52_status()
    }
    pub fn new(
        d: &'d Device,
        m: &'m Modules<'d>,
        attention_module: &'m Module<'d>,
        w: &Weights,
        bank: &Bank,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(w.layer < 48 && w.layer % 4 == 3, "full attention layer");
        let mut mlp = cuda_lowbit_moe::Engine::new(
            d, &m.router, &m.linear, &m.project, &m.moe, &m.affine, &m.split, &m.shared, &m.join,
            &w.router, bank, &w.shared,
        )?;
        mlp.enable_tensor(&m.tensor)?;
        let mut attention = cuda_lowbit_attention::Attention::new(d, &m.affine, attention_module, &w.attention, capacity)?;
        if let Some(r) = &m.rows {
            attention.enable_rows(r)?;
        }
        Ok(Self {
            d,
            hc: [
                cuda_hc::Hc::new(d, &m.affine, &m.split, &m.hc, &w.hc[0])?,
                cuda_hc::Hc::new(d, &m.affine, &m.split, &m.hc, &w.hc[1])?,
            ],
            attention,
            mlp,
            input: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            output: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            fwd: vec![Fwd::default()],
            active: Cell::new(0),
            layer: w.layer,
        })
    }
    /// R60 pipeline split: one canonical full-attention layer (target weights: affine8 trunk and
    /// QSA index projection, K4 experts, BF16 router) on the RTX 5090.
    pub fn new_canonical(
        d: &'d Device,
        m: &'m Modules<'d>,
        attention_module: &'m Module<'d>,
        m8: &'m Module<'d>,
        w: &crate::target_attention_layer::Weights,
        bank: &Bank,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(w.layer < 48 && w.layer % 4 == 3, "canonical full attention layer");
        let mlp = cuda_lowbit_moe::Engine::new_canonical(
            d, &m.router, &m.linear, &m.project, &m.moe, m8, &m.shared, &m.join, &m.k4, &w.mlp, bank,
        )?;
        let attention = cuda_lowbit_attention::Attention::new_canonical(d, m8, attention_module, &w.attention, &w.index, capacity)?;
        Ok(Self {
            d,
            hc: [
                cuda_hc::Hc::new_canonical(d, m8, &m.hc, &w.hc[0])?,
                cuda_hc::Hc::new_canonical(d, m8, &m.hc, &w.hc[1])?,
            ],
            attention,
            mlp,
            input: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            output: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            fwd: vec![Fwd::default()],
            active: Cell::new(0),
            layer: w.layer,
        })
    }
    pub fn description(&self) -> Value {
        json!({"layer":self.layer,"hc":[self.hc[0].description(),self.hc[1].description()],"attention":self.attention.description(),"mlp":self.mlp.description(),"boundary_bytes":self.input.byte_len()+self.output.byte_len(),"host_intermediate_transfers":0,"scope":"one full attention layer; all512 original expert weights; no embedding/PLE/head/token sampling"})
    }
    pub fn prepare(&self, x: &[u16], cache: &a::State, index: &qsa::State) -> Result<()> {
        self.prepare_state(cache, index)?;
        self.prepare_input(x)
    }
    pub(crate) fn prepare_state(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.prepare_state(cache, index)
    }
    /// R60 S4: prepare_state without poison or read-back (see Attention::prepare_state_lean).
    pub(crate) fn prepare_state_lean(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.prepare_state_lean(cache, index)
    }
    /// R60 S4: rotary offset for the next appended rows (M-RoPE decode delta; 0 for text).
    pub(crate) fn set_rope_delta(&self, delta: i32) {
        self.attention.set_rope_delta(delta);
    }
    /// R59 incremental refresh (see Attention::refresh_rows).
    pub(crate) fn refresh_state(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.refresh_rows(cache, index)
    }
    pub(crate) fn prepare_resident(&self, rows: usize) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.prepare_resident_input(rows)?;
        self.f().rows.set(rows);
        Ok(())
    }
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
    pub(crate) fn output(&self) -> &Buffer<'d> {
        &self.output
    }
    pub fn prepare_input(&self, x: &[u16]) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
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
        self.attention.prepare_resident_input(rows)?;
        self.f().rows.set(rows);
        Ok(())
    }
    pub fn state(&self) -> Result<(a::State, qsa::State)> {
        self.attention.state()
    }
    /// R56 lean drafting commit.
    pub(crate) fn record_commit(&self, accepted: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.attention.record_commit(accepted)
    }
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        self.attention.state_buffers()
    }
    pub(crate) fn export_state(&self) -> Result<(a::State, qsa::State)> {
        self.attention.export_state()
    }
    /// R66d: the committed open group's raw keys (see Attention::open_keys).
    pub(crate) fn open_keys(&self) -> Result<Vec<u8>> {
        self.attention.open_keys()
    }
    /// R66: committed K codes of rows [from, to) (see Attention::key_codes).
    pub(crate) fn key_codes(&self, from: usize, to: usize) -> Result<Vec<u8>> {
        self.attention.key_codes(from, to)
    }
    /// R60 S4 (texas-split parking): see Attention::snapshot_ranges.
    pub(crate) fn snapshot_ranges(&self) -> Result<(usize, Vec<Option<Buffer<'d>>>)> {
        self.attention.snapshot_ranges()
    }
    pub(crate) fn position_meta(&self) -> Option<(usize, usize)> {
        self.attention.position_meta()
    }
    pub(crate) fn set_position_meta(&self, meta: Option<(usize, usize)>) {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.set_position_meta(meta);
    }
    pub(crate) fn finish_commit(&self, accepted: usize) -> Result<()> {
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.finish_commit(accepted)
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let (epoch, rows) = self
            .f().valid
            .get()
            .ok_or_else(|| anyhow::anyhow!("layer commit requires all stages checked"))?;
        ensure!(
            epoch == self.f().epoch.get() && accepted <= rows,
            "layer unavailable epoch/prefix"
        );
        // Validation errors leave the last completed result usable. Once the
        // state mutation starts, any launch/sync failure invalidates old plans.
        self.f().rows.set(0);
        self.f().valid.set(None);
        self.f().epoch.set(self.f().epoch.get() + 1);
        self.attention.commit(accepted)
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
            rows == self.f().rows.get() && rows > 0 && !input.aliases(&self.output),
            "attention layer resident readiness/alias"
        );
        let pre = &self.hc[0];
        let post = &self.hc[1];
        Ok(vec![
            pre.record_from(input, None, rows, p.hc())?,
            self.attention
                .record_from(pre.mixed(), rows, p.attention())?,
            post.record_from(
                input,
                Some((self.attention.output(), pre.injection())),
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
        self.f().valid.set(None);
        self.hc[0].poison(p.hc())?;
        self.attention.poison()?;
        self.hc[1].poison(p.hc())?;
        self.mlp.poison(p.shared())?;
        self.output.upload(&vec![255; self.output.byte_len()])
    }
    pub(crate) fn collect_from(&self, input: &Buffer<'d>, rows: usize, p: Policy) -> Result<Run> {
        ensure!(
            rows == self.f().rows.get() && rows > 0,
            "attention layer collect readiness"
        );
        let pre = self.hc[0].collect_from(input, rows, false, p.hc())?;
        let attention = self.attention.collect(rows, p.attention())?;
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
        self.f().valid.set(Some((self.f().epoch.get(), rows)));
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
        let rows = self.f().rows.get();
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
            exec,
            rows,
            p,
            repetitions,
            calls,
            epoch: self.f().epoch.get(),
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Groups(Vec<Vec<PreparedCall<'a, 'm, 'd>>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Layer<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    p: Policy,
    pub repetitions: usize,
    pub calls: usize,
    epoch: u64,
}
pub struct Run {
    pub pre: cuda_hc::Run,
    pub attention: cuda_lowbit_attention::Run,
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
            a.f().rows.get() == self.rows && a.f().epoch.get() == self.epoch,
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

impl<'m, 'd> Layer<'m, 'd> {
    /// R60 S4 (prefill v2): this attention layer over chunk rows (as the recurrent layer, with the
    /// attention block in row sub-batches of `s.attn`). The caller advances the attention position
    /// with `finish_prefill` once every call is enqueued.
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a crate::cuda_lowbit_layer::PrefillScratch<'m, 'd>,
        views: &'a [Vec<Buffer<'d>>],
        sub: usize,
        input: &'a Buffer<'d>,
        output: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && !input.aliases(output) && views.len() == rows.div_ceil(sub), "attention layer prefill");
        let (pre, post) = (&self.hc[0], &self.hc[1]);
        let mut calls = pre.record_prefill(&s.hc, input, None, rows)?;
        calls.extend(self.attention.record_prefill(&s.attn, s.hc.mixed(), rows)?);
        for (j, v) in views.iter().enumerate() {
            let (r0, r1) = (j * sub, ((j + 1) * sub).min(rows));
            calls.extend(self.attention.record_prefill_attend(&s.attn, v, r0, r1)?);
        }
        calls.extend(self.attention.record_prefill_output(&s.attn, rows)?);
        calls.extend(post.record_prefill(&s.hc, input, Some((s.attn.output(), s.hc.injection())), rows)?);
        calls.extend(self.mlp.record_prefill(&s.moe, &s.k64, s.hc.mixed(), rows)?);
        calls.push(post.record_write_prefill(s.hc.updated_stream(), s.moe.output(), s.hc.injection(), output, rows)?);
        calls.extend(self.attention.record_commit_prefill(&s.attn, rows)?);
        Ok(calls)
    }
    pub(crate) fn finish_prefill(&self, rows: usize) -> Result<()> {
        self.attention.finish_prefill(rows)
    }
    pub(crate) fn projection_widths(&self) -> [usize; 5] {
        self.attention.projection_widths()
    }
}

impl<'m, 'd> Layer<'m, 'd> {
    pub(crate) fn export_prefill_rows(&self, from: usize, to: usize, out: &mut Vec<u8>) -> Result<()> {
        self.attention.export_prefill_rows(from, to, out)
    }
}
