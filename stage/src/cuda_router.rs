//! Resident full512-expert CUDA routing with explicit projection/score policies.
use crate::{
    bf16_linear::Matrix,
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_linear::{Arm, Linear, Output},
    lowbit_router::{EXPERTS, H, MAX_ROWS, Routing, Scores, TOPK},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "b20417bf9b26ad439481da21eb0620c04e63d5ac9d7e10a249d35c7596687a48";
const GUARD: usize = 256;
/// R38 (Opus, direct): unchanged lowbit_router.cu plus router_qmv_ahead.
pub const R38_SHA: &str = "18c54c67bb345224c071500e0652639cdc95f2a9661fc71af96680f3b2953eda";
pub fn module(d: &Device) -> Result<Module<'_>> {
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_router_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_router.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(x: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
#[derive(Clone, Copy, Debug)]
pub enum Projection {
    Qmv,
    Tensor(Arm),
}
impl Projection {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qmv => "qmv",
            Self::Tensor(a) => a.name(),
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Empty,
    Activations,
    Logits,
}
/// Only the qualified router constructs this descriptor. Calls producing it
/// must precede consumers on the same stream. Invalid numeric rows have bounded
/// unique IDs and NaN scores; reject before committing any model state/output.
pub(crate) struct DeviceRoutes<'a, 'd> {
    pub(crate) ids: &'a Buffer<'d>,
    pub(crate) scores: &'a Buffer<'d>,
    pub(crate) rows: usize,
}
pub struct Router<'m, 'd> {
    d: &'d Device,
    qmv: Kernel<'m, 'd>,
    qmv_ahead: Option<Kernel<'m, 'd>>,
    select: Kernel<'m, 'd>,
    tc: Linear<'m, 'd>,
    w: Buffer<'d>,
    x: Buffer<'d>,
    raw: Buffer<'d>,
    logits: Buffer<'d>,
    ids: Buffer<'d>,
    scores: Buffer<'d>,
    rows: Cell<usize>,
    input: Cell<Input>,
}
impl<'m, 'd> Router<'m, 'd> {
    pub fn new(
        d: &'d Device,
        ops: &'m Module<'d>,
        tensor: &'m Module<'d>,
        weight: &Matrix,
    ) -> Result<Self> {
        weight.validate()?;
        ensure!(weight.k == H && weight.n == EXPERTS, "CUDA router geometry");
        let w = d.alloc(H * EXPERTS * 2)?;
        w.upload(bytes(&weight.values))?;
        let mut raw = vec![0; H * EXPERTS * 2];
        w.download(&mut raw)?;
        ensure!(raw == bytes(&weight.values), "router QMV weight readback");
        Ok(Self {
            d,
            qmv: ops.kernel("router_qmv")?,
            qmv_ahead: if crate::cuda_affine4::r38_enabled() {
                Some(ops.kernel("router_qmv_ahead")?)
            } else {
                None
            },
            select: ops.kernel("router_select")?,
            tc: Linear::new(d, tensor, weight)?,
            w,
            x: d.alloc(MAX_ROWS * H * 2)?,
            raw: d.alloc(MAX_ROWS * EXPERTS * 4 + GUARD)?,
            logits: d.alloc(MAX_ROWS * EXPERTS * 2 + GUARD)?,
            ids: d.alloc(MAX_ROWS * TOPK * 4 + GUARD)?,
            scores: d.alloc(MAX_ROWS * TOPK * 4 + GUARD)?,
            rows: Cell::new(0),
            input: Cell::new(Input::Empty),
        })
    }
    pub fn description(&self) -> Value {
        let extra_bytes: usize = [
            &self.w,
            &self.x,
            &self.raw,
            &self.logits,
            &self.ids,
            &self.scores,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":SHA,"experts":EXPERTS,"topk":TOPK,"max_rows":MAX_ROWS,
            "tie_break":"lowest original ID","logit_dtype":"BF16 after FP32 dot","score_policies":["staged_bf16","selected_f32"],
            "nonfinite_row":"IDs remain bounded/unique; NaN scores require rejection before publishing output",
            "tc":self.tc.description(),"qmv":self.qmv.description(),"select":self.select.description(),
            "extra_bytes": extra_bytes,
            "scope":"A/B retains duplicate2.5MiB BF16 weight and TC probe scratch; host routing absent"})
    }
    pub fn prepare_input(&self, x: &[u16]) -> Result<()> {
        self.rows.set(0);
        self.input.set(Input::Empty);
        ensure!(
            !x.is_empty()
                && x.len() % H == 0
                && x.len() / H <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "router input geometry/finite"
        );
        self.x.upload(bytes(x))?;
        let mut check = vec![0; x.len() * 2];
        self.x.download(&mut check)?;
        ensure!(check == bytes(x), "router input readback");
        self.rows.set(x.len() / H);
        self.input.set(Input::Activations);
        Ok(())
    }
    pub fn prepare_logits(&self, x: &[f32]) -> Result<()> {
        self.rows.set(0);
        self.input.set(Input::Empty);
        ensure!(
            !x.is_empty()
                && x.len() % EXPERTS == 0
                && x.len() / EXPERTS <= MAX_ROWS
                && x.iter()
                    .all(|&v| v.is_finite() && bf16::from_f32(v).is_finite()),
            "router logits geometry/finite"
        );
        let mut raw = vec![0xff; self.raw.byte_len()];
        raw[..x.len() * 4].copy_from_slice(bytes(x));
        self.raw.upload(&raw)?;
        let mut check = vec![0; raw.len()];
        self.raw.download(&mut check)?;
        ensure!(raw == check, "router given logits readback");
        self.rows.set(x.len() / EXPERTS);
        self.input.set(Input::Logits);
        Ok(())
    }
    fn select_call(&self, rows: usize, scores: Scores) -> Result<PreparedCall<'_, 'm, 'd>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "router selection rows");
        // Fixed512 logits,10 unique IDs/scores per row, a full32-thread warp;
        // owner allocations are disjoint and survive every prepared call/graph.
        unsafe {
            self.select.prepare(
                vec![
                    Arg::Buffer(&self.raw),
                    Arg::Buffer(&self.logits),
                    Arg::Buffer(&self.ids),
                    Arg::Buffer(&self.scores),
                    Arg::U32(u32::from(scores == Scores::SelectedF32)),
                ],
                [rows as u32, 1, 1],
                [32, 1, 1],
            )
        }
    }
    fn record_input<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        projection: Option<Projection>,
        scores: Scores,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && x.byte_len() >= rows * H * 2,
            "router resident input bound"
        );
        ensure!(
            [&self.raw, &self.logits, &self.ids, &self.scores]
                .iter()
                .all(|b| !x.aliases(b)),
            "router input/output alias"
        );
        let mut calls = vec![];
        if let Some(p) = projection {
            calls.push(match p {
                Projection::Qmv => unsafe {
                    let qmv = match (&self.qmv_ahead, crate::cuda_affine4::ahead_selected()) {
                        (Some(k), true) => k,
                        (None, true) => anyhow::bail!("R38 router kernel not loaded"),
                        _ => &self.qmv,
                    };
                    qmv.prepare(
                        vec![Arg::Buffer(x), Arg::Buffer(&self.w), Arg::Buffer(&self.raw)],
                        [(EXPERTS / 4) as u32, rows as u32, 1],
                        [128, 1, 1],
                    )?
                },
                Projection::Tensor(arm) => {
                    self.tc.record_into(x, &self.raw, rows, arm, Output::F32)?
                }
            });
        }
        calls.push(self.select_call(rows, scores)?);
        Ok(calls)
    }
    /// R38 island: BF16 weight, raw/BF16 logits, IDs and scores.
    pub(crate) fn island_parts(&self) -> [&Buffer<'d>; 5] {
        [&self.w, &self.raw, &self.logits, &self.ids, &self.scores]
    }
    pub(crate) fn input_buffer(&self) -> &Buffer<'d> {
        &self.x
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        projection: Projection,
        scores: Scores,
    ) -> Result<(Vec<PreparedCall<'a, 'm, 'd>>, DeviceRoutes<'a, 'd>)> {
        Ok((
            self.record_input(x, rows, Some(projection), scores)?,
            DeviceRoutes {
                ids: &self.ids,
                scores: &self.scores,
                rows,
            },
        ))
    }
    pub(crate) fn poison(&self, raw: bool) -> Result<()> {
        for b in [&self.logits, &self.ids, &self.scores] {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        if raw {
            self.raw.upload(&vec![0xff; self.raw.byte_len()])?;
        }
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize) -> Result<Run> {
        let raw: Vec<_> = read(&self.raw, rows * EXPERTS * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let logits: Vec<_> = read(&self.logits, rows * EXPERTS * 2)?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            raw.iter().zip(&logits).all(|(&f, &h)| f.is_finite()
                && bf16::from_bits(h).is_finite()
                && bf16::from_f32(f).to_bits() == h),
            "router logit rounding/finite"
        );
        let ids = read(&self.ids, rows * TOPK * 4)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let scores = read(&self.scores, rows * TOPK * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let routing = Routing { rows, ids, scores };
        routing.validate()?;
        let counter_words_checked = self.tc.check_counters()?;
        Ok(Run {
            raw,
            logits,
            routing,
            counter_words_checked,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn plan(
        &self,
        rows: usize,
        projection: Option<Projection>,
        scores: Scores,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows > 0
                && rows == self.rows.get()
                && (1..=16).contains(&repetitions)
                && self.input.get()
                    == if projection.is_some() {
                        Input::Activations
                    } else {
                        Input::Logits
                    },
            "router plan input/bounds"
        );
        let c = self.record_input(&self.x, rows, projection, scores)?;
        let execute = if graph {
            let mut all = c;
            for _ in 1..repetitions {
                all.extend(self.record_input(&self.x, rows, projection, scores)?);
            }
            Executor::Graph(self.d.capture(all)?)
        } else {
            Executor::Prepared(c)
        };
        Ok(Plan {
            owner: self,
            execute,
            rows,
            projection,
            repetitions,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Router<'m, 'd>,
    execute: Executor<'a, 'm, 'd>,
    rows: usize,
    projection: Option<Projection>,
    repetitions: usize,
}
pub struct Run {
    pub raw: Vec<f32>,
    pub logits: Vec<u16>,
    pub routing: Routing,
    pub counter_words_checked: usize,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
fn read(b: &Buffer<'_>, active: usize) -> Result<Vec<u8>> {
    ensure!(active + GUARD <= b.byte_len(), "router read bounds");
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(
        raw[active..].iter().all(|&v| v == 0xff),
        "router canary/tail"
    );
    raw.truncate(active);
    Ok(raw)
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        ensure!(
            a.rows.get() == self.rows
                && a.input.get()
                    == if self.projection.is_some() {
                        Input::Activations
                    } else {
                        Input::Logits
                    },
            "stale router plan input"
        );
        for b in [&a.logits, &a.ids, &a.scores] {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        if self.projection.is_some() {
            a.raw.upload(&vec![0xff; a.raw.byte_len()])?;
        }
        a.d.sync()?;
        let t = Instant::now();
        match &self.execute {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(calls) => {
                for _ in 0..self.repetitions {
                    for c in calls {
                        c.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = t.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let mut run = a.collect(self.rows)?;
        run.total_us = total_us;
        run.enqueue_us = enqueue_us;
        run.wait_us = wait_us;
        Ok(run)
    }
}

/// R60 S4 (prefill v2): router outputs for chunk-sized row counts.
pub struct RouterScratch<'d> {
    rows: usize,
    raw: Buffer<'d>,
    logits: Buffer<'d>,
    pub(crate) ids: Buffer<'d>,
    pub(crate) scores: Buffer<'d>,
}
impl<'d> RouterScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        Ok(Self {
            rows,
            raw: d.alloc(rows * EXPERTS * 4 + GUARD)?,
            logits: d.alloc(rows * EXPERTS * 2 + GUARD)?,
            ids: d.alloc(rows * TOPK * 4 + GUARD)?,
            scores: d.alloc(rows * TOPK * 4 + GUARD)?,
        })
    }
}
impl<'m, 'd> Router<'m, 'd> {
    /// R60 S4 (prefill v2): router_qmv + router_select (staged BF16 scores) for chunk rows.
    pub(crate) fn record_prefill<'a>(&'a self, s: &'a RouterScratch<'d>, x: &'a Buffer<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && x.byte_len() >= rows * H * 2, "router prefill shape");
        ensure!([&s.raw, &s.logits, &s.ids, &s.scores].iter().all(|b| !x.aliases(b)), "router prefill alias");
        // SAFETY: kernels/lowbit_router.cu ABIs (the decode path's Qmv projection and selection).
        Ok(vec![
            unsafe { self.qmv.prepare(vec![Arg::Buffer(x), Arg::Buffer(&self.w), Arg::Buffer(&s.raw)], [(EXPERTS / 4) as u32, rows as u32, 1], [128, 1, 1])? },
            unsafe {
                self.select.prepare(
                    vec![Arg::Buffer(&s.raw), Arg::Buffer(&s.logits), Arg::Buffer(&s.ids), Arg::Buffer(&s.scores), Arg::U32(0)],
                    [rows as u32, 1, 1],
                    [32, 1, 1],
                )?
            },
        ])
    }
}
