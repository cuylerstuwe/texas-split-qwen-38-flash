//! Resident CUDA lowbit shared expert composition; no host intermediate tensors.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::Linear,
    lowbit_shared::{Arithmetic, H, I, MAX_ROWS, Stages, Weights},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Serialize;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "f286caea10d3cdcab38aa87fe6508c3b2e2ac5e9561b478b8438a40bdff512ca";
const GUARD: usize = 256;
/// R38 (Opus, direct): unchanged lowbit_shared.cu plus shared_finish_ahead.
pub const R38_SHA: &str = "465ca09c11946c63aca88924de0465837850463ff396240fe1ff661b1e229667";
pub fn module(d: &Device) -> Result<Module<'_>> {
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_shared_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_shared.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(x: &[T]) -> &[u8] {
    // Initialized numerical arrays on this little-endian host.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub enum Projection {
    Original(usize),
    Split { width: usize, splits: usize },
}
impl Projection {
    /// R60: affine8 projections have no split-K variant; a split request becomes the plain arm
    /// with the same rows per block.
    fn for_linear(self, l: &Linear<'_, '_>) -> Self {
        match self {
            Self::Split { width, .. } if l.is_affine8() => Self::Original(match width {
                1 => 0,
                2 => 2,
                _ => 3,
            }),
            other => other,
        }
    }
    fn record<'a, 'm, 'd>(
        self,
        l: &'a Linear<'m, 'd>,
        x: &'a Buffer<'d>,
        y: &'a Pair<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        match self.for_linear(l) {
            Self::Original(arm) => Ok(vec![l.record_into(x, &y.fp32, &y.bf16, rows, arm)?]),
            Self::Split { width, splits } => {
                l.record_split_into(x, &y.fp32, &y.bf16, rows, width, splits)
            }
        }
    }
    fn poison(self, l: &Linear<'_, '_>) -> Result<()> {
        if matches!(self.for_linear(l), Self::Split { .. }) {
            l.poison_split()?;
        }
        Ok(())
    }
    fn check(self, l: &Linear<'_, '_>, rows: usize) -> Result<usize> {
        match self.for_linear(l) {
            Self::Original(_) => Ok(0),
            Self::Split { splits, .. } => l.check_split(rows, splits),
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Policy {
    pub combined: bool,
    pub projection: Projection,
    pub down: Projection,
    pub arithmetic: Arithmetic,
}
struct Pair<'d> {
    fp32: Buffer<'d>,
    bf16: Buffer<'d>,
}
impl<'d> Pair<'d> {
    fn new(d: &'d Device, n: usize) -> Result<Self> {
        Ok(Self {
            fp32: d.alloc(n * 4 + GUARD)?,
            bf16: d.alloc(n * 2 + GUARD)?,
        })
    }
    fn poison(&self) -> Result<()> {
        poison(&self.fp32)?;
        poison(&self.bf16)
    }
    fn read(&self, n: usize) -> Result<(Vec<f32>, Vec<u16>)> {
        let raw = read_guarded(&self.fp32, n * 4)?;
        let f: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let b = read_bf16(&self.bf16, n)?;
        ensure!(
            f.iter()
                .zip(&b)
                .all(|(&f, &b)| f.is_finite() && bf16::from_f32(f).to_bits() == b),
            "shared projection rounding/finite"
        );
        Ok((f, b))
    }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![0xff; b.byte_len()])
}
fn read_guarded(b: &Buffer<'_>, bytes: usize) -> Result<Vec<u8>> {
    ensure!(bytes + GUARD <= b.byte_len(), "shared read bounds");
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(
        raw[bytes..].iter().all(|&v| v == 0xff),
        "shared output/tail canary"
    );
    raw.truncate(bytes);
    Ok(raw)
}
fn read_bf16(b: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let raw = read_guarded(b, count * 2)?;
    let values: Vec<_> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|&b| bf16::from_bits(b).is_finite()),
        "shared stage finite/fully written"
    );
    Ok(values)
}
pub struct Shared<'m, 'd> {
    d: &'d Device,
    gate: Linear<'m, 'd>,
    up: Linear<'m, 'd>,
    combined: Linear<'m, 'd>,
    down: Linear<'m, 'd>,
    activate: Kernel<'m, 'd>,
    finish: Kernel<'m, 'd>,
    finish_ahead: Option<Kernel<'m, 'd>>,
    scalar_gate: Buffer<'d>,
    x: Buffer<'d>,
    gu: Pair<'d>,
    u: Pair<'d>,
    projected: Pair<'d>,
    act: Buffer<'d>,
    out: Buffer<'d>,
    scalar_raw: Buffer<'d>,
    logits: Buffer<'d>,
    sigmoid: Buffer<'d>,
    rows: Cell<usize>,
}
impl<'m, 'd> Shared<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        split: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        ensure!(
            w.projections[0].k == H
                && w.projections[0].n == I
                && w.projections[1].k == H
                && w.projections[1].n == I
                && w.projections[2].k == I
                && w.projections[2].n == H
                && w.scalar_gate.k == H
                && w.scalar_gate.n == 1,
            "shared fixed geometry"
        );
        w.scalar_gate.validate()?;
        let joined = w.projections[0].concat_rows(&w.projections[1])?;
        Self::assemble(
            d,
            ops,
            [
                Linear::new_split(d, affine, split, &w.projections[0])?,
                Linear::new_split(d, affine, split, &w.projections[1])?,
                Linear::new_split(d, affine, split, &joined)?,
                Linear::new_split(d, affine, split, &w.projections[2])?,
            ],
            &w.scalar_gate.values,
            true,
        )
    }
    /// R60 pipeline split: the canonical target's shared expert (affine8 gate/up/down from `m8`,
    /// the affine8 module; BF16 scalar gate). Record with Projection::Original arms only.
    pub fn new_canonical(d: &'d Device, m8: &'m Module<'d>, ops: &'m Module<'d>, w: &crate::target_mlp::Weights) -> Result<Self> {
        let [g, u, dn] = &w.shared;
        ensure!(
            (g.k, g.n, u.k, u.n, dn.k, dn.n) == (H, I, H, I, I, H) && w.shared_gate.k == H && w.shared_gate.n == 1,
            "canonical shared geometry"
        );
        w.shared_gate.validate()?;
        let joined = g.concat_rows(u)?;
        Self::assemble(
            d,
            ops,
            [
                Linear::new_affine8(d, m8, g)?,
                Linear::new_affine8(d, m8, u)?,
                Linear::new_affine8(d, m8, &joined)?,
                Linear::new_affine8(d, m8, dn)?,
            ],
            &w.shared_gate.values,
            false,
        )
    }
    fn assemble(d: &'d Device, ops: &'m Module<'d>, linears: [Linear<'m, 'd>; 4], gate_values: &[u16], fast: bool) -> Result<Self> {
        ensure!(gate_values.len() == H, "shared scalar gate length");
        let scalar_gate = d.alloc(H * 2)?;
        scalar_gate.upload(bytes(gate_values))?;
        let mut check = vec![0; H * 2];
        scalar_gate.download(&mut check)?;
        ensure!(check == bytes(gate_values), "shared scalar weight upload readback");
        let [gate, up, combined, down] = linears;
        Ok(Self {
            d,
            gate,
            up,
            combined,
            down,
            activate: ops.kernel("shared_activate")?,
            finish: ops.kernel("shared_finish")?,
            finish_ahead: if fast && crate::cuda_affine4::r38_enabled() {
                Some(ops.kernel("shared_finish_ahead")?)
            } else {
                None
            },
            scalar_gate,
            x: d.alloc(MAX_ROWS * H * 2)?,
            gu: Pair::new(d, MAX_ROWS * 2 * I)?,
            u: Pair::new(d, MAX_ROWS * I)?,
            projected: Pair::new(d, MAX_ROWS * H)?,
            act: d.alloc(MAX_ROWS * I * 2 + GUARD)?,
            out: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            scalar_raw: d.alloc(MAX_ROWS * 4 + GUARD)?,
            logits: d.alloc(MAX_ROWS * 2 + GUARD)?,
            sigmoid: d.alloc(MAX_ROWS * 2 + GUARD)?,
            rows: Cell::new(0),
        })
    }
    pub fn description(&self) -> Value {
        let extra_bytes: usize = [
            &self.scalar_gate,
            &self.x,
            &self.gu.fp32,
            &self.gu.bf16,
            &self.u.fp32,
            &self.u.bf16,
            &self.projected.fp32,
            &self.projected.bf16,
            &self.act,
            &self.out,
            &self.scalar_raw,
            &self.logits,
            &self.sigmoid,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":SHA,"hidden":H,"intermediate":I,"max_rows":MAX_ROWS,
            "resident_intermediates":true,"host_intermediates":0,"extra_bytes":extra_bytes,
            "linears":[self.gate.description(),self.up.description(),self.combined.description(),self.down.description()],
            "allocation_scope":"A/B probe retains both separate and combined packed weights plus Linear probe scratch; production would choose one",
            "kernels":[self.activate.description(),self.finish.description()]})
    }
    pub fn prepare(&self, x: &[u16]) -> Result<()> {
        self.rows.set(0);
        ensure!(
            !x.is_empty()
                && x.len() % H == 0
                && x.len() / H <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "shared input shape/finite"
        );
        self.x.upload(bytes(x))?;
        let mut check = vec![0; x.len() * 2];
        self.x.download(&mut check)?;
        ensure!(check == bytes(x), "shared input readback");
        self.rows.set(x.len() / H);
        Ok(())
    }
    fn record(&self, rows: usize, p: Policy) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.record_from(&self.x, rows, p)
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(x.byte_len() >= rows * H * 2, "shared external BF16 size");
        ensure!((1..=MAX_ROWS).contains(&rows), "shared record rows");
        let mut calls = if p.combined {
            p.projection.record(&self.combined, x, &self.gu, rows)?
        } else {
            let mut c = p.projection.record(&self.gate, x, &self.gu, rows)?;
            c.extend(p.projection.record(&self.up, x, &self.u, rows)?);
            c
        };
        let arithmetic = if p.arithmetic == Arithmetic::StagedBf16 {
            0
        } else {
            1
        };
        // Fixed geometry, private disjoint buffers and bounded rows; CUDA guards
        // the activation tail and finish owns exactly one row per128-thread block.
        unsafe {
            calls.push(self.activate.prepare(
                vec![
                    Arg::Buffer(&self.gu.bf16),
                    Arg::Buffer(&self.u.bf16),
                    Arg::Buffer(&self.act),
                    Arg::U32(rows as u32),
                    Arg::U32(I as u32),
                    Arg::U32(u32::from(p.combined)),
                    Arg::U32(arithmetic),
                ],
                [(rows * I).div_ceil(128) as u32, 1, 1],
                [128, 1, 1],
            )?);
        }
        calls.extend(
            p.down
                .record(&self.down, &self.act, &self.projected, rows)?,
        );
        let finish = match (&self.finish_ahead, crate::cuda_affine4::ahead_selected()) {
            (Some(k), true) => k,
            (None, true) => anyhow::bail!("R38 shared finish kernel not loaded"),
            _ => &self.finish,
        };
        unsafe {
            calls.push(finish.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.scalar_gate),
                    Arg::Buffer(&self.projected.bf16),
                    Arg::Buffer(&self.out),
                    Arg::Buffer(&self.scalar_raw),
                    Arg::Buffer(&self.logits),
                    Arg::Buffer(&self.sigmoid),
                    Arg::U32(H as u32),
                    Arg::U32(arithmetic),
                ],
                [rows as u32, 1, 1],
                [128, 1, 1],
            )?);
        }
        Ok(calls)
    }
    /// R38 island buffers, in kernel argument order (see moe_island.cu).
    pub(crate) fn island_parts(&self) -> Result<(Vec<&Buffer<'d>>, [usize; 2])> {
        let (gw, gs, gb, ggroup) = self.combined.packed_weights();
        let (dw, ds, db, dgroup) = self.down.packed_weights();
        Ok((
            vec![
                gw,
                gs,
                gb,
                self.combined.split_partials()?,
                &self.gu.fp32,
                &self.gu.bf16,
                &self.act,
                dw,
                ds,
                db,
                &self.projected.fp32,
                &self.projected.bf16,
                &self.scalar_gate,
                &self.out,
                &self.scalar_raw,
                &self.logits,
                &self.sigmoid,
            ],
            [ggroup, dgroup],
        ))
    }
    pub(crate) fn output(&self) -> &Buffer<'d> {
        &self.out
    }
    pub(crate) fn poison(&self, p: Policy) -> Result<()> {
        for pair in [&self.gu, &self.u, &self.projected] {
            pair.poison()?;
        }
        for b in [
            &self.act,
            &self.out,
            &self.scalar_raw,
            &self.logits,
            &self.sigmoid,
        ] {
            poison(b)?;
        }
        if p.combined {
            p.projection.poison(&self.combined)?;
        } else {
            p.projection.poison(&self.gate)?;
            p.projection.poison(&self.up)?;
        }
        p.down.poison(&self.down)?;
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize, p: Policy) -> Result<Run> {
        let (graw, g) = self.gu.read(rows * I * if p.combined { 2 } else { 1 })?;
        let (gate_raw, up_raw, gate, up) = if p.combined {
            let mut gr = Vec::new();
            let mut ur = Vec::new();
            let mut gb = Vec::new();
            let mut ub = Vec::new();
            for r in 0..rows {
                gr.extend_from_slice(&graw[r * 2 * I..r * 2 * I + I]);
                ur.extend_from_slice(&graw[r * 2 * I + I..(r + 1) * 2 * I]);
                gb.extend_from_slice(&g[r * 2 * I..r * 2 * I + I]);
                ub.extend_from_slice(&g[r * 2 * I + I..(r + 1) * 2 * I]);
            }
            // Unused up outputs must remain untouched in the combined path.
            read_guarded(&self.u.fp32, 0)?;
            read_guarded(&self.u.bf16, 0)?;
            (gr, ur, gb, ub)
        } else {
            let (ur, ub) = self.u.read(rows * I)?;
            (graw, ur, g, ub)
        };
        let activation = read_bf16(&self.act, rows * I)?;
        let (down_raw, down) = self.projected.read(rows * H)?;
        let gate_logits = read_bf16(&self.logits, rows)?;
        let gate_sigmoid = read_bf16(&self.sigmoid, rows)?;
        let output = read_bf16(&self.out, rows * H)?;
        let scalar_raw: Vec<_> = read_guarded(&self.scalar_raw, rows * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            scalar_raw
                .iter()
                .zip(&gate_logits)
                .all(|(&f, &b)| f.is_finite() && bf16::from_f32(f).to_bits() == b),
            "shared scalar rounding/finite"
        );
        let partial_values_checked = if p.combined {
            p.projection.check(&self.combined, rows)?
        } else {
            p.projection.check(&self.gate, rows)? + p.projection.check(&self.up, rows)?
        } + p.down.check(&self.down, rows)?;
        Ok(Run {
            stages: Stages {
                gate,
                up,
                activation,
                down,
                gate_logits,
                gate_sigmoid,
                output,
            },
            gate_raw,
            up_raw,
            down_raw,
            scalar_raw,
            partial_values_checked,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn plan(
        &self,
        rows: usize,
        p: Policy,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows > 0 && rows == self.rows.get() && (1..=16).contains(&repetitions),
            "shared plan bounds"
        );
        let c = self.record(rows, p)?;
        let calls_per_branch = c.len();
        let exec = if graph {
            let mut all = c;
            for _ in 1..repetitions {
                all.extend(self.record(rows, p)?);
            }
            Executor::Graph(self.d.capture(all)?)
        } else {
            Executor::Prepared(c)
        };
        Ok(Plan {
            owner: self,
            exec,
            rows,
            policy: p,
            repetitions,
            calls_per_branch,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Shared<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    policy: Policy,
    repetitions: usize,
    pub calls_per_branch: usize,
}
pub struct Run {
    pub stages: Stages,
    pub gate_raw: Vec<f32>,
    pub up_raw: Vec<f32>,
    pub down_raw: Vec<f32>,
    pub scalar_raw: Vec<f32>,
    pub partial_values_checked: usize,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        let p = self.policy;
        let rows = self.rows;
        ensure!(a.rows.get() == rows, "shared plan input geometry changed");
        for pair in [&a.gu, &a.u, &a.projected] {
            pair.poison()?;
        }
        for b in [&a.act, &a.out, &a.scalar_raw, &a.logits, &a.sigmoid] {
            poison(b)?;
        }
        if p.combined {
            p.projection.poison(&a.combined)?;
        } else {
            p.projection.poison(&a.gate)?;
            p.projection.poison(&a.up)?;
        }
        p.down.poison(&a.down)?;
        a.d.sync()?;
        let start = Instant::now();
        match &self.exec {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(calls) => {
                for _ in 0..self.repetitions {
                    for c in calls {
                        c.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut run = a.collect(rows, p)?;
        run.total_us = total_us;
        run.enqueue_us = enqueue_us;
        run.wait_us = wait_us;
        Ok(run)
    }
}

/// R60 S4 (prefill v2): shared-expert intermediates for chunk-sized row counts.
pub struct SharedScratch<'d> {
    rows: usize,
    gu: Pair<'d>,
    u: Pair<'d>,
    projected: Pair<'d>,
    act: Buffer<'d>,
    out: Buffer<'d>,
    scalar_raw: Buffer<'d>,
    logits: Buffer<'d>,
    sigmoid: Buffer<'d>,
}
impl<'d> SharedScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        Ok(Self {
            rows,
            gu: Pair::new(d, rows * 2 * I)?,
            u: Pair::new(d, rows * I)?,
            projected: Pair::new(d, rows * H)?,
            act: d.alloc(rows * I * 2 + GUARD)?,
            out: d.alloc(rows * H * 2 + GUARD)?,
            scalar_raw: d.alloc(rows * 4 + GUARD)?,
            logits: d.alloc(rows * 2 + GUARD)?,
            sigmoid: d.alloc(rows * 2 + GUARD)?,
        })
    }
    pub fn output(&self) -> &Buffer<'d> {
        &self.out
    }
}
impl<'m, 'd> Shared<'m, 'd> {
    /// R60 S4 (prefill v2): combined gate/up and down on the 64-token tensor tile, the staged-BF16
    /// activation and the scalar-gated finish, for chunk rows.
    pub(crate) fn record_prefill<'a>(&'a self, s: &'a SharedScratch<'d>, x: &'a Buffer<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && x.byte_len() >= rows * H * 2, "shared prefill shape");
        let mut calls = vec![self.combined.record_rows(x, &s.gu.fp32, &s.gu.bf16, rows)?];
        // SAFETY: kernels/lowbit_shared.cu ABIs (the decode path's combined, staged-BF16 arms).
        calls.push(unsafe {
            self.activate.prepare(
                vec![Arg::Buffer(&s.gu.bf16), Arg::Buffer(&s.u.bf16), Arg::Buffer(&s.act), Arg::U32(rows as u32), Arg::U32(I as u32), Arg::U32(1), Arg::U32(0)],
                [(rows * I).div_ceil(128) as u32, 1, 1],
                [128, 1, 1],
            )?
        });
        calls.push(self.down.record_rows(&s.act, &s.projected.fp32, &s.projected.bf16, rows)?);
        calls.push(unsafe {
            self.finish.prepare(
                vec![
                    Arg::Buffer(x), Arg::Buffer(&self.scalar_gate), Arg::Buffer(&s.projected.bf16), Arg::Buffer(&s.out),
                    Arg::Buffer(&s.scalar_raw), Arg::Buffer(&s.logits), Arg::Buffer(&s.sigmoid), Arg::U32(H as u32), Arg::U32(0),
                ],
                [rows as u32, 1, 1],
                [128, 1, 1],
            )?
        });
        Ok(calls)
    }
}
