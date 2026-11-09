//! Native connected DFlash2 MLP; no intermediate host transfers or upstream runtime.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_linear::{Arm, Linear, Output},
    dflash_mlp::{Arithmetic, DYN, EPS, F, H, MAX_ROWS, Weights},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
use std::{cell::Cell, path::Path, time::Instant};
pub const CUBIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/kernels/dflash_ops.sm120.cubin"
);
pub const SHA: &str = "0a4100d5bdf41f39c8badb7fbec4653d62c4e3cf44ea79ec564fa31b3883a8ac";
const GUARD: usize = 256;
const COLS: [usize; 8] = [H, DYN, H, F, F, F, H, H];
const NAMES: [&str; 8] = [
    "norm",
    "dynamic",
    "prepare_conv",
    "gate",
    "up",
    "swiglu",
    "down",
    "finish_residual",
];
fn bytes(x: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len() * 2) }
}
pub fn load_module(d: &Device) -> Result<Module<'_>> {
    d.module(Path::new(CUBIN), SHA)
}
pub struct Mlp<'m, 'd> {
    device: &'d Device,
    rms: Kernel<'m, 'd>,
    conv: Kernel<'m, 'd>,
    act: Kernel<'m, 'd>,
    norm: Buffer<'d>,
    base: Buffer<'d>,
    input: Buffer<'d>,
    bufs: Vec<Buffer<'d>>,
    dynamic: Linear<'m, 'd>,
    gate: Linear<'m, 'd>,
    up: Linear<'m, 'd>,
    down: Linear<'m, 'd>,
    rows: Cell<usize>,
}
impl<'m, 'd> Mlp<'m, 'd> {
    pub fn new(
        d: &'d Device,
        gemm: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        let upload = |v: &[u16]| -> Result<Buffer<'d>> {
            let b = d.alloc(v.len() * 2)?;
            b.upload(bytes(v))?;
            let mut chk = vec![0; v.len() * 2];
            b.download(&mut chk)?;
            ensure!(chk == bytes(v), "MLP weight transfer");
            Ok(b)
        };
        let this = Self {
            device: d,
            rms: ops.kernel("dflash_rms")?,
            conv: ops.kernel("dflash_conv")?,
            act: ops.kernel("dflash_swiglu")?,
            norm: upload(&w.norm)?,
            base: upload(&w.base)?,
            input: d.alloc(MAX_ROWS * H * 2)?,
            bufs: COLS
                .iter()
                .map(|&n| d.alloc(MAX_ROWS * n * 2 + GUARD))
                .collect::<Result<_>>()?,
            dynamic: Linear::new(d, gemm, &w.dynamic)?,
            gate: Linear::new(d, gemm, &w.gate)?,
            up: Linear::new(d, gemm, &w.up)?,
            down: Linear::new(d, gemm, &w.down)?,
            rows: Cell::new(0),
        };
        Ok(this)
    }
    pub fn description(&self) -> serde_json::Value {
        json!({"backend":"native Rust/CUDA/libtinynv","ops_cubin_sha256":SHA,
            "kernels":[self.rms.description(),self.conv.description(),self.act.description()],
            "linears":[self.dynamic.description(),self.gate.description(),self.up.description(),self.down.description()],
            "weight_readback_verified":true,"dependent_device_calls":8,
            "intermediate_host_transfers":0,"maximum_rows":MAX_ROWS,
            "linear_private_probe_buffers_retained":true})
    }
    pub fn prepare(&self, x: &[u16]) -> Result<f64> {
        ensure!(
            !x.is_empty()
                && x.len() % H == 0
                && x.len() / H <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "connected MLP input"
        );
        self.rows.set(0);
        let t = Instant::now();
        self.input.upload(bytes(x))?;
        let mut check = vec![0; x.len() * 2];
        self.input.download(&mut check)?;
        ensure!(check == bytes(x), "connected MLP input upload");
        self.rows.set(x.len() / H);
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        rows: usize,
        arm: Arithmetic,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "MLP graph shape");
        ensure!(
            input.byte_len() >= rows * H * 2 && self.bufs.iter().all(|b| !input.aliases(b)),
            "MLP external input size/alias"
        );
        let b = &self.bufs;
        let mut c = vec![];
        // Audited source ABI. All fixed-width tensors are owned, fully sized,
        // disjoint, and retained by the plan. All calls share one device stream.
        unsafe {
            c.push(self.rms.prepare(
                vec![
                    Arg::Buffer(input),
                    Arg::Buffer(&self.norm),
                    Arg::Buffer(&b[0]),
                    Arg::U32(H as u32),
                    Arg::F32(EPS as f32),
                ],
                [rows as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        c.push(
            self.dynamic
                .record_into(&b[0], &b[1], rows, Arm::Split(8), Output::Bf16)?,
        );
        unsafe {
            c.push(self.conv.prepare(
                vec![
                    Arg::Buffer(&b[0]),
                    Arg::Buffer(&b[1]),
                    Arg::Buffer(&self.base),
                    Arg::Buffer(input),
                    Arg::Buffer(&b[2]),
                    Arg::U32(rows as u32),
                    Arg::U32(H as u32),
                    Arg::U32(16),
                    Arg::U32(0),
                    Arg::U32(arm.code()),
                    Arg::U32(0),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        c.push(
            self.gate
                .record_into(&b[2], &b[3], rows, Arm::Split(4), Output::Bf16)?,
        );
        c.push(
            self.up
                .record_into(&b[2], &b[4], rows, Arm::Split(4), Output::Bf16)?,
        );
        unsafe {
            c.push(self.act.prepare(
                vec![
                    Arg::Buffer(&b[3]),
                    Arg::Buffer(&b[4]),
                    Arg::Buffer(&b[5]),
                    Arg::U32((rows * F) as u32),
                    Arg::U32(arm.code()),
                ],
                [(rows * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        c.push(
            self.down
                .record_into(&b[5], &b[6], rows, Arm::Split(4), Output::Bf16)?,
        );
        unsafe {
            c.push(self.conv.prepare(
                vec![
                    Arg::Buffer(&b[6]),
                    Arg::Buffer(&b[1]),
                    Arg::Buffer(&self.base),
                    Arg::Buffer(input),
                    Arg::Buffer(&b[7]),
                    Arg::U32(rows as u32),
                    Arg::U32(H as u32),
                    Arg::U32(16),
                    Arg::U32(1),
                    Arg::U32(arm.code()),
                    Arg::U32(1),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        Ok(c)
    }
    pub fn plan(&self, rows: usize, arm: Arithmetic, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows == self.rows.get() && rows > 0,
            "prepare MLP shape before recording"
        );
        let t = Instant::now();
        let calls = self.record_from(&self.input, rows, arm)?;
        let execute = if graph {
            Executor::Graph(self.device.capture(calls)?)
        } else {
            Executor::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            execute,
            rows,
            setup_us: t.elapsed().as_secs_f64() * 1e6,
        })
    }
    pub(crate) fn poison(&self) -> Result<()> {
        for b in &self.bufs {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        Ok(())
    }
    pub(crate) fn output_buffer(&self) -> &Buffer<'d> {
        &self.bufs[7]
    }
    pub(crate) fn collect(&self, rows: usize) -> Result<Collected> {
        ensure!((1..=MAX_ROWS).contains(&rows), "MLP collect rows");
        let mut stages = vec![];
        let mut canary_bytes_checked = 0;
        for (i, b) in self.bufs.iter().enumerate() {
            let mut raw = vec![0; b.byte_len()];
            b.download(&mut raw)?;
            let active = rows * COLS[i] * 2;
            ensure!(
                raw[active..].iter().all(|&v| v == 0xff),
                "MLP stage {} output guard",
                NAMES[i]
            );
            canary_bytes_checked += raw.len() - active;
            let values: Vec<_> = raw[..active]
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "MLP stage {} nonfinite/unwritten",
                NAMES[i]
            );
            stages.push((NAMES[i], values));
        }
        let mut counters_checked = 0;
        for l in [&self.dynamic, &self.gate, &self.up, &self.down] {
            counters_checked += l.check_counters()?;
        }
        Ok(Collected {
            stages,
            counters_checked,
            canary_bytes_checked,
        })
    }
}
pub(crate) struct Collected {
    pub stages: Vec<(&'static str, Vec<f32>)>,
    pub counters_checked: usize,
    pub canary_bytes_checked: usize,
}
enum Executor<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Mlp<'m, 'd>,
    execute: Executor<'a, 'm, 'd>,
    rows: usize,
    pub setup_us: f64,
}
pub struct Run {
    pub stages: Vec<(&'static str, Vec<f32>)>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub counters_checked: usize,
    pub canary_bytes_checked: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, repetitions: usize, wait_each: bool) -> Result<Run> {
        ensure!(
            self.rows == self.owner.rows.get() && (1..=32).contains(&repetitions),
            "MLP changed shape/repetitions"
        );
        ensure!(
            !wait_each || matches!(&self.execute, Executor::Calls(_)),
            "wait-each graph unsupported"
        );
        let m = self.owner;
        m.poison()?;
        m.device.sync()?;
        let total = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repetitions {
            match &self.execute {
                Executor::Calls(calls) => {
                    for c in calls {
                        let t = Instant::now();
                        c.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            m.device.sync()?;
                            completion_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    }
                }
                Executor::Graph(g) => {
                    let t = Instant::now();
                    g.launch()?;
                    enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                }
            }
        }
        let t = Instant::now();
        m.device.sync()?;
        completion_us += t.elapsed().as_secs_f64() * 1e6;
        let total_us = total.elapsed().as_secs_f64() * 1e6;
        let Collected {
            stages,
            counters_checked,
            canary_bytes_checked,
        } = m.collect(self.rows)?;
        Ok(Run {
            stages,
            total_us,
            enqueue_us,
            completion_us,
            counters_checked,
            canary_bytes_checked,
        })
    }
}
