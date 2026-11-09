//! Complete original-weight DFlash2 layer, one native CUDA queue and resident KV.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_dflash_attention::Attention,
    cuda_dflash_mlp::Mlp,
    cuda_linear::{Arm, Linear, Output},
    dflash_attention::{D, KH, QH, Shape},
    dflash_layer::{Fixture, Weights},
    dflash_mlp::{Arithmetic, DYN, EPS, H, MAX_ROWS},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::{cell::Cell, time::Instant};
const GUARD: usize = 256;
const COLS: [usize; 5] = [H, DYN, H, H, H];
fn bytes(x: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len() * 2) }
}
pub struct Layer<'m, 'd> {
    device: &'d Device,
    rms: Kernel<'m, 'd>,
    conv: Kernel<'m, 'd>,
    norm: Buffer<'d>,
    base: Buffer<'d>,
    input: Buffer<'d>,
    bufs: Vec<Buffer<'d>>,
    dynamic: Linear<'m, 'd>,
    q: Linear<'m, 'd>,
    k: Linear<'m, 'd>,
    v: Linear<'m, 'd>,
    o: Linear<'m, 'd>,
    attention: Attention<'m, 'd>,
    mlp: Mlp<'m, 'd>,
    shape: Cell<Option<Shape>>,
}
impl<'m, 'd> Layer<'m, 'd> {
    pub fn new(
        d: &'d Device,
        gemm: &'m Module<'d>,
        ops: &'m Module<'d>,
        attn: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        let upload = |x: &[u16]| -> Result<Buffer<'d>> {
            let b = d.alloc(x.len() * 2)?;
            b.upload(bytes(x))?;
            let mut got = vec![0; x.len() * 2];
            b.download(&mut got)?;
            ensure!(got == bytes(x), "layer weight upload/readback");
            Ok(b)
        };
        Ok(Self {
            device: d,
            rms: ops.kernel("dflash_rms")?,
            conv: ops.kernel("dflash_conv")?,
            norm: upload(&w.norm)?,
            base: upload(&w.base)?,
            input: d.alloc(MAX_ROWS * H * 2)?,
            bufs: COLS
                .iter()
                .map(|&c| d.alloc(MAX_ROWS * c * 2 + GUARD))
                .collect::<Result<_>>()?,
            dynamic: Linear::new(d, gemm, &w.dynamic)?,
            q: Linear::new(d, gemm, &w.q)?,
            k: Linear::new(d, gemm, &w.k)?,
            v: Linear::new(d, gemm, &w.v)?,
            o: Linear::new(d, gemm, &w.o)?,
            attention: Attention::new(d, attn, &w.qk)?,
            mlp: Mlp::new(d, gemm, ops, &w.mlp)?,
            shape: Cell::new(None),
        })
    }
    pub fn description(&self) -> serde_json::Value {
        json!({"backend":"native Rust/CUDA/libtinynv","dependent_device_calls":19,"intermediate_host_transfers":0,
            "attention_core":self.attention.description(),"mlp":self.mlp.description(),
            "attention_projections":[self.dynamic.description(),self.q.description(),self.k.description(),self.v.description(),self.o.description()],
            "attention_extra_buffer_bytes":self.norm.byte_len()+self.base.byte_len()+self.input.byte_len()+self.bufs.iter().map(|b|b.byte_len()).sum::<usize>(),
            "original_bf16_weights":true,"context_append_not_timed":true,"full_drafter_not_implemented":true})
    }
    pub fn prepare(&self, f: &Fixture) -> Result<f64> {
        self.shape.set(None);
        f.validate()?;
        let t = Instant::now();
        self.input.upload(bytes(&f.x))?;
        let mut got = vec![0; f.x.len() * 2];
        self.input.download(&mut got)?;
        ensure!(got == bytes(&f.x), "layer input transfer");
        self.attention.prepare_context(f.shape, &f.ck, &f.cv)?;
        self.shape.set(Some(f.shape));
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    pub fn check_context(&self, f: &Fixture) -> Result<usize> {
        ensure!(
            self.shape.get() == Some(f.shape),
            "layer context shape changed"
        );
        self.attention.check_context(&f.ck, &f.cv)
    }
    pub(crate) fn prepare_context(&self, shape: Shape, k: &[u16], v: &[u16]) -> Result<()> {
        self.shape.set(None);
        self.attention.prepare_context(shape, k, v)?;
        self.shape.set(Some(shape));
        Ok(())
    }
    pub(crate) fn prepared_shape(&self) -> Option<Shape> {
        self.shape
            .get()
            .filter(|s| Some(*s) == self.attention.prepared_shape())
    }
    pub(crate) fn output_buffer(&self) -> &Buffer<'d> {
        self.mlp.output_buffer()
    }
    fn record(&self, rows: usize, splits: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.record_from(&self.input, rows, splits)
    }
    pub(crate) fn record_from<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        rows: usize,
        splits: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "DFlash layer rows");
        ensure!(
            input.byte_len() >= rows * H * 2
                && self.bufs.iter().all(|b| !input.aliases(b))
                && !input.aliases(self.output_buffer()),
            "layer external input size/alias"
        );
        let b = &self.bufs;
        let mut calls = vec![];
        // Audited fixed-width kernel ABIs. All buffers disjoint, retained and
        // fully sized; dependent projections/elementary calls share one stream.
        unsafe {
            calls.push(self.rms.prepare(
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
        calls.push(
            self.dynamic
                .record_into(&b[0], &b[1], rows, Arm::Split(8), Output::Bf16)?,
        );
        unsafe {
            calls.push(self.conv.prepare(
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
                    Arg::U32(0),
                    Arg::U32(0),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        let (q, k, v) = self.attention.block_buffers();
        calls.push(
            self.q
                .record_into(&b[2], q, rows, Arm::Split(4), Output::Bf16)?,
        );
        calls.push(
            self.k
                .record_into(&b[2], k, rows, Arm::Split(8), Output::Bf16)?,
        );
        calls.push(
            self.v
                .record_into(&b[2], v, rows, Arm::Split(8), Output::Bf16)?,
        );
        calls.extend(self.attention.record(splits)?);
        calls.push(self.o.record_into(
            self.attention.output_buffer(),
            &b[3],
            rows,
            Arm::Split(4),
            Output::Bf16,
        )?);
        unsafe {
            calls.push(self.conv.prepare(
                vec![
                    Arg::Buffer(&b[3]),
                    Arg::Buffer(&b[1]),
                    Arg::Buffer(&self.base),
                    Arg::Buffer(input),
                    Arg::Buffer(&b[4]),
                    Arg::U32(rows as u32),
                    Arg::U32(H as u32),
                    Arg::U32(16),
                    Arg::U32(1),
                    Arg::U32(0),
                    Arg::U32(1),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        calls.extend(self.mlp.record_from(&b[4], rows, Arithmetic::Staged)?);
        ensure!(calls.len() == 19, "complete layer call count");
        Ok(calls)
    }
    pub(crate) fn poison(&self) -> Result<()> {
        let (q, k, v) = self.attention.block_buffers();
        for b in self.bufs.iter().chain([q, k, v]) {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        self.attention.poison()?;
        self.mlp.poison()?;
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize, splits: usize) -> Result<Collected> {
        ensure!(
            self.prepared_shape().is_some_and(|s| s.rows == rows),
            "collect layer shape"
        );
        let l = self;
        let (q, k, v) = l.attention.block_buffers();
        let mut canary_bytes_checked = 0;
        let mut read = |b: &Buffer<'_>, cols: usize| -> Result<Vec<f32>> {
            let mut raw = vec![0; b.byte_len()];
            b.download(&mut raw)?;
            let active = rows * cols * 2;
            ensure!(
                raw[active..].iter().all(|&v| v == 0xff),
                "layer stage trailing canary"
            );
            canary_bytes_checked += raw.len() - active;
            let out: Vec<_> = raw[..active]
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
                .collect();
            ensure!(
                out.iter().all(|v| v.is_finite()),
                "layer stage nonfinite/unwritten"
            );
            Ok(out)
        };
        let mut stages = vec![
            ("attn_norm", read(&l.bufs[0], H)?),
            ("attn_dynamic", read(&l.bufs[1], DYN)?),
            ("attn_prepare_conv", read(&l.bufs[2], H)?),
            ("q_projection", read(q, QH * D)?),
            ("k_projection", read(k, KH * D)?),
            ("v_projection", read(v, KH * D)?),
        ];
        let a = l.attention.collect(splits)?;
        stages.extend([
            ("q_rope", a.q),
            ("k_rope", a.k),
            ("attention", a.out),
            ("o_projection", read(&l.bufs[3], H)?),
            ("attn_finish_residual", read(&l.bufs[4], H)?),
        ]);
        let m = l.mlp.collect(rows)?;
        stages.extend(m.stages);
        let mut counters_checked = m.counters_checked;
        for linear in [&l.dynamic, &l.q, &l.k, &l.v, &l.o] {
            counters_checked += linear.check_counters()?;
        }
        Ok(Collected {
            stages,
            canary_bytes_checked: canary_bytes_checked
                + a.canary_bytes_checked
                + m.canary_bytes_checked,
            counters_checked,
            partial_values_checked: a.partial_values_checked,
        })
    }
    pub fn plan(&self, rows: usize, splits: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            self.shape.get().is_some_and(|s| s.rows == rows)
                && self.shape.get() == self.attention.prepared_shape(),
            "prepare layer before planning"
        );
        let t = Instant::now();
        let calls = self.record(rows, splits)?;
        let execute = if graph {
            Executor::Graph(self.device.capture(calls)?)
        } else {
            Executor::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            rows,
            splits,
            execute,
            setup_us: t.elapsed().as_secs_f64() * 1e6,
        })
    }
}
pub(crate) struct Collected {
    pub stages: Vec<(&'static str, Vec<f32>)>,
    pub canary_bytes_checked: usize,
    pub counters_checked: usize,
    pub partial_values_checked: usize,
}
enum Executor<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Layer<'m, 'd>,
    rows: usize,
    splits: usize,
    execute: Executor<'a, 'm, 'd>,
    pub setup_us: f64,
}
pub struct Run {
    pub stages: Vec<(&'static str, Vec<f32>)>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub canary_bytes_checked: usize,
    pub counters_checked: usize,
    pub partial_values_checked: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, repetitions: usize, wait_each: bool) -> Result<Run> {
        let l = self.owner;
        let s = l.shape.get().context("layer input not prepared")?;
        ensure!(
            s.rows == self.rows
                && l.attention.prepared_shape() == Some(s)
                && (1..=32).contains(&repetitions),
            "layer changed shape/repetitions"
        );
        ensure!(
            !wait_each || matches!(&self.execute, Executor::Calls(_)),
            "layer graph wait-each"
        );
        l.poison()?;
        l.device.sync()?;
        let t = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repetitions {
            match &self.execute {
                Executor::Calls(c) => {
                    for call in c {
                        let t = Instant::now();
                        call.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            l.device.sync()?;
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
        let wait = Instant::now();
        l.device.sync()?;
        completion_us += wait.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let collected = l.collect(self.rows, self.splits)?;
        Ok(Run {
            stages: collected.stages,
            total_us,
            enqueue_us,
            completion_us,
            canary_bytes_checked: collected.canary_bytes_checked,
            counters_checked: collected.counters_checked,
            partial_values_checked: collected.partial_values_checked,
        })
    }
}
