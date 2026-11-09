//! R10f five connected original-BF16 DFlash layers plus final norm, one stream.
//! Resident context replay; context append, vocabulary head and selector excluded.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_dflash_layer::Layer,
    dflash_attention::Shape,
    dflash_body::{Fixture, LAYERS, Weights},
    dflash_layer::Fixture as LayerFixture,
    dflash_mlp::{EPS, H, MAX_ROWS},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, time::Instant};
const GUARD: usize = 256;
fn bytes(x: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
pub struct Body<'m, 'd> {
    device: &'d Device,
    layers: Vec<Layer<'m, 'd>>,
    rms: Kernel<'m, 'd>,
    norm: Buffer<'d>,
    input: Buffer<'d>,
    output: Buffer<'d>,
    shape: Cell<Option<Shape>>,
    ticket: Cell<u64>,
}
impl<'m, 'd> Body<'m, 'd> {
    pub fn new(
        d: &'d Device,
        gemm: &'m Module<'d>,
        ops: &'m Module<'d>,
        attn: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        ensure!(
            w.layers.len() == LAYERS && w.norm.len() == H,
            "body weight geometry"
        );
        let norm = d.alloc(H * 2)?;
        norm.upload(bytes(&w.norm))?;
        let mut check = vec![0; H * 2];
        norm.download(&mut check)?;
        ensure!(check == bytes(&w.norm), "body norm transfer");
        Ok(Self {
            device: d,
            layers: w
                .layers
                .iter()
                .map(|w| Layer::new(d, gemm, ops, attn, w))
                .collect::<Result<_>>()?,
            rms: ops.kernel("dflash_rms")?,
            norm,
            input: d.alloc(MAX_ROWS * H * 2)?,
            output: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            shape: Cell::new(None),
            ticket: Cell::new(0),
        })
    }
    pub fn description(&self) -> Value {
        json!({"backend":"native Rust/CUDA/libtinynv","layers":LAYERS,"dependent_calls":96,
            "intermediate_host_transfers":0,"original_bf16_weights":true,
            "scope":"five body layers plus final norm, resident per-layer context",
            "excludes":["feature encoder/context append","target vocabulary head","selector","sampling","cross-device loop"],
            "layer_details":self.layers.iter().map(Layer::description).collect::<Vec<_>>()})
    }
    pub fn prepare(&self, f: &Fixture) -> Result<f64> {
        self.shape.set(None);
        self.ticket.set(
            self.ticket
                .get()
                .checked_add(1)
                .context("body ticket overflow")?,
        );
        f.validate()?;
        let start = Instant::now();
        self.input.upload(bytes(&f.x))?;
        let mut check = vec![0; f.x.len() * 2];
        self.input.download(&mut check)?;
        ensure!(check == bytes(&f.x), "body input transfer");
        for (layer, (k, v)) in self.layers.iter().zip(&f.contexts) {
            layer.prepare_context(f.shape, k, v)?;
        }
        self.shape.set(Some(f.shape));
        Ok(start.elapsed().as_secs_f64() * 1e6)
    }
    pub fn check_context(&self, f: &Fixture) -> Result<usize> {
        f.validate()?;
        ensure!(
            self.shape.get() == Some(f.shape),
            "body context shape changed"
        );
        let mut count = 0;
        for (layer, (k, v)) in self.layers.iter().zip(&f.contexts) {
            count += layer.check_context(&LayerFixture {
                shape: f.shape,
                x: vec![],
                ck: k.clone(),
                cv: v.clone(),
            })?;
        }
        Ok(count)
    }
    pub fn plan(&self, splits: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        let shape = self.shape.get().context("prepare body before plan")?;
        ensure!(
            self.layers
                .iter()
                .all(|l| l.prepared_shape() == Some(shape)),
            "body layer shape changed"
        );
        let start = Instant::now();
        let mut calls = vec![];
        let mut previous = &self.input;
        for layer in &self.layers {
            calls.extend(layer.record_from(previous, shape.rows, splits)?);
            previous = layer.output_buffer();
        }
        // Audited fixed-width RMS ABI; final input/output/norm are disjoint and
        // the plan retains every layer and buffer until graph completion.
        unsafe {
            calls.push(self.rms.prepare(
                vec![
                    Arg::Buffer(previous),
                    Arg::Buffer(&self.norm),
                    Arg::Buffer(&self.output),
                    Arg::U32(H as u32),
                    Arg::F32(EPS as f32),
                ],
                [shape.rows as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        ensure!(calls.len() == 96, "body call count");
        let execute = if graph {
            Executor::Graph(self.device.capture(calls)?)
        } else {
            Executor::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            shape,
            splits,
            ticket: self.ticket.get(),
            execute,
            setup_us: start.elapsed().as_secs_f64() * 1e6,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Body<'m, 'd>,
    shape: Shape,
    splits: usize,
    ticket: u64,
    execute: Executor<'a, 'm, 'd>,
    pub setup_us: f64,
}
pub struct Run {
    pub layers: Vec<Vec<(&'static str, Vec<f32>)>>,
    pub output: Vec<f32>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub canary_bytes_checked: usize,
    pub counters_checked: usize,
    pub partial_values_checked: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, repetitions: usize, wait_each: bool) -> Result<Run> {
        let b = self.owner;
        ensure!(
            b.ticket.get() == self.ticket
                && b.shape.get() == Some(self.shape)
                && b.layers
                    .iter()
                    .all(|l| l.prepared_shape() == Some(self.shape)),
            "stale or incomplete body plan"
        );
        ensure!(
            (1..=16).contains(&repetitions)
                && (!wait_each || matches!(&self.execute, Executor::Calls(_))),
            "body repeats/wait policy"
        );
        for l in &b.layers {
            l.poison()?;
        }
        b.output.upload(&vec![0xff; b.output.byte_len()])?;
        b.device.sync()?;
        let start = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repetitions {
            match &self.execute {
                Executor::Calls(calls) => {
                    for call in calls {
                        let t = Instant::now();
                        call.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            b.device.sync()?;
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
        b.device.sync()?;
        completion_us += t.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut layers = vec![];
        let mut canary_bytes_checked = 0;
        let mut counters_checked = 0;
        let mut partial_values_checked = 0;
        for l in &b.layers {
            let c = l.collect(self.shape.rows, self.splits)?;
            layers.push(c.stages);
            canary_bytes_checked += c.canary_bytes_checked;
            counters_checked += c.counters_checked;
            partial_values_checked += c.partial_values_checked;
        }
        let mut raw = vec![0; b.output.byte_len()];
        b.output.download(&mut raw)?;
        let active = self.shape.rows * H * 2;
        ensure!(
            raw[active..].iter().all(|&v| v == 0xff),
            "body output trailing guard"
        );
        canary_bytes_checked += raw.len() - active;
        let output = raw[..active]
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "body final output finite/written"
        );
        Ok(Run {
            layers,
            output,
            total_us,
            enqueue_us,
            completion_us,
            canary_bytes_checked,
            counters_checked,
            partial_values_checked,
        })
    }
}
