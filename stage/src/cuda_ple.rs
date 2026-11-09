//! CUDA PLE: host selects packed rows; CUDA owns unpack, projections, nonlinear
//! work and convolution state. Candidate execution never commits state implicitly.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::Linear,
    ple::{self, Core, F, H, HC, HEADS, HISTORY, MAX_ROWS, PackedRows, Weights},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    path::Path,
    time::Instant,
};
pub const SHA: &str = "aa120866f0b456ed60491e07fa5c3370aaa8146a3b385f7fa7ad90dc9739876e";
const GUARD: usize = 256;
pub const NAMES: [&str; 15] = [
    "embedding",
    "key",
    "value",
    "key_norm",
    "query_norm",
    "dots",
    "scaled",
    "roots",
    "gates",
    "gated",
    "normalized",
    "conv",
    "activated",
    "addend",
    "output",
];
const WIDTHS: [usize; 15] = [H, F, H, F, F, HC, HC, HC, HC, F, F, F, F, F, F];
/// R53: QWEN_R53_TAILX=1 loads lowbit_ple_r53 (ple_normx, ple_norm_gatex: weights
/// and the gate input requested before the store loops; same values).
fn r53_ple_sha() -> String {
    let pin: serde_json::Value = serde_json::from_str(include_str!("../pins/R53-PLE-CUDA-BUILD.json")).expect("R53-PLE record");
    pin["cubin_sha256"].as_str().expect("R53-PLE cubin_sha256").to_string()
}
/// The cubin this process loads for the PLE kernels.
pub fn active_sha() -> String {
    if crate::cuda_lowbit_io::r53_tailx_enabled() { r53_ple_sha() } else { SHA.to_string() }
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if crate::cuda_lowbit_io::r53_tailx_enabled() {
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_ple_r53.sm120.cubin"), &r53_ple_sha());
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_ple.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![0xff; b.byte_len()])
}
fn raw(b: &Buffer<'_>, n: usize) -> Result<Vec<u8>> {
    ensure!(n + GUARD <= b.byte_len(), "PLE read bound");
    let mut out = vec![0; b.byte_len()];
    b.download(&mut out)?;
    ensure!(out[n..].iter().all(|&v| v == 0xff), "PLE tail canary");
    out.truncate(n);
    Ok(out)
}
fn read(b: &Buffer<'_>, n: usize) -> Result<Vec<u16>> {
    let out: Vec<_> = raw(b, n * 2)?
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        out.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "PLE finite/fully-written BF16"
    );
    Ok(out)
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub fused: bool,
    pub projection: usize,
}
pub struct Ple<'m, 'd> {
    d: &'d Device,
    hash: ple::Hash,
    linears: [Linear<'m, 'd>; 2],
    weights: Vec<Buffer<'d>>,
    x: Buffer<'d>,
    packed: Buffer<'d>,
    initial: Buffer<'d>,
    next: Buffer<'d>,
    fields: Vec<Buffer<'d>>,
    projection_raw: [Buffer<'d>; 2],
    unpack: Kernel<'m, 'd>,
    norm_pair: Kernel<'m, 'd>,
    gate: Kernel<'m, 'd>,
    fused: Kernel<'m, 'd>,
    norm: Kernel<'m, 'd>,
    conv: Kernel<'m, 'd>,
    gather: Kernel<'m, 'd>,
    restore: Kernel<'m, 'd>,
    rows: Cell<usize>,
    epoch: Cell<u64>,
    has_state: Cell<bool>,
    valid: Cell<Option<u64>>,
    previous: Cell<[u32; 2]>,
    tokens: RefCell<Vec<u32>>,
}
impl<'m, 'd> Ple<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        w.validate()?;
        let upload = |v: &[u16]| -> Result<Buffer<'d>> {
            let b = d.alloc(v.len() * 2)?;
            b.upload(bytes(v))?;
            let mut r = vec![0; v.len() * 2];
            b.download(&mut r)?;
            ensure!(r == bytes(v), "PLE weight readback");
            Ok(b)
        };
        Ok(Self {
            d,
            hash: w.hash.clone(),
            linears: [
                Linear::new(d, affine, &w.key)?,
                Linear::new(d, affine, &w.value)?,
            ],
            weights: w
                .norms
                .iter()
                .chain(std::iter::once(&w.conv))
                .map(|v| upload(v))
                .collect::<Result<_>>()?,
            x: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            packed: d.alloc(MAX_ROWS * HEADS * 100 + GUARD)?,
            initial: d.alloc(HISTORY * F * 2 + GUARD)?,
            next: d.alloc(HISTORY * F * 2 + GUARD)?,
            fields: WIDTHS
                .iter()
                .map(|&w| d.alloc(MAX_ROWS * w * 2 + GUARD))
                .collect::<Result<_>>()?,
            projection_raw: [
                d.alloc(MAX_ROWS * F * 4 + GUARD)?,
                d.alloc(MAX_ROWS * H * 4 + GUARD)?,
            ],
            unpack: ops.kernel("ple_unpack")?,
            norm_pair: ops.kernel("ple_norm_pair")?,
            gate: ops.kernel("ple_gate")?,
            fused: ops.kernel(if crate::cuda_lowbit_io::r53_tailx_enabled() { "ple_norm_gatex" } else { "ple_norm_gate" })?,
            norm: ops.kernel(if crate::cuda_lowbit_io::r53_tailx_enabled() { "ple_normx" } else { "ple_norm" })?,
            conv: ops.kernel("ple_conv")?,
            gather: ops.kernel("ple_commit")?,
            restore: ops.kernel("ple_restore")?,
            rows: Cell::new(0),
            epoch: Cell::new(0),
            has_state: Cell::new(false),
            valid: Cell::new(None),
            previous: Cell::new([ple::EOS; 2]),
            tokens: RefCell::new(vec![]),
        })
    }
    pub fn description(&self) -> Value {
        let extra: usize = [&self.x, &self.packed, &self.initial, &self.next]
            .into_iter()
            .chain(&self.weights)
            .chain(&self.fields)
            .chain(&self.projection_raw)
            .map(|b| b.byte_len())
            .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":active_sha(),"extra_bytes":extra,
            "linears":self.linears.iter().map(|l|l.description()).collect::<Vec<_>>(),
            "packed_host_bytes_per_token":1600,"host_dequantization":false,"intermediate_host_handoffs":0,
            "state":"9 normalized rows on CUDA plus 2 token IDs on host; atomic logical prefix commit",
            "kernels":[self.unpack.description(),self.norm_pair.description(),self.gate.description(),self.fused.description(),self.norm.description(),self.conv.description(),self.gather.description(),self.restore.description()]})
    }
    fn invalidate(&self) {
        self.rows.set(0);
        self.valid.set(None);
        self.epoch.set(self.epoch.get().wrapping_add(1));
    }
    pub fn initialize(&self, previous: [u32; 2], initial: &[u16]) -> Result<()> {
        self.invalidate();
        self.has_state.set(false);
        ensure!(
            previous.iter().all(|&v| v < ple::VOCAB)
                && initial.len() == HISTORY * F
                && initial.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "PLE initial state"
        );
        poison(&self.initial)?;
        self.initial.upload(bytes(initial))?;
        ensure!(
            read(&self.initial, HISTORY * F)? == initial,
            "PLE initial readback"
        );
        self.previous.set(previous);
        self.tokens.borrow_mut().clear();
        self.has_state.set(true);
        Ok(())
    }
    /// Input validation checks row IDs against this exact committed token history.
    pub fn prepare(&self, tokens: &[u32], packed: &PackedRows, x: &[u16]) -> Result<()> {
        self.invalidate();
        ensure!(
            x.len() == tokens.len() * F && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "PLE stream shape/finite"
        );
        self.prepare_tokens(tokens, packed)?;
        poison(&self.x)?;
        self.x.upload(bytes(x))?;
        ensure!(read(&self.x, x.len())? == x, "PLE input readback");
        Ok(())
    }
    pub(crate) fn prepare_tokens(&self, tokens: &[u32], packed: &PackedRows) -> Result<()> {
        self.invalidate();
        ensure!(self.has_state.get(), "PLE uninitialized state");
        packed.validate()?;
        ensure!(
            packed.ids == self.hash.rows(self.previous.get(), tokens)?,
            "PLE wrong-prefix packed row IDs"
        );
        let mut p = packed.codes.clone();
        p.extend_from_slice(bytes(&packed.scales));
        p.extend_from_slice(bytes(&packed.biases));
        ensure!(
            p.len() == tokens.len() * HEADS * 100,
            "PLE packed payload layout"
        );
        poison(&self.packed)?;
        self.packed.upload(&p)?;
        ensure!(
            raw(&self.packed, p.len())? == p,
            "PLE token payload readback"
        );
        self.tokens.replace(tokens.to_vec());
        self.rows.set(tokens.len());
        Ok(())
    }
    /// R56 graph-step prepare: the packed rows are uploaded once (no poison, no readback).
    pub(crate) fn prepare_tokens_lean(&self, tokens: &[u32], packed: &PackedRows) -> Result<()> {
        self.invalidate();
        ensure!(self.has_state.get(), "PLE uninitialized state");
        ensure!(
            packed.ids == self.hash.rows(self.previous.get(), tokens)?,
            "PLE wrong-prefix packed row IDs"
        );
        let mut p = Vec::with_capacity(tokens.len() * HEADS * 100);
        p.extend_from_slice(&packed.codes);
        p.extend_from_slice(bytes(&packed.scales));
        p.extend_from_slice(bytes(&packed.biases));
        ensure!(p.len() == tokens.len() * HEADS * 100, "PLE packed payload layout");
        self.packed.upload(&p)?;
        self.tokens.replace(tokens.to_vec());
        self.rows.set(tokens.len());
        Ok(())
    }
    pub fn state(&self) -> Result<([u32; 2], Vec<u16>)> {
        ensure!(self.has_state.get(), "PLE unavailable state");
        Ok((self.previous.get(), read(&self.initial, HISTORY * F)?))
    }
    pub fn output(&self) -> &Buffer<'d> {
        &self.fields[14]
    }
    pub(crate) fn input(&self) -> &Buffer<'d> {
        &self.x
    }
    pub(crate) fn invalidate_output(&self) {
        self.valid.set(None);
    }
    pub fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && rows == self.rows.get()
                && self.has_state.get()
                && p.projection < 4
                && x.byte_len() >= rows * F * 2,
            "PLE record shape/state"
        );
        ensure!(
            self.fields
                .iter()
                .chain(&self.projection_raw)
                .chain(&self.weights)
                .chain([&self.initial, &self.next, &self.packed])
                .all(|b| !x.aliases(b)),
            "PLE input/output alias"
        );
        let b = &self.fields;
        let w = &self.weights;
        let grid = [HC as u32, rows as u32, 1];
        let block = [256, 1, 1];
        let mut out = vec![unsafe {
            self.unpack.prepare(
                vec![
                    Arg::Buffer(&self.packed),
                    Arg::Buffer(&b[0]),
                    Arg::U32(rows as u32),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                block,
            )?
        }];
        for i in 0..2 {
            out.push(self.linears[i].record_into(
                &b[0],
                &self.projection_raw[i],
                &b[1 + i],
                rows,
                p.projection,
            )?);
        }
        if p.fused {
            out.push(unsafe {
                self.fused.prepare(
                    vec![
                        Arg::Buffer(&b[1]),
                        Arg::Buffer(x),
                        Arg::Buffer(&w[0]),
                        Arg::Buffer(&w[1]),
                        Arg::Buffer(&b[2]),
                        Arg::Buffer(&b[3]),
                        Arg::Buffer(&b[4]),
                        Arg::Buffer(&b[5]),
                        Arg::Buffer(&b[6]),
                        Arg::Buffer(&b[7]),
                        Arg::Buffer(&b[8]),
                        Arg::Buffer(&b[9]),
                    ],
                    grid,
                    block,
                )?
            });
        } else {
            out.push(unsafe {
                self.norm_pair.prepare(
                    vec![
                        Arg::Buffer(&b[1]),
                        Arg::Buffer(x),
                        Arg::Buffer(&w[0]),
                        Arg::Buffer(&w[1]),
                        Arg::Buffer(&b[3]),
                        Arg::Buffer(&b[4]),
                    ],
                    grid,
                    block,
                )?
            });
            out.push(unsafe {
                self.gate.prepare(
                    vec![
                        Arg::Buffer(&b[3]),
                        Arg::Buffer(&b[4]),
                        Arg::Buffer(&b[2]),
                        Arg::Buffer(&b[5]),
                        Arg::Buffer(&b[6]),
                        Arg::Buffer(&b[7]),
                        Arg::Buffer(&b[8]),
                        Arg::Buffer(&b[9]),
                    ],
                    grid,
                    block,
                )?
            });
        }
        out.push(unsafe {
            self.norm.prepare(
                vec![Arg::Buffer(&b[9]), Arg::Buffer(&w[2]), Arg::Buffer(&b[10])],
                grid,
                block,
            )?
        });
        out.push(unsafe {
            self.conv.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&b[9]),
                    Arg::Buffer(&b[10]),
                    Arg::Buffer(&self.initial),
                    Arg::Buffer(&w[3]),
                    Arg::Buffer(&b[11]),
                    Arg::Buffer(&b[12]),
                    Arg::Buffer(&b[13]),
                    Arg::Buffer(&b[14]),
                    Arg::U32(rows as u32),
                ],
                [(rows * F).div_ceil(256) as u32, 1, 1],
                block,
            )?
        });
        Ok(out)
    }
    pub(crate) fn poison(&self) -> Result<()> {
        for b in self.fields.iter().chain(&self.projection_raw) {
            poison(b)?;
        }
        Ok(())
    }
    pub(crate) fn collect(&self) -> Result<Run> {
        let rows = self.rows.get();
        ensure!(rows > 0, "PLE missing rows");
        let mut fields = self
            .fields
            .iter()
            .zip(WIDTHS)
            .map(|(b, w)| read(b, rows * w))
            .collect::<Result<Vec<_>>>()?;
        let mut projection_raw = vec![];
        for i in 0..2 {
            let f = raw(&self.projection_raw[i], rows * WIDTHS[i + 1] * 4)?
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            ensure!(
                f.iter()
                    .zip(&fields[i + 1])
                    .all(|(&v, &b)| v.is_finite() && bf16::from_f32(v).to_bits() == b),
                "PLE projection rounding/finite"
            );
            projection_raw.push(f);
        }
        let mut get = |i| std::mem::take(&mut fields[i]);
        let embedding = get(0);
        let key = get(1);
        let value = get(2);
        let core = Core {
            key_norm: get(3),
            query_norm: get(4),
            dots: get(5),
            scaled: get(6),
            roots: get(7),
            gates: get(8),
            gated: get(9),
            normalized: get(10),
            conv: get(11),
            activated: get(12),
            addend: get(13),
            output: get(14),
        };
        self.valid.set(Some(self.epoch.get()));
        Ok(Run {
            embedding,
            key,
            value,
            core,
            projection_raw,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        ensure!((1..=16).contains(&repetitions), "PLE repetition bound");
        let rows = self.rows.get();
        let mut c = self.record_from(&self.x, rows, p)?;
        let calls = c.len();
        let exec = if graph {
            for _ in 1..repetitions {
                c.extend(self.record_from(&self.x, rows, p)?);
            }
            Executor::Graph(self.d.capture(c)?)
        } else {
            Executor::Prepared(c)
        };
        Ok(Plan {
            owner: self,
            epoch: self.epoch.get(),
            rows,
            exec,
            repetitions,
            calls,
        })
    }
    /// R56 lean drafting commit: the history gather/restore kernels as prepared calls.
    pub(crate) fn record_commit(&self, accepted: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        ensure!(self.rows.get() > 0 && accepted <= self.rows.get(), "PLE lean prefix");
        let gather = unsafe {
            self.gather.prepare(
                vec![
                    Arg::Buffer(&self.initial),
                    Arg::Buffer(&self.fields[10]),
                    Arg::Buffer(&self.next),
                    Arg::U32(accepted as u32),
                ],
                [(HISTORY * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let restore = unsafe {
            self.restore.prepare(
                vec![Arg::Buffer(&self.next), Arg::Buffer(&self.initial)],
                [(HISTORY * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        Ok(vec![gather, restore])
    }
    /// R56: the committed 9-row convolution history; the two previous tokens are host state.
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        vec![&self.initial]
    }
    pub(crate) fn previous_tokens(&self) -> [u32; 2] {
        self.previous.get()
    }
    pub(crate) fn set_previous_tokens(&self, p: [u32; 2]) {
        self.invalidate();
        self.previous.set(p);
        self.tokens.borrow_mut().clear();
        self.has_state.set(true);
    }
    pub(crate) fn finish_commit(&self, accepted: usize) -> Result<()> {
        let next_tokens = ple::accepted_tokens(self.previous.get(), &self.tokens.borrow(), accepted)?;
        self.invalidate();
        self.previous.set(next_tokens);
        self.tokens.borrow_mut().clear();
        self.has_state.set(true);
        Ok(())
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        ensure!(
            self.valid.get() == Some(self.epoch.get()) && self.rows.get() > 0,
            "PLE commit requires completed current output"
        );
        let next_tokens =
            ple::accepted_tokens(self.previous.get(), &self.tokens.borrow(), accepted)?;
        self.invalidate();
        self.has_state.set(false);
        poison(&self.next)?;
        let gather = unsafe {
            self.gather.prepare(
                vec![
                    Arg::Buffer(&self.initial),
                    Arg::Buffer(&self.fields[10]),
                    Arg::Buffer(&self.next),
                    Arg::U32(accepted as u32),
                ],
                [(HISTORY * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let restore = unsafe {
            self.restore.prepare(
                vec![Arg::Buffer(&self.next), Arg::Buffer(&self.initial)],
                [(HISTORY * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        gather.enqueue()?;
        restore.enqueue()?;
        self.d.sync()?;
        self.previous.set(next_tokens);
        self.tokens.borrow_mut().clear();
        self.has_state.set(true);
        Ok(())
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Ple<'m, 'd>,
    epoch: u64,
    rows: usize,
    exec: Executor<'a, 'm, 'd>,
    repetitions: usize,
    pub calls: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Run {
    pub embedding: Vec<u16>,
    pub key: Vec<u16>,
    pub value: Vec<u16>,
    pub core: Core,
    pub projection_raw: Vec<Vec<f32>>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Run {
    pub fn fields(&self) -> [(&'static str, &[u16]); 15] {
        let c = &self.core;
        [
            (NAMES[0], &self.embedding),
            (NAMES[1], &self.key),
            (NAMES[2], &self.value),
            (NAMES[3], &c.key_norm),
            (NAMES[4], &c.query_norm),
            (NAMES[5], &c.dots),
            (NAMES[6], &c.scaled),
            (NAMES[7], &c.roots),
            (NAMES[8], &c.gates),
            (NAMES[9], &c.gated),
            (NAMES[10], &c.normalized),
            (NAMES[11], &c.conv),
            (NAMES[12], &c.activated),
            (NAMES[13], &c.addend),
            (NAMES[14], &c.output),
        ]
    }
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        ensure!(
            a.has_state.get() && a.epoch.get() == self.epoch && a.rows.get() == self.rows,
            "PLE stale plan"
        );
        a.valid.set(None);
        a.poison()?;
        a.d.sync()?;
        let start = Instant::now();
        match &self.exec {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(c) => {
                for _ in 0..self.repetitions {
                    for k in c {
                        k.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut out = a.collect()?;
        out.total_us = total_us;
        out.enqueue_us = enqueue_us;
        out.wait_us = wait_us;
        a.valid.set(Some(self.epoch));
        Ok(out)
    }
}
