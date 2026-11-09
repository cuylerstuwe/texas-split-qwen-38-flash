//! Native QK normalization/RoPE, affine8 append, packed SDPA and output gate.
//! Q/K/V/output projections and QSA selection are separate qualified components
//! awaiting one-chain composition. No host work between this module's stages.
use crate::{
    metal_ops::{Buffer, Obj, alloc, upload},
    qsa,
    target_attention::*,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const GUARD: usize = 64;
const BPOISON: u16 = 0x7fc1;
const FPOISON: u32 = 0x7fc12345;
const CPOISON: u8 = 0xcd;
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    offset: u32,
    rows: u32,
    base: u32,
    capacity: u32,
    sparse: u32,
    splits: u32,
    log2_scores: u32,
}
#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Simd1,
    Simd8,
    Simd16,
    Nax,
    Nax4,
    Nax8,
    Nax16,
}
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Simd1 => "simd1",
            Self::Simd8 => "simd8",
            Self::Simd16 => "simd16",
            Self::Nax => "nax",
            Self::Nax4 => "nax4",
            Self::Nax8 => "nax8",
            Self::Nax16 => "nax16",
        }
    }
    fn splits(self) -> usize {
        match self {
            Self::Nax4 => 4,
            Self::Simd8 | Self::Nax8 => 8,
            Self::Simd16 | Self::Nax16 => 16,
            _ => 1,
        }
    }
}
pub(crate) struct Ops {
    pipelines: Vec<Obj<dyn MTLComputePipelineState>>,
}
pub(crate) struct Workspace {
    qg: Buffer,
    kraw: Buffer,
    vraw: Buffer,
    query: Buffer,
    key: Buffer,
    gate: Buffer,
    cache: Vec<Buffer>,
    ids: Buffer,
    partial: Buffer,
    stats: Buffer,
    attention: Buffer,
    gated: Buffer,
    capacity: usize,
}
fn poison16(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(BPOISON);
    }
}
fn poison32(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            .fill(FPOISON);
    }
}
fn poison8(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u8>().as_ptr(), b.length())
            .fill(CPOISON);
    }
}
fn read16(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "attention BF16 read bound");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        v[n..].iter().all(|&v| v == BPOISON),
        "attention BF16 canary"
    );
    ensure!(
        v[..n].iter().all(|&v| bf16::from_bits(v).is_finite()),
        "attention BF16 nonfinite/unwritten"
    );
    Ok(v[..n].to_vec())
}
fn read8(b: &Buffer, n: usize) -> Result<Vec<u8>> {
    ensure!(n <= b.length(), "attention packed read bound");
    let v = unsafe { std::slice::from_raw_parts(b.contents().cast::<u8>().as_ptr(), b.length()) };
    ensure!(
        v[n..].iter().all(|&v| v == CPOISON),
        "attention packed canary"
    );
    Ok(v[..n].to_vec())
}
fn read32(b: &Buffer, n: usize, stats: bool) -> Result<Vec<f32>> {
    ensure!(n <= b.length() / 4, "attention FP32 read bound");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        v[n..].iter().all(|&v| v == FPOISON),
        "attention FP32 canary"
    );
    let v: Vec<_> = v[..n].iter().map(|&v| f32::from_bits(v)).collect();
    if stats {
        for p in v.chunks_exact(2) {
            ensure!(
                p[1].is_finite()
                    && p[1] >= 0.
                    && if p[1] == 0. {
                        p[0] == f32::NEG_INFINITY
                    } else {
                        p[0].is_finite()
                    },
                "attention partial mass/max"
            );
        }
    } else {
        ensure!(
            v.iter().all(|v| v.is_finite()),
            "attention FP32 nonfinite/unwritten"
        );
    }
    Ok(v)
}
impl Ops {
    fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        ensure!(
            d.name().to_string().contains("M5"),
            "NAX gate requires observed M5"
        );
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let source = [
            include_str!("../kernels/target_attention.metal"),
            include_str!("../kernels/target_attention_nax.metal"),
        ]
        .concat();
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| anyhow::anyhow!("attention MSL {e}"))?;
        let mut pipelines = vec![];
        for (name, threads) in [
            ("attn_prepare", 32),
            ("attn_pack", 32),
            ("attn_simd", 32),
            ("attn_merge", 256),
            ("attn_gate", 256),
            ("attn_nax", 64),
            ("attn_nax_split", 64),
        ] {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("attention function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("attention pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= threads
                    && p.staticThreadgroupMemoryLength() <= d.maxThreadgroupMemoryLength(),
                "attention launch geometry"
            );
            pipelines.push(p);
        }
        Ok(Self { pipelines })
    }
    fn dispatch(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        idx: usize,
        b: &[&Buffer],
        p: Params,
        grid: [usize; 3],
        threads: usize,
    ) {
        e.setComputePipelineState(&self.pipelines[idx]);
        for (i, b) in b.iter().enumerate() {
            unsafe {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
        }
        unsafe {
            e.setBytes_length_atIndex(
                NonNull::from(&p).cast(),
                std::mem::size_of::<Params>(),
                b.len(),
            );
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid[0],
                height: grid[1],
                depth: grid[2],
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        crate::metal_ops::barrier(&e);
    }
    fn encode(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &[Buffer; 3],
        s: &Workspace,
        p: Params,
        arm: Arm,
        ids: Option<&Buffer>,
    ) {
        self.dispatch(
            e,
            0,
            &[
                &s.qg, &s.kraw, &w[0], &w[1], &w[2], &s.query, &s.key, &s.gate,
            ],
            p,
            [QH + KH, p.rows as usize, 1],
            32,
        );
        self.dispatch(
            e,
            1,
            &[
                &s.key,
                &s.vraw,
                &s.cache[0],
                &s.cache[1],
                &s.cache[2],
                &s.cache[3],
                &s.cache[4],
                &s.cache[5],
            ],
            p,
            [p.rows as usize * META, 2, 1],
            32,
        );
        let mut b = vec![
            &s.query,
            &s.cache[0],
            &s.cache[3],
            &s.cache[1],
            &s.cache[2],
            &s.cache[4],
            &s.cache[5],
            ids.unwrap_or(&s.ids),
        ];
        if matches!(arm, Arm::Nax) {
            b.push(&s.attention);
            self.dispatch(e, 5, &b, p, [p.rows as usize, KH, 1], 64);
        } else {
            b.extend([&s.partial, &s.stats]);
            if matches!(arm, Arm::Nax4 | Arm::Nax8 | Arm::Nax16) {
                self.dispatch(e, 6, &b, p, [p.rows as usize, KH, p.splits as usize], 64);
            } else {
                self.dispatch(e, 2, &b, p, [QH, p.rows as usize, p.splits as usize], 32);
            }
            self.dispatch(
                e,
                3,
                &[&s.partial, &s.stats, &s.attention],
                p,
                [QH, p.rows as usize, 1],
                256,
            );
        }
        self.dispatch(
            e,
            4,
            &[&s.attention, &s.gate, &s.gated],
            p,
            [(p.rows as usize * Q).div_ceil(256), 1, 1],
            256,
        );
    }
}
/// R55 resident-model context bound (reference fixtures stop at MAX_REFERENCE_CTX). R58: raised
/// from 64K to the QSA position bound (262,144, the user's context target); past 64K it is
/// smoke-tested (greedy identity against Metal-only, speed, memory), not fixture-scored.
pub const RESIDENT_MAX_CTX: usize = crate::qsa::MAX_POSITION;
impl Workspace {
    fn new(d: &ProtocolObject<dyn MTLDevice>, capacity: usize) -> Result<Self> {
        // R55: the resident model may exceed the CPU reference's 32K context (the kernels
        // address positions to qsa::MAX_POSITION); 64K is an explicit, separately scored bound.
        ensure!(
            (MAX_ROWS..=RESIDENT_MAX_CTX + MAX_ROWS).contains(&capacity),
            "bounded attention component capacity"
        );
        let bf = |n| alloc(d, (n + GUARD) * 2);
        let mut cache = vec![];
        for _ in 0..2 {
            cache.push(alloc(d, capacity * KV + GUARD)?);
            cache.push(bf(capacity * META)?);
            cache.push(bf(capacity * META)?);
        }
        Ok(Self {
            qg: bf(MAX_ROWS * QG)?,
            kraw: bf(MAX_ROWS * KV)?,
            vraw: bf(MAX_ROWS * KV)?,
            query: bf(MAX_ROWS * Q)?,
            key: bf(MAX_ROWS * KV)?,
            gate: bf(MAX_ROWS * Q)?,
            cache,
            ids: alloc(d, (MAX_ROWS * qsa::TOPK + GUARD) * 4)?,
            partial: alloc(d, (MAX_ROWS * Q * 16 + GUARD) * 4)?,
            stats: alloc(d, (MAX_ROWS * QH * 16 * 2 + GUARD) * 4)?,
            attention: bf(MAX_ROWS * Q)?,
            gated: bf(MAX_ROWS * Q)?,
            capacity,
        })
    }
    fn prepare(&mut self, initial: &State) -> Result<()> {
        for b in [
            &self.qg,
            &self.kraw,
            &self.vraw,
            &self.query,
            &self.key,
            &self.gate,
            &self.attention,
            &self.gated,
        ] {
            poison16(b);
        }
        for b in [&self.ids, &self.partial, &self.stats] {
            poison32(b);
        }
        for (i, p) in [&initial.k, &initial.v].iter().enumerate() {
            poison8(&self.cache[i * 3]);
            poison16(&self.cache[i * 3 + 1]);
            poison16(&self.cache[i * 3 + 2]);
            upload(&self.cache[i * 3], &p.codes)?;
            upload(&self.cache[i * 3 + 1], &p.scales)?;
            upload(&self.cache[i * 3 + 2], &p.biases)?;
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Stages {
    pub prepared: Prepared,
    pub new_k: Packed,
    pub new_v: Packed,
    pub attention: Vec<u16>,
    pub gated: Vec<u16>,
}
pub struct Run {
    pub stages: Stages,
    pub state: State,
    pub gpu_us: f64,
    pub encode_submit_wait_us: f64,
    pub partial_values: usize,
}
pub struct Engine {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: Ops,
    weights: [Buffer; 3],
    s: Workspace,
    _single_owner: PhantomData<Rc<()>>,
}
impl Engine {
    pub fn new(n: &Norms, capacity: usize) -> Result<Self> {
        n.validate()?;
        let device = MTLCreateSystemDefaultDevice().context("attention device")?;
        let queue = device.newCommandQueue().context("attention queue")?;
        Self::new_on(n, capacity, device, queue)
    }
    pub(crate) fn new_on(
        n: &Norms,
        capacity: usize,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        n.validate()?;
        let q = alloc(&device, D * 2)?;
        upload(&q, &n.q)?;
        let k = alloc(&device, D * 2)?;
        upload(&k, &n.k)?;
        let f = alloc(&device, 32 * 4)?;
        upload(&f, &qsa::frequencies())?;
        Ok(Self {
            ops: Ops::new(&device)?,
            weights: [q, k, f],
            s: Workspace::new(&device, capacity)?,
            device,
            queue,
            _single_owner: PhantomData,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        json_info(self)
    }
    pub fn run(
        &mut self,
        input: &Input,
        initial: &State,
        selection: Option<&qsa::Selection>,
        arm: Arm,
        repeats: usize,
    ) -> Result<Run> {
        input.validate()?;
        initial.validate()?;
        let rows = input.rows();
        let total = initial.len + rows;
        ensure!(
            total <= self.s.capacity
                && total <= qsa::MAX_POSITION - initial.base
                && (1..=8).contains(&repeats),
            "attention append/repeat bounds"
        );
        if let Some(s) = selection {
            s.validate()?;
            ensure!(
                s.offset == initial.len && s.rows == rows,
                "attention selection alignment"
            );
        }
        self.prepare(initial, rows)?;
        upload(&self.s.qg, &input.qg)?;
        upload(&self.s.kraw, &input.k)?;
        upload(&self.s.vraw, &input.v)?;
        if let Some(s) = selection {
            upload(&self.s.ids, &s.ids)?;
        }
        let p = Params {
            offset: initial.len as u32,
            rows: rows as u32,
            base: initial.base as u32,
            capacity: self.s.capacity as u32,
            sparse: selection.is_some() as u32,
            splits: arm.splits() as u32,
            log2_scores: matches!(arm, Arm::Nax4 | Arm::Nax8 | Arm::Nax16) as u32,
        };
        let start = Instant::now();
        let cb = self.queue.commandBuffer().context("attention command")?;
        let enc = cb.computeCommandEncoder().context("attention encoder")?;
        for _ in 0..repeats {
            self.ops.encode(&enc, &self.weights, &self.s, p, arm, None);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "attention command failure {:?}",
            cb.error()
        );
        let gpu = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
        ensure!(gpu.is_finite() && gpu > 0., "attention GPU timestamps");
        ensure!(
            read16(&self.s.qg, rows * QG)? == input.qg
                && read16(&self.s.kraw, rows * KV)? == input.k
                && read16(&self.s.vraw, rows * KV)? == input.v,
            "attention input mutated"
        );
        let (stages, state, partial_values) = self.collect(initial, rows, arm)?;
        Ok(Run {
            stages,
            state,
            partial_values,
            gpu_us: gpu,
            encode_submit_wait_us: elapsed,
        })
    }
    pub(crate) fn prepare(&mut self, initial: &State, rows: usize) -> Result<()> {
        initial.validate()?;
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && initial.len + rows <= self.s.capacity
                && initial.len + rows <= qsa::MAX_POSITION - initial.base,
            "attention component append bounds"
        );
        self.s.prepare(initial)
    }
    /// R58 prefill: ([q norm, k norm, RoPE frequencies], the KV cache [k codes, k scales, k
    /// biases, v codes, v scales, v biases], capacity).
    pub(crate) fn prefill_parts(&self) -> (&[Buffer; 3], &[Buffer], usize) {
        (&self.weights, &self.s.cache, self.s.capacity)
    }
    pub(crate) fn projection_buffers(&self) -> [&Buffer; 3] {
        [&self.s.qg, &self.s.kraw, &self.s.vraw]
    }
    pub(crate) fn gated_buffer(&self) -> &Buffer {
        &self.s.gated
    }
    pub(crate) fn projected(&self, rows: usize) -> Result<Input> {
        Ok(Input {
            qg: read16(&self.s.qg, rows * QG)?,
            k: read16(&self.s.kraw, rows * KV)?,
            v: read16(&self.s.vraw, rows * KV)?,
        })
    }
    pub(crate) fn encode(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        initial: &State,
        rows: usize,
        arm: Arm,
        ids: Option<&Buffer>,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && initial.len + rows <= self.s.capacity,
            "attention encoded shape"
        );
        let p = Params {
            offset: initial.len as u32,
            rows: rows as u32,
            base: initial.base as u32,
            capacity: self.s.capacity as u32,
            sparse: ids.is_some() as u32,
            splits: arm.splits() as u32,
            log2_scores: matches!(arm, Arm::Nax4 | Arm::Nax8 | Arm::Nax16) as u32,
        };
        self.ops.encode(e, &self.weights, &self.s, p, arm, ids);
        Ok(())
    }
    pub(crate) fn collect(
        &self,
        initial: &State,
        rows: usize,
        arm: Arm,
    ) -> Result<(Stages, State, usize)> {
        let total = initial.len + rows;
        let mut packed = vec![];
        let mut new = vec![];
        for (i, old) in [&initial.k, &initial.v].iter().enumerate() {
            let c = Packed {
                codes: read8(&self.s.cache[i * 3], total * KV)?,
                scales: read16(&self.s.cache[i * 3 + 1], total * META)?,
                biases: read16(&self.s.cache[i * 3 + 2], total * META)?,
            };
            c.validate(total)?;
            ensure!(
                c.codes[..initial.len * KV] == old.codes
                    && c.scales[..initial.len * META] == old.scales
                    && c.biases[..initial.len * META] == old.biases,
                "committed KV mutated"
            );
            new.push(Packed {
                codes: c.codes[initial.len * KV..].to_vec(),
                scales: c.scales[initial.len * META..].to_vec(),
                biases: c.biases[initial.len * META..].to_vec(),
            });
            packed.push(c);
        }
        let n = if matches!(arm, Arm::Nax) {
            0
        } else {
            rows * QH * arm.splits()
        };
        read32(&self.s.partial, n * D, false)?;
        read32(&self.s.stats, n * 2, true)?;
        let stage = Stages {
            prepared: Prepared {
                q: read16(&self.s.query, rows * Q)?,
                k: read16(&self.s.key, rows * KV)?,
                gate: read16(&self.s.gate, rows * Q)?,
            },
            new_k: new.remove(0),
            new_v: new.remove(0),
            attention: read16(&self.s.attention, rows * Q)?,
            gated: read16(&self.s.gated, rows * Q)?,
        };
        let state = State {
            base: initial.base,
            len: total,
            k: packed.remove(0),
            v: packed.remove(0),
        };
        state.validate()?;
        Ok((stage, state, n * (D + 2)))
    }
}
fn json_info(e: &Engine) -> serde_json::Value {
    let s = &e.s;
    let b = [
        &s.qg,
        &s.kraw,
        &s.vraw,
        &s.query,
        &s.key,
        &s.gate,
        &s.ids,
        &s.partial,
        &s.stats,
        &s.attention,
        &s.gated,
    ];
    serde_json::json!({"device":e.device.name().to_string(),"capacity":s.capacity,"workspace_bytes":b.iter().map(|b|b.length()).sum::<usize>()+s.cache.iter().map(|b|b.length()).sum::<usize>(),"weight_bytes":e.weights.iter().map(|b|b.length()).sum::<usize>(),"threadgroup_bytes":e.ops.pipelines.iter().map(|p|p.staticThreadgroupMemoryLength()).collect::<Vec<_>>(),"host_intermediate_handoffs":0,"KV":"affine8 group64, token-major, BF16 metadata","safe_math":true,"full_attention_projections":false})
}
