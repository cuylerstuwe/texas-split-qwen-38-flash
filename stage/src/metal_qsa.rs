//! Native index projection/pooling/query/score/select chain. No host route or
//! score handoff. Pool ownership is append-only past the committed prefix;
//! the accepted raw tail is built on-device. Harness imports/exports state for
//! independent checks; whole-model resident state orchestration is separate.
use crate::{
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, alloc, dense_scratch_bytes, upload},
    qsa::*,
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const GUARD: usize = 64;
const POISON: u32 = 0x7fc12345;
const BPOISON: u16 = 0x7fc1;
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    offset: u32,
    rows: u32,
    base: u32,
    accepted: u32,
}
#[derive(Clone, Copy, Debug)]
pub enum Precision {
    Qmv,
    PackedF16x2,
}
impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qmv => "qmv",
            Self::PackedF16x2 => "packed_f16x2",
        }
    }
    pub(crate) fn arm(self) -> Arm {
        match self {
            Self::Qmv => Arm::Qmv4,
            Self::PackedF16x2 => Arm::FragmentPacked {
                splits: 8,
                residual: true,
            },
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Selector {
    Radix8,
    Radix11,
    /// R58: radix11 in two stages past 2,048 blocks (kernels/r58_qsa_select2.metal): chunk
    /// top-512s on many threadgroups per row, then one merge per row. Same ids, same order.
    Radix11Two,
}
impl Selector {
    pub fn name(self) -> &'static str {
        match self {
            Self::Radix8 => "radix8",
            Self::Radix11 => "radix11",
            Self::Radix11Two => "radix11_two_stage",
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Radix8 => 4,
            Self::Radix11 | Self::Radix11Two => 5,
        }
    }
}
/// R58 two-stage selection geometry: the chunk size and count for `nb` score blocks (at most 16
/// chunks; None up to 12,288 blocks, ~48K tokens, where one threadgroup per row is as fast).
fn select2_chunks(nb: usize) -> Option<(usize, usize)> {
    // r58-qsa-select2-check-01: slower than one threadgroup below ~16K blocks, up to ~3x faster at 64K
    if nb <= 12288 {
        return None;
    }
    let chunk = nb.div_ceil(16).max(2048).div_ceil(256) * 256;
    Some((chunk, nb.div_ceil(chunk)))
}
const SELECT2_MAX_CHUNKS: usize = 16;
pub(crate) struct Ops {
    dense: DenseOps,
    pipelines: Vec<Obj<dyn MTLComputePipelineState>>,
}
pub(crate) struct DeviceWeights {
    projection: DenseWeight,
    q: Buffer,
    k: Buffer,
    freq: Buffer,
}
pub(crate) struct Workspace {
    x: Buffer,
    raw: Buffer,
    qk: Buffer,
    query: Buffer,
    pool: Buffer,
    old_tail: Buffer,
    new_tail: Buffer,
    scores: Buffer,
    ids: Buffer,
    /// R58 two-stage selection: per row and chunk, the chunk's top-512 ids.
    cand: Buffer,
    scratch: Buffer,
    capacity: usize,
}
fn poison32(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            .fill(POISON);
    }
}
fn poison16(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(BPOISON);
    }
}
fn read_bits(b: &Buffer, n: usize) -> Result<Vec<u32>> {
    ensure!(n <= b.length() / 4, "QSA read bound");
    let a =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        a[n..].iter().all(|&v| v == POISON),
        "QSA 32-bit trailing canary"
    );
    Ok(a[..n].to_vec())
}
fn read32(b: &Buffer, n: usize) -> Result<Vec<f32>> {
    let a: Vec<_> = read_bits(b, n)?.into_iter().map(f32::from_bits).collect();
    ensure!(
        a.iter().all(|v| v.is_finite()),
        "QSA nonfinite/unwritten FP32"
    );
    Ok(a)
}
fn read16(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "QSA BF16 read bound");
    let a =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        a[n..].iter().all(|&v| v == BPOISON),
        "QSA BF16 trailing canary"
    );
    ensure!(
        a[..n].iter().all(|&v| bf16::from_bits(v).is_finite()),
        "QSA nonfinite/unwritten BF16"
    );
    Ok(a[..n].to_vec())
}
impl Ops {
    fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let source = [
            include_str!("../kernels/qsa_index.metal"),
            include_str!("../kernels/qsa_select.metal"),
            include_str!("../kernels/r58_qsa_select2.metal"),
        ]
        .concat();
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| anyhow::anyhow!("QSA MSL {e}"))?;
        let mut pipelines = vec![];
        for (name, threads) in [
            ("qsa_pool", 32),
            ("qsa_query", 32),
            ("qsa_score", 128),
            ("qsa_commit_tail", 128),
            ("qsa_select8", 256),
            ("qsa_select11", 256),
            ("qsa_select11_chunk", 256),
            ("qsa_select11_merge", 256),
        ] {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("QSA function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("QSA pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= threads
                    && p.staticThreadgroupMemoryLength() <= d.maxThreadgroupMemoryLength(),
                "QSA geometry"
            );
            pipelines.push(p);
        }
        Ok(Self {
            dense: DenseOps::new(d)?,
            pipelines,
        })
    }
    /// The top-512 selection of `rows` rows of scores into `ids` (with a barrier after it).
    fn encode_select(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, scores: &Buffer, ids: &Buffer, cand: &Buffer, p: Params, rows: usize, sel: Selector) {
        let nb = (p.offset + p.rows) as usize / RATIO;
        let two = match sel {
            Selector::Radix11Two => select2_chunks(nb),
            _ => None,
        };
        let Some((chunk, chunks)) = two else {
            self.dispatch(e, sel.index(), &[scores, ids], p, 1, rows, 256);
            return;
        };
        let q = [chunk as u32, chunks as u32, 0u32, 0u32];
        let set = |e: &ProtocolObject<dyn MTLComputeCommandEncoder>, idx: usize, bufs: &[&Buffer]| unsafe {
            e.setComputePipelineState(&self.pipelines[idx]);
            for (i, b) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, if i < 2 { i } else { i + 2 });
            }
            e.setBytes_length_atIndex(NonNull::from(&p).cast(), std::mem::size_of::<Params>(), 2);
            e.setBytes_length_atIndex(NonNull::from(&q).cast(), 16, 3);
        };
        set(e, 6, &[scores, cand]);
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: chunks, height: rows, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        crate::metal_ops::barrier(e);
        set(e, 7, &[scores, ids, cand]);
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: 1, height: rows, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        crate::metal_ops::barrier(e);
    }
    fn dispatch(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        idx: usize,
        b: &[&Buffer],
        p: Params,
        x: usize,
        y: usize,
        threads: usize,
    ) {
        if x == 0 {
            return;
        }
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
                width: x,
                height: y,
                depth: 1,
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
        w: &DeviceWeights,
        s: &Workspace,
        p: Params,
        precision: Option<Precision>,
        sel: Selector,
        input: Option<&Buffer>,
    ) -> Result<()> {
        if let Some(a) = precision {
            self.dense.encode(
                e,
                &w.projection,
                input.unwrap_or(&s.x),
                &s.raw,
                &s.qk,
                &s.scratch,
                p.rows as usize,
                0,
                a.arm(),
            )?;
            crate::metal_ops::barrier(&e);
        }
        self.dispatch(
            e,
            0,
            &[&s.qk, &s.old_tail, &w.k, &w.freq, &s.pool],
            p,
            ((p.offset % 4 + p.rows) / 4) as usize,
            1,
            32,
        );
        self.dispatch(
            e,
            1,
            &[&s.qk, &w.q, &w.freq, &s.query],
            p,
            HEADS,
            p.rows as usize,
            32,
        );
        let nb = (p.offset + p.rows) as usize / RATIO;
        self.dispatch(
            e,
            2,
            &[&s.query, &s.pool, &s.scores],
            p,
            nb.div_ceil(4),
            p.rows as usize,
            128,
        );
        self.encode_select(e, &s.scores, &s.ids, &s.cand, p, p.rows as usize, sel);
        self.dispatch(e, 3, &[&s.qk, &s.old_tail, &s.new_tail], p, 3, 1, 128);
        Ok(())
    }
}
impl DeviceWeights {
    fn new(d: &ProtocolObject<dyn MTLDevice>, w: &Weights) -> Result<Self> {
        w.norms.validate()?;
        let q = alloc(d, D * 2)?;
        upload(&q, &w.norms.q)?;
        let k = alloc(d, D * 2)?;
        upload(&k, &w.norms.k)?;
        let freq = alloc(d, ROPE / 2 * 4)?;
        upload(&freq, &frequencies())?;
        Ok(Self {
            projection: DenseWeight::new(d, &w.projection)?,
            q,
            k,
            freq,
        })
    }
}
impl Workspace {
    fn new(d: &ProtocolObject<dyn MTLDevice>, capacity: usize) -> Result<Self> {
        ensure!((16..=MAX_POSITION).contains(&capacity), "QSA capacity");
        Ok(Self {
            x: alloc(d, MAX_ROWS * H * 2)?,
            raw: alloc(d, (MAX_ROWS * QK + GUARD) * 4)?,
            qk: alloc(d, (MAX_ROWS * QK + GUARD) * 2)?,
            query: alloc(d, (MAX_ROWS * HEADS * D + GUARD) * 2)?,
            pool: alloc(d, (capacity / RATIO * D + GUARD) * 2)?,
            old_tail: alloc(d, 3 * D * 2)?,
            new_tail: alloc(d, (3 * D + GUARD) * 2)?,
            scores: alloc(d, (MAX_ROWS * (capacity / RATIO) + GUARD) * 4)?,
            ids: alloc(d, (MAX_ROWS * TOPK + GUARD) * 4)?,
            cand: alloc(d, (MAX_ROWS * SELECT2_MAX_CHUNKS * TOPK + GUARD) * 4)?,
            scratch: alloc(d, dense_scratch_bytes(H, QK, MAX_ROWS))?,
            capacity,
        })
    }
    fn prepare(&mut self, initial: &State) -> Result<()> {
        initial.validate()?;
        ensure!(initial.len <= self.capacity, "QSA state capacity");
        for b in [&self.raw, &self.scores, &self.ids] {
            poison32(b);
        }
        for b in [
            &self.qk,
            &self.query,
            &self.pool,
            &self.old_tail,
            &self.new_tail,
        ] {
            poison16(b);
        }
        upload(&self.pool, &initial.pooled)?;
        upload(&self.old_tail, &initial.tail)?;
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Stages {
    pub projected: Vec<u16>,
    pub queries: Vec<u16>,
    pub scores: Vec<f32>,
    pub selection: Selection,
    pub pooled: Vec<u16>,
    pub committed: State,
}
pub struct Run {
    pub stages: Stages,
    pub gpu_us: f64,
    pub encode_submit_wait_us: f64,
}
pub struct Engine {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: Ops,
    w: DeviceWeights,
    s: Workspace,
    _single_owner: PhantomData<Rc<()>>,
}
impl Engine {
    pub fn new(w: &Weights, capacity: usize) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("QSA device")?;
        let queue = device.newCommandQueue().context("QSA queue")?;
        Self::new_on(w, capacity, device, queue)
    }
    pub(crate) fn new_on(
        w: &Weights,
        capacity: usize,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        Ok(Self {
            ops: Ops::new(&device)?,
            w: DeviceWeights::new(&device, w)?,
            s: Workspace::new(&device, capacity)?,
            device,
            queue,
            _single_owner: PhantomData,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        let buffers = [
            &self.s.x,
            &self.s.raw,
            &self.s.qk,
            &self.s.query,
            &self.s.pool,
            &self.s.old_tail,
            &self.s.new_tail,
            &self.s.scores,
            &self.s.ids,
            &self.s.scratch,
        ];
        serde_json::json!({"device":self.device.name().to_string(),"weight_bytes":self.w.projection.bytes()+2*D*2+ROPE/2*4,
            "workspace_bytes":buffers.iter().map(|b|b.length()).sum::<usize>(),"capacity_tokens":self.s.capacity,
            "threadgroup_bytes":self.ops.pipelines.iter().map(|p|p.staticThreadgroupMemoryLength()).collect::<Vec<_>>(),
            "host_intermediate_handoffs":0,"safe_math":true,"dense_short_context_fast_path":false,
            "commit_tail_on_device":true,"state_import_export_for_diagnostics":true})
    }
    pub fn run(
        &mut self,
        input: &[u16],
        initial: &State,
        precision: Option<Precision>,
        sel: Selector,
        accepted: usize,
        repeats: usize,
    ) -> Result<Run> {
        ensure!((1..=8).contains(&repeats), "QSA repeat bound");
        let rows = self.prepare(input, initial, precision, accepted)?;
        let p = Params {
            offset: initial.len as u32,
            rows: rows as u32,
            base: initial.base as u32,
            accepted: accepted as u32,
        };
        let start = Instant::now();
        let cb = self.queue.commandBuffer().context("QSA command")?;
        let enc = cb.computeCommandEncoder().context("QSA encoder")?;
        for _ in 0..repeats {
            self.ops
                .encode(&enc, &self.w, &self.s, p, precision, sel, None)?;
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "QSA command failed {:?}",
            cb.error()
        );
        let gpu_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
        ensure!(gpu_us.is_finite() && gpu_us > 0., "QSA GPU timestamp");
        Ok(Run {
            stages: self.collect(initial, rows, precision, accepted)?,
            gpu_us,
            encode_submit_wait_us: elapsed,
        })
    }
    pub(crate) fn prepare(
        &mut self,
        input: &[u16],
        initial: &State,
        precision: Option<Precision>,
        accepted: usize,
    ) -> Result<usize> {
        let width = if precision.is_some() { H } else { QK };
        ensure!(
            !input.is_empty() && input.len() % width == 0 && input.len() / width <= MAX_ROWS,
            "QSA input shape"
        );
        let rows = input.len() / width;
        ensure!(
            input.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "QSA input finite"
        );
        if matches!(precision, Some(Precision::PackedF16x2)) {
            ensure!(
                input
                    .iter()
                    .all(|&v| f16::from_f32(bf16::from_bits(v).to_f32()).is_finite()),
                "QSA FP16 overflow"
            );
        }
        initial.validate()?;
        ensure!(
            initial.len + rows <= self.s.capacity
                && initial.base + initial.len + rows <= MAX_POSITION
                && accepted <= rows,
            "QSA append/accept geometry"
        );
        self.s.prepare(initial)?;
        upload(
            if precision.is_some() {
                &self.s.x
            } else {
                &self.s.qk
            },
            input,
        )?;
        Ok(rows)
    }
    /// R55 resident model, any accepted prefix: the next forward's old tail is the last
    /// (len + accepted) % 4 raw keys of [old tail (len % 4 rows); this forward's first
    /// `accepted` rows], as qsa_commit_tail builds it. Host side, after completion.
    pub(crate) fn commit_tail(&self, len: usize, accepted: usize) -> Result<()> {
        let (old, n) = (len % RATIO, (len + accepted) % RATIO);
        let mut tail = vec![0u16; n * D];
        unsafe {
            let ot = self.s.old_tail.contents().cast::<u16>().as_ptr();
            let qk = self.s.qk.contents().cast::<u16>().as_ptr();
            for i in 0..n {
                let row = old + accepted - n + i;
                let src = if row < old { ot.add(row * D) } else { qk.add((row - old) * QK + 512) };
                std::ptr::copy_nonoverlapping(src, tail.as_mut_ptr().add(i * D), D);
            }
            std::ptr::copy_nonoverlapping(tail.as_ptr(), ot, n * D);
        }
        Ok(())
    }
    /// R55 resident model: copy the committed raw tail (len_after % 4 rows) into the
    /// buffer the next forward reads. Host side, after the forward completed.
    pub(crate) fn roll_tail(&self, len_after: usize) -> Result<()> {
        let n = len_after % RATIO * D;
        ensure!(n * 2 <= self.s.old_tail.length() && n * 2 <= self.s.new_tail.length(), "QSA tail roll");
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.s.new_tail.contents().cast::<u16>().as_ptr(),
                self.s.old_tail.contents().cast::<u16>().as_ptr(),
                n,
            );
        }
        Ok(())
    }
    pub(crate) fn input_buffer(&self) -> &Buffer {
        &self.s.x
    }
    /// R58 prefill: (index projection, [q norm, k norm, RoPE frequencies], [pool, old tail, new
    /// tail], capacity).
    pub(crate) fn prefill_parts(&self) -> (&DenseWeight, [&Buffer; 3], [&Buffer; 3], usize) {
        (&self.w.projection, [&self.w.q, &self.w.k, &self.w.freq], [&self.s.pool, &self.s.old_tail, &self.s.new_tail], self.s.capacity)
    }
    pub(crate) fn selection_buffer(&self) -> &Buffer {
        &self.s.ids
    }
    /// R58 selector-guided seed: the last forward's per-row group scores (row stride = the
    /// forward's total group count).
    pub(crate) fn scores_buffer(&self) -> &Buffer {
        &self.s.scores
    }
    pub(crate) fn prepare_device(
        &mut self,
        initial: &State,
        rows: usize,
        accepted: usize,
    ) -> Result<()> {
        initial.validate()?;
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && accepted <= rows
                && initial.len + rows <= self.s.capacity
                && initial.len + rows <= MAX_POSITION - initial.base,
            "QSA device geometry"
        );
        self.s.prepare(initial)
    }
    pub(crate) fn encode_external(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        initial: &State,
        rows: usize,
        precision: Option<Precision>,
        sel: Selector,
        accepted: usize,
        input: Option<&Buffer>,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && accepted <= rows
                && initial.len + rows <= self.s.capacity,
            "QSA encoded shape"
        );
        self.ops.encode(
            e,
            &self.w,
            &self.s,
            Params {
                offset: initial.len as u32,
                rows: rows as u32,
                base: initial.base as u32,
                accepted: accepted as u32,
            },
            precision,
            sel,
            input,
        )
    }
    pub(crate) fn collect(
        &self,
        initial: &State,
        rows: usize,
        precision: Option<Precision>,
        accepted: usize,
    ) -> Result<Stages> {
        read32(&self.s.raw, if precision.is_some() { rows * QK } else { 0 })?;
        let projected = read16(&self.s.qk, rows * QK)?;
        let queries = read16(&self.s.query, rows * HEADS * D)?;
        let pooled = read16(&self.s.pool, (initial.len + rows) / RATIO * D)?;
        ensure!(
            pooled[..initial.pooled.len()] == initial.pooled,
            "QSA committed pool mutated"
        );
        let scores = read32(&self.s.scores, rows * ((initial.len + rows) / RATIO))?;
        ensure!(scores.iter().all(|v| *v >= 0.), "QSA negative score");
        let selection = Selection {
            offset: initial.len,
            rows,
            total: initial.len + rows,
            ids: read_bits(&self.s.ids, rows * TOPK)?,
            counts: (0..rows)
                .map(|r| ((initial.len + r + 1) / RATIO).min(TOPK))
                .collect(),
        };
        selection.validate()?;
        let committed = State {
            base: initial.base,
            len: initial.len + accepted,
            pooled: pooled[..(initial.len + accepted) / RATIO * D].to_vec(),
            tail: read16(&self.s.new_tail, (initial.len + accepted) % RATIO * D)?,
        };
        committed.validate()?;
        Ok(Stages {
            projected,
            queries,
            scores,
            selection,
            pooled,
            committed,
        })
    }
    /// Selector-only gate: exact input scores avoid attributing numerical
    /// projection differences to selection. No projection or state work runs.
    pub fn select_scores(
        &mut self,
        scores: &[f32],
        offset: usize,
        rows: usize,
        sel: Selector,
        repeats: usize,
    ) -> Result<(Selection, f64, f64)> {
        let nb = score_blocks(offset, rows)?;
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && offset + rows <= self.s.capacity
                && (1..=8).contains(&repeats),
            "selector bounds"
        );
        ensure!(
            scores.len() == rows * nb && scores.iter().all(|v| v.is_finite() && *v >= 0.),
            "selector score input"
        );
        poison32(&self.s.scores);
        poison32(&self.s.ids);
        upload(&self.s.scores, scores)?;
        let p = Params {
            offset: offset as u32,
            rows: rows as u32,
            base: 0,
            accepted: 0,
        };
        let start = Instant::now();
        let cb = self.queue.commandBuffer().context("selector command")?;
        let enc = cb.computeCommandEncoder().context("selector encoder")?;
        for _ in 0..repeats {
            self.ops.encode_select(&enc, &self.s.scores, &self.s.ids, &self.s.cand, p, rows, sel);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "selector command failed"
        );
        let s = Selection {
            offset,
            rows,
            total: offset + rows,
            ids: read_bits(&self.s.ids, rows * TOPK)?,
            counts: (0..rows)
                .map(|r| ((offset + r + 1) / RATIO).min(TOPK))
                .collect(),
        };
        s.validate()?;
        ensure!(
            read32(&self.s.scores, scores.len())?
                .iter()
                .zip(scores)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "selector input mutated"
        );
        Ok((s, (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6, elapsed))
    }
}
