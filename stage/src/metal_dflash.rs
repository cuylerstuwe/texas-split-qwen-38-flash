//! R58 step 2: the DFlash2 drafter on Metal, sharing the target's GPU and command queue.
//!
//! Context: for each committed row, the target features (the four-stream mean after target
//! layers 5/15/25/35/45, read from the target's own layer outputs, or given by the host) pass
//! through fc, then the hidden norm, then every layer's k/v projections. The k rows are normed
//! and rotated at their absolute positions; then everything goes into per-layer BF16 K/V rings
//! that hold the 2048-token window.
//!
//! Body: an 8-row block [anchor, mask x 7] at positions len.. passes through five layers (dynamic
//! conv, GQA over the ring and the block, dynamic conv, then dynamic conv / SwiGLU MLP / dynamic
//! conv), then the final norm. The caller runs the target's head on rows 1..7 and the selector.
//!
//! The linear layers are quantized at load to affine8, group 64 (Sushi does the same), and run on
//! the target's fragment kernels. Norms, conv, attention and activations are in
//! kernels/dflash2.metal. Synchronous: each call is one command buffer.
//!
//! Oracles: src/dflash_body.rs and friends (CPU, staged BF16), src/bin/dflash_hybrid.rs (the loop).
use crate::{
    bf16_linear::{MODEL, Matrix},
    dense::Affine8,
    dflash_context, dflash_layer,
    embedding::{EmbeddingTable, MASK},
    metal_model::MetalModel,
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, alloc, dense_scratch_bytes, upload},
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, ptr::NonNull, time::Instant};

pub const H: usize = 2560;
pub const QH: usize = 32;
pub const KH: usize = 8;
pub const D: usize = 128;
pub const MLP: usize = 7680;
pub const GROUP: usize = 16;
pub const DYN: usize = 4 * H / GROUP;
pub const FEATURES: usize = 5 * H;
pub const TAPS: [usize; 5] = [5, 15, 25, 35, 45];
pub const BLOCK: usize = 8;
pub const WINDOW: usize = 2048;
pub const RING: usize = 2048;
pub const MAX_ROWS: usize = 16;
/// K-splits of the fragment arm for an n-wide output: enough simdgroups (n/32 x splits ~ 640) to
/// fill the GPU at 8 rows; the arm allows 32 for n <= 1024, else 8.
fn auto_splits(n: usize) -> usize {
    let want = (640usize).div_ceil(n.div_ceil(32)).next_power_of_two();
    want.clamp(1, if n <= 1024 { 32 } else { 8 })
}
const P_RMS: usize = 0;
const P_CONV: usize = 1;
const P_NORMROPE: usize = 2;
const P_ATTN: usize = 3;
const P_SWIGLU: usize = 4;
const P_RING: usize = 5;
const P_FEATURES: usize = 6;
const P_COPY: usize = 7;
const P_ATTN_SPLIT: usize = 8;
/// Simdgroups per (head, row) in the split-key attention (QWEN_R58_DFLASH_ATTN_SPLIT; 1 = the serial kernel).
/// Default: ~128 context keys per simdgroup, at most 8 (serial below 256 keys).
fn attn_split(ctx: usize) -> usize {
    match std::env::var("QWEN_R58_DFLASH_ATTN_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(s) => s.clamp(1, 32),
        None => (ctx / 128).clamp(1, 8),
    }
}
type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    rows: u32,
    n: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// MLX-style affine8, group 64, of a BF16 [n][k] matrix. Per (row, group):
/// scale = bf16((max - min) / 255), bias = bf16(min), code = clamp(round((w - bias) / scale), 0, 255).
pub fn quantize(m: &Matrix) -> Result<Affine8> {
    let (k, n, g) = (m.k, m.n, 64usize);
    ensure!(m.values.len() == k * n && k % g == 0, "quantize geometry");
    let groups = k / g;
    let mut codes = vec![0u8; n * k];
    let (mut scales, mut biases) = (vec![0u16; n * groups], vec![0u16; n * groups]);
    let mut vals = [0f32; 64];
    for row in 0..n {
        for gi in 0..groups {
            let at = row * k + gi * g;
            for (j, v) in vals.iter_mut().enumerate() {
                *v = bf16::from_bits(m.values[at + j]).to_f32();
            }
            let (lo, hi) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            let mut sb = bf16::from_f32(if hi > lo { (hi - lo) / 255. } else { 1. });
            if sb.to_f32() == 0. {
                sb = bf16::from_f32(1.);
            }
            let bb = bf16::from_f32(lo);
            let (s, b) = (sb.to_f32(), bb.to_f32());
            for (j, &v) in vals.iter().enumerate() {
                codes[at + j] = ((v - b) / s).round().clamp(0., 255.) as u8;
            }
            scales[row * groups + gi] = sb.to_bits();
            biases[row * groups + gi] = bb.to_bits();
        }
    }
    Ok(Affine8 {
        k,
        n,
        group: g,
        codes,
        scales,
        biases,
        provenance: json!({"quantized_from":m.provenance,"bits":8,"group":g,"rule":"per row/group: bf16 scale (max-min)/255, bf16 bias min, round to nearest"}),
    })
}

struct LayerW {
    in_norm: Buffer,
    attn_base: Buffer,
    attn_dyn: DenseWeight,
    q: DenseWeight,
    k: DenseWeight,
    v: DenseWeight,
    o: DenseWeight,
    qn: Buffer,
    kn: Buffer,
    post_norm: Buffer,
    mlp_base: Buffer,
    mlp_dyn: DenseWeight,
    gate: DenseWeight,
    up: DenseWeight,
    down: DenseWeight,
    ck: Buffer,
    cv: Buffer,
}

pub struct MetalDflash {
    dense: DenseOps,
    pipes: Vec<Obj<dyn MTLComputePipelineState>>,
    fc: DenseWeight,
    hidden_norm: Buffer,
    final_norm: Buffer,
    freq: Buffer,
    layers: Vec<LayerW>,
    emb: EmbeddingTable,
    mask_row: Vec<u16>,
    feat: Buffer,
    x: Buffer,
    xn: Buffer,
    dynb: Buffer,
    prep: Buffer,
    qb: Buffer,
    kb: Buffer,
    vb: Buffer,
    qr: Buffer,
    kr: Buffer,
    att: Buffer,
    ob: Buffer,
    res: Buffer,
    gb: Buffer,
    ub: Buffer,
    act: Buffer,
    dn: Buffer,
    hidden: Buffer,
    y: Buffer,
    sums: Buffer,
    /// Prefill taps of up to RING rows (rows x FEATURES), filled during the target's engine prefill.
    staging: Buffer,
    len: usize,
    /// Lowest context position held in the ring (a context rebuilt from a later point has a gap below).
    start: usize,
    weight_bytes: usize,
    load_s: f64,
    /// Fixed K-splits for every projection, or None for auto_splits (QWEN_R58_DFLASH_SPLITS).
    splits: Option<usize>,
}

fn read_u16(b: &Buffer, n: usize) -> Vec<u16> {
    // SAFETY: shared-storage buffer of at least n u16; the command that wrote it has completed.
    unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), n) }.to_vec()
}

impl MetalDflash {
    /// Load the DFlash2 checkpoint (pinned by the CPU loaders) onto the target's GPU.
    pub fn load(target: &MetalModel) -> Result<Self> {
        let t0 = Instant::now();
        let (d, _) = target.metal();
        let dense = DenseOps::new(d)?;
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("../kernels/dflash2.metal")), Some(&options))
            .map_err(|e| anyhow::anyhow!("dflash2 MSL {e}"))?;
        let mut pipes = vec![];
        for name in ["df_rms", "df_conv", "df_normrope", "df_attn", "df_swiglu", "df_ring", "df_features", "df_copy", "df_attn_split"] {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).context("dflash2 function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("dflash2 pipeline {name}: {e}"))?;
            ensure!(p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= 256, "dflash2 pipeline geometry {name}");
            pipes.push(p);
        }
        let bytes = Cell::new(0usize);
        let q8 = |m: &Matrix| -> Result<DenseWeight> {
            let w = DenseWeight::fragment_only(d, &quantize(m)?)?;
            bytes.set(bytes.get() + w.fragment_bytes());
            Ok(w)
        };
        let vec_buf = |v: &[u16]| -> Result<Buffer> {
            let b = alloc(d, v.len() * 2)?;
            upload(&b, v)?;
            Ok(b)
        };
        let ctx = dflash_context::Weights::load(0)?;
        let fc = q8(&ctx.fc)?;
        let hidden_norm = vec_buf(&ctx.norm)?;
        drop(ctx);
        let mut layers = vec![];
        for l in 0..5 {
            let w = dflash_layer::Weights::load(l)?;
            layers.push(LayerW {
                in_norm: vec_buf(&w.norm)?,
                attn_base: vec_buf(&w.base)?,
                attn_dyn: q8(&w.dynamic)?,
                q: q8(&w.q)?,
                k: q8(&w.k)?,
                v: q8(&w.v)?,
                o: q8(&w.o)?,
                qn: vec_buf(&w.qk.q)?,
                kn: vec_buf(&w.qk.k)?,
                post_norm: vec_buf(&w.mlp.norm)?,
                mlp_base: vec_buf(&w.mlp.base)?,
                mlp_dyn: q8(&w.mlp.dynamic)?,
                gate: q8(&w.mlp.gate)?,
                up: q8(&w.mlp.up)?,
                down: q8(&w.mlp.down)?,
                ck: alloc(d, RING * KH * D * 2)?,
                cv: alloc(d, RING * KH * D * 2)?,
            });
        }
        let s = Shard::open(Path::new(MODEL))?;
        let t = s.tensors.get("norm.weight").context("DFlash2 final norm absent")?;
        ensure!(t.dtype == "BF16" && t.shape == [H as u64], "DFlash2 final norm geometry");
        let raw = s.read_range("norm.weight", 0, t.bytes())?;
        let fnorm: Vec<u16> = raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
        let final_norm = vec_buf(&fnorm)?;
        let freq = alloc(d, 64 * 4)?;
        upload(&freq, &crate::dflash_attention::frequencies())?;
        let emb = EmbeddingTable::open(target.pack())?;
        let mask_row = emb.rows(&[MASK as u32])?;
        let r = MAX_ROWS;
        let mut all: Vec<&DenseWeight> = vec![&fc];
        for l in &layers {
            all.extend([&l.attn_dyn, &l.q, &l.k, &l.v, &l.o, &l.mlp_dyn, &l.gate, &l.up, &l.down]);
        }
        let sums_bytes = all
            .iter()
            .map(|w| dense_scratch_bytes(w.k, w.n, r).max(r * (w.k / w.group) * 4).max(r * w.n * 4))
            .max()
            .unwrap();
        let a = |n: usize| alloc(d, n);
        Ok(Self {
            dense,
            pipes,
            fc,
            hidden_norm,
            final_norm,
            freq,
            emb,
            mask_row,
            feat: a(r * FEATURES * 2)?,
            x: a(r * H * 2)?,
            xn: a(r * H * 2)?,
            dynb: a(r * DYN * 2)?,
            prep: a(r * H * 2)?,
            qb: a(r * QH * D * 2)?,
            kb: a(r * KH * D * 2)?,
            vb: a(r * KH * D * 2)?,
            qr: a(r * QH * D * 2)?,
            kr: a(r * KH * D * 2)?,
            att: a(r * QH * D * 2)?,
            ob: a(r * H * 2)?,
            res: a(r * H * 2)?,
            gb: a(r * MLP * 2)?,
            ub: a(r * MLP * 2)?,
            act: a(r * MLP * 2)?,
            dn: a(r * H * 2)?,
            hidden: a(r * H * 2)?,
            y: a(r * MLP * 4)?,
            sums: a(sums_bytes)?,
            staging: a(RING * FEATURES * 2)?,
            layers,
            len: 0,
            start: 0,
            weight_bytes: bytes.get(),
            load_s: t0.elapsed().as_secs_f64(),
            splits: match std::env::var("QWEN_R58_DFLASH_SPLITS").as_deref() {
                Ok("auto") | Err(_) => None,
                Ok(v) => Some(v.parse()?),
            },
        })
    }
    pub fn description(&self) -> Value {
        json!({"kind":"metal_dflash_loaded","weights":"affine8 group 64, fragment layout only","weight_bytes":self.weight_bytes,
            "ring_tokens":RING,"window":WINDOW,"block":BLOCK,"taps":TAPS,"load_s":self.load_s,
            "dense_arm":"FragmentPacked f16x2","splits":self.splits.map_or(json!("auto"), |s| json!(s))})
    }
    /// Context positions appended so far (the anchor's position for the next block).
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Forget the context (the rings are only ever read between start and len).
    pub fn reset(&mut self) {
        self.len = 0;
        self.start = 0;
    }
    fn run(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, pipe: usize, bufs: &[&Buffer], p: Params, grid: (usize, usize), threads: usize) {
        e.setComputePipelineState(&self.pipes[pipe]);
        unsafe {
            for (i, b) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(*b), 0, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&p).cast(), std::mem::size_of::<Params>(), bufs.len());
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: 1 },
            MTLSize { width: threads, height: 1, depth: 1 },
        );
    }
    /// Fix the projections' K-splits (Some) or use auto_splits (None).
    pub fn set_splits(&mut self, splits: Option<usize>) {
        self.splits = splits;
    }
    fn arm(&self, n: usize) -> Arm {
        let s = self.splits.unwrap_or_else(|| auto_splits(n));
        Arm::FragmentPacked { splits: s.clamp(1, if n <= 1024 { 32 } else { 8 }), residual: true }
    }
    /// `run` with byte offsets per buffer.
    fn run_at(&self, e: &Enc, pipe: usize, bufs: &[(&Buffer, usize)], p: Params, grid: (usize, usize), threads: usize) {
        e.setComputePipelineState(&self.pipes[pipe]);
        unsafe {
            for (i, (b, off)) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(*b), *off, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&p).cast(), std::mem::size_of::<Params>(), bufs.len());
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: 1 },
            MTLSize { width: threads, height: 1, depth: 1 },
        );
    }
    fn dense(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, w: &DenseWeight, x: &Buffer, out: &Buffer, rows: usize) -> Result<()> {
        self.dense.encode(e, w, x, &self.y, out, &self.sums, rows, 0, self.arm(w.n))
    }
    /// Commit, wait and check; returns the command buffer's GPU seconds.
    fn finish(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<f64> {
        cb.commit();
        cb.waitUntilCompleted();
        ensure!(cb.status() == MTLCommandBufferStatus::Completed, "dflash2 command: {:?}", cb.error());
        Ok(cb.GPUEndTime() - cb.GPUStartTime())
    }
    /// Context rows already in `feat` -> fc -> hidden norm -> each layer's k (normed, rotated at
    /// positions len..) and v -> the rings.
    fn encode_context(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, rows: usize, pos: usize) -> Result<()> {
        let (r, pos) = (rows as u32, pos as u32);
        self.dense(e, &self.fc, &self.feat, &self.res, rows)?;
        self.run(e, P_RMS, &[&self.res, &self.hidden_norm, &self.xn], Params { rows: r, n: H as u32, ..Default::default() }, (rows, 1), 256);
        for l in &self.layers {
            self.dense(e, &l.k, &self.xn, &self.kb, rows)?;
            self.dense(e, &l.v, &self.xn, &self.vb, rows)?;
            self.run(e, P_NORMROPE, &[&self.kb, &l.kn, &self.kr, &self.freq], Params { rows: r, n: KH as u32, a: pos, ..Default::default() }, (KH, rows), 32);
            self.run(e, P_RING, &[&self.kr, &self.vb, &l.ck, &l.cv], Params { rows: r, a: pos, c: RING as u32, ..Default::default() }, (1024 / 256, rows), 256);
        }
        Ok(())
    }
    /// Append the first `rows` rows of the target's last (completed) forward as context positions
    /// len.., reading the target's tap-layer outputs on the GPU. Returns the GPU seconds.
    pub fn append_target(&mut self, target: &MetalModel, rows: usize) -> Result<f64> {
        ensure!((1..=MAX_ROWS).contains(&rows), "DFlash2 append rows");
        let (_, queue) = target.metal();
        let cb = queue.commandBuffer().context("dflash2 append command")?;
        let e = cb.computeCommandEncoder().context("dflash2 append encoder")?;
        for (t, &layer) in TAPS.iter().enumerate() {
            let s = target.layer_output(layer)?;
            ensure!(s.length() >= rows * 4 * H * 2, "tap layer output size");
            self.run(&e, P_FEATURES, &[s, &self.feat], Params { rows: rows as u32, a: (t * H) as u32, ..Default::default() }, (H.div_ceil(256), rows), 256);
        }
        self.encode_context(&e, rows, self.len)?;
        e.endEncoding();
        let g = Self::finish(&cb)?;
        self.len += rows;
        Ok(g)
    }
    /// Append host features (rows x 12800 BF16, the same reduction as df_features) as context
    /// positions len... Returns the GPU seconds.
    pub fn append_features(&mut self, target: &MetalModel, feats: &[u16]) -> Result<f64> {
        ensure!(!feats.is_empty() && feats.len() % FEATURES == 0, "DFlash2 feature rows");
        let (_, queue) = target.metal();
        let mut g = 0.;
        for part in feats.chunks(MAX_ROWS * FEATURES) {
            let rows = part.len() / FEATURES;
            upload(&self.feat, part)?;
            let cb = queue.commandBuffer().context("dflash2 append command")?;
            let e = cb.computeCommandEncoder().context("dflash2 append encoder")?;
            self.encode_context(&e, rows, self.len)?;
            e.endEncoding();
            g += Self::finish(&cb)?;
            self.len += rows;
        }
        Ok(g)
    }
    /// The target's engine prefill of `tokens` (from its current position), capturing the taps of the
    /// last min(n, RING) rows on the GPU, then those rows' context (fc, norms, k/v, rings) at their
    /// absolute positions. If this context does not continue at the captured rows, it restarts there.
    /// Returns the target's prefill timing and the context build's GPU seconds.
    pub fn prefill_target(&mut self, target: &mut MetalModel, tokens: &[u32], chunk: usize) -> Result<(crate::metal_model::PrefillTiming, f64)> {
        let n = tokens.len();
        let w = n.min(RING);
        let first = n - w;
        let abs_first = target.position() + first;
        let (pipe, staging) = (&self.pipes[P_FEATURES], &self.staging);
        let mut hook = |e: &Enc, layer: usize, out: &Buffer, t: usize, s: usize| -> Result<()> {
            let Some(ti) = TAPS.iter().position(|&x| x == layer) else { return Ok(()) };
            let (lo, hi) = (s.max(first), s + t);
            if hi <= lo {
                return Ok(());
            }
            let rows = hi - lo;
            e.setComputePipelineState(pipe);
            let p = Params { rows: rows as u32, a: (ti * H) as u32, ..Default::default() };
            unsafe {
                e.setBuffer_offset_atIndex(Some(out), (lo - s) * 4 * H * 2, 0);
                e.setBuffer_offset_atIndex(Some(staging), (lo - first) * FEATURES * 2, 1);
                e.setBytes_length_atIndex(NonNull::from(&p).cast(), std::mem::size_of::<Params>(), 2);
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: H.div_ceil(256), height: rows, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
            Ok(())
        };
        let timing = target.prefill_with(tokens, chunk, &mut hook)?;
        if self.len != abs_first {
            self.len = abs_first;
            self.start = abs_first;
        }
        let (_, queue) = target.metal();
        let cb = queue.commandBuffer().context("dflash2 context command")?;
        let e = cb.computeCommandEncoder().context("dflash2 context encoder")?;
        let mut at = 0;
        while at < w {
            let rows = MAX_ROWS.min(w - at);
            let words = rows * FEATURES / 2;
            self.run_at(&e, P_COPY, &[(&self.staging, at * FEATURES * 2), (&self.feat, 0)], Params { n: words as u32, ..Default::default() }, (words.div_ceil(256), 1), 256);
            self.encode_context(&e, rows, self.len + at)?;
            at += rows;
        }
        e.endEncoding();
        let g = Self::finish(&cb)?;
        self.len += w;
        Ok((timing, g))
    }
    /// Stage the block [anchor, mask x 7] in `x`.
    fn stage_block(&self, anchor: u32) -> Result<()> {
        let mut x = self.emb.rows(&[anchor])?;
        for _ in 1..BLOCK {
            x.extend_from_slice(&self.mask_row);
        }
        upload(&self.x, &x)
    }
    /// The body's dispatches, in order, through `sink(stage label, encode)`: a single encoder for
    /// a forward, or one timestamped encoder per stage when profiling.
    fn encode_body(&self, sink: &mut dyn FnMut(&'static str, &dyn Fn(&Enc) -> Result<()>) -> Result<()>) -> Result<()> {
        let rows = BLOCK;
        let (r, pos) = (rows as u32, self.len as u32);
        let ctx_lo = self.len.saturating_sub(RING).max(self.start) as u32;
        let p = |n: usize, a: u32, b: u32, c: u32| Params { rows: r, n: n as u32, a, b, c, d: 0 };
        let conv_grid = (H.div_ceil(256), rows);
        let g16 = GROUP as u32;
        for l in &self.layers {
            // attention half: norm, dynamic kernel, conv (set 0), q/k/v, norm+rope, GQA, o, conv (set 1) + residual
            sink("attn_norm", &|e| Ok(self.run(e, P_RMS, &[&self.x, &l.in_norm, &self.xn], p(H, 0, 0, 0), (rows, 1), 256)))?;
            sink("attn_dyn", &|e| self.dense(e, &l.attn_dyn, &self.xn, &self.dynb, rows))?;
            sink("attn_conv0", &|e| Ok(self.run(e, P_CONV, &[&self.xn, &self.dynb, &l.attn_base, &self.xn, &self.prep], p(H, g16, 0, 0), conv_grid, 256)))?;
            sink("q", &|e| self.dense(e, &l.q, &self.prep, &self.qb, rows))?;
            sink("k", &|e| self.dense(e, &l.k, &self.prep, &self.kb, rows))?;
            sink("v", &|e| self.dense(e, &l.v, &self.prep, &self.vb, rows))?;
            sink("normrope_q", &|e| Ok(self.run(e, P_NORMROPE, &[&self.qb, &l.qn, &self.qr, &self.freq], p(QH, pos, 0, 0), (QH, rows), 32)))?;
            sink("normrope_k", &|e| Ok(self.run(e, P_NORMROPE, &[&self.kb, &l.kn, &self.kr, &self.freq], p(KH, pos, 0, 0), (KH, rows), 32)))?;
            sink("attention", &|e| {
                let split = attn_split((pos - ctx_lo) as usize);
                if split > 1 {
                    self.run(e, P_ATTN_SPLIT, &[&self.qr, &self.kr, &self.vb, &l.ck, &l.cv, &self.att], p(ctx_lo as usize, pos, WINDOW as u32, RING as u32), (QH, rows), 32 * split);
                } else {
                    self.run(e, P_ATTN, &[&self.qr, &self.kr, &self.vb, &l.ck, &l.cv, &self.att], p(ctx_lo as usize, pos, WINDOW as u32, RING as u32), (QH, rows), 32);
                }
                Ok(())
            })?;
            sink("o", &|e| self.dense(e, &l.o, &self.att, &self.ob, rows))?;
            sink("attn_conv1", &|e| Ok(self.run(e, P_CONV, &[&self.ob, &self.dynb, &l.attn_base, &self.x, &self.res], p(H, g16, 1, 1), conv_grid, 256)))?;
            // MLP half: norm, dynamic kernel, conv (set 0), gate/up, SwiGLU, down, conv (set 1) + residual
            sink("mlp_norm", &|e| Ok(self.run(e, P_RMS, &[&self.res, &l.post_norm, &self.xn], p(H, 0, 0, 0), (rows, 1), 256)))?;
            sink("mlp_dyn", &|e| self.dense(e, &l.mlp_dyn, &self.xn, &self.dynb, rows))?;
            sink("mlp_conv0", &|e| Ok(self.run(e, P_CONV, &[&self.xn, &self.dynb, &l.mlp_base, &self.xn, &self.prep], p(H, g16, 0, 0), conv_grid, 256)))?;
            sink("gate", &|e| self.dense(e, &l.gate, &self.prep, &self.gb, rows))?;
            sink("up", &|e| self.dense(e, &l.up, &self.prep, &self.ub, rows))?;
            sink("swiglu", &|e| Ok(self.run(e, P_SWIGLU, &[&self.gb, &self.ub, &self.act], p(rows * MLP, 0, 0, 0), ((rows * MLP).div_ceil(256), 1), 256)))?;
            sink("down", &|e| self.dense(e, &l.down, &self.act, &self.dn, rows))?;
            sink("mlp_conv1", &|e| Ok(self.run(e, P_CONV, &[&self.dn, &self.dynb, &l.mlp_base, &self.res, &self.x], p(H, g16, 1, 1), conv_grid, 256)))?;
        }
        sink("final_norm", &|e| Ok(self.run(e, P_RMS, &[&self.x, &self.final_norm, &self.hidden], p(H, 0, 0, 0), (rows, 1), 256)))
    }
    /// The final-norm hidden rows (BF16, BLOCK x H) of the block [anchor, mask x 7] at positions
    /// len..len+8, over the context ring. Returns them with the GPU seconds.
    pub fn draft_hidden(&mut self, target: &MetalModel, anchor: u32) -> Result<(Vec<u16>, f64)> {
        self.stage_block(anchor)?;
        let (_, queue) = target.metal();
        let cb = queue.commandBuffer().context("dflash2 body command")?;
        let e = cb.computeCommandEncoder().context("dflash2 body encoder")?;
        self.encode_body(&mut |_, f| f(&e))?;
        e.endEncoding();
        let g = Self::finish(&cb)?;
        Ok((read_u16(&self.hidden, BLOCK * H), g))
    }
    /// Profile: the same block with one timestamped encoder per stage. Returns each stage's
    /// (label, GPU microseconds) in order (five layers, then the final norm).
    pub fn draft_hidden_profiled(&mut self, target: &MetalModel, anchor: u32) -> Result<Vec<(String, f64)>> {
        self.stage_block(anchor)?;
        let mut clock = crate::metal_profile::StageClock::with_capacity(target.queue_obj(), 128)?;
        self.encode_body(&mut |label, f| clock.stage(label, |e| f(e)))?;
        Ok(clock.finish()?.into_iter().map(|s| (s.label, s.gpu_us)).collect())
    }
}
