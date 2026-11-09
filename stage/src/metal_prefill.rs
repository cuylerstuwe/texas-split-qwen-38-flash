//! R58 prefill engine: a model layer for a prefill chunk of T rows. Stages and kernels are the
//! decode layer's wherever those are row-generic (HC norm / activate / mix / write, GDN prepare /
//! norm-gate, router / select, expert prepare / mid / finish, shared-expert gate / activate /
//! join). The large-row replacements are MLX's NAX affine GEMM for every dense projection
//! (metal_qmm_nax), MLX's chunked gated delta for the recurrence (metal_gdn_nax: final state only,
//! updated in place in the committed state), and expert-blocked MPP projections over
//! expert-sorted slots (metal_moe_prefill's kernel). A layer is two halves: the front (HC read,
//! mixer, fused HC write + MLP-side read, router, top-10 select) and the MLP tail (routed and
//! shared experts, join, final HC write); the routing plan is built on the host between them, so
//! a chunk runs as one command buffer per layer boundary (tail of one layer + front of the next).
//! Decode-path state contracts are kept: after a chunk a GDN layer's committed state and
//! convolution history are what the decode path's commit of the same rows leaves.
use crate::{
    gdn::{C, DK, HK, HV, V},
    hc::{F, H, HC, R},
    metal_attention_layer::Engine as AttnLayer,
    metal_gdn::MetalGdn,
    metal_gdn_nax::GdnNax,
    metal_hc::Hc,
    metal_mlp,
    metal_moe_prefill::plan,
    metal_ops::{Buffer, Obj, alloc, upload},
    metal_ple_target::PleTarget,
    metal_qmm_nax::QmmNax,
    metal_target_layer::Engine as GdnLayer,
    moe::{I, TOPK},
    ple::HISTORY,
    qsa,
    target_attention::{KH, KV, META, Q, QG, QH},
    target_mlp::RoutingArithmetic,
};
use anyhow::{Context, Result, ensure};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type Pipeline = Obj<dyn MTLComputePipelineState>;
const GUARD: usize = 1024;
const EXPERTS: usize = 512;
#[repr(C)]
#[derive(Clone, Copy)]
struct QsaParams {
    offset: u32,
    rows: u32,
    base: u32,
    accepted: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    offset: u32,
    rows: u32,
    base: u32,
    capacity: u32,
    sparse: u32,
    splits: u32,
    log2_scores: u32,
}

fn library(d: &ProtocolObject<dyn MTLDevice>, src: &str, what: &str) -> Result<Obj<dyn MTLLibrary>> {
    let opt = MTLCompileOptions::new();
    opt.setMathMode(crate::metal_ops::math_mode());
    d.newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opt)).map_err(|e| anyhow::anyhow!("{what} MSL: {e}"))
}
fn pipe(d: &ProtocolObject<dyn MTLDevice>, lib: &ProtocolObject<dyn MTLLibrary>, name: &str) -> Result<Pipeline> {
    let f = lib.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("prefill function {name}"))?;
    d.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow::anyhow!("prefill pipeline {name}: {e}"))
}
fn dispatch(e: &Encoder, x: usize, y: usize, threads: usize) {
    e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: x, height: y, depth: 1 }, MTLSize { width: threads, height: 1, depth: 1 });
}
fn bufs(e: &Encoder, b: &[&Buffer]) {
    unsafe {
        for (i, b) in b.iter().enumerate() {
            e.setBuffer_offset_atIndex(Some(b), 0, i);
        }
    }
}
fn at(e: &Encoder, b: &Buffer, offset: usize, index: usize) {
    unsafe { e.setBuffer_offset_atIndex(Some(b), offset, index) }
}
fn bytes<T: Copy>(e: &Encoder, v: &T, index: usize) {
    unsafe { e.setBytes_length_atIndex(NonNull::from(v).cast(), std::mem::size_of::<T>(), index) }
}
#[track_caller]
fn barrier(e: &Encoder) {
    crate::metal_ops::barrier(e);
}
pub(crate) fn router_mma() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R58_ROUTER_MMA").as_deref() != Ok("0"))
}
fn read16(b: &Buffer, n: usize) -> Vec<u16> {
    assert!(b.length() >= n * 2);
    // SAFETY: shared-storage buffer of at least n u16; the caller waited for the GPU work.
    unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), n).to_vec() }
}

/// R58: where a prefill stage's dispatches go: the command buffer's single encoder (One), or a
/// timestamped encoder per stage (Profiled, on a stage clock).
pub(crate) trait Sink {
    fn stage(&mut self, label: &'static str, f: &mut dyn FnMut(&Encoder) -> Result<()>) -> Result<()>;
}
pub(crate) struct One<'a>(pub &'a Encoder);
impl Sink for One<'_> {
    fn stage(&mut self, _: &'static str, f: &mut dyn FnMut(&Encoder) -> Result<()>) -> Result<()> {
        f(self.0)
    }
}
pub(crate) struct Profiled<'a> {
    pub clock: &'a mut crate::metal_profile::StageClock,
    pub prefix: &'static str,
}
impl Sink for Profiled<'_> {
    fn stage(&mut self, label: &'static str, f: &mut dyn FnMut(&Encoder) -> Result<()>) -> Result<()> {
        self.clock.stage_with(format!("{}.{}", self.prefix, label), true, |e| f(e))
    }
}

struct Pipes {
    hc_norm: Pipeline,
    hc_activate: Pipeline,
    hc_mix: Pipeline,
    hc_write: Pipeline,
    gdn_prepare: Pipeline,
    gdn_norm: Pipeline,
    split_gates: Pipeline,
    copy: Pipeline,
    router_mma: Pipeline,
    router: Pipeline,
    select: Pipeline,
    prepare: Pipeline,
    mid: Pipeline,
    finish: Pipeline,
    shared_gate: Pipeline,
    shared_act: Pipeline,
    join: Pipeline,
    mpp: Pipeline,
    attn_prepare: Pipeline,
    attn_pack: Pipeline,
    attn_nax: Pipeline,
    attn_gate: Pipeline,
    qsa_pool: Pipeline,
    qsa_query: Pipeline,
    qsa_score: Pipeline,
    qsa_commit_tail: Pipeline,
    qsa_select: Pipeline,
    ple_norm: Pipeline,
    ple_gate: Pipeline,
    ple_conv: Pipeline,
}

/// Scratch for one layer of a prefill chunk of up to `max` rows (reused layer after layer).
pub(crate) struct Prefill {
    pub max: usize,
    qmm: QmmNax,
    gdn: GdnNax,
    p: Pipes,
    // HC
    normalized: Buffer,
    parts: Buffer,
    dp: Buffer,
    act: Buffer,
    upv: Buffer,
    mixed0: Buffer,
    inject0: Buffer,
    mixed1: Buffer,
    inject1: Buffer,
    stream: Buffer,
    // GDN
    pqkv: Buffer,
    pz: Buffer,
    pa: Buffer,
    pb: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    gates: Buffer,
    g: Buffer,
    beta: Buffer,
    y: Buffer,
    gated: Buffer,
    gdn_out: Buffer,
    conv_out: Buffer,
    dummy: Buffer,
    // MLP
    logits_f: Buffer,
    logits_bf: Buffer,
    ids: Buffer,
    scores: Buffer,
    sorted: Buffer,
    order: Buffer,
    inverse: Buffer,
    blocks: Buffer,
    pg: Buffer,
    pu: Buffer,
    rg: Buffer,
    ru: Buffer,
    middle: Buffer,
    rd: Buffer,
    routed_f: Buffer,
    routed_bf: Buffer,
    gate_f: Buffer,
    gate_bf: Buffer,
    sg: Buffer,
    su: Buffer,
    sa: Buffer,
    sd: Buffer,
    sigmoid: Buffer,
    shared: Buffer,
    mlp_out: Buffer,
    // attention
    qg: Buffer,
    kraw: Buffer,
    vraw: Buffer,
    aquery: Buffer,
    akey: Buffer,
    agate: Buffer,
    attention: Buffer,
    agated: Buffer,
    attn_out: Buffer,
    // QSA indexer
    iqk: Buffer,
    iquery: Buffer,
    iscores: Buffer,
    iids: Buffer,
    // PLE
    pemb: Buffer,
    pkey: Buffer,
    pvalue: Buffer,
    pkn: Buffer,
    pqn: Buffer,
    pgated: Buffer,
    pgates: Buffer,
    pnorm: Buffer,
    /// The PLE stage's output rows (layer 1's input).
    pub ple_out: Buffer,
    /// Context capacity the indexer scores are sized for.
    capacity: usize,
    /// Layer input/output rows (BF16 [rows][F]); layers alternate between the two.
    pub io: [Buffer; 2],
}

impl Prefill {
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>, max: usize, capacity: usize) -> Result<Self> {
        ensure!((16..=4096).contains(&max) && max % 16 == 0, "prefill chunk rows");
        ensure!(capacity >= max && capacity <= qsa::MAX_POSITION, "prefill context capacity");
        let attn = library(d, &[include_str!("../kernels/target_attention.metal"), include_str!("../kernels/target_attention_nax.metal")].concat(), "attention")?;
        let index = library(d, &[include_str!("../kernels/qsa_index.metal"), include_str!("../kernels/qsa_select.metal")].concat(), "QSA")?;
        let ple = library(d, include_str!("../kernels/ple_target.metal"), "PLE")?;
        let hc = library(d, include_str!("../kernels/hc.metal"), "HC")?;
        let gdn = library(d, include_str!("../kernels/gdn_block.metal"), "GDN")?;
        let mlp_src = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../kernels/exl3_project.metal"),
            include_str!("../kernels/exl3_moe.metal"),
            include_str!("../kernels/target_mlp.metal"),
            include_str!("../kernels/exl3_grouped.metal")
        );
        let mlp = library(d, &mlp_src, "MLP")?;
        let mpp = library(d, include_str!("../kernels/exl3_mpp.metal"), "MPP")?;
        let helper = library(d, include_str!("../kernels/r58_prefill.metal"), "prefill helper")?;
        let p = Pipes {
            hc_norm: pipe(d, &hc, "hc_norm")?,
            hc_activate: pipe(d, &hc, "hc_activate")?,
            hc_mix: pipe(d, &hc, "hc_mix")?,
            hc_write: pipe(d, &hc, "hc_write")?,
            gdn_prepare: pipe(d, &gdn, "gdn_prepare")?,
            gdn_norm: pipe(d, &gdn, "gdn_norm_gate")?,
            split_gates: pipe(d, &helper, "r58_split_gates")?,
            copy: pipe(d, &helper, "r58_copy_words")?,
            router_mma: pipe(d, &helper, "r58_router_mma")?,
            router: pipe(d, &mlp, "tm_router_qmv")?,
            select: pipe(d, &mlp, "tm_router_select")?,
            prepare: pipe(d, &mlp, "tm_prepare_pair")?,
            mid: pipe(d, &mlp, "moe_mid")?,
            finish: pipe(d, &mlp, "moe_finish")?,
            shared_gate: pipe(d, &mlp, "tm_shared_gate")?,
            shared_act: pipe(d, &mlp, "tm_shared_activate")?,
            join: pipe(d, &mlp, "tm_shared_join")?,
            mpp: pipe(d, &mpp, "exl3_mpp")?,
            attn_prepare: pipe(d, &attn, "attn_prepare")?,
            attn_pack: pipe(d, &attn, "attn_pack")?,
            attn_nax: pipe(d, &attn, "attn_nax")?,
            attn_gate: pipe(d, &attn, "attn_gate")?,
            qsa_pool: pipe(d, &index, "qsa_pool")?,
            qsa_query: pipe(d, &index, "qsa_query")?,
            qsa_score: pipe(d, &index, "qsa_score")?,
            qsa_commit_tail: pipe(d, &index, "qsa_commit_tail")?,
            qsa_select: pipe(d, &index, "qsa_select11")?,
            ple_norm: pipe(d, &ple, "ple_norm")?,
            ple_gate: pipe(d, &ple, "ple_gate")?,
            ple_conv: pipe(d, &ple, "ple_conv_out")?,
        };
        let a = |n: usize| alloc(d, n + GUARD);
        let (t, slots) = (max, max * TOPK);
        Ok(Self {
            max,
            qmm: QmmNax::new(d)?,
            gdn: GdnNax::new(d)?,
            p,
            normalized: a(t * F * 2)?,
            parts: a(t * HC * HC * 4)?,
            dp: a(t * R * 2)?,
            act: a(t * R * 2)?,
            upv: a(t * F * 2)?,
            mixed0: a(t * H * 2)?,
            inject0: a(t * HC * 2)?,
            mixed1: a(t * H * 2)?,
            inject1: a(t * HC * 2)?,
            stream: a(t * F * 2)?,
            pqkv: a(t * C * 2)?,
            pz: a(t * V * 2)?,
            pa: a(t * HV * 2)?,
            pb: a(t * HV * 2)?,
            q: a(t * HK * DK * 2)?,
            k: a(t * HK * DK * 2)?,
            v: a(t * V * 2)?,
            gates: a(t * HV * 2 * 4)?,
            g: a(t * HV * 4)?,
            beta: a(t * HV * 4)?,
            y: a(t * V * 2)?,
            gated: a(t * V * 2)?,
            gdn_out: a(t * H * 2)?,
            conv_out: a(3 * C * 2)?,
            dummy: a(64)?,
            logits_f: a(t * EXPERTS * 4)?,
            logits_bf: a(t * EXPERTS * 2)?,
            ids: a(slots * 4)?,
            scores: a(slots * 4)?,
            sorted: a(slots * 4)?,
            order: a(slots * 4)?,
            inverse: a(slots * 4)?,
            blocks: a(slots * 16)?,
            pg: a(slots * H * 2)?,
            pu: a(slots * H * 2)?,
            rg: a(slots * I * 4)?,
            ru: a(slots * I * 4)?,
            middle: a(slots * I * 2)?,
            rd: a(slots * H * 4)?,
            routed_f: a(t * H * 4)?,
            routed_bf: a(t * H * 2)?,
            gate_f: a(t * 4)?,
            gate_bf: a(t * 2)?,
            sg: a(t * I * 2)?,
            su: a(t * I * 2)?,
            sa: a(t * I * 2)?,
            sd: a(t * H * 2)?,
            sigmoid: a(t * 2)?,
            shared: a(t * H * 2)?,
            mlp_out: a(t * H * 2)?,
            qg: a(t * QG * 2)?,
            kraw: a(t * KV * 2)?,
            vraw: a(t * KV * 2)?,
            aquery: a(t * Q * 2)?,
            akey: a(t * KV * 2)?,
            agate: a(t * Q * 2)?,
            attention: a(t * Q * 2)?,
            agated: a(t * Q * 2)?,
            attn_out: a(t * H * 2)?,
            iqk: a(t * qsa::QK * 2)?,
            iquery: a(t * qsa::HEADS * qsa::D * 2)?,
            iscores: a(t * (capacity / qsa::RATIO) * 4)?,
            iids: a(t * qsa::TOPK * 4)?,
            pemb: a(t * H * 2)?,
            pkey: a(t * F * 2)?,
            pvalue: a(t * H * 2)?,
            pkn: a(t * F * 2)?,
            pqn: a(t * F * 2)?,
            pgated: a(t * F * 2)?,
            pgates: a(t * HC * 2)?,
            pnorm: a(t * F * 2)?,
            ple_out: a(t * F * 2)?,
            capacity,
            io: [a(t * F * 2)?, a(t * F * 2)?],
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        [&self.normalized, &self.parts, &self.dp, &self.act, &self.upv, &self.mixed0, &self.inject0, &self.mixed1, &self.inject1, &self.stream,
         &self.pqkv, &self.pz, &self.pa, &self.pb, &self.q, &self.k, &self.v, &self.gates, &self.g, &self.beta, &self.y, &self.gated,
         &self.gdn_out, &self.conv_out, &self.dummy, &self.logits_f, &self.logits_bf, &self.ids, &self.scores, &self.sorted, &self.order,
         &self.inverse, &self.blocks, &self.pg, &self.pu, &self.rg, &self.ru, &self.middle, &self.rd, &self.routed_f, &self.routed_bf,
         &self.gate_f, &self.gate_bf, &self.sg, &self.su, &self.sa, &self.sd, &self.sigmoid, &self.shared, &self.mlp_out, &self.io[0], &self.io[1],
         &self.qg, &self.kraw, &self.vraw, &self.aquery, &self.akey, &self.agate, &self.attention, &self.agated, &self.attn_out, &self.iqk,
         &self.iquery, &self.iscores, &self.iids, &self.pemb, &self.pkey, &self.pvalue, &self.pkn, &self.pqn, &self.pgated, &self.pgates,
         &self.pnorm, &self.ple_out]
            .iter()
            .map(|b| b.length())
            .sum()
    }

    /// HC read for t rows of `x`: normalize the streams, down, activate, up, mix. With `pending`
    /// = (block, inject, next) the norm kernel first writes next = x + inject * block (the decode
    /// path's fused HC write) and reads that.
    #[allow(clippy::too_many_arguments)]
    fn hc_read(&self, e: &Encoder, h: &Hc, x: &Buffer, pending: Option<(&Buffer, &Buffer, &Buffer)>, mixed: &Buffer, inject: &Buffer, t: usize) -> Result<()> {
        let (down, up, norm, iw) = h.prefill_weights();
        ensure!(down.k == F && down.n == R && up.k == R && up.n == F, "prefill HC geometry");
        let (block, inj, next) = pending.unwrap_or((x, x, x));
        let prm = [t as u32, pending.is_some() as u32];
        e.setComputePipelineState(&self.p.hc_norm);
        bufs(e, &[x, norm, iw, block, inj, &self.normalized, &self.parts, next]);
        bytes(e, &prm, 8);
        dispatch(e, HC, t, 256);
        barrier(e);
        self.qmm.encode(e, down.original(), down.group, &self.normalized, 0, &self.dp, 0, t, down.n, down.k)?;
        barrier(e);
        let prm = [t as u32, 0u32];
        e.setComputePipelineState(&self.p.hc_activate);
        bufs(e, &[&self.dp, &self.parts, &self.act, inject]);
        bytes(e, &prm, 8);
        dispatch(e, (t * (R + HC)).div_ceil(256), 1, 256);
        barrier(e);
        self.qmm.encode(e, up.original(), up.group, &self.act, 0, &self.upv, 0, t, up.n, up.k)?;
        barrier(e);
        e.setComputePipelineState(&self.p.hc_mix);
        bufs(e, &[&self.upv, &self.normalized, mixed]);
        bytes(e, &prm, 8);
        dispatch(e, (t * H).div_ceil(256), 1, 256);
        barrier(e);
        Ok(())
    }

    fn bind_gdn(&self, e: &Encoder, small: &[Buffer], conv: &Buffer, state: &Buffer, t: usize) {
        bufs(e, &[&self.pqkv, &self.pz, &self.pa, &self.pb, conv, state, &small[0], &small[1], &small[2], &small[3], &self.q, &self.k, &self.v,
                  &self.gates, &self.y, &self.gated, &self.dummy, &self.conv_out]);
        bytes(e, &(t as u32), 18);
    }

    /// GDN mixer for t rows of mixed0 into gdn_out. The final state replaces the layer's committed
    /// state in place; the new convolution history (last three pre-convolution rows) replaces the
    /// committed one once gdn_prepare has read it.
    fn gdn_core(&self, s: &mut dyn Sink, g: &MetalGdn, t: usize) -> Result<()> {
        let (w, small, conv, state, slot) = g.prefill_parts();
        ensure!(slot.is_none(), "prefill needs the committed GDN state in its own buffer (no zero-copy slot)");
        ensure!(w.len() == 5 && small.len() == 4 && conv.length() >= 3 * C * 2, "prefill GDN parts");
        s.stage("gdn_proj", &mut |e| {
            for (j, out) in [(0usize, &self.pqkv), (1, &self.pz), (2, &self.pa), (3, &self.pb)] {
                self.qmm.encode(e, w[j].original(), w[j].group, &self.mixed0, 0, out, 0, t, w[j].n, w[j].k)?;
            }
            barrier(e);
            Ok(())
        })?;
        s.stage("gdn_prepare", &mut |e| {
            e.setComputePipelineState(&self.p.gdn_prepare);
            self.bind_gdn(e, small, conv, state, t);
            dispatch(e, HV, t, 128);
            barrier(e);
            e.setComputePipelineState(&self.p.split_gates);
            bufs(e, &[&self.gates, &self.g, &self.beta]);
            bytes(e, &((t * HV) as u32), 3);
            dispatch(e, (t * HV).div_ceil(256), 1, 256);
            self.copy_words(e, &self.conv_out, 0, conv, 3 * C / 2);
            barrier(e);
            Ok(())
        })?;
        // In place: each SIMD group loads and stores only its own 16 x 128 slice of the state.
        s.stage("gdn_delta", &mut |e| {
            self.gdn.encode(e, &self.q, &self.k, &self.v, state, &self.g, &self.beta, &self.y, state, t)?;
            barrier(e);
            Ok(())
        })?;
        s.stage("gdn_norm_out", &mut |e| {
            e.setComputePipelineState(&self.p.gdn_norm);
            self.bind_gdn(e, small, conv, state, t);
            dispatch(e, HV, t, 32);
            barrier(e);
            self.qmm.encode(e, w[4].original(), w[4].group, &self.gated, 0, &self.gdn_out, 0, t, w[4].n, w[4].k)?;
            barrier(e);
            Ok(())
        })
    }

    /// The MLP's router and top-10 select on mixed1 (scores per the decode arm). Router logits
    /// on the matrix units (r58_router_mma) unless QWEN_R58_ROUTER_MMA=0 (the decode path's
    /// per-row kernel).
    pub(crate) fn route(&self, e: &Encoder, weights: &metal_mlp::Weights, t: usize, scores: RoutingArithmetic) {
        self.route_with(e, weights, t, scores, router_mma())
    }
    pub(crate) fn route_with(&self, e: &Encoder, weights: &metal_mlp::Weights, t: usize, scores: RoutingArithmetic, mma: bool) {
        let (router, _, _, _) = weights.prefill_parts();
        if mma {
            e.setComputePipelineState(&self.p.router_mma);
            bufs(e, &[&self.mixed1, router, &self.logits_f]);
            bytes(e, &(t as u32), 3);
            dispatch(e, EXPERTS / 128, t.div_ceil(32), 128);
        } else {
            e.setComputePipelineState(&self.p.router);
            bufs(e, &[&self.mixed1, router, &self.logits_f]);
            dispatch(e, 128, t, 128);
        }
        barrier(e);
        e.setComputePipelineState(&self.p.select);
        bufs(e, &[&self.logits_f, &self.logits_bf, &self.ids, &self.scores]);
        bytes(e, &(matches!(scores, RoutingArithmetic::SelectedF32) as u32), 4);
        // R58 lead 1: the same expert pruning as the decode path (default off)
        let (prune, renorm) = crate::metal_mlp::moe_prune();
        bytes(e, &prune.to_bits(), 5);
        bytes(e, &(renorm as u32), 6);
        dispatch(e, t, 1, 32);
        barrier(e);
    }

    /// Front of a GDN layer for t rows of `input`: HC read, GDN, fused HC write + MLP-side read,
    /// router and select. The routes are ready for plan_routes once the GPU has finished.
    pub(crate) fn encode_gdn_front(&self, s: &mut dyn Sink, layer: &GdnLayer, input: &Buffer, t: usize, scores: RoutingArithmetic) -> Result<()> {
        ensure!(t >= 1 && t <= self.max && input.length() >= t * F * 2, "prefill rows");
        let (front, weights) = layer.prefill_parts();
        let (hc0, hc1, gdn, _) = front.prefill_parts();
        s.stage("hc_read_attn", &mut |e| self.hc_read(e, hc0, input, None, &self.mixed0, &self.inject0, t))?;
        self.gdn_core(s, gdn, t)?;
        s.stage("hc_read_mlp", &mut |e| self.hc_read(e, hc1, input, Some((&self.gdn_out, &self.inject0, &self.stream)), &self.mixed1, &self.inject1, t))?;
        s.stage("route", &mut |e| {
            self.route(e, weights, t, scores);
            Ok(())
        })
    }

    pub(crate) fn copy_words(&self, e: &Encoder, src: &Buffer, src_offset: usize, dst: &Buffer, words: usize) {
        e.setComputePipelineState(&self.p.copy);
        at(e, src, src_offset, 0);
        at(e, dst, 0, 1);
        bytes(e, &(words as u32), 2);
        dispatch(e, words.div_ceil(256), 1, 256);
    }

    /// Front of an attention layer for t rows of `input` at positions len..len+t: HC read, the
    /// q/k/v/index projections, QSA pooling (blocks completed by these rows), queries, scores and
    /// top-512 selection (only when the decode path would be sparse: len + t > 2051), QK norm /
    /// RoPE, affine8 append to the layer's KV cache, attention, gate, output projection, fused HC
    /// write + MLP-side read, router and select. The index's raw tail is rolled on the device.
    pub(crate) fn encode_attn_front(&self, s: &mut dyn Sink, layer: &AttnLayer, input: &Buffer, t: usize, len: usize, scores: RoutingArithmetic) -> Result<()> {
        ensure!(t >= 1 && t <= self.max && input.length() >= t * F * 2, "prefill rows");
        let (hc0, hc1, att, weights) = layer.prefill_parts();
        let (proj, index, core) = att.prefill_parts();
        let (iproj, inorm, itail, icap) = index.prefill_parts();
        let (anorm, cache, acap) = core.prefill_parts();
        ensure!(proj.len() == 4 && cache.len() == 6, "prefill attention parts");
        ensure!(len + t <= acap && len + t <= icap && icap <= self.capacity, "prefill attention capacity");
        let sparse = len + t > qsa::BUDGET + qsa::RATIO - 1;
        let qp = QsaParams { offset: len as u32, rows: t as u32, base: 0, accepted: t as u32 };
        let ap = AttnParams { offset: len as u32, rows: t as u32, base: 0, capacity: acap as u32, sparse: sparse as u32, splits: 1, log2_scores: 0 };
        s.stage("hc_read_attn", &mut |e| self.hc_read(e, hc0, input, None, &self.mixed0, &self.inject0, t))?;
        s.stage("attn_proj", &mut |e| {
            for (w, out) in [(&proj[0], &self.qg), (&proj[1], &self.kraw), (&proj[2], &self.vraw), (iproj, &self.iqk)] {
                self.qmm.encode(e, w.original(), w.group, &self.mixed0, 0, out, 0, t, w.n, w.k)?;
            }
            barrier(e);
            Ok(())
        })?;
        s.stage("attn_qsa_prep", &mut |e| {
            let pooled = (len % qsa::RATIO + t) / qsa::RATIO;
            if pooled > 0 {
                e.setComputePipelineState(&self.p.qsa_pool);
                bufs(e, &[&self.iqk, itail[1], inorm[1], inorm[2], itail[0]]);
                bytes(e, &qp, 5);
                dispatch(e, pooled, 1, 32);
            }
            e.setComputePipelineState(&self.p.qsa_query);
            bufs(e, &[&self.iqk, inorm[0], inorm[2], &self.iquery]);
            bytes(e, &qp, 4);
            dispatch(e, qsa::HEADS, t, 32);
            e.setComputePipelineState(&self.p.attn_prepare);
            bufs(e, &[&self.qg, &self.kraw, &anorm[0], &anorm[1], &anorm[2], &self.aquery, &self.akey, &self.agate]);
            bytes(e, &ap, 8);
            dispatch(e, QH + KH, t, 32);
            barrier(e);
            Ok(())
        })?;
        s.stage("attn_pack_score", &mut |e| {
            e.setComputePipelineState(&self.p.attn_pack);
            bufs(e, &[&self.akey, &self.vraw, &cache[0], &cache[1], &cache[2], &cache[3], &cache[4], &cache[5]]);
            bytes(e, &ap, 8);
            dispatch(e, t * META, 2, 32);
            if sparse {
                let nb = (len + t) / qsa::RATIO;
                e.setComputePipelineState(&self.p.qsa_score);
                bufs(e, &[&self.iquery, itail[0], &self.iscores]);
                bytes(e, &qp, 3);
                dispatch(e, nb.div_ceil(4), t, 128);
            }
            e.setComputePipelineState(&self.p.qsa_commit_tail);
            bufs(e, &[&self.iqk, itail[1], itail[2]]);
            bytes(e, &qp, 3);
            dispatch(e, 3, 1, 128);
            barrier(e);
            Ok(())
        })?;
        if sparse {
            s.stage("attn_select", &mut |e| {
                e.setComputePipelineState(&self.p.qsa_select);
                bufs(e, &[&self.iscores, &self.iids]);
                bytes(e, &qp, 2);
                dispatch(e, 1, t, 256);
                barrier(e);
                Ok(())
            })?;
        }
        s.stage("attn_core", &mut |e| {
            e.setComputePipelineState(&self.p.attn_nax);
            bufs(e, &[&self.aquery, &cache[0], &cache[3], &cache[1], &cache[2], &cache[4], &cache[5], &self.iids, &self.attention]);
            bytes(e, &ap, 9);
            e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: t, height: KH, depth: 1 }, MTLSize { width: 64, height: 1, depth: 1 });
            // The next forward's old raw tail: what qsa_commit_tail built (pool and tail reads done).
            let n_tail = (len + t) % qsa::RATIO;
            if n_tail > 0 {
                self.copy_words(e, itail[2], 0, itail[1], n_tail * qsa::D / 2);
            }
            barrier(e);
            Ok(())
        })?;
        s.stage("attn_gate_out", &mut |e| {
            e.setComputePipelineState(&self.p.attn_gate);
            bufs(e, &[&self.attention, &self.agate, &self.agated]);
            bytes(e, &ap, 3);
            dispatch(e, (t * Q).div_ceil(256), 1, 256);
            barrier(e);
            self.qmm.encode(e, proj[3].original(), proj[3].group, &self.agated, 0, &self.attn_out, 0, t, proj[3].n, proj[3].k)?;
            barrier(e);
            Ok(())
        })?;
        s.stage("hc_read_mlp", &mut |e| self.hc_read(e, hc1, input, Some((&self.attn_out, &self.inject0, &self.stream)), &self.mixed1, &self.inject1, t))?;
        s.stage("route", &mut |e| {
            self.route(e, weights, t, scores);
            Ok(())
        })
    }

    /// Host: stage the PLE n-gram embedding rows (t x H) for the next encode_ple.
    pub(crate) fn set_ple_rows(&self, emb: &[u16]) -> Result<()> {
        ensure!(emb.len() % H == 0 && emb.len() / H <= self.max, "PLE rows");
        upload(&self.pemb, emb)
    }

    /// Layer 1's PLE stage on t rows of `stream` into ple_out (key / value projections, norms,
    /// gate, dilated convolution over [history; rows]); the convolution history then becomes the
    /// last HISTORY normalized rows (t >= HISTORY).
    pub(crate) fn encode_ple(&self, s: &mut dyn Sink, ple: &PleTarget, stream: &Buffer, t: usize) -> Result<()> {
        ensure!(t >= HISTORY && t <= self.max && stream.length() >= t * F * 2, "prefill PLE rows");
        let (key, value, norms, conv, history) = ple.prefill_parts();
        ensure!(history.length() >= HISTORY * F * 2, "PLE history");
        s.stage("ple", &mut |e| self.ple_body(e, key, value, norms, conv, history, stream, t))
    }
    #[allow(clippy::too_many_arguments)]
    fn ple_body(&self, e: &Encoder, key: &crate::metal_ops::DenseWeight, value: &crate::metal_ops::DenseWeight, norms: &[Buffer; 3], conv: &Buffer,
                history: &Buffer, stream: &Buffer, t: usize) -> Result<()> {
        self.qmm.encode(e, key.original(), key.group, &self.pemb, 0, &self.pkey, 0, t, key.n, key.k)?;
        self.qmm.encode(e, value.original(), value.group, &self.pemb, 0, &self.pvalue, 0, t, value.n, value.k)?;
        barrier(e);
        let p = [t as u32, 0, 0, 0];
        e.setComputePipelineState(&self.p.ple_norm);
        bufs(e, &[&self.pkey, &norms[0], &self.pkn]);
        bytes(e, &p, 3);
        dispatch(e, HC, t, 256);
        bufs(e, &[stream, &norms[1], &self.pqn]);
        bytes(e, &p, 3);
        dispatch(e, HC, t, 256);
        barrier(e);
        e.setComputePipelineState(&self.p.ple_gate);
        bufs(e, &[&self.pkn, &self.pqn, &self.pvalue, &self.pgated, &self.pgates]);
        bytes(e, &p, 5);
        dispatch(e, HC, t, 256);
        barrier(e);
        e.setComputePipelineState(&self.p.ple_norm);
        bufs(e, &[&self.pgated, &norms[2], &self.pnorm]);
        bytes(e, &p, 3);
        dispatch(e, HC, t, 256);
        barrier(e);
        e.setComputePipelineState(&self.p.ple_conv);
        bufs(e, &[history, &self.pnorm, conv, &self.pgated, stream, &self.ple_out]);
        bytes(e, &p, 6);
        dispatch(e, (t * F).div_ceil(256), 1, 256);
        barrier(e);
        self.copy_words(e, &self.pnorm, (t - HISTORY) * F * 2, history, HISTORY * F / 2);
        barrier(e);
        Ok(())
    }

    /// Between the halves: sort the t rows' routed slots by expert (blocks of <= 32 rows).
    /// Returns the block count.
    pub(crate) fn plan_routes(&self, t: usize) -> Result<usize> {
        // SAFETY: shared storage; the front's command buffer has completed.
        let ids = unsafe { std::slice::from_raw_parts(self.ids.contents().cast::<u32>().as_ptr(), t * TOPK).to_vec() };
        ensure!(ids.iter().all(|&i| (i as usize) < EXPERTS), "routed expert id out of bank");
        let pl = plan(&ids, 32)?;
        upload(&self.sorted, &pl.ids)?;
        upload(&self.order, &pl.order)?;
        upload(&self.inverse, &pl.inverse)?;
        upload(&self.blocks, &pl.blocks)?;
        Ok(pl.blocks.len())
    }

    fn mpp(&self, e: &Encoder, x: &Buffer, w: &metal_mlp::ExpertParts<'_>, y: &Buffer, k: usize, n: usize, blocks: usize) {
        e.setComputePipelineState(&self.p.mpp);
        at(e, x, 0, 0);
        at(e, w.0, w.3[0], 1);
        at(e, &self.sorted, 0, 2);
        at(e, y, 0, 3);
        at(e, &self.blocks, 0, 5);
        bytes(e, &[k as u32, n as u32, 1u32, 0u32], 4);
        dispatch(e, n / 128, blocks, 128);
    }

    /// MLP tail for t rows: routed experts (expert-blocked), shared expert, join, then the final
    /// HC write of the layer into `output`.
    pub(crate) fn encode_mlp_tail(&self, s: &mut dyn Sink, weights: &metal_mlp::Weights, output: &Buffer, t: usize, blocks: usize) -> Result<()> {
        ensure!(t >= 1 && t <= self.max && output.length() >= t * F * 2 && blocks >= 1 && blocks <= t * TOPK, "prefill MLP tail");
        let (_, sgate, ex, shared) = weights.prefill_parts();
        let slots = t * TOPK;
        let p = [H as u32, I as u32, TOPK as u32, 1, 1, 0, 0, 0];
        s.stage("moe_prepare_shared", &mut |e| {
            e.setComputePipelineState(&self.p.prepare);
            bufs(e, &[&self.mixed1, ex[0].1, ex[1].1, &self.sorted, &self.order, &self.pg, &self.pu]);
            at(e, ex[0].1, ex[0].3[1], 1);
            at(e, ex[1].1, ex[1].3[1], 2);
            bytes(e, &p, 7);
            dispatch(e, H / 128, slots, 32);
            // The shared expert's gate and projections read only mixed1: same dependency interval.
            e.setComputePipelineState(&self.p.shared_gate);
            bufs(e, &[&self.mixed1, sgate, &self.gate_f, &self.gate_bf]);
            dispatch(e, t, 1, 32);
            self.qmm.encode(e, shared[0].original(), shared[0].group, &self.mixed1, 0, &self.sg, 0, t, shared[0].n, shared[0].k)?;
            self.qmm.encode(e, shared[1].original(), shared[1].group, &self.mixed1, 0, &self.su, 0, t, shared[1].n, shared[1].k)?;
            barrier(e);
            Ok(())
        })?;
        s.stage("moe_gate_up", &mut |e| {
            self.mpp(e, &self.pg, &ex[0], &self.rg, H, I, blocks);
            self.mpp(e, &self.pu, &ex[1], &self.ru, H, I, blocks);
            e.setComputePipelineState(&self.p.shared_act);
            bufs(e, &[&self.sg, &self.su, &self.sa]);
            bytes(e, &((t * I) as u32), 3);
            dispatch(e, (t * I).div_ceil(256), 1, 256);
            barrier(e);
            Ok(())
        })?;
        s.stage("moe_mid_shared_down", &mut |e| {
            e.setComputePipelineState(&self.p.mid);
            bufs(e, &[&self.rg, &self.ru, ex[0].2, ex[1].2, ex[2].1, &self.sorted, &self.middle]);
            at(e, ex[0].2, ex[0].3[2], 2);
            at(e, ex[1].2, ex[1].3[2], 3);
            at(e, ex[2].1, ex[2].3[1], 4);
            bytes(e, &p, 7);
            dispatch(e, I / 128, slots, 32);
            self.qmm.encode(e, shared[2].original(), shared[2].group, &self.sa, 0, &self.sd, 0, t, shared[2].n, shared[2].k)?;
            barrier(e);
            Ok(())
        })?;
        s.stage("moe_down", &mut |e| {
            self.mpp(e, &self.middle, &ex[2], &self.rd, I, H, blocks);
            barrier(e);
            Ok(())
        })?;
        s.stage("moe_finish_write", &mut |e| {
            e.setComputePipelineState(&self.p.finish);
            bufs(e, &[&self.rd, ex[2].2, &self.sorted, &self.inverse, &self.scores, &self.routed_f, &self.routed_bf]);
            at(e, ex[2].2, ex[2].3[2], 1);
            bytes(e, &p, 7);
            dispatch(e, H / 128, t, TOPK * 32);
            barrier(e);
            e.setComputePipelineState(&self.p.join);
            bufs(e, &[&self.sd, &self.gate_bf, &self.routed_bf, &self.sigmoid, &self.shared, &self.mlp_out]);
            bytes(e, &((t * H) as u32), 6);
            dispatch(e, (t * H).div_ceil(256), 1, 256);
            barrier(e);
            e.setComputePipelineState(&self.p.hc_write);
            bufs(e, &[&self.stream, &self.mlp_out, &self.inject1, output]);
            bytes(e, &[t as u32, 0u32], 8);
            dispatch(e, (t * F).div_ceil(256), 1, 256);
            barrier(e);
            Ok(())
        })
    }

    /// Diagnostics after a completed front: (post-mixer stream [t][F], MLP input [t][H]).
    pub(crate) fn front_outputs(&self, t: usize) -> (Vec<u16>, Vec<u16>) {
        (read16(&self.stream, t * F), read16(&self.mixed1, t * H))
    }
    pub(crate) fn read_rows(b: &Buffer, t: usize) -> Vec<u16> {
        read16(b, t * F)
    }
    /// Diagnostics: the MLP output rows [t][H], the selected experts [t][TOPK], and replacing the
    /// MLP input rows (e.g. with the decode path's, to isolate the tail).
    pub(crate) fn mlp_outputs(&self, t: usize) -> Vec<u16> {
        read16(&self.mlp_out, t * H)
    }
    pub(crate) fn routes(&self, t: usize) -> Vec<u32> {
        // SAFETY: shared storage; the routing command buffer has completed.
        unsafe { std::slice::from_raw_parts(self.ids.contents().cast::<u32>().as_ptr(), t * TOPK).to_vec() }
    }
    pub(crate) fn set_mlp_input(&self, rows: &[u16]) -> Result<()> {
        ensure!(rows.len() % H == 0 && rows.len() / H <= self.max, "MLP input rows");
        upload(&self.mixed1, rows)
    }
}
