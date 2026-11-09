//! R55 native Metal 48-layer target (canonical EXL3 4bpw pack): resident weights and
//! state, one command buffer per forward of 1-16 rows, any accepted prefix committed.
//! Embedding rows are decoded on the host; layer 1's PLE gathers its BF16 n-gram rows on
//! the host; everything else stays on the GPU. Logits are FP32 in shared memory.
use crate::{
    embedding::{RawEmbeddings, VOCAB},
    hc::{self, F, H, MAX_ROWS},
    metal_attention_layer as al,
    metal_full_attention as fa,
    metal_hc::{Arm as FrontArm, Hc, HcOps, Precision},
    metal_mlp::{Experts, SharedPrecision},
    metal_ops::{Arm as DenseArm, Buffer, DenseOps, DenseWeight, Obj, alloc, upload},
    metal_prefill::Prefill,
    metal_ple_target::PleTarget,
    metal_qsa::{Precision as QsaPrecision, Selector},
    metal_target_attention::Arm as Core,
    metal_target_layer as gl,
    moe::FullExpertBank,
    projection::Pack,
    target_attention_layer, target_layer,
    target_mlp::RoutingArithmetic,
};
use anyhow::{Context, Result, ensure};
use objc2_metal::*;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

/// Default model directory for diagnostics; the stage server takes QWEN_HYBRID_MODEL.
pub const MODEL: &str = "models/Qwen3.8-Flash-Next-Sushi-4bpw";
const HEAD_TILE: usize = 16384;
/// R56: threads for the commit's state snapshot copies.
const COMMIT_THREADS: usize = 6;

enum Layer {
    Gdn(gl::Engine),
    Attn(al::Engine),
}

pub struct Timing {
    pub host_prepare_us: f64,
    pub encode_us: f64,
    pub gpu_us: f64,
    pub wait_us: f64,
    pub commit_us: f64,
    /// R56: buffer barriers encoded in this forward (dependency intervals).
    pub barriers: usize,
    /// R58: when the GPU started and finished this forward, in host time (seconds, the clock of
    /// host_time_s), so a host-side trace can place GPU work on its own timeline.
    pub gpu_start_s: f64,
    pub gpu_end_s: f64,
    /// R58: commit to GPU start, microseconds.
    pub launch_us: f64,
    /// R58 QWEN_R58_EVENT_WAIT=1: host time when the forward's shared-event signal was seen
    /// (0 otherwise); waitUntilCompleted returned at notice_s.
    pub event_s: f64,
    pub notice_s: f64,
}

/// R58 DFlash2: a per-layer callback during the engine prefill: (encoder, layer, the layer's output
/// streams for the chunk, rows, the chunk's first row).
pub(crate) type TapHook<'a> = dyn FnMut(&objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, usize, &Buffer, usize, usize) -> Result<()> + 'a;
/// R58: host time in seconds on the clock Metal uses for GPUStartTime/GPUEndTime
/// (mach_absolute_time scaled by the timebase, as CACurrentMediaTime).
pub fn host_time_s() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    static TB: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    let (n, d) = *TB.get_or_init(|| {
        let mut t = Timebase { numer: 0, denom: 0 };
        unsafe { mach_timebase_info(&mut t) };
        (t.numer as f64, t.denom as f64)
    });
    let ticks = unsafe { mach_absolute_time() };
    ticks as f64 * n / d / 1e9
}

/// R56: a submitted forward whose command buffer may still be executing (see forward_submit).
pub struct InFlight {
    cb: Obj<dyn MTLCommandBuffer>,
    /// R58: the first command buffer when the forward is split (QWEN_R58_SPLIT_CB); its GPU start
    /// is the forward's.
    first: Option<Obj<dyn MTLCommandBuffer>>,
    /// R58: host time right after commit (launch latency = GPU start - this).
    commit_s: f64,
    rows: usize,
    host_prepare_us: f64,
    encode_us: f64,
    barriers: usize,
    /// R58: the shared-event value signalled at this forward's end (0: no event).
    seq: u64,
}
impl InFlight {
    /// R58: true once the command buffer has finished (completed or failed); forward_wait then
    /// returns without blocking. A cheap status read for completion-aware scheduling.
    pub fn done(&self) -> bool {
        matches!(self.cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
    }
    /// R58: the predicted host time this forward's GPU work ends (commit + launch + GPU, from
    /// running averages per row count kept by forward_wait); infinity before the first sample.
    pub fn predicted_end_s(&self) -> f64 {
        let est = VERIFY_EST.lock().unwrap();
        let (g, l) = est[self.rows.min(MAX_ROWS)];
        if g > 0. { self.commit_s + l + g } else { f64::INFINITY }
    }
}
/// R58 GPU top-K over the target's tiled vocabulary logits (kernels/r58_topk.metal): the same
/// rule as top_rows_by on the host, K = 20 per row. QWEN_R58_GPU_TOPK: "1"/unset on, "0" off,
/// "check" on and compared against the host scan (any difference is an error). It adds ~0.2-0.4 ms
/// of GPU to an 8-row verify and saves ~0.9 ms of host scan per page/round.
pub(crate) struct GpuTopK {
    pipes: [Obj<dyn MTLComputePipelineState>; 2],
    cv: Buffer,
    ci: Buffer,
    ov: Buffer,
    oi: Buffer,
    check: bool,
}
const TOPK_K: usize = 20;
const TOPK_BLOCK: usize = 2048;
impl GpuTopK {
    fn new(device: &objc2::runtime::ProtocolObject<dyn MTLDevice>) -> Result<Option<Self>> {
        // default on (r58-topk-*: DFlash2 round -0.6 ms net, identical top-k in check mode)
        let mode = std::env::var("QWEN_R58_GPU_TOPK").unwrap_or_else(|_| "1".into());
        if mode != "1" && mode != "check" {
            return Ok(None);
        }
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let lib = device
            .newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(include_str!("../kernels/r58_topk.metal")), Some(&options))
            .map_err(|e| anyhow::anyhow!("top-k MSL: {e}"))?;
        let pipe = |name: &str| -> Result<Obj<dyn MTLComputePipelineState>> {
            let f = lib.newFunctionWithName(&objc2_foundation::NSString::from_str(name)).context("top-k function")?;
            device.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow::anyhow!("top-k pipeline {name}: {e}"))
        };
        let blocks = VOCAB.div_ceil(TOPK_BLOCK);
        Ok(Some(Self {
            pipes: [pipe("r58_topk_partial")?, pipe("r58_topk_merge")?],
            cv: crate::metal_ops::alloc(device, MAX_ROWS * blocks * TOPK_K * 4)?,
            ci: crate::metal_ops::alloc(device, MAX_ROWS * blocks * TOPK_K * 4)?,
            ov: crate::metal_ops::alloc(device, MAX_ROWS * TOPK_K * 4)?,
            oi: crate::metal_ops::alloc(device, MAX_ROWS * TOPK_K * 4)?,
            check: mode == "check",
        }))
    }
    /// After the head's dispatches: per-row top-K of the `n`-row logits layout, first `rows` rows.
    fn encode(&self, e: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, logits: &Buffer, n: usize, rows: usize, bf16: bool) {
        let blocks = VOCAB.div_ceil(TOPK_BLOCK);
        let p: [u32; 8] = [n as u32, HEAD_TILE as u32, rows as u32, bf16 as u32, blocks as u32, VOCAB as u32, 0, 0];
        crate::metal_ops::barrier(e);
        e.setComputePipelineState(&self.pipes[0]);
        unsafe {
            for (i, b) in [logits, &self.cv, &self.ci].into_iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(std::ptr::NonNull::from(&p).cast(), 32, 3);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: blocks, height: rows, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        crate::metal_ops::barrier(e);
        e.setComputePipelineState(&self.pipes[1]);
        unsafe {
            for (i, b) in [&self.cv, &self.ci, &self.ov, &self.oi].into_iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(std::ptr::NonNull::from(&p).cast(), 32, 4);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: rows, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
    }
    fn read(&self, rows: usize, k: usize) -> Vec<Vec<(u32, f32)>> {
        // SAFETY: shared storage, written by a completed command buffer.
        let v = unsafe { std::slice::from_raw_parts(self.ov.contents().cast::<f32>().as_ptr(), MAX_ROWS * TOPK_K) };
        let i = unsafe { std::slice::from_raw_parts(self.oi.contents().cast::<u32>().as_ptr(), MAX_ROWS * TOPK_K) };
        (0..rows).map(|r| (0..k).map(|j| (i[r * TOPK_K + j], v[r * TOPK_K + j])).collect()).collect()
    }
}
/// R58: running averages (GPU seconds, launch seconds) of forwards by row count.
static VERIFY_EST: std::sync::Mutex<[(f64, f64); MAX_ROWS + 1]> = std::sync::Mutex::new([(0., 0.); MAX_ROWS + 1]);
pub struct MetalModel {
    _device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    pack: Pack,
    dense: DenseOps,
    hc_ops: HcOps,
    layers: Vec<Layer>,
    ple: PleTarget,
    mixer: Hc,
    head: Vec<(usize, DenseWeight)>,
    x0: Buffer,
    logits: Buffer,
    logits_bf: Buffer,
    head_sums: Buffer,
    len: usize,
    capacity: usize,
    pending: usize,
    /// R56: a submitted forward not yet waited for.
    in_flight: bool,
    /// R58 selector-guided seed: (offset, rows) of the last decode forward, whose per-layer QSA
    /// selections the engines still hold (rows 0: none since the last reset or prefill).
    last_forward: (usize, usize),
    /// R58 QWEN_R58_EVENT_WAIT=1: a shared event each forward signals at its end, and its count.
    event: Option<Obj<dyn MTLSharedEvent>>,
    event_seq: u64,
    /// R58 GPU top-K (QWEN_R58_GPU_TOPK): pipelines (partial, merge), block candidates and the
    /// per-row results; `topk_ready` = (rows, bf16 ranking) the last forward/head appended.
    topk: Option<GpuTopK>,
    topk_ready: Option<(usize, bool)>,
    concurrent: bool,
    gdn_arm: gl::Arm,
    attn_arm: al::Arm,
    /// R58: the large-row prefill engine (created on first use, sized for the largest chunk).
    prefill: Option<Prefill>,
    /// R58: the logits buffer holds the last prefilled token's row.
    prefill_logits: bool,
    /// R58: embedding rows for prefill (opened on first use).
    embeddings: Option<crate::embedding::EmbeddingTable>,
    /// R58: the routed experts' buffers (60 GB of file mappings) in one residency set. A prefill
    /// requests it and attaches it to the queue (without it every prefill command buffer paid
    /// ~10 ms of residency work at commit: 8K prompt 1.9 s of 7.1 s) and leaves it so (ending it
    /// right after cost the next decode forward ~0.4 s of launch latency); release_residency()
    /// ends it, e.g. when idle. Off with QWEN_R58_RESIDENCY=0.
    residency: Option<Obj<dyn MTLResidencySet>>,
    residency_active: bool,
}

/// R58: a whole prefill (see MetalModel::prefill).
#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct PrefillTiming {
    pub rows: usize,
    pub chunks: Vec<usize>,
    pub wall_s: f64,
    /// Sum of the command buffers' GPU spans.
    pub gpu_s: f64,
    /// Host: embedding and PLE n-gram rows (per chunk, before its first command buffer); the PLE
    /// gather's part.
    pub host_rows_s: f64,
    pub ple_gather_s: f64,
    /// Host: routing plans between command buffers.
    pub plan_s: f64,
    pub command_buffers: usize,
    pub engine_bytes: usize,
    /// Per chunk: wall and GPU seconds; GPU seconds of each command buffer (front of layer 0, then
    /// tail of layer l + front of layer l + 1).
    pub chunk_wall_s: Vec<f64>,
    pub chunk_gpu_s: Vec<f64>,
    pub command_gpu_s: Vec<Vec<f64>>,
    /// Host-side command buffer costs summed over the prefill: encoding (creation to commit),
    /// launch (commit to GPU start), completion (GPU end to the host's wait returning).
    pub encode_s: f64,
    pub launch_s: f64,
    pub complete_s: f64,
}

/// R58: one layer run alone through the prefill engine (see MetalModel::prefill_layer).
pub struct PrefillLayer {
    /// BF16 rows x F (the layer's output streams).
    pub output: Vec<u16>,
    /// BF16 rows x F (streams after the mixer's HC write) and rows x H (the MLP input).
    pub stream: Vec<u16>,
    pub mlp_input: Vec<u16>,
    /// BF16 rows x H: the MLP output.
    pub mlp_output: Vec<u16>,
    /// With a reference MLP input: the MLP output from it, and per row whether the top-10 expert
    /// set selected from the engine's own MLP input differs from the one selected from it.
    pub ref_mlp_output: Vec<u16>,
    pub route_flips: Vec<bool>,
    /// Rows whose top-10 set from the engine's router differs from the decode path's per-row
    /// router kernel on the same MLP input (only when the engine's router is the MMA one).
    pub router_kernel_flips: usize,
    /// GPU seconds per chunk: (front command buffer, MLP tail command buffer).
    pub gpu_s: Vec<(f64, f64)>,
    /// Host seconds per chunk spent building the routing plan.
    pub plan_s: Vec<f64>,
    pub engine_bytes: usize,
}

impl MetalModel {
    pub fn load(capacity: usize, mut observe: impl FnMut(Value) -> Result<()>) -> Result<Self> {
        ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use the guarded runner");
        ensure!((64..=crate::metal_target_attention::RESIDENT_MAX_CTX).contains(&capacity), "context capacity");
        let t = Instant::now();
        let root = Path::new(MODEL);
        let pack = Pack::open(root)?;
        let device = MTLCreateSystemDefaultDevice().context("Metal device")?;
        let queue = device.newCommandQueue().context("Metal queue")?;
        let dense = DenseOps::new(&device)?;
        let hc_ops = HcOps::new(&device)?;
        let ple = PleTarget::new(&device, &pack, root)?;
        let mixer = Hc::new(&device, &hc::Weights::load_mixer(&pack)?)?;
        // Head tiles of 16384 vocabulary rows (Affine8 component bounds; 512 MiB buffers).
        let mut head = vec![];
        let mut first = 0;
        while first < VOCAB {
            let n = HEAD_TILE.min(VOCAB - first);
            let w = crate::dense::Affine8::load_rows(&pack, "language_model.lm_head", H, first, n)?;
            head.push((first, DenseWeight::new(&device, &w)?));
            first += n;
        }
        observe(json!({"kind":"metal_model_load","part":"io","seconds":t.elapsed().as_secs_f64(),"head_tiles":head.len()}))?;
        let mut layers = Vec::with_capacity(48);
        // Routed experts are no-copy mappings of the pack files by default (page cache,
        // reclaimable); QWEN_R55_MAPPED=0 copies them into owned buffers instead.
        let mapped = std::env::var("QWEN_R55_MAPPED").as_deref() != Ok("0");
        for i in 0..48 {
            let s = Instant::now();
            let bank = if mapped {
                None
            } else {
                Some(FullExpertBank::load(&pack, i).with_context(|| format!("layer {i} experts"))?)
            };
            let experts = match &bank {
                Some(b) => Experts::Bank(b),
                None => Experts::Mapped(&pack, i),
            };
            let layer = if i % 4 == 3 {
                let w = target_attention_layer::Weights::load(&pack, i)?;
                Layer::Attn(al::Engine::new_on(&w, experts, capacity, device.clone(), queue.clone())?)
            } else {
                let w = target_layer::Weights::load_body(&pack, i)?;
                Layer::Gdn(gl::Engine::new_on(&w, experts, device.clone(), queue.clone())?)
            };
            layers.push(layer);
            observe(json!({"kind":"metal_model_load","layer":i,"seconds":s.elapsed().as_secs_f64()}))?;
        }
        let concurrent = std::env::var("QWEN_R55_CONCURRENT").as_deref() != Ok("0");
        let residency = if std::env::var("QWEN_R58_RESIDENCY").as_deref() != Ok("0") {
            let t = Instant::now();
            let desc = MTLResidencySetDescriptor::new();
            desc.setLabel(Some(&objc2_foundation::NSString::from_str("routed experts")));
            let set = device.newResidencySetWithDescriptor_error(&desc).map_err(|e| anyhow::anyhow!("residency set: {e}"))?;
            let mut n = 0;
            for l in &layers {
                let w = match l {
                    Layer::Gdn(g) => g.prefill_parts().1,
                    Layer::Attn(a) => a.prefill_parts().3,
                };
                for b in w.expert_buffers() {
                    set.addAllocation(objc2::runtime::ProtocolObject::from_ref(&**b));
                    n += 1;
                }
            }
            set.commit();
            observe(json!({"kind":"metal_model_residency","allocations":n,"bytes":set.allocatedSize(),"seconds":t.elapsed().as_secs_f64()}))?;
            Some(set)
        } else {
            None
        };
        let event = if std::env::var("QWEN_R58_EVENT_WAIT").as_deref() == Ok("1") { device.newSharedEvent() } else { None };
        let topk = GpuTopK::new(&device)?;
        let mut m = Self {
            x0: alloc(&device, MAX_ROWS * F * 2)?,
            logits: alloc(&device, MAX_ROWS * VOCAB * 4)?,
            logits_bf: alloc(&device, MAX_ROWS * VOCAB * 2)?,
            head_sums: alloc(&device, MAX_ROWS * HEAD_TILE * 4)?,
            _device: device,
            queue,
            pack,
            dense,
            hc_ops,
            layers,
            ple,
            mixer,
            head,
            len: 0,
            capacity,
            pending: 0,
            in_flight: false,
            last_forward: (0, 0),
            event,
            topk,
            topk_ready: None,
            event_seq: 0,
            concurrent,
            gdn_arm: gl::Arm {
                front: FrontArm { precision: Precision::PackedF16x2, fused_write: true },
                scores: RoutingArithmetic::StagedBf16,
                shared: if std::env::var("QWEN_R55_SHARED").as_deref() == Ok("auto") { SharedPrecision::Auto } else { SharedPrecision::Qmv },
            },
            attn_arm: al::Arm {
                hc: Precision::PackedF16x2,
                attention: fa::Arm {
                    precision: QsaPrecision::PackedF16x2,
                    core: Core::Nax8,
                    // R58: two-stage selection past ~48K tokens, default on (same ids; r58-sel2-192k:
                    // verify -1.2 ms at 192K); QWEN_R58_QSA_SELECT2=0 keeps one threadgroup per row
                    selector: if std::env::var("QWEN_R58_QSA_SELECT2").as_deref() == Ok("0") { Selector::Radix11 } else { Selector::Radix11Two },
                },
            },
            prefill: None,
            prefill_logits: false,
            embeddings: None,
            residency,
            residency_active: false,
        };
        m.reset()?;
        observe(json!({"kind":"metal_model_loaded","seconds":t.elapsed().as_secs_f64(),"capacity":capacity,"concurrent_dispatch":concurrent}))?;
        Ok(m)
    }
    /// R57: the 4 HC streams (BF16, rows x F) layer `layer` wrote in the last completed forward
    /// (DFlash2's target taps are their means after layers 5, 15, 25, 35 and 45).
    pub fn layer_streams(&self, layer: usize, rows: usize) -> Result<Vec<u16>> {
        ensure!(!self.in_flight, "layer streams with a forward in flight");
        ensure!((1..=hc::MAX_ROWS).contains(&rows) && rows <= self.pending.max(1) && layer < self.layers.len(), "layer streams bounds");
        let b = match &self.layers[layer] {
            Layer::Gdn(g) => g.output(),
            Layer::Attn(a) => a.output(),
        };
        ensure!(b.length() >= rows * hc::F * 2, "layer output size");
        // SAFETY: shared-storage buffer; the forward that wrote it has completed (not in flight).
        let v = unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), rows * hc::F) };
        Ok(v.to_vec())
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    /// R58 DFlash2: the device and queue, for components that share the target's GPU.
    pub(crate) fn metal(&self) -> (&objc2::runtime::ProtocolObject<dyn MTLDevice>, &objc2::runtime::ProtocolObject<dyn MTLCommandQueue>) {
        (&self._device, &self.queue)
    }
    /// R58 DFlash2: the queue as a retained object (for the stage clock).
    pub(crate) fn queue_obj(&self) -> &Obj<dyn MTLCommandQueue> {
        &self.queue
    }
    /// R58 DFlash2: the target's canonical pack (embedding rows for draft blocks).
    pub fn pack(&self) -> &Pack {
        &self.pack
    }
    /// R58 DFlash2: the buffer that holds the 4 HC streams (rows x F, BF16) layer `layer` wrote in
    /// the last forward. It stays valid until the next forward; read it only once that forward has completed.
    pub(crate) fn layer_output(&self, layer: usize) -> Result<&Buffer> {
        ensure!(!self.in_flight && layer < self.layers.len(), "layer output bounds");
        Ok(match &self.layers[layer] {
            Layer::Gdn(g) => g.output(),
            Layer::Attn(a) => a.output(),
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        ensure!(!self.in_flight, "reset with a forward in flight");
        self.last_forward = (0, 0);
        for l in &mut self.layers {
            match l {
                Layer::Gdn(g) => g.reset_state()?,
                Layer::Attn(a) => a.reset_state()?,
            }
        }
        self.ple.reset()?;
        self.len = 0;
        self.pending = 0;
        self.prefill_logits = false;
        Ok(())
    }
    pub fn position(&self) -> usize {
        self.len
    }
    /// R56: start background reads of the PLE n-gram rows a later forward over `tokens` will
    /// need, hashed after `previous` (the two tokens before tokens[0]); returns at once.
    pub fn prefetch_ple(&self, previous: [u32; 2], tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= MAX_ROWS, "PLE prefetch rows");
        self.ple.prefetch(previous, tokens)
    }
    /// R56: zero-copy GDN commits (the next forward reads the accepted row's state snapshot in
    /// place, saving ~108 MiB of copies per commit). commit(0) after a forward is then refused.
    pub fn set_zero_copy_commit(&mut self, on: bool) -> Result<()> {
        ensure!(self.pending == 0 && !self.in_flight, "zero-copy switch with a forward pending");
        for l in &self.layers {
            if let Layer::Gdn(g) = l {
                g.set_zero_copy(on);
            }
        }
        Ok(())
    }
    /// One forward over `tokens` at the current position. Logits stay pending until
    /// `commit(accepted)`; no other forward may run before it.
    /// R58 screen: the decode attention arm at context length `len`. QWEN_R58_ATTN_CORE_LONG
    /// (nax4/nax8/nax16/simd8/simd16) replaces the core from QWEN_R58_ATTN_LONG_FROM tokens on
    /// (default 16384); unset keeps Nax8 everywhere. Positions keep one arm whatever the page shape.
    fn attn_arm_at(&self, len: usize) -> al::Arm {
        use crate::metal_target_attention::Arm as Core;
        static LONG: std::sync::OnceLock<Option<(Core, usize)>> = std::sync::OnceLock::new();
        let long = *LONG.get_or_init(|| {
            let core = match std::env::var("QWEN_R58_ATTN_CORE_LONG").as_deref() {
                Ok("nax4") => Core::Nax4,
                Ok("nax8") => Core::Nax8,
                Ok("nax16") => Core::Nax16,
                Ok("simd8") => Core::Simd8,
                Ok("simd16") => Core::Simd16,
                _ => return None,
            };
            let from = std::env::var("QWEN_R58_ATTN_LONG_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
            Some((core, from))
        });
        let mut arm = self.attn_arm;
        if let Some((core, from)) = long {
            if len >= from {
                arm.attention.core = core;
            }
        }
        arm
    }
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Timing> {
        let f = self.forward_submit(tokens)?;
        self.forward_wait(f)
    }
    /// R56: encode and commit the forward over `tokens`, returning without waiting, so the host
    /// can work (e.g. drive the CUDA drafter) while the GPU executes. forward_wait completes it;
    /// nothing else may touch this model in between.
    pub fn forward_submit(&mut self, tokens: &[u32]) -> Result<InFlight> {
        ensure!(self.pending == 0 && !self.in_flight, "uncommitted forward");
        self.prefill_logits = false;
        let rows = tokens.len();
        ensure!((1..=MAX_ROWS).contains(&rows) && self.len + rows <= self.capacity, "forward rows/capacity");
        self.last_forward = (self.len, rows);
        let t = Instant::now();
        let idx: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let emb = RawEmbeddings::load(&self.pack, &idx)?;
        let mut x = Vec::with_capacity(rows * F);
        for r in 0..rows {
            for _ in 0..hc::HC {
                x.extend_from_slice(&emb.values[r * H..(r + 1) * H]);
            }
        }
        upload(&self.x0, &x)?;
        self.ple.prepare(tokens)?;
        let host_prepare_us = t.elapsed().as_secs_f64() * 1e6;
        let t = Instant::now();
        let b0 = crate::metal_ops::BARRIERS.load(std::sync::atomic::Ordering::Relaxed);
        let mut cb = self.queue.commandBuffer().context("forward command")?;
        let new_encoder = |cb: &Obj<dyn MTLCommandBuffer>| {
            if self.concurrent {
                cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            } else {
                cb.computeCommandEncoder()
            }
            .context("forward encoder")
        };
        let mut e = new_encoder(&cb)?;
        // R58 QWEN_R58_SPLIT_CB=N: commit the first N layers as their own command buffer, so the GPU
        // starts while the rest is encoded. Same dispatches; the tracked buffers keep the order.
        // Default 6 (measured: verify 46.6 -> 45.2 ms at 8 rows; 0 turns it off).
        let split: usize = std::env::var("QWEN_R58_SPLIT_CB").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let mut first: Option<Obj<dyn MTLCommandBuffer>> = None;
        let mut input: &Buffer = &self.x0;
        // R56 QWEN_R56_BARRIER_TRACE=1: barriers per component of this forward.
        let trace = std::env::var("QWEN_R56_BARRIER_TRACE").as_deref() == Ok("1");
        let count = || crate::metal_ops::BARRIERS.load(std::sync::atomic::Ordering::Relaxed);
        let (mut marks, mut last) = (vec![], count());
        let mut mark = |label: String, marks: &mut Vec<(String, usize)>| {
            if trace {
                let c = count();
                marks.push((label, c - last));
                last = c;
            }
        };
        for (i, l) in self.layers.iter().enumerate() {
            if split > 1 && i == split {
                e.endEncoding();
                cb.commit();
                first = Some(cb);
                cb = self.queue.commandBuffer().context("forward command (second part)")?;
                e = new_encoder(&cb)?;
            }
            if i == 1 {
                self.ple.encode(&e, input, rows, &self.dense)?;
                input = self.ple.output();
                mark("ple".into(), &mut marks);
            }
            match l {
                Layer::Gdn(g) => {
                    g.encode_resident(&e, input, rows, self.gdn_arm)?;
                    input = g.output();
                    mark(format!("gdn{i}"), &mut marks);
                }
                Layer::Attn(a) => {
                    a.encode_resident(&e, input, self.len, rows, rows, self.attn_arm_at(self.len))?;
                    input = a.output();
                    mark(format!("attn{i}"), &mut marks);
                }
            }
        }
        self.hc_ops.read(&e, &self.mixer, input, rows, Precision::PackedF16x2, None)?;
        mark("mixer".into(), &mut marks);
        // R56: one head arm for every row count, so a position's logits do not depend on the
        // page size (QWEN_R56_HEAD_QMV1=1 restores QMV at one row).
        let arm = if rows == 1 && std::env::var("QWEN_R56_HEAD_QMV1").as_deref() == Ok("1") { DenseArm::Qmv4 } else { DenseArm::FragmentPacked { splits: 1, residual: true } };
        for (first, w) in &self.head {
            self.dense.encode(&e, w, self.mixer.mixed(), &self.logits, &self.logits_bf, &self.head_sums, rows, first * rows, arm)?;
        }
        self.topk_ready = None;
        if let Some(tk) = &self.topk {
            tk.encode(&e, &self.logits, rows, rows, false);
            self.topk_ready = Some((rows, false));
        }
        mark("head".into(), &mut marks);
        if trace {
            println!("{}", serde_json::json!({"kind":"barrier_trace","rows":rows,"marks":marks}));
            let mut sites = crate::metal_ops::SITES.lock().unwrap();
            if !sites.is_empty() {
                let s: Vec<String> = sites.iter().map(|l| format!("{}:{}", l.file(), l.line())).collect();
                println!("{}", serde_json::json!({"kind":"barrier_sites","rows":rows,"sites":s}));
                sites.clear();
            }
        }
        e.endEncoding();
        let seq = match &self.event {
            Some(ev) => {
                self.event_seq += 1;
                cb.encodeSignalEvent_value(objc2::runtime::ProtocolObject::from_ref(&**ev), self.event_seq);
                self.event_seq
            }
            None => 0,
        };
        cb.commit();
        let commit_s = host_time_s();
        let encode_us = t.elapsed().as_secs_f64() * 1e6;
        self.in_flight = true;
        let barriers = crate::metal_ops::BARRIERS.load(std::sync::atomic::Ordering::Relaxed) - b0;
        Ok(InFlight { cb, first, commit_s, rows, host_prepare_us, encode_us, barriers, seq })
    }
    /// R56: wait for a submitted forward; its logits then stay pending until commit.
    pub fn forward_wait(&mut self, f: InFlight) -> Result<Timing> {
        ensure!(self.in_flight, "no forward in flight");
        let w = Instant::now();
        // R58 QWEN_R58_SPIN_WAIT=1: poll the command buffer's status (one core spins) instead of
        // sleeping in waitUntilCompleted, to cut the wake-up latency after the GPU finishes.
        if std::env::var("QWEN_R58_SPIN_WAIT").as_deref() == Ok("1") {
            while !matches!(f.cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error) {
                std::hint::spin_loop();
            }
        }
        // R58: with the shared event, spin until the GPU's end-of-forward signal (bounded), to
        // see how much later waitUntilCompleted returns than the signal itself.
        let mut event_s = 0.;
        if let (Some(ev), true) = (&self.event, f.seq > 0) {
            let t0 = Instant::now();
            while ev.signaledValue() < f.seq && t0.elapsed().as_secs_f64() < 10. && !matches!(f.cb.status(), MTLCommandBufferStatus::Error) {
                std::hint::spin_loop();
            }
            event_s = host_time_s();
        }
        f.cb.waitUntilCompleted();
        let notice_s = host_time_s();
        let wait_us = w.elapsed().as_secs_f64() * 1e6;
        self.in_flight = false;
        ensure!(f.cb.status() == MTLCommandBufferStatus::Completed, "forward failed: {:?}", f.cb.error());
        if let Some(c) = &f.first {
            ensure!(c.status() == MTLCommandBufferStatus::Completed, "forward (first part) failed: {:?}", c.error());
        }
        let (gpu_start_s, gpu_end_s) = (f.first.as_ref().map_or(f.cb.GPUStartTime(), |c| c.GPUStartTime()), f.cb.GPUEndTime());
        let gpu_us = (gpu_end_s - gpu_start_s) * 1e6;
        {
            let (g, l) = (gpu_us / 1e6, (gpu_start_s - f.commit_s).max(0.));
            let mut est = VERIFY_EST.lock().unwrap();
            let e = &mut est[f.rows.min(MAX_ROWS)];
            *e = if e.0 > 0. { (0.8 * e.0 + 0.2 * g, 0.8 * e.1 + 0.2 * l) } else { (g, l) };
        }
        self.pending = f.rows;
        Ok(Timing { host_prepare_us: f.host_prepare_us, encode_us: f.encode_us, gpu_us, wait_us, commit_us: 0., barriers: f.barriers,
                    gpu_start_s, gpu_end_s, launch_us: (gpu_start_s - f.commit_s) * 1e6, event_s, notice_s })
    }
    /// R55 diagnostic: the same forward with one timestamped encoder per stage (embedding
    /// upload excluded): ple, each layer, mixer, head. Returns (label, GPU us) in order.
    pub fn forward_profiled(&mut self, tokens: &[u32]) -> Result<Vec<(String, f64)>> {
        ensure!(self.pending == 0, "uncommitted forward");
        let rows = tokens.len();
        ensure!((1..=MAX_ROWS).contains(&rows) && self.len + rows <= self.capacity, "forward rows/capacity");
        let idx: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let emb = RawEmbeddings::load(&self.pack, &idx)?;
        let mut x = Vec::with_capacity(rows * F);
        for r in 0..rows {
            for _ in 0..hc::HC {
                x.extend_from_slice(&emb.values[r * H..(r + 1) * H]);
            }
        }
        upload(&self.x0, &x)?;
        self.ple.prepare(tokens)?;
        let mut clock = crate::metal_profile::StageClock::new(&self.queue)?;
        let conc = self.concurrent;
        let mut input: &Buffer = &self.x0;
        for (i, l) in self.layers.iter().enumerate() {
            if i == 1 {
                clock.stage_with("ple", conc, |e| self.ple.encode(e, input, rows, &self.dense))?;
                input = self.ple.output();
            }
            match l {
                Layer::Gdn(g) => {
                    if std::env::var("QWEN_R58_PROFILE_FINE").as_deref() == Ok("1") {
                        g.encode_resident_profiled(&mut clock, input, rows, self.gdn_arm)?;
                    } else {
                        clock.stage_with(format!("gdn{i}"), conc, |e| g.encode_resident(e, input, rows, self.gdn_arm))?;
                    }
                    input = g.output();
                }
                Layer::Attn(a) => {
                    if std::env::var("QWEN_R58_PROFILE_FINE").as_deref() == Ok("1") {
                        a.encode_resident_profiled(&mut clock, input, self.len, rows, rows, self.attn_arm_at(self.len))?;
                    } else {
                        clock.stage_with(format!("attn{i}"), conc, |e| a.encode_resident(e, input, self.len, rows, rows, self.attn_arm_at(self.len)))?;
                    }
                    input = a.output();
                }
            }
        }
        clock.stage_with("mixer", conc, |e| self.hc_ops.read(e, &self.mixer, input, rows, Precision::PackedF16x2, None))?;
        // R56: one head arm for every row count, so a position's logits do not depend on the
        // page size (QWEN_R56_HEAD_QMV1=1 restores QMV at one row).
        let arm = if rows == 1 && std::env::var("QWEN_R56_HEAD_QMV1").as_deref() == Ok("1") { DenseArm::Qmv4 } else { DenseArm::FragmentPacked { splits: 1, residual: true } };
        clock.stage_with("head", conc, |e| {
            for (first, w) in &self.head {
                self.dense.encode(e, w, self.mixer.mixed(), &self.logits, &self.logits_bf, &self.head_sums, rows, first * rows, arm)?;
            }
            Ok(())
        })?;
        let stages = clock.finish()?;
        self.pending = rows;
        Ok(stages.into_iter().map(|s| (s.label, s.gpu_us + s.gap_us)).collect())
    }
    /// R57: the target's vocabulary head alone on given hidden rows (BF16, rows x H: e.g. DFlash2's
    /// drafted positions), with the forward's own head arm; FP32 logits per row, vocabulary order.
    /// Between forwards only (stages the rows in the mixer's output buffer, which every forward
    /// rewrites).
    pub fn head_logits(&mut self, hidden: &[u16]) -> Result<Vec<Vec<f32>>> {
        ensure!(!self.in_flight && self.pending == 0, "head logits between forwards only");
        let rows = hidden.len() / H;
        ensure!((1..=MAX_ROWS).contains(&rows) && hidden.len() == rows * H, "head rows");
        upload(self.mixer.mixed(), hidden)?;
        let cb = self.queue.commandBuffer().context("head command")?;
        let e = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).context("head encoder")?;
        let arm = DenseArm::FragmentPacked { splits: 1, residual: true };
        for (first, w) in &self.head {
            self.dense.encode(&e, w, self.mixer.mixed(), &self.logits, &self.logits_bf, &self.head_sums, rows, first * rows, arm)?;
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        ensure!(cb.status() == MTLCommandBufferStatus::Completed, "head command: {:?}", cb.error());
        self.pending = rows;
        let out = (0..rows).map(|r| self.logits(r)).collect::<Result<Vec<_>>>();
        self.pending = 0;
        out
    }
    /// R58: the top `k` (id, logit) of each of the pending forward's first `rows` rows, larger first,
    /// ties to the lower id, NaN as -inf. One thread per row. These are the lists a serial top-k
    /// scan of logits(r) gives, without copying each vocabulary row.
    pub fn top_rows(&self, rows: usize, k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        self.top_rows_by(rows, k, false)
    }
    /// top_rows ranking by BF16-rounded logits (ties to the lower id), returning the BF16 values:
    /// the DFlash2 selector's candidate rule (Candidates::from_logits on BF16 logits).
    fn top_rows_by(&self, rows: usize, k: usize, bf16_rank: bool) -> Result<Vec<Vec<(u32, f32)>>> {
        ensure!(rows >= 1 && rows <= self.pending && (1..=64).contains(&k) && !self.in_flight, "top rows");
        if let (Some(tk), Some((r, bf))) = (&self.topk, self.topk_ready) {
            if rows <= r && bf == bf16_rank && k <= TOPK_K {
                let gpu = tk.read(rows, k);
                if tk.check {
                    let host = self.top_rows_host(rows, k, bf16_rank)?;
                    ensure!(gpu == host, "GPU top-k differs from the host scan: {:?} vs {:?}", &gpu[0][..3.min(k)], &host[0][..3.min(k)]);
                }
                return Ok(gpu);
            }
        }
        self.top_rows_host(rows, k, bf16_rank)
    }
    fn top_rows_host(&self, rows: usize, k: usize, bf16_rank: bool) -> Result<Vec<Vec<(u32, f32)>>> {
        ensure!(rows >= 1 && rows <= self.pending && (1..=64).contains(&k) && !self.in_flight, "top rows");
        let n = self.pending;
        // SAFETY: shared storage; the pending forward has completed (not in flight).
        let all = unsafe { std::slice::from_raw_parts(self.logits.contents().cast::<f32>().as_ptr(), MAX_ROWS * VOCAB) };
        let mut chunks = vec![];
        let mut id0 = 0usize;
        for (first, w) in &self.head {
            chunks.push((*first, w.n, id0));
            id0 += w.n;
        }
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..rows)
                .map(|r| {
                    let chunks = &chunks;
                    sc.spawn(move || {
                        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
                        let better = |a: (u32, f32), b: (u32, f32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
                        for &(first, wn, id0) in chunks {
                            let base = first * n + r * wn;
                            for (j, &x) in all[base..base + wn].iter().enumerate() {
                                let v = if x.is_nan() { f32::NEG_INFINITY } else if bf16_rank { half::bf16::from_f32(x).to_f32() } else { x };
                                let c = ((id0 + j) as u32, v);
                                if top.len() == k && !better(c, top[k - 1]) {
                                    continue;
                                }
                                let at = top.iter().position(|&t| better(c, t)).unwrap_or(top.len());
                                top.insert(at, c);
                                top.truncate(k);
                            }
                        }
                        top
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().map_err(|_| anyhow::anyhow!("top rows thread"))).collect()
        })
    }
    /// R58 DFlash2: the target's head on given hidden rows (as head_logits), returning only each
    /// row's top `k` by BF16-rounded logit (ties to the lower id) with those BF16 values, read in
    /// place: no copy of the vocabulary rows.
    pub fn head_top(&mut self, hidden: &[u16], k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        ensure!(!self.in_flight && self.pending == 0, "head top between forwards only");
        let rows = hidden.len() / H;
        ensure!((1..=MAX_ROWS).contains(&rows) && hidden.len() == rows * H, "head rows");
        upload(self.mixer.mixed(), hidden)?;
        let cb = self.queue.commandBuffer().context("head command")?;
        let e = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).context("head encoder")?;
        let arm = DenseArm::FragmentPacked { splits: 1, residual: true };
        for (first, w) in &self.head {
            self.dense.encode(&e, w, self.mixer.mixed(), &self.logits, &self.logits_bf, &self.head_sums, rows, first * rows, arm)?;
        }
        self.topk_ready = None;
        if let Some(tk) = &self.topk {
            tk.encode(&e, &self.logits, rows, rows, true);
            self.topk_ready = Some((rows, true));
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        ensure!(cb.status() == MTLCommandBufferStatus::Completed, "head command: {:?}", cb.error());
        self.pending = rows;
        let out = self.top_rows_by(rows, k, true);
        self.pending = 0;
        out
    }
    /// FP32 logits of the pending forward's row `r` (vocabulary order).
    pub fn logits(&self, r: usize) -> Result<Vec<f32>> {
        ensure!(r < self.pending, "logits row");
        let rows = self.pending;
        let all = unsafe {
            std::slice::from_raw_parts(self.logits.contents().cast::<f32>().as_ptr(), MAX_ROWS * VOCAB)
        };
        let mut out = Vec::with_capacity(VOCAB);
        for (first, w) in &self.head {
            let base = first * rows + r * w.n;
            out.extend_from_slice(&all[base..base + w.n]);
        }
        ensure!(out.iter().all(|v| v.is_finite()), "nonfinite logits");
        Ok(out)
    }
    /// R58 diagnostic: the rows layer `layer` read and wrote in the last completed forward (BF16,
    /// rows x F; layer 0 reads the embedding rows, layer 1 the PLE stage's output).
    pub fn layer_io(&self, layer: usize, rows: usize) -> Result<(Vec<u16>, Vec<u16>)> {
        ensure!(!self.in_flight && layer < self.layers.len() && (1..=MAX_ROWS).contains(&rows) && rows <= self.pending.max(1), "layer io bounds");
        fn out(l: &Layer) -> &Buffer {
            match l {
                Layer::Gdn(g) => g.output(),
                Layer::Attn(a) => a.output(),
            }
        }
        let input = match layer {
            0 => &self.x0,
            1 => self.ple.output(),
            _ => out(&self.layers[layer - 1]),
        };
        Ok((read_rows(input, rows * F)?, read_rows(out(&self.layers[layer]), rows * F)?))
    }
    /// R58 diagnostic: a GDN layer's streams after the mixer's HC write (rows x F), its MLP input
    /// and its MLP output (rows x H) from the last completed forward.
    pub fn front_io(&self, layer: usize, rows: usize) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
        ensure!(!self.in_flight && (1..=MAX_ROWS).contains(&rows) && layer < self.layers.len(), "front io bounds");
        let (stream, mlp_in, mlp_out) = match &self.layers[layer] {
            Layer::Gdn(g) => {
                let (front, _) = g.prefill_parts();
                let (_, hc1, _, stream) = front.prefill_parts();
                (stream, hc1.mixed(), g.mlp_output())
            }
            Layer::Attn(a) => a.diag_buffers(),
        };
        Ok((read_rows(stream, rows * F)?, read_rows(mlp_in, rows * H)?, read_rows(mlp_out, rows * H)?))
    }
    /// R58 diagnostic: an attention layer's KV cache for positions 0..len, dequantized (K and V,
    /// each len x 2 x 256 FP32), its QSA pool (len/4 x 128 BF16) and raw index tail (len%4 x 128).
    pub fn attn_state(&self, layer: usize, len: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<u16>, Vec<u16>)> {
        ensure!(!self.in_flight, "state read with a forward in flight");
        let Some(Layer::Attn(a)) = self.layers.get(layer) else { anyhow::bail!("layer {layer} is not an attention layer") };
        let (_, _, att, _) = a.prefill_parts();
        let (_, index, core) = att.prefill_parts();
        let (_, _, tails, _) = index.prefill_parts();
        let (_, cache, cap) = core.prefill_parts();
        ensure!(len <= cap, "attention state length");
        let kv = crate::target_attention::KV;
        let deq = |c: &Buffer, s: &Buffer, b: &Buffer| -> Result<Vec<f32>> {
            ensure!(c.length() >= len * kv, "cache size");
            // SAFETY: shared storage, nothing in flight.
            let codes = unsafe { std::slice::from_raw_parts(c.contents().cast::<u8>().as_ptr(), len * kv) };
            let (sc, bi) = (read_rows(s, len * kv / 64)?, read_rows(b, len * kv / 64)?);
            Ok((0..len * kv).map(|i| {
                let (s, b) = (half::bf16::from_bits(sc[i / 64]).to_f32(), half::bf16::from_bits(bi[i / 64]).to_f32());
                half::bf16::from_f32(codes[i] as f32 * s + b).to_f32()
            }).collect())
        };
        let k = deq(&cache[0], &cache[1], &cache[2])?;
        let v = deq(&cache[3], &cache[4], &cache[5])?;
        Ok((k, v, read_rows(tails[0], len / 4 * 128)?, read_rows(tails[1], len % 4 * 128)?))
    }
    /// R58 seeding: an attention layer's committed KV cache in its raw packed form (affine8 codes,
    /// BF16 group-64 scales/biases, token-major) plus the QSA index pool and raw tail: the structs
    /// the drafter's attention imports.
    pub fn attn_state_raw(&self, layer: usize, len: usize) -> Result<(crate::target_attention::State, crate::qsa::State)> {
        self.attn_state_window(layer, 0, len)
    }
    /// attn_state_raw for the window [start, len) (start a multiple of 4; base = start).
    pub fn attn_state_window(&self, layer: usize, start: usize, len: usize) -> Result<(crate::target_attention::State, crate::qsa::State)> {
        ensure!(!self.in_flight, "state read with a forward in flight");
        let Some(Layer::Attn(a)) = self.layers.get(layer) else { anyhow::bail!("layer {layer} is not an attention layer") };
        let (_, _, att, _) = a.prefill_parts();
        let (_, index, core) = att.prefill_parts();
        let (_, _, tails, _) = index.prefill_parts();
        let (_, cache, cap) = core.prefill_parts();
        ensure!(len <= cap, "attention state length");
        ensure!(start % 4 == 0 && start <= len, "attention window start");
        let (kv, meta, w) = (crate::target_attention::KV, crate::target_attention::KV / 64, len - start);
        // only the window [start, len) is read: token rows from start, pooled groups from start / 4
        let packed = |c: &Buffer, s: &Buffer, b: &Buffer| -> Result<crate::target_attention::Packed> {
            ensure!(c.length() >= len * kv, "cache size");
            // SAFETY: shared storage, nothing in flight.
            let codes = unsafe { std::slice::from_raw_parts(c.contents().cast::<u8>().as_ptr().add(start * kv), w * kv) }.to_vec();
            Ok(crate::target_attention::Packed { codes, scales: read_rows_at(s, start * meta, w * meta)?, biases: read_rows_at(b, start * meta, w * meta)? })
        };
        let cache_state = crate::target_attention::State { base: start, len: w, k: packed(&cache[0], &cache[1], &cache[2])?, v: packed(&cache[3], &cache[4], &cache[5])? };
        let index_state = crate::qsa::State { base: start, len: w, pooled: read_rows_at(tails[0], start / 4 * 128, (len / 4 - start / 4) * 128)?, tail: read_rows(tails[1], len % 4 * 128)? };
        cache_state.validate()?;
        index_state.validate()?;
        Ok((cache_state, index_state))
    }
    /// R58 selector-guided seed: attention layer `layer`'s QSA selection for row `row` of the last
    /// forward (offset = the committed length before it, rows = its row count): the selected group
    /// ids (ascending, at most TOPK) and each one's indexer score.
    pub fn qsa_selection(&self, layer: usize, row: usize, offset: usize, rows: usize) -> Result<Vec<(usize, f32)>> {
        ensure!(!self.in_flight && row < rows && rows <= MAX_ROWS, "QSA selection read");
        let Some(Layer::Attn(a)) = self.layers.get(layer) else { anyhow::bail!("layer {layer} is not an attention layer") };
        let (_, _, att, _) = a.prefill_parts();
        let (_, index, _) = att.prefill_parts();
        let total = (offset + rows) / crate::qsa::RATIO;
        let count = ((offset + row + 1) / crate::qsa::RATIO).min(crate::qsa::TOPK);
        // SAFETY: shared storage; nothing in flight.
        let ids = unsafe { std::slice::from_raw_parts(index.selection_buffer().contents().cast::<u32>().as_ptr().add(row * crate::qsa::TOPK), count) }.to_vec();
        let scores = unsafe { std::slice::from_raw_parts(index.scores_buffer().contents().cast::<f32>().as_ptr().add(row * total), total) };
        ids.iter().map(|&g| {
            ensure!((g as usize) < total, "QSA selected group bound");
            Ok((g as usize, scores[g as usize]))
        }).collect()
    }
    /// R58 selector-guided seed: attention layer `layer`'s committed KV and QSA index for the given
    /// whole groups (ascending, all before `tail_start`) followed by the contiguous window
    /// [tail_start, len), packed contiguously with base = len - slots. Keys carry their RoPE, so the
    /// drafter attends to them at their true positions; tokens it appends continue at len.
    pub fn attn_state_groups(&self, layer: usize, groups: &[usize], tail_start: usize, len: usize) -> Result<(crate::target_attention::State, crate::qsa::State)> {
        ensure!(!self.in_flight, "state read with a forward in flight");
        ensure!(tail_start % 4 == 0 && tail_start <= len, "selector seed tail start");
        ensure!(groups.windows(2).all(|w| w[0] < w[1]) && groups.last().is_none_or(|&g| (g + 1) * 4 <= tail_start), "selector seed groups");
        let Some(Layer::Attn(a)) = self.layers.get(layer) else { anyhow::bail!("layer {layer} is not an attention layer") };
        let (_, _, att, _) = a.prefill_parts();
        let (_, index, core) = att.prefill_parts();
        let (_, _, tails, _) = index.prefill_parts();
        let (_, cache, cap) = core.prefill_parts();
        ensure!(len <= cap, "attention state length");
        let (kv, meta) = (crate::target_attention::KV, crate::target_attention::KV / 64);
        let slots = groups.len() * 4 + (len - tail_start);
        let base = len - slots;
        // token rows: each group's four rows, then the tail window
        let mut rows: Vec<(usize, usize)> = groups.iter().map(|&g| (g * 4, 4)).collect();
        rows.push((tail_start, len - tail_start));
        let packed = |c: &Buffer, s: &Buffer, b: &Buffer| -> Result<crate::target_attention::Packed> {
            ensure!(c.length() >= len * kv, "cache size");
            let (mut codes, mut scales, mut biases) = (Vec::with_capacity(slots * kv), vec![], vec![]);
            for &(at, n) in &rows {
                // SAFETY: shared storage, nothing in flight.
                codes.extend_from_slice(unsafe { std::slice::from_raw_parts(c.contents().cast::<u8>().as_ptr().add(at * kv), n * kv) });
                scales.extend(read_rows_at(s, at * meta, n * meta)?);
                biases.extend(read_rows_at(b, at * meta, n * meta)?);
            }
            Ok(crate::target_attention::Packed { codes, scales, biases })
        };
        let cache_state = crate::target_attention::State { base, len: slots, k: packed(&cache[0], &cache[1], &cache[2])?, v: packed(&cache[3], &cache[4], &cache[5])? };
        let mut pooled = vec![];
        for &g in groups {
            pooled.extend(read_rows_at(tails[0], g * 128, 128)?);
        }
        pooled.extend(read_rows_at(tails[0], tail_start / 4 * 128, (len / 4 - tail_start / 4) * 128)?);
        let index_state = crate::qsa::State { base, len: slots, pooled, tail: read_rows(tails[1], len % 4 * 128)? };
        cache_state.validate()?;
        index_state.validate()?;
        Ok((cache_state, index_state))
    }
    /// R58 seeding: the PLE stage's normalized history (HISTORY x F BF16) and its two previous tokens.
    pub fn ple_state(&self) -> Result<(Vec<u16>, [u32; 2])> {
        ensure!(!self.in_flight, "state read with a forward in flight");
        Ok((read_rows(self.ple.history_rows(), crate::ple::HISTORY * F)?, self.ple.previous_tokens()))
    }
    /// R58 bounded q epochs: copy every GDN layer's referenced zero-copy snapshot into its committed
    /// state (zero-copy commits stay selected), so the committed state can be read between verifies
    /// (gdn_state, seeding). The next forward reads the same state either way.
    pub fn materialize_committed(&self) -> Result<()> {
        ensure!(!self.in_flight && self.pending == 0, "materialize between forwards only");
        for l in &self.layers {
            if let Layer::Gdn(g) = l {
                g.prefill_parts().0.prefill_parts().2.prefill_ready();
            }
        }
        Ok(())
    }
    /// R58: a GDN layer's committed convolution history (BF16 3 x C) and recurrent state (FP32).
    pub fn gdn_state(&self, layer: usize) -> Result<(Vec<u16>, Vec<f32>)> {
        ensure!(!self.in_flight, "state read with a forward in flight");
        let Some(Layer::Gdn(g)) = self.layers.get(layer) else { anyhow::bail!("layer {layer} is not a GDN layer") };
        let (front, _) = g.prefill_parts();
        let (_, _, gdn, _) = front.prefill_parts();
        let (_, _, conv, state, slot) = gdn.prefill_parts();
        ensure!(slot.is_none(), "GDN state read with a zero-copy slot");
        let n = crate::gdn::STATE;
        ensure!(state.length() >= n * 4, "state size");
        // SAFETY: shared storage; nothing is in flight.
        let st = unsafe { std::slice::from_raw_parts(state.contents().cast::<f32>().as_ptr(), n).to_vec() };
        Ok((read_rows(conv, 3 * crate::gdn::C)?, st))
    }
    fn prefill_engine(&mut self, rows: usize) -> Result<&Prefill> {
        let need = rows.div_ceil(16) * 16;
        if self.prefill.as_ref().is_none_or(|p| p.max < need) {
            self.prefill = None;
            self.prefill = Some(Prefill::new(&self._device, need, self.capacity)?);
        }
        Ok(self.prefill.as_ref().unwrap())
    }
    /// R58 prefill check: layer `layer` alone over `input` (BF16 rows x F, positions from the
    /// committed length) through the prefill engine in chunks of up to `chunk` rows, continuing
    /// from and updating the layer's committed state (GDN state / KV cache and index). With
    /// `reference` MLP input rows (rows x H) the MLP also runs on those (diagnostic).
    pub fn prefill_layer(&mut self, layer: usize, input: &[u16], chunk: usize, reference: Option<&[u16]>) -> Result<PrefillLayer> {
        ensure!(!self.in_flight && self.pending == 0, "prefill between forwards only");
        let rows = input.len() / F;
        ensure!(rows >= 1 && input.len() == rows * F && chunk >= 1 && layer < self.layers.len(), "prefill input rows");
        if let Layer::Gdn(g) = &self.layers[layer] {
            g.prefill_parts().0.prefill_parts().2.prefill_ready();
        }
        self.prefill_engine(chunk.min(rows))?;
        let pf = self.prefill.as_ref().unwrap();
        let weights = match &self.layers[layer] {
            Layer::Gdn(g) => g.prefill_parts().1,
            Layer::Attn(a) => a.prefill_parts().3,
        };
        let base = self.len;
        ensure!(reference.is_none_or(|r| r.len() == rows * H), "reference MLP input rows");
        let mut run = PrefillLayer { output: vec![], stream: vec![], mlp_input: vec![], mlp_output: vec![], ref_mlp_output: vec![], route_flips: vec![],
                                     router_kernel_flips: 0, gpu_s: vec![], plan_s: vec![], engine_bytes: pf.bytes() };
        let scores = if matches!(self.layers[layer], Layer::Gdn(_)) { self.gdn_arm.scores } else { RoutingArithmetic::StagedBf16 };
        let submit = |encode: &dyn Fn(&objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<()>| -> Result<f64> {
            let cb = self.queue.commandBuffer().context("prefill command")?;
            let e = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).context("prefill encoder")?;
            encode(&e)?;
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            ensure!(cb.status() == MTLCommandBufferStatus::Completed, "prefill command: {:?}", cb.error());
            Ok(cb.GPUEndTime() - cb.GPUStartTime())
        };
        for start in (0..rows).step_by(chunk) {
            let t = chunk.min(rows - start);
            upload(&pf.io[0], &input[start * F..(start + t) * F])?;
            let front = submit(&|e| match &self.layers[layer] {
                Layer::Gdn(g) => pf.encode_gdn_front(&mut crate::metal_prefill::One(e), g, &pf.io[0], t, self.gdn_arm.scores),
                Layer::Attn(a) => pf.encode_attn_front(&mut crate::metal_prefill::One(e), a, &pf.io[0], t, base + start, RoutingArithmetic::StagedBf16),
            })?;
            let p = Instant::now();
            let blocks = pf.plan_routes(t)?;
            run.plan_s.push(p.elapsed().as_secs_f64());
            let (stream, mlp_input) = pf.front_outputs(t);
            run.stream.extend(stream);
            run.mlp_input.extend(mlp_input);
            let own = pf.routes(t);
            if crate::metal_prefill::router_mma() {
                submit(&|e| {
                    pf.route_with(e, weights, t, scores, false);
                    Ok(())
                })?;
                let qmv = pf.routes(t);
                for row in 0..t {
                    let (mut a, mut b) = (own[row * 10..row * 10 + 10].to_vec(), qmv[row * 10..row * 10 + 10].to_vec());
                    a.sort_unstable();
                    b.sort_unstable();
                    run.router_kernel_flips += (a != b) as usize;
                }
                // Restore the engine's own routes for the tail (same input, same kernel).
                submit(&|e| {
                    pf.route_with(e, weights, t, scores, true);
                    Ok(())
                })?;
                pf.plan_routes(t)?;
            }
            let tail = submit(&|e| pf.encode_mlp_tail(&mut crate::metal_prefill::One(e), weights, &pf.io[1], t, blocks))?;
            run.output.extend(Prefill::read_rows(&pf.io[1], t));
            run.mlp_output.extend(pf.mlp_outputs(t));
            run.gpu_s.push((front, tail));
            if let Some(r) = reference {
                pf.set_mlp_input(&r[start * H..(start + t) * H])?;
                submit(&|e| {
                    pf.route(e, weights, t, scores);
                    Ok(())
                })?;
                let theirs = pf.routes(t);
                for row in 0..t {
                    let (mut a, mut b) = (own[row * 10..row * 10 + 10].to_vec(), theirs[row * 10..row * 10 + 10].to_vec());
                    a.sort_unstable();
                    b.sort_unstable();
                    run.route_flips.push(a != b);
                }
                let blocks = pf.plan_routes(t)?;
                submit(&|e| pf.encode_mlp_tail(&mut crate::metal_prefill::One(e), weights, &pf.io[1], t, blocks))?;
                run.ref_mlp_output.extend(pf.mlp_outputs(t));
            }
        }
        Ok(run)
    }
    /// R58: prefill `tokens` at the current position through the large-row engine, in chunks of
    /// up to `chunk` rows (a short remainder joins the last chunk), as one committed forward:
    /// every layer's state advances (GDN state and convolution history, KV cache, QSA pool and raw
    /// tail, PLE history and hash) and the position moves by tokens.len(). The last token's
    /// logits come from the decode path's mixer and head on that row (read with prefill_logits).
    /// One command buffer per layer boundary; the host builds each layer's routing plan between.
    pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Result<PrefillTiming> {
        Ok(self.prefill_run(tokens, chunk, false, None)?.0)
    }
    /// R58 lead 1: layer `layer`'s routed expert ids and weights for the last forward's `rows` rows.
    pub fn moe_routes(&self, layer: usize, rows: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(!self.in_flight, "route read with a forward in flight");
        match self.layers.get(layer).context("layer index")? {
            Layer::Gdn(g) => g.moe_routes(rows),
            Layer::Attn(a) => a.moe_routes(rows),
        }
    }
    /// R58 selector-guided seed: (offset, rows) of the last decode forward (rows 0: none usable).
    pub fn last_forward(&self) -> (usize, usize) {
        self.last_forward
    }
    /// R58 DFlash2: the same prefill, calling `hook(encoder, layer, output, rows, first_row)` in
    /// each layer's command buffer once that layer's 4 HC streams for the chunk's rows are in
    /// `output` (rows x F, BF16). `first_row` is the chunk's first row within `tokens`. The hook
    /// may only read `output`, which the next layer overwrites.
    pub(crate) fn prefill_with(&mut self, tokens: &[u32], chunk: usize, hook: &mut TapHook) -> Result<PrefillTiming> {
        Ok(self.prefill_run(tokens, chunk, false, Some(hook))?.0)
    }
    /// R58 profile: the same prefill with one timestamped encoder per stage; also returns each
    /// stage's (label "kind.stage", GPU microseconds) in execution order. Slower than prefill.
    pub fn prefill_profiled(&mut self, tokens: &[u32], chunk: usize) -> Result<(PrefillTiming, Vec<(String, f64)>)> {
        self.prefill_run(tokens, chunk, true, None)
    }
    fn prefill_run(&mut self, tokens: &[u32], chunk: usize, profile: bool, hook: Option<&mut TapHook>) -> Result<(PrefillTiming, Vec<(String, f64)>)> {
        ensure!(!self.in_flight && self.pending == 0, "prefill between forwards only");
        // the decode engines' selections predate this prefill: not a seed source any more
        self.last_forward = (self.len, 0);
        if let (Some(set), false) = (&self.residency, self.residency_active) {
            set.requestResidency();
            self.queue.addResidencySet(set);
            self.residency_active = true;
        }
        self.prefill_inner(tokens, chunk, profile, hook)
    }
    /// R58: end the experts' residency (a later prefill requests it again).
    pub fn release_residency(&mut self) -> Result<()> {
        ensure!(!self.in_flight, "release residency with a forward in flight");
        if let (Some(set), true) = (&self.residency, self.residency_active) {
            self.queue.removeResidencySet(set);
            set.endResidency();
            self.residency_active = false;
        }
        Ok(())
    }
    fn prefill_inner(&mut self, tokens: &[u32], chunk: usize, profile: bool, mut hook: Option<&mut TapHook>) -> Result<(PrefillTiming, Vec<(String, f64)>)> {
        use crate::metal_prefill::{One, Profiled, Sink};
        let n = tokens.len();
        let hist = crate::ple::HISTORY;
        ensure!(n >= hist && self.len + n <= self.capacity && (16..=4080).contains(&chunk), "prefill rows/capacity/chunk");
        // R58: the indexer-score workspace is chunk x (capacity / 4) x 4 bytes under a 512 MiB
        // per-buffer bound; past 128K contexts the chunk shrinks so it fits (2,032 rows at 262K)
        let chunk = chunk.min((((512usize << 20) - (1 << 20)) / (self.capacity / crate::qsa::RATIO * 4) / 16 * 16).max(16));
        let wall = Instant::now();
        let mut bounds = vec![];
        let mut s = 0;
        while s < n {
            let mut t = chunk.min(n - s);
            if n - s - t < hist {
                t = n - s;
            }
            bounds.push((s, t));
            s += t;
        }
        let max_t = bounds.iter().map(|b| b.1).max().unwrap();
        for l in &self.layers {
            if let Layer::Gdn(g) = l {
                g.prefill_parts().0.prefill_parts().2.prefill_ready();
            }
        }
        self.prefill_logits = false;
        self.ple.prefill_prefetch(tokens)?;
        if self.embeddings.is_none() {
            let table = crate::embedding::EmbeddingTable::open(&self.pack)?;
            // Once: the table's rows are the decode path's (RawEmbeddings), bit for bit.
            let probe = &tokens[..tokens.len().min(64)];
            let idx: Vec<usize> = probe.iter().map(|&v| v as usize).collect();
            ensure!(table.rows(probe)? == RawEmbeddings::load(&self.pack, &idx)?.values, "prefill embedding rows differ from the decode path's");
            self.embeddings = Some(table);
        }
        self.prefill_engine(max_t)?;
        let pf = self.prefill.as_ref().unwrap();
        let mut timing = PrefillTiming { rows: n, chunks: bounds.iter().map(|b| b.1).collect(), engine_bytes: pf.bytes(), ..Default::default() };
        let mut stages: Vec<(String, f64)> = vec![];
        let head_arm = DenseArm::FragmentPacked { splits: 1, residual: true };
        CB_COSTS.with(|c| *c.borrow_mut() = [0.; 3]);
        // One command buffer: a single concurrent encoder, or (profile) a stage clock whose stages
        // are separate timestamped encoders. Returns the GPU seconds.
        enum Target<'a> {
            One(&'a objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>),
            Clock(&'a mut crate::metal_profile::StageClock),
        }
        impl Target<'_> {
            fn sink(&mut self, prefix: &'static str) -> Box<dyn Sink + '_> {
                match self {
                    Target::One(e) => Box::new(One(e)),
                    Target::Clock(c) => Box::new(Profiled { clock: c, prefix }),
                }
            }
        }
        let queue = &self.queue;
        let run = |stages: &mut Vec<(String, f64)>, body: &mut dyn FnMut(&mut Target) -> Result<()>| -> Result<f64> {
            if profile {
                let mut clock = crate::metal_profile::StageClock::with_capacity(queue, 64)?;
                body(&mut Target::Clock(&mut clock))?;
                let st = clock.finish()?;
                let g = st.iter().map(|s| s.gpu_us + s.gap_us).sum::<f64>() / 1e6;
                stages.extend(st.into_iter().map(|s| (s.label, s.gpu_us)));
                Ok(g)
            } else {
                submit_wait(queue, |e| body(&mut Target::One(e)))
            }
        };
        let kind = |l: &Layer| if matches!(l, Layer::Gdn(_)) { "gdn" } else { "attn" };
        for (ci, &(s, t)) in bounds.iter().enumerate() {
            let c = &tokens[s..s + t];
            let h = Instant::now();
            let (chunk_wall, mut spans) = (Instant::now(), vec![]);
            let emb = self.embeddings.as_ref().unwrap().rows(c)?;
            let mut x = Vec::with_capacity(t * F);
            for r in 0..t {
                for _ in 0..hc::HC {
                    x.extend_from_slice(&emb[r * H..(r + 1) * H]);
                }
            }
            upload(&pf.io[0], &x)?;
            let g = Instant::now();
            pf.set_ple_rows(&self.ple.prefill_embed(c)?)?;
            timing.ple_gather_s += g.elapsed().as_secs_f64();
            timing.host_rows_s += h.elapsed().as_secs_f64();
            let len = self.len;
            let layers = &self.layers;
            let gdn_scores = self.gdn_arm.scores;
            let front = |tg: &mut Target, l: usize, input: &Buffer| -> Result<()> {
                let mut sink = tg.sink(kind(&layers[l]));
                match &layers[l] {
                    Layer::Gdn(g) => pf.encode_gdn_front(&mut *sink, g, input, t, gdn_scores),
                    Layer::Attn(a) => pf.encode_attn_front(&mut *sink, a, input, t, len, RoutingArithmetic::StagedBf16),
                }
            };
            spans.push(run(&mut stages, &mut |tg| front(tg, 0, &pf.io[0]))?);
            timing.command_buffers += 1;
            let p = Instant::now();
            let mut blocks = pf.plan_routes(t)?;
            timing.plan_s += p.elapsed().as_secs_f64();
            let mut cur = 0usize;
            for l in 0..layers.len() {
                let weights = match &layers[l] {
                    Layer::Gdn(g) => g.prefill_parts().1,
                    Layer::Attn(a) => a.prefill_parts().3,
                };
                let out = 1 - cur;
                let last = l + 1 == layers.len();
                let final_chunk = ci + 1 == bounds.len();
                spans.push(run(&mut stages, &mut |tg| {
                    pf.encode_mlp_tail(&mut *tg.sink(kind(&layers[l])), weights, &pf.io[out], t, blocks)?;
                    if let Some(h) = hook.as_deref_mut() {
                        tg.sink("taps").stage("taps", &mut |e| h(e, l, &pf.io[out], t, s))?;
                    }
                    if l == 0 {
                        pf.encode_ple(&mut *tg.sink("ple"), &self.ple, &pf.io[out], t)?;
                    }
                    if !last {
                        front(tg, l + 1, if l == 0 { &pf.ple_out } else { &pf.io[out] })?;
                    } else if final_chunk {
                        tg.sink("head").stage("mixer_head", &mut |e| {
                            pf.copy_words(e, &pf.io[out], (t - 1) * F * 2, &self.x0, F / 2);
                            crate::metal_ops::barrier(e);
                            self.hc_ops.read(e, &self.mixer, &self.x0, 1, Precision::PackedF16x2, None)?;
                            for (first, w) in &self.head {
                                self.dense.encode(e, w, self.mixer.mixed(), &self.logits, &self.logits_bf, &self.head_sums, 1, *first, head_arm)?;
                            }
                            Ok(())
                        })?;
                    }
                    Ok(())
                })?);
                timing.command_buffers += 1;
                cur = out;
                if !last {
                    let p = Instant::now();
                    blocks = pf.plan_routes(t)?;
                    timing.plan_s += p.elapsed().as_secs_f64();
                }
            }
            self.ple.prefill_advance(c)?;
            self.len += t;
            let g: f64 = spans.iter().sum();
            timing.gpu_s += g;
            timing.chunk_gpu_s.push(g);
            timing.chunk_wall_s.push(chunk_wall.elapsed().as_secs_f64());
            timing.command_gpu_s.push(spans);
        }
        self.prefill_logits = true;
        timing.wall_s = wall.elapsed().as_secs_f64();
        let [enc, launch, done] = CB_COSTS.with(|c| *c.borrow());
        (timing.encode_s, timing.launch_s, timing.complete_s) = (enc, launch, done);
        Ok((timing, stages))
    }
    /// R58: FP32 logits of the last prefilled token (vocabulary order), until the next forward.
    pub fn prefill_logits(&self) -> Result<Vec<f32>> {
        ensure!(self.prefill_logits && !self.in_flight && self.pending == 0, "no prefill logits");
        // SAFETY: shared storage; the prefill's command buffers have completed.
        let all = unsafe { std::slice::from_raw_parts(self.logits.contents().cast::<f32>().as_ptr(), MAX_ROWS * VOCAB) };
        let mut out = Vec::with_capacity(VOCAB);
        for (first, w) in &self.head {
            out.extend_from_slice(&all[*first..*first + w.n]);
        }
        ensure!(out.iter().all(|v| v.is_finite()), "nonfinite logits");
        Ok(out)
    }
    /// Keep the first `accepted` rows of the pending forward (0..=rows).
    pub fn commit(&mut self, accepted: usize) -> Result<f64> {
        let rows = self.pending;
        ensure!(rows > 0 && accepted <= rows, "commit");
        let t = Instant::now();
        // R56: the GDN layers' 3 MiB state snapshot copies (~108 MiB per commit) run on several
        // threads; the convolution histories and attention tails stay on this thread.
        let mut jobs = vec![];
        for l in &self.layers {
            match l {
                Layer::Gdn(g) => jobs.extend(g.commit_split(accepted, rows)?),
                Layer::Attn(a) => a.commit(self.len, accepted)?,
            }
        }
        let per = jobs.len().div_ceil(COMMIT_THREADS).max(1);
        std::thread::scope(|s| {
            for chunk in jobs.chunks(per) {
                s.spawn(move || {
                    for &(src, dst, n) in chunk {
                        // SAFETY: distinct per-layer shared-memory buffers of n bytes; the
                        // forward that wrote the source has completed.
                        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n) };
                    }
                });
            }
        });
        self.ple.commit(accepted)?;
        self.len += accepted;
        self.pending = 0;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
}
/// R58: BF16 rows from a shared-storage buffer the GPU has finished writing.
/// R58 diagnostic (verify-time probe only): skip every buffer barrier while on. Results are wrong;
/// it bounds what the dependency stalls cost a forward.
pub fn diagnostic_skip_barriers(on: bool) {
    crate::metal_ops::SKIP_BARRIERS.store(on, std::sync::atomic::Ordering::Relaxed);
}
/// R58 diagnostic (verify-time probe only): skip the barriers at the call sites matching
/// QWEN_R58_SKIP_SITES while on; returns how many were skipped so far.
pub fn diagnostic_skip_sites(on: bool) -> usize {
    crate::metal_ops::SKIP_SELECTED.store(on, std::sync::atomic::Ordering::Relaxed);
    crate::metal_ops::SKIPPED.load(std::sync::atomic::Ordering::Relaxed)
}
/// R58 diagnostic: the call-site patterns diagnostic_skip_sites matches ("file:line" substrings).
pub fn diagnostic_skip_patterns(pats: &[String]) {
    *crate::metal_ops::SKIP_PATTERNS.lock().unwrap() = pats.to_vec();
}
fn read_rows(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    read_rows_at(b, 0, n)
}
fn read_rows_at(b: &Buffer, offset: usize, n: usize) -> Result<Vec<u16>> {
    ensure!(b.length() >= (offset + n) * 2, "buffer rows");
    // SAFETY: shared storage of at least offset + n u16; no forward is in flight (callers check).
    Ok(unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr().add(offset), n).to_vec() })
}
/// R58: commit one command buffer (concurrent dispatch, explicit barriers) and wait; its GPU span
/// in seconds. Host-side costs accumulate in CB_COSTS (encode, launch, completion seconds).
fn submit_wait(
    queue: &objc2::runtime::ProtocolObject<dyn MTLCommandQueue>,
    encode: impl FnOnce(&objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<()>,
) -> Result<f64> {
    let t0 = host_time_s();
    let cb = queue.commandBuffer().context("prefill command")?;
    let e = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).context("prefill encoder")?;
    encode(&e)?;
    e.endEncoding();
    let t1 = host_time_s();
    cb.commit();
    cb.waitUntilCompleted();
    let t2 = host_time_s();
    ensure!(cb.status() == MTLCommandBufferStatus::Completed, "prefill command: {:?}", cb.error());
    let (s, f) = (cb.GPUStartTime(), cb.GPUEndTime());
    CB_COSTS.with(|c| {
        let mut c = c.borrow_mut();
        c[0] += t1 - t0;
        c[1] += (s - t1).max(0.);
        c[2] += (t2 - f).max(0.);
    });
    Ok(f - s)
}
thread_local! {
    static CB_COSTS: std::cell::RefCell<[f64; 3]> = const { std::cell::RefCell::new([0.; 3]) };
}
