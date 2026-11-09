//! Diagnostic resident model assembly. No host activation handoffs, expanded
//! expert weights, sampling, or performance-ready serving loop. Single rows
//! only: the first gate intentionally reserves one recurrent checkpoint.
use crate::{
    cuda::{Arg, Buffer, Device, Kernel, Module, PreparedCall},
    cuda_lowbit_attention, cuda_lowbit_attention_layer as al,
    cuda_lowbit_io::{self, Io, Kind},
    cuda_lowbit_layer as rl, cuda_ple, cuda_ple_layer as pl, gdn, hc, lowbit_attention_layer,
    lowbit_io, lowbit_layer,
    mcg_moe::Bank,
    ple, qsa, target_attention,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};

pub struct Modules<'d> {
    body: rl::Modules<'d>,
    attention: Module<'d>,
    ple: Module<'d>,
    io: Module<'d>,
    /// R53 calibration kernels (null twins, clock probe); loaded only when asked.
    cal: Option<Module<'d>>,
    /// R56: macuda's device-to-device byte copy (drafter checkpoints).
    copy: Module<'d>,
    /// R56: GPU top-20 of the head's logits (kernels/r56_ops.cu, research/R56-OPS-CUDA-BUILD.json).
    ops: Module<'d>,
}
impl<'d> Modules<'d> {
    pub fn load(d: &'d Device) -> Result<Self> {
        Ok(Self {
            body: rl::Modules::load(d)?,
            attention: cuda_lowbit_attention::module(d)?,
            ple: cuda_ple::module(d)?,
            io: cuda_lowbit_io::module(d)?,
            cal: if crate::r53_cal::module_needed() { Some(crate::r53_cal::module(d)?) } else { None },
            copy: d.module(
                Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/macuda/copy1d.sm_120.cubin")),
                "03a644f4a55d702f0aa2bf969fa0e716fcf89e2d948f42f0bbcddc7e0ede4c58",
            )?,
            ops: d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/r56_ops.sm120.cubin"), R56_OPS_SHA)?,
        })
    }
}
/// R56: the early-launch driver mode this process selected (model-probe's translation of
/// QWEN_R38_PDL_MODE / QWEN_R53_CHAIN_DEPTH into the project-local driver's own settings).
static PDL_DRIVER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static ZERO_COPY_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Before Device::open, single-threaded: select the project-local driver's early-launch mode
/// (TINYNV_R38_PDL) and launch-chain depth (TINYNV_CHAIN_DEPTH) from the R53 best environment,
/// as model-probe does. Without this the r39pdl_ kernels run fully serialized (~5.2 ms forward
/// instead of ~3.9 ms). Returns the mode. Only graph replays are qualified with early launch;
/// the queued paths (run, step_lean) refuse to run while a mode is selected.
pub fn select_driver_settings() -> Result<Option<u32>> {
    ensure!(std::env::var("QWEN_R53_NOBAR").is_err(), "QWEN_R53_NOBAR faulted the model (r53-nobar-l4)");
    for q in ["QWEN_R53_TRIG", "QWEN_R53_DOWNTRIG", "QWEN_R53_HCF4", "QWEN_R53_RING_ASYNC", "QWEN_R53_RING_PIPELINE", "QWEN_R53_SKIP_DONE_ACQUIRE", "QWEN_R53_SMEM_FLOOR_KB"] {
        ensure!(std::env::var(q).is_err(), "{q} is a model-probe-only setting");
    }
    let archive = option_env!("QWEN_TINYNV_ARCHIVE").unwrap_or("");
    let mut mode = None;
    if let Ok(m) = std::env::var("QWEN_R38_PDL_MODE") {
        let n: u32 = m.parse()?;
        ensure!(archive.ends_with("driver-r38/build/r53nv/libtinynv.a") && n == 37, "QWEN_R38_PDL_MODE=37 with the r53nv archive only");
        // SAFETY: called single-threaded before the driver is opened (it reads this once).
        unsafe { std::env::set_var("TINYNV_R38_PDL", &m) };
        PDL_DRIVER.store(n, std::sync::atomic::Ordering::Relaxed);
        mode = Some(n);
    }
    if let Ok(depth) = std::env::var("QWEN_R53_CHAIN_DEPTH") {
        let n: u32 = depth.parse()?;
        ensure!(!archive.is_empty() && (1..=1024).contains(&n), "QWEN_R53_CHAIN_DEPTH needs the project-local archive and 1..=1024");
        // SAFETY: as above.
        unsafe { std::env::set_var("TINYNV_CHAIN_DEPTH", n.to_string()) };
    }
    Ok(mode)
}
fn queued_allowed() -> Result<()> {
    ensure!(!ZERO_COPY_ON.load(std::sync::atomic::Ordering::Relaxed), "zero-copy drafting runs graph steps only");
    ensure!(
        PDL_DRIVER.load(std::sync::atomic::Ordering::Relaxed) == 0,
        "queued submissions are not qualified with the early-launch driver mode; use the graph step"
    );
    Ok(())
}
const R56_OPS_SHA: &str = "af3efba49e41f2f9218726d9937c9015cb84b9cade4434f5bd393d9b7f0dc1fd";
/// R56 GPU top-k: the sampler's candidate count, the partial pass's chunk (256 threads x 8)
/// and the rows it serves (the logits buffer's rows).
pub const TOPK: usize = 20;
const TOPK_CHUNK: usize = 2048;
const TOPK_BLOCKS: usize = lowbit_io::VOCAB.div_ceil(TOPK_CHUNK);
const TOPK_ROWS: usize = crate::hc::MAX_ROWS;
/// R56 zero-copy drafting: copy tables committing every GDN layer's convolution history.
struct ZeroCopy<'d> {
    gather: Buffer<'d>,
    gather_n: usize,
    restore: Buffer<'d>,
    restore_n: usize,
}
struct TopK<'m, 'd> {
    partial: Kernel<'m, 'd>,
    last: Kernel<'m, 'd>,
    cand_v: Buffer<'d>,
    cand_i: Buffer<'d>,
    out: Buffer<'d>,
}
enum Layer<'m, 'd> {
    Recurrent(rl::Layer<'m, 'd>),
    Ple(pl::Layer<'m, 'd>),
    Attention(al::Layer<'m, 'd>),
}
impl<'m, 'd> Layer<'m, 'd> {
    fn output(&self) -> &Buffer<'d> {
        match self {
            Self::Recurrent(a) => a.output(),
            Self::Ple(a) => a.output(),
            Self::Attention(a) => a.output(),
        }
    }
    fn commit(&self, n: usize) -> Result<()> {
        match self {
            Self::Recurrent(a) => a.commit(n),
            Self::Ple(a) => a.commit(n),
            Self::Attention(a) => a.commit(n),
        }
    }
}
fn recurrent_policy() -> rl::Policy {
    rl::Policy {
        hc_split: true,
        separate_write: true,
        gdn_fused: true,
        all_states: false,
        wide: false,
        expert: rl::Expert::Mma,
    }
}
fn attention_policy() -> al::Policy {
    al::Policy {
        hc_split: true,
        separate_write: true,
        attention_parts: 16,
        wide: false,
        expert: rl::Expert::Mma,
    }
}
fn ple_policy() -> pl::Policy {
    pl::Policy {
        body: recurrent_policy(),
        ple_fused: true,
    }
}
/// R58 batched prefill policies (QWEN_R58_DRAFTER_PREFILL): "decode" (default) = the drafting
/// policies; "unfused" = GDN prepare over all rows in parallel, then the recurrence (same
/// arithmetic as the fused kernel); "wide" = unfused plus the 4-rows-per-block dense projections,
/// width-4 shared expert and tensor-core router. At 16 rows all three ran at the same speed
/// (r58-drafter-prefill-03), so the default keeps the drafting kernels.
fn prefill_policies() -> (rl::Policy, al::Policy, pl::Policy) {
    let mode = std::env::var("QWEN_R58_DRAFTER_PREFILL").unwrap_or_else(|_| "decode".into());
    let (unfused, wide) = match mode.as_str() {
        "decode" => (false, false),
        "unfused" => (true, false),
        _ => (true, true),
    };
    // R58 (prepared for a morning screen, not run overnight): QWEN_R58_DRAFTER_EXPERT=hybrid puts
    // the multi-row forwards (catch-up, replay, branch rows) on the R22-qualified grouped arm
    // (mcg_group512 + decode-once MMA); the 1-row graph step keeps direct MMA.
    let expert = match std::env::var("QWEN_R58_DRAFTER_EXPERT").as_deref() {
        Ok("hybrid") => rl::Expert::Hybrid,
        _ => rl::Expert::Mma,
    };
    let r = rl::Policy { gdn_fused: !unfused, wide, expert, ..recurrent_policy() };
    let a = al::Policy { wide, expert, ..attention_policy() };
    (r, a, pl::Policy { body: r, ..ple_policy() })
}

pub struct Model<'m, 'd> {
    d: &'d Device,
    io: Io<'m, 'd>,
    layers: Vec<Layer<'m, 'd>>,
    hash: ple::Hash,
    table: ple::Table,
    previous: Cell<[u32; 2]>,
    position: Cell<usize>,
    /// R58 seeding: the first absolute position held in the KV caches (0 unless a windowed seed
    /// was imported); occupancy = position - kv_base is what the capacity bounds.
    kv_base: Cell<usize>,
    ready: Cell<bool>,
    pending: Cell<Option<u32>>,
    /// R58 asynchronous drafting: a launched graph step not yet finished (token, start, patch us,
    /// recaptured, patched).
    inflight: Cell<Option<(u32, Instant, f64, bool, usize)>>,
    checked: Cell<bool>,
    capacity: usize,
    /// R53: [null, r39pdl_ null, clock probe] when the calibration module is loaded.
    cal: Option<[Kernel<'m, 'd>; 3]>,
    topk: TopK<'m, 'd>,
    copy_table: Kernel<'m, 'd>,
    zero_copy: std::cell::OnceCell<ZeroCopy<'d>>,
    /// R58 batched prefill, zero-copy: per row count, the copy table writing every GDN layer's
    /// convolution history (table, entries).
    conv_tables: std::cell::RefCell<Vec<(usize, Buffer<'d>, usize)>>,
}

/// R58 S3 (Model::set_skip_mlp): q's MoE MLP blocks to skip, bit i = layer i, read whenever layer
/// calls are recorded. A skipped layer keeps its GDN or attention mixer and its HC stages; its MLP
/// output stays zero, so the HC write adds nothing, and the g4w2h write still clears that MoE's idle
/// counter block. Acceptance stays exact (drafts are accepted under the skipped q's own law).
pub static SKIP_MLP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn skip_mlp(layer: usize) -> bool {
    layer < 64 && (SKIP_MLP.load(std::sync::atomic::Ordering::Relaxed) >> layer) & 1 == 1
}

/// R58: one batched drafter prefill (see Model::prefill).
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct PrefillStats {
    pub rows: usize,
    pub chunks: Vec<usize>,
    pub seconds: f64,
    /// Host: n-gram gather and staging; recording the launches; enqueueing; waiting.
    pub prepare_s: f64,
    pub record_s: f64,
    pub enqueue_s: f64,
    pub sync_s: f64,
    pub launches: usize,
    /// QWEN_R58_DRAFTER_PROFILE=1: the first chunk's stage groups each run as their own graph
    /// (host time of launch + sync, microseconds): (layer.stage, us).
    pub profile: Vec<(String, f64)>,
}
/// R56 captured drafting plan plus where the position-dependent launches sit in it.
#[cfg(feature = "r53-driver")]
pub struct DraftGraph<'s, 'm, 'd> {
    graph: crate::cuda::Graph<'s, 'm, 'd>,
    /// (layer, first launch, launches) of each attention layer's forward and commit calls.
    fwd: Vec<(usize, usize, usize)>,
    commits: Vec<(usize, usize, usize)>,
    /// The plan ends its forward with the GPU top-k (two launches before the commits).
    topk: bool,
}
/// R56 drafter page checkpoint: device copies of every committed state buffer plus the host
/// metadata (attention lengths, PLE tokens, model position).
/// R58 seeding: a drafter state outside the device (export_state / import_state).
#[derive(Clone)]
pub enum LayerState {
    Recurrent { conv: Vec<u16>, state: Vec<f32> },
    Ple { conv: Vec<u16>, state: Vec<f32>, history: Vec<u16>, previous: [u32; 2] },
    Attention { cache: crate::target_attention::State, index: crate::qsa::State },
}
#[derive(Clone)]
pub struct DrafterState {
    pub position: usize,
    pub previous: [u32; 2],
    pub layers: Vec<LayerState>,
}
impl DrafterState {
    /// Bytes of the useful payload (for transfer accounting).
    pub fn payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Recurrent { conv, state } => conv.len() * 2 + state.len() * 4,
                LayerState::Ple { conv, state, history, .. } => conv.len() * 2 + state.len() * 4 + history.len() * 2,
                LayerState::Attention { cache, index } => {
                    cache.k.codes.len() + cache.v.codes.len() + (cache.k.scales.len() + cache.k.biases.len() + cache.v.scales.len() + cache.v.biases.len()) * 2
                        + (index.pooled.len() + index.tail.len()) * 2
                }
            })
            .sum()
    }
}
pub struct Checkpoint<'d> {
    bufs: Vec<Buffer<'d>>,
    positions: Vec<Option<(usize, usize)>>,
    ple_previous: [u32; 2],
    previous: [u32; 2],
    position: usize,
    saved: bool,
    /// R58: {dst, src, bytes} entries (live state -> this checkpoint) for one copy-table launch.
    table: Option<(Buffer<'d>, usize)>,
    /// R58: the reverse entries (this checkpoint -> live state), for checkpoint_restore_fast.
    rtable: Option<(Buffer<'d>, usize)>,
}
pub struct Run {
    pub io: cuda_lowbit_io::Run,
    pub outputs: Vec<Vec<u16>>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
    pub collection_us: f64,
    pub calls: usize,
    pub max_queued_calls: usize,
    pub layer_us: Vec<f64>,
    pub execution: &'static str,
    pub graph_capture_and_first_execution_us: f64,
}
impl<'m, 'd> Model<'m, 'd> {
    /// Four layers are a composition control, not a meaningful model. Forty-eight
    /// layers are the real backbone. Both retain the same embedding and head.
    pub fn load(
        d: &'d Device,
        m: &'m Modules<'d>,
        root: &Path,
        count: usize,
        capacity: usize,
        mut observe: impl FnMut(Value) -> Result<()>,
    ) -> Result<Self> {
        ensure!(
            // R59: up to the attention component's qualified bound (32,768 + 16; QSA selection past
            // 2,051), so seeded q can hold the full context (agentic session A: +19%, r59-agentS-A-full-31)
            [4, 48].contains(&count) && (16..=crate::cuda_lowbit_attention::MAX_CAPACITY).contains(&capacity),
            "model load bounds"
        );
        let before = d.memory_info()?;
        // R58: the model's small long-lived buffers come out of shared arena chunks, saving the
        // driver's per-allocation slack and page rounding (~11 MiB per layer; QWEN_R58_ARENA=0 off)
        let _arena = d.arena_scope();
        let emb = lowbit_io::Vocab::load(root, true)?;
        let head = lowbit_io::Vocab::load(root, false)?;
        let mix = lowbit_io::Mixer::load(root)?;
        let io = Io::new(d, &m.body.affine, &m.body.hc, &m.io, &emb, &head, &mix)?;
        observe(
            json!({"kind":"model_load_io", "description":io.description(), "allocator":d.memory_info()?}),
        )?;
        drop((emb, head, mix));
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let start = Instant::now();
            let bank =
                Bank::full_packed(root, i).with_context(|| format!("layer {i} expert bank"))?;
            let (layer, provenance) = if i % 4 == 3 {
                let w = lowbit_attention_layer::Weights::load(root, i)?;
                let a = al::Layer::new(d, &m.body, &m.attention, &w, &bank, capacity)?;
                (Layer::Attention(a), w.provenance())
            } else if i == 1 {
                let w = lowbit_layer::Weights::load_ple_body(root)?;
                let pw = ple::Weights::load(root)?;
                let a = pl::Layer::new_with_prefix_capacity(d, &m.body, &m.ple, &w, &pw, &bank, 1)?;
                (
                    Layer::Ple(a),
                    json!({"body":w.provenance(),"ple":pw.provenance}),
                )
            } else {
                let w = lowbit_layer::Weights::load(root, i)?;
                let a = rl::Layer::new_with_prefix_capacity(d, &m.body, &w, &bank, 1)?;
                (Layer::Recurrent(a), w.provenance())
            };
            layers.push(layer);
            // R56: this layer's weights are in device memory now (see the sweep below)
            let lf = root.join(format!("layer-{i}.safetensors"));
            if std::env::var("QWEN_R56_EVICT_WEIGHTS").as_deref() != Ok("0") && lf.is_file() {
                crate::io_pool::evict_cached(&lf)?;
            }
            observe(
                json!({"kind":"model_load_layer", "layer":i, "seconds":start.elapsed().as_secs_f64(),
                "provenance":provenance, "experts":bank.provenance(), "allocator":d.memory_info()?}),
            )?;
        }
        // R56: the weights now live in device memory; drop their files' cached pages so they do
        // not crowd the n-gram tables and the Metal target's mapped experts (on by default;
        // QWEN_R56_EVICT_WEIGHTS=0 keeps them). Only this pack's .safetensors files.
        if std::env::var("QWEN_R56_EVICT_WEIGHTS").as_deref() != Ok("0") {
            let (mut files, mut bytes) = (0usize, 0u64);
            for e in std::fs::read_dir(root)? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "safetensors") {
                    bytes += crate::io_pool::evict_cached(&p)?;
                    files += 1;
                }
            }
            observe(json!({"kind":"drafter_weights_evicted","files":files,"bytes":bytes}))?;
        }
        // R52 multi4 pump: each layer's MoE warms the next GDN layer's multi4 weights
        // (opt-in QWEN_R52_MPUMP=1; used only by the candidate A/B arm).
        if std::env::var("QWEN_R52_MPUMP").as_deref() == Ok("1") {
            for i in 0..layers.len().saturating_sub(1) {
                let next = match &layers[i + 1] {
                    Layer::Recurrent(a) => Some(a.first_projection_regions()),
                    Layer::Ple(a) => Some(a.first_projection_regions()),
                    Layer::Attention(_) => None,
                };
                if let Some(regions) = next {
                    match &layers[i] {
                        Layer::Recurrent(a) => a.set_mlp_pump(&regions)?,
                        Layer::Ple(a) => a.set_mlp_pump(&regions)?,
                        Layer::Attention(a) => a.set_mlp_pump(&regions)?,
                    }
                }
            }
        }
        observe(
            json!({"kind":"model_loaded", "layers":count, "capacity":capacity,
            "before":before, "allocator":d.memory_info()?, "arena":d.arena_stats(), "all_weights_simultaneously_resident":true}),
        )?;
        Ok(Self {
            d,
            io,
            layers,
            hash: ple::Hash::load(root)?,
            table: ple::Table::open(root)?,
            previous: Cell::new([ple::EOS; 2]),
            position: Cell::new(0),
            kv_base: Cell::new(0),
            ready: Cell::new(false),
            pending: Cell::new(None),
            inflight: Cell::new(None),
            checked: Cell::new(false),
            capacity,
            cal: match &m.cal {
                Some(c) => Some([c.kernel("r53_null")?, c.kernel("r39pdl_r53_null")?, c.kernel("r53_clock")?]),
                None => None,
            },
            topk: TopK {
                partial: m.ops.kernel("r56_topk_partial")?,
                last: m.ops.kernel("r56_topk_final")?,
                cand_v: d.alloc(TOPK_ROWS * TOPK_BLOCKS * TOPK * 4)?,
                cand_i: d.alloc(TOPK_ROWS * TOPK_BLOCKS * TOPK * 4)?,
                out: d.alloc(TOPK_ROWS * 2 * TOPK * 4)?,
            },
            copy_table: m.ops.kernel("r56_copy_table")?,
            zero_copy: std::cell::OnceCell::new(),
            conv_tables: std::cell::RefCell::new(vec![]),
        })
    }
    pub fn reset(&self) -> Result<()> {
        self.ready.set(false);
        self.checked.set(false);
        self.pending.set(None);
        let initial = gdn::Initial {
            conv: vec![0; 3 * gdn::C],
            state: vec![0.; gdn::STATE],
        };
        let conv = vec![0; ple::HISTORY * hc::F];
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.prepare_initial(&initial)?,
                Layer::Ple(a) => a.prepare_state([ple::EOS; 2], &conv, &initial)?,
                Layer::Attention(a) => {
                    a.prepare_state(&target_attention::State::empty(0)?, &qsa::State::empty(0)?)?
                }
            }
        }
        self.previous.set([ple::EOS; 2]);
        self.position.set(0);
        self.kv_base.set(0);
        self.ready.set(true);
        Ok(())
    }
    pub fn prepare(&self, token: u32) -> Result<()> {
        self.checked.set(false);
        self.pending.set(None);
        ensure!(
            self.ready.get() && token < ple::VOCAB && self.position.get() - self.kv_base.get() < self.capacity,
            "model state/token/context bounds"
        );
        // Any partial preparation failure requires explicit reset. Never keep a
        // candidate whose layer states have only partially advanced/prepared.
        self.ready.set(false);
        let packed = self
            .table
            .gather(&self.hash.rows(self.previous.get(), &[token])?)?;
        // R56: when only graph steps may run (early-launch mode or zero-copy), the staging
        // buffers are uploaded once, without the queued paths' poison + readback canaries
        // (2 transfers per step instead of 7). QWEN_R56_LEAN_PREPARE=0 keeps the full checks.
        let lean = queued_allowed().is_err() && std::env::var("QWEN_R56_LEAN_PREPARE").as_deref() != Ok("0");
        if lean {
            self.io.prepare_lean(&[token])?;
        } else {
            self.io.prepare(&[token], None)?;
        }
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.prepare_resident(1)?,
                Layer::Ple(a) if lean => a.prepare_resident_lean(&[token], &packed)?,
                Layer::Ple(a) => a.prepare_resident(&[token], &packed)?,
                Layer::Attention(a) => a.prepare_resident(1)?,
            }
        }
        self.pending.set(Some(token));
        self.ready.set(true);
        Ok(())
    }
    fn record(&self) -> Result<Vec<Vec<PreparedCall<'_, 'm, 'd>>>> {
        let mut groups = vec![vec![self.io.record_start()?]];
        let mut input = self.io.stream();
        for layer in &self.layers {
            let calls = match layer {
                Layer::Recurrent(a) => a.record_from(input, 1, recurrent_policy())?,
                Layer::Ple(a) => a.record_from(input, ple_policy())?,
                Layer::Attention(a) => a.record_from(input, 1, attention_policy())?,
            };
            groups.push(calls.into_iter().flatten().collect());
            input = layer.output();
        }
        groups.push(self.io.record_tail(input, 1, 0)?);
        Ok(groups)
    }
    pub fn run(&self, wait_layer: bool) -> Result<Run> {
        self.run_submission(wait_layer, 128, 1)
    }
    /// Bounded scheduling isolation on the already prepared candidate. Repeated
    /// forwards never commit state and are allowed only for the four-layer gate.
    pub fn run_submission(
        &self,
        wait_layer: bool,
        max_queued_calls: usize,
        repetitions: usize,
    ) -> Result<Run> {
        self.run_execution(wait_layer, max_queued_calls, repetitions, false)
    }
    /// Capture this exact prepared candidate, execute once during capture, then
    /// time one replay. Position-dependent arguments are frozen in this graph;
    /// this is not a reusable multi-token generation plan.
    pub fn run_graph_diagnostic(&self) -> Result<Run> {
        self.run_execution(false, 2048, 1, true)
    }
    fn run_execution(
        &self,
        wait_layer: bool,
        max_queued_calls: usize,
        repetitions: usize,
        use_graph: bool,
    ) -> Result<Run> {
        if !use_graph {
            queued_allowed()?;
        }
        ensure!(
            (1..=4096).contains(&max_queued_calls)
                && (1..=12).contains(&repetitions)
                && (repetitions == 1 || self.layers.len() == 4),
            "submission diagnostic bound"
        );
        ensure!(
            self.ready.get() && self.pending.get().is_some(),
            "model forward needs prepared token"
        );
        self.ready.set(false);
        self.checked.set(false);
        self.io.poison()?;
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.poison(recurrent_policy())?,
                Layer::Ple(a) => a.poison(ple_policy())?,
                Layer::Attention(a) => a.poison(attention_policy())?,
            }
        }
        let mut groups = self.record()?;
        // R53 warm-rerun calibration (candidate arm of whole-graph replays only).
        if use_graph && crate::cuda_affine4::PDL.load(std::sync::atomic::Ordering::Relaxed) {
            let names = crate::r53_cal::dup_names();
            if !names.is_empty() {
                let max = crate::r53_cal::dup_max_blocks();
                groups = groups
                    .into_iter()
                    .map(|g| {
                        let mut out = Vec::with_capacity(g.len() * 2);
                        for c in g {
                            let (n, grid, _) = c.meta();
                            let twin = (names.iter().any(|m| m == n) && grid[0] * grid[1] * grid[2] <= max)
                                .then(|| c.dup());
                            out.push(c);
                            if let Some(t) = twin {
                                out.push(t);
                            }
                        }
                        out
                    })
                    .collect();
            }
        }
        let calls = groups.iter().map(Vec::len).sum::<usize>() * repetitions;
        // R53 calibration (opt-in, whole-graph replays only): labels, null twins
        // with identical geometry/edges, and a driver stamp region.
        let r53 = use_graph && crate::r53_cal::active();
        let arm = crate::cuda_affine4::PDL.load(std::sync::atomic::Ordering::Relaxed);
        let mut nulls: Vec<(&'static str, Vec<PreparedCall<'_, 'm, 'd>>)> = vec![];
        if r53 {
            let meta: Vec<_> = groups
                .iter()
                .flatten()
                .map(|c| {
                    let (n, g, b) = c.meta();
                    (n.to_string(), g, b)
                })
                .collect();
            crate::r53_cal::report_meta(arm, &meta);
            if let Some([plain, pdl, _]) = &self.cal {
                for v in crate::r53_cal::null_variants() {
                    let twin = if v == "serial" { plain } else { pdl };
                    nulls.push((
                        v,
                        groups.iter().flatten().map(|c| c.null_twin(plain, twin)).collect::<Result<_>>()?,
                    ));
                }
            }
        }
        let stamp = if r53 && crate::r53_cal::stamp_enabled() {
            let s = self.d.alloc(crate::r53_cal::stamp_bytes(calls))?;
            s.upload(&vec![0u8; crate::r53_cal::stamp_bytes(calls)])?;
            Some(s)
        } else {
            None
        };
        self.d.sync()?;
        let capture = Instant::now();
        let graph = if use_graph {
            #[cfg(feature = "r53-driver")]
            if let Some(s) = &stamp {
                self.d.r53_stamp(Some(s), calls as u32)?;
            }
            let g = self.d.capture_model_diagnostic(
                std::mem::take(&mut groups).into_iter().flatten().collect(),
            );
            #[cfg(feature = "r53-driver")]
            if stamp.is_some() {
                self.d.r53_stamp(None, 0)?;
            }
            Some(g?)
        } else {
            None
        };
        self.d.sync()?;
        let capture_us = if use_graph {
            capture.elapsed().as_secs_f64() * 1e6
        } else {
            0.
        };
        // R39 opt-in timing discriminator. Same frozen candidate, no commit;
        // avoid letting long diagnostic downloads determine the GPU's clock
        // state for a single sub-millisecond sample. Original default is kept.
        // R52: prepared-queue profiling submissions ignore the graph replay count.
        let replay_samples: usize = if use_graph { std::env::var("QWEN_R39_REPLAY_SAMPLES")
            .unwrap_or_else(|_| "1".into()).parse()? } else { 1 };
        ensure!((1..=16).contains(&replay_samples) && (replay_samples==1 || use_graph),
            "bounded graph replay samples");
        if r53 && crate::r53_cal::clock_enabled() {
            if let Some([_, _, clock]) = &self.cal {
                crate::r53_cal::clock(self.d, clock, "before_warmup", arm)?;
            }
        }
        if replay_samples>1 {
            for _ in 0..2 { graph.as_ref().unwrap().launch()?; self.d.sync()?; }
        }
        if let Some(s) = &stamp {
            s.upload(&vec![0u8; crate::r53_cal::stamp_bytes(calls)])?;
        }
        let mut stamps_real = vec![];
        let start = Instant::now();
        let (mut enqueue_us, mut wait_us) = (0., 0.);
        let mut replay_times = vec![];
        let mut layer_us = vec![];
        let mut pending = 0;
        // R53: QWEN_R53_MOETRACE_ALL=1 keeps the traced layer's MoE block timeline from
        // every replay (downloaded between replays, outside the timed interval).
        let trace_layer = std::env::var("QWEN_R53_MOETRACE").ok().and_then(|v| v.parse::<usize>().ok());
        let trace_all = r53 && trace_layer.is_some() && std::env::var("QWEN_R53_MOETRACE_ALL").as_deref() == Ok("1");
        let mut traces: Vec<Vec<u64>> = vec![];
        if let Some(graph) = &graph {
            if replay_samples==1 {
                graph.launch()?;
                enqueue_us = start.elapsed().as_secs_f64() * 1e6;
            } else {
                for _ in 0..replay_samples {
                    let t=Instant::now(); graph.launch()?;
                    enqueue_us+=t.elapsed().as_secs_f64()*1e6;
                    let w=Instant::now(); self.d.sync()?;
                    wait_us+=w.elapsed().as_secs_f64()*1e6;
                    replay_times.push(t.elapsed().as_secs_f64()*1e6);
                    // R53: outside the timed interval; clears the region again.
                    if let Some(s) = &stamp {
                        stamps_real.push(crate::r53_cal::read_stamps(s, calls)?);
                    }
                    if trace_all {
                        let tr = match self.layers.get(trace_layer.unwrap()) {
                            Some(Layer::Recurrent(a)) => a.moe_trace()?,
                            Some(Layer::Attention(a)) => a.moe_trace()?,
                            _ => None,
                        };
                        if let Some((_, words)) = tr { traces.push(words); }
                    }
                }
            }
        }
        if r53 && replay_samples > 1 {
            crate::r53_cal::report("model", arm, calls, &stamps_real, &replay_times)?;
            // R53: one layer's MoE block trace from the last replay, with that replay's
            // per-call stamps (same GPU clock) for alignment (QWEN_R53_MOETRACE=layer).
            if let Some(li) = std::env::var("QWEN_R53_MOETRACE").ok().and_then(|v| v.parse::<usize>().ok()) {
                let tr = match self.layers.get(li) {
                    Some(Layer::Recurrent(a)) => a.moe_trace()?,
                    Some(Layer::Attention(a)) => a.moe_trace()?,
                    _ => None,
                };
                let emit = |k: usize, words: &[u64], st: &[[u64; 2]]| {
                    let rows = |base: usize, n: usize| -> Vec<&[u64]> { (0..n).map(|q| &words[base + q * 6..base + q * 6 + 6]).collect() };
                    println!("{}", json!({"kind":"r53_moetrace","layer":li,"candidate":arm,"replay":k,
                        "back":rows(0, 960),"f0":rows(16320, 768),"select":&words[16320 + 768 * 6..16320 + 768 * 6 + 4],
                        "down":rows(21120, 400),"stamps":st}));
                };
                if trace_all && traces.len() == stamps_real.len() {
                    for (k, (w, st)) in traces.iter().zip(&stamps_real).enumerate() { emit(k, w, st); }
                } else if let (Some((_, words)), Some(last)) = (tr, stamps_real.last()) {
                    emit(stamps_real.len() - 1, &words, last);
                }
            }
            // R53: per-layer MoE spin-timeout words (sticky; 0 everywhere = no timeout).
            let mut statuses = vec![];
            for l in &self.layers {
                let s = match l { Layer::Recurrent(a) => a.moe_status()?, Layer::Attention(a) => a.moe_status()?, _ => None };
                if let Some(s) = s { statuses.push(s); }
            }
            if !statuses.is_empty() {
                let timeouts = statuses.iter().filter(|&&s| s != 0).count();
                println!("{}", json!({"kind":"r53_moe_status","candidate":arm,"timeouts":timeouts,"status":statuses}));
                // A timed-out spin means a hand-off never completed, so this run's values are not
                // valid (research/R53-G4W2H-SAFETY.txt). Fail loudly instead of reporting timings.
                ensure!(timeouts == 0, "MoE spin timeout in {timeouts} layer(s), status words {statuses:?}");
            }
            if crate::r53_cal::clock_enabled() {
                if let Some([_, _, clock]) = &self.cal {
                    crate::r53_cal::clock(self.d, clock, "after_replays", arm)?;
                }
            }
            // Null twins read and write nothing, so the real replay's outputs stay
            // in place for the collection below.
            for (v, list) in std::mem::take(&mut nulls) {
                #[cfg(feature = "r53-driver")]
                if let Some(s) = &stamp {
                    self.d.r53_stamp(Some(s), calls as u32)?;
                }
                let g = self.d.capture_model_diagnostic(list);
                #[cfg(feature = "r53-driver")]
                if stamp.is_some() {
                    self.d.r53_stamp(None, 0)?;
                }
                let g = g?;
                for _ in 0..2 {
                    g.launch()?;
                    self.d.sync()?;
                }
                if let Some(s) = &stamp {
                    s.upload(&vec![0u8; crate::r53_cal::stamp_bytes(calls)])?;
                }
                let (mut times, mut st) = (vec![], vec![]);
                for _ in 0..replay_samples {
                    let t = Instant::now();
                    g.launch()?;
                    self.d.sync()?;
                    times.push(t.elapsed().as_secs_f64() * 1e6);
                    if let Some(s) = &stamp {
                        st.push(crate::r53_cal::read_stamps(s, calls)?);
                    }
                }
                crate::r53_cal::report(&format!("null-{v}"), arm, calls, &st, &times)?;
            }
        }
        for group in (0..repetitions).flat_map(|_| groups.iter()) {
            let phase = Instant::now();
            for call in group {
                let enqueue = Instant::now();
                call.enqueue()?;
                enqueue_us += enqueue.elapsed().as_secs_f64() * 1e6;
                pending += 1;
                // First full-model unbounded submission faulted in the pinned
                // driver. Keep explicit bounded controls alongside the large
                // submission arm; no change to the shared driver configuration.
                if pending == max_queued_calls {
                    let wait = Instant::now();
                    self.d.sync()?;
                    wait_us += wait.elapsed().as_secs_f64() * 1e6;
                    pending = 0;
                }
            }
            if wait_layer {
                let wait = Instant::now();
                self.d.sync()?;
                wait_us += wait.elapsed().as_secs_f64() * 1e6;
                layer_us.push(phase.elapsed().as_secs_f64() * 1e6);
                pending = 0;
            }
        }
        let wait = Instant::now();
        self.d.sync()?;
        wait_us += wait.elapsed().as_secs_f64() * 1e6;
        let mut total_us = start.elapsed().as_secs_f64() * 1e6;
        if replay_samples>1 {
            let mut sorted=replay_times.clone();sorted.sort_by(f64::total_cmp);
            total_us=(sorted[(sorted.len()-1)/2]+sorted[sorted.len()/2])/2.;
            enqueue_us/=replay_samples as f64;wait_us/=replay_samples as f64;
            println!("{}",json!({"kind":"r39_replay_group","samples":replay_samples,
                "warmups":2,"times_us":replay_times,"median_us":total_us,
                "scope":"same frozen candidate; final replay output collected; no generation throughput"}));
        }
        let collect = Instant::now();
        let mut outputs = vec![];
        let mut input = self.io.stream();
        for (i, layer) in self.layers.iter().enumerate() {
            let output = match layer {
                Layer::Recurrent(a) => {
                    a.collect_from(input, 1, recurrent_policy())
                        .with_context(|| format!("collect layer {i}"))?
                        .output
                }
                Layer::Ple(a) => {
                    a.collect(ple_policy())
                        .context("collect PLE layer")?
                        .body
                        .output
                }
                Layer::Attention(a) => {
                    a.collect_from(input, 1, attention_policy())
                        .with_context(|| format!("collect layer {i}"))?
                        .output
                }
            };
            outputs.push(output);
            input = layer.output();
        }
        let io = self.io.collect(Kind::Roundtrip, 1)?;
        self.checked.set(true);
        self.ready.set(true);
        Ok(Run {
            io,
            outputs,
            total_us,
            enqueue_us,
            wait_us,
            calls,
            max_queued_calls,
            collection_us: collect.elapsed().as_secs_f64() * 1e6,
            layer_us,
            execution: if use_graph {
                "whole_graph_replay"
            } else {
                "prepared_queue"
            },
            graph_capture_and_first_execution_us: capture_us,
        })
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        queued_allowed()?;
        ensure!(
            self.ready.get() && self.checked.get() && accepted <= 1,
            "model unchecked/invalid commit"
        );
        let token = self.pending.get().context("missing model candidate")?;
        self.ready.set(false);
        self.checked.set(false);
        self.pending.set(None);
        for layer in &self.layers {
            layer.commit(accepted)?;
        }
        if accepted == 1 {
            self.previous
                .set(ple::accepted_tokens(self.previous.get(), &[token], 1)?);
            self.position.set(self.position.get() + 1);
        }
        self.ready.set(true);
        Ok(())
    }
    pub fn position(&self) -> usize {
        self.position.get()
    }
    /// R58 seeding: KV occupancy (position minus the first position the KV window holds).
    /// R58 S3: skip the MLP of every layer whose bit is set in `mask` (zeroing their outputs
    /// first). Recorded graphs must be re-recorded after a change.
    pub fn set_skip_mlp(&self, mask: u64) -> Result<()> {
        for (i, l) in self.layers.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                match l {
                    Layer::Recurrent(a) => a.zero_mlp_output()?,
                    Layer::Ple(a) => a.zero_mlp_output()?,
                    Layer::Attention(a) => a.zero_mlp_output()?,
                }
            }
        }
        SKIP_MLP.store(mask, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    /// R58 sibling screen: every layer's routed expert ids from the last forward ([layer]
    /// [row * TOPK + k]); host reads of the routers' id buffers, no launch.
    pub fn route_ids(&self, rows: usize) -> Result<Vec<Vec<u32>>> {
        self.layers
            .iter()
            .map(|l| match l {
                Layer::Recurrent(a) => a.route_ids(rows),
                Layer::Ple(a) => a.route_ids(rows),
                Layer::Attention(a) => a.route_ids(rows),
            })
            .collect()
    }
    pub fn occupancy(&self) -> usize {
        self.position.get() - self.kv_base.get()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// R58 seeding: the committed drafter state, per layer (GDN convolution history and recurrent
    /// state from the live buffers, PLE history and its previous tokens, attention KV cache, index
    /// pool and raw tail), plus the model's position and previous tokens. No forward may be
    /// pending. import_state restores it into this or a reset drafter.
    pub fn export_state(&self) -> Result<DrafterState> {
        ensure!(self.ready.get() && self.pending.get().is_none(), "export needs a committed state");
        let down = |b: &Buffer<'d>, n: usize| -> Result<Vec<u8>> {
            let mut v = vec![0u8; b.byte_len()];
            b.download(&mut v)?;
            ensure!(v.len() >= n, "state buffer size");
            v.truncate(n);
            Ok(v)
        };
        let u16s = |v: Vec<u8>| -> Vec<u16> { v.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect() };
        let f32s = |v: Vec<u8>| -> Vec<f32> { v.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect() };
        let mut layers = vec![];
        for l in &self.layers {
            layers.push(match l {
                Layer::Recurrent(a) => {
                    let b = a.state_buffers();
                    ensure!(b.len() == 2, "recurrent state layout");
                    LayerState::Recurrent { conv: u16s(down(b[0], 3 * gdn::C * 2)?), state: f32s(down(b[1], gdn::STATE * 4)?) }
                }
                Layer::Ple(a) => {
                    let b = a.state_buffers();
                    ensure!(b.len() == 3, "PLE state layout");
                    LayerState::Ple {
                        conv: u16s(down(b[0], 3 * gdn::C * 2)?),
                        state: f32s(down(b[1], gdn::STATE * 4)?),
                        history: u16s(down(b[2], ple::HISTORY * hc::F * 2)?),
                        previous: a.previous_tokens(),
                    }
                }
                Layer::Attention(a) => {
                    let (cache, index) = a.export_state()?;
                    LayerState::Attention { cache, index }
                }
            });
        }
        Ok(DrafterState { position: self.position.get(), previous: self.previous.get(), layers })
    }
    /// R58 seeding: load an exported (or translated) state, as reset() loads the empty one. Any
    /// failure leaves the drafter invalid until a reset or a successful import.
    pub fn import_state(&self, st: &DrafterState) -> Result<()> {
        self.import_state_with(st, false)
    }
    /// import_state with lean GDN uploads (no canaries; zero-copy: live slot only): for trusted
    /// seeds on the hot path.
    pub fn import_state_lean(&self, st: &DrafterState) -> Result<()> {
        self.import_state_with(st, true)
    }
    /// R59 incremental refresh: the GDN/conv/PLE states are replaced in full (lean), but each
    /// attention layer only takes the rows from its exported `cache.base` on (the target's exact
    /// KV for the tokens since q's last seed); older rows and q's window base stay. Requires a
    /// held window whose base is <= every exported base, at a group boundary.
    pub fn import_state_incremental(&self, st: &DrafterState) -> Result<()> {
        ensure!(st.layers.len() == self.layers.len() && self.pending.get().is_none(), "incremental import layout");
        let kv_base = self.kv_base.get();
        ensure!(st.position >= kv_base && st.position - kv_base <= self.capacity, "incremental import position");
        for s in &st.layers {
            if let LayerState::Attention { cache, .. } = s {
                ensure!(cache.base >= kv_base && (cache.base - kv_base) % 4 == 0 && cache.base + cache.len == st.position, "incremental import rows");
            }
        }
        self.ready.set(false);
        self.checked.set(false);
        self.pending.set(None);
        for (l, s) in self.layers.iter().zip(&st.layers) {
            match (l, s) {
                (Layer::Recurrent(a), LayerState::Recurrent { conv, state }) => {
                    a.prepare_initial_lean(&gdn::Initial { conv: conv.clone(), state: state.clone() })?
                }
                (Layer::Ple(a), LayerState::Ple { conv, state, history, previous }) => {
                    a.prepare_state_lean(*previous, history, &gdn::Initial { conv: conv.clone(), state: state.clone() })?
                }
                (Layer::Attention(a), LayerState::Attention { cache, index }) => a.refresh_state(cache, index)?,
                _ => anyhow::bail!("import layer kind mismatch"),
            }
        }
        self.previous.set(st.previous);
        self.position.set(st.position);
        self.ready.set(true);
        Ok(())
    }
    /// R59: the window base q's attention KV currently starts at (absolute position).
    pub fn kv_base(&self) -> usize {
        self.kv_base.get()
    }
    fn import_state_with(&self, st: &DrafterState, lean: bool) -> Result<()> {
        ensure!(st.layers.len() == self.layers.len() && self.pending.get().is_none(), "import layout");
        self.ready.set(false);
        self.checked.set(false);
        self.pending.set(None);
        for (l, s) in self.layers.iter().zip(&st.layers) {
            match (l, s) {
                (Layer::Recurrent(a), LayerState::Recurrent { conv, state }) => {
                    let init = gdn::Initial { conv: conv.clone(), state: state.clone() };
                    if lean { a.prepare_initial_lean(&init)? } else { a.prepare_initial(&init)? }
                }
                (Layer::Ple(a), LayerState::Ple { conv, state, history, previous }) => {
                    let init = gdn::Initial { conv: conv.clone(), state: state.clone() };
                    if lean { a.prepare_state_lean(*previous, history, &init)? } else { a.prepare_state(*previous, history, &init)? }
                }
                (Layer::Attention(a), LayerState::Attention { cache, index }) => a.prepare_state(cache, index)?,
                _ => anyhow::bail!("import layer kind mismatch"),
            }
        }
        let bases: Vec<usize> = st.layers.iter().filter_map(|l| match l { LayerState::Attention { cache, .. } => Some(cache.base), _ => None }).collect();
        let kv_base = bases.first().copied().unwrap_or(0);
        ensure!(bases.iter().all(|&b| b == kv_base) && st.position >= kv_base && st.position - kv_base <= self.capacity, "import window/position");
        self.kv_base.set(kv_base);
        self.previous.set(st.previous);
        self.position.set(st.position);
        self.ready.set(true);
        Ok(())
    }
    /// R58 diagnostic: every committed state buffer (the checkpoint set) with a label and its
    /// element type: per GDN layer the convolution history (bf16) and state (f32), the PLE
    /// history (bf16), each attention layer's raw index tail (bf16). No forward may be pending.
    pub fn state_snapshot(&self) -> Result<Vec<(String, &'static str, Vec<u8>)>> {
        ensure!(self.ready.get() && self.pending.get().is_none(), "state snapshot with a forward pending");
        let mut out = vec![];
        for (i, l) in self.layers.iter().enumerate() {
            let (names, bufs): (&[(&str, &'static str)], Vec<&Buffer<'d>>) = match l {
                Layer::Recurrent(a) => (&[("conv", "bf16"), ("state", "f32")], a.state_buffers()),
                Layer::Ple(a) => (&[("conv", "bf16"), ("state", "f32"), ("ple_history", "bf16")], a.state_buffers()),
                Layer::Attention(a) => (&[("index_tail", "bf16")], a.state_buffers()),
            };
            ensure!(names.len() == bufs.len(), "state layout of layer {i}");
            for ((name, ty), b) in names.iter().zip(bufs) {
                let mut v = vec![0u8; b.byte_len()];
                b.download(&mut v)?;
                out.push((format!("{i}.{name}"), *ty, v));
            }
        }
        Ok(out)
    }
    /// R58 batched prefill: commit `tokens` (at least 3) in forwards of up to 16 rows (the
    /// layers' row limit) without the head, every layer's committed state advancing over all of
    /// them (GDN state and convolution history, KV cache, index pool and raw tail, PLE history and
    /// hash). A short remainder is avoided by shortening the second-to-last chunk. Graph and
    /// queued drafting both continue from it.
    pub fn prefill(&self, tokens: &[u32]) -> Result<PrefillStats> {
        let n = tokens.len();
        ensure!(n >= 3, "batched prefill needs at least 3 tokens");
        let t = Instant::now();
        let mut st = PrefillStats { rows: n, ..Default::default() };
        let mut s = 0;
        while s < n {
            let mut rows = hc::MAX_ROWS.min(n - s);
            let rest = n - s - rows;
            if rest > 0 && rest < 3 {
                rows -= 3 - rest;
            }
            self.prefill_rows(&tokens[s..s + rows], &mut st)?;
            st.chunks.push(rows);
            s += rows;
        }
        st.seconds = t.elapsed().as_secs_f64();
        Ok(st)
    }
    fn ensure_conv_table(&self, rows: usize) -> Result<()> {
        if self.conv_tables.borrow().iter().any(|e| e.0 == rows) {
            return Ok(());
        }
        let mut entries = vec![];
        for l in &self.layers {
            let g = match l {
                Layer::Recurrent(a) => a.gdn_ref(),
                Layer::Ple(a) => a.body_gdn(),
                Layer::Attention(_) => continue,
            };
            entries.push(g.conv_copy_rows(rows)?);
        }
        let raw: Vec<u8> = entries.iter().flat_map(|&(d, s, n)| [d, s, n]).flat_map(|v| v.to_le_bytes()).collect();
        let b = self.d.alloc(raw.len())?;
        b.upload(&raw)?;
        let n = entries.len();
        self.conv_tables.borrow_mut().push((rows, b, n));
        Ok(())
    }
    /// One batched forward of 3..=16 rows with every layer's commit, enqueued in order.
    fn prefill_rows(&self, tokens: &[u32], st: &mut PrefillStats) -> Result<()> {
        let rows = tokens.len();
        ensure!(
            (3..=hc::MAX_ROWS).contains(&rows)
                && self.ready.get()
                && self.pending.get().is_none()
                && self.position.get() - self.kv_base.get() + rows <= self.capacity
                && tokens.iter().all(|&t| t < ple::VOCAB),
            "drafter prefill rows/state/capacity/token"
        );
        let t0 = Instant::now();
        // As in prepare(): a failure part-way requires an explicit reset.
        self.ready.set(false);
        let packed = self.table.gather(&self.hash.rows(self.previous.get(), tokens)?)?;
        let lean = queued_allowed().is_err() && std::env::var("QWEN_R56_LEAN_PREPARE").as_deref() != Ok("0");
        if lean {
            self.io.prepare_lean(tokens)?;
        } else {
            self.io.prepare(tokens, None)?;
        }
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.prepare_resident(rows)?,
                Layer::Ple(a) if lean => a.prepare_resident_lean(tokens, &packed)?,
                Layer::Ple(a) => a.prepare_resident(tokens, &packed)?,
                Layer::Attention(a) => a.prepare_resident(rows)?,
            }
        }
        let t1 = Instant::now();
        let (rp, ap, pp) = prefill_policies();
        // R58 (opt-in, QWEN_R58_DRAFTER_ROWS=1): the dense GDN/attention projections with the
        // multi-row MMA kernel while recording this chunk. ~8% faster prefill; not yet screened
        // in isolation, and one 1.5K-token hybrid run drafted slightly worse (r58-hybrid-prefill-05).
        let _rows_arm = (std::env::var("QWEN_R58_DRAFTER_ROWS").as_deref() == Ok("1")).then(crate::cuda_affine4::RowsArm::set);
        let zc = self.zero_copy.get().is_some();
        if zc {
            self.ensure_conv_table(rows)?;
        }
        let tables = self.conv_tables.borrow();
        let table = if zc { tables.iter().find(|e| e.0 == rows).map(|e| (&e.1, e.2)) } else { None };
        let mut calls = vec![self.io.record_start()?];
        let mut input = self.io.stream();
        for layer in &self.layers {
            let c = match layer {
                Layer::Recurrent(a) => a.record_from(input, rows, rp)?,
                Layer::Ple(a) => a.record_from(input, pp)?,
                Layer::Attention(a) => a.record_from(input, rows, ap)?,
            };
            calls.extend(c.into_iter().flatten());
            input = layer.output();
        }
        if let Some((b, n)) = table {
            // Zero-copy: GDN states were updated in place by the cores; conv histories here.
            calls.push(unsafe { self.copy_table.prepare(vec![Arg::Buffer(b)], [8, n as u32, 1], [256, 1, 1])? });
        }
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) if !zc => calls.extend(a.record_commit(rows, rp)?),
                Layer::Recurrent(_) => {}
                Layer::Ple(a) if zc => calls.extend(a.record_commit_ple(rows)?),
                Layer::Ple(a) => calls.extend(a.record_commit(rows, pp)?),
                Layer::Attention(a) => calls.extend(a.record_commit(rows)?),
            }
        }
        if st.profile.is_empty() && std::env::var("QWEN_R58_DRAFTER_PROFILE").as_deref() == Ok("1") {
            // Diagnostic: the same launches, one graph per stage group, each timed to completion.
            let mut groups: Vec<(String, Vec<PreparedCall<'_, 'm, 'd>>)> = vec![];
            let mut it = calls.into_iter();
            groups.push(("io".into(), vec![it.next().unwrap()]));
            for (i, layer) in self.layers.iter().enumerate() {
                let sizes: Vec<usize> = match layer {
                    Layer::Recurrent(a) => a.record_from(if i == 0 { self.io.stream() } else { self.layers[i - 1].output() }, rows, rp)?.iter().map(|g| g.len()).collect(),
                    Layer::Ple(a) => a.record_from(self.layers[i - 1].output(), pp)?.iter().map(|g| g.len()).collect(),
                    Layer::Attention(a) => a.record_from(self.layers[i - 1].output(), rows, ap)?.iter().map(|g| g.len()).collect(),
                };
                for (j, n) in sizes.into_iter().enumerate() {
                    let g: Vec<PreparedCall<'_, 'm, 'd>> = it.by_ref().take(n).collect();
                    // QWEN_R58_DRAFTER_PROFILE_CALLS=LAYER: that layer's calls one by one.
                    if std::env::var("QWEN_R58_DRAFTER_PROFILE_CALLS").ok().and_then(|v| v.parse::<usize>().ok()) == Some(i) {
                        for (k, c) in g.into_iter().enumerate() {
                            let (name, grid, block) = c.meta();
                            let name = name.to_string();
                            groups.push((format!("{i}.{j}.{k}:{name}:{grid:?}x{block:?}"), vec![c]));
                        }
                    } else {
                        groups.push((format!("{i}.{j}"), g));
                    }
                }
            }
            groups.push(("commit".into(), it.collect()));
            for (label, g) in groups {
                if g.is_empty() {
                    continue;
                }
                let s = Instant::now();
                if queued_allowed().is_ok() {
                    for c in &g {
                        c.enqueue()?;
                    }
                    self.d.sync()?;
                } else {
                    let graph = self.d.capture_model_diagnostic(g)?;
                    self.d.sync()?;
                    drop(graph);
                }
                st.profile.push((label, s.elapsed().as_secs_f64() * 1e6));
            }
            drop(tables);
            for layer in &self.layers {
                match layer {
                    Layer::Recurrent(a) => a.finish_commit(),
                    Layer::Ple(a) => a.finish_commit(rows)?,
                    Layer::Attention(a) => a.finish_commit(rows)?,
                }
            }
            for piece in tokens.chunks(hc::MAX_ROWS) {
                self.previous.set(ple::accepted_tokens(self.previous.get(), piece, piece.len())?);
            }
            self.position.set(self.position.get() + rows);
            self.ready.set(true);
            return Ok(());
        }
        let t2 = Instant::now();
        let launches = calls.len();
        // Queued launches where allowed; under the early-launch driver mode or zero-copy only
        // graph replays are qualified, so the chunk runs as a one-shot captured graph (capture
        // executes it).
        let graph = if queued_allowed().is_ok() {
            for c in &calls {
                c.enqueue()?;
            }
            None
        } else {
            Some(self.d.capture_model_diagnostic(calls)?)
        };
        let t3 = Instant::now();
        self.d.sync()?;
        let t4 = Instant::now();
        drop(graph);
        drop(tables);
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.finish_commit(),
                Layer::Ple(a) => a.finish_commit(rows)?,
                Layer::Attention(a) => a.finish_commit(rows)?,
            }
        }
        for piece in tokens.chunks(hc::MAX_ROWS) {
            self.previous.set(ple::accepted_tokens(self.previous.get(), piece, piece.len())?);
        }
        self.position.set(self.position.get() + rows);
        self.ready.set(true);
        st.prepare_s += (t1 - t0).as_secs_f64();
        st.record_s += (t2 - t1).as_secs_f64();
        st.enqueue_s += (t3 - t2).as_secs_f64();
        st.sync_s += (t4 - t3).as_secs_f64();
        st.launches += launches;
        Ok(())
    }
    fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        let mut v = vec![];
        for l in &self.layers {
            v.extend(match l {
                Layer::Recurrent(a) => a.state_buffers(),
                Layer::Ple(a) => a.state_buffers(),
                Layer::Attention(a) => a.state_buffers(),
            });
        }
        v
    }
    pub fn checkpoint_new(&self) -> Result<Checkpoint<'d>> {
        let _arena = self.d.arena_scope();
        let bufs = self.state_buffers().iter().map(|b| self.d.alloc(b.byte_len())).collect::<Result<_>>()?;
        Ok(Checkpoint { bufs, positions: vec![], ple_previous: [0; 2], previous: [0; 2], position: 0, saved: false, table: None, rtable: None })
    }
    fn copy_all(&self, m: &Modules<'d>, pairs: Vec<(&Buffer<'d>, &Buffer<'d>)>) -> Result<()> {
        let k = m.copy.kernel("tinycudart_copy1d")?;
        let mut calls = vec![];
        for (dst, src) in pairs {
            ensure!(dst.byte_len() == src.byte_len(), "checkpoint buffer size");
            let n = src.byte_len();
            let grid = (n / 16).div_ceil(256).clamp(1, 1024) as u32;
            calls.push(unsafe { k.prepare(vec![Arg::Buffer(dst), Arg::Buffer(src), Arg::U64(n as u64)], [grid, 1, 1], [256, 1, 1])? });
        }
        for c in &calls {
            c.enqueue()?;
        }
        self.d.sync()
    }
    /// Save the committed drafter state (no forward may be pending).
    pub fn checkpoint_save(&self, m: &Modules<'d>, ck: &mut Checkpoint<'d>) -> Result<()> {
        ensure!(self.ready.get() && self.pending.get().is_none() && self.inflight.get().is_none(), "checkpoint needs a committed state");
        let src = self.state_buffers();
        ensure!(src.len() == ck.bufs.len(), "checkpoint layout");
        self.copy_all(m, ck.bufs.iter().zip(src).map(|(d, s)| (d, s)).collect())?;
        ck.positions = self.layers.iter().map(|l| match l { Layer::Attention(a) => a.position_meta(), _ => None }).collect();
        ck.ple_previous = self.layers.iter().find_map(|l| match l { Layer::Ple(a) => Some(a.previous_tokens()), _ => None }).context("PLE layer")?;
        ck.previous = self.previous.get();
        ck.position = self.position.get();
        ck.saved = true;
        Ok(())
    }
    /// R58: checkpoint_save without a host wait: one copy-table launch queued on the drafter's
    /// stream (it completes before anything queued after it, e.g. the next graph step).
    pub fn checkpoint_save_async(&self, m: &Modules<'d>, ck: &mut Checkpoint<'d>) -> Result<()> {
        ensure!(self.ready.get() && self.pending.get().is_none() && self.inflight.get().is_none(), "async checkpoint needs a committed state");
        let src = self.state_buffers();
        ensure!(src.len() == ck.bufs.len(), "checkpoint layout");
        if ck.table.is_none() {
            let _arena = self.d.arena_scope();
            let raw: Vec<u8> = ck.bufs.iter().zip(&src).flat_map(|(d, s)| [d.device_ptr(), s.device_ptr(), s.byte_len() as u64]).flat_map(|v| v.to_le_bytes()).collect();
            let b = self.d.alloc(raw.len())?;
            b.upload(&raw)?;
            ck.table = Some((b, src.len()));
        }
        let _ = m;
        let (table, n) = ck.table.as_ref().unwrap();
        let call = unsafe { self.copy_table.prepare(vec![Arg::Buffer(table)], [64, *n as u32, 1], [256, 1, 1])? };
        call.enqueue()?;
        ck.positions = self.layers.iter().map(|l| match l { Layer::Attention(a) => a.position_meta(), _ => None }).collect();
        ck.ple_previous = self.layers.iter().find_map(|l| match l { Layer::Ple(a) => Some(a.previous_tokens()), _ => None }).context("PLE layer")?;
        ck.previous = self.previous.get();
        ck.position = self.position.get();
        ck.saved = true;
        Ok(())
    }
    /// R58: exchange two checkpoints' contents (an asynchronous save into a scratch checkpoint
    /// becomes a boundary checkpoint without a copy).
    pub fn checkpoint_swap(a: &mut Checkpoint<'d>, b: &mut Checkpoint<'d>) {
        std::mem::swap(a, b);
    }
    /// Restore a saved checkpoint (no forward may be pending).
    pub fn checkpoint_restore(&self, m: &Modules<'d>, ck: &Checkpoint<'d>) -> Result<()> {
        ensure!(ck.saved && self.pending.get().is_none() && self.inflight.get().is_none(), "restore needs a saved checkpoint and no pending forward");
        let dst = self.state_buffers();
        ensure!(dst.len() == ck.bufs.len(), "checkpoint layout");
        self.copy_all(m, dst.into_iter().zip(ck.bufs.iter()).collect())?;
        for (l, meta) in self.layers.iter().zip(&ck.positions) {
            match l {
                Layer::Recurrent(a) => a.invalidate_pending(),
                Layer::Ple(a) => a.restore_meta(ck.ple_previous),
                Layer::Attention(a) => a.set_position_meta(*meta),
            }
        }
        self.previous.set(ck.previous);
        self.position.set(ck.position);
        self.ready.set(true);
        Ok(())
    }
    /// R58: checkpoint_restore by one queued copy-table launch (no host wait; the copies complete
    /// before anything queued after them on the drafter's stream).
    pub fn checkpoint_restore_fast(&self, ck: &mut Checkpoint<'d>) -> Result<()> {
        ensure!(ck.saved && self.pending.get().is_none() && self.inflight.get().is_none(), "restore needs a saved checkpoint and no pending forward");
        let dst = self.state_buffers();
        ensure!(dst.len() == ck.bufs.len(), "checkpoint layout");
        if ck.rtable.is_none() {
            let _arena = self.d.arena_scope();
            let raw: Vec<u8> = dst.iter().zip(&ck.bufs).flat_map(|(d, s)| [d.device_ptr(), s.device_ptr(), s.byte_len() as u64]).flat_map(|v| v.to_le_bytes()).collect();
            let b = self.d.alloc(raw.len())?;
            b.upload(&raw)?;
            ck.rtable = Some((b, dst.len()));
        }
        let (table, n) = ck.rtable.as_ref().unwrap();
        let call = unsafe { self.copy_table.prepare(vec![Arg::Buffer(table)], [64, *n as u32, 1], [256, 1, 1])? };
        call.enqueue()?;
        for (l, meta) in self.layers.iter().zip(&ck.positions) {
            match l {
                Layer::Recurrent(a) => a.invalidate_pending(),
                Layer::Ple(a) => a.restore_meta(ck.ple_previous),
                Layer::Attention(a) => a.set_position_meta(*meta),
            }
        }
        self.previous.set(ck.previous);
        self.position.set(ck.position);
        self.ready.set(true);
        Ok(())
    }
    /// The head's BF16-rounded logits of the last executed forward.
    pub fn logits_bf16(&self) -> Result<Vec<u16>> {
        // Row 0 only: the buffer holds MAX_ROWS rows (~7.9 MB); one row is ~0.5 MB.
        let b = self.io.logits();
        let mut raw = vec![0u8; lowbit_io::VOCAB * 2];
        ensure!(b.byte_len() >= raw.len(), "logits buffer size");
        b.download(&mut raw)?;
        Ok(raw.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect())
    }
    /// R56 graph drafting step: the forward + commit plan is captured once into `graph`; later
    /// steps re-record only the attention layers' forward and commit calls (the only
    /// position-dependent launches) and patch their changed scalar arguments in place. A
    /// structural change re-captures. Returns (logits, wall us, recaptured, patched args).
    #[cfg(feature = "r53-driver")]
    pub fn step_graph<'s>(&'s self, graph: &mut Option<DraftGraph<'s, 'm, 'd>>) -> Result<(Vec<u16>, f64, bool, usize)> {
        let (l, _, us, re, n) = self.graph_step(graph, false)?;
        Ok((l.context("logits")?, us, re, n))
    }
    /// R56 graph drafting step returning only the GPU top-20 (token id, BF16-rounded logit),
    /// best first with ties to the lower id: everything the T=1 / top-k 20 / top-p sampler
    /// reads. Downloads 160 bytes instead of the logits row.
    #[cfg(feature = "r53-driver")]
    pub fn step_graph_topk<'s>(&'s self, graph: &mut Option<DraftGraph<'s, 'm, 'd>>) -> Result<(Vec<(u32, f32)>, f64, bool, usize)> {
        let (_, t, us, re, n) = self.graph_step(graph, true)?;
        Ok((t.context("top-k")?, us, re, n))
    }
    /// R56 zero-copy drafting (one-way switch, before checkpoints are allocated): every GDN core
    /// reads its initial state from its snapshot slot, so the 3 MiB FP32 state per layer is never
    /// copied; the convolution histories are committed by two copy-table launches. Graph steps
    /// only (the queued paths refuse). Resets the drafter.
    pub fn enable_zero_copy(&self) -> Result<()> {
        ensure!(self.zero_copy.get().is_none() && self.pending.get().is_none(), "zero-copy switch");
        let gdns: Vec<&crate::cuda_gdn::Gdn<'m, 'd>> = self
            .layers
            .iter()
            .filter_map(|l| match l {
                Layer::Recurrent(a) => Some(a.gdn_ref()),
                Layer::Ple(a) => Some(a.body_gdn()),
                Layer::Attention(_) => None,
            })
            .collect();
        let (mut gather, mut restore) = (vec![], vec![]);
        for g in &gdns {
            g.set_zero_copy(true);
            let (gs, r) = g.conv_copies();
            gather.extend(gs);
            restore.push(r);
        }
        let table = |e: &[(u64, u64, u64)]| -> Result<Buffer<'d>> {
            let b = self.d.alloc(e.len() * 24)?;
            let raw: Vec<u8> = e.iter().flat_map(|&(d, s, n)| [d, s, n]).flat_map(|v| v.to_le_bytes()).collect();
            b.upload(&raw)?;
            Ok(b)
        };
        let z = ZeroCopy { gather: table(&gather)?, gather_n: gather.len(), restore: table(&restore)?, restore_n: restore.len() };
        let _ = self.zero_copy.set(z);
        ZERO_COPY_ON.store(true, std::sync::atomic::Ordering::Relaxed);
        self.reset()
    }
    fn record_zero_copy(&self) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        let Some(z) = self.zero_copy.get() else { return Ok(vec![]) };
        Ok(vec![
            unsafe { self.copy_table.prepare(vec![Arg::Buffer(&z.gather)], [8, z.gather_n as u32, 1], [256, 1, 1])? },
            unsafe { self.copy_table.prepare(vec![Arg::Buffer(&z.restore)], [8, z.restore_n as u32, 1], [256, 1, 1])? },
        ])
    }
    /// GPU top-k over the first `rows` logits rows (two launches).
    fn record_topk(&self, rows: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        ensure!((1..=TOPK_ROWS).contains(&rows) && TOPK_BLOCKS * TOPK <= 4096, "top-k rows");
        let t = &self.topk;
        let v = lowbit_io::VOCAB as u32;
        let partial = unsafe {
            t.partial.prepare(
                vec![Arg::Buffer(self.io.logits()), Arg::U32(v), Arg::U32(v), Arg::Buffer(&t.cand_v), Arg::Buffer(&t.cand_i)],
                [TOPK_BLOCKS as u32, rows as u32, 1],
                [256, 1, 1],
            )?
        };
        let last = unsafe {
            t.last.prepare(
                vec![Arg::Buffer(&t.cand_v), Arg::Buffer(&t.cand_i), Arg::U32((TOPK_BLOCKS * TOPK) as u32), Arg::Buffer(&t.out)],
                [1, rows as u32, 1],
                [256, 1, 1],
            )?
        };
        Ok(vec![partial, last])
    }
    fn read_topk(&self, rows: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        let mut raw = vec![0u8; rows * 2 * TOPK * 4];
        self.topk.out.download(&mut raw)?;
        let w: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        w.chunks_exact(2 * TOPK)
            .map(|r| {
                let out: Vec<(u32, f32)> = (0..TOPK).map(|i| (r[TOPK + i], f32::from_bits(r[i]))).collect();
                ensure!(out.iter().all(|x| (x.0 as usize) < lowbit_io::VOCAB), "top-k id range");
                Ok(out)
            })
            .collect()
    }
    #[cfg(feature = "r53-driver")]
    /// R58: graph_step's launch half: patch or capture, launch, no wait. Returns (token, start,
    /// patch us, launch us, recaptured, patched) for graph_finish.
    fn graph_launch<'s>(
        &'s self,
        graph: &mut Option<DraftGraph<'s, 'm, 'd>>,
        topk: bool,
    ) -> Result<(u32, Instant, f64, f64, bool, usize)> {
        ensure!(self.ready.get(), "graph step bounds");
        let token = self.pending.get().context("graph step needs a prepared token")?;
        self.ready.set(false);
        let t = Instant::now();
        let mut patched = 0usize;
        let mut reused = false;
        if let Some(dg) = graph.as_mut().filter(|g| g.topk == topk) {
            reused = true;
            for &(li, start, len) in &dg.fwd {
                let input = if li == 0 { self.io.stream() } else { self.layers[li - 1].output() };
                let Layer::Attention(a) = &self.layers[li] else { anyhow::bail!("graph layout") };
                let calls: Vec<PreparedCall<'s, 'm, 'd>> = a.record_from(input, 1, attention_policy())?.into_iter().flatten().collect();
                if calls.len() != len {
                    reused = false;
                    break;
                }
                match dg.graph.patch_range(start, calls)? {
                    Some(n) => patched += n,
                    None => {
                        reused = false;
                        break;
                    }
                }
            }
            if reused {
                for &(li, start, len) in &dg.commits {
                    let Layer::Attention(a) = &self.layers[li] else { anyhow::bail!("graph layout") };
                    let calls = a.record_commit(1)?;
                    if calls.len() != len {
                        reused = false;
                        break;
                    }
                    match dg.graph.patch_range(start, calls)? {
                        Some(n) => patched += n,
                        None => {
                            reused = false;
                            break;
                        }
                    }
                }
            }
        }
        let recaptured = !reused;
        let t_patch = t.elapsed().as_secs_f64() * 1e6;
        if reused {
            graph.as_ref().unwrap().graph.launch()?;
        } else {
            // fresh capture executes the plan once (no separate launch)
            *graph = None;
            let groups = self.record()?;
            let (mut fwd, mut commits) = (vec![], vec![]);
            let mut at = groups[0].len();
            for (li, g) in groups[1..groups.len() - 1].iter().enumerate() {
                if matches!(self.layers[li], Layer::Attention(_)) {
                    fwd.push((li, at, g.len()));
                }
                at += g.len();
            }
            at += groups.last().unwrap().len();
            let mut calls: Vec<PreparedCall<'s, 'm, 'd>> = groups.into_iter().flatten().collect();
            if topk {
                let t = self.record_topk(1)?;
                at += t.len();
                calls.extend(t);
            }
            let z = self.record_zero_copy()?;
            at += z.len();
            calls.extend(z);
            for (li, layer) in self.layers.iter().enumerate() {
                let c = match layer {
                    Layer::Recurrent(a) => a.record_commit(1, recurrent_policy())?,
                    Layer::Ple(a) => a.record_commit(1, ple_policy())?,
                    Layer::Attention(a) => a.record_commit(1)?,
                };
                if matches!(layer, Layer::Attention(_)) {
                    commits.push((li, at, c.len()));
                }
                at += c.len();
                calls.extend(c);
            }
            ensure!(at == calls.len(), "graph layout accounting");
            *graph = Some(DraftGraph { graph: self.d.capture_model_diagnostic(calls)?, fwd, commits, topk });
        }
        let t_launch = t.elapsed().as_secs_f64() * 1e6;
        Ok((token, t, t_patch, t_launch, recaptured, patched))
    }
    /// R58: graph_step's finish half: wait, read the logits or top-k, commit the row.
    fn graph_finish(&self, token: u32, t: Instant, t_patch: f64, t_launch: f64, topk: bool, recaptured: bool, patched: usize) -> Result<(Option<Vec<u16>>, Option<Vec<(u32, f32)>>, f64, bool, usize)> {
        self.d.sync()?;
        let t_sync = t.elapsed().as_secs_f64() * 1e6;
        let (logits, top) = if topk { (None, Some(self.read_topk(1)?.remove(0))) } else { (Some(self.logits_bf16()?), None) };
        let t_logits = t.elapsed().as_secs_f64() * 1e6;
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.finish_commit(),
                Layer::Ple(a) => a.finish_commit(1)?,
                Layer::Attention(a) => a.finish_commit(1)?,
            }
        }
        self.previous.set(ple::accepted_tokens(self.previous.get(), &[token], 1)?);
        self.position.set(self.position.get() + 1);
        self.pending.set(None);
        self.ready.set(true);
        let total = t.elapsed().as_secs_f64() * 1e6;
        if std::env::var("QWEN_R56_STEP_TRACE").as_deref() == Ok("1") {
            println!("{}", json!({"kind":"graph_step_phases","recaptured":recaptured,"patched":patched,"record_patch_us":t_patch,
                "launch_us":t_launch - t_patch,"sync_us":t_sync - t_launch,"logits_us":t_logits - t_sync,"finish_us":total - t_logits}));
        }
        Ok((logits, top, total, recaptured, patched))
    }
    fn graph_step<'s>(
        &'s self,
        graph: &mut Option<DraftGraph<'s, 'm, 'd>>,
        topk: bool,
    ) -> Result<(Option<Vec<u16>>, Option<Vec<(u32, f32)>>, f64, bool, usize)> {
        ensure!(self.inflight.get().is_none(), "graph step with an asynchronous step in flight");
        let (token, t, t_patch, t_launch, recaptured, patched) = self.graph_launch(graph, topk)?;
        self.graph_finish(token, t, t_patch, t_launch, topk, recaptured, patched)
    }
    /// R58 asynchronous drafting: launch the prepared token's graph step (top-k) and return at
    /// once; step_graph_done polls it, step_graph_finish completes it (the row is committed then).
    pub fn step_graph_launch<'s>(&'s self, graph: &mut Option<DraftGraph<'s, 'm, 'd>>) -> Result<()> {
        ensure!(self.inflight.get().is_none(), "one asynchronous step at a time");
        let (token, t, t_patch, t_launch, recaptured, patched) = self.graph_launch(graph, true)?;
        self.d.mark()?;
        let _ = t_launch;
        self.inflight.set(Some((token, t, t_patch, recaptured, patched)));
        Ok(())
    }
    pub fn step_graph_done(&self) -> Result<bool> {
        ensure!(self.inflight.get().is_some(), "no asynchronous step in flight");
        self.d.marker_done()
    }
    pub fn step_in_flight(&self) -> bool {
        self.inflight.get().is_some()
    }
    pub fn step_graph_finish(&self) -> Result<(Vec<(u32, f32)>, f64, bool, usize)> {
        let (token, t, t_patch, recaptured, patched) = self.inflight.take().context("no asynchronous step in flight")?;
        let t_launch = t_patch;
        let (_, top, us, re, n) = self.graph_finish(token, t, t_patch, t_launch, true, recaptured, patched)?;
        Ok((top.context("top-k")?, us, re, n))
    }
    /// R56: sample the SM clock with the R53 clock probe (QWEN_R53_CLOCK=1 loads it); prints an
    /// r53_clock line. A plain one-block-per-SM spin kernel, normal ordering.
    pub fn sample_clock(&self, when: &str) -> Result<()> {
        if let Some([_, _, clock]) = &self.cal {
            crate::r53_cal::clock(self.d, clock, when, true)?;
        }
        Ok(())
    }
    /// R56 diagnostic: where the graph drafting step's GPU time goes. Captures the prepared
    /// forward alone, then the commit(1) calls alone (per layer kind and all together), replays
    /// each `reps` times (median launch-to-sync wall us) and times the logits download.
    /// Repeated commits shift the histories (finite values, unchanged addresses); the caller
    /// must reset afterwards.
    #[cfg(feature = "r53-driver")]
    pub fn split_timing(&self, reps: usize) -> Result<Value> {
        ensure!(self.ready.get() && self.pending.get().is_some() && (1..=64).contains(&reps), "split timing bounds");
        self.ready.set(false);
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let time = |g: &crate::cuda::Graph<'_, 'm, 'd>| -> Result<f64> {
            let mut v = vec![];
            for _ in 0..reps {
                let t = Instant::now();
                g.launch()?;
                self.d.sync()?;
                v.push(t.elapsed().as_secs_f64() * 1e6);
            }
            Ok(median(v))
        };
        let fwd: Vec<PreparedCall<'_, 'm, 'd>> = self.record()?.into_iter().flatten().collect();
        let n_fwd = fwd.len();
        let g = self.d.capture_model_diagnostic(fwd)?;
        self.d.sync()?;
        let forward_us = time(&g)?;
        drop(g);
        // g4w2h MoE spin-timeout words (sticky; nonzero = a hand-off timed out, values invalid)
        let mut statuses = vec![];
        for l in &self.layers {
            let s = match l {
                Layer::Recurrent(a) => a.moe_status()?,
                Layer::Attention(a) => a.moe_status()?,
                _ => None,
            };
            statuses.extend(s);
        }
        let mut out = json!({"kind":"graph_split_timing","reps":reps,"forward_launches":n_fwd,"forward_us":forward_us,
            "moe_status_layers":statuses.len(),"moe_timeouts":statuses.iter().filter(|&&s| s != 0).count()});
        for kind in ["recurrent", "ple", "attention", "all"] {
            let mut calls = vec![];
            for layer in &self.layers {
                calls.extend(match layer {
                    Layer::Recurrent(a) if kind == "recurrent" || kind == "all" => a.record_commit(1, recurrent_policy())?,
                    Layer::Ple(a) if kind == "ple" || kind == "all" => a.record_commit(1, ple_policy())?,
                    Layer::Attention(a) if kind == "attention" || kind == "all" => a.record_commit(1)?,
                    _ => vec![],
                });
            }
            if calls.is_empty() {
                continue;
            }
            out[format!("commit_{kind}_launches")] = json!(calls.len());
            let g = self.d.capture_model_diagnostic(calls)?;
            self.d.sync()?;
            out[format!("commit_{kind}_us")] = json!(time(&g)?);
        }
        let mut v = vec![];
        for _ in 0..reps {
            let t = Instant::now();
            let _ = self.logits_bf16()?;
            v.push(t.elapsed().as_secs_f64() * 1e6);
        }
        out["logits_download_us"] = json!(median(v));
        Ok(out)
    }
    /// R56 lean drafting step for the prepared token: the forward and every layer's
    /// commit(1) in one queued submission (bounded by `max_queued_calls`), one sync, then only
    /// the logits are downloaded. No diagnostic poison uploads or CPU collection; layers are
    /// marked complete. Returns (BF16 logits, wall us).
    pub fn step_lean(&self, max_queued_calls: usize) -> Result<(Vec<u16>, f64)> {
        queued_allowed()?;
        ensure!(self.ready.get() && (1..=4096).contains(&max_queued_calls), "lean step bounds");
        let token = self.pending.get().context("lean step needs a prepared token")?;
        self.ready.set(false);
        let t = Instant::now();
        {
            let groups = self.record()?;
            let mut commits = vec![];
            for layer in &self.layers {
                commits.extend(match layer {
                    Layer::Recurrent(a) => a.record_commit(1, recurrent_policy())?,
                    Layer::Ple(a) => a.record_commit(1, ple_policy())?,
                    Layer::Attention(a) => a.record_commit(1)?,
                });
            }
            let mut pending = 0;
            for call in groups.iter().flatten().chain(commits.iter()) {
                call.enqueue()?;
                pending += 1;
                if pending == max_queued_calls {
                    self.d.sync()?;
                    pending = 0;
                }
            }
            self.d.sync()?;
        }
        let logits = self.logits_bf16()?;
        for layer in &self.layers {
            match layer {
                Layer::Recurrent(a) => a.finish_commit(),
                Layer::Ple(a) => a.finish_commit(1)?,
                Layer::Attention(a) => a.finish_commit(1)?,
            }
        }
        self.previous.set(ple::accepted_tokens(self.previous.get(), &[token], 1)?);
        self.position.set(self.position.get() + 1);
        self.pending.set(None);
        self.ready.set(true);
        Ok((logits, t.elapsed().as_secs_f64() * 1e6))
    }
}
