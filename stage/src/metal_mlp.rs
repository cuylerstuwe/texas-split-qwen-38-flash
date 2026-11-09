//! Complete original EXL3 target MLP on one Metal command stream. The encode
//! interface has no submit/wait/readback, so a layer can keep dependencies on
//! device. Engine is only the bounded component harness around that interface.
use crate::{
    metal_ops::{
        Arm as DenseArm, Buffer, DenseOps, DenseWeight, Obj, alloc, dense_scratch_bytes, upload,
    },
    moe::{FullExpertBank, H, I, TOPK},
    projection::ScaledProjection,
    target_mlp::{RoutingArithmetic, Weights as HostWeights},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
pub const MAX_ROWS: usize = 16;
const GUARD: usize = 64;
const U16_SENTINEL: u16 = 0x7fc1;
const U32_SENTINEL: u32 = 0x7fc12345;
const SPLITS: usize = 2;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type Pipeline = Obj<dyn MTLComputePipelineState>;

#[derive(Clone, Copy, Debug)]
pub enum SharedPrecision {
    Qmv,
    PackedF16x2,
    /// R55: QMV (RMV kernel) up to 4 rows, packed f16x2 fragments above, private scratch.
    Auto,
}
impl SharedPrecision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qmv => "qmv",
            Self::PackedF16x2 => "packed_f16x2",
            Self::Auto => "auto_qmv4_packed",
        }
    }
    fn dense(self) -> DenseArm {
        self.dense_rows(1)
    }
    fn dense_rows(self, rows: usize) -> DenseArm {
        self.dense_shape(rows, I)
    }
    /// R55: Auto sizes split-K to the output width (the fragment kernel tiles 128 columns
    /// per threadgroup, so a 640-wide projection needs many splits to occupy the GPU).
    fn dense_shape(self, rows: usize, n: usize) -> DenseArm {
        match self {
            Self::Qmv => DenseArm::Qmv4,
            Self::Auto if rows <= 4 => DenseArm::Qmv4,
            Self::Auto => DenseArm::FragmentPacked {
                splits: if n <= 1024 { 16 } else { 8 },
                residual: true,
            },
            Self::PackedF16x2 => DenseArm::FragmentPacked {
                splits: 2,
                residual: true,
            },
        }
    }
}

struct PackedWeight {
    trellis: Buffer,
    suh: Buffer,
    svh: Buffer,
    /// R55: byte offsets of trellis/suh/svh inside their buffers (non-zero when the
    /// buffers are no-copy mappings of the pack's expert files).
    offs: [usize; 3],
}
/// R55: where a layer's routed-expert weights come from.
pub enum Experts<'a> {
    /// Host copies uploaded into owned buffers.
    Bank(&'a FullExpertBank),
    /// Read-only mappings of the canonical pack's expert files (page cache, no copy).
    Mapped(&'a crate::projection::Pack, usize),
}
impl Experts<'_> {
    pub fn layer(&self) -> usize {
        match self {
            Self::Bank(b) => b.layer,
            Self::Mapped(_, l) => *l,
        }
    }
}
unsafe extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
}
/// Map a whole file read-only and wrap it in a no-copy shared buffer. The mapping is
/// never unmapped (the model lives for the process).
pub(crate) fn map_file(d: &ProtocolObject<dyn MTLDevice>, path: &std::path::Path) -> Result<Buffer> {
    use std::os::fd::AsRawFd;
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size = f.metadata()?.len() as usize;
    let len = size.div_ceil(16384) * 16384;
    // PROT_READ, MAP_SHARED
    let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 1, f.as_raw_fd(), 0) };
    ensure!(!ptr.is_null() && ptr as isize != -1, "mmap {}", path.display());
    let p = std::ptr::NonNull::new(ptr).context("mmap pointer")?;
    unsafe {
        d.newBufferWithBytesNoCopy_length_options_deallocator(p, len, MTLResourceOptions::StorageModeShared, None)
    }
    .with_context(|| format!("no-copy buffer {}", path.display()))
}
impl PackedWeight {
    fn mapped(
        d: &ProtocolObject<dyn MTLDevice>,
        pack: &crate::projection::Pack,
        layer: usize,
        projection: &str,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        let p = format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}");
        let path = pack.file_of(&format!("{p}.trellis"))?;
        for s in ["suh", "svh"] {
            ensure!(pack.file_of(&format!("{p}.{s}"))? == path, "expert tensors in one file");
        }
        let shard = crate::weights::Shard::open(&path)?;
        for (key, expected) in [("format", "exl3"), ("k", "4"), ("codebook", "mcg"), ("window", "15")] {
            ensure!(shard.metadata.get(key).map(String::as_str) == Some(expected), "pack stamp {key}");
        }
        let t = &shard.tensors[&format!("{p}.trellis")];
        ensure!(t.dtype == "U16" && t.shape == [512, k as u64 / 16, n as u64 / 16, 64], "trellis shape");
        let suh = &shard.tensors[&format!("{p}.suh")];
        let svh = &shard.tensors[&format!("{p}.svh")];
        ensure!(
            suh.dtype == "F16" && suh.shape == [512, k as u64] && svh.dtype == "F16" && svh.shape == [512, n as u64],
            "expert scale shapes"
        );
        let b = map_file(d, &path)?;
        let offs = [t, suh, svh].map(|x| (shard.data_start + x.data_offsets[0]) as usize);
        ensure!(offs.iter().all(|o| o % 8 == 0), "expert tensor alignment");
        // The sign/scale vectors must be finite (as the copying loader checks).
        for (o, count) in [(offs[1], 512 * k), (offs[2], 512 * n)] {
            let v = unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(o).cast::<u16>(), count) };
            ensure!(v.iter().all(|&x| half::f16::from_bits(x).is_finite()), "nonfinite expert scale");
        }
        Ok(Self { trellis: b.clone(), suh: b.clone(), svh: b, offs })
    }
    fn new(
        d: &ProtocolObject<dyn MTLDevice>,
        p: &ScaledProjection,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        ensure!(
            p.raw.experts == 512
                && p.raw.k == k
                && p.raw.n == n
                && p.raw.packed.len() == 512 * k * n / 2
                && p.suh.len() == 512 * k
                && p.svh.len() == 512 * n,
            "full packed expert geometry"
        );
        let w = Self {
            trellis: alloc(d, p.raw.packed.len())?,
            suh: alloc(d, p.suh.len() * 2)?,
            svh: alloc(d, p.svh.len() * 2)?,
            offs: [0; 3],
        };
        upload(&w.trellis, &p.raw.packed)?;
        upload(&w.suh, &p.suh)?;
        upload(&w.svh, &p.svh)?;
        Ok(w)
    }
    fn bytes(&self) -> usize {
        if self.offs[0] != 0 {
            return self.trellis.length();
        }
        self.trellis.length() + self.suh.length() + self.svh.length()
    }
}
pub(crate) struct Weights {
    router: Buffer,
    shared_gate: Buffer,
    experts: [PackedWeight; 3],
    shared: [DenseWeight; 3],
}
/// R58 prefill: one routed projection's (trellis, suh, svh) buffers and their byte offsets.
pub(crate) type ExpertParts<'a> = (&'a Buffer, &'a Buffer, &'a Buffer, [usize; 3]);
impl Weights {
    /// R58: the routed experts' buffers (no-copy file mappings or owned copies), deduplicated.
    pub(crate) fn expert_buffers(&self) -> Vec<&Buffer> {
        let mut v: Vec<&Buffer> = vec![];
        for w in &self.experts {
            for b in [&w.trellis, &w.suh, &w.svh] {
                if !v.iter().any(|x| objc2::rc::Retained::as_ptr(x) == objc2::rc::Retained::as_ptr(b)) {
                    v.push(b);
                }
            }
        }
        v
    }
    /// R58 prefill: (router BF16 [512][2560], shared-expert gate BF16 [2560], routed gate/up/down,
    /// shared-expert gate/up/down affine8).
    pub(crate) fn prefill_parts(&self) -> (&Buffer, &Buffer, [ExpertParts<'_>; 3], &[DenseWeight; 3]) {
        fn e(w: &PackedWeight) -> ExpertParts<'_> {
            (&w.trellis, &w.suh, &w.svh, w.offs)
        }
        (&self.router, &self.shared_gate, [e(&self.experts[0]), e(&self.experts[1]), e(&self.experts[2])], &self.shared)
    }
    pub(crate) fn new(
        d: &ProtocolObject<dyn MTLDevice>,
        bank: &FullExpertBank,
        w: &HostWeights,
    ) -> Result<Self> {
        Self::new_from(d, Experts::Bank(bank), w)
    }
    pub(crate) fn new_from(
        d: &ProtocolObject<dyn MTLDevice>,
        experts: Experts<'_>,
        w: &HostWeights,
    ) -> Result<Self> {
        ensure!(experts.layer() == w.layer, "MLP layer identity");
        w.router.validate()?;
        w.shared_gate.validate()?;
        ensure!(
            w.router.k == H && w.router.n == 512 && w.shared_gate.k == H && w.shared_gate.n == 1,
            "MLP router dimensions"
        );
        let router = alloc(d, w.router.values.len() * 2)?;
        upload(&router, &w.router.values)?;
        let shared_gate = alloc(d, w.shared_gate.values.len() * 2)?;
        upload(&shared_gate, &w.shared_gate.values)?;
        Ok(Self {
            router,
            shared_gate,
            experts: match experts {
                Experts::Bank(bank) => [
                    PackedWeight::new(d, &bank.gate, H, I)?,
                    PackedWeight::new(d, &bank.up, H, I)?,
                    PackedWeight::new(d, &bank.down, I, H)?,
                ],
                Experts::Mapped(pack, layer) => [
                    PackedWeight::mapped(d, pack, layer, "gate_proj", H, I)?,
                    PackedWeight::mapped(d, pack, layer, "up_proj", H, I)?,
                    PackedWeight::mapped(d, pack, layer, "down_proj", I, H)?,
                ],
            },
            shared: [
                DenseWeight::new(d, &w.shared[0])?,
                DenseWeight::new(d, &w.shared[1])?,
                DenseWeight::new(d, &w.shared[2])?,
            ],
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.router.length()
            + self.shared_gate.length()
            + self.experts.iter().map(PackedWeight::bytes).sum::<usize>()
            + self.shared.iter().map(DenseWeight::bytes).sum::<usize>()
    }
}
struct Pair {
    full: Buffer,
    bf: Buffer,
    n: usize,
}
impl Pair {
    fn new(d: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Self> {
        Ok(Self {
            full: alloc(d, (MAX_ROWS * n + GUARD) * 4)?,
            bf: alloc(d, (MAX_ROWS * n + GUARD) * 2)?,
            n,
        })
    }
    fn poison(&self) {
        poison32(&self.full);
        poison16(&self.bf);
    }
    fn collect(&self, rows: usize) -> Result<Vec<u16>> {
        read32(&self.full, rows * self.n)?;
        read16(&self.bf, rows * self.n, false)
    }
    fn bytes(&self) -> usize {
        self.full.length() + self.bf.length()
    }
}
/// One workspace can be reused by sequential layers. Weights and pipelines
/// are separate; no model-sized multiplication of activation allocations.
impl Workspace {
    /// R58 lead 1: the last forward's routed ids and weights for `rows` rows ([row * 10 + rank]).
    pub(crate) fn routes(&self, rows: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(rows <= MAX_ROWS, "route rows");
        // SAFETY: shared storage, read between forwards.
        let ids = unsafe { std::slice::from_raw_parts(self.ids.contents().cast::<u32>().as_ptr(), rows * TOPK) }.to_vec();
        let w = unsafe { std::slice::from_raw_parts(self.scores.contents().cast::<f32>().as_ptr(), rows * TOPK) }.to_vec();
        Ok((ids, w))
    }
}
pub(crate) struct Workspace {
    router: Pair,
    ids: Buffer,
    scores: Buffer,
    identity: Buffer,
    pg: Buffer,
    pu: Buffer,
    rg: Buffer,
    ru: Buffer,
    middle: Buffer,
    rd: Buffer,
    routed: Pair,
    sg: Pair,
    su: Pair,
    sa: Buffer,
    sd: Pair,
    gate: Pair,
    sigmoid: Buffer,
    shared: Buffer,
    sums: Buffer,
    /// R55 expert grouping table.
    table: Buffer,
    /// R55: private split-K scratch for the shared gate/up/down projections.
    ssums: [Buffer; 3],
}
impl Workspace {
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let slots = MAX_ROWS * TOPK;
        let identity = alloc(d, slots * 4)?;
        upload(&identity, &(0..slots as u32).collect::<Vec<_>>())?;
        Ok(Self {
            router: Pair::new(d, 512)?,
            ids: alloc(d, (slots + GUARD) * 4)?,
            scores: alloc(d, (slots + GUARD) * 4)?,
            identity,
            pg: alloc(d, (slots * H + GUARD) * 2)?,
            pu: alloc(d, (slots * H + GUARD) * 2)?,
            rg: alloc(d, (slots * I * SPLITS + GUARD) * 4)?,
            ru: alloc(d, (slots * I * SPLITS + GUARD) * 4)?,
            middle: alloc(d, (slots * I + GUARD) * 2)?,
            rd: alloc(d, (slots * H * SPLITS + GUARD) * 4)?,
            routed: Pair::new(d, H)?,
            sg: Pair::new(d, I)?,
            su: Pair::new(d, I)?,
            sa: alloc(d, (MAX_ROWS * I + GUARD) * 2)?,
            sd: Pair::new(d, H)?,
            gate: Pair::new(d, 1)?,
            sigmoid: alloc(d, (MAX_ROWS + GUARD) * 2)?,
            shared: alloc(d, (MAX_ROWS * H + GUARD) * 2)?,
            sums: alloc(
                d,
                dense_scratch_bytes(H, I, MAX_ROWS).max(dense_scratch_bytes(I, H, MAX_ROWS)),
            )?,
            table: alloc(d, (64 + 8 * 12 * slots + GUARD) * 4)?,
            ssums: [
                alloc(d, dense_scratch_bytes(H, I, MAX_ROWS))?,
                alloc(d, dense_scratch_bytes(H, I, MAX_ROWS))?,
                alloc(d, dense_scratch_bytes(I, H, MAX_ROWS))?,
            ],
        })
    }
    pub(crate) fn poison(&self) {
        for p in [
            &self.router,
            &self.routed,
            &self.sg,
            &self.su,
            &self.sd,
            &self.gate,
        ] {
            p.poison();
        }
        for b in [&self.ids, &self.scores, &self.rg, &self.ru, &self.rd] {
            poison32(b);
        }
        for b in [
            &self.pg,
            &self.pu,
            &self.middle,
            &self.sa,
            &self.sigmoid,
            &self.shared,
        ] {
            poison16(b);
        }
    }
    pub(crate) fn collect(&self, out: &Buffer, rows: usize) -> Result<Stages> {
        ensure!((1..=MAX_ROWS).contains(&rows), "MLP collect rows");
        let slots = rows * TOPK;
        let logits = self.router.collect(rows)?;
        let ids = read_ids(&self.ids, slots)?;
        let scores = read32(&self.scores, slots)?;
        ensure!(scores.iter().all(|&v| v >= 0.), "MLP nonnegative scores");
        let rg = read32(&self.rg, slots * I * SPLITS)?;
        let ru = read32(&self.ru, slots * I * SPLITS)?;
        let rd = read32(&self.rd, slots * H * SPLITS)?;
        let routed_bf = self.routed.collect(rows)?;
        Ok(Stages {
            router_raw: read32(&self.router.full, rows * 512)?,
            logits,
            ids,
            scores,
            routed: vec![
                half_values(read16(&self.pg, slots * H, true)?),
                half_values(read16(&self.pu, slots * H, true)?),
                sum_splits(rg, I),
                sum_splits(ru, I),
                half_values(read16(&self.middle, slots * I, true)?),
                sum_splits(rd, H),
                read32(&self.routed.full, rows * H)?,
                routed_bf
                    .iter()
                    .map(|&v| bf16::from_bits(v).to_f32())
                    .collect(),
            ],
            shared: vec![
                self.sg.collect(rows)?,
                self.su.collect(rows)?,
                read16(&self.sa, rows * I, false)?,
                self.sd.collect(rows)?,
                self.gate.collect(rows)?,
                read16(&self.sigmoid, rows, false)?,
                read16(&self.shared, rows * H, false)?,
            ],
            output: read16(out, rows * H, false)?,
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        [
            &self.router,
            &self.routed,
            &self.sg,
            &self.su,
            &self.sd,
            &self.gate,
        ]
        .iter()
        .map(|p| p.bytes())
        .sum::<usize>()
            + [
                &self.ids,
                &self.scores,
                &self.identity,
                &self.pg,
                &self.pu,
                &self.rg,
                &self.ru,
                &self.middle,
                &self.rd,
                &self.sa,
                &self.sigmoid,
                &self.shared,
                &self.sums,
            ]
            .iter()
            .map(|b| b.length())
            .sum::<usize>()
    }
}
#[track_caller]
fn barrier(e: &Encoder) {
    crate::metal_ops::barrier(&e);
}
fn dispatch(e: &Encoder, x: usize, y: usize, z: usize, threads: usize) {
    e.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: x,
            height: y,
            depth: z,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}
/// Re-bind one argument at a byte offset (mapped expert tensors).
fn at(e: &Encoder, b: &Buffer, offset: usize, index: usize) {
    unsafe { e.setBuffer_offset_atIndex(Some(b), offset, index) }
}
fn bind(e: &Encoder, b: &[&Buffer]) {
    // The owning layer/engine retains every allocation until completion. Fixed
    // ABIs match the included MSL, and encode validates external buffer bounds.
    unsafe {
        for (i, b) in b.iter().enumerate() {
            e.setBuffer_offset_atIndex(Some(b), 0, i);
        }
    }
}
fn params<const N: usize>(e: &Encoder, p: &[u32; N], index: usize) {
    unsafe {
        e.setBytes_length_atIndex(NonNull::from(p).cast(), N * 4, index);
    }
}
pub(crate) struct Ops {
    dense: DenseOps,
    project: Pipeline,
    prepare: Pipeline,
    mid: Pipeline,
    finish: Pipeline,
    router: Pipeline,
    select: Pipeline,
    /// R58: select folded into prepare (QWEN_R58_SELECT_PREPARE): per slot, and row-grouped.
    select_prepare: Pipeline,
    select_prepare_rows: Pipeline,
    shared_gate: Pipeline,
    activate: Pipeline,
    join: Pipeline,
    finish_join: Pipeline,
    /// R55: per-expert slot grouping and the grouped projections (S = 2, 4, 8).
    group: Pipeline,
    grouped: [Pipeline; 4],
    bucket_group: Pipeline,
    bucket: Vec<Pipeline>,
}
fn mlp_overlap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_MLP_OVERLAP").as_deref() != Ok("0"))
}
/// R58 (QWEN_R58_FINISH_JOIN=1): the routed finish and the shared join in one kernel
/// (moe_finish_join; bit-identical; one dispatch and one barrier fewer per layer).
/// R58 lead 1 (QWEN_R58_MOE_PRUNE=0.90, QWEN_R58_MOE_RENORM=1): drop each token's routed experts
/// past the cumulative gate weight threshold (weight 0; every slot is still computed, so this shows
/// the quality cost; the speed needs the slot kernels to skip them). Default off (bit-identical).
pub(crate) fn moe_prune() -> (f32, bool) {
    static V: std::sync::OnceLock<(f32, bool)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let t = std::env::var("QWEN_R58_MOE_PRUNE").ok().and_then(|v| v.parse::<f32>().ok()).filter(|t| *t > 0. && *t < 1.).unwrap_or(1.0);
        (t, std::env::var("QWEN_R58_MOE_RENORM").as_deref() == Ok("1"))
    })
}
/// R58 QWEN_R58_SELECT_PREPARE=1: the router selection runs inside the expert-input prepare
/// (tm_select_prepare_pair), dropping one dispatch and one barrier per layer (bit-identical).
fn select_prepare() -> bool {
    select_prepare_kind() > 0
}
/// 0 off, 1 per-slot simdgroups (QWEN_R58_SELECT_PREPARE=1), 2 row-grouped (=rows).
fn select_prepare_kind() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("QWEN_R58_SELECT_PREPARE").as_deref() {
        Ok("1") => 1,
        Ok("rows") => 2,
        _ => 0,
    })
}
fn finish_join() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R58_FINISH_JOIN").as_deref() == Ok("1"))
}
fn bucket_kind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_GROUPED_KIND").as_deref() == Ok("bucket"))
}
const BUCKET_STRIDE: usize = MAX_ROWS * TOPK;
/// R55 opt-in (QWEN_R55_GROUPED=1): decode each expert tile once for all its slots.
pub(crate) fn grouped_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_GROUPED").as_deref() == Ok("1"))
}
impl Ops {
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let source = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../kernels/exl3_project.metal"),
            include_str!("../kernels/exl3_moe.metal"),
            include_str!("../kernels/target_mlp.metal"),
            include_str!("../kernels/exl3_grouped.metal")
        );
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| anyhow::anyhow!("target MLP compile: {e}"))?;
        let pipe = |name: &str, threads: usize| -> Result<Pipeline> {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("MLP function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("MLP pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= threads
                    && p.staticThreadgroupMemoryLength() <= d.maxThreadgroupMemoryLength(),
                "MLP launch geometry"
            );
            Ok(p)
        };
        Ok(Self {
            dense: DenseOps::new(d)?,
            project: pipe("exl3_project", 128)?,
            prepare: pipe("tm_prepare_pair", 32)?,
            mid: pipe("moe_mid", 32)?,
            finish: pipe("moe_finish", 320)?,
            router: pipe("tm_router_qmv", 128)?,
            select: pipe("tm_router_select", 32)?,
            select_prepare: pipe("tm_select_prepare_pair", 32)?,
            select_prepare_rows: pipe("tm_select_prepare_rows", 320)?,
            shared_gate: pipe("tm_shared_gate", 32)?,
            activate: pipe("tm_shared_activate", 256)?,
            join: pipe("tm_shared_join", 256)?,
            finish_join: pipe("moe_finish_join", 320)?,
            group: pipe("r55_group_slots", 512)?,
            grouped: [
                pipe("r55_exl3_project_grouped_s2", 128)?,
                pipe("r55_exl3_project_grouped_s4", 128)?,
                pipe("r55_exl3_project_grouped_s8", 128)?,
                pipe("r55_exl3_project_grouped_loop", 128)?,
            ],
            bucket_group: pipe("r55_bucket_slots", 512)?,
            bucket: {
                // QWEN_R55_UNROLL=2|4 selects the latency variants (default 1).
                let u = std::env::var("QWEN_R55_UNROLL").unwrap_or_default();
                (1..=8)
                    .map(|b| match u.as_str() {
                        "2" | "4" => pipe(&format!("r55_exl3_project_bucket_u{u}_s{b}"), 128),
                        _ => pipe(&format!("r55_exl3_project_bucket_s{b}"), 128),
                    })
                    .collect::<Result<_>>()?
            },
        })
    }
    fn projection(
        &self,
        e: &Encoder,
        w: &PackedWeight,
        x: &Buffer,
        ids: &Buffer,
        y: &Buffer,
        k: usize,
        n: usize,
        slots: usize,
    ) {
        e.setComputePipelineState(&self.project);
        bind(e, &[x, &w.trellis, ids, y]);
        at(e, &w.trellis, w.offs[0], 1);
        params(e, &[k as u32, n as u32, SPLITS as u32, 0], 4);
        dispatch(e, n / 16, slots, SPLITS, 128);
    }
    /// Exact-size buckets: eight indirect dispatches whose grids the bucket kernel wrote.
    fn projection_bucket(&self, e: &Encoder, w: &PackedWeight, x: &Buffer, table: &Buffer, y: &Buffer, k: usize, n: usize, args: usize) {
        for b in 0..8 {
            e.setComputePipelineState(&self.bucket[b]);
            bind(e, &[x, &w.trellis, table, y]);
            at(e, &w.trellis, w.offs[0], 1);
            params(e, &[k as u32, n as u32, SPLITS as u32, BUCKET_STRIDE as u32], 4);
            unsafe {
                e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    table,
                    (args + b * 3) * 4,
                    MTLSize { width: 128, height: 1, depth: 1 },
                );
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn projection_grouped(
        &self,
        e: &Encoder,
        w: &PackedWeight,
        x: &Buffer,
        table: &Buffer,
        y: &Buffer,
        k: usize,
        n: usize,
        slots: usize,
        rows: usize,
    ) {
        let pipe = if std::env::var("QWEN_R55_GROUPED_KIND").as_deref() == Ok("loop") {
            &self.grouped[3]
        } else {
            match rows {
                2 => &self.grouped[0],
                3..=4 => &self.grouped[1],
                _ => &self.grouped[2],
            }
        };
        e.setComputePipelineState(pipe);
        bind(e, &[x, &w.trellis, table, y]);
        at(e, &w.trellis, w.offs[0], 1);
        params(e, &[k as u32, n as u32, SPLITS as u32, 0], 4);
        dispatch(e, n / 16, slots, SPLITS, 128);
    }
    /// Caller supplies distinct BF16 router/expert inputs, same-device buffers,
    /// and an encoder; no input mutation/readback or CPU route plan is performed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        e: &Encoder,
        w: &Weights,
        s: &Workspace,
        rx: &Buffer,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        score: RoutingArithmetic,
        shared: SharedPrecision,
    ) -> Result<()> {
        if mlp_overlap() && matches!(shared, SharedPrecision::Qmv | SharedPrecision::Auto) {
            // R55: the shared-expert chain depends only on x; run it beside the router,
            // select and prepare steps (no shared split-K scratch with QMV projections).
            // R58 select_prepare (identity slot order only): stage 12 is select + prepare, so the
            // shared down projection moves beside the gate/up projections
            let fused = select_prepare() && !(grouped_enabled() && rows > 1);
            let groups: &[&[usize]] = match (fused, finish_join()) {
                (true, true) => &[&[0usize, 7, 8][..], &[12, 9], &[3, 10], &[4], &[5], &[6]],
                (true, false) => &[&[0usize, 7, 8][..], &[12, 9], &[3, 10], &[4], &[5], &[6], &[11]],
                (false, true) => &[&[0usize, 7, 8][..], &[1, 9], &[2, 10], &[3], &[4], &[5], &[6]],
                (false, false) => &[&[0usize, 7, 8][..], &[1, 9], &[2, 10], &[3], &[4], &[5], &[6], &[11]],
            };
            for (gi, group) in groups.iter().copied().enumerate() {
                for &stage in group {
                    self.encode_stage(e, stage, w, s, rx, x, out, rows, score, shared)?;
                }
                // one call site per group, so barrier attribution can tell them apart
                match gi {
                    0 => barrier(e),
                    1 => barrier(e),
                    2 => barrier(e),
                    3 => barrier(e),
                    4 => barrier(e),
                    5 => barrier(e),
                    6 => barrier(e),
                    _ => barrier(e),
                }
                for &stage in group {
                    if crate::metal_profile::dup(6 + stage as u32) {
                        self.encode_stage(e, stage, w, s, rx, x, out, rows, score, shared)?;
                        barrier(e);
                    }
                }
            }
            return Ok(());
        }
        for stage in 0..STAGES.len() {
            self.encode_stage(e, stage, w, s, rx, x, out, rows, score, shared)?;
            barrier(e);
        }
        Ok(())
    }
    /// R55 stage clock: the same stages, one command buffer each.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        w: &Weights,
        s: &Workspace,
        rx: &Buffer,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        score: RoutingArithmetic,
        shared: SharedPrecision,
    ) -> Result<()> {
        for (stage, label) in STAGES.iter().enumerate() {
            clock.stage(*label, |e| self.encode_stage(e, stage, w, s, rx, x, out, rows, score, shared))?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn encode_stage(
        &self,
        e: &Encoder,
        stage: usize,
        w: &Weights,
        s: &Workspace,
        rx: &Buffer,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        score: RoutingArithmetic,
        shared: SharedPrecision,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && rx.length() >= rows * H * 2
                && x.length() >= rows * H * 2
                && out.length() >= rows * H * 2,
            "MLP external buffer geometry"
        );
        let slots = rows * TOPK;
        let p = [
            H as u32,
            I as u32,
            TOPK as u32,
            SPLITS as u32,
            SPLITS as u32,
            0,
            0,
            0,
        ];
        match stage {
            0 => {
                e.setComputePipelineState(&self.router);
                bind(e, &[rx, &w.router, &s.router.full]);
                dispatch(e, 128, rows, 1, 128);
            }
            1 => {
                e.setComputePipelineState(&self.select);
                bind(e, &[&s.router.full, &s.router.bf, &s.ids, &s.scores]);
                params(
                    e,
                    &[matches!(score, RoutingArithmetic::SelectedF32) as u32],
                    4,
                );
                let (prune, renorm) = moe_prune();
                params(e, &[prune.to_bits()], 5);
                params(e, &[renorm as u32], 6);
                dispatch(e, rows, 1, 1, 32);
            }
            2 => {
                e.setComputePipelineState(&self.prepare);
                bind(
                    e,
                    &[
                        x,
                        &w.experts[0].suh,
                        &w.experts[1].suh,
                        &s.ids,
                        &s.identity,
                        &s.pg,
                        &s.pu,
                    ],
                );
                at(e, &w.experts[0].suh, w.experts[0].offs[1], 1);
                at(e, &w.experts[1].suh, w.experts[1].offs[1], 2);
                params(e, &p, 7);
                dispatch(e, H / 128, slots, 1, 32);
                if grouped_enabled() && rows > 1 && bucket_kind() {
                    e.setComputePipelineState(&self.bucket_group);
                    bind(e, &[&s.ids, &s.table]);
                    params(
                        e,
                        &[slots as u32, 512, BUCKET_STRIDE as u32, (I / 16) as u32, (H / 16) as u32, SPLITS as u32, 0, 0],
                        2,
                    );
                    dispatch(e, 1, 1, 1, 512);
                } else if grouped_enabled() && rows > 1 {
                    // Independent of prepare: reads ids, writes only the table.
                    e.setComputePipelineState(&self.group);
                    bind(e, &[&s.ids, &s.table]);
                    params(e, &[slots as u32, 512, 0, 0], 2);
                    dispatch(e, 1, 1, 1, 512);
                }
            }
            3 if grouped_enabled() && rows > 1 && bucket_kind() => {
                self.projection_bucket(e, &w.experts[0], &s.pg, &s.table, &s.rg, H, I, 8);
                self.projection_bucket(e, &w.experts[1], &s.pu, &s.table, &s.ru, H, I, 8);
            }
            3 if grouped_enabled() && rows > 1 => {
                self.projection_grouped(e, &w.experts[0], &s.pg, &s.table, &s.rg, H, I, slots, rows);
                self.projection_grouped(e, &w.experts[1], &s.pu, &s.table, &s.ru, H, I, slots, rows);
            }
            3 => {
                self.projection(e, &w.experts[0], &s.pg, &s.ids, &s.rg, H, I, slots);
                self.projection(e, &w.experts[1], &s.pu, &s.ids, &s.ru, H, I, slots);
            }
            4 => {
                e.setComputePipelineState(&self.mid);
                bind(
                    e,
                    &[
                        &s.rg,
                        &s.ru,
                        &w.experts[0].svh,
                        &w.experts[1].svh,
                        &w.experts[2].suh,
                        &s.ids,
                        &s.middle,
                    ],
                );
                at(e, &w.experts[0].svh, w.experts[0].offs[2], 2);
                at(e, &w.experts[1].svh, w.experts[1].offs[2], 3);
                at(e, &w.experts[2].suh, w.experts[2].offs[1], 4);
                params(e, &p, 7);
                dispatch(e, I / 128, slots, 1, 32);
            }
            5 if grouped_enabled() && rows > 1 && bucket_kind() => {
                self.projection_bucket(e, &w.experts[2], &s.middle, &s.table, &s.rd, I, H, 32)
            }
            5 if grouped_enabled() && rows > 1 => {
                self.projection_grouped(e, &w.experts[2], &s.middle, &s.table, &s.rd, I, H, slots, rows)
            }
            5 => self.projection(e, &w.experts[2], &s.middle, &s.ids, &s.rd, I, H, slots),
            6 => {
                e.setComputePipelineState(if finish_join() { &self.finish_join } else { &self.finish });
                bind(
                    e,
                    &[
                        &s.rd,
                        &w.experts[2].svh,
                        &s.ids,
                        &s.identity,
                        &s.scores,
                        &s.routed.full,
                        &s.routed.bf,
                    ],
                );
                at(e, &w.experts[2].svh, w.experts[2].offs[2], 1);
                params(e, &p, 7);
                if finish_join() {
                    // the shared down projection and gate (stages 10, 7) completed in earlier groups
                    for (i, b) in [&s.sd.bf, &s.gate.bf, &s.sigmoid, &s.shared, out].into_iter().enumerate() {
                        at(e, b, 0, 8 + i);
                    }
                }
                dispatch(e, H / 128, rows, 1, 320);
            }
            7 => {
                e.setComputePipelineState(&self.shared_gate);
                bind(e, &[x, &w.shared_gate, &s.gate.full, &s.gate.bf]);
                dispatch(e, rows, 1, 1, 32);
                self.dense.encode(
                    e,
                    &w.shared[0],
                    x,
                    &s.sg.full,
                    &s.sg.bf,
                    &s.ssums[0],
                    rows,
                    0,
                    shared.dense_rows(rows),
                )?;
            }
            8 => self.dense.encode(
                e,
                &w.shared[1],
                x,
                &s.su.full,
                &s.su.bf,
                &s.ssums[1],
                rows,
                0,
                shared.dense_rows(rows),
            )?,
            9 => {
                e.setComputePipelineState(&self.activate);
                bind(e, &[&s.sg.bf, &s.su.bf, &s.sa]);
                params(e, &[(rows * I) as u32], 3);
                dispatch(e, (rows * I).div_ceil(256), 1, 1, 256);
            }
            10 => self.dense.encode(
                e,
                &w.shared[2],
                &s.sa,
                &s.sd.full,
                &s.sd.bf,
                &s.ssums[2],
                rows,
                0,
                shared.dense_shape(rows, H),
            )?,
            12 => {
                ensure!(!(grouped_enabled() && rows > 1), "select_prepare needs the identity slot order");
                let row_grouped = select_prepare_kind() == 2;
                e.setComputePipelineState(if row_grouped { &self.select_prepare_rows } else { &self.select_prepare });
                bind(e, &[&s.router.full, &s.router.bf, &s.ids, &s.scores]);
                params(e, &[matches!(score, RoutingArithmetic::SelectedF32) as u32], 4);
                for (i, b) in [x, &w.experts[0].suh, &w.experts[1].suh, &s.identity, &s.pg, &s.pu].into_iter().enumerate() {
                    at(e, b, 0, 5 + i);
                }
                at(e, &w.experts[0].suh, w.experts[0].offs[1], 6);
                at(e, &w.experts[1].suh, w.experts[1].offs[1], 7);
                params(e, &p, 11);
                if row_grouped {
                    dispatch(e, H / 128, rows, 1, 320);
                } else {
                    dispatch(e, H / 128, slots, 1, 32);
                }
            }
            11 if finish_join() => {} // folded into stage 6
            11 => {
                e.setComputePipelineState(&self.join);
                bind(
                    e,
                    &[
                        &s.sd.bf,
                        &s.gate.bf,
                        &s.routed.bf,
                        &s.sigmoid,
                        &s.shared,
                        out,
                    ],
                );
                params(e, &[(rows * H) as u32], 6);
                dispatch(e, (rows * H).div_ceil(256), 1, 1, 256);
            }
            _ => anyhow::bail!("MLP stage"),
        }
        Ok(())
    }
}
const STAGES: [&str; 12] = [
    "mlp.router",
    "mlp.select",
    "mlp.expert_prepare",
    "mlp.expert_gate_up",
    "mlp.expert_mid",
    "mlp.expert_down",
    "mlp.expert_finish",
    "mlp.shared_gate_and_gate_proj",
    "mlp.shared_up_proj",
    "mlp.shared_activate",
    "mlp.shared_down_proj",
    "mlp.join",
];
fn poison16(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(U16_SENTINEL);
    }
}
fn poison32(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            .fill(U32_SENTINEL);
    }
}
fn read16(b: &Buffer, n: usize, fp16: bool) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "MLP read bounds");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        v[n..].iter().all(|&v| v == U16_SENTINEL),
        "MLP BF16/F16 trailing canary"
    );
    ensure!(
        v[..n].iter().all(|&v| if fp16 {
            f16::from_bits(v).is_finite()
        } else {
            bf16::from_bits(v).is_finite()
        }),
        "MLP BF16/F16 finite stage"
    );
    Ok(v[..n].to_vec())
}
fn read_ids(b: &Buffer, n: usize) -> Result<Vec<u32>> {
    ensure!(n <= b.length() / 4, "MLP ID read bounds");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        v[n..].iter().all(|&v| v == U32_SENTINEL),
        "MLP ID trailing canary"
    );
    ensure!(v[..n].iter().all(|&v| v < 512), "MLP expert ID range");
    ensure!(
        v[..n]
            .chunks_exact(TOPK)
            .all(|r| r.iter().enumerate().all(|(i, v)| !r[..i].contains(v))),
        "MLP unique top-k"
    );
    Ok(v[..n].to_vec())
}
fn read32(b: &Buffer, n: usize) -> Result<Vec<f32>> {
    ensure!(n <= b.length() / 4, "MLP FP32 read bounds");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        v[n..].iter().all(|&v| v == U32_SENTINEL),
        "MLP FP32 trailing canary"
    );
    let v: Vec<_> = v[..n].iter().map(|&v| f32::from_bits(v)).collect();
    ensure!(v.iter().all(|v| v.is_finite()), "MLP FP32 finite stage");
    Ok(v)
}
fn half_values(v: Vec<u16>) -> Vec<f32> {
    v.into_iter().map(|v| f16::from_bits(v).to_f32()).collect()
}
fn sum_splits(v: Vec<f32>, n: usize) -> Vec<f32> {
    v.chunks_exact(n * SPLITS)
        .flat_map(|r| (0..n).map(move |i| (0..SPLITS).map(|s| r[s * n + i]).sum()))
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Stages {
    pub router_raw: Vec<f32>,
    pub logits: Vec<u16>,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
    /// prepared gate/up, raw gate/up, middle, raw down, unrounded/rounded routed.
    pub routed: Vec<Vec<f32>>,
    /// gate/up, activation, down, gate logits, gate sigmoid, shared output.
    pub shared: Vec<Vec<u16>>,
    pub output: Vec<u16>,
}
pub struct Run {
    pub stages: Stages,
    pub gpu_span_us: f64,
    pub submit_wait_us: f64,
    pub repetitions: usize,
    pub partial_values: usize,
}
pub struct Engine {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: Ops,
    w: Weights,
    s: Workspace,
    rx: Buffer,
    x: Buffer,
    out: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
impl Engine {
    pub fn new(bank: &FullExpertBank, w: &HostWeights) -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use exclusive guarded runner"
        );
        let device = MTLCreateSystemDefaultDevice().context("Metal device")?;
        let queue = device.newCommandQueue().context("Metal queue")?;
        let ops = Ops::new(&device)?;
        let w = Weights::new(&device, bank, w)?;
        let s = Workspace::new(&device)?;
        let rx = alloc(&device, (MAX_ROWS * H + GUARD) * 2)?;
        let x = alloc(&device, (MAX_ROWS * H + GUARD) * 2)?;
        let out = alloc(&device, (MAX_ROWS * H + GUARD) * 2)?;
        Ok(Self {
            device,
            queue,
            ops,
            w,
            s,
            rx,
            x,
            out,
            _single_owner: PhantomData,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"device":self.device.name().to_string(),"packed_weight_bytes":self.w.bytes(),"workspace_bytes":self.s.bytes(),"input_output_bytes":self.rx.length()+self.x.length()+self.out.length(),"expert_addresses":512,"host_route_handoffs":0,"full_expert_FP32_decode":false})
    }
    pub fn run(
        &mut self,
        rx: &[u16],
        x: &[u16],
        score: RoutingArithmetic,
        shared: SharedPrecision,
        repetitions: usize,
    ) -> Result<Run> {
        ensure!(
            !x.is_empty()
                && x.len() % H == 0
                && x.len() == rx.len()
                && x.len() / H <= MAX_ROWS
                && (1..=8).contains(&repetitions),
            "bounded MLP input/repetitions"
        );
        ensure!(
            rx.iter().chain(x).all(|&v| bf16::from_bits(v).is_finite()),
            "nonfinite MLP input"
        );
        let rows = x.len() / H;
        let slots = rows * TOPK;
        self.s.poison();
        for b in [&self.rx, &self.x, &self.out] {
            poison16(b);
        }
        upload(&self.rx, rx)?;
        upload(&self.x, x)?;
        let command = self.queue.commandBuffer().context("MLP command")?;
        let encoder = command.computeCommandEncoder().context("MLP encoder")?;
        for _ in 0..repetitions {
            self.ops.encode(
                &encoder, &self.w, &self.s, &self.rx, &self.x, &self.out, rows, score, shared,
            )?;
        }
        encoder.endEncoding();
        let start = Instant::now();
        command.commit();
        command.waitUntilCompleted();
        let submit_wait_us = start.elapsed().as_secs_f64() * 1e6;
        ensure!(
            command.status() == MTLCommandBufferStatus::Completed,
            "MLP command failed: {:?}",
            command.error()
        );
        let gpu_span_us = (command.GPUEndTime() - command.GPUStartTime()) * 1e6;
        ensure!(
            gpu_span_us.is_finite() && gpu_span_us > 0.,
            "MLP GPU timestamp"
        );
        ensure!(
            read16(&self.rx, rx.len(), false)? == rx && read16(&self.x, x.len(), false)? == x,
            "MLP immutable inputs"
        );
        let stages = self.s.collect(&self.out, rows)?;
        Ok(Run {
            stages,
            gpu_span_us,
            submit_wait_us,
            repetitions,
            partial_values: slots * (I * 2 + H) * SPLITS,
        })
    }
}
