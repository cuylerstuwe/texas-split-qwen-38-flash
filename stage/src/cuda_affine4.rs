//! Native packed affine4 projection. No expanded matrix or host dequantization.
use crate::{
    affine4::{Affine4, MAX_ROWS},
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4_split::Split,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "f11059e7ff71b6aeca519ecab03de39a81922276e6db89dff5e65569a1e7a915";
const GUARD: usize = 256;
const NAMES: [&str; 4] = [
    "affine4_direct1",
    "affine4_factored1",
    "affine4_factored2",
    "affine4_factored4",
];
/// R60: the canonical target's affine8 projections reuse the same ABI and launch geometry
/// (kernels/affine8.cu): arms 0/1 one row per block, arm 2 two rows, arm 3 four rows.
const NAMES8: [&str; 4] = ["affine8_f1", "affine8_f1", "affine8_f2", "affine8_f4"];
/// R60 S4: affine8 QMV v2 (16-byte loads, a chunk's loads issued before its arithmetic), chosen
/// by row count: 1, 2, 3-4 and 5+ rows per block. QWEN_AFFINE8_V2=0 keeps the v1 arms.
const NAMES8_V2: [&str; 4] = ["affine8_v2_1", "affine8_v2_2", "affine8_v2_4", "affine8_v2_8"];
pub fn affine8_v2() -> bool {
    std::env::var("QWEN_AFFINE8_V2").as_deref() != Ok("0")
}
fn v2_slot(rows: usize) -> (usize, usize) {
    match rows {
        1 => (0, 1),
        2 => (1, 2),
        3 | 4 => (2, 4),
        _ => (3, 8),
    }
}
/// R60 S4: tensor-core affine8 (kernels/affine8.cu affine8_mma_*) from this many rows up
/// (QWEN_AFFINE8_MMA_MIN, default 3; QWEN_AFFINE8_MMA=0 turns it off).
pub fn affine8_mma_min() -> usize {
    if std::env::var("QWEN_AFFINE8_MMA").as_deref() == Ok("0") {
        return usize::MAX;
    }
    std::env::var("QWEN_AFFINE8_MMA_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}
/// R60 S4: the affine8 kernel set past v1: the v2 QMV (1/2/4/8 rows per block) and the
/// tensor-core arms (8 and 16 rows per block row).
pub struct A8Kernels<'m, 'd> {
    v2: [Kernel<'m, 'd>; 4],
    /// [t1, t2] x [4, 8, 16 warps]
    mma: Option<[[Kernel<'m, 'd>; 3]; 2]>,
    /// R60 S4 (prefill v2): the 64-token tile for chunk-sized row counts
    mma8: Option<Kernel<'m, 'd>>,
}
impl<'m, 'd> A8Kernels<'m, 'd> {
    /// R60 S4 (prefill v2): (kernel, grid, block) for chunk-sized row counts (> 16): the 64-token
    /// tensor tile, one block row per 64 tokens. Smaller counts take `select`.
    fn select_rows(&self, n: usize, k: usize, rows: usize, group: usize) -> Result<(&Kernel<'m, 'd>, [u32; 3], [u32; 3])> {
        if rows <= MAX_ROWS {
            return Ok(self.select(n, k, rows, group));
        }
        let m = self.mma8.as_ref().ok_or_else(|| anyhow::anyhow!("affine8 64-token arm not loaded"))?;
        ensure!(group % 64 == 0, "affine8 prefill arm needs 64-code groups");
        Ok((m, [n.div_ceil(16) as u32, rows.div_ceil(64) as u32, 1], [128, 1, 1]))
    }
    /// (kernel, grid, block) for an n-output projection of `rows` rows.
    fn select(&self, n: usize, k: usize, rows: usize, group: usize) -> (&Kernel<'m, 'd>, [u32; 3], [u32; 3]) {
        if let Some(m) = &self.mma {
            if rows >= affine8_mma_min() && group % 64 == 0 {
                let (i, t) = if rows <= 8 { (0, 1) } else { (1, 2) };
                // enough warps to fill the card: narrow outputs split K across 8 or 16 warps per block
                let blocks = n.div_ceil(16) * rows.div_ceil(8 * t);
                // K caps the split: each warp should get at least one U=4 group of 64-code blocks
                let cap = k / 256;
                let (j, warps) = if blocks < 256 && cap >= 16 { (2, 16) } else if blocks < 512 && cap >= 8 { (1, 8) } else { (0, 4) };
                return (&m[i][j], [n.div_ceil(16) as u32, rows.div_ceil(8 * t) as u32, 1], [32 * warps, 1, 1]);
            }
        }
        let (i, per) = v2_slot(rows);
        (&self.v2[i], [n.div_ceil(4) as u32, rows.div_ceil(per) as u32, 1], [64, 1, 1])
    }
}
fn v2_kernels<'m, 'd>(m8: &'m Module<'d>, k: usize) -> Result<Option<A8Kernels<'m, 'd>>> {
    Ok(if affine8_v2() && k % 16 == 0 {
        Some(A8Kernels {
            v2: [m8.kernel(NAMES8_V2[0])?, m8.kernel(NAMES8_V2[1])?, m8.kernel(NAMES8_V2[2])?, m8.kernel(NAMES8_V2[3])?],
            mma: if affine8_mma_min() < usize::MAX {
                Some([
                    [m8.kernel("affine8_mma_t1")?, m8.kernel("affine8_mma_t1_w8")?, m8.kernel("affine8_mma_t1_w16")?],
                    [m8.kernel("affine8_mma_t2")?, m8.kernel("affine8_mma_t2_w8")?, m8.kernel("affine8_mma_t2_w16")?],
                ])
            } else {
                None
            },
            mma8: Some(m8.kernel("affine8_mma_t8")?),
        })
    } else {
        None
    })
}
/// R60 S4: the four-projection affine8 launch: v1 (one row per block row) and, when enabled,
/// the v2 bodies with 1/4/8 rows per block row.
pub struct Multi8<'m, 'd> {
    v1: Kernel<'m, 'd>,
    v2: Option<[Kernel<'m, 'd>; 3]>,
    mma: Option<[Kernel<'m, 'd>; 2]>,
    /// R60 S4 (prefill v2): affine8_mma_multi4_t8 for chunk-sized row counts
    mma8: Option<Kernel<'m, 'd>>,
}
impl<'m, 'd> Multi8<'m, 'd> {
    pub fn load(m8: &'m Module<'d>) -> Result<Self> {
        Ok(Self {
            v1: m8.kernel("affine8_multi4")?,
            v2: if affine8_v2() {
                Some([m8.kernel("affine8_v2_multi4_1")?, m8.kernel("affine8_v2_multi4_4")?, m8.kernel("affine8_v2_multi4_8")?])
            } else {
                None
            },
            mma: if affine8_v2() && affine8_mma_min() < usize::MAX {
                Some([m8.kernel("affine8_mma_multi4_t1")?, m8.kernel("affine8_mma_multi4_t2")?])
            } else {
                None
            },
            mma8: Some(m8.kernel("affine8_mma_multi4_t8")?),
        })
    }
}
/// R38 (Opus, direct): the unchanged affine4.cu kernels plus bit-identical
/// load-ahead variants, compiled together. Loaded only with QWEN_AFFINE4_R38=1.
pub const R38_SHA: &str = "9746180bd08bdfcd32640ac230009b83bafc2b1a91e064c2f0b2783322cdd55b";
pub const AHEAD_KERNEL: &str = "affine4_ahead_direct1_d2";
/// Record-time R38 switch. When set (and the R38 module is loaded), arm 0 records
/// the load-ahead kernel, which has the same arguments, grid and arithmetic.
pub static AHEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn r38_enabled() -> bool {
    std::env::var("QWEN_AFFINE4_R38").as_deref() == Ok("1")
}
pub fn ahead_selected() -> bool {
    AHEAD.load(std::sync::atomic::Ordering::Relaxed)
}
/// R56: row capacity of each projection's own x/y/ybf scratch (default MAX_ROWS = 64). The
/// resident model's layer inputs are bounded to 16 rows; record paths bound-check every buffer.
pub static SCRATCH_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(MAX_ROWS);
fn scratch_rows() -> usize {
    SCRATCH_ROWS.load(std::sync::atomic::Ordering::Relaxed).clamp(1, MAX_ROWS)
}
pub static PDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn r39_enabled() -> bool {
    std::env::var("QWEN_R39_AFFINE_PDL").as_deref() == Ok("1")
}
pub fn pdl_affine_selected() -> bool { pdl_selected() && std::env::var("QWEN_R39_AFFINE_OFF").as_deref()!=Ok("1") }
pub fn pdl_selected() -> bool { PDL.load(std::sync::atomic::Ordering::Relaxed) }
/// R52: one-row K=6144 projections preload all weights before the wait (8 columns
/// per 128-thread block); candidate arm with affine PDL only. Loads affine4_r52.
pub fn r52_outp_enabled() -> bool { r39_enabled() && std::env::var("QWEN_R52_OUTP").as_deref() == Ok("1") }
/// R53: deeper load-ahead for the early-launch projections (candidate arm):
/// QWEN_R53_OUTD = K-step depth for the one-row K=6144 output projections
/// (2 = R39 default; 4, 6, 8, 12; "2p", "8p", "12p" add a bulk L2 self-prefetch
/// of the block's rows at entry), QWEN_R53_M4D = depth for the GDN fused input
/// projection (qkv/z/a/b; 3, 5, 10, "2p", "3p", "5p"), QWEN_R53_M4DA the same for
/// attention's (q/k/v/indexer). Same arithmetic order; loads affine4_r53.
fn r53_knob(name: &str, allowed: &[&str]) -> Option<String> {
    std::env::var(name).ok().filter(|v| allowed.contains(&v.as_str()))
}
pub fn r53_outd() -> Option<String> { r53_knob("QWEN_R53_OUTD", &["4", "6", "8", "12", "2p", "8p", "12p"]) }
pub fn r53_m4d() -> Option<String> { r53_knob("QWEN_R53_M4D", &["3", "5", "10", "2p", "3p", "5p"]) }
pub fn r53_m4da() -> Option<String> { r53_knob("QWEN_R53_M4DA", &["3", "5", "10", "2p", "3p", "5p"]) }
/// QWEN_R53_M4LATE=1: the fused input projections (both layer kinds) launched
/// normally instead of early (same kernel body under a non-r39pdl_ name).
pub fn r53_m4late() -> bool { std::env::var("QWEN_R53_M4LATE").as_deref() == Ok("1") }
pub fn r53_affine_enabled() -> bool {
    r39_enabled() && (r53_outd().is_some() || r53_m4d().is_some() || r53_m4da().is_some() || r53_m4late())
}
/// R58: the multi-row MMA projection module (kernels/r58_affine4_rows.cu; drafter batched prefill)
/// when QWEN_R58_DRAFTER_ROWS=1 and its cubin is present.
pub fn rows_module(d: &Device) -> Result<Option<Module<'_>>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/r58_affine4_rows.sm120.cubin");
    if !path.is_file() || std::env::var("QWEN_R58_DRAFTER_ROWS").as_deref() != Ok("1") {
        return Ok(None);
    }
    let pin: Value = serde_json::from_str(include_str!("../pins/R58-AFFINE-ROWS-CUDA-BUILD.json"))?;
    Ok(Some(d.module(&path, pin["cubin_sha256"].as_str().unwrap())?))
}
/// R58: while set, layers record their dense projections with arm 4 (the multi-row MMA kernel).
/// Only the drafter's batched prefill sets it, around its recording (RowsArm guard).
static ROWS_ARM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) fn rows_arm() -> bool {
    ROWS_ARM.load(std::sync::atomic::Ordering::Relaxed)
}
pub(crate) struct RowsArm;
impl RowsArm {
    pub(crate) fn set() -> Self {
        ROWS_ARM.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}
impl Drop for RowsArm {
    fn drop(&mut self) {
        ROWS_ARM.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if r53_affine_enabled() {
        ensure!(r38_enabled(), "R53 requires R38 module selection");
        let pin: Value = serde_json::from_str(include_str!("../pins/R53-AFFINE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_r53.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r52_outp_enabled() {
        ensure!(r38_enabled(), "R52 requires R38 module selection");
        let pin: Value = serde_json::from_str(include_str!("../pins/R52-AFFINE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_r52.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r39_enabled() {
        ensure!(r38_enabled(), "R39 requires R38 module selection");
        let pin: Value = serde_json::from_str(include_str!("../pins/R39-AFFINE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_r39.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // Only initialized u16/f32 arrays on this little-endian host.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
/// R38: four same-input, same-K single-row projections in one launch, each
/// written to its usual caller-owned FP32/BF16 outputs. Bit-identical per block
/// to four affine4_ahead_direct1_d2 launches.
/// R60: four affine8 projections of one input in one launch (kernels/affine8.cu affine8_multi4);
/// each output equals its own arm-0 (affine8_f1) launch.
pub(crate) fn record_multi4_affine8<'a, 'm, 'd>(
    multi: &'a Multi8<'m, 'd>,
    linears: [&'a Linear<'m, 'd>; 4],
    x: &'a Buffer<'d>,
    outs: [(&'a Buffer<'d>, &'a Buffer<'d>); 4],
    rows: usize,
) -> Result<PreparedCall<'a, 'm, 'd>> {
    let k = linears[0].k;
    ensure!(
        linears.iter().all(|l| l.bits8 && l.k == k) && (1..=MAX_ROWS).contains(&rows) && x.byte_len() >= rows * k * 2,
        "affine8 multi4 geometry"
    );
    let tensor = multi.mma.is_some() && rows >= affine8_mma_min() && linears.iter().all(|l| l.group % 64 == 0);
    let (kernel, per) = match (&multi.v2, &multi.mma) {
        (_, Some(m)) if tensor => if rows <= 8 { (&m[0], 8) } else { (&m[1], 16) },
        (Some(v2), _) if k % 16 == 0 => match rows {
            1 => (&v2[0], 1),
            2..=4 => (&v2[1], 4),
            _ => (&v2[2], 8),
        },
        _ => (&multi.v1, 1),
    };
    let mut args = vec![Arg::Buffer(x)];
    let mut blocks = 0usize;
    for (l, (y, yb)) in linears.iter().zip(outs) {
        ensure!(y.byte_len() >= rows * l.n * 4 && yb.byte_len() >= rows * l.n * 2 && !x.aliases(y) && !x.aliases(yb), "affine8 multi4 outputs");
        args.extend([Arg::Buffer(&l.w), Arg::Buffer(&l.s), Arg::Buffer(&l.b), Arg::Buffer(y), Arg::Buffer(yb)]);
        blocks += if tensor { l.n.div_ceil(16) } else { l.n.div_ceil(4) };
    }
    for l in &linears {
        args.push(Arg::U32(l.n as u32));
    }
    for l in &linears {
        args.push(Arg::U32(l.group as u32));
    }
    args.extend([Arg::U32(k as u32), Arg::U32(rows as u32)]);
    // SAFETY: fixed kernel ABI; buffers checked above and borrowed for the call's lifetime.
    unsafe { kernel.prepare(args, [blocks as u32, rows.div_ceil(per) as u32, 1], [if tensor { 128 } else { 64 }, 1, 1]) }
}
/// R60 S4 (prefill v2): the four-projection affine8 launch for chunk-sized row counts (> 16) on
/// the 64-token tensor tile; smaller counts take record_multi4_affine8.
pub(crate) fn record_multi4_affine8_rows<'a, 'm, 'd>(
    multi: &'a Multi8<'m, 'd>,
    linears: [&'a Linear<'m, 'd>; 4],
    x: &'a Buffer<'d>,
    outs: [(&'a Buffer<'d>, &'a Buffer<'d>); 4],
    rows: usize,
) -> Result<PreparedCall<'a, 'm, 'd>> {
    if rows <= MAX_ROWS {
        return record_multi4_affine8(multi, linears, x, outs, rows);
    }
    let k = linears[0].k;
    let kernel = multi.mma8.as_ref().ok_or_else(|| anyhow::anyhow!("affine8 multi4 64-token arm not loaded"))?;
    ensure!(
        linears.iter().all(|l| l.bits8 && l.k == k && l.group % 64 == 0) && x.byte_len() >= rows * k * 2,
        "affine8 multi4 prefill geometry"
    );
    let mut args = vec![Arg::Buffer(x)];
    let mut blocks = 0usize;
    for (l, (y, yb)) in linears.iter().zip(outs) {
        ensure!(y.byte_len() >= rows * l.n * 4 && yb.byte_len() >= rows * l.n * 2 && !x.aliases(y) && !x.aliases(yb), "affine8 multi4 prefill outputs");
        args.extend([Arg::Buffer(&l.w), Arg::Buffer(&l.s), Arg::Buffer(&l.b), Arg::Buffer(y), Arg::Buffer(yb)]);
        blocks += l.n.div_ceil(16);
    }
    for l in &linears {
        args.push(Arg::U32(l.n as u32));
    }
    for l in &linears {
        args.push(Arg::U32(l.group as u32));
    }
    args.extend([Arg::U32(k as u32), Arg::U32(rows as u32)]);
    // SAFETY: fixed kernel ABI (affine8_mma_multi4_t8); buffers checked above.
    unsafe { kernel.prepare(args, [blocks as u32, rows.div_ceil(64) as u32, 1], [128, 1, 1]) }
}
pub(crate) fn record_multi4<'a, 'm, 'd>(
    linears: [&'a Linear<'m, 'd>; 4],
    x: &'a Buffer<'d>,
    outs: [(&'a Buffer<'d>, &'a Buffer<'d>); 4],
    rows: usize,
) -> Result<PreparedCall<'a, 'm, 'd>> {
    let kernel = (if pdl_affine_selected() { &linears[0].pdl_multi } else { &linears[0].multi })
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("R38 multi-projection kernel not loaded"))?;
    let (k, group) = (linears[0].k, linears[0].group);
    ensure!(
        (1..=MAX_ROWS).contains(&rows)
            && x.byte_len() >= rows * k * 2
            && linears.iter().all(|l| l.k == k && l.group == group),
        "R38 multi-projection geometry"
    );
    let mut args = vec![Arg::Buffer(x)];
    let mut blocks = 0;
    for (l, (y, yb)) in linears.iter().zip(outs) {
        ensure!(
            y.byte_len() >= rows * l.n * 4
                && yb.byte_len() >= rows * l.n * 2
                && !x.aliases(y)
                && !x.aliases(yb)
                && !y.aliases(yb),
            "R38 multi-projection outputs"
        );
        args.extend([
            Arg::Buffer(&l.w),
            Arg::Buffer(&l.s),
            Arg::Buffer(&l.b),
            Arg::Buffer(y),
            Arg::Buffer(yb),
        ]);
        blocks += l.n.div_ceil(4);
    }
    for l in &linears {
        args.push(Arg::U32(l.n as u32));
    }
    args.extend([Arg::U32(k as u32), Arg::U32(group as u32), Arg::U32(rows as u32)]);
    unsafe { kernel.prepare(args, [blocks as u32, rows as u32, 1], [64, 1, 1]) }
}
/// A single packed weight allocation with caller-owned activation/workspace.
/// The original standalone Linear below is retained as a qualified control.
pub struct Packed<'m, 'd> {
    kernels: [Kernel<'m, 'd>; 4],
    ahead: Option<Kernel<'m, 'd>>,
    pdl: Option<Kernel<'m, 'd>>,
    /// R52: deeper load-ahead (QWEN_R52_HEAD=d4|d8) for the vocabulary-sized head.
    r52_head: Option<Kernel<'m, 'd>>,
    pub(crate) w: Buffer<'d>,
    pub(crate) s: Buffer<'d>,
    pub(crate) b: Buffer<'d>,
    k: usize,
    n: usize,
    group: usize,
    /// R60: canonical affine8 weights (byte codes) through kernels/affine8.cu; no fast paths
    bits8: bool,
    /// R60 S4: affine8 QMV v2 kernels (1/2/4/8 rows per block), when enabled.
    v2: Option<A8Kernels<'m, 'd>>,
}
impl<'m, 'd> Packed<'m, 'd> {
    pub fn new(d: &'d Device, m: &'m Module<'d>, a: &Affine4) -> Result<Self> {
        a.validate()?;
        Self::build(d, m, a.k, a.n, a.group, &a.codes, &a.scales, &a.biases, false)
    }
    /// R60: the canonical target's affine8 projection (`m8`: the affine8 module).
    pub fn new_affine8(d: &'d Device, m8: &'m Module<'d>, a: &crate::dense::Affine8) -> Result<Self> {
        a.validate()?;
        Self::build(d, m8, a.k, a.n, a.group, &a.codes, &a.scales, &a.biases, true)
    }
    /// R64: the canonical 8-bit vocabulary head (hidden x 248,320), past `Affine8::validate`'s dense-size
    /// cap: assembled from validated row tiles, so only the joined geometry is checked here.
    pub fn new_affine8_head(d: &'d Device, m8: &'m Module<'d>, a: &crate::dense::Affine8) -> Result<Self> {
        let groups = a.k / a.group;
        ensure!(
            a.k == crate::hc::H && a.n == crate::embedding::VOCAB && [64, 128].contains(&a.group) && a.k % a.group == 0
                && a.codes.len() == a.k * a.n && a.scales.len() == a.n * groups && a.biases.len() == a.n * groups,
            "affine8 head geometry"
        );
        Self::build(d, m8, a.k, a.n, a.group, &a.codes, &a.scales, &a.biases, true)
    }
    pub fn vocabulary(
        d: &'d Device,
        m: &'m Module<'d>,
        v: &crate::lowbit_io::Vocab,
    ) -> Result<Self> {
        v.validate()?;
        Self::build(
            d,
            m,
            crate::hc::H,
            crate::lowbit_io::VOCAB,
            crate::lowbit_io::GROUP,
            &v.codes,
            &v.scales,
            &v.biases,
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn build(
        d: &'d Device,
        m: &'m Module<'d>,
        k: usize,
        n: usize,
        group: usize,
        w: &[u8],
        s: &[u16],
        b: &[u16],
        bits8: bool,
    ) -> Result<Self> {
        let names = if bits8 { NAMES8 } else { NAMES };
        let upload = |raw: &[u8]| -> Result<Buffer<'d>> {
            let dst = d.alloc(raw.len())?;
            dst.upload(raw)?;
            let mut check = vec![0; raw.len()];
            dst.download(&mut check)?;
            ensure!(check == raw, "packed affine4 complete weight readback");
            Ok(dst)
        };
        Ok(Self {
            kernels: [
                m.kernel(names[0])?,
                m.kernel(names[1])?,
                m.kernel(names[2])?,
                m.kernel(names[3])?,
            ],
            pdl: if r39_enabled() && !bits8 { Some(m.kernel("r39pdl_affine4_direct1_d2")?) } else { None },
            ahead: if r38_enabled() && !bits8 {
                Some(m.kernel(AHEAD_KERNEL)?)
            } else {
                None
            },
            r52_head: match (r39_enabled() && n > 100_000 && !bits8, std::env::var("QWEN_R52_HEAD").as_deref()) {
                (true, Ok("d4")) => Some(m.kernel("r39pdl_affine4_direct1_d4")?),
                (true, Ok("d8")) => Some(m.kernel("r39pdl_affine4_direct1_d8")?),
                _ => None,
            },
            w: upload(w)?,
            s: upload(bytes(s))?,
            b: upload(bytes(b))?,
            k,
            n,
            group,
            bits8,
            v2: if bits8 { v2_kernels(m, k)? } else { None },
        })
    }
    pub fn description(&self) -> Value {
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":if self.pdl.is_some() { serde_json::from_str::<Value>(include_str!("../pins/R39-AFFINE-CUDA-BUILD.json")).unwrap()["cubin_sha256"].as_str().unwrap().to_owned() } else if self.ahead.is_some() {R38_SHA.to_owned()} else {SHA.to_owned()},
            "r38_ahead_loaded":self.ahead.is_some(),"k":self.k,"n":self.n,"group":self.group,
            "packed_weight_bytes":self.w.byte_len()+self.s.byte_len()+self.b.byte_len(),"activation_scratch_bytes":0,
            "weight_readback_checked":true,"whole_matrix_expanded":false,
            "kernels":self.kernels.iter().map(|k|k.description()).collect::<Vec<_>>()})
    }
    pub(crate) fn group_size(&self) -> usize {
        self.group
    }
    /// R60 S4 (prefill v2): any row count; above 16 rows the 64-token tensor tile (affine8 only).
    pub fn record_rows<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        yb: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        if rows <= MAX_ROWS {
            return self.record_into(x, y, yb, rows, 0);
        }
        let a8 = self
            .v2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("chunk-row projection needs the affine8 v2/MMA kernels"))?;
        ensure!(
            self.bits8
                && x.byte_len() >= rows * self.k * 2
                && y.byte_len() >= rows * self.n * 4
                && yb.byte_len() >= rows * self.n * 2
                && !x.aliases(y)
                && !x.aliases(yb)
                && !y.aliases(yb),
            "chunk-row projection buffers"
        );
        let (kernel, grid, block) = a8.select_rows(self.n, self.k, rows, self.group)?;
        // SAFETY: fixed kernel ABI (affine8_mma_t8 = affine8_f1's); buffers checked above.
        unsafe {
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.w),
                    Arg::Buffer(&self.s),
                    Arg::Buffer(&self.b),
                    Arg::Buffer(y),
                    Arg::Buffer(yb),
                    Arg::U32(self.k as u32),
                    Arg::U32(self.n as u32),
                    Arg::U32(self.group as u32),
                    Arg::U32(rows as u32),
                ],
                grid,
                block,
            )
        }
    }
    pub fn record_into<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        yb: &'a Buffer<'d>,
        rows: usize,
        arm: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && arm < 4,
            "packed projection row/arm bounds"
        );
        ensure!(
            x.byte_len() >= rows * self.k * 2
                && y.byte_len() >= rows * self.n * 4
                && yb.byte_len() >= rows * self.n * 2
                && !x.aliases(y)
                && !x.aliases(yb)
                && !y.aliases(yb)
                && [&self.w, &self.s, &self.b]
                    .iter()
                    .all(|b| !b.aliases(y) && !b.aliases(yb)),
            "packed projection buffers/alias"
        );
        unsafe {
            let v2 = self.v2.as_ref().map(|a| a.select(self.n, self.k, rows, self.group));
            let kernel = if let Some((k, _, _)) = v2 {
                k
            } else if arm == 0 && ahead_selected() && !self.bits8 {
                if pdl_affine_selected() && self.r52_head.is_some() { self.r52_head.as_ref() } else
                { if pdl_affine_selected() { self.pdl.as_ref() } else { self.ahead.as_ref() } }
                    .ok_or_else(|| anyhow::anyhow!("R38 load-ahead kernel not loaded"))?
            } else {
                &self.kernels[arm]
            };
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.w),
                    Arg::Buffer(&self.s),
                    Arg::Buffer(&self.b),
                    Arg::Buffer(y),
                    Arg::Buffer(yb),
                    Arg::U32(self.k as u32),
                    Arg::U32(self.n as u32),
                    Arg::U32(self.group as u32),
                    Arg::U32(rows as u32),
                ],
                v2.map_or([self.n.div_ceil(4) as u32, rows.div_ceil([1, 1, 2, 4][arm]) as u32, 1], |(_, g, _)| g),
                v2.map_or([64, 1, 1], |(_, _, b)| b),
            )
        }
    }
}
pub struct Linear<'m, 'd> {
    d: &'d Device,
    kernels: [Kernel<'m, 'd>; 4],
    ahead: Option<Kernel<'m, 'd>>,
    pdl: Option<Kernel<'m, 'd>>,
    multi: Option<Kernel<'m, 'd>>,
    pdl_multi: Option<Kernel<'m, 'd>>,
    r52_all: Option<Kernel<'m, 'd>>,
    /// R58: the multi-row MMA kernels (4 and 8 warps), when enabled.
    rows_k: Option<[Kernel<'m, 'd>; 2]>,
    /// R60 S4: affine8 QMV v2 kernels (1/2/4/8 rows per block), when enabled.
    v2: Option<A8Kernels<'m, 'd>>,
    w: Buffer<'d>,
    s: Buffer<'d>,
    b: Buffer<'d>,
    x: Buffer<'d>,
    y: Buffer<'d>,
    ybf: Buffer<'d>,
    k: usize,
    n: usize,
    group: usize,
    rows: Cell<usize>,
    split: Option<Split<'m, 'd>>,
    /// R60: canonical affine8 weights through kernels/affine8.cu (no split, no fast paths)
    bits8: bool,
}
pub struct Run {
    pub fp32: Vec<f32>,
    pub bf16: Vec<f32>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
    pub partial_values_checked: usize,
}
impl<'m, 'd> Linear<'m, 'd> {
    pub fn new(d: &'d Device, m: &'m Module<'d>, a: &Affine4) -> Result<Self> {
        Self::build(d, m, None, a)
    }
    /// R60: the canonical target's affine8 projection (`m8`: the affine8 module). Same ABI and
    /// arms 0-3 as affine4; no split-K, load-ahead, PDL or multi-row MMA variants.
    pub fn new_affine8(d: &'d Device, m8: &'m Module<'d>, a: &crate::dense::Affine8) -> Result<Self> {
        Self::build_affine8(d, m8, a, false)
    }
    /// R60: an affine8 projection that also owns split-K partial scratch (tall-skinny shapes such
    /// as the HC down projection, K=10240 -> 320).
    pub fn new_affine8_split(d: &'d Device, m8: &'m Module<'d>, a: &crate::dense::Affine8) -> Result<Self> {
        Self::build_affine8(d, m8, a, true)
    }
    fn build_affine8(d: &'d Device, m8: &'m Module<'d>, a: &crate::dense::Affine8, split: bool) -> Result<Self> {
        a.validate()?;
        let upload = |raw: &[u8]| -> Result<Buffer<'d>> {
            let b = d.alloc(raw.len())?;
            b.upload(raw)?;
            let mut check = vec![0; raw.len()];
            b.download(&mut check)?;
            ensure!(check == raw, "affine8 weight upload readback");
            Ok(b)
        };
        Ok(Self {
            d,
            kernels: [m8.kernel(NAMES8[0])?, m8.kernel(NAMES8[1])?, m8.kernel(NAMES8[2])?, m8.kernel(NAMES8[3])?],
            ahead: None,
            pdl: None,
            multi: None,
            pdl_multi: None,
            r52_all: None,
            rows_k: None,
            v2: v2_kernels(m8, a.k)?,
            w: upload(&a.codes)?,
            s: upload(bytes(&a.scales))?,
            b: upload(bytes(&a.biases))?,
            x: d.alloc(scratch_rows() * a.k * 2)?,
            y: d.alloc(scratch_rows() * a.n * 4 + GUARD)?,
            ybf: d.alloc(scratch_rows() * a.n * 2 + GUARD)?,
            k: a.k,
            n: a.n,
            group: a.group,
            rows: Cell::new(0),
            split: if split { Some(Split::new8(d, m8, a.n)?) } else { None },
            bits8: true,
        })
    }
    /// Output width.
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn has_split(&self) -> bool {
        self.split.is_some()
    }
    pub fn is_affine8(&self) -> bool {
        self.bits8
    }
    pub fn new_split(
        d: &'d Device,
        m: &'m Module<'d>,
        split: &'m Module<'d>,
        a: &Affine4,
    ) -> Result<Self> {
        Self::build(d, m, Some(split), a)
    }
    fn build(
        d: &'d Device,
        m: &'m Module<'d>,
        split: Option<&'m Module<'d>>,
        a: &Affine4,
    ) -> Result<Self> {
        a.validate()?;
        let upload = |raw: &[u8]| -> Result<Buffer<'d>> {
            let b = d.alloc(raw.len())?;
            b.upload(raw)?;
            let mut check = vec![0; raw.len()];
            b.download(&mut check)?;
            ensure!(check == raw, "affine4 weight upload readback");
            Ok(b)
        };
        Ok(Self {
            d,
            kernels: [
                m.kernel(NAMES[0])?,
                m.kernel(NAMES[1])?,
                m.kernel(NAMES[2])?,
                m.kernel(NAMES[3])?,
            ],
            pdl: if r39_enabled() {
                Some(m.kernel(&match (a.k, r53_outd()) {
                    (6144, Some(depth)) => format!("r39pdl_affine4_direct1_d{depth}"),
                    _ => "r39pdl_affine4_direct1_d2".to_string(),
                })?)
                // (the first linear of a fused group selects the multi4 kernel below:
                // GDN qkv has n=10240, attention q has n=12288)
            } else {
                None
            },
            ahead: if r38_enabled() {
                Some(m.kernel(AHEAD_KERNEL)?)
            } else {
                None
            },
            pdl_multi: if r39_enabled() {
                Some(m.kernel(&match (a.n, r53_m4d(), r53_m4da(), r53_m4late()) {
                    (10240, Some(depth), _, true) | (12288, _, Some(depth), true) => format!("affine4_multi4_d{depth}_late"),
                    (_, _, _, true) => "affine4_multi4_d2_late".to_string(),
                    (10240, Some(depth), _, false) | (12288, _, Some(depth), false) => format!("r39pdl_affine4_multi4_d{depth}"),
                    _ => "r39pdl_affine4_multi4_d2".to_string(),
                })?)
            } else {
                None
            },
            multi: if r38_enabled() {
                Some(m.kernel("affine4_ahead_multi4_d2")?)
            } else {
                None
            },
            r52_all: if r52_outp_enabled() && a.k == 6144 { Some(m.kernel("r39pdl_affine4_direct8_all24")?) } else { None },
            rows_k: None,
            v2: None,
            w: upload(&a.codes)?,
            s: upload(bytes(&a.scales))?,
            b: upload(bytes(&a.biases))?,
            x: d.alloc(scratch_rows() * a.k * 2)?,
            y: d.alloc(scratch_rows() * a.n * 4 + GUARD)?,
            ybf: d.alloc(scratch_rows() * a.n * 2 + GUARD)?,
            k: a.k,
            n: a.n,
            group: a.group,
            rows: Cell::new(0),
            split: split.map(|m| Split::new(d, m, a.n)).transpose()?,
            bits8: false,
        })
    }
    pub fn description(&self) -> Value {
        let allocated: usize = [&self.w, &self.s, &self.b, &self.x, &self.y, &self.ybf]
            .iter()
            .map(|b| b.byte_len())
            .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":if self.pdl.is_some() { serde_json::from_str::<Value>(include_str!("../pins/R39-AFFINE-CUDA-BUILD.json")).unwrap()["cubin_sha256"].as_str().unwrap().to_owned() } else if self.ahead.is_some() {R38_SHA.to_owned()} else {SHA.to_owned()},
            "r38_ahead_loaded":self.ahead.is_some(),"k":self.k,"n":self.n,"group":self.group,
            "weight_format":"packed affine4 U32, BF16 scales/biases","weight_readback_checked":true,
            "allocated_bytes_excluding_optional_split":allocated,"split":self.split.as_ref().map(|s|s.description()),
            "kernels":self.kernels.iter().map(|k|k.description()).collect::<Vec<_>>(),
            "output":"FP32 diagnostic and BF16, both stores included","whole_matrix_expanded":false})
    }
    /// R58: enable arm 4, the multi-row MMA projection (needs k % 16 == 0 and groups of 16k).
    pub(crate) fn enable_rows(&mut self, m: &'m Module<'d>) -> Result<()> {
        ensure!(self.k % 16 == 0 && self.group % 16 == 0 && self.k % self.group == 0, "multi-row MMA geometry");
        self.rows_k = Some([m.kernel("r58_affine4_rows_w4")?, m.kernel("r58_affine4_rows_w8")?]);
        Ok(())
    }
    pub fn prepare(&self, x: &[u16]) -> Result<()> {
        self.rows.set(0);
        ensure!(
            !x.is_empty()
                && x.len() % self.k == 0
                && x.len() / self.k <= MAX_ROWS
                && x.len() * 2 <= self.x.byte_len()
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "affine4 input shape/finite"
        );
        self.x.upload(bytes(x))?;
        let mut check = vec![0; x.len() * 2];
        self.x.download(&mut check)?;
        ensure!(check == bytes(x), "affine4 input readback");
        self.rows.set(x.len() / self.k);
        Ok(())
    }
    /// R60 S4 (prefill v2): any row count; above 16 rows the 64-token tensor tile (affine8 only).
    pub fn record_rows<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        yb: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        if rows <= MAX_ROWS {
            return self.record_into(x, y, yb, rows, 0);
        }
        let a8 = self
            .v2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("chunk-row projection needs the affine8 v2/MMA kernels"))?;
        ensure!(
            self.bits8
                && x.byte_len() >= rows * self.k * 2
                && y.byte_len() >= rows * self.n * 4
                && yb.byte_len() >= rows * self.n * 2
                && !x.aliases(y)
                && !x.aliases(yb)
                && !y.aliases(yb),
            "chunk-row projection buffers"
        );
        let (kernel, grid, block) = a8.select_rows(self.n, self.k, rows, self.group)?;
        // SAFETY: fixed kernel ABI (affine8_mma_t8 = affine8_f1's); buffers checked above.
        unsafe {
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.w),
                    Arg::Buffer(&self.s),
                    Arg::Buffer(&self.b),
                    Arg::Buffer(y),
                    Arg::Buffer(yb),
                    Arg::U32(self.k as u32),
                    Arg::U32(self.n as u32),
                    Arg::U32(self.group as u32),
                    Arg::U32(rows as u32),
                ],
                grid,
                block,
            )
        }
    }
    pub fn record_into<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        ybf: &'a Buffer<'d>,
        rows: usize,
        arm: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && (arm < 4 || (arm == 4 && self.rows_k.is_some())),
            "affine4 record shape/arm"
        );
        ensure!(
            x.byte_len() >= rows * self.k * 2
                && y.byte_len() >= rows * self.n * 4
                && ybf.byte_len() >= rows * self.n * 2
                && !x.aliases(y)
                && !x.aliases(ybf)
                && !y.aliases(ybf)
                && [&self.w, &self.s, &self.b]
                    .iter()
                    .all(|b| !b.aliases(y) && !b.aliases(ybf)),
            "affine4 record buffers"
        );
        if arm == 4 {
            // R58: 16 output channels x 16 rows per block; 8 warps when there are few blocks.
            let ks = self.rows_k.as_ref().unwrap();
            let tiles = self.n.div_ceil(16);
            let (kernel, warps) = if tiles < 340 { (&ks[1], 8u32) } else { (&ks[0], 4u32) };
            return unsafe {
                kernel.prepare(
                    vec![
                        Arg::Buffer(x), Arg::Buffer(&self.w), Arg::Buffer(&self.s), Arg::Buffer(&self.b),
                        Arg::Buffer(y), Arg::Buffer(ybf), Arg::U32(self.k as u32), Arg::U32(self.n as u32),
                        Arg::U32(self.group as u32), Arg::U32(rows as u32),
                    ],
                    [tiles as u32, rows.div_ceil(16) as u32, 1],
                    [32 * warps, 1, 1],
                )
            };
        }
        let width = [1, 1, 2, 4][arm];
        if let (true, Some(all)) = (arm == 0 && !self.bits8 && ahead_selected() && pdl_affine_selected() && rows == 1 && self.k == 6144, &self.r52_all) {
            return unsafe {
                all.prepare(
                    vec![
                        Arg::Buffer(x), Arg::Buffer(&self.w), Arg::Buffer(&self.s), Arg::Buffer(&self.b),
                        Arg::Buffer(y), Arg::Buffer(ybf), Arg::U32(self.k as u32), Arg::U32(self.n as u32),
                        Arg::U32(self.group as u32), Arg::U32(rows as u32),
                    ],
                    [self.n.div_ceil(8) as u32, 1, 1],
                    [128, 1, 1],
                )
            };
        }
        // Checked dimensions, disjoint buffers and pinned ABI; k%64 means each
        // packed U32 stays inside a single quantization group and matrix row.
        unsafe {
            let v2 = self.v2.as_ref().map(|a| a.select(self.n, self.k, rows, self.group));
            let kernel = if let Some((k, _, _)) = v2 {
                k
            } else if arm == 0 && ahead_selected() && !self.bits8 {
                (if pdl_affine_selected() { &self.pdl } else { &self.ahead })
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("R38 load-ahead kernel not loaded"))?
            } else {
                &self.kernels[arm]
            };
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.w),
                    Arg::Buffer(&self.s),
                    Arg::Buffer(&self.b),
                    Arg::Buffer(y),
                    Arg::Buffer(ybf),
                    Arg::U32(self.k as u32),
                    Arg::U32(self.n as u32),
                    Arg::U32(self.group as u32),
                    Arg::U32(rows as u32),
                ],
                v2.map_or([self.n.div_ceil(4) as u32, rows.div_ceil(width) as u32, 1], |(_, g, _)| g),
                v2.map_or([64, 1, 1], |(_, _, b)| b),
            )
        }
    }
    pub fn plan(
        &self,
        rows: usize,
        arm: usize,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows == self.rows.get() && rows > 0 && (1..=32).contains(&repetitions),
            "affine4 plan bounds"
        );
        let make = || self.record_into(&self.x, &self.y, &self.ybf, rows, arm);
        let exec = if graph {
            Executor::Graph(
                self.d
                    .capture((0..repetitions).map(|_| make()).collect::<Result<_>>()?)?,
            )
        } else {
            Executor::Prepared(vec![make()?])
        };
        Ok(Plan {
            owner: self,
            exec,
            rows,
            repetitions,
            splits: 0,
        })
    }
    pub fn record_split_into<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        ybf: &'a Buffer<'d>,
        rows: usize,
        width: usize,
        splits: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        let split = self
            .split
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("split module not loaded"))?;
        split.record(
            x, &self.w, &self.s, &self.b, y, ybf, self.k, self.group, rows, width, splits,
        )
    }
    /// R38: packed weight buffers and quantization group, for fused consumers.
    pub(crate) fn packed_weights(&self) -> (&Buffer<'d>, &Buffer<'d>, &Buffer<'d>, usize) {
        (&self.w, &self.s, &self.b, self.group)
    }
    /// Composition probes check the same private scratch as standalone plans.
    pub(crate) fn split_partials(&self) -> Result<&Buffer<'d>> {
        Ok(self
            .split
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("split module not loaded"))?
            .partials())
    }
    pub fn poison_split(&self) -> Result<()> {
        self.split
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("split module not loaded"))?
            .poison()
    }
    pub fn check_split(&self, rows: usize, splits: usize) -> Result<usize> {
        self.split
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("split module not loaded"))?
            .check(rows, splits)
    }
    pub fn plan_split(
        &self,
        rows: usize,
        width: usize,
        splits: usize,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows > 0 && rows == self.rows.get() && (1..=32).contains(&repetitions),
            "affine4 split plan bounds"
        );
        let make = || self.record_split_into(&self.x, &self.y, &self.ybf, rows, width, splits);
        let exec = if graph {
            let mut calls = Vec::new();
            for _ in 0..repetitions {
                calls.extend(make()?);
            }
            Executor::Graph(self.d.capture(calls)?)
        } else {
            Executor::Prepared(make()?)
        };
        Ok(Plan {
            owner: self,
            exec,
            rows,
            repetitions,
            splits,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Linear<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    repetitions: usize,
    splits: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        ensure!(a.rows.get() == self.rows, "affine4 prepared shape changed");
        a.y.upload(&vec![0xff; a.y.byte_len()])?;
        a.ybf.upload(&vec![0xff; a.ybf.byte_len()])?;
        if self.splits > 0 {
            a.split.as_ref().unwrap().poison()?;
        }
        a.d.sync()?;
        let t = Instant::now();
        match &self.exec {
            Executor::Prepared(calls) => {
                for _ in 0..self.repetitions {
                    for c in calls {
                        c.enqueue()?;
                    }
                }
            }
            Executor::Graph(g) => g.launch()?,
        }
        let enqueue_us = t.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let mut f = vec![0; a.y.byte_len()];
        let mut b = vec![0; a.ybf.byte_len()];
        a.y.download(&mut f)?;
        a.ybf.download(&mut b)?;
        let count = self.rows * a.n;
        ensure!(
            f[count * 4..]
                .iter()
                .chain(&b[count * 2..])
                .all(|&v| v == 0xff),
            "affine4 output/tail canary"
        );
        let fp32: Vec<_> = f[..count * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let bf16: Vec<_> = b[..count * 2]
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        ensure!(
            fp32.iter().chain(&bf16).all(|v| v.is_finite()),
            "affine4 nonfinite/unwritten output"
        );
        Ok(Run {
            fp32,
            bf16,
            total_us,
            enqueue_us,
            wait_us,
            partial_values_checked: if self.splits > 0 {
                a.split.as_ref().unwrap().check(self.rows, self.splits)?
            } else {
                0
            },
        })
    }
}
