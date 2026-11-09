//! Internal encode-only dense operations. The caller owns device/queue, buffer
//! lifetime, dependencies and completion; no hidden submit, wait or CPU copy.
use crate::dense::{Affine8, MAX_ROWS};
use anyhow::{Context, Result, ensure};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;
pub(crate) type Obj<T> = Retained<ProtocolObject<T>>;
/// R58: the Metal math mode for every kernel library: Safe (default), or QWEN_R58_FAST_MATH=1 for
/// Fast, =relaxed for Relaxed (a speed screen; numerics need the KL panel).
pub(crate) fn math_mode() -> MTLMathMode {
    match std::env::var("QWEN_R58_FAST_MATH").as_deref() {
        Ok("1") => MTLMathMode::Fast,
        Ok("relaxed") => MTLMathMode::Relaxed,
        _ => MTLMathMode::Safe,
    }
}
/// R56: every buffer barrier goes through here; BARRIERS counts them (per-forward dependency
/// intervals on the critical path).
pub static BARRIERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// R56: dispatches encoded (counted by callers that report them).
/// R56 QWEN_R56_BARRIER_SITES=1: call sites of the barriers, in order (diagnostic).
pub static SITES: std::sync::Mutex<Vec<&'static std::panic::Location<'static>>> = std::sync::Mutex::new(Vec::new());
#[track_caller]
pub(crate) fn barrier(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("QWEN_R56_BARRIER_SITES").as_deref() == Ok("1")) {
        SITES.lock().unwrap().push(std::panic::Location::caller());
    }
    // R58 diagnostic: while SKIP_BARRIERS is set every barrier is skipped (wrong results; bounds
    // what the dependency stalls cost a forward). Only the verify-time probe sets it.
    if SKIP_BARRIERS.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    // R58 diagnostic: while SKIP_SELECTED is set, barriers whose call site ("file:line") contains
    // one of QWEN_R58_SKIP_SITES's comma-separated patterns are skipped (attribution; wrong results)
    if SKIP_SELECTED.load(std::sync::atomic::Ordering::Relaxed) {
        let l = std::panic::Location::caller();
        let site = format!("{}:{}", l.file(), l.line());
        if SKIP_PATTERNS.lock().unwrap().iter().any(|p| site.contains(p.as_str())) {
            SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
}
/// R58 diagnostic switch for the selective skip (QWEN_R58_SKIP_SITES); set only by the
/// verify-time probe after its warm-up forwards. SKIPPED counts the barriers it dropped.
pub static SKIP_SELECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static SKIP_PATTERNS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
pub static SKIPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// R58 diagnostic switch for barrier(): set only by the verify-time probe (QWEN_R58_NO_BARRIERS=1),
/// after its warm-up forwards, so every buffer already holds finite values.
pub static SKIP_BARRIERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) type Buffer = Obj<dyn MTLBuffer>;

#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Qmv4,
    MppBf16,
    MppF16,
    MppF16x2,
    MppPostscale,
    Fragment { splits: usize, residual: bool },
    FragmentPacked { splits: usize, residual: bool },
}
impl Arm {
    pub const ALL: [Self; 5] = [
        Self::Qmv4,
        Self::MppBf16,
        Self::MppF16,
        Self::MppF16x2,
        Self::MppPostscale,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Qmv4 => "qmv_rows_f32_affine",
            Self::MppBf16 => "mpp16_bf16",
            Self::MppF16 => "mpp16_f16",
            Self::MppF16x2 => "mpp16_f16x2",
            Self::MppPostscale => "mpp16_affine_postscale",
            Self::Fragment { splits, residual } => match (splits, residual) {
                (1, false) => "frag32_f16_s1",
                (2, false) => "frag32_f16_s2",
                (4, false) => "frag32_f16_s4",
                (8, false) => "frag32_f16_s8",
                (16, false) => "frag32_f16_s16",
                (32, false) => "frag32_f16_s32",
                (1, true) => "frag32_f16x2_s1",
                (2, true) => "frag32_f16x2_s2",
                (4, true) => "frag32_f16x2_s4",
                (8, true) => "frag32_f16x2_s8",
                (16, true) => "frag32_f16x2_s16",
                (32, true) => "frag32_f16x2_s32",
                _ => "invalid_fragment_splits",
            },
            Self::FragmentPacked { splits, residual } => match (splits, residual) {
                (1, false) => "packed32_f16_s1",
                (2, false) => "packed32_f16_s2",
                (4, false) => "packed32_f16_s4",
                (8, false) => "packed32_f16_s8",
                (16, false) => "packed32_f16_s16",
                (32, false) => "packed32_f16_s32",
                (1, true) => "packed32_f16x2_s1",
                (2, true) => "packed32_f16x2_s2",
                (4, true) => "packed32_f16x2_s4",
                (8, true) => "packed32_f16x2_s8",
                (16, true) => "packed32_f16x2_s16",
                (32, true) => "packed32_f16x2_s32",
                _ => "invalid_packed_splits",
            },
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Qmv4 => 0,
            Self::MppBf16 => 1,
            Self::MppF16 => 2,
            Self::MppF16x2 => 3,
            Self::MppPostscale => 6,
            Self::Fragment {
                residual: false, ..
            } => 8,
            Self::Fragment { residual: true, .. } => 9,
            Self::FragmentPacked {
                residual: false, ..
            } => 11,
            Self::FragmentPacked { residual: true, .. } => 12,
        }
    }
}
impl Arm {
    fn fragment(self) -> Option<(usize, bool)> {
        match self {
            Self::Fragment { splits, .. } => Some((splits, false)),
            Self::FragmentPacked { splits, .. } => Some((splits, true)),
            _ => None,
        }
    }
}
/// R55: a dense projection can be encoded whole, or as its GEMM and its split-K reduce
/// separately so independent projections with private scratch overlap (caller barriers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Phase {
    All,
    Gemm,
    Reduce,
}
fn rmv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_RMV").as_deref() != Ok("0"))
}
fn rmv1_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_RMV1").as_deref() == Ok("1"))
}
/// Reused per operator, with extra K parallelism for narrow outputs. Distinct
/// output slots do not make scratch independent: encode barriers protect reuse.
pub(crate) fn dense_scratch_bytes(k: usize, n: usize, rows: usize) -> usize {
    rows * (k / 64).max(n * if n <= 1024 { 32 } else { 8 }) * 4
}
pub(crate) fn alloc(d: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Buffer> {
    ensure!(n > 0 && n <= 512 << 20, "bounded component allocation");
    d.newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared)
        .context("Metal allocation")
}
pub(crate) fn upload<T: Copy>(b: &ProtocolObject<dyn MTLBuffer>, v: &[T]) -> Result<()> {
    ensure!(std::mem::size_of_val(v) <= b.length(), "Metal upload bound");
    // Caller has exclusive access and completed all previous GPU use.
    unsafe {
        std::ptr::copy_nonoverlapping(
            v.as_ptr().cast::<u8>(),
            b.contents().cast::<u8>().as_ptr(),
            std::mem::size_of_val(v),
        );
    }
    Ok(())
}
pub(crate) struct DenseWeight {
    codes: Buffer,
    scales: Buffer,
    biases: Buffer,
    fragment_codes: Buffer,
    fragment_scales: Buffer,
    fragment_biases: Buffer,
    pub k: usize,
    pub n: usize,
    pub group: usize,
}
impl DenseWeight {
    /// Original [n][k] codes and [n][groups] scales/biases (R55 RMV kernels).
    pub(crate) fn original(&self) -> [&Buffer; 3] {
        [&self.codes, &self.scales, &self.biases]
    }
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, w: &Affine8) -> Result<Self> {
        Self::build(device, w, true)
    }
    /// R58: only the fragment layout (all the FragmentPacked arm reads); the original-layout
    /// buffers are 16-byte placeholders, so the weights are held once. Other arms must not be
    /// used with such a weight.
    pub fn fragment_only(device: &ProtocolObject<dyn MTLDevice>, w: &Affine8) -> Result<Self> {
        Self::build(device, w, false)
    }
    fn build(device: &ProtocolObject<dyn MTLDevice>, w: &Affine8, original: bool) -> Result<Self> {
        w.validate()?;
        let (codes, scales, biases) = if original {
            let codes = alloc(device, w.codes.len())?;
            upload(&codes, &w.codes)?;
            let scales = alloc(device, w.scales.len() * 2)?;
            upload(&scales, &w.scales)?;
            let biases = alloc(device, w.biases.len() * 2)?;
            upload(&biases, &w.biases)?;
            (codes, scales, biases)
        } else {
            (alloc(device, 16)?, alloc(device, 16)?, alloc(device, 16)?)
        };
        // Alternate component representation for same-process A/B controls.
        // No checkpoint copy or quantization; all real values remain unchanged.
        let np = w.n.div_ceil(32) * 32;
        let mut tiled = vec![0u8; np * w.k];
        let mut ts = vec![0u16; np * (w.k / w.group)];
        let mut tb = vec![0u16; ts.len()];
        for n in 0..w.n {
            for k in (0..w.k).step_by(16) {
                let at = ((k / 16) * (np / 32) + n / 32) * 512 + (n % 32) * 16;
                tiled[at..at + 16].copy_from_slice(&w.codes[n * w.k + k..n * w.k + k + 16]);
            }
            for g in 0..w.k / w.group {
                ts[g * np + n] = w.scales[n * (w.k / w.group) + g];
                tb[g * np + n] = w.biases[n * (w.k / w.group) + g];
            }
        }
        let fragment_codes = alloc(device, tiled.len())?;
        upload(&fragment_codes, &tiled)?;
        let fragment_scales = alloc(device, ts.len() * 2)?;
        upload(&fragment_scales, &ts)?;
        let fragment_biases = alloc(device, tb.len() * 2)?;
        upload(&fragment_biases, &tb)?;
        Ok(Self {
            codes,
            scales,
            biases,
            fragment_codes,
            fragment_scales,
            fragment_biases,
            k: w.k,
            n: w.n,
            group: w.group,
        })
    }
    pub fn bytes(&self) -> usize {
        self.original_bytes() + self.fragment_bytes()
    }
    pub fn original_bytes(&self) -> usize {
        self.codes.length() + self.scales.length() + self.biases.length()
    }
    pub fn fragment_bytes(&self) -> usize {
        self.fragment_codes.length() + self.fragment_scales.length() + self.fragment_biases.length()
    }
}
pub(crate) struct DenseOps {
    pipelines: Vec<Obj<dyn MTLComputePipelineState>>,
    rmv: Vec<((usize, usize), Obj<dyn MTLComputePipelineState>)>,
}
impl DenseOps {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        ensure!(
            device.name().to_string().contains("M5"),
            "MPP requires observed M5"
        );
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let lib = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(
                    &[
                        include_str!("../kernels/affine8_dense.metal"),
                        include_str!("../kernels/affine8_fragment.metal"),
                        include_str!("../kernels/affine8_rmv.metal"),
                    ]
                    .concat(),
                ),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("dense MSL: {e}"))?;
        let mut pipelines = vec![];
        for (name, threads) in [
            ("affine8_qmv4", 64),
            ("affine8_mpp_bf16", 256),
            ("affine8_mpp_f16", 256),
            ("affine8_mpp_f16x2", 256),
            ("affine8_qmv1", 64),
            ("affine8_qmv2", 64),
            ("affine8_mpp_postscale", 256),
            ("affine8_xsum", 32),
            ("affine8_frag_f16", 128),
            ("affine8_frag_f16x2", 128),
            ("affine8_frag_reduce", 256),
            ("affine8_packed_f16", 128),
            ("affine8_packed_f16x2", 128),
        ] {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("dense function")?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("dense pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= threads,
                "dense launch geometry"
            );
            ensure!(
                p.staticThreadgroupMemoryLength() <= device.maxThreadgroupMemoryLength(),
                "dense threadgroup memory"
            );
            pipelines.push(p);
        }
        // R55 multi-row matrix-vector kernels (QMV's per-lane arithmetic; bit-identical).
        let mut rmv = vec![];
        for r in [1usize, 2, 4] {
            for c in [2usize, 4, 8] {
                let name = format!("affine8_rmv_r{r}_c{c}_s2");
                let f = lib.newFunctionWithName(&NSString::from_str(&name)).context("rmv function")?;
                let p = device
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(|e| anyhow::anyhow!("dense pipeline {name}: {e}"))?;
                ensure!(p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= 64, "rmv geometry");
                rmv.push(((r, c), p));
            }
        }
        Ok(Self { pipelines, rmv })
    }
    /// R55: the split count when `encode` would write split-K partials for this arm and row
    /// count (a fragment arm with splits > 1 not diverted to RMV), else None.
    pub(crate) fn partial_splits(&self, arm: Arm, rows: usize) -> Option<usize> {
        if rows <= 4 && rmv_enabled() && (matches!(arm, Arm::Qmv4) || (rows == 1 && arm.fragment().is_some() && rmv1_enabled())) {
            return None;
        }
        arm.fragment().map(|f| f.0).filter(|&s| s > 1)
    }
    pub fn threadgroup_bytes(&self) -> Vec<usize> {
        self.pipelines
            .iter()
            .map(|p| p.staticThreadgroupMemoryLength())
            .collect()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &DenseWeight,
        x: &Buffer,
        y: &Buffer,
        ybf: &Buffer,
        sums: &Buffer,
        rows: usize,
        output_offset: usize,
        arm: Arm,
    ) -> Result<()> {
        self.encode_phase(enc, w, x, y, ybf, sums, rows, output_offset, arm, Phase::All)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode_phase(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &DenseWeight,
        x: &Buffer,
        y: &Buffer,
        ybf: &Buffer,
        sums: &Buffer,
        rows: usize,
        output_offset: usize,
        arm: Arm,
        phase: Phase,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && x.length() >= rows * w.k * 2,
            "dense device input geometry"
        );
        let end = output_offset
            .checked_add(rows * w.n)
            .context("dense output overflow")?;
        ensure!(
            end <= y.length() / 4
                && end <= ybf.length() / 2
                && sums.length() >= rows * (w.k / w.group) * 4,
            "dense device output geometry"
        );
        // R55: QMV at 1-4 rows runs the bit-identical RMV kernel (weights read once per
        // pass, more columns per simdgroup). QWEN_R55_RMV=0 restores the original kernel;
        // QWEN_R55_RMV1=1 also moves 1-row fragment (tensor-op) projections onto it.
        if rows <= 4
            && rmv_enabled()
            && (matches!(arm, Arm::Qmv4) || (rows == 1 && arm.fragment().is_some() && rmv1_enabled()))
        {
            if matches!(phase, Phase::Reduce) {
                return Ok(());
            }
            let r = if rows == 1 { 1 } else if rows == 2 { 2 } else { 4 };
            let c = if w.n >= 8192 { 8 } else if w.n >= 4096 { 4 } else { 2 };
            let p = &self.rmv.iter().find(|(k, _)| *k == (r, c)).context("rmv variant")?.1;
            enc.setComputePipelineState(p);
            let params = [w.k as u32, w.n as u32, w.group as u32, rows as u32, 1u32, 0u32];
            unsafe {
                for (i, b) in [x, &w.codes, &w.scales, &w.biases].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(b), 0, i);
                }
                enc.setBuffer_offset_atIndex(Some(y), output_offset * 4, 4);
                enc.setBuffer_offset_atIndex(Some(ybf), output_offset * 2, 5);
                enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 24, 6);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: w.n.div_ceil(c * 2), height: rows.div_ceil(r), depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            return Ok(());
        }
        let nv = if rows == 1 {
            1
        } else if rows == 2 {
            2
        } else {
            4
        };
        let pidx = if matches!(arm, Arm::Qmv4) && nv < 4 {
            if nv == 1 { 4 } else { 5 }
        } else {
            arm.index()
        };
        let splits = if let Some((splits, _)) = arm.fragment() {
            ensure!(
                [1, 2, 4, 8, 16, 32].contains(&splits) && (splits <= 8 || w.n <= 1024),
                "fragment split bound"
            );
            ensure!(
                sums.length() >= rows * w.n * splits * 4,
                "fragment scratch bound"
            );
            splits
        } else {
            1
        };
        if matches!(phase, Phase::Reduce) && !(arm.fragment().is_some() && splits > 1) {
            return Ok(());
        }
        enc.setComputePipelineState(&self.pipelines[pidx]);
        let params = [
            w.k as u32,
            w.n as u32,
            w.group as u32,
            rows as u32,
            splits as u32,
            matches!(arm, Arm::FragmentPacked { .. }) as u32,
        ];
        // All buffers are from the caller's device. The caller retains them
        // through completion and inserts barriers before dependent consumers.
        unsafe {
            let wb = if matches!(arm, Arm::FragmentPacked { .. }) {
                [&w.fragment_codes, &w.fragment_scales, &w.fragment_biases]
            } else {
                [&w.codes, &w.scales, &w.biases]
            };
            for (i, b) in [x, wb[0], wb[1], wb[2]].into_iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            enc.setBuffer_offset_atIndex(Some(y), output_offset * 4, 4);
            enc.setBuffer_offset_atIndex(Some(ybf), output_offset * 2, 5);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 24, 6);
            enc.setBuffer_offset_atIndex(Some(sums), 0, 7);
        }
        if matches!(arm, Arm::MppPostscale) && !matches!(phase, Phase::Reduce) {
            enc.setComputePipelineState(&self.pipelines[7]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: w.k / w.group,
                    height: rows,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            crate::metal_ops::barrier(&enc);
            enc.setComputePipelineState(&self.pipelines[pidx]);
        }
        let (gx, gy, threads) = if arm.fragment().is_some() {
            (w.n.div_ceil(128), rows.div_ceil(32), 128)
        } else if matches!(arm, Arm::Qmv4) {
            (w.n.div_ceil(4), rows.div_ceil(nv), 64)
        } else {
            (w.n.div_ceil(32), rows.div_ceil(16), 256)
        };
        if !matches!(phase, Phase::Reduce) {
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: gx,
                height: gy,
                depth: splits,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        }
        if matches!(phase, Phase::Gemm) {
            return Ok(());
        }
        if arm.fragment().is_some() && splits > 1 {
            if matches!(phase, Phase::All) {
                crate::metal_ops::barrier(&enc);
            }
            enc.setComputePipelineState(&self.pipelines[10]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (rows * w.n).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            // A later independent projection may reuse this same scratch before
            // the caller's next barrier. Complete its read before that write.
            if matches!(phase, Phase::All) {
                crate::metal_ops::barrier(&enc);
            }
        }
        Ok(())
    }
}
