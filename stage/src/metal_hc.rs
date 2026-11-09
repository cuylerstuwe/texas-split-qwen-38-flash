//! Native HC -> GDN -> HC read on one queue, with a separate/fused residual
//! write control. No router, experts, QSA or full model is implied.
use crate::{
    gdn,
    hc::*,
    metal_gdn::{BlockArm, MetalGdn},
    metal_ops::{
        Arm as DenseArm, Buffer, DenseOps, DenseWeight, Obj, Phase, alloc, dense_scratch_bytes, upload,
    },
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const GUARD: usize = 64;
const SENTINEL: u16 = 0x7fc1;
const FSENTINEL: u32 = 0x7fc12345;

#[derive(Clone, Copy, Debug)]
pub enum Precision {
    Qmv,
    F16,
    F16x2,
    PackedF16,
    PackedF16x2,
}
impl Precision {
    pub const ALL: [Self; 5] = [
        Self::Qmv,
        Self::F16,
        Self::F16x2,
        Self::PackedF16,
        Self::PackedF16x2,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Qmv => "qmv",
            Self::F16 => "f16",
            Self::F16x2 => "f16x2",
            Self::PackedF16 => "packed_f16",
            Self::PackedF16x2 => "packed_f16x2",
        }
    }
    fn dense(self, rows: usize, n: usize) -> DenseArm {
        if matches!(self, Self::PackedF16 | Self::PackedF16x2) {
            return DenseArm::FragmentPacked {
                splits: if n == R { 32 } else { 2 },
                residual: matches!(self, Self::PackedF16x2),
            };
        }
        if rows == 1 {
            return DenseArm::Qmv4;
        }
        match self {
            Self::Qmv => DenseArm::Qmv4,
            Self::F16 => DenseArm::MppF16,
            Self::F16x2 => DenseArm::MppF16x2,
            Self::PackedF16 | Self::PackedF16x2 => unreachable!(),
        }
    }
    fn gdn(self) -> BlockArm {
        match self {
            Self::Qmv => BlockArm::Fused,
            Self::F16 => BlockArm::F16,
            Self::F16x2 => BlockArm::F16x2,
            Self::PackedF16 => BlockArm::PackedF16,
            Self::PackedF16x2 => BlockArm::PackedF16x2,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub precision: Precision,
    pub fused_write: bool,
}
impl Arm {
    pub fn all() -> Vec<Self> {
        Precision::ALL
            .into_iter()
            .flat_map(|precision| {
                [false, true].map(|fused_write| Self {
                    precision,
                    fused_write,
                })
            })
            .collect()
    }
    pub fn name(self) -> String {
        format!(
            "{}_{}",
            self.precision.name(),
            if self.fused_write {
                "write_norm"
            } else {
                "separate_write"
            }
        )
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
    fn read(&self, rows: usize) -> Result<Vec<u16>> {
        read32(&self.full, rows * self.n)?;
        read16(&self.bf, rows * self.n)
    }
}
fn poison16(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(SENTINEL);
    }
}
fn poison32(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            .fill(FSENTINEL);
    }
}
fn read16(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "HC read bound");
    let all =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        all[n..].iter().all(|&v| v == SENTINEL),
        "HC trailing BF16 canary"
    );
    ensure!(
        all[..n].iter().all(|&v| bf16::from_bits(v).is_finite()),
        "HC unwritten/nonfinite BF16"
    );
    Ok(all[..n].to_vec())
}
fn read32(b: &Buffer, n: usize) -> Result<Vec<f32>> {
    ensure!(n <= b.length() / 4, "HC read bound");
    let all =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        all[n..].iter().all(|&v| v == FSENTINEL),
        "HC trailing FP32 canary"
    );
    let v: Vec<_> = all[..n].iter().map(|&v| f32::from_bits(v)).collect();
    ensure!(
        v.iter().all(|v| v.is_finite()),
        "HC unwritten/nonfinite FP32"
    );
    Ok(v)
}
/// R56 (QWEN_R56_HC_UPMIX=1, opt-in until requalified): the HC read's up GEMV and mix in one
/// kernel (hc_up_mix_r*), RMV arithmetic, no split-K partials: 4 dependent stages instead of 5.
/// Measured -0.45 ms per 1-row forward, -0.15 ms at 2 rows; slower beyond (rows <= 2 only).
fn hc_upmix() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R56_HC_UPMIX").as_deref() == Ok("1"))
}
/// R57/R58 (default on; QWEN_R57_HC_ROWS=0 turns it off): the HC read as hc_norm + Sushi's fused
/// down (+act, +inject) and up (+mix) kernels, 3 dispatches instead of 5, for pages of up to 8 rows
/// (one row group); 16-row prefill chunks keep the old path. r57-hcrows-model: GPU per forward
/// 21.9 -> 19.3 ms at 1 row, -2.0 ms at 4, level at 8 (+2.5 ms at 16, hence not used there).
/// Quality: on the 24-prompt panel at 8-row chunks KL .004189 -> .004410 (paired CI
/// +.000055..+.000382; r57-panel-hc8, -hc8off), inside the user's band (Sushi native .004581,
/// MTPLX .023); user direction 2026-11-07 is to take it. Page shapes stay coherent: target logits
/// are bitwise identical across 1/2/4/8-row pages with either setting (r58-rowcheck-hcon/-hcoff).
fn hc_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R57_HC_ROWS").as_deref() != Ok("0"))
}
fn hc_fuse() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_HC_FUSE").as_deref() != Ok("0"))
}
pub(crate) struct Hc {
    down: DenseWeight,
    up: DenseWeight,
    norm: Buffer,
    iw: Buffer,
    normalized: Buffer,
    parts: Buffer,
    dp: Pair,
    act: Buffer,
    upv: Pair,
    mixed: Buffer,
    inject: Buffer,
    sums: Buffer,
}
impl Hc {
    /// R58 prefill: (down, up, norm, input-mix weight) for the large-row path.
    pub(crate) fn prefill_weights(&self) -> (&DenseWeight, &DenseWeight, &Buffer, &Buffer) {
        (&self.down, &self.up, &self.norm, &self.iw)
    }
    pub(crate) fn mixed(&self) -> &Buffer {
        &self.mixed
    }
    pub(crate) fn inject(&self) -> &Buffer {
        &self.inject
    }
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>, w: &Weights) -> Result<Self> {
        w.validate()?;
        let norm = alloc(d, F * 2)?;
        upload(&norm, &w.norm)?;
        let iw = alloc(d, HC * F * 2)?;
        upload(&iw, &w.inject)?;
        Ok(Self {
            down: DenseWeight::new(d, &w.down)?,
            up: DenseWeight::new(d, &w.up)?,
            norm,
            iw,
            normalized: alloc(d, (MAX_ROWS * F + GUARD) * 2)?,
            parts: alloc(d, (MAX_ROWS * HC * HC + GUARD) * 4)?,
            dp: Pair::new(d, R)?,
            act: alloc(d, (MAX_ROWS * R + GUARD) * 2)?,
            upv: Pair::new(d, F)?,
            mixed: alloc(d, (MAX_ROWS * H + GUARD) * 2)?,
            inject: alloc(d, (MAX_ROWS * HC + GUARD) * 2)?,
            sums: alloc(
                d,
                dense_scratch_bytes(F, R, MAX_ROWS).max(dense_scratch_bytes(R, F, MAX_ROWS)),
            )?,
        })
    }
    pub(crate) fn poison(&self) {
        for b in [&self.normalized, &self.act, &self.mixed, &self.inject] {
            poison16(b);
        }
        poison32(&self.parts);
        self.dp.poison();
        self.upv.poison();
    }
    pub(crate) fn collect(&self, rows: usize) -> Result<Read> {
        read32(&self.parts, rows * HC * HC)?;
        Ok(Read {
            normalized: read16(&self.normalized, rows * F)?,
            down: self.dp.read(rows)?,
            activated: read16(&self.act, rows * R)?,
            up: self.upv.read(rows)?,
            mixed: read16(&self.mixed, rows * H)?,
            inject: read16(&self.inject, rows * HC)?,
        })
    }
    pub(crate) fn weight_bytes(&self) -> usize {
        self.down.bytes() + self.up.bytes() + self.norm.length() + self.iw.length()
    }
    pub(crate) fn bytes(&self) -> usize {
        self.weight_bytes()
            + [
                &self.normalized,
                &self.parts,
                &self.dp.full,
                &self.dp.bf,
                &self.act,
                &self.upv.full,
                &self.upv.bf,
                &self.mixed,
                &self.inject,
                &self.sums,
            ]
            .iter()
            .map(|b| b.length())
            .sum::<usize>()
    }
}
pub(crate) struct HcOps {
    pipes: Vec<Obj<dyn MTLComputePipelineState>>,
    dense: DenseOps,
}
fn grid(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, x: usize, y: usize, threads: usize) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
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
}
fn bind(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &[&Buffer],
    rows: usize,
    write: bool,
) {
    let p = [rows as u32, write as u32];
    unsafe {
        for (i, b) in buf.iter().enumerate() {
            enc.setBuffer_offset_atIndex(Some(b), 0, i);
        }
        enc.setBytes_length_atIndex(NonNull::from(&p).cast(), 8, 8);
    }
}
impl HcOps {
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let opt = MTLCompileOptions::new();
        opt.setMathMode(crate::metal_ops::math_mode());
        let lib = d
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../kernels/hc.metal")),
                Some(&opt),
            )
            .map_err(|e| anyhow::anyhow!("HC MSL: {e}"))?;
        let mut pipes = vec![];
        let mut names: Vec<String> = ["hc_norm", "hc_activate", "hc_mix", "hc_write", "hc_act_reduce", "hc_mix_reduce", "hc_up_mix_r1", "hc_up_mix_r2", "hc_up_mix_r4", "hc_up_mix_r8"]
            .iter().map(|s| s.to_string()).collect();
        // R57: Sushi's fused HC down (10..=17) and up+mix (18..=25) kernels, ROWS 1..=8
        names.extend((1..=8).map(|r| format!("hc_down_rows_r{r}")));
        names.extend((1..=8).map(|r| format!("hc_up_rows_r{r}")));
        for name in names.iter().map(String::as_str) {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("HC function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("HC pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= 256
                    && p.staticThreadgroupMemoryLength() <= d.maxThreadgroupMemoryLength(),
                "HC pipeline geometry"
            );
            pipes.push(p);
        }
        Ok(Self {
            pipes,
            dense: DenseOps::new(d)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        h: &Hc,
        x: &Buffer,
        rows: usize,
        arm: Precision,
        pending: Option<(&Buffer, &Buffer, &Buffer)>,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && x.length() >= rows * F * 2,
            "HC encode input"
        );
        let (block, inj, next) = pending.unwrap_or((x, x, x));
        if pending.is_some() {
            ensure!(
                block.length() >= rows * H * 2
                    && inj.length() >= rows * HC * 2
                    && next.length() >= rows * F * 2,
                "HC pending write buffers"
            );
        }
        enc.setComputePipelineState(&self.pipes[0]);
        bind(
            enc,
            &[x, &h.norm, &h.iw, block, inj, &h.normalized, &h.parts, next],
            rows,
            pending.is_some(),
        );
        grid(enc, HC, rows, 256);
        crate::metal_ops::barrier(&enc);
        if hc_rows() && rows <= 8 {
            ensure!(h.down.k == F && h.down.n == R && h.up.k == R && h.up.n == F
                && F % h.down.group == 0 && R % h.up.group == 0 && h.down.group % 4 == 0 && h.up.group % 4 == 0, "HC rows geometry");
            let groups = rows.div_ceil(8);
            let per = rows.div_ceil(groups);
            let [dc, ds, db] = h.down.original();
            let [uc, us, ub] = h.up.original();
            let set = |bufs: &[&Buffer], group: usize| unsafe {
                for (i, b) in bufs.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(b), 0, i);
                }
                let prm = [rows as u32, group as u32];
                enc.setBytes_length_atIndex(NonNull::from(&prm).cast(), 8, 8);
            };
            enc.setComputePipelineState(&self.pipes[10 + per - 1]);
            set(&[&h.normalized, dc, ds, db, &h.parts, &h.act, &h.inject], h.down.group);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: R + HC, depth: groups },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            crate::metal_ops::barrier(&enc);
            enc.setComputePipelineState(&self.pipes[18 + per - 1]);
            set(&[&h.act, uc, us, ub, &h.normalized, &h.mixed], h.up.group);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: H / 8, depth: groups },
                MTLSize { width: 32, height: 8, depth: 1 },
            );
            crate::metal_ops::barrier(&enc);
            return Ok(());
        }
        // R55 (QWEN_R55_HC_FUSE=1): each split-K reduction is folded into its consumer.
        let (da, ua) = (arm.dense(rows, h.down.n), arm.dense(rows, h.up.n));
        if hc_fuse() {
            if let (Some(ds), Some(us)) = (self.dense.partial_splits(da, rows), self.dense.partial_splits(ua, rows)) {
                let fp = [rows as u32, ds as u32];
                self.dense.encode_phase(enc, &h.down, &h.normalized, &h.dp.full, &h.dp.bf, &h.sums, rows, 0, da, Phase::Gemm)?;
                crate::metal_ops::barrier(&enc);
                enc.setComputePipelineState(&self.pipes[4]);
                unsafe {
                    for (i, b) in [&h.sums, &h.parts, &h.act, &h.inject, &h.dp.full, &h.dp.bf].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(b), 0, i);
                    }
                    enc.setBytes_length_atIndex(NonNull::from(&fp).cast(), 8, 8);
                }
                grid(enc, (rows * (R + HC)).div_ceil(256), 1, 256);
                crate::metal_ops::barrier(&enc);
                // RMV arithmetic is compute-bound past 2 rows (r56-hc-upmix-ab: 4 rows +1.5 ms,
                // 8 rows +7.2 ms per forward), so the fused kernel serves 1-2 rows only.
                if hc_upmix() && rows <= 2 {
                    ensure!(h.up.k == R && h.up.n == F && R % h.up.group == 0 && h.up.group % 8 == 0, "HC up geometry");
                    let (pipe, rr) = match rows {
                        1 => (6, 1),
                        2 => (7, 2),
                        3 | 4 => (8, 4),
                        _ => (9, 8),
                    };
                    let [codes, scales, biases] = h.up.original();
                    let up = [rows as u32, h.up.group as u32];
                    enc.setComputePipelineState(&self.pipes[pipe]);
                    unsafe {
                        for (i, b) in [&h.act, codes, scales, biases, &h.normalized, &h.mixed].into_iter().enumerate() {
                            enc.setBuffer_offset_atIndex(Some(b), 0, i);
                        }
                        enc.setBytes_length_atIndex(NonNull::from(&up).cast(), 8, 8);
                    }
                    grid(enc, H / 8, rows.div_ceil(rr), 256);
                    crate::metal_ops::barrier(&enc);
                    return Ok(());
                }
                self.dense.encode_phase(enc, &h.up, &h.act, &h.upv.full, &h.upv.bf, &h.sums, rows, 0, ua, Phase::Gemm)?;
                crate::metal_ops::barrier(&enc);
                let fp = [rows as u32, us as u32];
                enc.setComputePipelineState(&self.pipes[5]);
                unsafe {
                    for (i, b) in [&h.sums, &h.normalized, &h.mixed, &h.upv.full, &h.upv.bf].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(b), 0, i);
                    }
                    enc.setBytes_length_atIndex(NonNull::from(&fp).cast(), 8, 8);
                }
                grid(enc, (rows * H).div_ceil(256), 1, 256);
                crate::metal_ops::barrier(&enc);
                return Ok(());
            }
        }
        self.dense.encode(
            enc,
            &h.down,
            &h.normalized,
            &h.dp.full,
            &h.dp.bf,
            &h.sums,
            rows,
            0,
            arm.dense(rows, h.down.n),
        )?;
        crate::metal_ops::barrier(&enc);
        enc.setComputePipelineState(&self.pipes[1]);
        bind(enc, &[&h.dp.bf, &h.parts, &h.act, &h.inject], rows, false);
        grid(enc, (rows * (R + HC)).div_ceil(256), 1, 256);
        crate::metal_ops::barrier(&enc);
        self.dense.encode(
            enc,
            &h.up,
            &h.act,
            &h.upv.full,
            &h.upv.bf,
            &h.sums,
            rows,
            0,
            arm.dense(rows, h.up.n),
        )?;
        crate::metal_ops::barrier(&enc);
        enc.setComputePipelineState(&self.pipes[2]);
        bind(enc, &[&h.upv.bf, &h.normalized, &h.mixed], rows, false);
        grid(enc, (rows * H).div_ceil(256), 1, 256);
        crate::metal_ops::barrier(&enc);
        Ok(())
    }
    pub(crate) fn write(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &Buffer,
        block: &Buffer,
        inj: &Buffer,
        next: &Buffer,
        rows: usize,
    ) {
        enc.setComputePipelineState(&self.pipes[3]);
        bind(enc, &[x, block, inj, next], rows, false);
        grid(enc, (rows * F).div_ceil(256), 1, 256);
        crate::metal_ops::barrier(&enc);
    }
}
pub struct MetalFront {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: HcOps,
    hcs: [Hc; 2],
    gdn: MetalGdn,
    x: Buffer,
    stream: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
pub struct Run {
    pub pre: Read,
    pub gdn: crate::metal_gdn::Run,
    pub stream: Vec<u16>,
    pub post: Read,
    pub stage_us: f64,
    pub encode_queue_wait_us: f64,
    pub gpu_us: f64,
}
impl MetalFront {
    pub fn new(pre: &Weights, gw: &gdn::Weights, post: &Weights) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("HC/GDN Metal device")?;
        let queue = device.newCommandQueue().context("HC/GDN queue")?;
        Self::new_on(pre, gw, post, device, queue)
    }
    pub(crate) fn new_on(
        pre: &Weights,
        gw: &gdn::Weights,
        post: &Weights,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        Ok(Self {
            ops: HcOps::new(&device)?,
            hcs: [Hc::new(&device, pre)?, Hc::new(&device, post)?],
            gdn: MetalGdn::new_on(gw, device.clone(), queue.clone())?,
            x: alloc(&device, (MAX_ROWS * F + GUARD) * 2)?,
            stream: alloc(&device, (MAX_ROWS * F + GUARD) * 2)?,
            device,
            queue,
            _single_owner: PhantomData,
        })
    }
    pub fn description(&self) -> serde_json::Value {
        serde_json::json!({"device":self.device.name().to_string(),"gdn":self.gdn.description(),
            "hc_weight_bytes":self.hcs.iter().map(Hc::weight_bytes).sum::<usize>(),
            "hc_total_bytes":self.hcs.iter().map(Hc::bytes).sum::<usize>()+self.x.length()+self.stream.length(),
            "hc_threadgroup_bytes":self.ops.pipes.iter().map(|p|p.staticThreadgroupMemoryLength()).collect::<Vec<_>>(),
            "scope":"attention HC read -> GDN -> attention HC write -> MLP HC read; no MoE/router/shared expert",
            "host_intermediate_transfers":0,"safe_math":true,"state_snapshots":"full FP32 per position, compact replay not implemented"})
    }
    pub(crate) fn prepare(&mut self, x: &[u16], init: &gdn::Initial) -> Result<usize> {
        ensure!(
            !x.is_empty() && x.len() % F == 0 && x.len() / F <= MAX_ROWS,
            "HC/GDN geometry"
        );
        ensure!(
            x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "HC/GDN nonfinite input"
        );
        init.validate()?;
        for h in &self.hcs {
            h.poison();
        }
        poison16(&self.x);
        poison16(&self.stream);
        upload(&self.x, x)?;
        self.gdn.prepare(init)?;
        Ok(x.len() / F)
    }
    /// Encode the entire front without a queue submission or host handoff.
    pub(crate) fn encode(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        self.encode_from(enc, &self.x, rows, arm)
    }
    /// R55 resident model: the same front reading the layer input from `x` (the previous
    /// layer's output buffer) instead of this front's own staging buffer.
    pub(crate) fn encode_from(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &Buffer,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        ensure!(x.length() >= rows * F * 2, "HC/GDN front input");
        for _ in 0..1 + crate::metal_profile::dup(0) as usize {
            self.ops
                .read(enc, &self.hcs[0], x, rows, arm.precision, None)?;
        }
        self.gdn
            .encode(enc, &self.hcs[0].mixed, rows, arm.precision.gdn())?;
        if arm.fused_write {
            for _ in 0..1 + crate::metal_profile::dup(5) as usize {
                self.ops.read(
                    enc,
                    &self.hcs[1],
                    x,
                    rows,
                    arm.precision,
                    Some((self.gdn.output_buffer(), &self.hcs[0].inject, &self.stream)),
                )?;
            }
        } else {
            self.ops.write(
                enc,
                x,
                self.gdn.output_buffer(),
                &self.hcs[0].inject,
                &self.stream,
                rows,
            );
            self.ops
                .read(enc, &self.hcs[1], &self.stream, rows, arm.precision, None)?;
        }
        Ok(())
    }
    /// R55 resident model: load this layer's recurrent state (once, at reset).
    pub(crate) fn reset_state(&mut self, init: &gdn::Initial) -> Result<()> {
        self.gdn.prepare(init)
    }
    /// R55 resident model: keep the state after `accepted` of the last forward's rows.
    pub(crate) fn commit(&self, accepted: usize, rows: usize) -> Result<()> {
        self.gdn.commit(accepted, rows)
    }
    pub(crate) fn commit_split(&self, accepted: usize, rows: usize) -> Result<Option<(usize, usize, usize)>> {
        self.gdn.commit_split(accepted, rows)
    }
    pub(crate) fn set_zero_copy(&self, on: bool) {
        self.gdn.set_zero_copy(on)
    }
    pub(crate) fn mlp_input(&self) -> &Buffer {
        &self.hcs[1].mixed
    }
    /// R58 prefill: (attention-side HC, MLP-side HC, GDN engine, post-attention stream buffer).
    pub(crate) fn prefill_parts(&self) -> (&Hc, &Hc, &MetalGdn, &Buffer) {
        (&self.hcs[0], &self.hcs[1], &self.gdn, &self.stream)
    }
    /// R55 stage clock version of encode (same order, one command buffer per stage).
    pub(crate) fn encode_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        self.encode_profiled_from(clock, &self.x, rows, arm)
    }
    /// R58: encode_profiled reading the layer input from `x` (the resident model's buffers).
    pub(crate) fn encode_profiled_from(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        x: &Buffer,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        clock.stage("hc.read_attention", |e| {
            self.ops.read(e, &self.hcs[0], x, rows, arm.precision, None)
        })?;
        self.gdn
            .encode_profiled(clock, &self.hcs[0].mixed, rows, arm.precision.gdn())?;
        if arm.fused_write {
            clock.stage("hc.write_attention_read_mlp", |e| {
                self.ops.read(
                    e,
                    &self.hcs[1],
                    x,
                    rows,
                    arm.precision,
                    Some((self.gdn.output_buffer(), &self.hcs[0].inject, &self.stream)),
                )
            })?;
        } else {
            clock.stage("hc.write_attention", |e| {
                self.ops.write(
                    e,
                    x,
                    self.gdn.output_buffer(),
                    &self.hcs[0].inject,
                    &self.stream,
                    rows,
                );
                Ok(())
            })?;
            clock.stage("hc.read_mlp", |e| {
                self.ops.read(e, &self.hcs[1], &self.stream, rows, arm.precision, None)
            })?;
        }
        Ok(())
    }
    pub(crate) fn queue(&self) -> &Obj<dyn MTLCommandQueue> {
        &self.queue
    }
    pub(crate) fn finish(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        block: &Buffer,
        out: &Buffer,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && block.length() >= rows * H * 2
                && out.length() >= rows * F * 2,
            "final HC write geometry"
        );
        for _ in 0..1 + crate::metal_profile::dup(18) as usize {
            self.ops
                .write(enc, &self.stream, block, &self.hcs[1].inject, out, rows);
        }
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize) -> Result<Run> {
        read16(&self.x, rows * F)?;
        Ok(Run {
            pre: self.hcs[0].collect(rows)?,
            gdn: self.gdn.collect(rows, false)?,
            stream: read16(&self.stream, rows * F)?,
            post: self.hcs[1].collect(rows)?,
            stage_us: 0.,
            encode_queue_wait_us: 0.,
            gpu_us: 0.,
        })
    }
    pub fn run(
        &mut self,
        x: &[u16],
        init: &gdn::Initial,
        arm: Arm,
        repetitions: usize,
    ) -> Result<Run> {
        Ok(self.run_batch(x, init, &[arm], repetitions)?.0)
    }
    /// Every queued command restarts the same state; only the last arrays survive.
    pub fn run_batch(
        &mut self,
        x: &[u16],
        init: &gdn::Initial,
        arms: &[Arm],
        repetitions: usize,
    ) -> Result<(Run, Vec<f64>)> {
        ensure!((1..=4).contains(&repetitions), "HC/GDN repetitions");
        ensure!(!arms.is_empty() && arms.len() <= 64, "HC/GDN queue bound");
        let stage = Instant::now();
        let rows = self.prepare(x, init)?;
        let stage_us = stage.elapsed().as_secs_f64() * 1e6;
        let time = Instant::now();
        let mut commands = vec![];
        for &arm in arms {
            let cb = self.queue.commandBuffer().context("HC/GDN command")?;
            let enc = cb.computeCommandEncoder().context("HC/GDN encoder")?;
            for _ in 0..repetitions {
                self.encode(&enc, rows, arm)?;
            }
            enc.endEncoding();
            commands.push(cb);
        }
        // All fallible construction precedes commit; retained resources live
        // until the queue completes. No CPU writes/readbacks while in flight.
        for cb in &commands {
            cb.commit();
        }
        commands.last().unwrap().waitUntilCompleted();
        let encode_queue_wait_us = time.elapsed().as_secs_f64() * 1e6;
        let timings = commands
            .iter()
            .map(|cb| {
                ensure!(
                    cb.status() == MTLCommandBufferStatus::Completed,
                    "HC/GDN GPU error {:?}",
                    cb.error()
                );
                let us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
                ensure!(us > 0. && us.is_finite(), "HC/GDN timer");
                Ok(us)
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_us = (commands.last().unwrap().GPUEndTime() - commands[0].GPUStartTime()) * 1e6;
        let mut run = self.collect(rows)?;
        run.stage_us = stage_us;
        run.encode_queue_wait_us = encode_queue_wait_us;
        run.gpu_us = gpu_us;
        Ok((run, timings))
    }
}
