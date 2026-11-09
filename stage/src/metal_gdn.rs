//! One dependent GDN block on one native Metal queue. The four input
//! projections, state update, norm/gate and output projection stay on-device.
use crate::{
    gdn::*,
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, Phase, alloc, dense_scratch_bytes, upload},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const GUARD: usize = 64;
const SENTINEL: u32 = 0x7fc12345;
const BSENTINEL: u16 = 0x7fc1;
#[derive(Clone, Copy, Debug)]
pub enum BlockArm {
    Staged,
    Fused,
    F16,
    F16x2,
    F16Staged,
    F16x2Staged,
    PackedF16,
    PackedF16x2,
}
impl BlockArm {
    pub const ALL: [Self; 6] = [
        Self::Staged,
        Self::Fused,
        Self::F16Staged,
        Self::F16,
        Self::F16x2Staged,
        Self::F16x2,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Staged => "qmv_staged",
            Self::Fused => "qmv_fused",
            Self::F16 => "f16_mpp_fused",
            Self::F16x2 => "f16x2_mpp_fused",
            Self::F16Staged => "f16_mpp_staged",
            Self::F16x2Staged => "f16x2_mpp_staged",
            Self::PackedF16 => "f16_packed_fused",
            Self::PackedF16x2 => "f16x2_packed_fused",
        }
    }
    fn dense(self, rows: usize, n: usize) -> Arm {
        if n == HV {
            return Arm::Qmv4;
        }
        if matches!(self, Self::PackedF16 | Self::PackedF16x2) {
            // R04h bank measurements, specific to these three Qwen geometries.
            // Keep the tiny gate projections on QMV regardless of batch width.
            // R58 screen: QWEN_R58_GDN_SPLITS=c,v,out overrides the split counts.
            let (sc, sv, so) = gdn_splits();
            return Arm::FragmentPacked {
                splits: if n == C {
                    sc
                } else if n == V {
                    sv
                } else {
                    so
                },
                residual: matches!(self, Self::PackedF16x2),
            };
        }
        if rows == 1 {
            return Arm::Qmv4;
        }
        match self {
            Self::Staged | Self::Fused => Arm::Qmv4,
            Self::F16 | Self::F16Staged => Arm::MppF16,
            Self::F16x2 | Self::F16x2Staged => Arm::MppF16x2,
            Self::PackedF16 | Self::PackedF16x2 => unreachable!(),
        }
    }
    fn staged(self) -> bool {
        matches!(self, Self::Staged | Self::F16Staged | Self::F16x2Staged)
    }
}
struct Pair {
    full: Buffer,
    bf: Buffer,
    n: usize,
}
impl Pair {
    fn new(d: &objc2::runtime::ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Self> {
        Ok(Self {
            full: alloc(d, (MAX_T * n + GUARD) * 4)?,
            bf: alloc(d, (MAX_T * n + GUARD) * 2)?,
            n,
        })
    }
}
pub struct MetalGdn {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: DenseOps,
    pipelines: Vec<Obj<dyn MTLComputePipelineState>>,
    weights: Vec<DenseWeight>,
    small: Vec<Buffer>,
    x: Buffer,
    initial_conv: Buffer,
    initial_state: Buffer,
    projected: Vec<Pair>,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    gates: Buffer,
    y: Buffer,
    gated: Buffer,
    states: Buffer,
    conv: Buffer,
    sums: Buffer,
    /// R55: private split-K scratch per input projection (overlapped projections).
    sums4: Vec<Buffer>,
    _single_owner: PhantomData<Rc<()>>,
    /// R56 zero-copy commit (opt-in): the next forward reads its initial state from snapshot
    /// slot i of `states` instead of `initial_state` (gdn_recur and the fused kernels load
    /// each thread's state elements before any snapshot store to the same elements).
    zero_copy: std::cell::Cell<bool>,
    init_slot: std::cell::Cell<Option<usize>>,
}
pub struct Run {
    pub projected: Vec<Vec<u16>>,
    pub output: Vec<f32>,
    pub bf_output: Vec<u16>,
    pub core: Core,
    pub stage_us: f64,
    pub submit_wait_us: f64,
    pub hot_us: f64,
    pub gpu_us: f64,
}
fn poison32(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            .fill(SENTINEL);
    }
}
fn poison16(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(BSENTINEL);
    }
}
fn read32(b: &Buffer, count: usize) -> Result<Vec<f32>> {
    ensure!(count <= b.length() / 4, "GDN FP32 read bound");
    let all =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4) };
    ensure!(
        all[count..].iter().all(|&x| x == SENTINEL),
        "GDN FP32 trailing canary"
    );
    let out: Vec<_> = all[..count].iter().map(|&x| f32::from_bits(x)).collect();
    ensure!(
        out.iter().all(|x| x.is_finite()),
        "GDN FP32 unwritten/nonfinite"
    );
    Ok(out)
}
fn read16(b: &Buffer, count: usize) -> Result<Vec<u16>> {
    ensure!(count <= b.length() / 2, "GDN BF16 read bound");
    let all =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        all[count..].iter().all(|&x| x == BSENTINEL),
        "GDN BF16 trailing canary"
    );
    ensure!(
        all[..count].iter().all(|&x| bf16::from_bits(x).is_finite()),
        "GDN BF16 unwritten/nonfinite"
    );
    Ok(all[..count].to_vec())
}
fn gdn_splits() -> (usize, usize, usize) {
    static S: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let v: Vec<usize> = std::env::var("QWEN_R58_GDN_SPLITS").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if v.len() == 3 && v.iter().all(|&x| (1..=16).contains(&x)) { (v[0], v[1], v[2]) } else { (2, 4, 8) }
    })
}
fn proj_overlap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R55_PROJ_OVERLAP").as_deref() != Ok("0"))
}
impl MetalGdn {
    pub fn new(w: &Weights) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("GDN Metal device")?;
        let queue = device.newCommandQueue().context("GDN queue")?;
        Self::new_on(w, device, queue)
    }
    pub(crate) fn new_on(
        w: &Weights,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        w.validate()?;
        let ops = DenseOps::new(&device)?;
        let weights = w
            .projections
            .iter()
            .map(|w| DenseWeight::new(&device, w))
            .collect::<Result<Vec<_>>>()?;
        let mut small = vec![];
        for data in [&w.conv, &w.a_log, &w.dt_bias, &w.norm] {
            let b = alloc(&device, data.len() * 2)?;
            upload(&b, data)?;
            small.push(b);
        }
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let lib = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../kernels/gdn_block.metal")),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("GDN MSL: {e}"))?;
        let mut pipelines = vec![];
        for (name, threads) in [
            ("gdn_prepare", 128),
            ("gdn_recur", 128),
            ("gdn_norm_gate", 32),
            ("gdn_fused1", 128),
            ("gdn_fused4", 128),
            ("gdn_fused8", 128),
            ("gdn_fused16", 128),
        ] {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .context("GDN function")?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("GDN pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= threads
                    && p.staticThreadgroupMemoryLength() <= device.maxThreadgroupMemoryLength(),
                "GDN pipeline geometry"
            );
            pipelines.push(p);
        }
        let projected = [C, V, HV, HV, H]
            .into_iter()
            .map(|n| Pair::new(&device, n))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x: alloc(&device, MAX_T * H * 2)?,
            initial_conv: alloc(&device, 3 * C * 2)?,
            initial_state: alloc(&device, STATE * 4)?,
            q: alloc(&device, (MAX_T * HK * DK + GUARD) * 2)?,
            k: alloc(&device, (MAX_T * HK * DK + GUARD) * 2)?,
            v: alloc(&device, (MAX_T * V + GUARD) * 2)?,
            gates: alloc(&device, (MAX_T * HV * 2 + GUARD) * 4)?,
            y: alloc(&device, (MAX_T * V + GUARD) * 2)?,
            gated: alloc(&device, (MAX_T * V + GUARD) * 2)?,
            states: alloc(&device, (MAX_T * STATE + GUARD) * 4)?,
            conv: alloc(&device, (3 * C + GUARD) * 2)?,
            sums: alloc(
                &device,
                w.projections
                    .iter()
                    .map(|w| dense_scratch_bytes(w.k, w.n, MAX_T))
                    .max()
                    .unwrap(),
            )?,
            sums4: [C, V, HV, HV]
                .iter()
                .map(|&n| alloc(&device, dense_scratch_bytes(H, n, MAX_T)))
                .collect::<Result<_>>()?,
            device,
            queue,
            ops,
            pipelines,
            weights,
            small,
            projected,
            _single_owner: PhantomData,
            zero_copy: std::cell::Cell::new(false),
            init_slot: std::cell::Cell::new(None),
        })
    }
    pub fn description(&self) -> serde_json::Value {
        let weights = self.weights.iter().map(DenseWeight::bytes).sum::<usize>()
            + self.small.iter().map(|b| b.length()).sum::<usize>();
        let scratch = [
            &self.x,
            &self.initial_conv,
            &self.initial_state,
            &self.q,
            &self.k,
            &self.v,
            &self.gates,
            &self.y,
            &self.gated,
            &self.states,
            &self.conv,
            &self.sums,
        ]
        .iter()
        .map(|b| b.length())
        .sum::<usize>()
            + self
                .projected
                .iter()
                .map(|b| b.full.length() + b.bf.length())
                .sum::<usize>();
        serde_json::json!({"device":self.device.name().to_string(),"weight_bytes":weights,"allocated_bytes":weights+scratch,
            "original_dense_weight_bytes":self.weights.iter().map(DenseWeight::original_bytes).sum::<usize>(),
            "alternate_repacked_bytes":self.weights.iter().map(DenseWeight::fragment_bytes).sum::<usize>(),
            "packed_split_policy":{"qkv":2,"z":4,"out":8,"a_b":"QMV"},
            "gdn_threadgroup_bytes":self.pipelines.iter().map(|p|p.staticThreadgroupMemoryLength()).collect::<Vec<_>>(),
            "max_tokens":MAX_T,"state_dtype":"FP32","gate_dtype":"FP32","output_gate":"sigmoid","safe_math":true,
            "host_intermediate_transfers":0,"command_buffers_per_run":1,"full_state_snapshot_per_position":true})
    }
    /// Encode a dependent block onto the caller's queue. No submit or host read.
    /// Caller owns buffer lifetimes and must complete before prepare/collect.
    pub(crate) fn encode(
        &self,
        enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        rows: usize,
        arm: BlockArm,
    ) -> Result<()> {
        self.check_encode(input, rows)?;
        use crate::metal_profile::dup;
        for k in 0..1 + dup(1) as usize {
            let _ = k;
            self.encode_projections(enc, input, rows, arm)?;
            crate::metal_ops::barrier(&enc);
        }
        for _ in 0..1 + dup(2) as usize {
            self.encode_core(enc, rows, arm);
            crate::metal_ops::barrier(&enc);
        }
        for _ in 0..1 + dup(3) as usize {
            self.encode_norm(enc, rows);
            crate::metal_ops::barrier(&enc);
        }
        for _ in 0..1 + dup(4) as usize {
            self.encode_output(enc, rows, arm)?;
            crate::metal_ops::barrier(&enc);
        }
        Ok(())
    }
    /// R55 stage clock: the same four stages, one command buffer each.
    pub(crate) fn encode_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        input: &Buffer,
        rows: usize,
        arm: BlockArm,
    ) -> Result<()> {
        self.check_encode(input, rows)?;
        clock.stage("gdn.projections", |e| self.encode_projections(e, input, rows, arm))?;
        clock.stage("gdn.core", |e| {
            self.encode_core(e, rows, arm);
            Ok(())
        })?;
        clock.stage("gdn.norm_gate", |e| {
            self.encode_norm(e, rows);
            Ok(())
        })?;
        clock.stage("gdn.output_projection", |e| self.encode_output(e, rows, arm))
    }
    fn check_encode(&self, input: &Buffer, rows: usize) -> Result<()> {
        ensure!(
            (1..=MAX_T).contains(&rows) && input.length() >= rows * H * 2,
            "GDN encoded shape"
        );
        Ok(())
    }
    fn encode_projections(
        &self,
        enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        rows: usize,
        arm: BlockArm,
    ) -> Result<()> {
        if proj_overlap() {
            // All four GEMMs (private scratch) may overlap; then all four reductions.
            for phase in [Phase::Gemm, Phase::Reduce] {
                for j in 0..4 {
                    self.ops.encode_phase(
                        &enc,
                        &self.weights[j],
                        input,
                        &self.projected[j].full,
                        &self.projected[j].bf,
                        &self.sums4[j],
                        rows,
                        0,
                        arm.dense(rows, self.weights[j].n),
                        phase,
                    )?;
                }
                if phase == Phase::Gemm {
                    crate::metal_ops::barrier(&enc);
                }
            }
            return Ok(());
        }
        for j in 0..4 {
            self.ops.encode(
                &enc,
                &self.weights[j],
                input,
                &self.projected[j].full,
                &self.projected[j].bf,
                &self.sums,
                rows,
                0,
                arm.dense(rows, self.weights[j].n),
            )?;
        }
        Ok(())
    }
    fn bind_core(&self, enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, rows: usize) {
        let buffers = [
            &self.projected[0].bf,
            &self.projected[1].bf,
            &self.projected[2].bf,
            &self.projected[3].bf,
            &self.initial_conv,
            &self.initial_state,
            &self.small[0],
            &self.small[1],
            &self.small[2],
            &self.small[3],
            &self.q,
            &self.k,
            &self.v,
            &self.gates,
            &self.y,
            &self.gated,
            &self.states,
            &self.conv,
        ];
        let param = [rows as u32];
        unsafe {
            for (i, b) in buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            if let Some(slot) = self.init_slot.get() {
                enc.setBuffer_offset_atIndex(Some(&self.states), slot * STATE * 4, 5);
            }
            enc.setBytes_length_atIndex(NonNull::from(&param).cast(), 4, 18);
        }
    }
    fn encode_core(
        &self,
        enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        rows: usize,
        arm: BlockArm,
    ) {
        self.bind_core(enc, rows);
        if arm.staged() {
            enc.setComputePipelineState(&self.pipelines[0]);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: HV,
                    height: rows,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            crate::metal_ops::barrier(&enc);
            enc.setComputePipelineState(&self.pipelines[1]);
        } else {
            let idx = if rows == 1 {
                3
            } else if rows <= 4 {
                4
            } else if rows <= 8 {
                5
            } else {
                6
            };
            enc.setComputePipelineState(&self.pipelines[idx]);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: HV * 4,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    fn encode_norm(&self, enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>, rows: usize) {
        self.bind_core(enc, rows);
        enc.setComputePipelineState(&self.pipelines[2]);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: HV,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
    fn encode_output(
        &self,
        enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        rows: usize,
        arm: BlockArm,
    ) -> Result<()> {
        self.ops.encode(
            &enc,
            &self.weights[4],
            &self.gated,
            &self.projected[4].full,
            &self.projected[4].bf,
            &self.sums,
            rows,
            0,
            arm.dense(rows, H),
        )
    }
    /// R58 prefill: (projections qkv/z/a/b/out, small weights conv/A_log/dt_bias/norm, committed
    /// convolution history, committed state, zero-copy initial-state slot if one is in use).
    pub(crate) fn prefill_parts(&self) -> (&[DenseWeight], &[Buffer], &Buffer, &Buffer, Option<usize>) {
        (&self.weights, &self.small, &self.initial_conv, &self.initial_state, self.init_slot.get())
    }
    /// R55 resident model, host side on shared memory after the forward completed: the
    /// next initial state is the snapshot after row `accepted`-1 and the convolution
    /// history is the last three pre-convolution rows of [initial; accepted rows].
    pub(crate) fn commit(&self, accepted: usize, rows: usize) -> Result<()> {
        if let Some((src, dst, n)) = self.commit_split(accepted, rows)? {
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n) };
        }
        Ok(())
    }
    /// R56: commit's convolution part now, and the 3 MiB FP32 state snapshot copy returned as
    /// (source address, destination address, bytes) for the caller to run (in parallel with
    /// the other layers'). Shared-memory buffers; the forward has completed.
    pub(crate) fn commit_split(&self, accepted: usize, rows: usize) -> Result<Option<(usize, usize, usize)>> {
        ensure!(accepted <= rows && rows <= MAX_T, "GDN commit bounds");
        if accepted == 0 {
            if let (true, Some(slot)) = (self.zero_copy.get(), self.init_slot.get()) {
                ensure!(slot >= rows, "zero-copy commit(0) after a forward that overwrote the initial-state slot");
            }
            return Ok(None);
        }
        unsafe {
            let states = self.states.contents().cast::<f32>().as_ptr();
            let init = self.initial_state.contents().cast::<f32>().as_ptr();
            let job = (states.add((accepted - 1) * STATE) as usize, init as usize, STATE * 4);
            let conv = self.initial_conv.contents().cast::<u16>().as_ptr();
            let pre = self.projected[0].bf.contents().cast::<u16>().as_ptr();
            let mut cat = Vec::with_capacity((3 + accepted) * C);
            cat.extend_from_slice(std::slice::from_raw_parts(conv, 3 * C));
            cat.extend_from_slice(std::slice::from_raw_parts(pre, accepted * C));
            std::ptr::copy_nonoverlapping(cat[accepted * C..].as_ptr(), conv, 3 * C);
            if self.zero_copy.get() {
                self.init_slot.set(Some(accepted - 1));
                return Ok(None);
            }
            Ok(Some(job))
        }
    }
    /// R56: select zero-copy commits; turning it off copies a referenced slot back first.
    pub(crate) fn set_zero_copy(&self, on: bool) {
        if !on {
            self.materialize();
        }
        self.zero_copy.set(on);
    }
    /// R58 prefill: the committed state must live in `initial_state` (copies a referenced
    /// zero-copy slot back; zero-copy commits stay selected).
    pub(crate) fn prefill_ready(&self) {
        self.materialize();
    }
    /// Copy the referenced snapshot slot into `initial_state` (no forward may be pending).
    fn materialize(&self) {
        if let Some(slot) = self.init_slot.take() {
            unsafe {
                let states = self.states.contents().cast::<f32>().as_ptr();
                let init = self.initial_state.contents().cast::<f32>().as_ptr();
                std::ptr::copy_nonoverlapping(states.add(slot * STATE), init, STATE);
            }
        }
    }
    pub(crate) fn output_buffer(&self) -> &Buffer {
        &self.projected[4].bf
    }
    pub(crate) fn prepare(&mut self, initial: &Initial) -> Result<()> {
        initial.validate()?;
        for p in &self.projected {
            poison32(&p.full);
            poison16(&p.bf);
        }
        for b in [&self.q, &self.k, &self.v, &self.y, &self.gated, &self.conv] {
            poison16(b);
        }
        for b in [&self.gates, &self.states] {
            poison32(b);
        }
        upload(&self.initial_conv, &initial.conv)?;
        upload(&self.initial_state, &initial.state)?;
        self.init_slot.set(None);
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize, any_staged: bool) -> Result<Run> {
        ensure!((1..=MAX_T).contains(&rows), "GDN collected shape");
        let projected = self.projected[..4]
            .iter()
            .map(|p| read16(&p.bf, rows * p.n))
            .collect::<Result<Vec<_>>>()?;
        for p in &self.projected[..4] {
            read32(&p.full, rows * p.n)?;
        }
        let t = if any_staged { rows } else { 0 };
        read16(&self.q, t * HK * DK)?;
        read16(&self.k, t * HK * DK)?;
        read16(&self.v, t * V)?;
        read32(&self.gates, t * HV * 2)?;
        Ok(Run {
            projected,
            output: read32(&self.projected[4].full, rows * H)?,
            bf_output: read16(&self.projected[4].bf, rows * H)?,
            core: Core {
                y: read16(&self.y, rows * V)?,
                gated: read16(&self.gated, rows * V)?,
                states: read32(&self.states, rows * STATE)?,
                conv: read16(&self.conv, 3 * C)?,
            },
            stage_us: 0.,
            submit_wait_us: 0.,
            hot_us: 0.,
            gpu_us: 0.,
        })
    }
    pub fn run(
        &mut self,
        input: &[u16],
        initial: &Initial,
        arm: BlockArm,
        repetitions: usize,
    ) -> Result<Run> {
        Ok(self.run_batch(input, initial, &[arm], repetitions)?.0)
    }
    /// Queue all commands before waiting. Every entry restarts from the same
    /// supplied state. Only the final entry's full arrays survive for validation;
    /// individual command GPU spans are returned separately. No concurrent client.
    pub fn run_batch(
        &mut self,
        input: &[u16],
        initial: &Initial,
        arms: &[BlockArm],
        repetitions: usize,
    ) -> Result<(Run, Vec<f64>)> {
        ensure!(
            !arms.is_empty() && arms.len() <= 48,
            "GDN queued batch bound"
        );
        initial.validate()?;
        ensure!(
            !input.is_empty()
                && input.len() % H == 0
                && input.len() / H <= MAX_T
                && (1..=8).contains(&repetitions),
            "GDN run geometry"
        );
        ensure!(
            input.iter().all(|&x| bf16::from_bits(x).is_finite()),
            "GDN nonfinite input"
        );
        if arms
            .iter()
            .any(|a| !matches!(a, BlockArm::Staged | BlockArm::Fused))
        {
            ensure!(
                input
                    .iter()
                    .all(|&x| f16::from_f32(bf16::from_bits(x).to_f32()).is_finite()),
                "GDN half input overflow"
            );
        }
        let rows = input.len() / H;
        self.prepare(initial)?;
        let hot = Instant::now();
        let stage = Instant::now();
        upload(&self.x, input)?;
        upload(&self.initial_conv, &initial.conv)?;
        upload(&self.initial_state, &initial.state)?;
        let stage_us = stage.elapsed().as_secs_f64() * 1e6;
        let submit = Instant::now();
        let mut commands = Vec::with_capacity(arms.len());
        for &arm in arms {
            let cb = self.queue.commandBuffer().context("GDN command buffer")?;
            let enc = cb.computeCommandEncoder().context("GDN encoder")?;
            for _ in 0..repetitions {
                self.encode(&enc, &self.x, rows, arm)?;
            }
            enc.endEncoding();
            commands.push(cb);
        }
        // All fallible encoding completes before submission. The single queue's
        // tracked resources order successive writes to the reused output slots.
        for command in &commands {
            command.commit();
        }
        let cb = commands.last().unwrap();
        cb.waitUntilCompleted();
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "GDN GPU error {:?}: {:?}",
            cb.status(),
            cb.error()
        );
        let submit_wait_us = submit.elapsed().as_secs_f64() * 1e6;
        let hot_us = hot.elapsed().as_secs_f64() * 1e6;
        let timings = commands
            .iter()
            .map(|c| {
                ensure!(
                    c.status() == MTLCommandBufferStatus::Completed,
                    "queued GDN command failed"
                );
                let us = (c.GPUEndTime() - c.GPUStartTime()) * 1e6;
                ensure!(us.is_finite() && us > 0., "queued GDN timer");
                Ok(us)
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_us = (cb.GPUEndTime() - commands.first().unwrap().GPUStartTime()) * 1e6;
        ensure!(gpu_us.is_finite() && gpu_us > 0., "GDN GPU timer");
        let mut run = self.collect(rows, arms.iter().any(|a| a.staged()))?;
        run.stage_us = stage_us;
        run.submit_wait_us = submit_wait_us;
        run.hot_us = hot_us;
        run.gpu_us = gpu_us;
        Ok((run, timings))
    }
}
