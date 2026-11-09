//! Synchronous, single-owner native Metal routed-expert pipeline. Allocations
//! persist across calls; run() waits before any CPU reuse/readback. No CUDA/MLX.
use crate::moe::{ExpertBank, H, I, MAX_ROWS, Routes, TOPK};
use crate::projection::ScaledProjection;
use anyhow::{Context, Result, ensure};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const SIMD: &str = include_str!("../kernels/exl3_project.metal");
const MPP: &str = include_str!("../kernels/exl3_mpp.metal");
const TRANSFORMS: &str = include_str!("../kernels/exl3_moe.metal");
type Obj<T> = Retained<ProtocolObject<T>>;
type Buffer = Obj<dyn MTLBuffer>;
type Pipeline = Obj<dyn MTLComputePipelineState>;

#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Simd,
    Mpp16,
    Adaptive,
}
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Simd => "simd_s2_s2",
            Self::Mpp16 => "mpp16_s4_s2",
            Self::Adaptive => "adaptive_g4_d2_s4_s2",
        }
    }
    fn splits(self) -> (usize, usize) {
        match self {
            Self::Simd => (2, 2),
            _ => (4, 2),
        }
    }
    fn thresholds(self) -> (usize, usize) {
        match self {
            Self::Simd => (usize::MAX, usize::MAX),
            Self::Mpp16 => (0, 0),
            Self::Adaptive => (4, 2),
        }
    }
}
struct Weight {
    trellis: Buffer,
    suh: Buffer,
    svh: Buffer,
}
pub struct MetalMoe {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    simd: Pipeline,
    mpp: Pipeline,
    prepare: Pipeline,
    mid: Pipeline,
    finish: Pipeline,
    gate: Weight,
    up: Weight,
    down: Weight,
    experts: usize,
    x: Buffer,
    ids: Buffer,
    order: Buffer,
    inverse: Buffer,
    scores: Buffer,
    gblocks: Buffer,
    dblocks: Buffer,
    pg: Buffer,
    pu: Buffer,
    rg: Buffer,
    ru: Buffer,
    md: Buffer,
    rd: Buffer,
    out: Buffer,
    out_bf: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
pub struct DebugVectors {
    pub prepared_gate: Vec<f32>,
    pub prepared_up: Vec<f32>,
    pub raw_gate: Vec<f32>,
    pub raw_up: Vec<f32>,
    pub middle: Vec<f32>,
    pub raw_down: Vec<f32>,
}
pub struct Run {
    pub output: Vec<f32>,
    pub bf16_output: Vec<f32>,
    pub diagnostic: Option<DebugVectors>,
    pub route_plan_us: f64,
    pub stage_us: f64,
    pub submit_wait_us: f64,
    pub gpu_span_us: f64,
    pub total_hot_us: f64,
    pub gate_simd_rows: usize,
    pub down_simd_rows: usize,
    pub gate_mpp_blocks: usize,
    pub down_mpp_blocks: usize,
    pub used_experts: usize,
}
fn alloc(d: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer> {
    d.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .context("Metal allocation")
}
fn upload<T: Copy>(b: &ProtocolObject<dyn MTLBuffer>, data: &[T]) -> Result<()> {
    ensure!(
        std::mem::size_of_val(data) <= b.length(),
        "Metal upload bound"
    );
    // The owning run is exclusive and the preceding command completed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            b.contents().cast::<u8>().as_ptr(),
            std::mem::size_of_val(data),
        );
    }
    Ok(())
}
fn weight(
    d: &ProtocolObject<dyn MTLDevice>,
    p: &ScaledProjection,
    k: usize,
    n: usize,
    experts: usize,
) -> Result<Weight> {
    ensure!(
        p.raw.k == k
            && p.raw.n == n
            && p.raw.experts == experts
            && p.raw.packed.len() == experts * k * n / 2
            && p.suh.len() == experts * k
            && p.svh.len() == experts * n,
        "Metal weight shape"
    );
    let w = Weight {
        trellis: alloc(d, p.raw.packed.len())?,
        suh: alloc(d, p.suh.len() * 2)?,
        svh: alloc(d, p.svh.len() * 2)?,
    };
    upload(&w.trellis, &p.raw.packed)?;
    upload(&w.suh, &p.suh)?;
    upload(&w.svh, &p.svh)?;
    Ok(w)
}
fn dispatch(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: usize,
    y: usize,
    z: usize,
    threads: usize,
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
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
fn bind(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    params: &[u32; 8],
) {
    // Every kernel's fixed buffer ABI is visible in exl3_moe.metal. Lifetimes
    // extend through waitUntilCompleted; counts and allocation bounds checked.
    unsafe {
        for (i, b) in buffers.iter().enumerate() {
            enc.setBuffer_offset_atIndex(Some(b), 0, i);
        }
        enc.setBytes_length_atIndex(NonNull::from(params).cast(), 32, 7);
    }
}
impl MetalMoe {
    pub fn new(bank: &ExpertBank) -> Result<Self> {
        let experts = bank.original_ids.len();
        ensure!((TOPK..=128).contains(&experts), "Metal expert bank");
        let device = MTLCreateSystemDefaultDevice().context("Metal device")?;
        let queue = device.newCommandQueue().context("Metal queue")?;
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(&format!("{SIMD}\n{MPP}\n{TRANSFORMS}")),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("MoE MSL compile: {e}"))?;
        let pipeline = |name: &str, threads: usize| -> Result<Pipeline> {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .context("MoE Metal function")?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("MoE pipeline: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= threads,
                "MoE SIMD geometry"
            );
            Ok(p)
        };
        let simd = pipeline("exl3_project", 128)?;
        let mpp = pipeline("exl3_mpp", 128)?;
        let prepare = pipeline("moe_prepare_pair", 32)?;
        let mid = pipeline("moe_mid", 32)?;
        let finish = pipeline("moe_finish", TOPK * 32)?;
        let gate = weight(&device, &bank.gate, H, I, experts)?;
        let up = weight(&device, &bank.up, H, I, experts)?;
        let down = weight(&device, &bank.down, I, H, experts)?;
        let slots = MAX_ROWS * TOPK;
        Ok(Self {
            x: alloc(&device, MAX_ROWS * H * 4)?,
            ids: alloc(&device, slots * 4)?,
            order: alloc(&device, slots * 4)?,
            inverse: alloc(&device, slots * 4)?,
            scores: alloc(&device, slots * 4)?,
            gblocks: alloc(&device, slots * 16)?,
            dblocks: alloc(&device, slots * 16)?,
            pg: alloc(&device, slots * H * 2)?,
            pu: alloc(&device, slots * H * 2)?,
            rg: alloc(&device, slots * I * 4 * 4)?,
            ru: alloc(&device, slots * I * 4 * 4)?,
            md: alloc(&device, slots * I * 2)?,
            rd: alloc(&device, slots * H * 4 * 4)?,
            out: alloc(&device, MAX_ROWS * H * 4)?,
            out_bf: alloc(&device, MAX_ROWS * H * 2)?,
            device,
            queue,
            simd,
            mpp,
            prepare,
            mid,
            finish,
            gate,
            up,
            down,
            experts,
            _single_owner: PhantomData,
        })
    }
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
    pub fn allocated_bytes(&self) -> usize {
        [
            &self.x,
            &self.ids,
            &self.order,
            &self.inverse,
            &self.scores,
            &self.gblocks,
            &self.dblocks,
            &self.pg,
            &self.pu,
            &self.rg,
            &self.ru,
            &self.md,
            &self.rd,
            &self.out,
            &self.out_bf,
            &self.gate.trellis,
            &self.gate.suh,
            &self.gate.svh,
            &self.up.trellis,
            &self.up.suh,
            &self.up.svh,
            &self.down.trellis,
            &self.down.suh,
            &self.down.svh,
        ]
        .iter()
        .map(|b| b.length())
        .sum()
    }
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &ProtocolObject<dyn MTLBuffer>,
        w: &Weight,
        out: &ProtocolObject<dyn MTLBuffer>,
        blocks: &ProtocolObject<dyn MTLBuffer>,
        k: usize,
        n: usize,
        splits: usize,
        simd_rows: usize,
        block_count: usize,
    ) {
        let p = [k as u32, n as u32, splits as u32, 0];
        unsafe {
            for (i, b) in [
                (0, x),
                (1, &*w.trellis),
                (2, &*self.ids),
                (3, out),
                (5, blocks),
            ] {
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            enc.setBytes_length_atIndex(NonNull::from(&p).cast(), 16, 4);
        }
        if simd_rows > 0 {
            enc.setComputePipelineState(&self.simd);
            dispatch(enc, n / 16, simd_rows, splits, 128);
        }
        if block_count > 0 {
            enc.setComputePipelineState(&self.mpp);
            dispatch(enc, n / 128, block_count, splits, 128);
        }
    }
    pub fn run(
        &mut self,
        x: &[f32],
        routes: &Routes,
        arm: Arm,
        repetitions: usize,
        capture: bool,
    ) -> Result<Run> {
        routes.validate()?;
        ensure!(
            routes.experts == self.experts
                && x.len() == routes.rows * H
                && x.iter().all(|v| v.is_finite())
                && (1..=16).contains(&repetitions),
            "MoE run inputs"
        );
        let slots = routes.ids.len();
        let (gs, ds) = arm.splits();
        // Poison only active intermediate spans, outside timing. All are
        // overwritten every repetition; final values must reflect this input.
        unsafe {
            for (b, count) in [
                (&self.pg, slots * H),
                (&self.pu, slots * H),
                (&self.md, slots * I),
                (&self.out_bf, routes.rows * H),
            ] {
                std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), count)
                    .fill(0x7fff);
            }
            for (b, count) in [
                (&self.rg, slots * I * gs),
                (&self.ru, slots * I * gs),
                (&self.rd, slots * H * ds),
                (&self.out, routes.rows * H),
            ] {
                std::slice::from_raw_parts_mut(b.contents().cast::<f32>().as_ptr(), count)
                    .fill(f32::NAN);
            }
        }
        let total = Instant::now();
        let route_clock = Instant::now();
        let plan = routes.plan()?;
        let (gt, dt) = arm.thresholds();
        let (gprefix, gblocks) = plan.partition(gt);
        let (dprefix, dblocks) = plan.partition(dt);
        let route_plan_us = route_clock.elapsed().as_secs_f64() * 1e6;
        let stage = Instant::now();
        upload(&self.x, x)?;
        upload(&self.ids, &plan.ids)?;
        upload(&self.order, &plan.order)?;
        upload(&self.inverse, &plan.inverse)?;
        upload(&self.scores, &routes.scores)?;
        upload(&self.gblocks, &gblocks)?;
        upload(&self.dblocks, &dblocks)?;
        let stage_us = stage.elapsed().as_secs_f64() * 1e6;
        let submit = Instant::now();
        let cb = self.queue.commandBuffer().context("MoE command buffer")?;
        let enc = cb.computeCommandEncoder().context("MoE encoder")?;
        let p = [
            H as u32,
            I as u32,
            TOPK as u32,
            gs as u32,
            ds as u32,
            0,
            0,
            0,
        ];
        for _ in 0..repetitions {
            enc.setComputePipelineState(&self.prepare);
            bind(
                &enc,
                &[
                    &self.x,
                    &self.gate.suh,
                    &self.up.suh,
                    &self.ids,
                    &self.order,
                    &self.pg,
                    &self.pu,
                ],
                &p,
            );
            dispatch(&enc, H / 128, slots, 1, 32);
            crate::metal_ops::barrier(&enc);
            self.project(
                &enc,
                &self.pg,
                &self.gate,
                &self.rg,
                &self.gblocks,
                H,
                I,
                gs,
                gprefix,
                gblocks.len(),
            );
            self.project(
                &enc,
                &self.pu,
                &self.up,
                &self.ru,
                &self.gblocks,
                H,
                I,
                gs,
                gprefix,
                gblocks.len(),
            );
            crate::metal_ops::barrier(&enc);
            enc.setComputePipelineState(&self.mid);
            bind(
                &enc,
                &[
                    &self.rg,
                    &self.ru,
                    &self.gate.svh,
                    &self.up.svh,
                    &self.down.suh,
                    &self.ids,
                    &self.md,
                ],
                &p,
            );
            dispatch(&enc, I / 128, slots, 1, 32);
            crate::metal_ops::barrier(&enc);
            self.project(
                &enc,
                &self.md,
                &self.down,
                &self.rd,
                &self.dblocks,
                I,
                H,
                ds,
                dprefix,
                dblocks.len(),
            );
            crate::metal_ops::barrier(&enc);
            enc.setComputePipelineState(&self.finish);
            bind(
                &enc,
                &[
                    &self.rd,
                    &self.down.svh,
                    &self.ids,
                    &self.inverse,
                    &self.scores,
                    &self.out,
                    &self.out_bf,
                ],
                &p,
            );
            dispatch(&enc, H / 128, routes.rows, 1, TOPK * 32);
            crate::metal_ops::barrier(&enc);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "MoE command {:?}: {:?}",
            cb.status(),
            cb.error()
        );
        let submit_wait_us = submit.elapsed().as_secs_f64() * 1e6;
        let total_hot_us = total.elapsed().as_secs_f64() * 1e6;
        let gpu_span_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
        // Completed command owns no CPU-live writes now. Diagnostics below are
        // excluded from timing and do not perform any execution-stage reduction.
        let output = unsafe {
            std::slice::from_raw_parts(self.out.contents().cast::<f32>().as_ptr(), routes.rows * H)
        }
        .to_vec();
        let bf16_output = unsafe {
            std::slice::from_raw_parts(
                self.out_bf.contents().cast::<u16>().as_ptr(),
                routes.rows * H,
            )
        }
        .iter()
        .map(|&v| half::bf16::from_bits(v).to_f32())
        .collect();
        let diagnostic = if capture {
            let unpermute_half = |b: &Buffer, dim: usize| {
                let raw = unsafe {
                    std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), slots * dim)
                };
                let mut dst = vec![0.; slots * dim];
                for (slot, &original) in plan.order.iter().enumerate() {
                    for j in 0..dim {
                        dst[original as usize * dim + j] =
                            half::f16::from_bits(raw[slot * dim + j]).to_f32();
                    }
                }
                dst
            };
            let unpermute_sum = |b: &Buffer, dim: usize, splits: usize| {
                let raw = unsafe {
                    std::slice::from_raw_parts(
                        b.contents().cast::<f32>().as_ptr(),
                        slots * dim * splits,
                    )
                };
                let mut dst = vec![0.; slots * dim];
                for (slot, &original) in plan.order.iter().enumerate() {
                    for s in 0..splits {
                        for j in 0..dim {
                            dst[original as usize * dim + j] += raw[(slot * splits + s) * dim + j];
                        }
                    }
                }
                dst
            };
            Some(DebugVectors {
                prepared_gate: unpermute_half(&self.pg, H),
                prepared_up: unpermute_half(&self.pu, H),
                raw_gate: unpermute_sum(&self.rg, I, gs),
                raw_up: unpermute_sum(&self.ru, I, gs),
                middle: unpermute_half(&self.md, I),
                raw_down: unpermute_sum(&self.rd, H, ds),
            })
        } else {
            None
        };
        Ok(Run {
            output,
            bf16_output,
            diagnostic,
            route_plan_us,
            stage_us,
            submit_wait_us,
            gpu_span_us,
            total_hot_us,
            gate_simd_rows: gprefix,
            down_simd_rows: dprefix,
            gate_mpp_blocks: gblocks.len(),
            down_mpp_blocks: dblocks.len(),
            used_experts: plan.counts.iter().filter(|&&n| n > 0).count(),
        })
    }
}
