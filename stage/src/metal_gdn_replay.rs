//! R05c prepared-operand verification plus accepted-prefix commit. Each run
//! owns its buffers until command completion; no target/model wrapper here.
use crate::{
    gdn_replay::{DIM, Fixture, HEADS, MAX_ROWS, STATE},
    metal_ops::{Buffer, Obj, alloc},
};
use anyhow::{Context, Result, ensure};
use objc2_foundation::NSString;
use objc2_metal::*;
use serde::Serialize;
use std::{marker::PhantomData, ptr::NonNull, rc::Rc, time::Instant};
const GUARD: usize = 64;
const POISON: u32 = 0x7fc12345;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum Arm {
    StoreAll = 0,
    PrefixReplay = 1,
    Compact = 2,
}
impl Arm {
    pub const ALL: [Self; 3] = [Self::StoreAll, Self::PrefixReplay, Self::Compact];
}

struct GuardBuffer {
    buffer: Buffer,
    capacity: usize,
}
impl GuardBuffer {
    fn new(d: &objc2::runtime::ProtocolObject<dyn MTLDevice>, capacity: usize) -> Result<Self> {
        Ok(Self {
            buffer: alloc(d, (capacity + 2 * GUARD) * 4)?,
            capacity,
        })
    }
    fn poison(&mut self) {
        // All prior commands complete before any CPU access to shared storage.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().cast::<u32>().as_ptr(),
                self.capacity + 2 * GUARD,
            )
            .fill(POISON);
        }
    }
    fn write(&mut self, values: &[f32]) -> Result<()> {
        ensure!(values.len() <= self.capacity, "replay upload bound");
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.buffer.contents().cast::<f32>().as_ptr().add(GUARD),
                values.len(),
            );
        }
        Ok(())
    }
    fn inspect(&self, used: usize) -> Result<&[u32]> {
        ensure!(used <= self.capacity, "replay readback bound");
        let data = unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<u32>().as_ptr(),
                self.capacity + 2 * GUARD,
            )
        };
        ensure!(
            data[..GUARD]
                .iter()
                .chain(&data[GUARD + used..])
                .all(|&v| v == POISON),
            "replay canary or inactive buffer changed"
        );
        Ok(&data[GUARD..GUARD + used])
    }
    fn collect(&self, used: usize) -> Result<Vec<f32>> {
        let values: Vec<_> = self
            .inspect(used)?
            .iter()
            .map(|&x| f32::from_bits(x))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "replay nonfinite/unwritten output"
        );
        Ok(values)
    }
}
#[derive(Serialize)]
pub struct Timing {
    pub gpu_cycle_us: f64,
    pub host_encode_submit_wait_us: f64,
}
pub struct Run {
    pub y: Vec<f32>,
    pub snapshots: Vec<f32>,
    pub final_state: Vec<f32>,
    pub pseudo: Vec<f32>,
    pub committed: Vec<f32>,
    pub timing: Timing,
}
pub struct MetalReplay {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    pipelines: Vec<Obj<dyn MTLComputePipelineState>>,
    buffers: Vec<GuardBuffer>,
    _local: PhantomData<Rc<()>>,
}
impl MetalReplay {
    pub fn new() -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use exclusive guarded runner"
        );
        let device = MTLCreateSystemDefaultDevice().context("replay Metal device")?;
        let queue = device.newCommandQueue().context("replay queue")?;
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../kernels/gdn_replay.metal")),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("replay MSL: {e}"))?;
        let mut pipelines = vec![];
        for name in [
            "gdn_capture_all",
            "gdn_capture_final",
            "gdn_capture_compact",
            "gdn_commit",
        ] {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .context("replay function")?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("replay pipeline {name}: {e}"))?;
            ensure!(
                p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= 128,
                "replay SIMD geometry"
            );
            pipelines.push(p);
        }
        let vector = HEADS * MAX_ROWS * DIM;
        let scalar = HEADS * MAX_ROWS;
        let buffers = [
            STATE,
            vector,
            vector,
            vector,
            scalar,
            scalar,
            vector,
            MAX_ROWS * STATE,
            STATE,
            vector,
            STATE,
        ]
        .into_iter()
        .map(|n| GuardBuffer::new(&device, n))
        .collect::<Result<_>>()?;
        Ok(Self {
            device,
            queue,
            pipelines,
            buffers,
            _local: PhantomData,
        })
    }
    pub fn description(&self) -> serde_json::Value {
        serde_json::json!({"device":self.device.name().to_string(),"math_mode":"safe","state_dtype":"float32","all_arm_probe_allocated_bytes":self.buffers.iter().map(|b|b.buffer.length()).sum::<usize>(),"guard_words_per_end":GUARD,"command_buffers_per_cycle":1,"compute_encoders_per_cycle":2,"prepared_inputs_only":true,"host_intermediate_transfers":0,"source_sha256":crate::projection::sha(include_bytes!("../kernels/gdn_replay.metal")),"limitations":["All three arms retain separate A/B storage in this probe","Committed state is separate from base/final for integrity checks; production aliasing is not implemented","Acceptance count is supplied before dispatch, so model/sampler/feedback costs are excluded"]})
    }
    fn encode(
        &self,
        cb: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: usize,
        params: &[u32; 3],
    ) -> Result<()> {
        let encoder = cb
            .computeCommandEncoder()
            .context("replay compute encoder")?;
        encoder.setComputePipelineState(&self.pipelines[pipeline]);
        unsafe {
            for (i, b) in self.buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(&b.buffer), GUARD * 4, i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(params).cast(),
                std::mem::size_of_val(params),
                11,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: HEADS * 4,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }
    pub fn run(&mut self, fixture: &Fixture, accepted: usize, arm: Arm) -> Result<Run> {
        fixture.validate()?;
        ensure!(
            accepted <= fixture.rows,
            "acceptance outside verified prefix"
        );
        for b in &mut self.buffers {
            b.poison();
        }
        for (b, x) in self.buffers.iter_mut().zip(fixture.inputs()) {
            b.write(x)?;
        }
        // Timer excludes upload/poison, reference construction and diagnostics.
        // Tracked buffers and separate encoders order capture before commit.
        let start = Instant::now();
        let params = [fixture.rows as u32, accepted as u32, arm as u32];
        let cb = self
            .queue
            .commandBuffer()
            .context("replay command buffer")?;
        self.encode(&cb, arm as usize, &params)?;
        self.encode(&cb, 3, &params)?;
        cb.commit();
        cb.waitUntilCompleted();
        let host_encode_submit_wait_us = start.elapsed().as_secs_f64() * 1e6;
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "replay command failed {:?}: {:?}",
            cb.status(),
            cb.error()
        );
        let gpu_cycle_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
        ensure!(
            gpu_cycle_us.is_finite() && gpu_cycle_us > 0.,
            "replay GPU timing"
        );
        for (b, input) in self.buffers.iter().zip(fixture.inputs()) {
            ensure!(
                b.inspect(input.len())?
                    .iter()
                    .zip(input)
                    .all(|(&a, b)| a == b.to_bits()),
                "prepared input changed"
            );
        }
        let vector = HEADS * fixture.rows * DIM;
        let y = self.buffers[6].collect(vector)?;
        let snapshots = self.buffers[7].collect(if arm == Arm::StoreAll {
            fixture.rows * STATE
        } else {
            0
        })?;
        let final_state = self.buffers[8].collect(if arm == Arm::StoreAll { 0 } else { STATE })?;
        let pseudo = self.buffers[9].collect(if arm == Arm::Compact { vector } else { 0 })?;
        let committed = self.buffers[10].collect(STATE)?;
        // Pure copy cases are strict bit-preservation controls. Reordered state
        // arithmetic is compared quantitatively by the probe, not by this gate.
        let copied = if accepted == 0 {
            Some(fixture.initial.as_slice())
        } else if arm == Arm::StoreAll {
            Some(&snapshots[(accepted - 1) * STATE..accepted * STATE])
        } else if accepted == fixture.rows {
            Some(final_state.as_slice())
        } else {
            None
        };
        if let Some(source) = copied {
            ensure!(
                source
                    .iter()
                    .zip(&committed)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "accepted state copy mismatch"
            );
        }
        Ok(Run {
            y,
            snapshots,
            final_state,
            pseudo,
            committed,
            timing: Timing {
                gpu_cycle_us,
                host_encode_submit_wait_us,
            },
        })
    }
}
