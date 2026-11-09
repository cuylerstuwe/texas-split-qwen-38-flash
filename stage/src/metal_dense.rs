//! Single-owner synchronous affine8 Metal component. Persistent packed buffers;
//! no MLX host, full-model runtime or GPU matrix materialization.
use crate::dense::{Affine8, MAX_ROWS};
pub use crate::metal_ops::Arm;
use crate::metal_ops::{Buffer, DenseOps, DenseWeight, Obj, alloc, dense_scratch_bytes, upload};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use objc2_metal::*;
use std::{marker::PhantomData, rc::Rc, time::Instant};
const GUARD: usize = 64;
const SENTINEL: u32 = 0x7fc12345;
const MAX_BANK: usize = 8;
pub struct MetalDense {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    ops: DenseOps,
    x: Buffer,
    weights: Vec<DenseWeight>,
    y: Buffer,
    ybf: Buffer,
    sums: Buffer,
    k: usize,
    n: usize,
    _single_owner: PhantomData<Rc<()>>,
}
pub struct Run {
    pub output: Vec<f32>,
    pub bf16_output: Vec<f32>,
    pub stage_us: f64,
    pub submit_wait_us: f64,
    pub hot_us: f64,
    pub gpu_us: f64,
}
impl MetalDense {
    pub fn new(weight: &Affine8) -> Result<Self> {
        Self::new_bank(std::slice::from_ref(weight))
    }
    /// At most eight equal-shape, packed real matrices, compiled once. Per-matrix
    /// quantization group sizes may differ. Sequence
    /// outputs occupy distinct slots; this is not a dependent model layer chain.
    pub fn new_bank(bank: &[Affine8]) -> Result<Self> {
        ensure!((1..=MAX_BANK).contains(&bank.len()), "dense bank size");
        let weight = &bank[0];
        for w in bank {
            w.validate()?;
            ensure!(
                (w.k, w.n) == (weight.k, weight.n),
                "dense bank geometry differs"
            );
        }
        let device = MTLCreateSystemDefaultDevice().context("dense Metal device")?;
        ensure!(
            device.name().to_string().contains("M5"),
            "MPP experiment requires observed M5"
        );
        let queue = device.newCommandQueue().context("dense queue")?;
        let ops = DenseOps::new(&device)?;
        let weights = bank
            .iter()
            .map(|w| DenseWeight::new(&device, w))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x: alloc(&device, MAX_ROWS * weight.k * 2)?,
            y: alloc(&device, (bank.len() * MAX_ROWS * weight.n + GUARD) * 4)?,
            ybf: alloc(&device, (bank.len() * MAX_ROWS * weight.n + GUARD) * 2)?,
            sums: alloc(&device, dense_scratch_bytes(weight.k, weight.n, MAX_ROWS))?,
            device,
            queue,
            ops,
            weights,
            k: weight.k,
            n: weight.n,
            _single_owner: PhantomData,
        })
    }
    pub fn description(&self) -> serde_json::Value {
        let weight_bytes = self.weights.iter().map(DenseWeight::bytes).sum::<usize>();
        let allocated_bytes = [&self.x, &self.y, &self.ybf, &self.sums]
            .iter()
            .map(|b| b.length())
            .sum::<usize>()
            + weight_bytes;
        serde_json::json!({"device":self.device.name().to_string(),"allocated_bytes":allocated_bytes,"bank_count":self.weights.len(),"weight_bytes":weight_bytes,
            "original_weight_bytes":self.weights.iter().map(DenseWeight::original_bytes).sum::<usize>(),
            "alternate_repacked_bytes":self.weights.iter().map(DenseWeight::fragment_bytes).sum::<usize>(),
            "groups":self.weights.iter().map(|w|w.group).collect::<Vec<_>>(),"pipeline_threadgroup_bytes":self.ops.threadgroup_bytes(),"safe_math":true})
    }
    pub fn run(&mut self, input: &[u16], arm: Arm, repetitions: usize) -> Result<Run> {
        self.run_sequence(input, arm, &[0], repetitions)
    }
    /// Queue every selected matrix against the same input, retaining every
    /// result in sequence order. Each repeat reuses its own output slots only
    /// after a resource barrier. Full input upload and all encoding are timed.
    pub fn run_sequence(
        &mut self,
        input: &[u16],
        arm: Arm,
        order: &[usize],
        repetitions: usize,
    ) -> Result<Run> {
        self.run_schedule(input, arm, order, repetitions, false)
    }
    /// Blocked repeats each matrix before moving on; interleaved cycles through
    /// the whole bank before repeating. Both perform the same projections and
    /// keep each slot's last result. This does not impose model dependencies.
    pub fn run_schedule(
        &mut self,
        input: &[u16],
        arm: Arm,
        order: &[usize],
        repetitions: usize,
        blocked: bool,
    ) -> Result<Run> {
        Ok(self
            .run_queued(input, &[arm], order, repetitions, blocked)?
            .0)
    }
    /// Queue complete bank commands without CPU diagnostic gaps. Only the final
    /// command's arrays survive; every command returns its own GPU duration.
    pub fn run_queued(
        &mut self,
        input: &[u16],
        arms: &[Arm],
        order: &[usize],
        repetitions: usize,
        blocked: bool,
    ) -> Result<(Run, Vec<f64>)> {
        ensure!(!arms.is_empty() && arms.len() <= 48, "dense queued bound");
        ensure!(
            !input.is_empty()
                && input.len() % self.k == 0
                && input.len() / self.k <= MAX_ROWS
                && (1..=16).contains(&repetitions),
            "dense run shape"
        );
        ensure!(
            !order.is_empty()
                && order.len() <= self.weights.len()
                && order.iter().all(|&i| i < self.weights.len()),
            "dense sequence bounds"
        );
        ensure!(
            input.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "dense finite input"
        );
        if arms.iter().any(|a| {
            matches!(
                a,
                Arm::MppF16 | Arm::MppF16x2 | Arm::Fragment { .. } | Arm::FragmentPacked { .. }
            )
        }) {
            ensure!(
                input
                    .iter()
                    .all(|&v| f16::from_f32(bf16::from_bits(v).to_f32()).is_finite()),
                "candidate half input overflow"
            );
        }
        let rows = input.len() / self.k;
        let per_step = rows * self.n;
        let count = per_step * order.len();
        // Poison active writes and all trailing cells before the timing scope.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.y.contents().cast::<u32>().as_ptr(),
                self.y.length() / 4,
            )
            .fill(SENTINEL);
            std::slice::from_raw_parts_mut(
                self.ybf.contents().cast::<u16>().as_ptr(),
                self.ybf.length() / 2,
            )
            .fill(0x7fc1);
        }
        let hot = Instant::now();
        let stage = Instant::now();
        upload(&self.x, input)?;
        let stage_us = stage.elapsed().as_secs_f64() * 1e6;
        let submit = Instant::now();
        let mut commands = Vec::with_capacity(arms.len());
        for &arm in arms {
            let cb = self.queue.commandBuffer().context("dense command buffer")?;
            let enc = cb.computeCommandEncoder().context("dense encoder")?;
            for step in 0..repetitions * order.len() {
                let slot = if blocked {
                    step / repetitions
                } else {
                    step % order.len()
                };
                self.ops.encode(
                    &enc,
                    &self.weights[order[slot]],
                    &self.x,
                    &self.y,
                    &self.ybf,
                    &self.sums,
                    rows,
                    slot * per_step,
                    arm,
                )?;
                crate::metal_ops::barrier(&enc);
            }
            enc.endEncoding();
            commands.push(cb);
        }
        // Finish all fallible encoding before commit. Resources are retained
        // across the entire serial queue, with no CPU access while in flight.
        for cb in &commands {
            cb.commit();
        }
        let cb = commands.last().unwrap();
        cb.waitUntilCompleted();
        ensure!(
            cb.status() == MTLCommandBufferStatus::Completed,
            "dense GPU error {:?}: {:?}",
            cb.status(),
            cb.error()
        );
        let submit_wait_us = submit.elapsed().as_secs_f64() * 1e6;
        let hot_us = hot.elapsed().as_secs_f64() * 1e6;
        let timings = commands
            .iter()
            .map(|cb| {
                ensure!(
                    cb.status() == MTLCommandBufferStatus::Completed,
                    "dense queued GPU error {:?}",
                    cb.error()
                );
                let us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
                ensure!(us.is_finite() && us > 0., "dense queued timer");
                Ok(us)
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_us = (cb.GPUEndTime() - commands[0].GPUStartTime()) * 1e6;
        ensure!(
            gpu_us.is_finite() && gpu_us > 0.,
            "missing dense GPU timing"
        );
        let raw = unsafe {
            std::slice::from_raw_parts(
                self.y.contents().cast::<u32>().as_ptr(),
                self.y.length() / 4,
            )
        };
        let braw = unsafe {
            std::slice::from_raw_parts(
                self.ybf.contents().cast::<u16>().as_ptr(),
                self.ybf.length() / 2,
            )
        };
        ensure!(
            raw[count..].iter().all(|&v| v == SENTINEL)
                && braw[count..].iter().all(|&v| v == 0x7fc1),
            "dense trailing write guard changed"
        );
        let output = raw[..count].iter().map(|&v| f32::from_bits(v)).collect();
        let bf16_output = braw[..count]
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        Ok((
            Run {
                output,
                bf16_output,
                stage_us,
                submit_wait_us,
                hot_us,
                gpu_us,
            },
            timings,
        ))
    }
}
