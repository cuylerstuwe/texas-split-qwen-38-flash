//! Native target attention: shared-input Q/K/V/index projections -> QSA ->
//! packed KV/SDPA/gate -> output projection. No host activation or mask handoff.
//! Diagnostic import/export remains separate from resident model transactions.
use crate::{
    metal_ops::{Buffer, DenseOps, DenseWeight, Obj, Phase, alloc, dense_scratch_bytes},
    metal_qsa::{self, Precision, Selector},
    metal_target_attention::{self, Arm as Core},
    qsa, target_attention as a,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, time::Instant};
const GUARD: usize = 64;
const BPOISON: u16 = 0x7fc1;
const FPOISON: u32 = 0x7fc12345;

#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub precision: Precision,
    pub core: Core,
    pub selector: Selector,
}
impl Arm {
    pub fn name(self) -> String {
        format!(
            "{}_{}_{}",
            self.precision.name(),
            self.core.name(),
            self.selector.name()
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Schedule {
    Joined,
    WaitBetween,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Joined => "joined",
            Self::WaitBetween => "wait_between_projections_and_attention",
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Stages {
    pub projected: a::Input,
    pub index: metal_qsa::Stages,
    pub attention: metal_target_attention::Stages,
    pub output: Vec<u16>,
}
pub struct Run {
    pub stages: Stages,
    pub state: a::State,
    pub gpu_us: f64,
    pub gpu_span_us: f64,
    pub encode_submit_wait_us: f64,
    pub command_buffers: usize,
    pub partial_values: usize,
}
pub struct Engine {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    index: metal_qsa::Engine,
    core: metal_target_attention::Engine,
    dense: DenseOps,
    weights: Vec<DenseWeight>,
    raw: Vec<Buffer>,
    scratch: Buffer,
    /// R55: private split-K scratch for the q/k/v projections (overlapped).
    scratch3: Vec<Buffer>,
    output: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
impl Engine {
    pub fn new(w: &a::Weights, iw: &qsa::Weights, capacity: usize) -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use exclusive guarded runner"
        );
        ensure!(w.layer == iw.layer, "attention/index layer alignment");
        let device = MTLCreateSystemDefaultDevice().context("full attention device")?;
        let queue = device.newCommandQueue().context("full attention queue")?;
        Self::new_on(w, iw, capacity, device, queue)
    }
    pub(crate) fn new_on(
        w: &a::Weights,
        iw: &qsa::Weights,
        capacity: usize,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        ensure!(w.layer == iw.layer, "attention/index layer alignment");
        let mut weights = vec![];
        let mut raw = vec![];
        let mut scratch = 0;
        let mut scratch3 = vec![];
        for (i, w) in [&w.q, &w.k, &w.v, &w.o].into_iter().enumerate() {
            weights.push(DenseWeight::new(&device, w)?);
            raw.push(alloc(&device, (a::MAX_ROWS * w.n + GUARD) * 4)?);
            scratch = scratch.max(dense_scratch_bytes(w.k, w.n, a::MAX_ROWS));
            if i < 3 {
                scratch3.push(alloc(&device, dense_scratch_bytes(w.k, w.n, a::MAX_ROWS))?);
            }
        }
        Ok(Self {
            index: metal_qsa::Engine::new_on(iw, capacity, device.clone(), queue.clone())?,
            core: metal_target_attention::Engine::new_on(
                &w.norms,
                capacity,
                device.clone(),
                queue.clone(),
            )?,
            dense: DenseOps::new(&device)?,
            weights,
            raw,
            scratch: alloc(&device, scratch)?,
            scratch3,
            output: alloc(&device, (a::MAX_ROWS * a::H + GUARD) * 2)?,
            device,
            queue,
            _single_owner: PhantomData,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"device":self.device.name().to_string(),"index":self.index.info(),"core":self.core.info(),"projection_weight_bytes":self.weights.iter().map(|w|w.bytes()).sum::<usize>(),"additional_workspace_bytes":self.raw.iter().map(|b|b.length()).sum::<usize>()+self.scratch.length()+self.output.length(),"joined_host_intermediate_handoffs":0,"dense_short_context_selector_skip":false,"full_attention":true,"whole_layer_or_model":false})
    }
    fn projections(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        initial: &qsa::State,
        rows: usize,
        accepted: usize,
        arm: Arm,
        input: Option<&Buffer>,
    ) -> Result<()> {
        if std::env::var("QWEN_R55_PROJ_OVERLAP").as_deref() != Ok("0") {
            // R55: the three GEMMs (private scratch) overlap, then the three reductions.
            for phase in [Phase::Gemm, Phase::Reduce] {
                for (i, out) in self.core.projection_buffers().iter().enumerate() {
                    self.dense.encode_phase(
                        e,
                        &self.weights[i],
                        input.unwrap_or(self.index.input_buffer()),
                        &self.raw[i],
                        out,
                        &self.scratch3[i],
                        rows,
                        0,
                        arm.precision.arm(),
                        phase,
                    )?;
                }
                crate::metal_ops::barrier(&e);
            }
        } else {
            for (i, out) in self.core.projection_buffers().iter().enumerate() {
                self.dense.encode(
                    e,
                    &self.weights[i],
                    input.unwrap_or(self.index.input_buffer()),
                    &self.raw[i],
                    out,
                    &self.scratch,
                    rows,
                    0,
                    arm.precision.arm(),
                )?;
                crate::metal_ops::barrier(&e);
            }
        }
        self.index.encode_external(
            e,
            initial,
            rows,
            Some(arm.precision),
            arm.selector,
            accepted,
            input,
        )
    }
    fn attention(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        initial: &a::State,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        let ids = if initial.len + rows > qsa::BUDGET + qsa::RATIO - 1 {
            Some(self.index.selection_buffer())
        } else {
            None
        };
        self.core.encode(e, initial, rows, arm.core, ids)?;
        self.dense.encode(
            e,
            &self.weights[3],
            self.core.gated_buffer(),
            &self.raw[3],
            &self.output,
            &self.scratch,
            rows,
            0,
            arm.precision.arm(),
        )?;
        crate::metal_ops::barrier(&e);
        Ok(())
    }
    pub fn run(
        &mut self,
        input: &[u16],
        initial: &a::State,
        index_initial: &qsa::State,
        accepted: usize,
        arm: Arm,
        schedule: Schedule,
        repeats: usize,
    ) -> Result<Run> {
        Ok(self
            .run_plan(
                input,
                initial,
                index_initial,
                accepted,
                &[arm],
                schedule,
                repeats,
            )?
            .0)
    }
    /// Bounded diagnostic queue; commands share scratch on one serial queue.
    /// Inputs/committed state remain fixed, each command rebuilds candidates.
    /// Only the final command's state survives; per-command timers are returned.
    pub fn run_queued(
        &mut self,
        input: &[u16],
        initial: &a::State,
        index_initial: &qsa::State,
        accepted: usize,
        arms: &[Arm],
        repeats: usize,
    ) -> Result<(Run, Vec<f64>)> {
        self.run_plan(
            input,
            initial,
            index_initial,
            accepted,
            arms,
            Schedule::Joined,
            repeats,
        )
    }
    fn run_plan(
        &mut self,
        input: &[u16],
        initial: &a::State,
        index_initial: &qsa::State,
        accepted: usize,
        arms: &[Arm],
        schedule: Schedule,
        repeats: usize,
    ) -> Result<(Run, Vec<f64>)> {
        ensure!(
            !arms.is_empty() && arms.len() <= 32,
            "attention queue bound"
        );
        ensure!(
            arms.len() == 1 || matches!(schedule, Schedule::Joined),
            "queued split schedule unsupported"
        );
        // Mixed scratch extents would invalidate the final trailing-canary
        // diagnostic. This queue experiment intentionally uses eight-way arms.
        ensure!(
            arms.len() == 1
                || arms
                    .iter()
                    .all(|a| matches!(a.core, Core::Simd8 | Core::Nax8)),
            "queued experiment requires common eight-way partial geometry"
        );
        let arm = *arms.last().unwrap();
        let admission = if arms
            .iter()
            .any(|a| matches!(a.precision, Precision::PackedF16x2))
        {
            Precision::PackedF16x2
        } else {
            Precision::Qmv
        };
        ensure!(
            (1..=4).contains(&repeats),
            "full attention repetition bound"
        );
        ensure!(
            initial.base == index_initial.base && initial.len == index_initial.len,
            "KV/index prefix alignment"
        );
        let rows = self
            .index
            .prepare(input, index_initial, Some(admission), accepted)?;
        self.core.prepare(initial, rows)?;
        self.poison();
        let t = Instant::now();
        let mut commands = vec![];
        match schedule {
            Schedule::Joined => {
                for &arm in arms {
                    let cb = self
                        .queue
                        .commandBuffer()
                        .context("queued attention command")?;
                    let e = cb
                        .computeCommandEncoder()
                        .context("queued attention encoder")?;
                    for _ in 0..repeats {
                        self.projections(&e, index_initial, rows, accepted, arm, None)?;
                        self.attention(&e, initial, rows, arm)?;
                    }
                    e.endEncoding();
                    commands.push(cb);
                }
                // Build every command before commit: no fallible host construction
                // or CPU buffer accesses while shared scratch is in flight.
                for cb in &commands {
                    cb.commit();
                }
                commands.last().unwrap().waitUntilCompleted();
            }
            Schedule::WaitBetween => {
                for _ in 0..repeats {
                    for front in [true, false] {
                        let cb = self
                            .queue
                            .commandBuffer()
                            .context("split attention command")?;
                        let e = cb
                            .computeCommandEncoder()
                            .context("split attention encoder")?;
                        if front {
                            self.projections(&e, index_initial, rows, accepted, arm, None)?;
                        } else {
                            self.attention(&e, initial, rows, arm)?;
                        }
                        e.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        ensure!(
                            cb.status() == MTLCommandBufferStatus::Completed,
                            "split attention failure {:?}",
                            cb.error()
                        );
                        commands.push(cb);
                    }
                }
            }
        }
        let encode_submit_wait_us = t.elapsed().as_secs_f64() * 1e6;
        let mut gpu_us = 0.;
        let mut timings = vec![];
        for c in &commands {
            ensure!(
                c.status() == MTLCommandBufferStatus::Completed,
                "full attention failure {:?}",
                c.error()
            );
            let us = (c.GPUEndTime() - c.GPUStartTime()) * 1e6;
            ensure!(us.is_finite() && us > 0., "full attention GPU timestamp");
            gpu_us += us;
            timings.push(us);
        }
        let gpu_span_us =
            (commands.last().unwrap().GPUEndTime() - commands[0].GPUStartTime()) * 1e6;
        let source = unsafe {
            std::slice::from_raw_parts(
                self.index.input_buffer().contents().cast::<u16>().as_ptr(),
                input.len(),
            )
        };
        ensure!(source == input, "full attention input mutated");
        let (stages, state, partial_values) =
            self.collect(initial, index_initial, rows, accepted, arm)?;
        Ok((
            Run {
                stages,
                state,
                gpu_us,
                gpu_span_us,
                encode_submit_wait_us,
                command_buffers: commands.len(),
                partial_values,
            },
            timings,
        ))
    }
    fn poison(&self) {
        for b in &self.raw {
            unsafe {
                std::slice::from_raw_parts_mut(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
                    .fill(FPOISON);
            }
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.output.contents().cast::<u16>().as_ptr(),
                self.output.length() / 2,
            )
            .fill(BPOISON);
        }
    }
    pub(crate) fn prepare_device(
        &mut self,
        initial: &a::State,
        index_initial: &qsa::State,
        rows: usize,
        accepted: usize,
    ) -> Result<()> {
        ensure!(
            initial.base == index_initial.base && initial.len == index_initial.len,
            "attention device prefix alignment"
        );
        self.index.prepare_device(index_initial, rows, accepted)?;
        self.core.prepare(initial, rows)?;
        self.poison();
        Ok(())
    }
    pub(crate) fn encode(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        initial: &a::State,
        index_initial: &qsa::State,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<()> {
        self.projections(e, index_initial, rows, accepted, arm, Some(input))?;
        self.attention(e, initial, rows, arm)
    }
    /// R58 profile: encode() split into timestamped stages: the q/k/v projections, the QSA
    /// indexer (projections, pool, scores, selection), and the attention core with the output
    /// projection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        input: &Buffer,
        initial: &a::State,
        index_initial: &qsa::State,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<()> {
        clock.stage_with("attn.qkv_projections", true, |e| {
            for phase in [Phase::Gemm, Phase::Reduce] {
                for (i, out) in self.core.projection_buffers().iter().enumerate() {
                    self.dense.encode_phase(e, &self.weights[i], input, &self.raw[i], out, &self.scratch3[i], rows, 0, arm.precision.arm(), phase)?;
                }
                crate::metal_ops::barrier(e);
            }
            Ok(())
        })?;
        clock.stage_with("attn.qsa_index", true, |e| {
            self.index.encode_external(e, index_initial, rows, Some(arm.precision), arm.selector, accepted, Some(input))
        })?;
        clock.stage_with("attn.core_and_output", true, |e| self.attention(e, initial, rows, arm))
    }
    /// R55 resident model: the next forward's old index tail is this forward's new one.
    pub(crate) fn roll_tail(&self, len_after: usize) -> Result<()> {
        self.index.roll_tail(len_after)
    }
    /// R55 resident model: index tail for any accepted prefix of the last forward.
    pub(crate) fn commit_tail(&self, len: usize, accepted: usize) -> Result<()> {
        self.index.commit_tail(len, accepted)
    }
    pub(crate) fn output_buffer(&self) -> &Buffer {
        &self.output
    }
    /// R58 prefill: (q / k / v / output projections, QSA indexer, attention core).
    pub(crate) fn prefill_parts(&self) -> (&[DenseWeight], &metal_qsa::Engine, &metal_target_attention::Engine) {
        (&self.weights, &self.index, &self.core)
    }
    pub(crate) fn collect(
        &self,
        initial: &a::State,
        index_initial: &qsa::State,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<(Stages, a::State, usize)> {
        for (b, w) in self.raw.iter().zip(&self.weights) {
            let v = unsafe {
                std::slice::from_raw_parts(b.contents().cast::<u32>().as_ptr(), b.length() / 4)
            };
            ensure!(
                v[..rows * w.n]
                    .iter()
                    .all(|&x| f32::from_bits(x).is_finite())
                    && v[rows * w.n..].iter().all(|&x| x == FPOISON),
                "projection FP32 finite/canary"
            );
        }
        let out = unsafe {
            std::slice::from_raw_parts(
                self.output.contents().cast::<u16>().as_ptr(),
                self.output.length() / 2,
            )
        };
        ensure!(
            out[..rows * a::H]
                .iter()
                .all(|&x| bf16::from_bits(x).is_finite())
                && out[rows * a::H..].iter().all(|&x| x == BPOISON),
            "output finite/canary"
        );
        let (attention, state, partial_values) = self.core.collect(initial, rows, arm.core)?;
        let stages = Stages {
            projected: self.core.projected(rows)?,
            index: self
                .index
                .collect(index_initial, rows, Some(arm.precision), accepted)?,
            attention,
            output: out[..rows * a::H].to_vec(),
        };
        Ok((stages, state, partial_values))
    }
}
