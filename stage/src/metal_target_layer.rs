//! Native GDN layer: HC read -> GDN -> HC write/read -> full EXL3 MLP ->
//! final HC write. No CPU activation/routing handoffs in the joined schedule.
//! GDN candidate states remain uncommitted until the caller accepts a prefix.
use crate::{
    gdn, hc,
    metal_hc::{self, Arm as FrontArm, MetalFront},
    metal_mlp::{self, SharedPrecision},
    metal_ops::{Buffer, Obj, alloc},
    moe::FullExpertBank,
    target_layer::Weights,
    target_mlp::RoutingArithmetic,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::{marker::PhantomData, rc::Rc, time::Instant};
const GUARD: usize = 64;
const SENTINEL: u16 = 0x7fc1;
#[derive(Clone, Copy, Debug)]
pub enum Schedule {
    Joined,
    WaitBetween,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Joined => "joined",
            Self::WaitBetween => "wait_between_front_and_tail",
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub front: FrontArm,
    pub scores: RoutingArithmetic,
    pub shared: SharedPrecision,
}
impl Arm {
    pub fn name(self) -> String {
        format!(
            "{}_{}_{}",
            self.front.name(),
            if matches!(self.scores, RoutingArithmetic::StagedBf16) {
                "staged_scores"
            } else {
                "selected_f32_scores"
            },
            self.shared.name()
        )
    }
}
pub struct Run {
    pub front: metal_hc::Run,
    pub mlp: metal_mlp::Stages,
    pub output: Vec<u16>,
    pub gpu_us: f64,
    pub gpu_span_us: f64,
    pub encode_submit_wait_us: f64,
    pub command_buffers: usize,
}
pub struct Engine {
    queue: Obj<dyn MTLCommandQueue>,
    front: MetalFront,
    ops: metal_mlp::Ops,
    weights: metal_mlp::Weights,
    scratch: metal_mlp::Workspace,
    mlp_out: Buffer,
    out: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
fn poison(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(SENTINEL);
    }
}
fn read(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "layer output bound");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        v[n..].iter().all(|&v| v == SENTINEL),
        "layer output trailing canary"
    );
    ensure!(
        v[..n].iter().all(|&v| bf16::from_bits(v).is_finite()),
        "layer finite output"
    );
    Ok(v[..n].to_vec())
}
impl Engine {
    pub fn new(w: &Weights, bank: &FullExpertBank) -> Result<Self> {
        ensure!(
            w.layer == bank.layer && w.layer % 4 != 3 && w.layer != 1,
            "GDN layer without PLE"
        );
        let device = MTLCreateSystemDefaultDevice().context("target layer Metal device")?;
        let queue = device.newCommandQueue().context("target layer queue")?;
        Self::new_on(w, metal_mlp::Experts::Bank(bank), device, queue)
    }
    /// R55 resident model: a GDN layer (layer 1's body included; its PLE stage is
    /// separate) on the model's shared device and queue.
    pub(crate) fn new_on(
        w: &Weights,
        experts: metal_mlp::Experts<'_>,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use exclusive guarded runner"
        );
        ensure!(w.layer == experts.layer() && w.layer % 4 != 3, "GDN layer");
        let front = MetalFront::new_on(&w.hc[0], &w.gdn, &w.hc[1], device.clone(), queue.clone())?;
        let ops = metal_mlp::Ops::new(&device)?;
        let weights = metal_mlp::Weights::new_from(&device, experts, &w.mlp)?;
        let scratch = metal_mlp::Workspace::new(&device)?;
        let mlp_out = alloc(&device, (hc::MAX_ROWS * hc::H + GUARD) * 2)?;
        let out = alloc(&device, (hc::MAX_ROWS * hc::F + GUARD) * 2)?;
        Ok(Self {
            queue,
            front,
            ops,
            weights,
            scratch,
            mlp_out,
            out,
            _single_owner: PhantomData,
        })
    }
    /// R55 resident model: the whole layer from `input` into this layer's output buffer.
    pub(crate) fn encode_resident(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        self.front.encode_from(e, input, rows, arm.front)?;
        self.tail(e, rows, arm)
    }
    /// R58 profile: encode_resident with one timestamped encoder per stage (HC, GDN, MLP).
    pub(crate) fn encode_resident_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        input: &Buffer,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        self.front.encode_profiled_from(clock, input, rows, arm.front)?;
        self.ops.encode_profiled(clock, &self.weights, &self.scratch, self.front.mlp_input(), self.front.mlp_input(), &self.mlp_out, rows, arm.scores, arm.shared)?;
        clock.stage("hc.write_mlp", |e| self.front.finish(e, &self.mlp_out, &self.out, rows))
    }
    pub(crate) fn output(&self) -> &Buffer {
        &self.out
    }
    /// R58 prefill: the layer's front (HC + GDN) and MLP weights.
    pub(crate) fn prefill_parts(&self) -> (&MetalFront, &metal_mlp::Weights) {
        (&self.front, &self.weights)
    }
    /// R58 diagnostic: the MLP output rows (BF16 rows x H) of the last forward.
    /// R58 lead 1: this layer's last routed ids and weights.
    pub(crate) fn moe_routes(&self, rows: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        self.scratch.routes(rows)
    }
    pub(crate) fn mlp_output(&self) -> &Buffer {
        &self.mlp_out
    }
    pub(crate) fn reset_state(&mut self) -> Result<()> {
        self.front.reset_state(&gdn::Initial {
            conv: vec![0; 3 * gdn::C],
            state: vec![0.; gdn::STATE],
        })
    }
    pub(crate) fn commit(&self, accepted: usize, rows: usize) -> Result<()> {
        self.front.commit(accepted, rows)
    }
    pub(crate) fn commit_split(&self, accepted: usize, rows: usize) -> Result<Option<(usize, usize, usize)>> {
        self.front.commit_split(accepted, rows)
    }
    pub(crate) fn set_zero_copy(&self, on: bool) {
        self.front.set_zero_copy(on)
    }
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"front":self.front.description(),"mlp_weight_bytes":self.weights.bytes(),"mlp_workspace_bytes":self.scratch.bytes(),"tail_output_bytes":self.mlp_out.length()+self.out.length(),"joined_host_intermediate_handoffs":0,"state":"uncommitted full FP32 snapshots, per position","scope":"native HC/GDN/HC/full512EXL3MLP/HC GDN-layer; QSA/PLE/full model not implied"})
    }
    fn tail(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        rows: usize,
        arm: Arm,
    ) -> Result<()> {
        self.ops.encode(
            e,
            &self.weights,
            &self.scratch,
            self.front.mlp_input(),
            self.front.mlp_input(),
            &self.mlp_out,
            rows,
            arm.scores,
            arm.shared,
        )?;
        self.front.finish(e, &self.mlp_out, &self.out, rows)
    }
    /// R55 profile, all in one command buffer so the GPU stays busy. Each mode runs
    /// `reps` times in order: "serial" and "concurrent" are whole joined layers (one
    /// encoder each, the normal barriers inside, serial or concurrent dispatch); "split"
    /// is one encoder per stage. The layer only reads its input and initial state, so
    /// every repetition computes the same values; the output is from the last one.
    pub fn profile(
        &mut self,
        input: &[u16],
        initial: &gdn::Initial,
        arm: Arm,
        modes: &[&str],
        reps: usize,
    ) -> Result<(Vec<crate::metal_profile::Stage>, Vec<u16>)> {
        ensure!((1..=64).contains(&reps), "profile repetitions");
        let rows = self.front.prepare(input, initial)?;
        self.scratch.poison();
        poison(&self.mlp_out);
        poison(&self.out);
        let mut clock = crate::metal_profile::StageClock::new(self.front.queue())?;
        for &mode in modes {
            for _ in 0..reps {
                match mode {
                    "serial" | "concurrent" => clock.stage_with(mode, mode == "concurrent", |e| {
                        self.front.encode(e, rows, arm.front)?;
                        self.tail(e, rows, arm)
                    })?,
                    "split" => {
                        self.front.encode_profiled(&mut clock, rows, arm.front)?;
                        self.ops.encode_profiled(
                            &mut clock,
                            &self.weights,
                            &self.scratch,
                            self.front.mlp_input(),
                            self.front.mlp_input(),
                            &self.mlp_out,
                            rows,
                            arm.scores,
                            arm.shared,
                        )?;
                        clock.stage("hc.write_mlp", |e| self.front.finish(e, &self.mlp_out, &self.out, rows))?;
                    }
                    _ => anyhow::bail!("profile mode"),
                }
            }
        }
        let stages = clock.finish()?;
        let output = read(&self.out, rows * hc::F)?;
        Ok((stages, output))
    }
    pub fn run(
        &mut self,
        input: &[u16],
        initial: &gdn::Initial,
        arm: Arm,
        schedule: Schedule,
        repetitions: usize,
    ) -> Result<Run> {
        ensure!((1..=4).contains(&repetitions), "bounded layer repetitions");
        let rows = self.front.prepare(input, initial)?;
        self.scratch.poison();
        poison(&self.mlp_out);
        poison(&self.out);
        let t = Instant::now();
        let mut commands = vec![];
        match schedule {
            Schedule::Joined => {
                let cb = self.queue.commandBuffer().context("joined layer command")?;
                let e = cb.computeCommandEncoder().context("joined layer encoder")?;
                for _ in 0..repetitions {
                    self.front.encode(&e, rows, arm.front)?;
                    self.tail(&e, rows, arm)?;
                }
                e.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                commands.push(cb);
            }
            Schedule::WaitBetween => {
                for _ in 0..repetitions {
                    for front in [true, false] {
                        let cb = self.queue.commandBuffer().context("split layer command")?;
                        let e = cb.computeCommandEncoder().context("split layer encoder")?;
                        if front {
                            self.front.encode(&e, rows, arm.front)?;
                        } else {
                            self.tail(&e, rows, arm)?;
                        }
                        e.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        ensure!(
                            cb.status() == MTLCommandBufferStatus::Completed,
                            "split layer command: {:?}",
                            cb.error()
                        );
                        commands.push(cb);
                    }
                }
            }
        }
        let encode_submit_wait_us = t.elapsed().as_secs_f64() * 1e6;
        let mut gpu_us = 0.;
        for cb in &commands {
            ensure!(
                cb.status() == MTLCommandBufferStatus::Completed,
                "layer command: {:?}",
                cb.error()
            );
            let span = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
            ensure!(span.is_finite() && span > 0., "layer GPU timer");
            gpu_us += span;
        }
        let gpu_span_us =
            (commands.last().unwrap().GPUEndTime() - commands[0].GPUStartTime()) * 1e6;
        let front = self.front.collect(rows)?;
        let mlp = self.scratch.collect(&self.mlp_out, rows)?;
        let output = read(&self.out, rows * hc::F)?;
        Ok(Run {
            front,
            mlp,
            output,
            gpu_us,
            gpu_span_us,
            encode_submit_wait_us,
            command_buffers: commands.len(),
        })
    }
}
