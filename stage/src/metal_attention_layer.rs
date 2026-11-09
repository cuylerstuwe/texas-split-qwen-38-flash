//! Whole attention layer on one queue: HC/QSA/attention/HC/full EXL3 MLP/HC.
use crate::{
    hc, metal_full_attention as attention,
    metal_hc::{Hc, HcOps, Precision as HcPrecision},
    metal_mlp,
    metal_ops::{Buffer, Obj, alloc, upload},
    moe::FullExpertBank,
    qsa, target_attention as a,
    target_attention_layer::Weights,
    target_mlp,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, time::Instant};
const GUARD: usize = 64;
const POISON: u16 = 0x7fc1;
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub hc: HcPrecision,
    pub attention: attention::Arm,
}
impl Arm {
    pub fn name(self) -> String {
        format!("hc_{}_{}", self.hc.name(), self.attention.name())
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
            Self::WaitBetween => "wait_before_mlp",
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Mlp {
    pub logits: Vec<u16>,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
    pub routed: Vec<f32>,
    pub shared: Vec<u16>,
    pub output: Vec<u16>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Stages {
    pub pre: hc::Read,
    pub attention: attention::Stages,
    pub stream: Vec<u16>,
    pub post: hc::Read,
    pub mlp: Mlp,
    pub output: Vec<u16>,
}
pub struct Run {
    pub stages: Stages,
    pub state: a::State,
    pub gpu_us: f64,
    pub gpu_span_us: f64,
    pub encode_submit_wait_us: f64,
    pub command_buffers: usize,
}
pub struct Engine {
    queue: Obj<dyn MTLCommandQueue>,
    hc_ops: HcOps,
    hcs: [Hc; 2],
    attention: attention::Engine,
    mlp_ops: metal_mlp::Ops,
    mlp_weights: metal_mlp::Weights,
    mlp_work: metal_mlp::Workspace,
    input: Buffer,
    stream: Buffer,
    mlp_output: Buffer,
    output: Buffer,
    _single_owner: PhantomData<Rc<()>>,
}
fn poison(b: &Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(b.contents().cast::<u16>().as_ptr(), b.length() / 2)
            .fill(POISON);
    }
}
fn read(b: &Buffer, n: usize) -> Result<Vec<u16>> {
    ensure!(n <= b.length() / 2, "layer read bound");
    let v =
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), b.length() / 2) };
    ensure!(
        v[..n].iter().all(|&v| bf16::from_bits(v).is_finite())
            && v[n..].iter().all(|&v| v == POISON),
        "layer finite/canary"
    );
    Ok(v[..n].to_vec())
}
impl Engine {
    pub fn new(w: &Weights, bank: &FullExpertBank, capacity: usize) -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use guarded runner"
        );
        ensure!(w.layer == bank.layer, "layer/expert identity");
        let device = MTLCreateSystemDefaultDevice().context("attention layer device")?;
        let queue = device.newCommandQueue().context("attention layer queue")?;
        Self::new_on(w, metal_mlp::Experts::Bank(bank), capacity, device, queue)
    }
    /// R55 resident model: an attention layer on the model's shared device and queue.
    pub(crate) fn new_on(
        w: &Weights,
        experts: metal_mlp::Experts<'_>,
        capacity: usize,
        device: Obj<dyn MTLDevice>,
        queue: Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        ensure!(
            std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"),
            "use guarded runner"
        );
        ensure!(w.layer == experts.layer(), "layer/expert identity");
        Ok(Self {
            hc_ops: HcOps::new(&device)?,
            hcs: [Hc::new(&device, &w.hc[0])?, Hc::new(&device, &w.hc[1])?],
            attention: attention::Engine::new_on(
                &w.attention,
                &w.index,
                capacity,
                device.clone(),
                queue.clone(),
            )?,
            mlp_ops: metal_mlp::Ops::new(&device)?,
            mlp_weights: metal_mlp::Weights::new_from(&device, experts, &w.mlp)?,
            mlp_work: metal_mlp::Workspace::new(&device)?,
            input: alloc(&device, (hc::MAX_ROWS * hc::F + GUARD) * 2)?,
            stream: alloc(&device, (hc::MAX_ROWS * hc::F + GUARD) * 2)?,
            mlp_output: alloc(&device, (hc::MAX_ROWS * hc::H + GUARD) * 2)?,
            output: alloc(&device, (hc::MAX_ROWS * hc::F + GUARD) * 2)?,
            queue,
            _single_owner: PhantomData,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"attention":self.attention.info(),"hc_bytes":self.hcs.iter().map(Hc::bytes).sum::<usize>(),"mlp_weight_bytes":self.mlp_weights.bytes(),"mlp_workspace_bytes":self.mlp_work.bytes(),"layer_stream_bytes":self.input.length()+self.stream.length()+self.mlp_output.length()+self.output.length(),"host_intermediate_handoffs":0,"fused_HC_write_norm":true,"routing":"staged BF16","shared":"QMV","whole_attention_layer":true,"whole_model":false})
    }
    fn front(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        initial: &a::State,
        index: &qsa::State,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<()> {
        self.hc_ops
            .read(e, &self.hcs[0], &self.input, rows, arm.hc, None)?;
        self.attention.encode(
            e,
            self.hcs[0].mixed(),
            initial,
            index,
            rows,
            accepted,
            arm.attention,
        )?;
        self.hc_ops.read(
            e,
            &self.hcs[1],
            &self.input,
            rows,
            arm.hc,
            Some((
                self.attention.output_buffer(),
                self.hcs[0].inject(),
                &self.stream,
            )),
        )
    }
    /// R55 resident model: the whole layer from `input`, appending this forward's rows to
    /// the resident KV cache and index pool at `len` (metadata-only states). The index
    /// tail is built for `accepted` rows (the panel path accepts every row).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_resident(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        len: usize,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<()> {
        ensure!(input.length() >= rows * hc::F * 2, "attention layer input");
        let initial = a::State { base: 0, len, k: a::Packed::empty(), v: a::Packed::empty() };
        let index = qsa::State { base: 0, len, pooled: vec![], tail: vec![] };
        self.hc_ops.read(e, &self.hcs[0], input, rows, arm.hc, None)?;
        self.attention
            .encode(e, self.hcs[0].mixed(), &initial, &index, rows, accepted, arm.attention)?;
        self.hc_ops.read(
            e,
            &self.hcs[1],
            input,
            rows,
            arm.hc,
            Some((self.attention.output_buffer(), self.hcs[0].inject(), &self.stream)),
        )?;
        self.tail(e, rows)
    }
    /// R58 profile: encode_resident split into timestamped stages (HC, attention parts, MLP).
    pub(crate) fn encode_resident_profiled(
        &self,
        clock: &mut crate::metal_profile::StageClock,
        input: &Buffer,
        len: usize,
        rows: usize,
        accepted: usize,
        arm: Arm,
    ) -> Result<()> {
        let initial = a::State { base: 0, len, k: a::Packed::empty(), v: a::Packed::empty() };
        let index = qsa::State { base: 0, len, pooled: vec![], tail: vec![] };
        clock.stage_with("attn.hc_read_attention", true, |e| self.hc_ops.read(e, &self.hcs[0], input, rows, arm.hc, None))?;
        self.attention.encode_profiled(clock, self.hcs[0].mixed(), &initial, &index, rows, accepted, arm.attention)?;
        clock.stage_with("attn.hc_write_read_mlp", true, |e| {
            self.hc_ops.read(e, &self.hcs[1], input, rows, arm.hc, Some((self.attention.output_buffer(), self.hcs[0].inject(), &self.stream)))
        })?;
        clock.stage_with("attn.mlp_and_write", true, |e| self.tail(e, rows))
    }
    /// R58 lead 1: this layer's last routed ids and weights.
    pub(crate) fn moe_routes(&self, rows: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        self.mlp_work.routes(rows)
    }
    pub(crate) fn output(&self) -> &Buffer {
        &self.output
    }
    /// R58 prefill: (attention-side HC, MLP-side HC, attention, MLP weights).
    pub(crate) fn prefill_parts(&self) -> (&Hc, &Hc, &attention::Engine, &metal_mlp::Weights) {
        (&self.hcs[0], &self.hcs[1], &self.attention, &self.mlp_weights)
    }
    /// R58 diagnostic: the post-attention streams, MLP input and MLP output of the last forward.
    pub(crate) fn diag_buffers(&self) -> (&Buffer, &Buffer, &Buffer) {
        (&self.stream, self.hcs[1].mixed(), &self.mlp_output)
    }
    /// Empty cache and index (once, at reset).
    pub(crate) fn reset_state(&mut self) -> Result<()> {
        self.attention
            .prepare_device(&a::State::empty(0)?, &qsa::State::empty(0)?, 1, 0)
    }
    /// After a forward that appended its rows at `len`: keep the first `accepted` (the KV
    /// cache and index pool commit by length; the raw index tail is rebuilt).
    pub(crate) fn commit(&self, len: usize, accepted: usize) -> Result<()> {
        self.attention.commit_tail(len, accepted)
    }
    fn tail(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, rows: usize) -> Result<()> {
        self.mlp_ops.encode(
            e,
            &self.mlp_weights,
            &self.mlp_work,
            self.hcs[1].mixed(),
            self.hcs[1].mixed(),
            &self.mlp_output,
            rows,
            target_mlp::RoutingArithmetic::StagedBf16,
            if std::env::var("QWEN_R55_SHARED").as_deref() == Ok("auto") {
                metal_mlp::SharedPrecision::Auto
            } else {
                metal_mlp::SharedPrecision::Qmv
            },
        )?;
        self.hc_ops.write(
            e,
            &self.stream,
            &self.mlp_output,
            self.hcs[1].inject(),
            &self.output,
            rows,
        );
        Ok(())
    }
    pub fn run(
        &mut self,
        input: &[u16],
        initial: &a::State,
        index: &qsa::State,
        accepted: usize,
        arm: Arm,
        schedule: Schedule,
        repeats: usize,
    ) -> Result<Run> {
        ensure!(
            !input.is_empty()
                && input.len() % hc::F == 0
                && input.len() / hc::F <= 8
                && input.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "bounded attention layer input"
        );
        ensure!((1..=4).contains(&repeats), "layer repetitions");
        let rows = input.len() / hc::F;
        self.attention
            .prepare_device(initial, index, rows, accepted)?;
        for h in &self.hcs {
            h.poison();
        }
        self.mlp_work.poison();
        for b in [&self.input, &self.stream, &self.mlp_output, &self.output] {
            poison(b);
        }
        upload(&self.input, input)?;
        let start = Instant::now();
        let mut commands = vec![];
        match schedule {
            Schedule::Joined => {
                let cb = self.queue.commandBuffer().context("layer command")?;
                let e = cb.computeCommandEncoder().context("layer encoder")?;
                for _ in 0..repeats {
                    self.front(&e, initial, index, rows, accepted, arm)?;
                    self.tail(&e, rows)?;
                }
                e.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                commands.push(cb);
            }
            Schedule::WaitBetween => {
                for _ in 0..repeats {
                    for front in [true, false] {
                        let cb = self.queue.commandBuffer().context("split layer command")?;
                        let e = cb.computeCommandEncoder().context("split layer encoder")?;
                        if front {
                            self.front(&e, initial, index, rows, accepted, arm)?;
                        } else {
                            self.tail(&e, rows)?;
                        }
                        e.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        ensure!(
                            cb.status() == MTLCommandBufferStatus::Completed,
                            "split layer failure {:?}",
                            cb.error()
                        );
                        commands.push(cb);
                    }
                }
            }
        }
        let encode_submit_wait_us = start.elapsed().as_secs_f64() * 1e6;
        let mut gpu_us = 0.;
        for c in &commands {
            ensure!(
                c.status() == MTLCommandBufferStatus::Completed,
                "layer failure {:?}",
                c.error()
            );
            let us = (c.GPUEndTime() - c.GPUStartTime()) * 1e6;
            ensure!(us.is_finite() && us > 0., "layer GPU timestamp");
            gpu_us += us;
        }
        let gpu_span_us =
            (commands.last().unwrap().GPUEndTime() - commands[0].GPUStartTime()) * 1e6;
        ensure!(
            read(&self.input, input.len())? == input,
            "layer input mutated"
        );
        let (attention, state, _) =
            self.attention
                .collect(initial, index, rows, accepted, arm.attention)?;
        let m = self.mlp_work.collect(&self.mlp_output, rows)?;
        let mlp = Mlp {
            logits: m.logits,
            ids: m.ids,
            scores: m.scores,
            routed: m.routed[6].clone(),
            shared: m.shared[6].clone(),
            output: m.output,
        };
        let stages = Stages {
            pre: self.hcs[0].collect(rows)?,
            attention,
            stream: read(&self.stream, rows * hc::F)?,
            post: self.hcs[1].collect(rows)?,
            mlp,
            output: read(&self.output, rows * hc::F)?,
        };
        Ok(Run {
            stages,
            state,
            gpu_us,
            gpu_span_us,
            encode_submit_wait_us,
            command_buffers: commands.len(),
        })
    }
}
