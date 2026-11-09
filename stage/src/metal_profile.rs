//! R55 stage clock: one command buffer; every stage is its own compute encoder with GPU
//! timestamps sampled at the encoder's start and end (the M5 supports stage-boundary
//! sampling, not dispatch-boundary). Encoders in one command buffer run in order, so the
//! samples bracket each stage exactly; the encoder boundary itself is the only overhead.
//! A diagnostic, never part of a served forward.
use crate::metal_ops::Obj;
use anyhow::{Context, Result, ensure};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;

pub type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// R55 differential cost: bit i set encodes stage i a second time (after a barrier) in the
/// joined layer encodings. Every listed stage is idempotent. Bits: 0 HC read (attention side),
/// 1 GDN projections, 2 GDN core, 3 GDN norm/gate, 4 GDN output projection, 5 HC write+read,
/// 6..=17 MLP stages 0..=11, 18 final HC write.
pub static DUP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn dup(bit: u32) -> bool {
    DUP.load(std::sync::atomic::Ordering::Relaxed) & (1u64 << bit) != 0
}
pub const DUP_STAGES: [&str; 19] = [
    "hc_read_attention", "gdn_projections", "gdn_core", "gdn_norm_gate", "gdn_output_projection",
    "hc_write_read_mlp", "mlp_router", "mlp_select", "mlp_expert_prepare", "mlp_expert_gate_up",
    "mlp_expert_mid", "mlp_expert_down", "mlp_expert_finish", "mlp_shared_gate_and_gate_proj",
    "mlp_shared_up_proj", "mlp_shared_activate", "mlp_shared_down_proj", "mlp_join", "hc_write_final",
];

pub struct StageClock {
    cb: Obj<dyn MTLCommandBuffer>,
    samples: Obj<dyn MTLCounterSampleBuffer>,
    labels: Vec<String>,
    cap: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Stage {
    pub label: String,
    pub gpu_us: f64,
    /// Time between the previous encoder's end and this encoder's start.
    pub gap_us: f64,
}

impl StageClock {
    pub fn new(queue: &Obj<dyn MTLCommandQueue>) -> Result<Self> {
        Self::with_capacity(queue, 2048)
    }
    pub fn with_capacity(queue: &Obj<dyn MTLCommandQueue>, cap: usize) -> Result<Self> {
        let device = queue.device();
        ensure!(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            "stage-boundary counter sampling unsupported"
        );
        let sets = device.counterSets().context("counter sets")?;
        let mut set = None;
        for i in 0..sets.count() {
            let s = sets.objectAtIndex(i);
            if s.name().to_string() == "timestamp" {
                set = Some(s);
            }
        }
        let set = set.context("timestamp counter set")?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(2 * cap) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| anyhow::anyhow!("counter sample buffer: {e}"))?;
        let cb = queue.commandBuffer().context("stage command buffer")?;
        Ok(Self { cb, samples, labels: vec![], cap })
    }
    pub fn stage(&mut self, label: impl Into<String>, f: impl FnOnce(&Encoder) -> Result<()>) -> Result<()> {
        self.stage_with(label, false, f)
    }
    /// `concurrent` selects MTLDispatchTypeConcurrent: dispatches may overlap unless the
    /// encoded code separates them with memory barriers.
    pub fn stage_with(
        &mut self,
        label: impl Into<String>,
        concurrent: bool,
        f: impl FnOnce(&Encoder) -> Result<()>,
    ) -> Result<()> {
        let i = self.labels.len();
        ensure!(i < self.cap, "stage clock capacity");
        let pass = MTLComputePassDescriptor::computePassDescriptor();
        if concurrent {
            pass.setDispatchType(MTLDispatchType::Concurrent);
        }
        let att = unsafe { pass.sampleBufferAttachments().objectAtIndexedSubscript(0) };
        att.setSampleBuffer(Some(&self.samples));
        unsafe {
            att.setStartOfEncoderSampleIndex(2 * i);
            att.setEndOfEncoderSampleIndex(2 * i + 1);
        }
        let e = self.cb.computeCommandEncoderWithDescriptor(&pass).context("stage encoder")?;
        let r = f(&e);
        e.endEncoding();
        r?;
        self.labels.push(label.into());
        Ok(())
    }
    pub fn finish(self) -> Result<Vec<Stage>> {
        let n = self.labels.len();
        ensure!(n > 0, "empty stage clock");
        self.cb.commit();
        self.cb.waitUntilCompleted();
        ensure!(self.cb.status() == MTLCommandBufferStatus::Completed, "stage clock: {:?}", self.cb.error());
        let data = unsafe { self.samples.resolveCounterRange(NSRange { location: 0, length: 2 * n }) }
            .context("resolve counters")?;
        let bytes = unsafe { data.as_bytes_unchecked() };
        ensure!(bytes.len() >= 16 * n, "counter data size");
        let ticks: Vec<u64> = bytes[..16 * n]
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(ticks.iter().all(|&t| t != 0 && t != u64::MAX), "unsampled counter");
        // Scale ticks to microseconds with the command buffer's own GPU span.
        let span_s = self.cb.GPUEndTime() - self.cb.GPUStartTime();
        let span_ticks = (ticks[2 * n - 1] - ticks[0]) as f64;
        ensure!(span_s > 0. && span_ticks > 0., "stage clock span");
        let us_per_tick = span_s * 1e6 / span_ticks;
        let mut out = vec![];
        for i in 0..n {
            let (s, e) = (ticks[2 * i], ticks[2 * i + 1]);
            ensure!(e >= s, "stage {} counter order", self.labels[i]);
            out.push(Stage {
                label: self.labels[i].clone(),
                gpu_us: (e - s) as f64 * us_per_tick,
                gap_us: if i == 0 { 0. } else { (s as f64 - ticks[2 * i - 1] as f64) * us_per_tick },
            });
        }
        Ok(out)
    }
}
