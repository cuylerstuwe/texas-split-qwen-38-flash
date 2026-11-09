//! Qualified R22 packed K1.75 projections, with optional original512 grouping.
use crate::cuda::{Arg, Buffer, Device, Kernel, Module, PreparedCall};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub const SHA: &str = "0a16beea75145301b715b6f5faf713c9fb458c383c4fdf6d39403d77ec94da4b";
const SLOTS: usize = 160;
const GUARD: usize = 256;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Half2,
    Mma,
    Hybrid,
}
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Half2 => "half2_direct",
            Self::Mma => "mma_direct",
            Self::Hybrid => "hybrid8_grid1024",
        }
    }
    pub fn grouped(self) -> bool {
        self == Self::Hybrid
    }
}
/// R38 (Opus, direct): unchanged mcg_tensor.cu plus bit-identical load-ahead
/// mcg_mma_direct variants in one module. Loaded only with QWEN_AFFINE4_R38=1.
pub const R38_SHA: &str = "830ba83d0eea1057f53e214da6090599f3e87afa688a784206206d35f4c3dac6";
pub const AHEAD_KERNEL: &str = "mcg_mma_direct_ahead2";
/// R38 load-ahead depth for the MMA kernels (2, 4 or 8; QWEN_R38_MCG_D).
fn r38_depth() -> u32 {
    match std::env::var("QWEN_R38_MCG_D").ok().and_then(|v| v.parse().ok()) {
        Some(4) => 4,
        Some(8) => 8,
        _ => 2,
    }
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/mcg_tensor_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/mcg_tensor.sm120.cubin"),
        SHA,
    )
}
/// R60: the canonical EXL3 K4 variant (kernels/mcg4_tensor.cu, same kernel names and ABI).
pub fn module_k4(d: &Device) -> Result<Module<'_>> {
    let pin: Value = serde_json::from_str(include_str!("../pins/R60-MCG4-TENSOR-CUDA-BUILD.json"))?;
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/mcg4_tensor.sm120.cubin"),
        pin["cubin_sha256"].as_str().unwrap_or_default(),
    )
}
pub struct Tensor<'m, 'd> {
    group: Kernel<'m, 'd>,
    project: [Kernel<'m, 'd>; 3],
    ahead: Option<Kernel<'m, 'd>>,
    pair: Option<Kernel<'m, 'd>>,
    desc: Buffer<'d>,
    count: Buffer<'d>,
    /// MCG halfwords per 16x16 tile: 28 (K1.75, kernels/mcg_tensor.cu) or 64 (K4, mcg4_tensor.cu)
    halfwords: usize,
}
impl<'m, 'd> Tensor<'m, 'd> {
    /// R60: the canonical K4 kernels (no R38 load-ahead variants).
    pub fn new_k4(d: &'d Device, m: &'m Module<'d>) -> Result<Self> {
        Ok(Self {
            group: m.kernel("mcg_group512")?,
            project: [m.kernel("mcg_half2_direct")?, m.kernel("mcg_mma_direct")?, m.kernel("mcg_hybrid8")?],
            ahead: None,
            pair: None,
            desc: d.alloc(SLOTS * 40 + GUARD)?,
            count: d.alloc(4 + GUARD)?,
            halfwords: 64,
        })
    }
    pub fn new(d: &'d Device, m: &'m Module<'d>) -> Result<Self> {
        Ok(Self {
            halfwords: 28,
            group: m.kernel("mcg_group512")?,
            project: [
                m.kernel("mcg_half2_direct")?,
                m.kernel("mcg_mma_direct")?,
                m.kernel("mcg_hybrid8")?,
            ],
            ahead: if crate::cuda_affine4::r38_enabled() {
                Some(m.kernel(&format!("mcg_mma_direct_ahead{}", r38_depth()))?)
            } else {
                None
            },
            pair: if crate::cuda_affine4::r38_enabled() {
                Some(m.kernel(&format!("mcg_mma_pair_ahead{}", r38_depth()))?)
            } else {
                None
            },
            desc: d.alloc(SLOTS * 40 + GUARD)?,
            count: d.alloc(4 + GUARD)?,
        })
    }
    pub fn description(&self) -> Value {
        json!({"cubin":if self.ahead.is_some() {R38_SHA} else {SHA},"r38_ahead_loaded":self.ahead.is_some(),"arms":["half2_direct","mma_direct","hybrid8_grid1024"],"original_ids":512,"maximum_slots":SLOTS,"grouping_passes_per_MoE":1,"bytes":SLOTS*40+4+GUARD*2})
    }
    pub fn poison(&self) -> Result<()> {
        self.desc.upload(&vec![255; SLOTS * 40 + GUARD])?;
        self.count.upload(&vec![255; 4 + GUARD])
    }
    pub fn group<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        slots: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=SLOTS).contains(&slots) && ids.byte_len() >= slots * 4,
            "group512 bounds"
        );
        unsafe {
            self.group.prepare(
                vec![
                    Arg::Buffer(ids),
                    Arg::Buffer(&self.desc),
                    Arg::Buffer(&self.count),
                    Arg::U32(slots as u32),
                    Arg::U32(512),
                ],
                [1, 1, 1],
                [512, 1, 1],
            )
        }
    }
    /// R38: gate and up projections (same K/N, same routes) in one launch.
    pub(crate) fn project_pair<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        x0: &'a Buffer<'d>,
        w0: &'a Buffer<'d>,
        y0: &'a Buffer<'d>,
        x1: &'a Buffer<'d>,
        w1: &'a Buffer<'d>,
        y1: &'a Buffer<'d>,
        k: usize,
        n: usize,
        slots: usize,
        splits: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (k, n) == (2560, 640) && (1..=SLOTS).contains(&slots) && [1, 2, 4].contains(&splits),
            "R38 paired projection geometry"
        );
        for (x, w, y) in [(x0, w0, y0), (x1, w1, y1)] {
            ensure!(
                x.byte_len() >= slots * k * 2
                    && w.byte_len() == 512 * k * n / 256 * self.halfwords * 2
                    && y.byte_len() >= slots * n * splits * 4
                    && ids.byte_len() >= slots * 4,
                "paired packed tensor buffer geometry"
            );
        }
        let kernel = self
            .pair
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("R38 paired MMA kernel not loaded"))?;
        unsafe {
            kernel.prepare(
                vec![
                    Arg::Buffer(x0),
                    Arg::Buffer(w0),
                    Arg::Buffer(y0),
                    Arg::Buffer(x1),
                    Arg::Buffer(w1),
                    Arg::Buffer(y1),
                    Arg::Buffer(ids),
                    Arg::U32(slots as u32),
                    Arg::U32(k as u32),
                    Arg::U32(n as u32),
                    Arg::U32(splits as u32),
                ],
                [(n / 16) as u32, (2 * slots) as u32, splits as u32],
                [128, 1, 1],
            )
        }
    }
    pub fn project<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        x: &'a Buffer<'d>,
        w: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        k: usize,
        n: usize,
        slots: usize,
        splits: usize,
        arm: Arm,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            [(2560, 640), (640, 2560)].contains(&(k, n))
                && (1..=SLOTS).contains(&slots)
                && [1, 2, 4].contains(&splits),
            "tensor projection shape"
        );
        ensure!(
            x.byte_len() >= slots * k * 2
                && w.byte_len() == 512 * k * n / 256 * self.halfwords * 2
                && y.byte_len() >= slots * n * splits * 4
                && ids.byte_len() >= slots * 4,
            "full packed tensor buffer geometry"
        );
        let kernel = if arm == Arm::Mma && crate::cuda_affine4::ahead_selected() {
            self.ahead
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("R38 load-ahead MMA kernel not loaded"))?
        } else {
            &self.project[match arm {
                Arm::Half2 => 0,
                Arm::Mma => 1,
                Arm::Hybrid => 2,
            }]
        };
        let grid = if arm.grouped() {
            [(slots * (n / 16) * splits).min(1024) as u32, 1, 1]
        } else {
            [(n / 16) as u32, slots as u32, splits as u32]
        };
        unsafe {
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(w),
                    Arg::Buffer(if arm.grouped() { &self.desc } else { ids }),
                    Arg::Buffer(&self.count),
                    Arg::Buffer(y),
                    Arg::U32(k as u32),
                    Arg::U32(n as u32),
                    Arg::U32(splits as u32),
                ],
                grid,
                [128, 1, 1],
            )
        }
    }
    pub fn check(&self, ids: &[u32], arm: Arm) -> Result<usize> {
        ensure!(
            !ids.is_empty() && ids.len() <= SLOTS && ids.iter().all(|&v| v < 512),
            "collected route bound"
        );
        let mut want = vec![];
        if arm.grouped() {
            for e in 0..512 {
                let rows: Vec<_> = ids
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v == e)
                    .map(|(i, _)| i as u32)
                    .collect();
                for c in rows.chunks(8) {
                    want.push(e);
                    want.push(c.len() as u32);
                    want.extend(c);
                    want.extend(std::iter::repeat_n(u32::MAX, 8 - c.len()));
                }
            }
        }
        let mut desc = vec![0; SLOTS * 40 + GUARD];
        let mut count = vec![0; 4 + GUARD];
        self.desc.download(&mut desc)?;
        self.count.download(&mut count)?;
        let expected = want
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        ensure!(
            desc[..expected.len()] == expected && desc[expected.len()..].iter().all(|&v| v == 255),
            "device route descriptors/tail"
        );
        let wc = if arm.grouped() {
            (want.len() / 10) as u32
        } else {
            u32::MAX
        };
        ensure!(
            u32::from_le_bytes(count[..4].try_into()?) == wc
                && count[4..].iter().all(|&v| v == 255),
            "device route count/tail"
        );
        Ok(want.len())
    }
}
