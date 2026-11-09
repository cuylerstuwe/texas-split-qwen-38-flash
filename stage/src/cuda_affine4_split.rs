//! One-GPU reduction partitioning. This does not divide a model between GPUs.
use crate::cuda::{Arg, Buffer, Device, Kernel, Module, PreparedCall};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub const SHA: &str = "1c36148c7b5eb6d45731fc6162fa3ab14b06f8d16528dfbe7d0fd01020e22f84";
pub const MAX_ROWS: usize = 16;
pub const MAX_SPLITS: usize = 8;
const GUARD: usize = 256;
/// R38 (Opus, direct): unchanged affine4_split.cu plus bit-identical load-ahead
/// partials in one module. Loaded only with QWEN_AFFINE4_R38=1.
pub const R38_SHA: &str = "5dc473cd3dc2a21c0749abf78c43c466ac31bd33308e1b2c573eb2b0d44ab216";
pub const AHEAD_KERNEL: &str = "affine4_ahead_partial1_d2";
pub fn module(d: &Device) -> Result<Module<'_>> {
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_split_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine4_split.sm120.cubin"),
        SHA,
    )
}
pub struct Split<'m, 'd> {
    kernels: [Kernel<'m, 'd>; 3],
    ahead: Option<Kernel<'m, 'd>>,
    reduce: Kernel<'m, 'd>,
    partial: Buffer<'d>,
    n: usize,
    /// R60: canonical affine8 weights (kernels/affine8.cu partials and reduce)
    bits8: bool,
}
impl<'m, 'd> Split<'m, 'd> {
    /// R60: split-K for the canonical target's affine8 projections (`m8`: the affine8 module).
    pub fn new8(d: &'d Device, m8: &'m Module<'d>, n: usize) -> Result<Self> {
        ensure!((1..=32768).contains(&n), "affine8 split output bound");
        Ok(Self {
            kernels: [m8.kernel("affine8_partial1")?, m8.kernel("affine8_partial2")?, m8.kernel("affine8_partial4")?],
            ahead: None,
            reduce: m8.kernel("affine8_reduce")?,
            partial: d.alloc(MAX_ROWS * MAX_SPLITS * n * 4 + GUARD)?,
            n,
            bits8: true,
        })
    }
    pub fn new(d: &'d Device, m: &'m Module<'d>, n: usize) -> Result<Self> {
        ensure!((1..=32768).contains(&n), "affine4 split output bound");
        Ok(Self {
            kernels: [
                m.kernel("affine4_partial1")?,
                m.kernel("affine4_partial2")?,
                m.kernel("affine4_partial4")?,
            ],
            ahead: if crate::cuda_affine4::r38_enabled() {
                Some(m.kernel(AHEAD_KERNEL)?)
            } else {
                None
            },
            reduce: m.kernel("affine4_reduce")?,
            partial: d.alloc(MAX_ROWS * MAX_SPLITS * n * 4 + GUARD)?,
            n,
            bits8: false,
        })
    }
    pub fn description(&self) -> Value {
        json!({"cubin_sha256":if self.ahead.is_some() {R38_SHA} else {SHA},"r38_ahead_loaded":self.ahead.is_some(),"maximum_rows":MAX_ROWS,"maximum_splits":MAX_SPLITS,
            "partial_layout":"[row,split,column] FP32","partition":"disjoint complete256-K chunks, final chunk guarded",
            "partial_bytes":self.partial.byte_len(),"kernels":self.kernels.iter().map(|k|k.description()).collect::<Vec<_>>(),
            "reduce":self.reduce.description(),"device_calls":2,"host_intermediates":0})
    }
    pub fn record<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        w: &'a Buffer<'d>,
        s: &'a Buffer<'d>,
        b: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        ybf: &'a Buffer<'d>,
        k: usize,
        group: usize,
        rows: usize,
        width: usize,
        splits: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (128..=16384).contains(&k)
                && k % 64 == 0
                && [64, 128].contains(&group)
                && k % group == 0,
            "affine4 split weight geometry"
        );
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && [1, 2, 4].contains(&width)
                && [1, 2, 4, 8].contains(&splits)
                && splits <= k.div_ceil(256),
            "affine4 split shape/partition"
        );
        ensure!(
            x.byte_len() >= rows * k * 2
                && w.byte_len() == if self.bits8 { self.n * k } else { self.n * k / 2 }
                && s.byte_len() == self.n * k / group * 2
                && b.byte_len() == s.byte_len()
                && y.byte_len() >= rows * self.n * 4
                && ybf.byte_len() >= rows * self.n * 2,
            "affine4 split buffer size"
        );
        ensure!(
            !y.aliases(ybf)
                && [&self.partial, y, ybf]
                    .iter()
                    .all(|out| [x, w, s, b].iter().all(|input| !out.aliases(input)))
                && !self.partial.aliases(y)
                && !self.partial.aliases(ybf),
            "affine4 split alias"
        );
        // All tiles have a unique output slice. Complete chunks partition K
        // exactly; scratch includes the maximum row/split geometry plus canary.
        let partial = if width == 1 && crate::cuda_affine4::ahead_selected() && !self.bits8 {
            self.ahead
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("R38 load-ahead partial not loaded"))?
        } else {
            &self.kernels[match width {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => unreachable!(),
            }]
        };
        unsafe {
            Ok(vec![
                partial
                .prepare(
                    vec![
                        Arg::Buffer(x),
                        Arg::Buffer(w),
                        Arg::Buffer(s),
                        Arg::Buffer(b),
                        Arg::Buffer(&self.partial),
                        Arg::U32(k as u32),
                        Arg::U32(self.n as u32),
                        Arg::U32(group as u32),
                        Arg::U32(rows as u32),
                        Arg::U32(splits as u32),
                    ],
                    [
                        self.n.div_ceil(4) as u32,
                        rows.div_ceil(width) as u32,
                        splits as u32,
                    ],
                    [64, 1, 1],
                )?,
                self.reduce.prepare(
                    vec![
                        Arg::Buffer(&self.partial),
                        Arg::Buffer(y),
                        Arg::Buffer(ybf),
                        Arg::U32(self.n as u32),
                        Arg::U32(rows as u32),
                        Arg::U32(splits as u32),
                    ],
                    [(rows * self.n).div_ceil(128) as u32, 1, 1],
                    [128, 1, 1],
                )?,
            ])
        }
    }
    pub fn poison(&self) -> Result<()> {
        self.partial.upload(&vec![0xff; self.partial.byte_len()])
    }
    pub(crate) fn partials(&self) -> &Buffer<'d> {
        &self.partial
    }
    pub fn check(&self, rows: usize, splits: usize) -> Result<usize> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && (1..=MAX_SPLITS).contains(&splits),
            "affine4 split check bounds"
        );
        let active = rows * splits * self.n * 4;
        let mut bytes = vec![0; self.partial.byte_len()];
        self.partial.download(&mut bytes)?;
        ensure!(
            bytes[active..].iter().all(|&b| b == 0xff),
            "affine4 split partial canary"
        );
        ensure!(
            bytes[..active]
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "affine4 split partial unwritten/nonfinite"
        );
        Ok(active / 4)
    }
}
