//! R60 pipeline split: the canonical target's MLX affine8 projections (8-bit codes, BF16
//! scales/biases, group 64 or 128) on CUDA, 1-16 token rows. Kernel: kernels/affine8.cu (port
//! of affine8_dense.metal's QMV organization, MLX's factored arithmetic order).
use crate::{
    cuda::{Arg, Buffer, Device, Kernel, Module, PreparedCall},
    dense::Affine8,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub const MAX_ROWS: usize = 16;
fn pin() -> String {
    serde_json::from_str::<Value>(include_str!("../pins/R60-AFFINE8-CUDA-BUILD.json")).unwrap()["cubin_sha256"]
        .as_str()
        .unwrap()
        .to_owned()
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/affine8.sm120.cubin"), &pin())
}
/// The four row-blocking entries: 1, 2, 4 and 8 token rows per block.
pub struct Kernels<'m, 'd> {
    k: [Kernel<'m, 'd>; 4],
    /// R60 S4: the tensor-core arms ([t1, t2] x [4, 8, 16 warps]), used from affine8_mma_min()
    /// rows with the same width rule as cuda_affine4::A8Kernels.
    mma: Option<[[Kernel<'m, 'd>; 3]; 2]>,
}
impl<'m, 'd> Kernels<'m, 'd> {
    pub fn new(m: &'m Module<'d>) -> Result<Self> {
        // R60 S4: the v2 kernels have the same ABI and rows-per-block geometry (QWEN_AFFINE8_V2=0: v1).
        if crate::cuda_affine4::affine8_v2() {
            let mma = if crate::cuda_affine4::affine8_mma_min() < usize::MAX {
                Some([
                    [m.kernel("affine8_mma_t1")?, m.kernel("affine8_mma_t1_w8")?, m.kernel("affine8_mma_t1_w16")?],
                    [m.kernel("affine8_mma_t2")?, m.kernel("affine8_mma_t2_w8")?, m.kernel("affine8_mma_t2_w16")?],
                ])
            } else {
                None
            };
            return Ok(Self { k: [m.kernel("affine8_v2_1")?, m.kernel("affine8_v2_2")?, m.kernel("affine8_v2_4")?, m.kernel("affine8_v2_8")?], mma });
        }
        Ok(Self { k: [m.kernel("affine8_f1")?, m.kernel("affine8_f2")?, m.kernel("affine8_f4")?, m.kernel("affine8_f8")?], mma: None })
    }
}
/// One resident projection: y[rows, n] = x[rows, k] · W^T with W in MLX affine8 layout.
pub struct Dense<'d> {
    pub k: usize,
    pub n: usize,
    pub group: usize,
    w: Buffer<'d>,
    s: Buffer<'d>,
    b: Buffer<'d>,
}
fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
impl<'d> Dense<'d> {
    pub fn upload(d: &'d Device, a: &Affine8) -> Result<Self> {
        a.validate()?;
        ensure!(a.k % 256 == 0 || a.k % 8 == 0, "affine8 K stride");
        let put = |raw: &[u8]| -> Result<Buffer<'d>> {
            let buf = d.alloc(raw.len())?;
            buf.upload(raw)?;
            let mut check = vec![0; raw.len()];
            buf.download(&mut check)?;
            ensure!(check == raw, "affine8 weight upload readback");
            Ok(buf)
        };
        Ok(Self { k: a.k, n: a.n, group: a.group, w: put(&a.codes)?, s: put(&bytes16(&a.scales))?, b: put(&bytes16(&a.biases))? })
    }
    pub fn bytes(&self) -> usize {
        self.w.byte_len() + self.s.byte_len() + self.b.byte_len()
    }
    pub fn description(&self) -> Value {
        json!({"k":self.k,"n":self.n,"group":self.group,"bytes":self.bytes(),"cubin_sha256":pin(),"format":"MLX affine8, BF16 scales/biases"})
    }
    /// x: BF16 [rows, k]; y: FP32 [rows, n]; ybf: BF16 [rows, n].
    pub fn record<'a, 'm>(
        &'a self,
        kern: &'a Kernels<'m, 'd>,
        x: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        ybf: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && x.byte_len() >= rows * self.k * 2
                && y.byte_len() >= rows * self.n * 4
                && ybf.byte_len() >= rows * self.n * 2,
            "affine8 record geometry"
        );
        let (entry, nv) = match rows {
            1 => (0, 1),
            2 => (1, 2),
            3 | 4 => (2, 4),
            _ => (3, 8),
        };
        let tensor = kern.mma.as_ref().filter(|_| rows >= crate::cuda_affine4::affine8_mma_min() && self.group % 64 == 0);
        let (kernel, grid, block) = match tensor {
            Some(m) => {
                let (i, t) = if rows <= 8 { (0, 1) } else { (1, 2) };
                let blocks = self.n.div_ceil(16) * rows.div_ceil(8 * t);
                // K caps the split: each warp should get at least one U=4 group of 64-code blocks
                let cap = self.k / 256;
                let (j, warps) = if blocks < 256 && cap >= 16 { (2, 16) } else if blocks < 512 && cap >= 8 { (1, 8) } else { (0, 4) };
                (&m[i][j], [self.n.div_ceil(16) as u32, rows.div_ceil(8 * t) as u32, 1], [32 * warps, 1, 1])
            }
            None => (&kern.k[entry], [self.n.div_ceil(4) as u32, rows.div_ceil(nv) as u32, 1], [64, 1, 1]),
        };
        // SAFETY: fixed kernel ABI; buffers sized above and borrowed for the call's lifetime.
        unsafe {
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.w),
                    Arg::Buffer(&self.s),
                    Arg::Buffer(&self.b),
                    Arg::Buffer(y),
                    Arg::Buffer(ybf),
                    Arg::U32(self.k as u32),
                    Arg::U32(self.n as u32),
                    Arg::U32(self.group as u32),
                    Arg::U32(rows as u32),
                ],
                grid,
                block,
            )
        }
    }
}
