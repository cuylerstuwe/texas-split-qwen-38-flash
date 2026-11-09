//! Bounded real DFlash2 BF16 projections and an independent scalar FP64 oracle.
use crate::{projection::sha, weights::Shard};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::path::Path;

pub const MAX_ROWS: usize = 64;
pub const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/incoai/Qwen3.8-Flash-Next-DFlash2/model.safetensors"
);
pub const HEADER_SHA: &str = "5ef4772f77b6a3ea349c6515aceffdb999868946aafb04578416494c64ef0513";

pub struct Matrix {
    pub k: usize,
    pub n: usize,
    /// Original row-major [output, input], BF16; never re-quantized.
    pub values: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Matrix {
    pub fn load(name: &str) -> Result<Self> {
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.file_bytes == 1_206_140_008 && s.header_sha256 == HEADER_SHA,
            "pinned DFlash2 header/size changed"
        );
        let t = s.tensors.get(name).context("DFlash2 tensor absent")?;
        ensure!(
            t.dtype == "BF16" && t.shape.len() == 2,
            "DFlash2 matrix geometry"
        );
        let raw = s.read_range(name, 0, t.bytes())?;
        let out = Self {
            n: t.shape[0] as usize,
            k: t.shape[1] as usize,
            values: raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            provenance: json!({"path":MODEL,"revision":"a22a1744e85c17364dcc54a6a420af7135d4c7f6",
                "header_sha256":s.header_sha256,"tensor":name,"shape":t.shape,"dtype":t.dtype,
                "bytes":raw.len(),"tensor_sha256":sha(&raw)}),
        };
        out.validate()?;
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (8..=16384).contains(&self.k)
                && self.k % 8 == 0
                && (1..=16384).contains(&self.n)
                && self.n.checked_mul(self.k) == Some(self.values.len())
                && self.values.len() <= (32 << 20),
            "bounded BF16 matrix"
        );
        ensure!(
            self.values.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "nonfinite BF16 matrix"
        );
        Ok(())
    }
    pub fn reference(&self, input: &[u16]) -> Result<Vec<f64>> {
        self.validate()?;
        ensure!(
            !input.is_empty() && input.len() % self.k == 0 && input.len() / self.k <= MAX_ROWS,
            "BF16 reference geometry"
        );
        let x: Vec<_> = input.iter().map(|&v| bf16::from_bits(v).to_f64()).collect();
        ensure!(x.iter().all(|v| v.is_finite()), "BF16 reference input");
        let w: Vec<_> = self
            .values
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        let rows = x.len() / self.k;
        let mut y = vec![0.; rows * self.n];
        let per = rows.div_ceil(4);
        std::thread::scope(|scope| {
            for (block, dst) in y.chunks_mut(per * self.n).enumerate() {
                let x = &x[block * per * self.k..];
                let w = &w;
                scope.spawn(move || {
                    for (r, out) in dst.chunks_mut(self.n).enumerate() {
                        let xx = &x[r * self.k..(r + 1) * self.k];
                        for (col, ww) in w.chunks_exact(self.k).enumerate() {
                            let mut s = 0.;
                            for (&a, &b) in xx.iter().zip(ww) {
                                s += a * f64::from(b);
                            }
                            out[col] = s;
                        }
                    }
                });
            }
        });
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_and_output_axes_are_independent() {
        let m = Matrix {
            k: 8,
            n: 3,
            values: (0..24)
                .map(|v| bf16::from_f32(v as f32 - 12.).to_bits())
                .collect(),
            provenance: json!({}),
        };
        let mut x = vec![0; 16];
        x[0] = bf16::ONE.to_bits();
        x[15] = bf16::from_f32(-2.).to_bits();
        assert_eq!(m.reference(&x).unwrap(), [-12., -4., 4., 10., -6., -22.]);
        assert!(m.reference(&x[..15]).is_err());
    }
}
