//! Real lowbit affine4 trunk weights; bounded CPU weight-format/dot reference.
use crate::{
    projection::sha,
    weights::{Shard, Tensor},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use serde_json::{Value, json};
use std::path::{Component, Path};

pub const MAX_ROWS: usize = 64;
pub struct Affine4 {
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Affine4 {
    /// Append complete output rows without decoding or requantizing either input.
    pub fn concat_rows(&self, other: &Self) -> Result<Self> {
        self.validate()?;
        other.validate()?;
        ensure!(
            self.k == other.k && self.group == other.group,
            "concat affine geometry"
        );
        let n = self
            .n
            .checked_add(other.n)
            .context("concat output overflow")?;
        ensure!(n <= 32768 && n * self.k <= 64 << 20, "concat weight bound");
        let mut codes = self.codes.clone();
        codes.extend_from_slice(&other.codes);
        let mut scales = self.scales.clone();
        scales.extend_from_slice(&other.scales);
        let mut biases = self.biases.clone();
        biases.extend_from_slice(&other.biases);
        let result = Self {
            k: self.k,
            n,
            group: self.group,
            codes,
            scales,
            biases,
            provenance: json!({"operation":"exact packed output-row concatenation", "inputs":[self.provenance, other.provenance]}),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn load(root: &Path, prefix: &str, k: usize) -> Result<Self> {
        let config = std::fs::read(root.join("config.json"))?;
        let index_bytes = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&config) == "fd9d242648d6a5a1ef130f8f31234bc182b0c39ea175241c0e7c22a65d241454",
            "lowbit config pin"
        );
        ensure!(
            sha(&index_bytes) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
            "lowbit index pin"
        );
        let index: Value = serde_json::from_slice(&index_bytes)?;
        let read = |name: &str| -> Result<(Tensor, Vec<u8>, Value)> {
            let file = index["weight_map"][name]
                .as_str()
                .context("affine4 tensor index")?;
            let mut parts = Path::new(file).components();
            ensure!(
                matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
                "unsafe affine4 shard"
            );
            let shard = Shard::open(&root.join(file))?;
            let tensor = shard
                .tensors
                .get(name)
                .context("affine4 tensor missing")?
                .clone();
            ensure!(tensor.bytes() <= 64 << 20, "bounded affine4 tensor read");
            let bytes = shard.read_range(name, 0, tensor.bytes())?;
            let provenance = json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,
                "payload_sha256":sha(&bytes),"config_sha256":sha(&config),"index_sha256":sha(&index_bytes)});
            Ok((tensor, bytes, provenance))
        };
        let (w, codes, wp) = read(&format!("{prefix}.weight"))?;
        let (s, sb, sp) = read(&format!("{prefix}.scales"))?;
        let (b, bb, bp) = read(&format!("{prefix}.biases"))?;
        ensure!(
            (128..=16384).contains(&k) && k % 64 == 0,
            "dense input dimension"
        );
        ensure!(
            w.dtype == "U32" && w.shape.len() == 2 && w.shape[1] == (k / 8) as u64,
            "not affine4 packed geometry"
        );
        let n = w.shape[0] as usize;
        ensure!(
            (1..=32768).contains(&n) && k.checked_mul(n).is_some_and(|v| v <= 64 << 20),
            "dense bounded weight geometry"
        );
        ensure!(
            s.dtype == "BF16"
                && b.dtype == "BF16"
                && s.shape.len() == 2
                && s.shape == b.shape
                && s.shape[0] == n as u64
                && s.shape[1] > 0,
            "dense affine metadata"
        );
        let groups = s.shape[1] as usize;
        ensure!(
            k % groups == 0 && [64, 128].contains(&(k / groups)),
            "unsupported affine4 group"
        );
        let parse = |v: &[u8]| -> Vec<u16> {
            v.chunks_exact(2)
                .map(|x| u16::from_le_bytes([x[0], x[1]]))
                .collect()
        };
        let weight = Self {
            k,
            n,
            group: k / groups,
            codes,
            scales: parse(&sb),
            biases: parse(&bb),
            provenance: serde_json::json!([wp, sp, bp]),
        };
        weight.validate()?;
        Ok(weight)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (128..=16384).contains(&self.k)
                && self.k % 64 == 0
                && (1..=32768).contains(&self.n)
                && [64, 128].contains(&self.group)
                && self.k % self.group == 0,
            "dense shape"
        );
        let count = self
            .k
            .checked_mul(self.n)
            .filter(|&v| v <= 64 << 20)
            .ok_or_else(|| anyhow::anyhow!("dense weight size"))?;
        ensure!(
            self.codes.len() == count / 2
                && self.scales.len() == count / self.group
                && self.biases.len() == self.scales.len(),
            "dense weight lengths"
        );
        ensure!(
            self.scales
                .iter()
                .chain(&self.biases)
                .all(|&v| bf16::from_bits(v).is_finite()),
            "nonfinite affine metadata"
        );
        Ok(())
    }
    /// Materialization is for validation only. GPU execution must retain packed nibbles.
    pub fn oracle_weights(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.k * self.n);
        for (g, bytes) in self.codes.chunks_exact(self.group / 2).enumerate() {
            let scale = bf16::from_bits(self.scales[g]).to_f64();
            let bias = bf16::from_bits(self.biases[g]).to_f64();
            for &packed in bytes {
                out.push((scale * f64::from(packed & 15) + bias) as f32);
                out.push((scale * f64::from(packed >> 4) + bias) as f32);
            }
        }
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "nonfinite dequantized weight"
        );
        Ok(out)
    }
    pub fn reference(&self, decoded: &[f32], x: &[u16]) -> Result<Vec<f64>> {
        ensure!(
            !x.is_empty()
                && x.len() % self.k == 0
                && x.len() / self.k <= MAX_ROWS
                && decoded.len() == self.k * self.n,
            "dense oracle shape"
        );
        let inputs: Vec<f64> = x.iter().map(|&v| bf16::from_bits(v).to_f64()).collect();
        ensure!(
            inputs.iter().all(|v| v.is_finite()) && decoded.iter().all(|v| v.is_finite()),
            "dense oracle finite"
        );
        let rows = inputs.len() / self.k;
        let mut y = vec![0.; rows * self.n];
        // Four bounded CPU workers; independent scalar FP64 dot for each output.
        // This is not a GPU timing path or a BLAS dependency.
        let per = rows.div_ceil(4).max(1);
        std::thread::scope(|scope| {
            for (block, dst) in y.chunks_mut(per * self.n).enumerate() {
                let input = &inputs[block * per * self.k..];
                scope.spawn(move || {
                    for (r, out) in dst.chunks_mut(self.n).enumerate() {
                        let a = &input[r * self.k..(r + 1) * self.k];
                        for (j, w) in decoded.chunks_exact(self.k).enumerate() {
                            let mut sum = 0f64;
                            for (&v, &q) in a.iter().zip(w) {
                                sum += v * f64::from(q);
                            }
                            out[j] = sum;
                        }
                    }
                });
            }
        });
        Ok(y)
    }
    /// Input-half and weight-half conversion are explicit candidate policies.
    pub fn half_weight_error(&self, decoded: &[f32]) -> Result<(f64, f64)> {
        ensure!(decoded.len() == self.k * self.n, "half diagnostic shape");
        let (mut h, mut h2, mut norm) = (0., 0., 0.);
        for &w in decoded {
            let hi = f16::from_f32(w).to_f32();
            ensure!(hi.is_finite(), "weight outside candidate FP16 range");
            let lo = f16::from_f32(w - hi).to_f32();
            h += (f64::from(w) - f64::from(hi)).powi(2);
            h2 += (f64::from(w) - f64::from(hi) - f64::from(lo)).powi(2);
            norm += f64::from(w).powi(2);
        }
        Ok(((h / norm.max(1e-30)).sqrt(), (h2 / norm.max(1e-30)).sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concatenation_preserves_output_rows_and_metadata() {
        let make = |n, byte, scale, bias| Affine4 {
            k: 128,
            n,
            group: 64,
            codes: vec![byte; n * 64],
            scales: vec![bf16::from_f32(scale).to_bits(); n * 2],
            biases: vec![bf16::from_f32(bias).to_bits(); n * 2],
            provenance: Value::Null,
        };
        let a = make(1, 0x21, 0.5, -1.);
        let b = make(2, 0x43, -0.25, 2.);
        let c = a.concat_rows(&b).unwrap();
        let mut x = vec![0; 256];
        x[0] = bf16::from_f32(1.).to_bits();
        x[128 + 1] = bf16::from_f32(2.).to_bits();
        assert_eq!(
            c.reference(&c.oracle_weights().unwrap(), &x).unwrap(),
            vec![-0.5, 1.25, 1.25, 0., 2., 2.]
        );
        let mut wrong = b;
        wrong.group = 128;
        wrong.scales.truncate(2);
        wrong.biases.truncate(2);
        assert!(a.concat_rows(&wrong).is_err());
    }
    #[test]
    fn low_nibble_first_groups_and_outputs_are_independent() {
        let mut codes = vec![];
        for row in 0..2 {
            for k in (0..128).step_by(2) {
                let lo = if row == 0 { k % 16 } else { 15 - k % 16 };
                let hi = if row == 0 {
                    (k + 1) % 16
                } else {
                    15 - (k + 1) % 16
                };
                codes.push((lo | (hi << 4)) as u8);
            }
        }
        let h = |v: f32| bf16::from_f32(v).to_bits();
        let mut p = Affine4 {
            k: 128,
            n: 2,
            group: 64,
            codes,
            scales: vec![h(0.25), h(-0.125), h(0.5), h(0.0625)],
            biases: vec![h(-0.5), h(1.), h(0.25), h(-2.)],
            provenance: Value::Null,
        };
        assert_eq!(&p.codes[..4], &[0x10, 0x32, 0x54, 0x76]);
        let w = p.oracle_weights().unwrap();
        let mut x = vec![0; 4 * 128];
        for (row, k) in [0, 63, 64, 127].into_iter().enumerate() {
            x[row * 128 + k] = h(1.);
        }
        let y = p.reference(&w, &x).unwrap();
        assert_eq!(y, vec![-0.5, 7.75, 3.25, 0.25, 1., -1.0625, -0.875, -2.]);
        assert!(p.reference(&w, &[0; 127]).is_err());
        p.codes.pop();
        assert!(p.validate().is_err());
    }
}
