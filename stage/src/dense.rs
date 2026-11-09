//! Real affine8 trunk weights and independent FP64 projection reference.
use crate::projection::Pack;
use anyhow::{Result, ensure};
use half::{bf16, f16};

pub const MAX_ROWS: usize = 64;
pub struct Affine8 {
    pub k: usize,
    pub n: usize,
    pub group: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Affine8 {
    pub fn load(pack: &Pack, prefix: &str, k: usize) -> Result<Self> {
        let (w, codes, wp) = pack.tensor(&format!("{prefix}.weight"))?;
        let (s, sb, sp) = pack.tensor(&format!("{prefix}.scales"))?;
        let (b, bb, bp) = pack.tensor(&format!("{prefix}.biases"))?;
        ensure!(
            (128..=16384).contains(&k) && k % 64 == 0,
            "dense input dimension"
        );
        ensure!(
            w.dtype == "U32" && w.shape.len() == 2 && w.shape[1] == (k / 4) as u64,
            "not affine8 packed geometry"
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
            "unsupported affine8 group"
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
    /// R55: rows [first, first+n) of a larger affine8 matrix (the vocabulary head), read
    /// in bounded row ranges.
    pub fn load_rows(pack: &Pack, prefix: &str, k: usize, first: usize, n: usize) -> Result<Self> {
        ensure!((1..=32768).contains(&n) && k % 64 == 0, "row tile bounds");
        let mut codes = vec![];
        let mut sb = vec![];
        let mut bb = vec![];
        let mut groups = 0;
        let mut provenance = vec![];
        let mut at = first;
        while at < first + n {
            let take = (first + n - at).min(4096);
            let (w, c, wp) = pack.tensor_row_range(&format!("{prefix}.weight"), at, take)?;
            let (s, sc, _) = pack.tensor_row_range(&format!("{prefix}.scales"), at, take)?;
            let (_, bi, _) = pack.tensor_row_range(&format!("{prefix}.biases"), at, take)?;
            ensure!(w.dtype == "U32" && w.shape[1] == (k / 4) as u64 && s.dtype == "BF16", "row tile dtype");
            groups = s.shape[1] as usize;
            codes.extend_from_slice(&c);
            sb.extend_from_slice(&sc);
            bb.extend_from_slice(&bi);
            provenance.push(wp);
            at += take;
        }
        let parse = |v: &[u8]| -> Vec<u16> { v.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect() };
        ensure!(groups > 0 && k % groups == 0 && [64, 128].contains(&(k / groups)), "row tile group");
        let weight = Self {
            k,
            n,
            group: k / groups,
            codes,
            scales: parse(&sb),
            biases: parse(&bb),
            provenance: serde_json::json!({"rows":[first, first + n],"parts":provenance}),
        };
        weight.validate()?;
        Ok(weight)
    }
    /// R60: append complete output rows (same K and group), e.g. the shared expert's gate||up.
    pub fn concat_rows(&self, other: &Self) -> Result<Self> {
        self.validate()?;
        other.validate()?;
        ensure!(self.k == other.k && self.group == other.group, "concat affine8 geometry");
        let mut out = Self {
            k: self.k,
            n: self.n + other.n,
            group: self.group,
            codes: [self.codes.as_slice(), other.codes.as_slice()].concat(),
            scales: [self.scales.as_slice(), other.scales.as_slice()].concat(),
            biases: [self.biases.as_slice(), other.biases.as_slice()].concat(),
            provenance: serde_json::json!({"concat_rows":[self.provenance.clone(), other.provenance.clone()]}),
        };
        out.validate()?;
        out.provenance["rows"] = serde_json::json!([self.n, other.n]);
        Ok(out)
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
            self.codes.len() == count
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
    /// Materialization is for the CPU oracle only. Native Metal keeps byte codes.
    pub fn oracle_weights(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.codes.len());
        for (g, codes) in self.codes.chunks_exact(self.group).enumerate() {
            let s = bf16::from_bits(self.scales[g]).to_f64();
            let b = bf16::from_bits(self.biases[g]).to_f64();
            for &code in codes {
                out.push((s * f64::from(code) + b) as f32);
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
        ensure!(decoded.len() == self.codes.len(), "half diagnostic shape");
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
    fn affine_axes_groups_and_input_rows_are_distinct() {
        let p = Affine8 {
            k: 128,
            n: 3,
            group: 64,
            codes: (0..384).map(|i| (i % 251) as u8).collect(),
            scales: (0..6)
                .map(|i| bf16::from_f32((i + 1) as f32 / 32.).to_bits())
                .collect(),
            biases: (0..6)
                .map(|i| bf16::from_f32(-(i as f32)).to_bits())
                .collect(),
            provenance: serde_json::Value::Null,
        };
        let w = p.oracle_weights().unwrap();
        let mut x = vec![0; 256];
        x[63] = bf16::ONE.to_bits();
        x[128 + 64] = bf16::from_f32(-2.).to_bits();
        let y = p.reference(&w, &x).unwrap();
        for n in 0..3 {
            for (r, k) in [(0, 63), (1, 64)] {
                let g = (n * 128 + k) / 64;
                let expect = (f64::from(p.codes[n * 128 + k]) * (g + 1) as f64 / 32. - g as f64)
                    * if r == 0 { 1. } else { -2. };
                assert_eq!(y[r * 3 + n], expect);
            }
        }
        assert!(p.reference(&w, &[0; 127]).is_err());
    }
}
