//! Bounded real EXL3 expert slices and an independent FP64 projection oracle.
//! Raw trellis projection only: Hadamards/scales/activation belong to MoE later.
use crate::{
    exl3::decode_k4_tile,
    weights::{Shard, check_config},
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub struct Pack {
    root: PathBuf,
    index: BTreeMap<String, String>,
}
impl Pack {
    /// R55: the shard file holding `name` (validated single path component).
    pub(crate) fn file_of(&self, name: &str) -> Result<PathBuf> {
        let file = self.index.get(name).context("missing tensor index")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe shard name"
        );
        Ok(self.root.join(file))
    }
    /// One contiguous row tile, for scanning a vocabulary table without
    /// materializing or mapping the entire decoded matrix.
    pub(crate) fn tensor_row_range(
        &self,
        name: &str,
        first: usize,
        rows: usize,
    ) -> Result<(crate::weights::Tensor, Vec<u8>, serde_json::Value)> {
        ensure!((1..=4096).contains(&rows), "bounded contiguous row count");
        let file = self.index.get(name).context("missing row-range tensor")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe row-range shard"
        );
        let shard = Shard::open(&self.root.join(file))?;
        let t = shard
            .tensors
            .get(name)
            .context("row-range tensor absent")?
            .clone();
        ensure!(
            t.shape.len() == 2
                && t.shape[0] > 0
                && first
                    .checked_add(rows)
                    .is_some_and(|n| n as u64 <= t.shape[0]),
            "row-range geometry"
        );
        let stride = t.bytes() / t.shape[0];
        let bytes = stride
            .checked_mul(rows as u64)
            .context("row-range overflow")?;
        ensure!(stride > 0 && bytes <= 16 << 20, "bounded row-range bytes");
        let data = shard.read_range(name, first as u64 * stride, bytes)?;
        let provenance = json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"first_row":first,"rows":rows,"row_bytes":stride,"bytes":data.len(),"sha256":sha(&data)});
        Ok((t, data, provenance))
    }
    /// Bounded selected rows of a matrix, preserving caller order and repeats.
    /// Used for raw token embeddings without reading the full vocabulary table.
    pub(crate) fn tensor_rows(
        &self,
        name: &str,
        rows: &[usize],
    ) -> Result<(crate::weights::Tensor, Vec<u8>, serde_json::Value)> {
        ensure!(
            !rows.is_empty() && rows.len() <= 64,
            "bounded row selection"
        );
        let file = self.index.get(name).context("missing row tensor index")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe row tensor shard"
        );
        let shard = Shard::open(&self.root.join(file))?;
        let t = shard
            .tensors
            .get(name)
            .context("missing row tensor")?
            .clone();
        ensure!(
            t.shape.len() == 2 && t.shape[0] > 0 && rows.iter().all(|&r| (r as u64) < t.shape[0]),
            "row tensor geometry/index"
        );
        let stride = t.bytes() / t.shape[0];
        ensure!(
            stride > 0
                && stride
                    .checked_mul(rows.len() as u64)
                    .is_some_and(|n| n <= 16 << 20),
            "row selection byte bound"
        );
        let mut data = Vec::with_capacity(stride as usize * rows.len());
        let mut hashes = Vec::with_capacity(rows.len());
        for &row in rows {
            let raw = shard.read_range(name, row as u64 * stride, stride)?;
            hashes.push(json!({"row":row,"sha256":sha(&raw)}));
            data.extend(raw);
        }
        Ok((
            t,
            data,
            json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"row_bytes":stride,"rows":hashes}),
        ))
    }
    /// Read one bounded tensor from the pinned pack. No whole-shard mapping.
    pub(crate) fn tensor(
        &self,
        name: &str,
    ) -> Result<(crate::weights::Tensor, Vec<u8>, serde_json::Value)> {
        let file = self.index.get(name).context("missing tensor index")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe tensor shard name"
        );
        let shard = Shard::open(&self.root.join(file))?;
        let tensor = shard
            .tensors
            .get(name)
            .context("missing indexed tensor")?
            .clone();
        let data = shard.read_range(name, 0, tensor.bytes())?;
        let provenance = serde_json::json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"sha256":sha(&data),"bytes":data.len()});
        Ok((tensor, data, provenance))
    }
    /// Read only selected per-expert scales; never expand an entire checkpoint.
    pub fn scaled_projection(
        &self,
        layer: usize,
        projection: &str,
        experts: &[usize],
    ) -> Result<ScaledProjection> {
        ensure!(
            !experts.is_empty() && experts.len() <= 256,
            "compact projection bound"
        );
        self.scaled_projection_impl(layer, projection, experts)
    }
    /// Device-only packed bank. The public compact loader/oracle limits remain
    /// unchanged; the full bank is hidden behind moe::FullExpertBank.
    pub(crate) fn full_scaled_projection(
        &self,
        layer: usize,
        projection: &str,
    ) -> Result<ScaledProjection> {
        self.scaled_projection_impl(layer, projection, &(0..512).collect::<Vec<_>>())
    }
    fn scaled_projection_impl(
        &self,
        layer: usize,
        projection: &str,
        experts: &[usize],
    ) -> Result<ScaledProjection> {
        let raw = self.projection_impl(layer, projection, experts)?;
        let mut scales = Vec::new();
        let mut provenance = Vec::new();
        for (suffix, dim) in [("suh", raw.k), ("svh", raw.n)] {
            let name =
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.{suffix}");
            let file = self.index.get(&name).context("missing scale index")?;
            let mut parts = Path::new(file).components();
            ensure!(
                matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
                "unsafe scale shard name"
            );
            let shard = Shard::open(&self.root.join(file))?;
            let t = shard.tensors.get(&name).context("missing scale")?;
            ensure!(
                t.dtype == "F16" && t.shape == [512, dim as u64],
                "scale shape/dtype"
            );
            let mut values = Vec::with_capacity(experts.len() * dim);
            let mut slices = vec![];
            if experts.len() == 512 && experts.iter().copied().eq(0..512) {
                let bytes = shard.read_range(&name, 0, (512 * dim * 2) as u64)?;
                slices.push(serde_json::json!({"full_tensor":true,"sha256":sha(&bytes)}));
                values.extend(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
            } else {
                for &expert in experts {
                    let bytes = shard.read_range(&name, (expert * dim * 2) as u64, (dim * 2) as u64)?;
                    slices.push(serde_json::json!({"expert":expert,"sha256":sha(&bytes)}));
                    values.extend(
                        bytes
                            .chunks_exact(2)
                            .map(|b| u16::from_le_bytes([b[0], b[1]])),
                    );
                }
            }
            ensure!(
                values.iter().all(|&v| f16::from_bits(v).is_finite()),
                "nonfinite scale"
            );
            provenance.push(serde_json::json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"slices":slices}));
            scales.push(values);
        }
        let svh = scales.pop().unwrap();
        let suh = scales.pop().unwrap();
        Ok(ScaledProjection {
            raw,
            suh,
            svh,
            scale_provenance: serde_json::json!(provenance),
        })
    }
    pub fn open(root: &Path) -> Result<Self> {
        let config = std::fs::read(root.join("config.json"))?;
        let index = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&config) == "2446916c7a09b7a66bfb3eba2a19d62ded0af083b66e7c20b891bd82bfdde96a",
            "config pin changed"
        );
        ensure!(
            sha(&index) == "b76a06c532138ac35e916f73d21cb1011dcb2edcfdbae5e451ca5816948151f2",
            "index pin changed"
        );
        check_config(&serde_json::from_slice(&config)?)?;
        let value: serde_json::Value = serde_json::from_slice(&index)?;
        Ok(Self {
            root: root.into(),
            index: serde_json::from_value(value["weight_map"].clone())?,
        })
    }
    pub fn projection(
        &self,
        layer: usize,
        projection: &str,
        experts: &[usize],
    ) -> Result<Projection> {
        ensure!(
            !experts.is_empty() && experts.len() <= 256,
            "compact projection bound"
        );
        self.projection_impl(layer, projection, experts)
    }
    fn projection_impl(
        &self,
        layer: usize,
        projection: &str,
        experts: &[usize],
    ) -> Result<Projection> {
        ensure!(
            layer < 48 && ["gate_proj", "up_proj", "down_proj"].contains(&projection),
            "projection selection"
        );
        ensure!(
            !experts.is_empty() && experts.len() <= 512 && experts.iter().all(|&e| e < 512),
            "expert bounds"
        );
        let name =
            format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.trellis");
        let file = self.index.get(&name).context("missing projection")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe shard name"
        );
        let shard = Shard::open(&self.root.join(file))?;
        let tensor = shard.tensors.get(&name).context("missing tensor")?;
        let (k, n) = if projection == "down_proj" {
            (640, 2560)
        } else {
            (2560, 640)
        };
        ensure!(
            tensor.dtype == "U16" && tensor.shape == [512, k as u64 / 16, n as u64 / 16, 64],
            "trellis shape"
        );
        for (key, expected) in [
            ("format", "exl3"),
            ("k", "4"),
            ("codebook", "mcg"),
            ("window", "15"),
        ] {
            ensure!(
                shard.metadata.get(key).map(String::as_str) == Some(expected),
                "pack stamp {key}"
            );
        }
        let expert_bytes = k * n / 2;
        // R60: the full original bank (the pipeline split's resident layers) is one contiguous
        // read without per-expert hashing; compact subsets keep per-slice receipts.
        if experts.len() == 512 && experts.iter().copied().eq(0..512) {
            let total = 512 * expert_bytes;
            let mut packed = Vec::with_capacity(total);
            let chunk = 64 << 20;
            while packed.len() < total {
                let len = chunk.min(total - packed.len());
                packed.extend(shard.read_range(&name, packed.len() as u64, len as u64)?);
            }
            let provenance = serde_json::json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"full_tensor":true,"bytes":packed.len()});
            return Ok(Projection { k, n, experts: 512, packed, provenance });
        }
        let mut packed = Vec::with_capacity(experts.len() * expert_bytes);
        let mut slices = Vec::new();
        for &e in experts {
            let bytes = shard.read_range(&name, (e * expert_bytes) as u64, expert_bytes as u64)?;
            slices.push(serde_json::json!({"expert":e,"offset_in_tensor":e*expert_bytes,"bytes":expert_bytes,"sha256":sha(&bytes)}));
            packed.extend(bytes);
        }
        let provenance = serde_json::json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"slices":slices,"packed_sha256":sha(&packed)});
        Ok(Projection {
            k,
            n,
            experts: experts.len(),
            packed,
            provenance,
        })
    }
}
pub struct Projection {
    pub k: usize,
    pub n: usize,
    pub experts: usize,
    pub packed: Vec<u8>,
    pub provenance: serde_json::Value,
}
pub struct ScaledProjection {
    pub raw: Projection,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
    pub scale_provenance: serde_json::Value,
}
impl Projection {
    /// CPU reference materialization only. Both device candidates keep EXL3 packed.
    pub fn oracle_weights(&self) -> Vec<f32> {
        assert!(
            self.experts <= 256,
            "full expert bank CPU decode is forbidden; select a compact subset"
        );
        let mut values = vec![0.; self.experts * self.k * self.n];
        for expert in 0..self.experts {
            for ti in 0..self.k / 16 {
                for to in 0..self.n / 16 {
                    let offset = ((expert * self.k / 16 + ti) * (self.n / 16) + to) * 128;
                    let decoded =
                        decode_k4_tile(self.packed[offset..offset + 128].try_into().unwrap(), 15);
                    for i in 0..16 {
                        for o in 0..16 {
                            values[(expert * self.k + ti * 16 + i) * self.n + to * 16 + o] =
                                f16::from_bits(decoded[i * 16 + o]).to_f32();
                        }
                    }
                }
            }
        }
        values
    }
    pub fn reference(&self, decoded: &[f32], x: &[u16], ids: &[u32]) -> Result<Vec<f64>> {
        ensure!(
            decoded.len() == self.experts * self.k * self.n && x.len() == ids.len() * self.k,
            "oracle shape"
        );
        ensure!(
            ids.iter().all(|&e| (e as usize) < self.experts),
            "oracle expert id"
        );
        let mut out = vec![0f64; ids.len() * self.n];
        for (row, &e) in ids.iter().enumerate() {
            let y = &mut out[row * self.n..(row + 1) * self.n];
            let w = &decoded[e as usize * self.k * self.n..(e as usize + 1) * self.k * self.n];
            for i in 0..self.k {
                let a = f16::from_bits(x[row * self.k + i]).to_f64();
                for (dst, &weight) in y.iter_mut().zip(&w[i * self.n..(i + 1) * self.n]) {
                    *dst += a * f64::from(weight);
                }
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorMetrics {
    pub values: usize,
    pub relative_l2: f64,
    pub max_absolute: f64,
    pub max_scaled: f64,
}
pub fn compare(actual: &[f32], reference: &[f64]) -> Result<ErrorMetrics> {
    ensure!(
        !actual.is_empty() && actual.len() == reference.len(),
        "comparison shape"
    );
    let (mut err2, mut norm2, mut maxerr, mut maxref) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in actual.iter().zip(reference) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite projection output"
        );
        let e = (f64::from(a) - b).abs();
        err2 += e * e;
        norm2 += b * b;
        maxerr = maxerr.max(e);
        maxref = maxref.max(b.abs());
    }
    Ok(ErrorMetrics {
        values: actual.len(),
        relative_l2: (err2 / norm2.max(1e-30)).sqrt(),
        max_absolute: maxerr,
        max_scaled: maxerr / maxref.max(1e-15),
    })
}
pub fn check_accuracy(m: &ErrorMetrics) -> Result<()> {
    // Trusted-reference control only, not a universal optimization threshold.
    // Deliberately approximate candidates should retain compare() diagnostics
    // and use their named policy plus later matched end-to-end quality evidence;
    // a modest component error alone is not grounds to discard that direction.
    ensure!(
        m.relative_l2 < 2e-5 && m.max_scaled < 1e-4,
        "projection error exceeds screen: {m:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "full expert bank CPU decode is forbidden")]
    fn refuses_full_bank_decode_before_allocation() {
        Projection {
            k: 2560,
            n: 640,
            experts: 512,
            packed: vec![],
            provenance: serde_json::Value::Null,
        }
        .oracle_weights();
    }
    #[test]
    fn raw_projection_has_input_output_and_expert_axes() {
        let p = Projection {
            k: 32,
            n: 16,
            experts: 2,
            packed: vec![0; 512],
            provenance: serde_json::Value::Null,
        };
        let mut w = vec![0.; 1024];
        for e in 0..2 {
            for i in 0..32 {
                for o in 0..16 {
                    w[(e * 32 + i) * 16 + o] = (1000 * e + 100 * i + o) as f32;
                }
            }
        }
        let mut x = vec![0u16; 64];
        x[3] = f16::ONE.to_bits();
        x[32 + 17] = f16::from_f32(-2.).to_bits();
        let y = p.reference(&w, &x, &[1, 0]).unwrap();
        for o in 0..16 {
            assert_eq!(y[o], (1300 + o) as f64);
            assert_eq!(y[16 + o], -2. * (1700 + o) as f64);
        }
        assert!(p.reference(&w, &x, &[2, 0]).is_err());
    }
}
