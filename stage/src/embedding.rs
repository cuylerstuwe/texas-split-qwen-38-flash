//! Bounded raw token embeddings from the pinned target's affine8 table.
//! No PLE, hyper-connections, embedding normalization or GPU initialization.
use crate::projection::{Pack, sha};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};

pub const HIDDEN: usize = 2560;
pub const VOCAB: usize = 248320;
pub const MASK: usize = 248077;
const PREFIX: &str = "language_model.model.embed_tokens";

pub struct RawEmbeddings {
    pub tokens: Vec<usize>,
    pub values: Vec<u16>,
    pub provenance: Value,
}
impl RawEmbeddings {
    pub fn load(pack: &Pack, tokens: &[usize]) -> Result<Self> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= 64 && tokens.iter().all(|&t| t < VOCAB),
            "embedding token count/range"
        );
        let (w, codes, wp) = pack.tensor_rows(&format!("{PREFIX}.weight"), tokens)?;
        let (s, scales, sp) = pack.tensor_rows(&format!("{PREFIX}.scales"), tokens)?;
        let (b, biases, bp) = pack.tensor_rows(&format!("{PREFIX}.biases"), tokens)?;
        ensure!(
            w.dtype == "U32" && w.shape == [VOCAB as u64, (HIDDEN / 4) as u64],
            "raw embedding affine8 geometry"
        );
        ensure!(
            s.dtype == "BF16"
                && b.dtype == "BF16"
                && s.shape == b.shape
                && s.shape.len() == 2
                && s.shape[0] == VOCAB as u64
                && s.shape[1] > 0,
            "raw embedding affine metadata"
        );
        let groups = s.shape[1] as usize;
        ensure!(
            HIDDEN % groups == 0 && [64, 128].contains(&(HIDDEN / groups)),
            "raw embedding group size"
        );
        let group = HIDDEN / groups;
        let values = decode_affine8(&codes, &scales, &biases, group)?;
        ensure!(
            values.len() == tokens.len() * HIDDEN,
            "embedding output geometry"
        );
        let raw = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(Self {
            tokens: tokens.to_vec(),
            values,
            provenance: json!({"kind":"raw_affine8_target_embedding", "tokens":tokens,
                "shape":[tokens.len(),HIDDEN], "group":group, "sources":[wp,sp,bp],
                "arithmetic":"FP32 fused affine then one BF16 round-to-nearest-even",
                "output_sha256":sha(&raw), "gpu_opened":false,
                "upstream_metal_rounding_comparison_pending":true}),
        })
    }
}

/// R58 prefill: the same affine8 embedding rows as RawEmbeddings (identical arithmetic: FP32
/// fused affine, one BF16 round-to-nearest-even) for any number of tokens, from a shard opened
/// once (positioned reads, decoded on a few threads; no per-call provenance hashing).
pub struct EmbeddingTable {
    file: std::fs::File,
    at: [u64; 3],
    groups: usize,
}
impl EmbeddingTable {
    pub fn open(pack: &Pack) -> Result<Self> {
        let names = ["weight", "scales", "biases"].map(|s| format!("{PREFIX}.{s}"));
        let path = pack.file_of(&names[0])?;
        ensure!(names.iter().all(|n| pack.file_of(n).ok().as_ref() == Some(&path)), "embedding tensors in one shard");
        let shard = crate::weights::Shard::open(&path)?;
        let t = |n: &str| shard.tensors.get(n).cloned().ok_or_else(|| anyhow::anyhow!("missing {n}"));
        let (w, s, b) = (t(&names[0])?, t(&names[1])?, t(&names[2])?);
        ensure!(w.dtype == "U32" && w.shape == [VOCAB as u64, (HIDDEN / 4) as u64], "raw embedding affine8 geometry");
        ensure!(s.dtype == "BF16" && b.dtype == "BF16" && s.shape == b.shape && s.shape.len() == 2 && s.shape[0] == VOCAB as u64, "raw embedding affine metadata");
        let groups = s.shape[1] as usize;
        ensure!(groups > 0 && HIDDEN % groups == 0 && [64, 128].contains(&(HIDDEN / groups)), "raw embedding group size");
        Ok(Self { file: std::fs::File::open(&path)?, at: [w, s, b].map(|x| shard.data_start + x.data_offsets[0]), groups })
    }
    /// BF16 rows (tokens x HIDDEN).
    pub fn rows(&self, tokens: &[u32]) -> Result<Vec<u16>> {
        use std::os::unix::fs::FileExt;
        ensure!(tokens.iter().all(|&t| (t as usize) < VOCAB), "embedding token range");
        let (g, group) = (self.groups, HIDDEN / self.groups);
        let mut out = vec![0u16; tokens.len() * HIDDEN];
        let per = tokens.len().div_ceil(4).max(1);
        std::thread::scope(|sc| -> Result<()> {
            let mut handles = vec![];
            for (ts, o) in tokens.chunks(per).zip(out.chunks_mut(per * HIDDEN)) {
                handles.push(sc.spawn(move || -> Result<()> {
                    let (mut codes, mut sb) = (vec![0u8; HIDDEN], vec![0u8; g * 4]);
                    for (r, &tok) in ts.iter().enumerate() {
                        let t = tok as u64;
                        self.file.read_exact_at(&mut codes, self.at[0] + t * HIDDEN as u64)?;
                        self.file.read_exact_at(&mut sb[..g * 2], self.at[1] + t * g as u64 * 2)?;
                        self.file.read_exact_at(&mut sb[g * 2..], self.at[2] + t * g as u64 * 2)?;
                        let row = &mut o[r * HIDDEN..(r + 1) * HIDDEN];
                        for j in 0..g {
                            let s = bf16::from_bits(u16::from_le_bytes([sb[2 * j], sb[2 * j + 1]])).to_f32();
                            let b = bf16::from_bits(u16::from_le_bytes([sb[g * 2 + 2 * j], sb[g * 2 + 2 * j + 1]])).to_f32();
                            ensure!(s.is_finite() && b.is_finite(), "nonfinite embedding metadata");
                            for k in j * group..(j + 1) * group {
                                let v = bf16::from_f32(s.mul_add(f32::from(codes[k]), b));
                                ensure!(v.is_finite(), "nonfinite embedding output");
                                row[k] = v.to_bits();
                            }
                        }
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join().map_err(|_| anyhow::anyhow!("embedding thread"))??;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

fn decode_affine8(codes: &[u8], scales: &[u8], biases: &[u8], group: usize) -> Result<Vec<u16>> {
    ensure!(
        [64, 128].contains(&group)
            && !codes.is_empty()
            && codes.len() % group == 0
            && scales.len() == codes.len() / group * 2
            && biases.len() == scales.len(),
        "affine embedding selected row lengths"
    );
    let mut out = Vec::with_capacity(codes.len());
    for ((c, s), b) in codes
        .chunks_exact(group)
        .zip(scales.chunks_exact(2))
        .zip(biases.chunks_exact(2))
    {
        let s = bf16::from_bits(u16::from_le_bytes(s.try_into().unwrap())).to_f32();
        let b = bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32();
        ensure!(
            s.is_finite() && b.is_finite(),
            "nonfinite embedding metadata"
        );
        for &code in c {
            let v = bf16::from_f32(s.mul_add(f32::from(code), b));
            ensure!(v.is_finite(), "nonfinite embedding output");
            out.push(v.to_bits());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
            .collect()
    }
    #[test]
    fn affine_embedding_groups_sign_and_halfway_rounding() {
        // Distinct groups, all byte codes, signed scales and exact cancellation.
        let codes = (0..512).map(|i| (i % 256) as u8).collect::<Vec<_>>();
        let scales = [1.0 / 128.0, -1.0 / 64.0, 1.0 / 256.0, -1.0 / 512.0];
        let biases = [-1.0, 3.0, 1.0, 0.5];
        let got = decode_affine8(&codes, &bits(&scales), &bits(&biases), 128).unwrap();
        for i in 0..512 {
            let g = i / 128;
            let exact = f64::from(scales[g]) * f64::from(codes[i]) + f64::from(biases[g]);
            assert_eq!(
                got[i],
                crate::rounding::bf16_from_f64(exact).to_bits(),
                "element {i}"
            );
        }
        // Group2's values 1+1/256 and 1+3/256 are ties; opposite even neighbors.
        assert_eq!(got[257], 0x3f80);
        assert_eq!(got[259], 0x3f82);
        assert!(decode_affine8(&codes, &bits(&scales), &bits(&biases), 32).is_err());
        assert!(decode_affine8(&codes, &bits(&scales[..3]), &bits(&biases), 128).is_err());
        assert!(decode_affine8(&codes, &bits(&[f32::NAN; 4]), &bits(&biases), 128).is_err());
    }
}
