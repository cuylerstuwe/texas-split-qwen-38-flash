//! Explicit K1.75 packed slices; the original K4 Pack contract is unchanged.
use crate::{
    exl3::decode_mcg_tile,
    projection::{Projection, sha},
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Component, Path},
};

pub struct McgProjection {
    pub k: usize,
    pub n: usize,
    pub experts: usize,
    pub halfwords: usize,
    pub packed: Vec<u8>,
    pub provenance: serde_json::Value,
}

impl McgProjection {
    pub fn original(p: Projection) -> Self {
        Self {
            k: p.k,
            n: p.n,
            experts: p.experts,
            halfwords: 64,
            packed: p.packed,
            provenance: p.provenance,
        }
    }

    pub fn lowbit(root: &Path, layer: usize, projection: &str, experts: &[usize]) -> Result<Self> {
        ensure!(
            !experts.is_empty() && experts.len() <= 256,
            "compact projection bound"
        );
        Self::load_lowbit(root, layer, projection, experts)
    }

    /// Packed full address space only; never implicitly expands all experts.
    pub(crate) fn full_lowbit(root: &Path, layer: usize, projection: &str) -> Result<Self> {
        Self::load_lowbit(root, layer, projection, &(0..512).collect::<Vec<_>>())
    }

    fn load_lowbit(root: &Path, layer: usize, projection: &str, experts: &[usize]) -> Result<Self> {
        ensure!(
            layer < 48 && ["gate_proj", "up_proj", "down_proj"].contains(&projection),
            "lowbit layer/projection"
        );
        ensure!(
            !experts.is_empty() && experts.len() <= 512 && experts.iter().all(|&e| e < 512),
            "lowbit expert bounds"
        );
        let config = std::fs::read(root.join("config.json"))?;
        let index = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&config) == "fd9d242648d6a5a1ef130f8f31234bc182b0c39ea175241c0e7c22a65d241454",
            "completed lowbit config pin"
        );
        ensure!(
            sha(&index) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
            "completed lowbit index pin"
        );
        let value: serde_json::Value = serde_json::from_slice(&index)?;
        let map: BTreeMap<String, String> = serde_json::from_value(value["weight_map"].clone())?;
        let name =
            format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.trellis");
        let file = map.get(&name).context("lowbit projection index")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "unsafe lowbit shard"
        );
        let shard = Shard::open(&root.join(file))?;
        let tensor = shard
            .tensors
            .get(&name)
            .context("lowbit projection absent")?;
        let (k, n) = if projection == "down_proj" {
            (640, 2560)
        } else {
            (2560, 640)
        };
        ensure!(
            tensor.dtype == "U16" && tensor.shape == [512, k as u64 / 16, n as u64 / 16, 28],
            "K1.75 trellis geometry"
        );
        for (key, expected) in [
            ("format", "exl3"),
            ("k", "1.75"),
            ("codebook", "mcg"),
            ("window", "15"),
            ("encoder", "sushi-encoder-1"),
        ] {
            ensure!(
                shard.metadata.get(key).map(String::as_str) == Some(expected),
                "lowbit metadata {key}"
            );
        }
        let expert_bytes = k * n / 256 * 56;
        let mut packed = Vec::with_capacity(experts.len() * expert_bytes);
        let mut slices = Vec::new();
        for &e in experts {
            let data = shard.read_range(&name, (e * expert_bytes) as u64, expert_bytes as u64)?;
            slices.push(json!({"expert":e,"offset_in_tensor":e*expert_bytes,"bytes":expert_bytes,"sha256":sha(&data)}));
            packed.extend(data);
        }
        let provenance = json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"config_sha256":sha(&config),"index_sha256":sha(&index),"slices":slices,"packed_sha256":sha(&packed)});
        Ok(Self {
            k,
            n,
            experts: experts.len(),
            halfwords: 28,
            packed,
            provenance,
        })
    }

    pub fn oracle_weights(&self) -> Result<Vec<f32>> {
        ensure!(
            self.experts <= 256,
            "decode selected expert subsets, not a full bank"
        );
        ensure!(
            [28, 64].contains(&self.halfwords)
                && self.packed.len() == self.experts * self.k * self.n / 256 * self.halfwords * 2,
            "MCG storage geometry"
        );
        let bytes = self.halfwords * 2;
        let mut values = vec![0.; self.experts * self.k * self.n];
        for e in 0..self.experts {
            for ti in 0..self.k / 16 {
                for to in 0..self.n / 16 {
                    let at = ((e * self.k / 16 + ti) * (self.n / 16) + to) * bytes;
                    let tile = decode_mcg_tile(&self.packed[at..at + bytes], self.halfwords, 15)?;
                    for i in 0..16 {
                        for o in 0..16 {
                            values[(e * self.k + ti * 16 + i) * self.n + to * 16 + o] =
                                f16::from_bits(tile[i * 16 + o]).to_f32();
                        }
                    }
                }
            }
        }
        Ok(values)
    }

    pub fn reference(&self, w: &[f32], x: &[u16], ids: &[u32]) -> Result<Vec<f64>> {
        ensure!(
            w.len() == self.experts * self.k * self.n
                && x.len() == ids.len() * self.k
                && ids.iter().all(|&e| (e as usize) < self.experts),
            "MCG oracle geometry"
        );
        let mut out = vec![0.; ids.len() * self.n];
        for (r, &e) in ids.iter().enumerate() {
            for i in 0..self.k {
                let a = f16::from_bits(x[r * self.k + i]).to_f64();
                for o in 0..self.n {
                    out[r * self.n + o] += a * f64::from(w[(e as usize * self.k + i) * self.n + o]);
                }
            }
        }
        ensure!(out.iter().all(|v| v.is_finite()), "MCG oracle finite");
        Ok(out)
    }
}
