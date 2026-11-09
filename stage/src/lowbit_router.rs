//! Full512-expert lowbit router weights and independent CPU validation.
use crate::{
    bf16_linear::Matrix,
    projection::sha,
    target_mlp::{self, RoutingArithmetic},
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Serialize;
use serde_json::{Value, json};
use std::path::{Component, Path};
pub const H: usize = 2560;
pub const EXPERTS: usize = 512;
pub const TOPK: usize = 10;
pub const MAX_ROWS: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scores {
    StagedBf16,
    SelectedF32,
}
pub fn load(root: &Path, layer: usize) -> Result<Matrix> {
    ensure!(layer < 48, "router layer bound");
    let config = std::fs::read(root.join("config.json"))?;
    let index_bytes = std::fs::read(root.join("model.safetensors.index.json"))?;
    ensure!(
        sha(&config) == "fd9d242648d6a5a1ef130f8f31234bc182b0c39ea175241c0e7c22a65d241454",
        "router lowbit config pin"
    );
    ensure!(
        sha(&index_bytes) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
        "router lowbit index pin"
    );
    let index: Value = serde_json::from_slice(&index_bytes)?;
    let name = format!("language_model.model.layers.{layer}.mlp.gate.weight");
    let file = index["weight_map"][&name]
        .as_str()
        .context("router weight index")?;
    let mut parts = Path::new(file).components();
    ensure!(
        matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
        "router shard path"
    );
    let shard = Shard::open(&root.join(file))?;
    let t = shard.tensors.get(&name).context("router tensor")?;
    ensure!(
        t.dtype == "BF16"
            && t.shape == [EXPERTS as u64, H as u64]
            && t.bytes() == (H * EXPERTS * 2) as u64,
        "router BF16 geometry"
    );
    let raw = shard.read_range(&name, 0, t.bytes())?;
    let w = Matrix {
        k: H,
        n: EXPERTS,
        values: raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        provenance: json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"payload_sha256":sha(&raw),
            "config_sha256":sha(&config),"index_sha256":sha(&index_bytes)}),
    };
    w.validate()?;
    Ok(w)
}
#[derive(Clone)]
pub struct Routing {
    pub rows: usize,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}
impl Routing {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows)
                && self.ids.len() == self.rows * TOPK
                && self.scores.len() == self.ids.len(),
            "full router shape"
        );
        ensure!(
            self.ids.iter().all(|&e| e < EXPERTS as u32)
                && self
                    .scores
                    .iter()
                    .all(|&s| s.is_finite() && s >= 0. && s <= 1.),
            "full router values"
        );
        for (ids, scores) in self
            .ids
            .chunks_exact(TOPK)
            .zip(self.scores.chunks_exact(TOPK))
        {
            ensure!(
                ids.iter().enumerate().all(|(i, e)| !ids[..i].contains(e)),
                "full router duplicate ID"
            );
            ensure!(
                (scores.iter().map(|&s| s as f64).sum::<f64>() - 1.).abs() < 0.04,
                "full router normalization"
            );
        }
        Ok(())
    }
}
pub fn select(logits: &[u16], arm: Scores) -> Result<Routing> {
    ensure!(
        !logits.is_empty()
            && logits.len() % EXPERTS == 0
            && logits.len() / EXPERTS <= MAX_ROWS
            && logits.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "router logits geometry/finite"
    );
    let mut result = Routing {
        rows: logits.len() / EXPERTS,
        ids: vec![],
        scores: vec![],
    };
    // Reuse the source-grounded scalar reference row-by-row, without widening
    // the older target component's full-model/decoded-bank memory bounds.
    for row in logits.chunks_exact(EXPERTS) {
        let r = target_mlp::route(
            row,
            match arm {
                Scores::StagedBf16 => RoutingArithmetic::StagedBf16,
                Scores::SelectedF32 => RoutingArithmetic::SelectedF32,
            },
        )?;
        result.ids.extend(r.ids);
        result.scores.extend(r.scores);
    }
    result.validate()?;
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_ids_ties_and_tail_experts_are_preserved() {
        let h = |x: f32| bf16::from_f32(x).to_bits();
        let mut x = vec![0; 3 * EXPERTS];
        for (i, v) in x[EXPERTS..2 * EXPERTS].iter_mut().enumerate() {
            *v = h(i as f32 - 256.);
        }
        x[2 * EXPERTS + 511] = h(100.);
        x[2 * EXPERTS + 128] = h(100.);
        for arm in [Scores::StagedBf16, Scores::SelectedF32] {
            let r = select(&x, arm).unwrap();
            assert_eq!(&r.ids[..10], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            assert_eq!(
                &r.ids[10..20],
                &[511, 510, 509, 508, 507, 506, 505, 504, 503, 502]
            );
            assert_eq!(&r.ids[20..23], &[128, 511, 0]);
            assert!(select(&x[..EXPERTS - 1], arm).is_err());
        }
    }
}
