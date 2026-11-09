//! Source-grounded Qwen target router/shared-expert CPU reference. Expert
//! computation itself reuses the independently decoded EXL3 component oracle.
//! FP64 reductions preserve semantic BF16 boundaries, not GPU reduction order.
use crate::{
    bf16_linear::Matrix,
    dense::Affine8,
    moe::{self, ExpertBank, H, I, Routes, TOPK},
    projection::Pack,
};
use anyhow::{Result, ensure};
use half::{bf16, f16};
use serde::Serialize;
use serde_json::json;
pub const EXPERTS: usize = 512;
pub const MAX_ROWS: usize = 8;
fn number(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
fn bits(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
fn round(x: f64) -> f64 {
    number(bits(x))
}
/// Source's two-sided BF16 sigmoid with every intermediate dtype boundary.
/// CPU exp is an oracle, not a copy of MLX's GPU sigmoid lookup table.
fn sigmoid(x: f64) -> f64 {
    let small = round(1. / round(1. + round(x.abs().exp())));
    if x < 0. { small } else { round(1. - small) }
}
pub struct Weights {
    pub router: Matrix,
    pub shared_gate: Matrix,
    pub shared: [Affine8; 3],
    pub layer: usize,
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48, "target MLP layer");
        let p = format!("language_model.model.layers.{layer}.mlp");
        let load = |suffix: &str, n: usize| -> Result<Matrix> {
            let (t, b, provenance) = pack.tensor(&format!("{p}.{suffix}.weight"))?;
            ensure!(
                t.dtype == "BF16" && t.shape == [n as u64, H as u64],
                "target BF16 router/gate geometry"
            );
            let m = Matrix {
                k: H,
                n,
                values: b
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                provenance,
            };
            m.validate()?;
            Ok(m)
        };
        let router = load("gate", EXPERTS)?;
        let shared_gate = load("shared_expert_gate", 1)?;
        let shared = [
            Affine8::load(pack, &format!("{p}.shared_expert.gate_proj"), H)?,
            Affine8::load(pack, &format!("{p}.shared_expert.up_proj"), H)?,
            Affine8::load(pack, &format!("{p}.shared_expert.down_proj"), I)?,
        ];
        ensure!(
            shared[0].n == I && shared[1].n == I && shared[2].n == H,
            "target shared expert dimensions"
        );
        Ok(Self {
            router,
            shared_gate,
            shared,
            layer,
        })
    }
    pub fn decoded_shared(&self) -> Result<[Vec<f32>; 3]> {
        Ok([
            self.shared[0].oracle_weights()?,
            self.shared[1].oracle_weights()?,
            self.shared[2].oracle_weights()?,
        ])
    }
    pub fn provenance(&self) -> serde_json::Value {
        json!({"layer":self.layer,"router":self.router.provenance,"shared_gate":self.shared_gate.provenance,"shared":self.shared.iter().map(|w|&w.provenance).collect::<Vec<_>>()})
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingArithmetic {
    StagedBf16,
    SelectedF32,
}
pub struct Routing {
    pub rows: usize,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}
/// The source selects on BF16 logits, with lowest-ID ties. Its default scores
/// round the full softmax, selected sum (each add), and quotient to BF16.
/// SelectedF32 explicitly removes those score round trips; it is a candidate.
pub fn route(logits: &[u16], arm: RoutingArithmetic) -> Result<Routing> {
    ensure!(
        !logits.is_empty() && logits.len() % EXPERTS == 0 && logits.len() / EXPERTS <= MAX_ROWS,
        "router shape"
    );
    ensure!(
        logits.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "router nonfinite logits"
    );
    let rows = logits.len() / EXPERTS;
    let mut ids = vec![];
    let mut scores = vec![];
    for row in logits.chunks_exact(EXPERTS) {
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_unstable_by(|&a, &b| {
            number(row[b])
                .partial_cmp(&number(row[a]))
                .unwrap()
                .then(a.cmp(&b))
        });
        let top = &order[..TOPK];
        let max = number(row[top[0]]);
        let exp: Vec<f64> = row.iter().map(|&v| (number(v) - max).exp()).collect();
        let weights: Vec<f64> = match arm {
            RoutingArithmetic::StagedBf16 => {
                let sum = exp.iter().sum::<f64>();
                let selected: Vec<_> = top.iter().map(|&i| round(exp[i] / sum)).collect();
                let denom = selected.iter().fold(0., |s, &v| round(s + v));
                ensure!(denom > 0. && denom.is_finite(), "selected BF16 sum");
                selected.iter().map(|v| round(v / denom)).collect()
            }
            RoutingArithmetic::SelectedF32 => {
                let denom = top.iter().map(|&i| exp[i]).sum::<f64>();
                top.iter().map(|&i| exp[i] / denom).collect()
            }
        };
        ids.extend(top.iter().map(|&i| i as u32));
        scores.extend(weights.iter().map(|&v| v as f32));
    }
    Ok(Routing { rows, ids, scores })
}
impl Routing {
    /// Compact fixture bank: keep original model IDs explicitly, remap only the
    /// component's selected-weight storage. No routing pruning or hit-rate claim.
    pub fn bank_routes(&self) -> Result<(Vec<usize>, Routes)> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows)
                && self.ids.len() == self.rows * TOPK
                && self.scores.len() == self.ids.len(),
            "routing output shape"
        );
        ensure!(
            self.ids.iter().all(|&i| i < EXPERTS as u32),
            "model expert ID"
        );
        let mut original: Vec<usize> = self.ids.iter().map(|&i| i as usize).collect();
        original.sort_unstable();
        original.dedup();
        let routes = Routes {
            rows: self.rows,
            experts: original.len(),
            ids: self
                .ids
                .iter()
                .map(|&i| original.binary_search(&(i as usize)).unwrap() as u32)
                .collect(),
            scores: self.scores.clone(),
        };
        routes.validate()?;
        Ok((original, routes))
    }
}
pub struct Shared {
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub activation: Vec<u16>,
    pub down: Vec<u16>,
    pub gate_logits: Vec<u16>,
    pub gate_sigmoid: Vec<u16>,
    pub output: Vec<u16>,
}
pub fn shared_reference(w: &Weights, decoded: &[Vec<f32>; 3], input: &[u16]) -> Result<Shared> {
    ensure!(
        !input.is_empty() && input.len() % H == 0 && input.len() / H <= MAX_ROWS,
        "shared input rows"
    );
    let rows = input.len() / H;
    let gate: Vec<_> = w.shared[0]
        .reference(&decoded[0], input)?
        .into_iter()
        .map(bits)
        .collect();
    let up: Vec<_> = w.shared[1]
        .reference(&decoded[1], input)?
        .into_iter()
        .map(bits)
        .collect();
    let activation: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| bits(round(number(g) * sigmoid(number(g))) * number(u)))
        .collect();
    let down: Vec<_> = w.shared[2]
        .reference(&decoded[2], &activation)?
        .into_iter()
        .map(bits)
        .collect();
    let gate_logits: Vec<_> = w
        .shared_gate
        .reference(input)?
        .into_iter()
        .map(bits)
        .collect();
    let gate_sigmoid: Vec<_> = gate_logits
        .iter()
        .map(|&g| bits(sigmoid(number(g))))
        .collect();
    let output: Vec<_> = (0..rows * H)
        .map(|i| bits(number(down[i]) * number(gate_sigmoid[i / H])))
        .collect();
    ensure!(
        output.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "shared finite output"
    );
    Ok(Shared {
        gate,
        up,
        activation,
        down,
        gate_logits,
        gate_sigmoid,
        output,
    })
}
pub fn combine(routed: &[f64], shared: &[u16]) -> Result<Vec<u16>> {
    ensure!(
        !routed.is_empty()
            && routed.len() == shared.len()
            && routed.iter().all(|v| v.is_finite())
            && shared.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "MLP combine inputs"
    );
    Ok(routed
        .iter()
        .zip(shared)
        .map(|(&r, &s)| bits(round(r) + number(s)))
        .collect())
}
/// Reuse the exact same decoded expert outputs for a router-score-only arm.
/// The caller proves that both route lists have the identical original IDs.
pub fn reweight(bank: &ExpertBank, raw_down: &[f64], routes: &Routes) -> Result<Vec<f64>> {
    routes.validate()?;
    ensure!(
        bank.original_ids.len() == routes.experts
            && raw_down.len() == routes.ids.len() * H
            && raw_down.iter().all(|v| v.is_finite()),
        "reweight axes/finite inputs"
    );
    let mut output = vec![0.; routes.rows * H];
    for (slot, &e) in routes.ids.iter().enumerate() {
        let mut v = raw_down[slot * H..(slot + 1) * H].to_vec();
        moe::hadamard(&mut v);
        for j in 0..H {
            output[slot / TOPK * H + j] += v[j]
                * f16::from_bits(bank.down.svh[e as usize * H + j]).to_f64()
                * routes.scores[slot] as f64;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_routes_and_semantic_score_rounding() {
        let logits = vec![bits(3.); 2 * EXPERTS];
        let a = route(&logits, RoutingArithmetic::StagedBf16).unwrap();
        let b = route(&logits, RoutingArithmetic::SelectedF32).unwrap();
        assert_eq!(
            a.ids,
            (0..2).flat_map(|_| 0..TOPK as u32).collect::<Vec<_>>()
        );
        assert_eq!(a.ids, b.ids);
        assert!(
            a.scores
                .iter()
                .all(|&v| v == crate::rounding::bf16_from_f64(0.1).to_f32())
        );
        assert!(b.scores.iter().all(|&v| v == 0.1f32));
        let (original, r) = a.bank_routes().unwrap();
        assert_eq!(original, (0..TOPK).collect::<Vec<_>>());
        assert_eq!(r.ids, a.ids);
        assert!(route(&[], RoutingArithmetic::StagedBf16).is_err());
        let mut bad = logits;
        bad[0] = bf16::NAN.to_bits();
        assert!(route(&bad, RoutingArithmetic::SelectedF32).is_err());
    }
    #[test]
    fn learned_selection_preserves_original_ids() {
        let mut logits = vec![bits(-100.); EXPERTS];
        for i in 0..TOPK {
            logits[511 - i * 13] = bits(i as f64);
        }
        let a = route(&logits, RoutingArithmetic::StagedBf16).unwrap();
        assert_eq!(
            a.ids,
            (0..TOPK)
                .rev()
                .map(|i| (511 - i * 13) as u32)
                .collect::<Vec<_>>()
        );
        let (original, r) = a.bank_routes().unwrap();
        for (i, &id) in r.ids.iter().enumerate() {
            assert_eq!(original[id as usize], a.ids[i] as usize);
        }
        assert_eq!(r.scores, a.scores);
        assert!(sigmoid(-1000.) == 0. && sigmoid(1000.) == 1. && sigmoid(0.) == 0.5);
    }
}
