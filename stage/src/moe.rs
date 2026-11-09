//! Routed-expert branch (given route IDs/scores), not router/shared expert/HC.
use crate::projection::{Pack, ScaledProjection};
use anyhow::{Result, ensure};
use half::f16;
pub const H: usize = 2560;
pub const I: usize = 640;
pub const TOPK: usize = 10;
pub const MAX_ROWS: usize = 64;

/// Original 512-ID packed storage, with no public decoded-weight interface.
/// CPU references must load only selected compact subsets through ExpertBank.
pub struct FullExpertBank {
    pub(crate) gate: ScaledProjection,
    pub(crate) up: ScaledProjection,
    pub(crate) down: ScaledProjection,
    pub layer: usize,
}
impl FullExpertBank {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        Ok(Self {
            gate: pack.full_scaled_projection(layer, "gate_proj")?,
            up: pack.full_scaled_projection(layer, "up_proj")?,
            down: pack.full_scaled_projection(layer, "down_proj")?,
            layer,
        })
    }
    pub fn provenance(&self) -> serde_json::Value {
        serde_json::json!({"layer": self.layer, "original_experts":512,
            "projections":([&self.gate,&self.up,&self.down].map(|p|
                serde_json::json!({"trellis":p.raw.provenance,"scales":p.scale_provenance}))),
            "cpu_full_decode_available":false})
    }
}

pub struct ExpertBank {
    pub gate: ScaledProjection,
    pub up: ScaledProjection,
    pub down: ScaledProjection,
    pub original_ids: Vec<usize>,
}
impl ExpertBank {
    pub fn load(pack: &Pack, layer: usize, experts: &[usize]) -> Result<Self> {
        ensure!(
            experts.len() >= TOPK && experts.len() <= 128,
            "component expert-bank bound"
        );
        let mut sorted = experts.to_vec();
        sorted.sort_unstable();
        ensure!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "duplicate weight expert"
        );
        Ok(Self {
            gate: pack.scaled_projection(layer, "gate_proj", experts)?,
            up: pack.scaled_projection(layer, "up_proj", experts)?,
            down: pack.scaled_projection(layer, "down_proj", experts)?,
            original_ids: experts.to_vec(),
        })
    }
    pub fn decoded(&self) -> DecodedBank {
        DecodedBank {
            gate: self.gate.raw.oracle_weights(),
            up: self.up.raw.oracle_weights(),
            down: self.down.raw.oracle_weights(),
        }
    }
}
pub struct DecodedBank {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone)]
pub struct Routes {
    pub rows: usize,
    pub experts: usize,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}
impl Routes {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows) && (TOPK..=512).contains(&self.experts),
            "route dimensions"
        );
        ensure!(
            self.ids.len() == self.rows * TOPK && self.scores.len() == self.ids.len(),
            "route shape"
        );
        ensure!(
            self.ids.iter().all(|&i| (i as usize) < self.experts),
            "route expert out of bank"
        );
        ensure!(
            self.scores.iter().all(|x| x.is_finite() && *x >= 0.0),
            "route score invalid"
        );
        for row in 0..self.rows {
            let ids = &self.ids[row * TOPK..(row + 1) * TOPK];
            ensure!(
                ids.iter().enumerate().all(|(i, x)| !ids[..i].contains(x)),
                "duplicate top-k expert"
            );
        }
        Ok(())
    }
    pub fn plan(&self) -> Result<RoutePlan> {
        self.validate()?;
        let mut counts = vec![0usize; self.experts];
        for &id in &self.ids {
            counts[id as usize] += 1;
        }
        // Small groups first: each threshold's SIMD rows are a contiguous prefix.
        let mut order: Vec<u32> = (0..self.ids.len() as u32).collect();
        order.sort_unstable_by_key(|&slot| {
            let e = self.ids[slot as usize];
            (counts[e as usize], e, slot)
        });
        let ids: Vec<_> = order.iter().map(|&s| self.ids[s as usize]).collect();
        let mut inverse = vec![0u32; order.len()];
        for (sorted, &original) in order.iter().enumerate() {
            inverse[original as usize] = sorted as u32;
        }
        Ok(RoutePlan {
            order,
            inverse,
            ids,
            counts,
        })
    }
}
pub struct RoutePlan {
    pub order: Vec<u32>,
    pub inverse: Vec<u32>,
    pub ids: Vec<u32>,
    pub counts: Vec<usize>,
}
impl RoutePlan {
    /// (SIMD prefix rows, MPP16 descriptors) for a minimum reuse threshold.
    pub fn partition(&self, threshold: usize) -> (usize, Vec<[u32; 4]>) {
        let prefix = self
            .ids
            .iter()
            .take_while(|&&e| self.counts[e as usize] < threshold)
            .count();
        let mut blocks = vec![];
        let mut start = prefix;
        while start < self.ids.len() {
            let mut end = start + 1;
            while end < self.ids.len() && end - start < 16 && self.ids[end] == self.ids[start] {
                end += 1;
            }
            blocks.push([start as u32, (end - start) as u32, self.ids[start], 0]);
            start = end;
        }
        (prefix, blocks)
    }
}

/// Independent FP64 butterfly, normalized blockwise. Tests use a dense parity
/// matrix, independent of both this loop and the Metal SIMD shuffle layout.
pub fn hadamard(values: &mut [f64]) {
    assert!(values.len() % 128 == 0);
    for block in values.chunks_exact_mut(128) {
        for half in [1, 2, 4, 8, 16, 32, 64] {
            for base in (0..128).step_by(2 * half) {
                for j in 0..half {
                    let a = block[base + j];
                    let b = block[base + j + half];
                    block[base + j] = a + b;
                    block[base + j + half] = a - b;
                }
            }
        }
        for v in block {
            *v /= 128.0_f64.sqrt();
        }
    }
}
fn f16_round(values: &[f64]) -> Vec<u16> {
    values.iter().map(|&v| f16::from_f64(v).to_bits()).collect()
}
fn prepare(p: &ScaledProjection, x: &[f32], routes: &Routes) -> Vec<u16> {
    let mut out = vec![];
    for (slot, &expert) in routes.ids.iter().enumerate() {
        let mut v: Vec<_> = x[slot / TOPK * H..(slot / TOPK + 1) * H]
            .iter()
            .zip(&p.suh[expert as usize * H..(expert as usize + 1) * H])
            .map(|(&x, &s)| x as f64 * f16::from_bits(s).to_f64())
            .collect();
        hadamard(&mut v);
        out.extend(f16_round(&v));
    }
    out
}
pub struct Reference {
    pub prepared_gate: Vec<u16>,
    pub prepared_up: Vec<u16>,
    pub raw_gate: Vec<f64>,
    pub raw_up: Vec<f64>,
    pub middle: Vec<u16>,
    pub raw_down: Vec<f64>,
    pub output: Vec<f64>,
    pub bf16_output: Vec<f64>,
}
pub fn reference(
    bank: &ExpertBank,
    decoded: &DecodedBank,
    x: &[f32],
    routes: &Routes,
) -> Result<Reference> {
    routes.validate()?;
    ensure!(
        routes.experts == bank.original_ids.len()
            && x.len() == routes.rows * H
            && x.iter().all(|v| v.is_finite()),
        "reference inputs"
    );
    let prepared_gate = prepare(&bank.gate, x, routes);
    let prepared_up = prepare(&bank.up, x, routes);
    let raw_gate = bank
        .gate
        .raw
        .reference(&decoded.gate, &prepared_gate, &routes.ids)?;
    let raw_up = bank
        .up
        .raw
        .reference(&decoded.up, &prepared_up, &routes.ids)?;
    let mut middle = vec![];
    for (slot, &e) in routes.ids.iter().enumerate() {
        let mut g = raw_gate[slot * I..(slot + 1) * I].to_vec();
        let mut u = raw_up[slot * I..(slot + 1) * I].to_vec();
        hadamard(&mut g);
        hadamard(&mut u);
        let mut m = vec![0.; I];
        for j in 0..I {
            let at = e as usize * I + j;
            let gate = g[j] * f16::from_bits(bank.gate.svh[at]).to_f64();
            let up = u[j] * f16::from_bits(bank.up.svh[at]).to_f64();
            let ex = (-gate.abs()).exp();
            let sig = if gate < 0.0 {
                ex / (1.0 + ex)
            } else {
                1.0 / (1.0 + ex)
            };
            m[j] = gate * sig * up * f16::from_bits(bank.down.suh[at]).to_f64();
        }
        hadamard(&mut m);
        middle.extend(f16_round(&m));
    }
    let raw_down = bank
        .down
        .raw
        .reference(&decoded.down, &middle, &routes.ids)?;
    let mut output = vec![0.; routes.rows * H];
    for (slot, &e) in routes.ids.iter().enumerate() {
        let mut v = raw_down[slot * H..(slot + 1) * H].to_vec();
        hadamard(&mut v);
        for j in 0..H {
            output[slot / TOPK * H + j] += v[j]
                * f16::from_bits(bank.down.svh[e as usize * H + j]).to_f64()
                * routes.scores[slot] as f64;
        }
    }
    let bf16_output = output
        .iter()
        .map(|&v| crate::rounding::bf16_from_f64(v).to_f64())
        .collect();
    Ok(Reference {
        prepared_gate,
        prepared_up,
        raw_gate,
        raw_up,
        middle,
        raw_down,
        output,
        bf16_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn h128_matches_independent_parity_matrix() {
        let x: Vec<_> = (0..256)
            .map(|i| ((i * 71) % 101) as f64 / 37.0 - 1.0)
            .collect();
        let mut y = x.clone();
        hadamard(&mut y);
        for i in 0..256 {
            let want = (0..128)
                .map(|j| {
                    x[i / 128 * 128 + j]
                        * (if ((i % 128) & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        })
                        / 128.0_f64.sqrt()
                })
                .sum::<f64>();
            assert!((y[i] - want).abs() < 2e-14);
        }
        hadamard(&mut y);
        for (a, b) in y.iter().zip(x) {
            assert!((a - b).abs() < 2e-14);
        }
    }
    #[test]
    fn route_partition_preserves_every_original_score_and_row() {
        let mut ids = vec![];
        for r in 0..7 {
            for j in 0..TOPK {
                ids.push(((r * 3 + j) % 16) as u32);
            }
        }
        let r = Routes {
            rows: 7,
            experts: 16,
            ids,
            scores: vec![0.1; 70],
        };
        let p = r.plan().unwrap();
        for (i, &s) in p.order.iter().enumerate() {
            assert_eq!(p.inverse[s as usize], i as u32);
            assert_eq!(p.ids[i], r.ids[s as usize]);
        }
        for threshold in [0, 2, 4, 8, usize::MAX] {
            let (prefix, blocks) = p.partition(threshold);
            let mut covered = vec![0u8; 70];
            for c in &mut covered[..prefix] {
                *c += 1;
            }
            for [start, count, e, _] in blocks {
                assert!((1..=16).contains(&count));
                for i in start..start + count {
                    assert_eq!(p.ids[i as usize], e);
                    covered[i as usize] += 1;
                }
            }
            assert!(covered.iter().all(|&n| n == 1));
        }
        let mut bad = r.clone();
        bad.ids[1] = bad.ids[0];
        assert!(bad.validate().is_err());
        bad = r.clone();
        bad.scores[0] = f32::NAN;
        assert!(bad.validate().is_err());
    }
}
