//! Full lowbit routed-expert reference; CPU weight-format oracle, not CPU serving.
use crate::{
    mcg_projection::McgProjection,
    moe::{DecodedBank, H, I, Reference, Routes, TOPK, hadamard},
    projection::sha,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde_json::{Value, json};
use std::path::{Component, Path};

pub struct Scaled {
    pub raw: McgProjection,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
    pub scale_provenance: Value,
}
impl Scaled {
    fn load(root: &Path, layer: usize, name: &str, experts: &[usize]) -> Result<Self> {
        let raw = if experts.len() == 512 && experts.iter().copied().eq(0..512) {
            McgProjection::full_lowbit(root, layer, name)?
        } else {
            McgProjection::lowbit(root, layer, name, experts)?
        };
        let index: Value =
            serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json"))?)?;
        let mut scales = vec![];
        let mut provenance = vec![];
        for (suffix, dim) in [("suh", raw.k), ("svh", raw.n)] {
            let tensor =
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.{name}.{suffix}");
            let file = index["weight_map"][&tensor]
                .as_str()
                .context("scale index")?;
            let mut parts = Path::new(file).components();
            ensure!(
                matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
                "scale path"
            );
            let shard = Shard::open(&root.join(file))?;
            let t = shard.tensors.get(&tensor).context("scale tensor")?;
            ensure!(
                t.dtype == "F16" && t.shape == [512, dim as u64],
                "lowbit scale geometry"
            );
            let mut values = vec![];
            let mut receipts = vec![];
            for &e in experts {
                let b = shard.read_range(&tensor, (e * dim * 2) as u64, (dim * 2) as u64)?;
                receipts.push(json!({"expert":e,"sha256":sha(&b)}));
                values.extend(
                    b.chunks_exact(2)
                        .map(|x| u16::from_le_bytes(x.try_into().unwrap())),
                );
            }
            ensure!(
                values.iter().all(|&x| f16::from_bits(x).is_finite()),
                "nonfinite lowbit scale"
            );
            provenance.push(json!({"tensor":tensor,"file":file,"header_sha256":shard.header_sha256,"slices":receipts}));
            scales.push(values);
        }
        let svh = scales.pop().unwrap();
        let suh = scales.pop().unwrap();
        Ok(Self {
            raw,
            suh,
            svh,
            scale_provenance: json!(provenance),
        })
    }
}
pub struct Bank {
    pub gate: Scaled,
    pub up: Scaled,
    pub down: Scaled,
    pub original_ids: Vec<usize>,
}
impl Bank {
    pub fn load(root: &Path, layer: usize, experts: &[usize]) -> Result<Self> {
        ensure!((TOPK..=128).contains(&experts.len()), "lowbit bank bound");
        ensure!(
            experts
                .iter()
                .enumerate()
                .all(|(i, e)| !experts[..i].contains(e)),
            "duplicate lowbit expert"
        );
        Ok(Self {
            gate: Scaled::load(root, layer, "gate_proj", experts)?,
            up: Scaled::load(root, layer, "up_proj", experts)?,
            down: Scaled::load(root, layer, "down_proj", experts)?,
            original_ids: experts.to_vec(),
        })
    }
    /// Explicit original-ID bank. Compact CPU/GPU reference limits stay intact.
    pub fn full_packed(root: &Path, layer: usize) -> Result<Self> {
        let ids: Vec<_> = (0..512).collect();
        Ok(Self {
            gate: Scaled::load(root, layer, "gate_proj", &ids)?,
            up: Scaled::load(root, layer, "up_proj", &ids)?,
            down: Scaled::load(root, layer, "down_proj", &ids)?,
            original_ids: ids,
        })
    }

    /// R60 pipeline split: the canonical EXL3 K4 experts of `layer` from the target pack (all 512
    /// with `experts` None, else a compact subset of at most 256 for CPU reference checks).
    pub fn canonical(pack: &crate::projection::Pack, layer: usize, experts: Option<&[usize]>) -> Result<Self> {
        let ids: Vec<usize> = experts.map_or_else(|| (0..512).collect(), <[usize]>::to_vec);
        let load = |name: &str| -> Result<Scaled> {
            let p = match experts {
                None => pack.full_scaled_projection(layer, name)?,
                Some(e) => pack.scaled_projection(layer, name, e)?,
            };
            Ok(Scaled { raw: McgProjection::original(p.raw), suh: p.suh, svh: p.svh, scale_provenance: p.scale_provenance })
        };
        Ok(Self { gate: load("gate_proj")?, up: load("up_proj")?, down: load("down_proj")?, original_ids: ids })
    }
    pub fn decoded(&self) -> Result<DecodedBank> {
        ensure!(
            self.original_ids.len() <= 128,
            "compact CPU bank decode bound"
        );
        Ok(DecodedBank {
            gate: self.gate.raw.oracle_weights()?,
            up: self.up.raw.oracle_weights()?,
            down: self.down.raw.oracle_weights()?,
        })
    }
    pub fn provenance(&self) -> Value {
        json!({"original_ids":self.original_ids,"format":if self.gate.raw.halfwords == 64 {"K4/MCG/window15/H128"} else {"K1.75/MCG/window15/H128"},
            "projections":[self.gate.raw.provenance,self.up.raw.provenance,self.down.raw.provenance],
            "scales":[self.gate.scale_provenance,self.up.scale_provenance,self.down.scale_provenance]})
    }
}

fn f16_round(values: &[f64]) -> Vec<u16> {
    values.iter().map(|&v| f16::from_f64(v).to_bits()).collect()
}
fn prepare(p: &Scaled, x: &[f32], routes: &Routes) -> Vec<u16> {
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
pub fn reference(
    bank: &Bank,
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
