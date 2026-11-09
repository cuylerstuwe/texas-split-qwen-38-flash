//! Lowbit shared expert: real affine4 weights and a scalar FP64 validation oracle.
//! This module does not generate tokens or participate in timed GPU arithmetic.
use crate::{affine4::Affine4, bf16_linear::Matrix, projection::sha, weights::Shard};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Serialize;
use serde_json::{Value, json};
use std::path::{Component, Path};
pub const H: usize = 2560;
pub const I: usize = 640;
pub const MAX_ROWS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arithmetic {
    StagedBf16,
    FusedF32,
}
pub fn number(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
pub fn bits(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
fn round(x: f64) -> f64 {
    number(bits(x))
}
// Same semantic BF16 boundaries as target_mlp. The exp is an independent CPU
// oracle, not Sushi's Metal-generated 65536-entry sigmoid lookup table.
pub fn sigmoid(x: f64, arm: Arithmetic) -> f64 {
    match arm {
        Arithmetic::StagedBf16 => {
            let small = round(1. / round(1. + round(x.abs().exp())));
            if x < 0. { small } else { round(1. - small) }
        }
        Arithmetic::FusedF32 => 1. / (1. + (-x).exp()),
    }
}
pub fn activate(gate: &[u16], up: &[u16], arm: Arithmetic) -> Result<Vec<u16>> {
    ensure!(
        !gate.is_empty() && gate.len() == up.len(),
        "shared activation shape"
    );
    ensure!(
        gate.iter().chain(up).all(|&v| number(v).is_finite()),
        "shared activation finite"
    );
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&g, &u)| {
            let silu = number(g) * sigmoid(number(g), arm);
            bits(
                (if arm == Arithmetic::StagedBf16 {
                    round(silu)
                } else {
                    silu
                }) * number(u),
            )
        })
        .collect())
}
pub struct Weights {
    pub projections: [Affine4; 3],
    pub scalar_gate: Matrix,
    pub layer: usize,
}
impl Weights {
    pub fn load(root: &Path, layer: usize) -> Result<Self> {
        ensure!(layer < 48, "shared layer bound");
        let p = format!("language_model.model.layers.{layer}.mlp");
        let projections = [
            Affine4::load(root, &format!("{p}.shared_expert.gate_proj"), H)?,
            Affine4::load(root, &format!("{p}.shared_expert.up_proj"), H)?,
            Affine4::load(root, &format!("{p}.shared_expert.down_proj"), I)?,
        ];
        ensure!(
            projections[0].n == I && projections[1].n == I && projections[2].n == H,
            "shared projection geometry"
        );
        let index_bytes = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&index_bytes) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
            "shared lowbit index pin"
        );
        let index: Value = serde_json::from_slice(&index_bytes)?;
        let name = format!("{p}.shared_expert_gate.weight");
        let file = index["weight_map"][&name]
            .as_str()
            .context("shared scalar index")?;
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "shared shard path"
        );
        let shard = Shard::open(&root.join(file))?;
        let t = shard.tensors.get(&name).context("shared scalar tensor")?;
        ensure!(
            t.dtype == "BF16" && t.shape == [1, H as u64] && t.bytes() == (H * 2) as u64,
            "shared scalar geometry"
        );
        let raw = shard.read_range(&name, 0, t.bytes())?;
        let scalar_gate = Matrix {
            k: H,
            n: 1,
            values: raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            provenance: json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,
                "payload_sha256":sha(&raw),"index_sha256":sha(&index_bytes)}),
        };
        scalar_gate.validate()?;
        Ok(Self {
            projections,
            scalar_gate,
            layer,
        })
    }
    pub fn decoded(&self) -> Result<[Vec<f32>; 3]> {
        Ok([
            self.projections[0].oracle_weights()?,
            self.projections[1].oracle_weights()?,
            self.projections[2].oracle_weights()?,
        ])
    }
    pub fn provenance(&self) -> Value {
        json!({"layer":self.layer,"projections":self.projections.iter().map(|w|&w.provenance).collect::<Vec<_>>(),"scalar_gate":self.scalar_gate.provenance})
    }
}
#[derive(Clone)]
pub struct Stages {
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub activation: Vec<u16>,
    pub down: Vec<u16>,
    pub gate_logits: Vec<u16>,
    pub gate_sigmoid: Vec<u16>,
    pub output: Vec<u16>,
}
impl Stages {
    pub fn fields(&self) -> [(&'static str, &[u16]); 7] {
        [
            ("gate", &self.gate),
            ("up", &self.up),
            ("activation", &self.activation),
            ("down", &self.down),
            ("gate_logits", &self.gate_logits),
            ("gate_sigmoid", &self.gate_sigmoid),
            ("output", &self.output),
        ]
    }
}
pub struct Reference {
    pub stages: Stages,
    pub gate_raw: Vec<f64>,
    pub up_raw: Vec<f64>,
    pub down_raw: Vec<f64>,
    pub scalar_raw: Vec<f64>,
}
pub fn reference(
    w: &Weights,
    decoded: &[Vec<f32>; 3],
    input: &[u16],
    arm: Arithmetic,
) -> Result<Reference> {
    ensure!(
        !input.is_empty() && input.len() % H == 0 && input.len() / H <= MAX_ROWS,
        "shared oracle rows"
    );
    let gate_raw = w.projections[0].reference(&decoded[0], input)?;
    let up_raw = w.projections[1].reference(&decoded[1], input)?;
    let gate: Vec<_> = gate_raw.iter().map(|&v| bits(v)).collect();
    let up: Vec<_> = up_raw.iter().map(|&v| bits(v)).collect();
    let activation = activate(&gate, &up, arm)?;
    let down_raw = w.projections[2].reference(&decoded[2], &activation)?;
    let down: Vec<_> = down_raw.iter().map(|&v| bits(v)).collect();
    let scalar_raw = w.scalar_gate.reference(input)?;
    let gate_logits: Vec<_> = scalar_raw.iter().map(|&v| bits(v)).collect();
    let gate_sigmoid: Vec<_> = gate_logits
        .iter()
        .map(|&v| bits(sigmoid(number(v), arm)))
        .collect();
    // Fused arm also keeps the scalar sigmoid in FP32 until the final product.
    let output: Vec<_> = down
        .iter()
        .enumerate()
        .map(|(i, &v)| bits(number(v) * sigmoid(number(gate_logits[i / H]), arm)))
        .collect();
    ensure!(
        output.iter().all(|&v| number(v).is_finite()),
        "shared oracle output finite"
    );
    Ok(Reference {
        stages: Stages {
            gate,
            up,
            activation,
            down,
            gate_logits,
            gate_sigmoid,
            output,
        },
        gate_raw,
        up_raw,
        down_raw,
        scalar_raw,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staged_sigmoid_and_activation_saturation() {
        for arm in [Arithmetic::StagedBf16, Arithmetic::FusedF32] {
            assert_eq!(sigmoid(0., arm), 0.5);
            assert_eq!(sigmoid(-1000., arm), 0.);
            assert_eq!(sigmoid(1000., arm), 1.);
            let a = activate(
                &[bits(0.), bits(1000.), bits(-1000.)],
                &[bits(3.), bits(2.), bits(2.)],
                arm,
            )
            .unwrap();
            assert_eq!(
                a.iter().map(|&v| number(v)).collect::<Vec<_>>(),
                vec![0., 2000., 0.]
            );
        }
        assert!(activate(&[bits(1.)], &[], Arithmetic::StagedBf16).is_err());
    }
}
