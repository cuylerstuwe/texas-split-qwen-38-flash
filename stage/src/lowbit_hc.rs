//! Lowbit hyper-connection weights and an independent FP64 validation oracle.
//! CPU materialization is only for component checking, never timed inference.
use crate::{affine4::Affine4, hc, projection::sha, weights::Shard};
use anyhow::{Context, Result, ensure};
use half::bf16;
pub use hc::{F, H, HC, MAX_ROWS, R, Read};
use serde_json::{Value, json};
use std::path::{Component, Path};
fn num(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
fn bits(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
fn round(x: f64) -> f64 {
    num(bits(x))
}
fn sigmoid(x: f64) -> f64 {
    round(1. / (1. + (-x).exp()))
}

/// Bounded BF16 side tensors from the pinned lowbit pack. No model-wide read.
pub(crate) fn small(root: &Path, name: &str, shape: &[u64]) -> Result<(Vec<u16>, Value)> {
    let ib = std::fs::read(root.join("model.safetensors.index.json"))?;
    ensure!(
        sha(&ib) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
        "lowbit small index pin"
    );
    let index: Value = serde_json::from_slice(&ib)?;
    let file = index["weight_map"][name]
        .as_str()
        .context("small tensor index")?;
    let mut parts = Path::new(file).components();
    ensure!(
        matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
        "small shard path"
    );
    let shard = Shard::open(&root.join(file))?;
    let t = shard.tensors.get(name).context("small tensor absent")?;
    ensure!(
        t.dtype == "BF16" && t.shape == shape && t.bytes() <= 2 << 20,
        "small tensor shape/bound"
    );
    let raw = shard.read_range(name, 0, t.bytes())?;
    let values: Vec<u16> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "small tensor finite"
    );
    Ok((
        values,
        json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,"payload_sha256":sha(&raw),"index_sha256":sha(&ib)}),
    ))
}
pub struct Weights {
    pub down: Affine4,
    pub up: Affine4,
    pub norm: Vec<u16>,
    pub inject: Vec<u16>,
    pub provenance: Value,
}
impl Weights {
    pub fn load(root: &Path, layer: usize, attention: bool) -> Result<Self> {
        ensure!(layer < 48, "lowbit HC layer");
        let p = format!(
            "language_model.model.layers.{layer}.{}_hyper_connection",
            if attention { "attn" } else { "mlp" }
        );
        let mut down = Affine4::load(root, &format!("{p}.input_mix_weight_down"), F)?;
        let up = Affine4::load(root, &format!("{p}.input_mix_weight_up"), R)?;
        let (norm, np) = small(root, &format!("{p}.hc_norm.weight"), &[F as u64])?;
        let (mut inject, ip) = small(
            root,
            &format!("{p}.block_inject_weight.weight"),
            &[HC as u64, F as u64],
        )?;
        for v in down
            .scales
            .iter_mut()
            .chain(&mut down.biases)
            .chain(&mut inject)
        {
            *v = bits(num(*v) / HC as f64);
        }
        let provenance = json!({"layer":layer,"attention":attention,"down":down.provenance,"up":up.provenance,"norm":np,"inject":ip,"load_transform":"down scales+biases and injection divided by4, rounded BF16; canonical bytes unchanged"});
        let w = Self {
            down,
            up,
            norm,
            inject,
            provenance,
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        self.down.validate()?;
        self.up.validate()?;
        ensure!(
            (self.down.k, self.down.n) == (F, R) && (self.up.k, self.up.n) == (R, F),
            "lowbit HC projection geometry"
        );
        ensure!(
            self.norm.len() == F
                && self.inject.len() == HC * F
                && self
                    .norm
                    .iter()
                    .chain(&self.inject)
                    .all(|&v| bf16::from_bits(v).is_finite()),
            "lowbit HC side tensors"
        );
        Ok(())
    }
    pub fn decoded(&self) -> Result<[Vec<f32>; 2]> {
        Ok([self.down.oracle_weights()?, self.up.oracle_weights()?])
    }
}
pub fn reference(w: &Weights, decoded: &[Vec<f32>; 2], x: &[u16]) -> Result<Read> {
    w.validate()?;
    ensure!(
        !x.is_empty()
            && x.len() % F == 0
            && x.len() / F <= MAX_ROWS
            && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "lowbit HC oracle input"
    );
    let rows = x.len() / F;
    let mut normalized = vec![0; x.len()];
    for (r, row) in x.chunks_exact(F).enumerate() {
        for h in 0..HC {
            let v = &row[h * H..(h + 1) * H];
            let rs = (v.iter().map(|&v| num(v).powi(2)).sum::<f64>() / H as f64 + 1e-6)
                .sqrt()
                .recip();
            for j in 0..H {
                normalized[r * F + h * H + j] =
                    bits(round(num(v[j]) * rs) * num(w.norm[h * H + j]));
            }
        }
    }
    let down: Vec<_> = w
        .down
        .reference(&decoded[0], &normalized)?
        .into_iter()
        .map(bits)
        .collect();
    let activated: Vec<_> = down
        .iter()
        .map(|&v| bits(num(v) * sigmoid(num(v))))
        .collect();
    let up: Vec<_> =
        w.up.reference(&decoded[1], &activated)?
            .into_iter()
            .map(bits)
            .collect();
    let mut mixed = vec![0; rows * H];
    let mut inject = vec![0; rows * HC];
    for r in 0..rows {
        for j in 0..H {
            let sum = (0..HC)
                .map(|h| {
                    let i = r * F + h * H + j;
                    round(sigmoid(num(up[i])) * num(normalized[i]))
                })
                .sum();
            mixed[r * H + j] = bits(round(sum) / HC as f64);
        }
        for h in 0..HC {
            let raw = round(
                (0..F)
                    .map(|j| num(normalized[r * F + j]) * num(w.inject[h * F + j]))
                    .sum(),
            );
            inject[r * HC + h] = bits(2. * sigmoid(raw));
        }
    }
    Ok(Read {
        normalized,
        down,
        activated,
        up,
        mixed,
        inject,
    })
}
