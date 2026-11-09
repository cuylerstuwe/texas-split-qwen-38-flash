//! Original lowbit token tables and final HC mixer; independent CPU oracles.
//! CPU decoding here is validation only. Inference retains packed GPU weights.
pub use crate::embedding::VOCAB;
use crate::{
    affine4::Affine4,
    hc::{F, H, HC, MAX_ROWS, R},
    lowbit_hc::small,
    projection::sha,
    rounding::bf16_from_f64,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Component, Path};
pub const GROUP: usize = 64;

/// The only larger-than-component matrices admitted here are the two pinned
/// [248320,2560] affine4/group64 vocabulary tables. Ordinary Affine4 limits stay.
pub struct Vocab {
    pub(crate) codes: Vec<u8>,
    pub(crate) scales: Vec<u16>,
    pub(crate) biases: Vec<u16>,
    pub provenance: Value,
}
impl Vocab {
    pub fn load(root: &Path, embedding: bool) -> Result<Self> {
        let config = std::fs::read(root.join("config.json"))?;
        let ib = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&config) == "fd9d242648d6a5a1ef130f8f31234bc182b0c39ea175241c0e7c22a65d241454",
            "lowbit IO config pin"
        );
        ensure!(
            sha(&ib) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
            "lowbit IO index pin"
        );
        let index: Value = serde_json::from_slice(&ib)?;
        let prefix = if embedding {
            "language_model.model.embed_tokens"
        } else {
            "language_model.lm_head"
        };
        let read = |suffix: &str, dtype: &str, columns: usize| -> Result<(Vec<u8>, Value)> {
            let name = format!("{prefix}.{suffix}");
            let file = index["weight_map"][&name]
                .as_str()
                .context("IO tensor index")?;
            let mut parts = Path::new(file).components();
            ensure!(
                matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
                "IO shard path"
            );
            let shard = Shard::open(&root.join(file))?;
            let tensor = shard.tensors.get(&name).context("IO tensor missing")?;
            ensure!(
                tensor.dtype == dtype && tensor.shape == [VOCAB as u64, columns as u64],
                "IO exact vocabulary shape"
            );
            // Keep the safetensors reader's 64 MiB range guard intact.
            let mut data = Vec::with_capacity(tensor.bytes() as usize);
            for offset in (0..tensor.bytes()).step_by(32 << 20) {
                data.extend(shard.read_range(
                    &name,
                    offset,
                    (tensor.bytes() - offset).min(32 << 20),
                )?);
            }
            let p = json!({"tensor":name,"file":file,"header_sha256":shard.header_sha256,
                "payload_sha256":sha(&data),"bytes":data.len(),"config_sha256":sha(&config),"index_sha256":sha(&ib)});
            Ok((data, p))
        };
        let (codes, cp) = read("weight", "U32", H / 8)?;
        let (s, sp) = read("scales", "BF16", H / GROUP)?;
        let (b, bp) = read("biases", "BF16", H / GROUP)?;
        let words = |v: Vec<u8>| {
            v.chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect()
        };
        let result = Self {
            codes,
            scales: words(s),
            biases: words(b),
            provenance: json!({"embedding":embedding,"tensors":[cp,sp,bp],"format":"original affine4/group64"}),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.codes.len() == VOCAB * H / 2
                && self.scales.len() == VOCAB * H / GROUP
                && self.biases.len() == self.scales.len(),
            "vocabulary table lengths"
        );
        ensure!(
            self.scales
                .iter()
                .chain(&self.biases)
                .all(|&v| bf16::from_bits(v).is_finite()),
            "vocabulary metadata finite"
        );
        Ok(())
    }
    pub fn bytes(&self) -> usize {
        self.codes.len() + 2 * (self.scales.len() + self.biases.len())
    }
    /// Bounded CPU tile. No whole expanded vocabulary table is created.
    pub fn tile(&self, first: usize, n: usize) -> Result<Affine4> {
        ensure!(
            n > 0 && n <= 2048 && first.checked_add(n).is_some_and(|e| e <= VOCAB),
            "vocabulary tile bounds"
        );
        let a = Affine4 {
            k: H,
            n,
            group: GROUP,
            codes: self.codes[first * H / 2..(first + n) * H / 2].to_vec(),
            scales: self.scales[first * H / GROUP..(first + n) * H / GROUP].to_vec(),
            biases: self.biases[first * H / GROUP..(first + n) * H / GROUP].to_vec(),
            provenance: json!({"first":first,"rows":n,"table":self.provenance}),
        };
        a.validate()?;
        Ok(a)
    }
    pub fn embedding(&self, tokens: &[u32]) -> Result<Vec<u16>> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= MAX_ROWS
                && tokens.iter().all(|&t| t < (VOCAB as u32)),
            "IO token bounds"
        );
        let mut out = Vec::with_capacity(tokens.len() * F);
        for &token in tokens {
            let weights = self.tile(token as usize, 1)?.oracle_weights()?;
            let row: Vec<_> = weights
                .into_iter()
                .map(|v| bf16::from_f32(v).to_bits())
                .collect();
            for _ in 0..HC {
                out.extend_from_slice(&row);
            }
        }
        Ok(out)
    }
    pub fn reference(&self, x: &[u16]) -> Result<(Vec<f64>, Vec<u16>)> {
        ensure!(
            !x.is_empty()
                && x.len() % H == 0
                && x.len() / H <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "IO head input"
        );
        let rows = x.len() / H;
        let mut raw = vec![0.; rows * VOCAB];
        for first in (0..VOCAB).step_by(2048) {
            let n = (VOCAB - first).min(2048);
            let tile = self.tile(first, n)?;
            let projected = tile.reference(&tile.oracle_weights()?, x)?;
            for r in 0..rows {
                raw[r * VOCAB + first..r * VOCAB + first + n]
                    .copy_from_slice(&projected[r * n..(r + 1) * n]);
            }
        }
        let rounded = raw.iter().map(|&v| bf16_from_f64(v).to_bits()).collect();
        Ok((raw, rounded))
    }
}

pub struct Mixer {
    pub down: Affine4,
    pub up: Affine4,
    pub norm: Vec<u16>,
    pub provenance: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mixed {
    pub normalized: Vec<u16>,
    pub down: Vec<u16>,
    pub activated: Vec<u16>,
    pub up: Vec<u16>,
    pub mixed: Vec<u16>,
}
impl Mixer {
    pub fn load(root: &Path) -> Result<Self> {
        let p = "language_model.model.hyper_connection_mixer";
        let mut down = Affine4::load(root, &format!("{p}.input_mix_weight_down"), F)?;
        let up = Affine4::load(root, &format!("{p}.input_mix_weight_up"), R)?;
        let (norm, np) = small(root, &format!("{p}.hc_norm.weight"), &[F as u64])?;
        for v in down.scales.iter_mut().chain(&mut down.biases) {
            *v = bf16_from_f64(bf16::from_bits(*v).to_f64() / HC as f64).to_bits();
        }
        let w = Self {
            provenance: json!({"down":down.provenance,"up":up.provenance,"norm":np,
            "load_transform":"divide down scales/biases by4 with BF16 rounding, matching Sushi; no injection weights; no additional model.norm"}),
            down,
            up,
            norm,
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        self.down.validate()?;
        self.up.validate()?;
        ensure!(
            (self.down.k, self.down.n) == (F, R)
                && (self.up.k, self.up.n) == (R, F)
                && self.norm.len() == F
                && self.norm.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "final mixer geometry"
        );
        Ok(())
    }
    pub fn reference(&self, x: &[u16]) -> Result<Mixed> {
        self.validate()?;
        ensure!(
            !x.is_empty()
                && x.len() % F == 0
                && x.len() / F <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "final mixer input"
        );
        let num = |v: u16| bf16::from_bits(v).to_f64();
        let bits = |v: f64| bf16_from_f64(v).to_bits();
        let rb = |v: f64| num(bits(v));
        let sig = |v: f64| rb(1. / (1. + (-v).exp()));
        let mut normalized = vec![0; x.len()];
        for (r, row) in x.chunks_exact(F).enumerate() {
            for h in 0..HC {
                let v = &row[h * H..(h + 1) * H];
                let rs = (v.iter().map(|&a| num(a).powi(2)).sum::<f64>() / H as f64 + 1e-6)
                    .sqrt()
                    .recip();
                for j in 0..H {
                    normalized[r * F + h * H + j] =
                        bits(rb(num(v[j]) * rs) * num(self.norm[h * H + j]));
                }
            }
        }
        let down: Vec<_> = self
            .down
            .reference(&self.down.oracle_weights()?, &normalized)?
            .into_iter()
            .map(bits)
            .collect();
        let activated: Vec<_> = down.iter().map(|&v| bits(num(v) * sig(num(v)))).collect();
        let up: Vec<_> = self
            .up
            .reference(&self.up.oracle_weights()?, &activated)?
            .into_iter()
            .map(bits)
            .collect();
        let mut mixed = vec![0; x.len() / HC];
        for r in 0..x.len() / F {
            for j in 0..H {
                let sum = (0..HC)
                    .map(|h| {
                        let i = r * F + h * H + j;
                        rb(sig(num(up[i])) * num(normalized[i]))
                    })
                    .sum::<f64>();
                mixed[r * H + j] = bits(rb(sum) / HC as f64);
            }
        }
        Ok(Mixed {
            normalized,
            down,
            activated,
            up,
            mixed,
        })
    }
}
