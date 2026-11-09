//! Qwen four-stream hyper-connections, with real affine8 weights and an
//! independent scalar reference. This is a component, not a whole model.
use crate::{dense::Affine8, projection::Pack};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
pub const H: usize = 2560;
pub const HC: usize = 4;
pub const F: usize = H * HC;
pub const R: usize = 320;
pub const MAX_ROWS: usize = 16;
fn num(v: u16) -> f64 {
    bf16::from_bits(v).to_f64()
}
fn bf(v: f64) -> u16 {
    crate::rounding::bf16_from_f64(v).to_bits()
}
fn round(v: f64) -> f64 {
    num(bf(v))
}

pub struct Weights {
    pub down: Affine8,
    pub up: Affine8,
    pub norm: Vec<u16>,
    /// BF16 [HC,F], including the source's load-time /HC fold.
    pub inject: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize, attention: bool) -> Result<Self> {
        ensure!(layer < 48, "HC layer bound");
        let p = format!(
            "language_model.model.layers.{layer}.{}_hyper_connection",
            if attention { "attn" } else { "mlp" }
        );
        let mut down = Affine8::load(pack, &format!("{p}.input_mix_weight_down"), F)?;
        let up = Affine8::load(pack, &format!("{p}.input_mix_weight_up"), R)?;
        let mut provenance = vec![down.provenance.clone(), up.provenance.clone()];
        let mut small = |name: &str, shape: &[u64]| -> Result<Vec<u16>> {
            let (t, b, prov) = pack.tensor(&format!("{p}.{name}.weight"))?;
            ensure!(
                t.dtype == "BF16" && t.shape == shape,
                "HC small weight geometry"
            );
            provenance.push(prov);
            Ok(b.chunks_exact(2)
                .map(|v| u16::from_le_bytes([v[0], v[1]]))
                .collect())
        };
        let norm = small("hc_norm", &[F as u64])?;
        let mut inject = small("block_inject_weight", &[HC as u64, F as u64])?;
        // Match loadHcWeights/scaleTriple: divide affine metadata and injection
        // weights, then round to their stored BF16 dtype. No checkpoint mutation.
        for v in down
            .scales
            .iter_mut()
            .chain(&mut down.biases)
            .chain(&mut inject)
        {
            *v = bf(num(*v) / HC as f64);
        }
        let w = Self {
            down,
            up,
            norm,
            inject,
            provenance: json!({"layer":layer,"attention":attention,"raw_weights":provenance,"load_transform":"down scales+biases and inject /4 in BF16"}),
        };
        w.validate()?;
        Ok(w)
    }
    /// R55: the final hyper-connection mixer as an HC read with zero injection weights
    /// (same /HC load transform on the down projection's affine metadata, no model.norm).
    pub fn load_mixer(pack: &Pack) -> Result<Self> {
        let p = "language_model.model.hyper_connection_mixer";
        let mut down = Affine8::load(pack, &format!("{p}.input_mix_weight_down"), F)?;
        let up = Affine8::load(pack, &format!("{p}.input_mix_weight_up"), R)?;
        let (t, b, prov) = pack.tensor(&format!("{p}.hc_norm.weight"))?;
        ensure!(t.dtype == "BF16" && t.shape == [F as u64], "mixer norm geometry");
        let norm = b.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect();
        for v in down.scales.iter_mut().chain(&mut down.biases) {
            *v = bf(num(*v) / HC as f64);
        }
        let w = Self {
            provenance: serde_json::json!({"mixer":true,"raw_weights":[down.provenance.clone(),up.provenance.clone(),prov],
                "load_transform":"down scales+biases /4 in BF16; injection weights zero (unused)"}),
            down,
            up,
            norm,
            inject: vec![0; HC * F],
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        self.down.validate()?;
        self.up.validate()?;
        ensure!(
            (self.down.k, self.down.n) == (F, R) && (self.up.k, self.up.n) == (R, F),
            "HC projections"
        );
        ensure!(
            self.norm.len() == F && self.inject.len() == HC * F,
            "HC parameters"
        );
        ensure!(
            self.norm
                .iter()
                .chain(&self.inject)
                .all(|&v| bf16::from_bits(v).is_finite()),
            "HC nonfinite weights"
        );
        Ok(())
    }
    pub fn decoded(&self) -> Result<[Vec<f32>; 2]> {
        Ok([self.down.oracle_weights()?, self.up.oracle_weights()?])
    }
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Read {
    pub normalized: Vec<u16>,
    pub down: Vec<u16>,
    pub activated: Vec<u16>,
    pub up: Vec<u16>,
    pub mixed: Vec<u16>,
    pub inject: Vec<u16>,
}
/// CPU-only policy screen. The default preserves the already-qualified native
/// fused helper; StagedBf16 models the source's explicit BF16 sigmoid chain.
#[derive(Clone, Copy, Debug)]
pub enum SigmoidArithmetic {
    NativeFused,
    StagedBf16,
}
impl SigmoidArithmetic {
    fn apply(self, x: f64) -> f64 {
        match self {
            Self::NativeFused => round(1. / (1. + (-x).exp())),
            Self::StagedBf16 => {
                let small = round(1. / round(1. + round(x.abs().exp())));
                if x < 0. { small } else { round(1. - small) }
            }
        }
    }
}
/// The source's >16-row prefill mix rounds each addition to BF16. The
/// qualified native small-row kernel instead sums rounded products in FP32.
/// Both are explicit arithmetic policies; this CPU screen changes no GPU path.
#[derive(Clone, Copy, Debug)]
pub enum MixArithmetic {
    WideSum,
    SourcePrefillBf16,
}
fn mix_products(products: [f64; HC], arithmetic: MixArithmetic) -> f64 {
    let sum = match arithmetic {
        MixArithmetic::WideSum => products.iter().sum(),
        MixArithmetic::SourcePrefillBf16 => products.into_iter().fold(0., |sum, x| round(sum + x)),
    };
    round(round(sum) / HC as f64)
}
/// FP64 scalar reductions, with source-grounded BF16 semantic boundaries.
/// Small discrepancies from GPU reduction order are diagnostics, not rejection
/// of an otherwise promising approximate arithmetic policy.
pub fn reference(w: &Weights, decoded: &[Vec<f32>; 2], x: &[u16]) -> Result<Read> {
    reference_with_sigmoid(w, decoded, x, SigmoidArithmetic::NativeFused)
}
pub fn reference_with_sigmoid(
    w: &Weights,
    decoded: &[Vec<f32>; 2],
    x: &[u16],
    sigmoid: SigmoidArithmetic,
) -> Result<Read> {
    reference_with_policy(w, decoded, x, sigmoid, MixArithmetic::WideSum)
}
pub fn reference_with_policy(
    w: &Weights,
    decoded: &[Vec<f32>; 2],
    x: &[u16],
    sigmoid: SigmoidArithmetic,
    mix: MixArithmetic,
) -> Result<Read> {
    w.validate()?;
    ensure!(
        !x.is_empty() && x.len() % F == 0 && x.len() / F <= MAX_ROWS,
        "HC reference shape"
    );
    ensure!(
        x.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "HC nonfinite input"
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
                normalized[r * F + h * H + j] = bf(round(num(v[j]) * rs) * num(w.norm[h * H + j]));
            }
        }
    }
    let down: Vec<_> = w
        .down
        .reference(&decoded[0], &normalized)?
        .into_iter()
        .map(bf)
        .collect();
    let activated: Vec<_> = down
        .iter()
        .map(|&v| bf(num(v) * sigmoid.apply(num(v))))
        .collect();
    let up: Vec<_> =
        w.up.reference(&decoded[1], &activated)?
            .into_iter()
            .map(bf)
            .collect();
    let mut mixed = vec![0; rows * H];
    let mut inject = vec![0; rows * HC];
    for r in 0..rows {
        for j in 0..H {
            let products = std::array::from_fn(|h| {
                let i = r * F + h * H + j;
                round(sigmoid.apply(num(up[i])) * num(normalized[i]))
            });
            mixed[r * H + j] = bf(mix_products(products, mix));
        }
        for h in 0..HC {
            let raw = round(
                (0..F)
                    .map(|j| num(normalized[r * F + j]) * num(w.inject[h * F + j]))
                    .sum::<f64>(),
            );
            inject[r * HC + h] = bf(2. * sigmoid.apply(raw));
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
pub fn write(x: &[u16], block: &[u16], inject: &[u16]) -> Result<Vec<u16>> {
    ensure!(
        !x.is_empty() && x.len() % F == 0 && x.len() / F <= MAX_ROWS,
        "HC write rows"
    );
    let rows = x.len() / F;
    ensure!(
        block.len() == rows * H && inject.len() == rows * HC,
        "HC write shape"
    );
    ensure!(
        x.iter()
            .chain(block)
            .chain(inject)
            .all(|&v| bf16::from_bits(v).is_finite()),
        "HC write finite"
    );
    Ok((0..x.len())
        .map(|i| {
            let r = i / F;
            let h = i % F / H;
            let j = i % H;
            bf(num(x[i]) + round(num(block[r * H + j]) * num(inject[r * HC + h])))
        })
        .collect())
}

#[cfg(test)]
mod mix_tests {
    use super::*;
    #[test]
    fn source_prefill_rounds_each_addition() {
        // Two half-ulp increments each tie back to1 separately, but their
        // unrounded sum reaches the next BF16 value. Mean divides by4 exactly.
        let p = [1., 1. / 256., 1. / 256., 0.];
        assert_eq!(mix_products(p, MixArithmetic::SourcePrefillBf16), 0.25);
        assert_eq!(mix_products(p, MixArithmetic::WideSum), 0.251953125);
        assert_eq!(
            mix_products([1., -1., 2., -2.], MixArithmetic::SourcePrefillBf16),
            0.
        );
    }
}
