//! DFlash2 MLP sublayer: independent CPU oracle with explicit BF16 boundaries.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL, Matrix},
    projection::sha,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::path::Path;

pub const H: usize = 2560;
pub const F: usize = 7680;
pub const GROUP: usize = 16;
pub const DYN: usize = 4 * H / GROUP;
pub const MAX_ROWS: usize = 16;
pub const EPS: f64 = 1e-6;
pub fn bf(x: f64) -> f64 {
    crate::rounding::bf16_from_f64(x).to_f64()
}
pub fn decode(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&v| bf16::from_bits(v).to_f64()).collect()
}
pub fn encode(v: &[f64]) -> Vec<u16> {
    v.iter()
        .map(|&v| crate::rounding::bf16_from_f64(v).to_bits())
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub enum Arithmetic {
    Staged,
    Fused,
}
impl Arithmetic {
    pub fn name(self) -> &'static str {
        match self {
            Self::Staged => "bf16_staged",
            Self::Fused => "fp32_fused",
        }
    }
    pub fn code(self) -> u32 {
        match self {
            Self::Staged => 0,
            Self::Fused => 1,
        }
    }
}
pub struct Weights {
    pub norm: Vec<u16>,
    pub base: Vec<u16>,
    pub dynamic: Matrix,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    pub provenance: serde_json::Value,
}
impl Weights {
    pub fn load(layer: usize) -> Result<Self> {
        ensure!(layer < 5, "DFlash2 layer bound");
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "DFlash2 pin"
        );
        let mut vectors = vec![];
        let mut get = |suffix: &str, shape: &[u64]| -> Result<Vec<u16>> {
            let name = format!("layers.{layer}.{suffix}");
            let t = s.tensors.get(&name).context("DFlash2 vector absent")?;
            ensure!(
                t.dtype == "BF16" && t.shape == shape,
                "DFlash2 vector geometry"
            );
            let b = s.read_range(&name, 0, t.bytes())?;
            vectors.push(json!({"tensor":name,"shape":shape,"bytes":b.len(),"sha256":sha(&b)}));
            let v: Vec<_> = b
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect();
            ensure!(
                v.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "nonfinite weight"
            );
            Ok(v)
        };
        let norm = get("post_attention_layernorm.weight", &[H as u64])?;
        let base = get("mlp_conv.base_kernel", &[2, 2, H as u64])?;
        let dynamic = Matrix::load(&format!("layers.{layer}.mlp_conv.kernel_projection.weight"))?;
        let gate = Matrix::load(&format!("layers.{layer}.mlp.gate_proj.weight"))?;
        let up = Matrix::load(&format!("layers.{layer}.mlp.up_proj.weight"))?;
        let down = Matrix::load(&format!("layers.{layer}.mlp.down_proj.weight"))?;
        ensure!(
            (dynamic.k, dynamic.n) == (H, DYN)
                && (gate.k, gate.n) == (H, F)
                && (up.k, up.n) == (H, F)
                && (down.k, down.n) == (F, H),
            "DFlash2 MLP geometry"
        );
        let provenance = json!({"layer":layer,"header_sha256":HEADER_SHA,"vectors":vectors,
            "matrices":[dynamic.provenance,gate.provenance,up.provenance,down.provenance]});
        Ok(Self {
            norm,
            base,
            dynamic,
            gate,
            up,
            down,
            provenance,
        })
    }
    pub fn reference(&self, input: &[u16], arm: Arithmetic) -> Result<Trace> {
        ensure!(
            !input.is_empty() && input.len() % H == 0 && input.len() / H <= MAX_ROWS,
            "DFlash2 MLP input"
        );
        let rows = input.len() / H;
        let x = decode(input);
        ensure!(x.iter().all(|x| x.is_finite()), "DFlash2 finite input");
        let norm = rms(&x, &decode(&self.norm), H, EPS);
        let dyns = self
            .dynamic
            .reference(&encode(&norm))?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let base = decode(&self.base);
        let prep = conv(&norm, &dyns, &base, rows, H, GROUP, 0, arm);
        let prep_b = encode(&prep);
        let gate = self
            .gate
            .reference(&prep_b)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let up = self
            .up
            .reference(&prep_b)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let act = swiglu(&gate, &up, arm);
        let down = self
            .down
            .reference(&encode(&act))?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let finish = conv(&down, &dyns, &base, rows, H, GROUP, 1, arm);
        let out = x.iter().zip(finish).map(|(&x, f)| bf(x + f)).collect();
        Ok(Trace {
            norm,
            dyns,
            prep,
            gate,
            up,
            act,
            down,
            out,
        })
    }
}
pub struct Trace {
    pub norm: Vec<f64>,
    pub dyns: Vec<f64>,
    pub prep: Vec<f64>,
    pub gate: Vec<f64>,
    pub up: Vec<f64>,
    pub act: Vec<f64>,
    pub down: Vec<f64>,
    pub out: Vec<f64>,
}
impl Trace {
    pub fn stages(&self) -> [(&str, &[f64]); 8] {
        [
            ("norm", &self.norm),
            ("dynamic", &self.dyns),
            ("prepare_conv", &self.prep),
            ("gate", &self.gate),
            ("up", &self.up),
            ("swiglu", &self.act),
            ("down", &self.down),
            ("finish_residual", &self.out),
        ]
    }
}
pub fn rms(x: &[f64], weight: &[f64], h: usize, eps: f64) -> Vec<f64> {
    x.chunks_exact(h)
        .flat_map(|row| {
            let inv = (row.iter().map(|v| v * v).sum::<f64>() / h as f64 + eps)
                .sqrt()
                .recip();
            row.iter().zip(weight).map(move |(&v, &w)| bf(v * inv * w))
        })
        .collect()
}
pub fn conv(
    x: &[f64],
    dyns: &[f64],
    base: &[f64],
    rows: usize,
    h: usize,
    group: usize,
    set: usize,
    arm: Arithmetic,
) -> Vec<f64> {
    let groups = h / group;
    let mut y = vec![0.; x.len()];
    for r in 0..rows {
        for c in 0..h {
            let mut v = 0.;
            for tap in 0..2 {
                let z = if r >= tap { x[(r - tap) * h + c] } else { 0. };
                for a in [
                    base[(2 * set + tap) * h + c],
                    dyns[(r * 4 + set * 2 + tap) * groups + c / group],
                ] {
                    v = match arm {
                        Arithmetic::Staged => bf(v + bf(a * z)),
                        Arithmetic::Fused => v + a * z,
                    };
                }
            }
            y[r * h + c] = bf(v);
        }
    }
    y
}
pub fn swiglu(gate: &[f64], up: &[f64], arm: Arithmetic) -> Vec<f64> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let s = 1. / (1. + (-g).exp());
            bf(match arm {
                Arithmetic::Staged => bf(g * bf(s)) * u,
                Arithmetic::Fused => g * s * u,
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conv_axes_zero_padding_and_prefix_are_distinct() {
        // h2/group1; distinct base[set,tap,channel] and row-local dynamic values.
        let x = vec![1., 2., 3., 4.];
        let base = vec![2., 3., 5., 7., 11., 13., 17., 19.];
        let d = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        assert_eq!(
            conv(&x, &d, &base, 2, 2, 1, 0, Arithmetic::Staged),
            [3., 10., 49., 90.]
        );
        assert_eq!(
            conv(&x, &d, &base, 2, 2, 1, 1, Arithmetic::Staged),
            [16., 38., 104., 178.]
        );
        let mut changed = x.clone();
        changed[2] = 15.;
        assert_eq!(
            &conv(&x, &d, &base, 2, 2, 1, 1, Arithmetic::Staged)[..2],
            &conv(&changed, &d, &base, 2, 2, 1, 1, Arithmetic::Staged)[..2]
        );
    }
}
