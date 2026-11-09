//! R10d one DFlash2 block: learned attention convolution, GQA, residual and MLP.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL, Matrix},
    dflash_attention::{self as attn, D, KH, Norms, QH, Shape},
    dflash_mlp::{self as mlp, Arithmetic, DYN, EPS, GROUP, H, MAX_ROWS, bf, decode, encode},
    projection::sha,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::path::Path;
pub struct Weights {
    pub norm: Vec<u16>,
    pub base: Vec<u16>,
    pub dynamic: Matrix,
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub qk: Norms,
    pub mlp: mlp::Weights,
    pub provenance: serde_json::Value,
}
impl Weights {
    pub fn load(layer: usize) -> Result<Self> {
        ensure!(layer < 5, "DFlash layer bound");
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "DFlash pin"
        );
        let mut vecs = vec![];
        let mut get = |suffix: &str, shape: &[u64]| -> Result<Vec<u16>> {
            let name = format!("layers.{layer}.{suffix}");
            let t = s.tensors.get(&name).context("DFlash vector absent")?;
            ensure!(
                t.dtype == "BF16" && t.shape == shape,
                "DFlash vector geometry"
            );
            let raw = s.read_range(&name, 0, t.bytes())?;
            let v: Vec<_> = raw
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect();
            ensure!(
                v.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "DFlash vector nonfinite"
            );
            vecs.push(json!({"tensor":name,"shape":shape,"sha256":sha(&raw)}));
            Ok(v)
        };
        let norm = get("input_layernorm.weight", &[H as u64])?;
        let base = get("attention_conv.base_kernel", &[2, 2, H as u64])?;
        let dynamic = Matrix::load(&format!(
            "layers.{layer}.attention_conv.kernel_projection.weight"
        ))?;
        let q = Matrix::load(&format!("layers.{layer}.self_attn.q_proj.weight"))?;
        let k = Matrix::load(&format!("layers.{layer}.self_attn.k_proj.weight"))?;
        let v = Matrix::load(&format!("layers.{layer}.self_attn.v_proj.weight"))?;
        let o = Matrix::load(&format!("layers.{layer}.self_attn.o_proj.weight"))?;
        ensure!(
            (dynamic.k, dynamic.n) == (H, DYN)
                && (q.k, q.n) == (H, QH * D)
                && (k.k, k.n) == (H, KH * D)
                && (v.k, v.n) == (H, KH * D)
                && (o.k, o.n) == (QH * D, H),
            "DFlash attention projections"
        );
        let qk = Norms::load(layer)?;
        let mlp = mlp::Weights::load(layer)?;
        let provenance = json!({"layer":layer,"vectors":vecs,"matrices":[dynamic.provenance,q.provenance,k.provenance,v.provenance,o.provenance],"qk":qk.provenance,"mlp":mlp.provenance});
        Ok(Self {
            norm,
            base,
            dynamic,
            q,
            k,
            v,
            o,
            qk,
            mlp,
            provenance,
        })
    }
    pub fn reference(&self, f: &Fixture, arm: Arithmetic) -> Result<Trace> {
        f.validate()?;
        let rows = f.shape.rows;
        let x = decode(&f.x);
        let norm = mlp::rms(&x, &decode(&self.norm), H, EPS);
        let dynamic = self
            .dynamic
            .reference(&encode(&norm))?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let prep = mlp::conv(&norm, &dynamic, &decode(&self.base), rows, H, GROUP, 0, arm);
        let p = encode(&prep);
        let q = self
            .q
            .reference(&p)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let k = self
            .k
            .reference(&p)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let v = self
            .v
            .reference(&p)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let af = attn::Fixture {
            shape: f.shape,
            q: encode(&q),
            k: encode(&k),
            v: encode(&v),
            ck: f.ck.clone(),
            cv: f.cv.clone(),
        };
        let attention = attn::reference(&af, &self.qk)?;
        let o = self
            .o
            .reference(&encode(&attention.out))?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let finish = mlp::conv(&o, &dynamic, &decode(&self.base), rows, H, GROUP, 1, arm);
        let residual = x
            .iter()
            .zip(finish)
            .map(|(&x, f)| bf(x + f))
            .collect::<Vec<_>>();
        let mlp = self.mlp.reference(&encode(&residual), arm)?;
        Ok(Trace {
            norm,
            dynamic,
            prep,
            q,
            k,
            v,
            attention,
            o,
            residual,
            mlp,
        })
    }
}
#[derive(Clone)]
pub struct Fixture {
    pub shape: Shape,
    pub x: Vec<u16>,
    pub ck: Vec<u16>,
    pub cv: Vec<u16>,
}
impl Fixture {
    pub fn validate(&self) -> Result<()> {
        self.shape.validate()?;
        ensure!(
            self.shape.rows <= MAX_ROWS
                && self.x.len() == self.shape.rows * H
                && self.ck.len() == self.shape.ctx * KH * D
                && self.cv.len() == self.ck.len()
                && self
                    .x
                    .iter()
                    .chain(&self.ck)
                    .chain(&self.cv)
                    .all(|&v| bf16::from_bits(v).is_finite()),
            "DFlash layer fixture"
        );
        Ok(())
    }
    pub fn synthetic(shape: Shape, seed: usize, n: &Norms) -> Result<Self> {
        let f = attn::Fixture::synthetic(shape, seed, n)?;
        let x = (0..shape.rows * H)
            .map(|i| {
                let mut h = (i as u64 + 1)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(seed as u64 * 7919);
                h ^= h >> 27;
                h = h.wrapping_mul(0x94d049bb133111eb);
                h ^= h >> 31;
                bf16::from_f32(
                    ((h & 65535) as f32 / 32768. - 1.) * if i % 127 == 0 { 3. } else { 0.5 },
                )
                .to_bits()
            })
            .collect();
        let out = Self {
            shape,
            x,
            ck: f.ck,
            cv: f.cv,
        };
        out.validate()?;
        Ok(out)
    }
}
pub struct Trace {
    pub norm: Vec<f64>,
    pub dynamic: Vec<f64>,
    pub prep: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub attention: attn::Reference,
    pub o: Vec<f64>,
    pub residual: Vec<f64>,
    pub mlp: mlp::Trace,
}
impl Trace {
    pub fn stages(&self) -> Vec<(&str, &[f64])> {
        let mut v: Vec<(&str, &[f64])> = vec![
            ("attn_norm", &self.norm),
            ("attn_dynamic", &self.dynamic),
            ("attn_prepare_conv", &self.prep),
            ("q_projection", &self.q),
            ("k_projection", &self.k),
            ("v_projection", &self.v),
            ("q_rope", &self.attention.q),
            ("k_rope", &self.attention.k),
            ("attention", &self.attention.out),
            ("o_projection", &self.o),
            ("attn_finish_residual", &self.residual),
        ];
        v.extend(self.mlp.stages());
        v
    }
}
