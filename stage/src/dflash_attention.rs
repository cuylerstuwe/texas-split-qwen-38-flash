//! DFlash2 attention geometry and independent two-pass FP64 attention oracle.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL},
    dflash_mlp::{bf, decode, encode},
    projection::sha,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::path::Path;
pub const D: usize = 128;
pub const QH: usize = 32;
pub const KH: usize = 8;
pub const MAX_ROWS: usize = 16;
pub const MAX_CTX: usize = 2051;
pub const MAX_SPLITS: usize = 16;
pub fn frequencies() -> [f32; 64] {
    std::array::from_fn(|i| 10000f64.powf(-(i as f64) / 64.) as f32)
}
pub struct Norms {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Norms {
    pub fn load(layer: usize) -> Result<Self> {
        ensure!(layer < 5, "DFlash attention layer bound");
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "DFlash2 pin"
        );
        let mut records = vec![];
        let mut get = |kind: &str| -> Result<Vec<u16>> {
            let n = format!("layers.{layer}.self_attn.{kind}_norm.weight");
            let t = s.tensors.get(&n).context("QK norm absent")?;
            ensure!(t.dtype == "BF16" && t.shape == [128], "QK norm geometry");
            let raw = s.read_range(&n, 0, t.bytes())?;
            records.push(json!({"tensor":n,"sha256":sha(&raw)}));
            let v = raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            ensure!(
                v.iter().all(|&x| bf16::from_bits(x).is_finite()),
                "QK weight finite"
            );
            Ok(v)
        };
        let q = get("q")?;
        let k = get("k")?;
        Ok(Self {
            q,
            k,
            provenance: json!({"layer":layer,"header_sha256":HEADER_SHA,"tensors":records}),
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub rows: usize,
    pub ctx: usize,
    pub base: usize,
    pub anchor: usize,
    pub window: usize,
}
impl Shape {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows)
                && self.ctx <= MAX_CTX
                && (1..=2048).contains(&self.window)
                && self.base <= 262144
                && self.base.checked_add(self.ctx) == Some(self.anchor)
                && self
                    .anchor
                    .checked_add(self.rows)
                    .is_some_and(|n| n <= 262144),
            "attention shape/position"
        );
        Ok(())
    }
    pub fn control(self) -> [u32; 5] {
        [
            self.rows as u32,
            self.ctx as u32,
            self.base as u32,
            self.anchor as u32,
            self.window as u32,
        ]
    }
    pub fn allowed(self, r: usize, key: usize) -> bool {
        let pos = if key < self.ctx {
            self.base + key
        } else {
            self.anchor + key - self.ctx
        };
        (self.anchor + r).abs_diff(pos) < self.window
    }
}
#[derive(Clone)]
pub struct Fixture {
    pub shape: Shape,
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub ck: Vec<u16>,
    pub cv: Vec<u16>,
}
impl Fixture {
    pub fn validate(&self) -> Result<()> {
        self.shape.validate()?;
        for (v, n) in [
            (&self.q, self.shape.rows * QH * D),
            (&self.k, self.shape.rows * KH * D),
            (&self.v, self.shape.rows * KH * D),
            (&self.ck, self.shape.ctx * KH * D),
            (&self.cv, self.shape.ctx * KH * D),
        ] {
            ensure!(
                v.len() == n && v.iter().all(|&x| bf16::from_bits(x).is_finite()),
                "attention fixture shape/finite"
            );
        }
        Ok(())
    }
    pub fn synthetic(shape: Shape, seed: usize, n: &Norms) -> Result<Self> {
        shape.validate()?;
        fn data(count: usize, seed: usize) -> Vec<u16> {
            (0..count)
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
                .collect()
        }
        let rawck = data(shape.ctx * KH * D, seed + 7);
        let ck = encode(&norm_rope(&rawck, &n.k, KH, shape.base));
        let f = Self {
            shape,
            q: data(shape.rows * QH * D, seed + 1),
            k: data(shape.rows * KH * D, seed + 3),
            v: data(shape.rows * KH * D, seed + 5),
            ck,
            cv: data(shape.ctx * KH * D, seed + 9),
        };
        f.validate()?;
        Ok(f)
    }
    pub fn uniform_axis() -> Self {
        let shape = Shape {
            rows: 2,
            ctx: 3,
            base: 10,
            anchor: 13,
            window: 3,
        };
        let expand =
            |v: &[f64]| encode(&v.iter().flat_map(|&x| vec![x; KH * D]).collect::<Vec<_>>());
        Self {
            shape,
            q: vec![0; 2 * QH * D],
            k: vec![0; 2 * KH * D],
            v: expand(&[8., 16.]),
            ck: vec![0; 3 * KH * D],
            cv: expand(&[1., 2., 4.]),
        }
    }
}
// Deliberate arithmetic convention: FP64-generated frequencies rounded FP32,
// FP32 angle, scalar FP64 trig and products, then BF16. GPU uses FP32 trig/ops.
pub fn norm_rope(input: &[u16], w: &[u16], heads: usize, base: usize) -> Vec<f64> {
    let x = decode(input);
    let weight = decode(w);
    let freq = frequencies();
    let mut y = vec![0.; input.len()];
    for (g, row) in x.chunks_exact(D).enumerate() {
        let inv = (row.iter().map(|v| v * v).sum::<f64>() / D as f64 + 1e-6)
            .sqrt()
            .recip();
        let norm: Vec<_> = row
            .iter()
            .zip(&weight)
            .map(|(&v, &w)| bf(v * inv * w))
            .collect();
        let pos = base + g / heads;
        for i in 0..64 {
            let angle = ((pos as f32) * freq[i]) as f64;
            let (s, c) = angle.sin_cos();
            y[g * D + i] = bf(norm[i] * c - norm[i + 64] * s);
            y[g * D + i + 64] = bf(norm[i] * s + norm[i + 64] * c);
        }
    }
    y
}
pub struct Reference {
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub out: Vec<f64>,
}
pub fn reference(f: &Fixture, n: &Norms) -> Result<Reference> {
    f.validate()?;
    let q = norm_rope(&f.q, &n.q, QH, f.shape.anchor);
    let k = norm_rope(&f.k, &n.k, KH, f.shape.anchor);
    let v = decode(&f.v);
    let ck = decode(&f.ck);
    let cv = decode(&f.cv);
    let s = f.shape;
    let mut out = vec![0.; s.rows * QH * D];
    std::thread::scope(|scope| {
        let chunk = (s.rows * QH).div_ceil(4);
        for (block, dst) in out.chunks_mut(chunk * D).enumerate() {
            let (q, k, v, ck, cv) = (&q, &k, &v, &ck, &cv);
            scope.spawn(move || {
                for (i, yy) in dst.chunks_mut(D).enumerate() {
                    let qh = block * chunk + i;
                    let r = qh / QH;
                    let h = (qh % QH) / 4;
                    let mut scores = vec![f64::NEG_INFINITY; s.ctx + s.rows];
                    for (key, score) in scores.iter_mut().enumerate() {
                        if !s.allowed(r, key) {
                            continue;
                        }
                        let (buf, idx) = if key < s.ctx {
                            (ck, key)
                        } else {
                            (k, key - s.ctx)
                        };
                        let kp = &buf[(idx * KH + h) * D..(idx * KH + h + 1) * D];
                        *score = q[qh * D..(qh + 1) * D]
                            .iter()
                            .zip(kp)
                            .map(|(&a, &b)| a * b)
                            .sum::<f64>()
                            / (D as f64).sqrt();
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for v in &mut scores {
                        *v = (*v - max).exp();
                    }
                    let denom = scores.iter().sum::<f64>();
                    for (key, &p) in scores.iter().enumerate() {
                        if p == 0. {
                            continue;
                        }
                        let (buf, idx) = if key < s.ctx {
                            (cv, key)
                        } else {
                            (v, key - s.ctx)
                        };
                        for c in 0..D {
                            yy[c] += p / denom * buf[(idx * KH + h) * D + c];
                        }
                    }
                    for v in yy {
                        *v = bf(*v);
                    }
                }
            });
        }
    });
    Ok(Reference { q, k, out })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_edge_and_future_key_have_hand_derived_mass() {
        let n = Norms {
            q: vec![bf16::ONE.to_bits(); D],
            k: vec![bf16::ONE.to_bits(); D],
            provenance: json!({}),
        };
        let f = Fixture::uniform_axis();
        let y = reference(&f, &n).unwrap();
        assert!(y.out[..QH * D].iter().all(|&v| v == 7.5));
        assert!(y.out[QH * D..].iter().all(|&v| v == bf(28. / 3.)));
        assert!(!f.shape.allowed(0, 0));
        assert!(f.shape.allowed(0, 4));
        let mut bad = f.shape;
        bad.anchor += 1;
        assert!(bad.validate().is_err());
    }
}
