//! Captured target features and committed DFlash2 context; no full target load.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL, Matrix},
    dflash_attention::{D, KH, MAX_ROWS, Norms, norm_rope},
    dflash_mlp::{EPS, H, bf, decode, encode, rms},
    projection::sha,
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub const FEATURES: usize = 5 * H;
pub const CAPACITY: usize = 2047;
pub const KV: usize = KH * D;
pub const CAPTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/evidence/r10-target-capture-01"
);

pub struct Features {
    pub prompts: Vec<(String, Vec<u16>)>,
    pub provenance: Value,
}
impl Features {
    pub fn load() -> Result<Self> {
        let root = Path::new(CAPTURE);
        let manifest: Value =
            serde_json::from_slice(&fs::read(root.join("capture-manifest.json"))?)?;
        ensure!(manifest["shape"] == json!([107, 4, H]), "capture geometry");
        let tokens = fs::read(root.join("hidden/tokens.bin"))?;
        ensure!(
            sha(&tokens) == "5c9eb11cbb557e286d949cb57628f3f4201b90ddab96d8f99070a4e0aad2b928",
            "capture token pin"
        );
        let mut joined = vec![0; 107 * FEATURES];
        let mut pins = vec![];
        for (tap, b) in [6, 16, 26, 36, 46].into_iter().enumerate() {
            let raw = fs::read(root.join(format!("hidden/boundary-{b:02}.bin")))?;
            ensure!(
                raw.len() == 107 * 4 * H * 2 && sha(&raw) == manifest["boundaries"][b]["sha256"],
                "capture boundary pin"
            );
            let v: Vec<_> = raw
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect();
            ensure!(v.iter().all(|v| v.is_finite()), "capture finite");
            // Match the mathematical HC mean with FP32 accumulation and one BF16
            // result. Serving reduction order/evaluation schedule is a later audit.
            for r in 0..107 {
                for c in 0..H {
                    let sum = (0..4).map(|s| v[(r * 4 + s) * H + c]).sum::<f32>();
                    joined[r * FEATURES + tap * H + c] = bf16::from_f32(sum * 0.25).to_bits();
                }
            }
            pins.push(json!({"boundary":b,"sha256":sha(&raw)}));
        }
        let prompts = vec![
            ("code-01".into(), joined[..52 * FEATURES].to_vec()),
            ("prose-01".into(), joined[52 * FEATURES..].to_vec()),
        ];
        Ok(Self {
            prompts,
            provenance: json!({"capture":CAPTURE,"source_target":"EXL3 4bpw, not BF16 teacher","boundaries":pins,"reduction":"FP32 four-stream mean -> BF16","generic_capture_schedule_not_yet_matched_to_serving":true}),
        })
    }
}

pub struct Weights {
    pub fc: Matrix,
    pub norm: Vec<u16>,
    pub k: Matrix,
    pub v: Matrix,
    pub qk: Norms,
    pub provenance: Value,
}
impl Weights {
    pub fn load(layer: usize) -> Result<Self> {
        ensure!(layer < 5, "context layer");
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "DFlash context weight pin"
        );
        let name = "hidden_norm.weight";
        let t = s.tensors.get(name).context("encoder norm absent")?;
        ensure!(
            t.dtype == "BF16" && t.shape == [H as u64],
            "encoder norm shape"
        );
        let raw = s.read_range(name, 0, t.bytes())?;
        let norm: Vec<_> = raw
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect();
        ensure!(
            norm.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "encoder norm finite"
        );
        let fc = Matrix::load("fc.weight")?;
        let k = Matrix::load(&format!("layers.{layer}.self_attn.k_proj.weight"))?;
        let v = Matrix::load(&format!("layers.{layer}.self_attn.v_proj.weight"))?;
        let qk = Norms::load(layer)?;
        ensure!(
            (fc.k, fc.n) == (FEATURES, H) && (k.k, k.n) == (H, KV) && (v.k, v.n) == (H, KV),
            "context matrix geometry"
        );
        let provenance = json!({"layer":layer,"fc":fc.provenance,"norm_sha256":sha(&raw),"k":k.provenance,"v":v.provenance,"qk":qk.provenance});
        Ok(Self {
            fc,
            norm,
            k,
            v,
            qk,
            provenance,
        })
    }
    pub fn reference(&self, features: &[u16], first: usize) -> Result<Trace> {
        ensure!(
            !features.is_empty()
                && features.len() % FEATURES == 0
                && features.len() / FEATURES <= 64
                && first + features.len() / FEATURES <= 262144,
            "context reference bounds"
        );
        let projected: Vec<_> = self.fc.reference(features)?.into_iter().map(bf).collect();
        let encoded = rms(&projected, &decode(&self.norm), H, EPS);
        let enc = encode(&encoded);
        let k: Vec<_> = self.k.reference(&enc)?.into_iter().map(bf).collect();
        let v: Vec<_> = self.v.reference(&enc)?.into_iter().map(bf).collect();
        let rotated = norm_rope(&encode(&k), &self.qk.k, KH, first);
        Ok(Trace {
            projected,
            encoded,
            k,
            v,
            rotated,
        })
    }
}
pub struct Trace {
    pub projected: Vec<f64>,
    pub encoded: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub rotated: Vec<f64>,
}

/// State is published only after completed, validated GPU work. This cache
/// accepts committed target features only; speculative rollback is upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub base: usize,
    pub len: usize,
    pub generation: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct Append {
    pub before: State,
    pub after: State,
    pub first: usize,
    pub rows: usize,
    pub drop: usize,
}
impl State {
    pub fn empty(base: usize) -> Result<Self> {
        ensure!(base <= 262144, "context base");
        Ok(Self {
            base,
            len: 0,
            generation: 0,
        })
    }
    pub fn plan(self, first: usize, rows: usize) -> Result<Append> {
        ensure!(
            self.len <= CAPACITY
                && first == self.base + self.len
                && (1..=MAX_ROWS).contains(&rows)
                && first + rows <= 262144,
            "context append contiguity/bounds"
        );
        let drop = (self.len + rows).saturating_sub(CAPACITY);
        let after = Self {
            base: self.base + drop,
            len: self.len + rows - drop,
            generation: self
                .generation
                .checked_add(1)
                .context("context generation overflow")?,
        };
        Ok(Append {
            before: self,
            after,
            first,
            rows,
            drop,
        })
    }
}
impl Append {
    pub fn control(self) -> [u32; 6] {
        [
            self.rows as u32,
            self.before.len as u32,
            self.drop as u32,
            self.first as u32,
            self.after.len as u32,
            self.after.base as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn committed_append_retains_absolute_positions() {
        let s = State {
            base: 500,
            len: CAPACITY,
            generation: 7,
        };
        let p = s.plan(500 + CAPACITY, 8).unwrap();
        assert_eq!(p.drop, 8);
        assert_eq!(
            p.after,
            State {
                base: 508,
                len: CAPACITY,
                generation: 8
            }
        );
        assert!(s.plan(500 + CAPACITY - 1, 8).is_err());
        assert!(s.plan(500 + CAPACITY, 0).is_err());
        assert!(s.plan(500 + CAPACITY, 17).is_err());
        assert!(State::empty(262143).unwrap().plan(262143, 2).is_err());
        let a = State::empty(19).unwrap().plan(19, 8).unwrap().after;
        let b = State::empty(19)
            .unwrap()
            .plan(19, 3)
            .unwrap()
            .after
            .plan(22, 5)
            .unwrap()
            .after;
        assert_eq!((a.base, a.len), (b.base, b.len));
    }
}
