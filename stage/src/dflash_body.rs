//! Five-layer DFlash2 body with real target features and raw anchor/mask inputs.
//! CPU reference only; selector, shared target vocabulary head and sampling follow.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL},
    dflash_attention::{D, KH, Shape, norm_rope},
    dflash_context::{FEATURES, Weights as ContextWeights},
    dflash_layer::{Fixture as LayerFixture, Trace as LayerTrace, Weights as LayerWeights},
    dflash_mlp::{Arithmetic, EPS, H, MAX_ROWS, bf, decode, encode, rms},
    embedding::{MASK, RawEmbeddings},
    projection::{Pack, sha},
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::path::Path;

pub const LAYERS: usize = 5;
pub struct Weights {
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<u16>,
    pub provenance: Value,
}
impl Weights {
    pub fn load() -> Result<Self> {
        let layers = (0..LAYERS)
            .map(LayerWeights::load)
            .collect::<Result<Vec<_>>>()?;
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(s.header_sha256 == HEADER_SHA, "DFlash body header pin");
        let t = s
            .tensors
            .get("norm.weight")
            .context("body final norm absent")?;
        ensure!(
            t.dtype == "BF16" && t.shape == [H as u64],
            "body final norm geometry"
        );
        let raw = s.read_range("norm.weight", 0, t.bytes())?;
        let norm = raw
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(
            norm.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "body final norm finite"
        );
        let provenance = json!({"layers":layers.iter().map(|w| &w.provenance).collect::<Vec<_>>(),
            "final_norm":{"tensor":"norm.weight","sha256":sha(&raw)}, "checkpoint_header_sha256":HEADER_SHA});
        Ok(Self {
            layers,
            norm,
            provenance,
        })
    }
    pub fn fixture(
        &self,
        pack: &Pack,
        features: &[u16],
        anchor_token: usize,
        rows: usize,
    ) -> Result<Fixture> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && !features.is_empty()
                && features.len() % FEATURES == 0
                && features.len() / FEATURES <= 64,
            "body real feature bounds"
        );
        let ctx = features.len() / FEATURES;
        let encoder = ContextWeights::load(0)?;
        // The target-feature encoder is shared across all five layers.
        let projected = encoder
            .fc
            .reference(features)?
            .into_iter()
            .map(bf)
            .collect::<Vec<_>>();
        let encoded = encode(&rms(&projected, &decode(&encoder.norm), H, EPS));
        let mut contexts = Vec::new();
        for w in &self.layers {
            let k =
                w.k.reference(&encoded)?
                    .into_iter()
                    .map(bf)
                    .collect::<Vec<_>>();
            let v =
                w.v.reference(&encoded)?
                    .into_iter()
                    .map(bf)
                    .collect::<Vec<_>>();
            contexts.push((encode(&norm_rope(&encode(&k), &w.qk.k, KH, 0)), encode(&v)));
        }
        let mut tokens = vec![MASK; rows];
        tokens[0] = anchor_token;
        let embedding = RawEmbeddings::load(pack, &tokens)?;
        let provenance = json!({"embedding":embedding.provenance,
            "shared_encoder":encoder.provenance, "feature_rows":ctx,
            "context_source":"CPU FP64 original-weight projection, BF16 boundaries; not CUDA append timings"});
        let f = Fixture {
            shape: Shape {
                rows,
                ctx,
                base: 0,
                anchor: ctx,
                window: 2048,
            },
            x: embedding.values,
            contexts,
            provenance,
        };
        f.validate()?;
        Ok(f)
    }
    pub fn reference(&self, fixture: &Fixture) -> Result<Trace> {
        fixture.validate()?;
        ensure!(self.layers.len() == LAYERS, "body layer count");
        let mut x = fixture.x.clone();
        let mut layers = Vec::new();
        for (w, (k, v)) in self.layers.iter().zip(&fixture.contexts) {
            let f = LayerFixture {
                shape: fixture.shape,
                x,
                ck: k.clone(),
                cv: v.clone(),
            };
            let trace = w.reference(&f, Arithmetic::Staged)?;
            x = encode(&trace.mlp.out);
            layers.push(trace);
        }
        let output = rms(&decode(&x), &decode(&self.norm), H, EPS);
        Ok(Trace { layers, output })
    }
}
#[derive(Clone)]
pub struct Fixture {
    pub shape: Shape,
    pub x: Vec<u16>,
    pub contexts: Vec<(Vec<u16>, Vec<u16>)>,
    pub provenance: Value,
}
impl Fixture {
    pub fn validate(&self) -> Result<()> {
        self.shape.validate()?;
        ensure!(self.contexts.len() == LAYERS, "body context layer count");
        ensure!(
            self.x.len() == self.shape.rows * H
                && self.x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "body input shape/finite"
        );
        for (k, v) in &self.contexts {
            ensure!(
                k.len() == self.shape.ctx * KH * D
                    && v.len() == k.len()
                    && k.iter().chain(v).all(|&v| bf16::from_bits(v).is_finite()),
                "body context shape/finite"
            );
        }
        Ok(())
    }
}
pub struct Trace {
    pub layers: Vec<LayerTrace>,
    pub output: Vec<f64>,
}
