//! Bounded independent CPU oracle for the full HC/attention/EXL3-MLP layer.
//! Never a serving backend; expert decoding is limited to <=40 experts/chunk.
use crate::{hc, moe, projection::Pack, qsa, target_attention as a, target_mlp as m};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
pub struct Weights {
    pub layer: usize,
    pub hc: [hc::Weights; 2],
    pub attention: a::Weights,
    pub index: qsa::Weights,
    pub mlp: m::Weights,
}
pub struct Decoded {
    pub hc: [[Vec<f32>; 2]; 2],
    pub attention: Vec<Vec<f32>>,
    pub index: Vec<f32>,
    pub shared: [Vec<f32>; 3],
}
impl Weights {
    pub fn load(p: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 == 3, "full attention layer");
        Ok(Self {
            layer,
            hc: [
                hc::Weights::load(p, layer, true)?,
                hc::Weights::load(p, layer, false)?,
            ],
            attention: a::Weights::load(p, layer)?,
            index: qsa::Weights::load(p, layer)?,
            mlp: m::Weights::load(p, layer)?,
        })
    }
    pub fn decoded(&self) -> Result<Decoded> {
        Ok(Decoded {
            hc: [self.hc[0].decoded()?, self.hc[1].decoded()?],
            attention: [
                &self.attention.q,
                &self.attention.k,
                &self.attention.v,
                &self.attention.o,
            ]
            .iter()
            .map(|w| w.oracle_weights())
            .collect::<Result<_>>()?,
            index: self.index.projection.oracle_weights()?,
            shared: self.mlp.decoded_shared()?,
        })
    }
    pub fn provenance(&self) -> Value {
        json!({"layer":self.layer,"hc":[self.hc[0].provenance,self.hc[1].provenance],"attention":self.attention.provenance(),"index":self.index.provenance(),"mlp":self.mlp.provenance()})
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Mlp {
    pub logits: Vec<u16>,
    pub cpu_ids: Vec<u32>,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
    pub routed: Vec<f64>,
    pub shared: Vec<u16>,
    pub output: Vec<u16>,
    pub expert_pins: Value,
}
pub fn mlp_reference(
    p: &Pack,
    w: &m::Weights,
    d: &[Vec<f32>; 3],
    x: &[u16],
    actual: Option<(&[u32], &[f32])>,
) -> Result<Mlp> {
    ensure!(
        !x.is_empty() && x.len() % a::H == 0 && x.len() / a::H <= 8,
        "bounded layer MLP oracle"
    );
    let rows = x.len() / a::H;
    let logits: Vec<_> = w.router.reference(x)?.into_iter().map(a::bits).collect();
    let route = m::route(&logits, m::RoutingArithmetic::StagedBf16)?;
    let (ids, scores) = actual.unwrap_or((&route.ids, &route.scores));
    ensure!(
        ids.len() == rows * moe::TOPK
            && scores.len() == ids.len()
            && ids.iter().all(|&v| v < 512)
            && scores.iter().all(|&v| v.is_finite() && v >= 0.),
        "MLP conditional routes"
    );
    let mut routed = vec![];
    let mut pins = vec![];
    for first in (0..rows).step_by(4) {
        let end = (first + 4).min(rows);
        let ids = &ids[first * moe::TOPK..end * moe::TOPK];
        let originals: Vec<_> = ids
            .iter()
            .map(|&v| v as usize)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        ensure!(originals.len() <= 40, "compact decoded bank");
        let bank = moe::ExpertBank::load(p, w.layer, &originals)?;
        let decoded = bank.decoded();
        let local = moe::Routes {
            rows: end - first,
            experts: originals.len(),
            ids: ids
                .iter()
                .map(|&id| originals.binary_search(&(id as usize)).unwrap() as u32)
                .collect(),
            scores: scores[first * moe::TOPK..end * moe::TOPK].to_vec(),
        };
        let f: Vec<_> = x[first * a::H..end * a::H]
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        routed.extend(moe::reference(&bank, &decoded, &f, &local)?.output);
        pins.push(json!({"first":first,"end":end,"originals":originals,"projections":([&bank.gate,&bank.up,&bank.down].map(|w|json!({"trellis":w.raw.provenance,"scales":w.scale_provenance})))}));
    }
    let shared = m::shared_reference(w, d, x)?.output;
    let output = m::combine(&routed, &shared)?;
    Ok(Mlp {
        logits,
        cpu_ids: route.ids.clone(),
        ids: ids.to_vec(),
        scores: scores.to_vec(),
        routed,
        shared,
        output,
        expert_pins: json!(pins),
    })
}
pub struct Reference {
    pub pre: hc::Read,
    pub projected: a::Input,
    pub index_projected: Vec<u16>,
    pub index: qsa::Candidate,
    pub prepared: a::Prepared,
    pub state: a::State,
    pub attention: Vec<u16>,
    pub gated: Vec<u16>,
    pub attention_output: Vec<u16>,
    pub stream: Vec<u16>,
    pub post: hc::Read,
    pub mlp: Mlp,
    pub output: Vec<u16>,
}
pub fn reference(
    p: &Pack,
    w: &Weights,
    d: &Decoded,
    x: &[u16],
    initial: &a::State,
    index_initial: &qsa::State,
) -> Result<Reference> {
    initial.validate()?;
    index_initial.validate()?;
    ensure!(d.attention.len() == 4, "decoded attention projections");
    ensure!(
        !x.is_empty() && x.len() % hc::F == 0 && x.len() / hc::F <= 8,
        "bounded attention-layer oracle"
    );
    ensure!(
        initial.base == index_initial.base && initial.len == index_initial.len,
        "oracle state alignment"
    );
    let pre = hc::reference(&w.hc[0], &d.hc[0], x)?;
    let projected = [&w.attention.q, &w.attention.k, &w.attention.v]
        .iter()
        .zip(&d.attention)
        .map(|(w, d)| {
            Ok(w.reference(d, &pre.mixed)?
                .into_iter()
                .map(a::bits)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    let projected = a::Input {
        qg: projected[0].clone(),
        k: projected[1].clone(),
        v: projected[2].clone(),
    };
    let (index_projected, index) = qsa::reference(&w.index, &d.index, index_initial, &pre.mixed)?;
    let prepared = a::prepare(&projected, &w.attention.norms, initial.base + initial.len)?;
    let state = a::append(initial, &prepared, &projected.v)?;
    let selection = if state.len > 2051 {
        Some(&index.selection)
    } else {
        None
    };
    let attention = a::attend(&prepared.q, &state, selection)?;
    let gated = a::gate(&attention, &prepared.gate)?;
    let attention_output = w
        .attention
        .o
        .reference(&d.attention[3], &gated)?
        .into_iter()
        .map(a::bits)
        .collect::<Vec<_>>();
    let stream = hc::write(x, &attention_output, &pre.inject)?;
    let post = hc::reference(&w.hc[1], &d.hc[1], &stream)?;
    let mlp = mlp_reference(p, &w.mlp, &d.shared, &post.mixed, None)?;
    let output = hc::write(&stream, &mlp.output, &post.inject)?;
    Ok(Reference {
        pre,
        projected,
        index_projected,
        index,
        prepared,
        state,
        attention,
        gated,
        attention_output,
        stream,
        post,
        mlp,
        output,
    })
}
