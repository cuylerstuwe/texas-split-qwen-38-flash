//! Original lowbit non-PLE recurrent layer weights and bounded CPU oracle.
//! CPU decoding is validation only; execution uses the original packed banks.
use crate::{
    bf16_linear::Matrix, gdn, hc, lowbit_gdn, lowbit_hc, lowbit_router, lowbit_shared, mcg_moe, moe,
};
use anyhow::{Result, ensure};
use half::{bf16, f16};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

pub struct Weights {
    pub layer: usize,
    pub hc: [lowbit_hc::Weights; 2],
    pub gdn: lowbit_gdn::Weights,
    pub router: Matrix,
    pub shared: lowbit_shared::Weights,
}
pub struct Decoded {
    pub hc: [[Vec<f32>; 2]; 2],
    pub gdn: Vec<Vec<f32>>,
    pub shared: [Vec<f32>; 3],
}
impl Weights {
    pub fn load(root: &Path, layer: usize) -> Result<Self> {
        ensure!(
            layer < 48 && layer % 4 != 3 && layer != 1,
            "non-PLE GDN layer only"
        );
        Self::load_body(root, layer)
    }
    /// Layer1's body weights; its PLE must execute before this recurrent body.
    pub fn load_ple_body(root: &Path) -> Result<Self> {
        Self::load_body(root, 1)
    }
    fn load_body(root: &Path, layer: usize) -> Result<Self> {
        Ok(Self {
            layer,
            hc: [
                lowbit_hc::Weights::load(root, layer, true)?,
                lowbit_hc::Weights::load(root, layer, false)?,
            ],
            gdn: lowbit_gdn::Weights::load(root, layer)?,
            router: lowbit_router::load(root, layer)?,
            shared: lowbit_shared::Weights::load(root, layer)?,
        })
    }
    pub fn decoded(&self) -> Result<Decoded> {
        Ok(Decoded {
            hc: [self.hc[0].decoded()?, self.hc[1].decoded()?],
            gdn: self.gdn.decoded()?,
            shared: self.shared.decoded()?,
        })
    }
    pub fn provenance(&self) -> Value {
        json!({"layer":self.layer,"hc":[self.hc[0].provenance,self.hc[1].provenance],"gdn":self.gdn.provenance,"router":self.router.provenance,"shared":self.shared.provenance()})
    }
}
fn round(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
pub struct MlpReference {
    pub router_raw: Vec<f64>,
    pub logits: Vec<u16>,
    pub routing: lowbit_router::Routing,
    pub routed: Vec<Vec<f64>>,
    pub shared: lowbit_shared::Reference,
    pub output: Vec<u16>,
    pub provenance: Value,
}
/// At most four activation rows / forty unique experts decoded at a time.
/// A caller may supply actual GPU routes for conditional arithmetic isolation;
/// the independently recomputed router remains in the returned reference.
pub fn mlp_reference(
    root: &Path,
    w: &Weights,
    decoded: &Decoded,
    x: &[u16],
    actual: Option<&lowbit_router::Routing>,
) -> Result<MlpReference> {
    mlp_reference_weights(
        root,
        w.layer,
        &w.router,
        &w.shared,
        &decoded.shared,
        x,
        actual,
    )
}
/// Shared independent MLP oracle for recurrent and full-attention layers.
pub fn mlp_reference_weights(
    root: &Path,
    layer: usize,
    router: &Matrix,
    shared_weights: &lowbit_shared::Weights,
    decoded_shared: &[Vec<f32>; 3],
    x: &[u16],
    actual: Option<&lowbit_router::Routing>,
) -> Result<MlpReference> {
    ensure!(
        !x.is_empty() && x.len() % hc::H == 0 && x.len() / hc::H <= hc::MAX_ROWS,
        "MLP reference rows"
    );
    let rows = x.len() / hc::H;
    let router_raw = router.reference(x)?;
    let logits: Vec<_> = router_raw.iter().map(|&v| round(v)).collect();
    let routing = lowbit_router::select(&logits, lowbit_router::Scores::StagedBf16)?;
    let r = actual.unwrap_or(&routing);
    r.validate()?;
    ensure!(r.rows == rows, "MLP route rows");
    let mut routed: Vec<Vec<f64>> = (0..8).map(|_| vec![]).collect();
    let mut provenance = vec![];
    for first in (0..rows).step_by(4) {
        let end = (first + 4).min(rows);
        let ids = &r.ids[first * moe::TOPK..end * moe::TOPK];
        let originals: Vec<_> = ids
            .iter()
            .map(|&v| v as usize)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let bank = mcg_moe::Bank::load(root, layer, &originals)?;
        let bd = bank.decoded()?;
        let routes = moe::Routes {
            rows: end - first,
            experts: originals.len(),
            ids: ids
                .iter()
                .map(|&v| originals.binary_search(&(v as usize)).unwrap() as u32)
                .collect(),
            scores: r.scores[first * moe::TOPK..end * moe::TOPK].to_vec(),
        };
        let x: Vec<_> = x[first * hc::H..end * hc::H]
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        let v = mcg_moe::reference(&bank, &bd, &x, &routes)?;
        let halves = |x: Vec<u16>| {
            x.into_iter()
                .map(|v| f16::from_bits(v).to_f64())
                .collect::<Vec<_>>()
        };
        for (all, part) in routed.iter_mut().zip([
            halves(v.prepared_gate),
            halves(v.prepared_up),
            v.raw_gate,
            v.raw_up,
            halves(v.middle),
            v.raw_down,
            v.output,
            v.bf16_output,
        ]) {
            all.extend(part);
        }
        provenance.push(json!({"first":first,"end":end,"bank":bank.provenance()}));
    }
    let shared = lowbit_shared::reference(
        shared_weights,
        decoded_shared,
        x,
        lowbit_shared::Arithmetic::StagedBf16,
    )?;
    let output = routed[7]
        .iter()
        .zip(&shared.stages.output)
        .map(|(&a, &b)| round(a + bf16::from_bits(b).to_f64()))
        .collect();
    Ok(MlpReference {
        router_raw,
        logits,
        routing,
        routed,
        shared,
        output,
        provenance: json!(provenance),
    })
}
pub struct Reference {
    pub pre: hc::Read,
    pub projected: Vec<Vec<u16>>,
    pub core: gdn::Core,
    pub attention: Vec<u16>,
    pub stream: Vec<u16>,
    pub post: hc::Read,
    pub mlp: MlpReference,
    pub output: Vec<u16>,
}
pub fn reference(
    root: &Path,
    w: &Weights,
    d: &Decoded,
    x: &[u16],
    initial: &gdn::Initial,
) -> Result<Reference> {
    let pre = lowbit_hc::reference(&w.hc[0], &d.hc[0], x)?;
    let projected = (0..4)
        .map(|j| {
            Ok(w.gdn.projections[j]
                .reference(&d.gdn[j], &pre.mixed)?
                .into_iter()
                .map(round)
                .collect())
        })
        .collect::<Result<Vec<Vec<u16>>>>()?;
    let core = w.gdn.core_reference(initial, &projected)?;
    let attention: Vec<_> = w.gdn.projections[4]
        .reference(&d.gdn[4], &core.gated)?
        .into_iter()
        .map(round)
        .collect();
    let stream = hc::write(x, &attention, &pre.inject)?;
    let post = lowbit_hc::reference(&w.hc[1], &d.hc[1], &stream)?;
    let mlp = mlp_reference(root, w, d, &post.mixed, None)?;
    let output = hc::write(&stream, &mlp.output, &post.inject)?;
    Ok(Reference {
        pre,
        projected,
        core,
        attention,
        stream,
        post,
        mlp,
        output,
    })
}
