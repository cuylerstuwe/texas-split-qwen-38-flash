//! Original lowbit HC/attention/HC/512-expert MLP layer, independent CPU oracle.
//! Weight decoding here is bounded validation work, never the CUDA runtime.
use crate::{
    bf16_linear::Matrix, hc, lowbit_attention, lowbit_hc, lowbit_layer, lowbit_router,
    lowbit_shared, qsa, target_attention as a,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub struct Weights {
    pub layer: usize,
    pub hc: [lowbit_hc::Weights; 2],
    pub attention: lowbit_attention::Weights,
    pub router: Matrix,
    pub shared: lowbit_shared::Weights,
}
pub struct Decoded {
    pub hc: [[Vec<f32>; 2]; 2],
    pub attention: Vec<Vec<f32>>,
    pub shared: [Vec<f32>; 3],
}
impl Weights {
    pub fn load(root: &Path, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 == 3, "full attention layer only");
        Ok(Self {
            layer,
            hc: [
                lowbit_hc::Weights::load(root, layer, true)?,
                lowbit_hc::Weights::load(root, layer, false)?,
            ],
            attention: lowbit_attention::Weights::load(root, layer)?,
            router: lowbit_router::load(root, layer)?,
            shared: lowbit_shared::Weights::load(root, layer)?,
        })
    }
    pub fn decoded(&self) -> Result<Decoded> {
        Ok(Decoded {
            hc: [self.hc[0].decoded()?, self.hc[1].decoded()?],
            attention: self.attention.decoded()?,
            shared: self.shared.decoded()?,
        })
    }
    pub fn provenance(&self) -> Value {
        json!({"layer":self.layer,"hc":[self.hc[0].provenance,self.hc[1].provenance],"attention":self.attention.provenance(),"router":self.router.provenance,"shared":self.shared.provenance()})
    }
}
pub fn mlp_reference(
    root: &Path,
    w: &Weights,
    d: &Decoded,
    x: &[u16],
    actual: Option<&lowbit_router::Routing>,
) -> Result<lowbit_layer::MlpReference> {
    lowbit_layer::mlp_reference_weights(root, w.layer, &w.router, &w.shared, &d.shared, x, actual)
}
pub struct Reference {
    pub pre: hc::Read,
    pub attention: lowbit_attention::Reference,
    pub stream: Vec<u16>,
    pub post: hc::Read,
    pub mlp: lowbit_layer::MlpReference,
    pub output: Vec<u16>,
}
pub fn reference(
    root: &Path,
    w: &Weights,
    d: &Decoded,
    x: &[u16],
    initial: &a::State,
    index: &qsa::State,
) -> Result<Reference> {
    let pre = lowbit_hc::reference(&w.hc[0], &d.hc[0], x)?;
    let attention =
        lowbit_attention::reference(&w.attention, &d.attention, &pre.mixed, initial, index)?;
    let stream = hc::write(x, &attention.output, &pre.inject)?;
    let post = lowbit_hc::reference(&w.hc[1], &d.hc[1], &stream)?;
    let mlp = mlp_reference(root, w, d, &post.mixed, None)?;
    let output = hc::write(&stream, &mlp.output, &post.inject)?;
    Ok(Reference {
        pre,
        attention,
        stream,
        post,
        mlp,
        output,
    })
}
