//! Complete GDN+HC+MoE layer CPU oracle assembled from the qualified native
//! components. This is reference preparation, not a production inference path.
use crate::{
    gdn, hc, moe,
    projection::Pack,
    target_mlp::{self, RoutingArithmetic},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
pub struct Weights {
    pub layer: usize,
    pub hc: [hc::Weights; 2],
    pub gdn: gdn::Weights,
    pub mlp: target_mlp::Weights,
}
pub struct Decoded {
    pub hc: [[Vec<f32>; 2]; 2],
    pub gdn: Vec<Vec<f32>>,
    pub shared: [Vec<f32>; 3],
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(
            // The source config's ple_layer_ids=[2] is ONE-based; weights
            // and native execution place PLE in zero-based layer1.
            layer < 48 && layer % 4 != 3 && layer != 1,
            "GDN-only layer without PLE; QSA/PLE remain separate components"
        );
        Self::load_body(pack, layer)
    }
    /// R55: any GDN layer's HC/GDN/MoE body, layer 1 included (its PLE stage runs
    /// before the body, separately: metal_ple_target).
    pub fn load_body(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 != 3, "GDN layer body");
        Ok(Self {
            layer,
            hc: [
                hc::Weights::load(pack, layer, true)?,
                hc::Weights::load(pack, layer, false)?,
            ],
            gdn: gdn::Weights::load(pack, layer)?,
            mlp: target_mlp::Weights::load(pack, layer)?,
        })
    }
    pub fn decoded(&self) -> Result<Decoded> {
        Ok(Decoded {
            hc: [self.hc[0].decoded()?, self.hc[1].decoded()?],
            gdn: self
                .gdn
                .projections
                .iter()
                .map(|w| w.oracle_weights())
                .collect::<Result<_>>()?,
            shared: self.mlp.decoded_shared()?,
        })
    }
    pub fn provenance(&self) -> serde_json::Value {
        json!({"layer":self.layer,"hc":[self.hc[0].provenance,self.hc[1].provenance],"gdn":self.gdn.provenance,"mlp":self.mlp.provenance()})
    }
}
pub struct Reference {
    pub pre: hc::Read,
    pub projected: Vec<Vec<u16>>,
    pub core: gdn::Core,
    pub attention_output: Vec<u16>,
    pub after_attention: Vec<u16>,
    pub post: hc::Read,
    pub router_logits: Vec<u16>,
    pub routing: target_mlp::Routing,
    pub alternative_routing: target_mlp::Routing,
    pub original_experts: Vec<usize>,
    pub shared: target_mlp::Shared,
    pub routed: moe::Reference,
    pub alternative_routed: Vec<f64>,
    pub mlp_output: Vec<u16>,
    pub alternative_mlp_output: Vec<u16>,
    pub output: Vec<u16>,
    pub alternative_output: Vec<u16>,
    pub expert_provenance: serde_json::Value,
}
pub fn reference(
    pack: &Pack,
    w: &Weights,
    d: &Decoded,
    input: &[u16],
    initial: &gdn::Initial,
) -> Result<Reference> {
    reference_with_sigmoid(
        pack,
        w,
        d,
        input,
        initial,
        hc::SigmoidArithmetic::NativeFused,
    )
}
pub fn reference_with_sigmoid(
    pack: &Pack,
    w: &Weights,
    d: &Decoded,
    input: &[u16],
    initial: &gdn::Initial,
    sigmoid: hc::SigmoidArithmetic,
) -> Result<Reference> {
    reference_with_hc_policy(
        pack,
        w,
        d,
        input,
        initial,
        sigmoid,
        hc::MixArithmetic::WideSum,
    )
}
pub fn reference_with_hc_policy(
    pack: &Pack,
    w: &Weights,
    d: &Decoded,
    input: &[u16],
    initial: &gdn::Initial,
    sigmoid: hc::SigmoidArithmetic,
    mix: hc::MixArithmetic,
) -> Result<Reference> {
    ensure!(
        !input.is_empty()
            && input.len() % hc::F == 0
            && input.len() / hc::F <= target_mlp::MAX_ROWS,
        "bounded full-layer reference rows"
    );
    ensure!(d.gdn.len() == 5, "decoded GDN weights");
    initial.validate()?;
    let pre = hc::reference_with_policy(&w.hc[0], &d.hc[0], input, sigmoid, mix)?;
    let projected = (0..4)
        .map(|j| {
            Ok(w.gdn.projections[j]
                .reference(&d.gdn[j], &pre.mixed)?
                .into_iter()
                .map(|v| crate::rounding::bf16_from_f64(v).to_bits())
                .collect())
        })
        .collect::<Result<Vec<Vec<u16>>>>()?;
    let core = gdn::reference(&w.gdn, initial, &projected)?;
    let attention_output: Vec<_> = w.gdn.projections[4]
        .reference(&d.gdn[4], &core.gated)?
        .into_iter()
        .map(|v| crate::rounding::bf16_from_f64(v).to_bits())
        .collect();
    let after_attention = hc::write(input, &attention_output, &pre.inject)?;
    let post = hc::reference_with_policy(&w.hc[1], &d.hc[1], &after_attention, sigmoid, mix)?;
    let router_logits: Vec<_> = w
        .mlp
        .router
        .reference(&post.mixed)?
        .into_iter()
        .map(|v| crate::rounding::bf16_from_f64(v).to_bits())
        .collect();
    let routing = target_mlp::route(&router_logits, RoutingArithmetic::StagedBf16)?;
    let alternative_routing = target_mlp::route(&router_logits, RoutingArithmetic::SelectedF32)?;
    ensure!(
        routing.ids == alternative_routing.ids,
        "score-only arm changed selected experts"
    );
    let (original_experts, routes) = routing.bank_routes()?;
    let (alt_original, alt_routes) = alternative_routing.bank_routes()?;
    ensure!(
        original_experts == alt_original && routes.ids == alt_routes.ids,
        "score-only storage map changed"
    );
    let shared = target_mlp::shared_reference(&w.mlp, &d.shared, &post.mixed)?;
    let bank = moe::ExpertBank::load(pack, w.layer, &original_experts)?;
    let bd = bank.decoded();
    let floats: Vec<_> = post
        .mixed
        .iter()
        .map(|&v| bf16::from_bits(v).to_f32())
        .collect();
    let routed = moe::reference(&bank, &bd, &floats, &routes)?;
    let identity = target_mlp::reweight(&bank, &routed.raw_down, &routes)?;
    ensure!(
        identity == routed.output,
        "independent reduction/reweight identity control"
    );
    let alternative_routed = target_mlp::reweight(&bank, &routed.raw_down, &alt_routes)?;
    let mlp_output = target_mlp::combine(&routed.output, &shared.output)?;
    let alternative_mlp_output = target_mlp::combine(&alternative_routed, &shared.output)?;
    let output = hc::write(&after_attention, &mlp_output, &post.inject)?;
    let alternative_output = hc::write(&after_attention, &alternative_mlp_output, &post.inject)?;
    let projections: Vec<_> = [&bank.gate, &bank.up, &bank.down]
        .iter()
        .map(|p| json!({"trellis":p.raw.provenance,"scales":p.scale_provenance}))
        .collect();
    let expert_provenance = json!({"original_ids":original_experts,"projections":projections});
    Ok(Reference {
        pre,
        projected,
        core,
        attention_output,
        after_attention,
        post,
        router_logits,
        routing,
        alternative_routing,
        original_experts,
        shared,
        routed,
        alternative_routed,
        mlp_output,
        alternative_mlp_output,
        output,
        alternative_output,
        expert_provenance,
    })
}
