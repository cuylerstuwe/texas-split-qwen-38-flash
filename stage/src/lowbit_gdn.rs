//! Original packed affine4 GDN projections and small recurrent parameters.
use crate::{
    affine4::Affine4,
    gdn::{self, *},
    lowbit_hc::small,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub struct Weights {
    pub projections: Vec<Affine4>,
    pub conv: Vec<u16>,
    pub a_log: Vec<u16>,
    pub dt_bias: Vec<u16>,
    pub norm: Vec<u16>,
    pub provenance: Value,
}
impl Weights {
    pub fn load(root: &Path, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 != 3, "lowbit GDN layer");
        let p = format!("language_model.model.layers.{layer}.linear_attn");
        let mut projections = Vec::new();
        for (name, k, n) in [
            ("in_proj_qkv", H, C),
            ("in_proj_z", H, V),
            ("in_proj_a", H, HV),
            ("in_proj_b", H, HV),
            ("out_proj", V, H),
        ] {
            let w = Affine4::load(root, &format!("{p}.{name}"), k)?;
            ensure!(w.n == n, "lowbit GDN projection shape");
            projections.push(w);
        }
        let (conv, cp) = small(root, &format!("{p}.conv1d.weight"), &[C as u64, 4, 1])?;
        let (a_log, ap) = small(root, &format!("{p}.A_log"), &[HV as u64])?;
        let (dt_bias, dp) = small(root, &format!("{p}.dt_bias"), &[HV as u64])?;
        let (norm, np) = small(root, &format!("{p}.norm.weight"), &[DV as u64])?;
        let provenance = json!({"layer":layer,"projections":projections.iter().map(|w|&w.provenance).collect::<Vec<_>>(),"parameters":[cp,ap,dp,np]});
        let w = Self {
            projections,
            conv,
            a_log,
            dt_bias,
            norm,
            provenance,
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.projections.len() == 5, "GDN projection count");
        for (w, (k, n)) in self
            .projections
            .iter()
            .zip([(H, C), (H, V), (H, HV), (H, HV), (V, H)])
        {
            w.validate()?;
            ensure!((w.k, w.n) == (k, n), "lowbit GDN projection geometry");
        }
        self.parameters().validate()
    }
    pub fn parameters(&self) -> gdn::Parameters<'_> {
        gdn::Parameters {
            conv: &self.conv,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
        }
    }
    pub fn decoded(&self) -> Result<Vec<Vec<f32>>> {
        self.projections
            .iter()
            .map(|w| w.oracle_weights())
            .collect()
    }
    pub fn core_reference(&self, initial: &Initial, projected: &[Vec<u16>]) -> Result<Core> {
        gdn::reference_parameters(
            self.parameters(),
            initial,
            projected,
            |_, _, _, _, _, _, _| {},
        )
    }
}
