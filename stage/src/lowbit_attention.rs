//! Original affine4 attention/indexer weights and independent CPU validation.
use crate::{affine4::Affine4, lowbit_hc::small, qsa, target_attention as a};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub struct Weights {
    /// q/g, k, v, indexer, output; original packed representations.
    pub projections: Vec<Affine4>,
    pub norms: a::Norms,
    pub index_norms: qsa::Norms,
    pub layer: usize,
}
impl Weights {
    pub fn load(root: &Path, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 == 3, "full attention layer");
        let p = format!("language_model.model.layers.{layer}.self_attn");
        let mut projections = vec![];
        for (name, k, n) in [
            ("q_proj", a::H, a::QG),
            ("k_proj", a::H, a::KV),
            ("v_proj", a::H, a::KV),
            ("indexer.index_qk_proj", a::H, qsa::QK),
            ("o_proj", a::Q, a::H),
        ] {
            let w = Affine4::load(root, &format!("{p}.{name}"), k)?;
            ensure!(w.n == n, "attention projection shape");
            projections.push(w);
        }
        let (q, qp) = small(root, &format!("{p}.q_norm.weight"), &[a::D as u64])?;
        let (k, kp) = small(root, &format!("{p}.k_norm.weight"), &[a::D as u64])?;
        let (iq, iqp) = small(
            root,
            &format!("{p}.indexer.q_layernorm.weight"),
            &[qsa::D as u64],
        )?;
        let (ik, ikp) = small(
            root,
            &format!("{p}.indexer.k_layernorm.weight"),
            &[qsa::D as u64],
        )?;
        let w = Self {
            projections,
            norms: a::Norms {
                q,
                k,
                provenance: json!([qp, kp]),
            },
            index_norms: qsa::Norms {
                q: iq,
                k: ik,
                provenance: json!([iqp, ikp]),
            },
            layer,
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.layer < 48 && self.layer % 4 == 3 && self.projections.len() == 5,
            "attention weights"
        );
        for (w, (k, n)) in self.projections.iter().zip([
            (a::H, a::QG),
            (a::H, a::KV),
            (a::H, a::KV),
            (a::H, qsa::QK),
            (a::Q, a::H),
        ]) {
            w.validate()?;
            ensure!((w.k, w.n) == (k, n), "attention weight dimensions");
        }
        self.norms.validate()?;
        self.index_norms.validate()
    }
    pub fn decoded(&self) -> Result<Vec<Vec<f32>>> {
        self.projections
            .iter()
            .map(|w| w.oracle_weights())
            .collect()
    }
    pub fn provenance(&self) -> Value {
        json!({"layer":self.layer,"projections":self.projections.iter().map(|w|&w.provenance).collect::<Vec<_>>(),"norms":self.norms.provenance,"index_norms":self.index_norms.provenance})
    }
}
pub struct Reference {
    pub projected: a::Input,
    pub index_projected: Vec<u16>,
    pub index: qsa::Candidate,
    pub prepared: a::Prepared,
    pub cache: a::State,
    pub attention: Vec<u16>,
    pub gated: Vec<u16>,
    pub output: Vec<u16>,
}
pub fn reference(
    w: &Weights,
    d: &[Vec<f32>],
    x: &[u16],
    initial: &a::State,
    index_initial: &qsa::State,
) -> Result<Reference> {
    ensure!(
        d.len() == 5 && initial.base == index_initial.base && initial.len == index_initial.len,
        "attention reference alignment"
    );
    let project = |j: usize| -> Result<Vec<u16>> {
        Ok(w.projections[j]
            .reference(&d[j], x)?
            .into_iter()
            .map(a::bits)
            .collect())
    };
    let projected = a::Input {
        qg: project(0)?,
        k: project(1)?,
        v: project(2)?,
    };
    let index_projected = project(3)?;
    let index = qsa::reference_projected(&w.index_norms, index_initial, &index_projected)?;
    let prepared = a::prepare(&projected, &w.norms, initial.base + initial.len)?;
    let cache = a::append(initial, &prepared, &projected.v)?;
    let selection = (cache.len > qsa::BUDGET + qsa::RATIO - 1).then_some(&index.selection);
    let attention = a::attend(&prepared.q, &cache, selection)?;
    let gated = a::gate(&attention, &prepared.gate)?;
    let output = w.projections[4]
        .reference(&d[4], &gated)?
        .into_iter()
        .map(a::bits)
        .collect();
    Ok(Reference {
        projected,
        index_projected,
        index,
        prepared,
        cache,
        attention,
        gated,
        output,
    })
}
/// Diagnostic accepted-prefix reconstruction using actual device projection
/// and pooled results. Runtime commit does the tail gather on the CUDA device.
pub fn index_prefix(
    initial: &qsa::State,
    qk: &[u16],
    pooled: &[u16],
    accepted: usize,
) -> Result<qsa::State> {
    initial.validate()?;
    ensure!(
        qk.len() % qsa::QK == 0 && accepted <= qk.len() / qsa::QK,
        "index prefix bounds"
    );
    let len = initial.len + accepted;
    ensure!(
        pooled.len() >= len / 4 * qsa::D,
        "index pooled prefix bounds"
    );
    let mut raw = initial.tail.clone();
    raw.extend(
        qk.chunks_exact(qsa::QK)
            .take(accepted)
            .flat_map(|r| r[4 * qsa::D..].iter().copied()),
    );
    let tail = raw[raw.len() - len % 4 * qsa::D..].to_vec();
    let state = qsa::State {
        base: initial.base,
        len,
        pooled: pooled[..len / 4 * qsa::D].to_vec(),
        tail,
    };
    state.validate()?;
    Ok(state)
}
