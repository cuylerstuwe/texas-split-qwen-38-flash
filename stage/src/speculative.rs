//! CPU reference for sparse, integer-mass proposal packets and exact correction.
//! This is not a model/tree verifier: prefix identities, path-specific filters,
//! state commit/rollback and device visibility remain integration obligations.
//! Sample FROM the packet, never round a different q after sampling from it.
use anyhow::{Result, ensure};

pub const MASS: u32 = u16::MAX as u32;

#[derive(Clone, Debug)]
pub struct Distribution {
    probabilities: Vec<f64>,
}
impl Distribution {
    pub fn from_weights(weights: &[f64]) -> Result<Self> {
        ensure!(!weights.is_empty(), "empty distribution");
        ensure!(
            weights.iter().all(|x| x.is_finite() && *x >= 0.0),
            "invalid probability weight"
        );
        let max = weights.iter().copied().fold(0.0_f64, f64::max);
        ensure!(max > 0.0, "zero total weight");
        let total: f64 = weights.iter().map(|x| x / max).sum();
        Ok(Self {
            probabilities: weights.iter().map(|x| (x / max) / total).collect(),
        })
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    /// Caller supplies independent target randomness in [0,1).
    pub fn sample_uniform(&self, u: f64) -> Result<usize> {
        ensure!((0.0..1.0).contains(&u), "uniform bounds");
        Ok(self.sample(u))
    }
    fn sample(&self, u: f64) -> usize {
        let mut cumulative = 0.0;
        let mut last_positive = 0;
        for (i, &p) in self.probabilities.iter().enumerate() {
            if p > 0.0 {
                last_positive = i;
            }
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        // Account only for finite summation rounding at the upper CDF endpoint.
        last_positive
    }
}

#[derive(Clone, Debug)]
pub struct SparseProposal {
    vocab: usize,
    entries: Vec<(u32, u16)>,
}
/// A token sampled from the SAME packet subsequently used for verification.
pub struct Candidate<'a> {
    proposal: &'a SparseProposal,
    token: usize,
    mass: u16,
}
impl Candidate<'_> {
    pub fn token(&self) -> usize {
        self.token
    }
}

impl SparseProposal {
    pub fn from_masses(vocab: usize, mut entries: Vec<(u32, u16)>) -> Result<Self> {
        ensure!(vocab > 0 && vocab <= u32::MAX as usize, "vocab bounds");
        ensure!(
            !entries.is_empty() && entries.len() <= MASS as usize,
            "support bounds"
        );
        entries.sort_unstable_by_key(|e| e.0);
        ensure!(entries.windows(2).all(|e| e[0].0 != e[1].0), "duplicate ID");
        ensure!(
            entries
                .iter()
                .all(|&(id, m)| (id as usize) < vocab && m > 0),
            "invalid entry"
        );
        ensure!(
            entries.iter().map(|e| e.1 as u64).sum::<u64>() == MASS as u64,
            "proposal mass must sum to65535"
        );
        Ok(Self { vocab, entries })
    }
    /// Deliberately change q by top-k and largest-remainder mass quantization.
    /// The target distribution is unaffected. This isn't a quality guarantee.
    pub fn from_topk(draft: &Distribution, k: usize) -> Result<Self> {
        ensure!(k > 0 && k <= MASS as usize, "top-k bounds");
        ensure!(
            draft.probabilities.len() <= u32::MAX as usize,
            "vocab bounds"
        );
        let mut ranked: Vec<_> = draft
            .probabilities
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, p)| *p > 0.0)
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(k);
        Self::from_sparse_weights(draft.probabilities.len(), &ranked)
    }
    /// Quantize an already selected, unique sparse support. Sample from the
    /// returned masses; this explicitly changes the proposal law, not target p.
    pub fn from_sparse_weights(vocab: usize, weights: &[(usize, f64)]) -> Result<Self> {
        ensure!(
            vocab > 0
                && vocab <= u32::MAX as usize
                && !weights.is_empty()
                && weights.len() <= MASS as usize,
            "sparse support bounds"
        );
        ensure!(
            weights
                .iter()
                .all(|&(id, p)| id < vocab && p.is_finite() && p >= 0.),
            "invalid sparse weights"
        );
        let mut ids = weights.iter().map(|x| x.0).collect::<Vec<_>>();
        ids.sort_unstable();
        ensure!(ids.windows(2).all(|x| x[0] != x[1]), "duplicate sparse ID");
        let maximum = weights.iter().map(|x| x.1).fold(0f64, f64::max);
        ensure!(maximum > 0., "zero sparse weight");
        let ranked: Vec<_> = weights
            .iter()
            .filter(|x| x.1 > 0.)
            .map(|&(id, p)| (id, p / maximum))
            .collect();
        let total: f64 = ranked.iter().map(|e| e.1).sum();
        let mut quantized: Vec<_> = ranked
            .iter()
            .map(|&(id, p)| {
                let v = p / total * MASS as f64;
                (id as u32, v.floor() as u32, v.fract())
            })
            .collect();
        let used: u32 = quantized.iter().map(|e| e.1).sum();
        ensure!(used <= MASS, "mass rounding overflow");
        let extra = (MASS - used) as usize;
        ensure!(extra <= quantized.len(), "mass rounding deficit");
        quantized.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        for e in quantized.iter_mut().take(extra) {
            e.1 += 1;
        }
        Self::from_masses(
            vocab,
            quantized
                .into_iter()
                .filter(|e| e.1 > 0)
                .map(|e| (e.0, e.1 as u16))
                .collect(),
        )
    }
    pub fn entries(&self) -> &[(u32, u16)] {
        &self.entries
    }
    /// Packed entry payload only. An eventual transport also needs framing,
    /// version/vocab, prefix/epoch identity and completion/ownership metadata.
    pub fn encode_entries(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * 6);
        for &(id, m) in &self.entries {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&m.to_le_bytes());
        }
        out
    }
    pub fn decode_entries(vocab: usize, bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() % 6 == 0 && bytes.len() / 6 <= MASS as usize,
            "invalid packet size"
        );
        Self::from_masses(
            vocab,
            bytes
                .chunks_exact(6)
                .map(|b| {
                    (
                        u32::from_le_bytes(b[..4].try_into().unwrap()),
                        u16::from_le_bytes(b[4..].try_into().unwrap()),
                    )
                })
                .collect(),
        )
    }
    /// Caller provides a uniform ticket in [0,65535). Rejection of the u16
    /// value65535 is valid; mapping all65536 values with modulo is biased.
    pub fn sample_ticket(&self, ticket: u16) -> Result<Candidate<'_>> {
        ensure!((ticket as u32) < MASS, "ticket outside proposal measure");
        let mut end = 0_u32;
        for &(id, mass) in &self.entries {
            end += mass as u32;
            if (ticket as u32) < end {
                return Ok(Candidate {
                    proposal: self,
                    token: id as usize,
                    mass,
                });
            }
        }
        unreachable!("validated total mass")
    }
    pub fn overlap(&self, target: &Distribution) -> Result<f64> {
        ensure!(target.probabilities.len() == self.vocab, "vocab mismatch");
        Ok(self
            .entries
            .iter()
            .map(|&(id, mass)| target.probabilities[id as usize].min(mass as f64 / MASS as f64))
            .sum())
    }
}

#[derive(Debug)]
pub struct Decision {
    pub token: usize,
    pub accepted: bool,
}

/// Standard maximal-coupling speculative step. Uniforms must be independent
/// of the proposal ticket and each other, and conditional p must match prefix.
pub fn verify(
    target: &Distribution,
    candidate: &Candidate<'_>,
    acceptance_uniform: f64,
    correction_uniform: f64,
) -> Result<Decision> {
    ensure!(
        (0.0..1.0).contains(&acceptance_uniform) && (0.0..1.0).contains(&correction_uniform),
        "uniform bounds"
    );
    let q = candidate.proposal;
    ensure!(target.probabilities.len() == q.vocab, "vocab mismatch");
    let proposal_p = candidate.mass as f64 / MASS as f64;
    if acceptance_uniform < (target.probabilities[candidate.token] / proposal_p).min(1.0) {
        return Ok(Decision {
            token: candidate.token,
            accepted: true,
        });
    }
    let mut residual = target.probabilities.clone();
    for &(id, mass) in &q.entries {
        residual[id as usize] = (residual[id as usize] - mass as f64 / MASS as f64).max(0.0);
    }
    let corrected = Distribution::from_weights(&residual)?;
    Ok(Decision {
        token: corrected.sample(correction_uniform),
        accepted: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_ticket_matches_its_transmitted_mass() {
        let d = Distribution::from_weights(&[0.001, 2.0, 3.0, 0.0, 0.9]).unwrap();
        let q = SparseProposal::from_topk(&d, 3).unwrap();
        let remote = SparseProposal::decode_entries(5, &q.encode_entries()).unwrap();
        let mut counts = vec![0_u32; 5];
        for ticket in 0..MASS {
            counts[remote.sample_ticket(ticket as u16).unwrap().token()] += 1;
        }
        for &(id, m) in q.entries() {
            assert_eq!(counts[id as usize], m as u32);
        }
        assert_eq!(counts.iter().sum::<u32>(), MASS);
        assert!(q.sample_ticket(u16::MAX).is_err());
    }
    #[test]
    fn reject_invalid_packets_and_distribution_inputs() {
        assert!(SparseProposal::from_masses(2, vec![(0, 1)]).is_err());
        assert!(SparseProposal::from_masses(2, vec![(0, 3), (0, 65532)]).is_err());
        assert!(SparseProposal::from_masses(2, vec![(2, 65535)]).is_err());
        assert!(SparseProposal::decode_entries(2, &[0; 5]).is_err());
        for bad in [
            vec![],
            vec![0.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![-1.0],
        ] {
            assert!(Distribution::from_weights(&bad).is_err());
        }
        let p = Distribution::from_weights(&[1.0, 0.0]).unwrap();
        let q = SparseProposal::from_masses(2, vec![(1, 65535)]).unwrap();
        let c = q.sample_ticket(0).unwrap();
        let r = verify(&p, &c, 0.0, 0.5).unwrap();
        assert!(!r.accepted && r.token == 0);
        assert!(verify(&p, &c, f64::NAN, 0.5).is_err());
        assert!(verify(&p, &c, 1.0, 0.5).is_err());
    }
}
