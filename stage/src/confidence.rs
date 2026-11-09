//! CPU reference for two admissible ways to choose a rejection-sampling prefix.
//!
//! Confidence is a cost/coverage estimate, never an acceptance test. Actual
//! tokens still pass the original p/q verifier. The caller must supply confidence
//! BEFORE observing the token whose admission is being decided. Rust cannot
//! prove that an externally computed score obeys that information boundary.
//! A frozen, prior-round estimate permits global search across irregular costs;
//! a current-chain estimate requires irreversible, left-to-right admission.
//! The expected-length formula assumes no EOS or remaining-output truncation;
//! those limits need a separate cost estimate. A bad estimate can waste work,
//! but must never authorize a token without the declared verifier.
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct CycleCosts(Vec<f64>);

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Estimate {
    pub draft_tokens: usize,
    pub expected_committed: f64,
    pub cycle_cost: f64,
    pub committed_per_cost: f64,
}

impl CycleCosts {
    /// Entry k prices the WHOLE cycle with k draft tokens and a correction/bonus
    /// opportunity. Units must match throughout; do not mix component clocks.
    /// Include exposed drafting, transfer, verification, replay and contention.
    /// Entry0 is ordinary target decoding. Nonmonotonic hardware costs are valid.
    pub fn new(costs: Vec<f64>) -> Result<Self> {
        ensure!((1..=65).contains(&costs.len()), "cycle width bounds");
        ensure!(
            costs.iter().all(|c| c.is_finite() && *c > 0.),
            "positive finite cycle costs required"
        );
        Ok(Self(costs))
    }
    fn estimate(&self, draft_tokens: usize, expected_committed: f64) -> Estimate {
        let cost = self.0[draft_tokens];
        Estimate {
            draft_tokens,
            expected_committed,
            cycle_cost: cost,
            committed_per_cost: expected_committed / cost,
        }
    }
    /// Search every width using prefix survival estimates fixed independently
    /// of THIS round's candidate draws. Scores from current suffixes are NOT
    /// admissible here. The returned width must be fixed before those draws.
    pub fn choose_from_prior(&self, survival: &[f64]) -> Result<Estimate> {
        ensure!(
            survival.len() + 1 == self.0.len(),
            "survival width mismatch"
        );
        let (mut previous, mut expected) = (1., 1.);
        let mut best = self.estimate(0, expected);
        for (i, &s) in survival.iter().enumerate() {
            ensure!(
                s.is_finite() && (0.0..=previous).contains(&s),
                "invalid prefix survival"
            );
            previous = s;
            expected += s;
            let candidate = self.estimate(i + 1, expected);
            if candidate.committed_per_cost > best.committed_per_cost {
                best = candidate;
            }
        }
        Ok(best)
    }
}

#[derive(Debug)]
pub struct CausalPrefix {
    costs: CycleCosts,
    estimate: Estimate,
    survival: f64,
    stopped: bool,
}

impl CausalPrefix {
    pub fn new(costs: CycleCosts) -> Self {
        let estimate = costs.estimate(0, 1.);
        Self {
            costs,
            estimate,
            survival: 1.,
            stopped: false,
        }
    }
    /// Call before drawing the next candidate. It is safe to condition on
    /// earlier admitted draft tokens. A false return seals the prefix: later
    /// confidence cannot retrospectively revive it or remove earlier tokens.
    pub fn admit_next(&mut self, conditional_confidence: f64) -> Result<bool> {
        ensure!(!self.stopped, "prefix already sealed");
        ensure!(
            conditional_confidence.is_finite() && (0.0..=1.0).contains(&conditional_confidence),
            "invalid confidence"
        );
        let next = self.estimate.draft_tokens + 1;
        if next >= self.costs.0.len() {
            self.stopped = true;
            return Ok(false);
        }
        let survival = self.survival * conditional_confidence;
        let candidate = self
            .costs
            .estimate(next, self.estimate.expected_committed + survival);
        if candidate.committed_per_cost <= self.estimate.committed_per_cost {
            self.stopped = true;
            return Ok(false);
        }
        self.survival = survival;
        self.estimate = candidate;
        Ok(true)
    }
    pub fn estimate(&self) -> Estimate {
        self.estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_policy_cannot_reopen_after_a_cost_cliff() -> Result<()> {
        let costs = CycleCosts::new(vec![1., 2., 1. / 0.45])?;
        let mut causal = CausalPrefix::new(costs.clone());
        assert!(!causal.admit_next(0.8)?);
        assert!(causal.admit_next(0.9).is_err());
        assert_eq!(causal.estimate().draft_tokens, 0);
        // An older, independent estimate may cross the local performance dip.
        assert_eq!(costs.choose_from_prior(&[0.8, 0.72])?.draft_tokens, 2);
        assert_eq!(costs.choose_from_prior(&[0.8, 0.0])?.draft_tokens, 0);
        Ok(())
    }
    #[test]
    fn invalid_estimates_and_widths_are_rejected() -> Result<()> {
        for c in [vec![], vec![f64::NAN], vec![0.], vec![1.; 66]] {
            assert!(CycleCosts::new(c).is_err());
        }
        let c = CycleCosts::new(vec![1., 1.1, 1.2])?;
        for p in [
            vec![0.8],
            vec![0.8, 0.9],
            vec![1., f64::NAN],
            vec![1.1, 0.9],
        ] {
            assert!(c.choose_from_prior(&p).is_err());
        }
        let mut p = CausalPrefix::new(c);
        assert!(p.admit_next(-0.1).is_err());
        assert!(p.admit_next(0.9)?);
        assert!(p.admit_next(0.9)?);
        assert!(!p.admit_next(0.9)?);
        Ok(())
    }
}
