//! Bounded synchronous reference for recursive speculative refill.
//!
//! Models are ordered smallest proposer -> reviewer -> final target. A lower
//! call returns a fixed number of tokens (or stops at EOS), including its own
//! corrections/bonuses. Each token carries the generating level's conditional
//! law. This is a CPU protocol reference, not a GPU scheduler or wire protocol.
use crate::speculative::Distribution;
use anyhow::{Result, ensure};
use serde::Serialize;

pub const MAX_LEVELS: usize = 4;
pub const MAX_BLOCK: usize = 64;
pub const MAX_CONTEXT: usize = 4096;

/// Device adapters must keep separate causal state per model. score_chain
/// restores/ingests `prefix`, scores before each candidate, and represents all
/// candidate tokens in its private state. The optional bonus row is at the end.
/// restore must discard/replay any divergent suffix; it cannot copy another
/// model's numerical cache merely because its token prefix matches.
pub trait Backend {
    fn levels(&self) -> usize;
    fn vocab(&self) -> usize;
    fn score_chain(
        &mut self,
        level: usize,
        prefix: &[u32],
        candidates: &[u32],
        bonus_row: bool,
    ) -> Result<Vec<Distribution>>;
    fn restore(&mut self, level: usize, prefix: &[u32]) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ChoiceKind {
    Base,
    Acceptance,
    Correction,
    Bonus,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ChoiceEvent {
    pub level: usize,
    pub kind: ChoiceKind,
}

/// Production adapters would supply fresh conditional randomness. The finite
/// screen instead enumerates every positive-probability choice exactly.
pub trait Choose {
    fn choose(&mut self, law: &Distribution, event: ChoiceEvent) -> Result<usize>;
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub rounds: [usize; MAX_LEVELS],
    pub scored_rows: [usize; MAX_LEVELS],
    pub returned_tokens: [usize; MAX_LEVELS],
    pub accepted: [usize; MAX_LEVELS],
    pub rejected: [usize; MAX_LEVELS],
    pub bonus: [usize; MAX_LEVELS],
    pub discarded_after_rejection: [usize; MAX_LEVELS],
    pub restore_calls: [usize; MAX_LEVELS],
}

#[derive(Clone, Debug)]
pub struct Token {
    id: u32,
    law: Distribution,
}
impl Token {
    pub fn id(&self) -> u32 {
        self.id
    }
    /// Law at this token's exact prefix, including all model-side filtering.
    /// This is a full dense reference, not a compressed q transport format.
    pub fn law(&self) -> &Distribution {
        &self.law
    }
}

pub struct Cascade<'a, B: Backend> {
    backend: &'a mut B,
    windows: Vec<usize>,
    levels: usize,
    vocab: usize,
    eos: Option<u32>,
    stats: Stats,
}
impl<'a, B: Backend> Cascade<'a, B> {
    pub fn new(backend: &'a mut B, windows: &[usize], eos: Option<u32>) -> Result<Self> {
        let levels = backend.levels();
        let vocab = backend.vocab();
        ensure!(
            (2..=MAX_LEVELS).contains(&levels) && (1..=1_048_576).contains(&vocab),
            "cascade model bounds"
        );
        ensure!(
            windows.len() + 1 == levels && windows.iter().all(|w| (1..=MAX_BLOCK).contains(w)),
            "cascade window bounds"
        );
        ensure!(
            eos.is_none_or(|id| (id as usize) < vocab),
            "cascade EOS bounds"
        );
        Ok(Self {
            backend,
            windows: windows.to_vec(),
            levels,
            vocab,
            eos,
            stats: Stats::default(),
        })
    }
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
    pub fn generate<C: Choose>(
        &mut self,
        prefix: &[u32],
        count: usize,
        chooser: &mut C,
    ) -> Result<Vec<Token>> {
        ensure!(
            (1..=MAX_BLOCK).contains(&count) && prefix.len() + count <= MAX_CONTEXT,
            "cascade request bounds"
        );
        ensure!(
            prefix.iter().all(|&x| (x as usize) < self.vocab),
            "cascade prefix token"
        );
        self.fill(self.levels - 1, prefix, count, chooser)
    }
    fn score(
        &mut self,
        level: usize,
        prefix: &[u32],
        candidates: &[u32],
        bonus: bool,
    ) -> Result<Vec<Distribution>> {
        let rows = self.backend.score_chain(level, prefix, candidates, bonus)?;
        ensure!(
            rows.len() == candidates.len() + usize::from(bonus)
                && rows.iter().all(|r| r.probabilities().len() == self.vocab),
            "cascade score geometry"
        );
        self.stats.scored_rows[level] += rows.len();
        Ok(rows)
    }
    fn draw<C: Choose>(
        &self,
        chooser: &mut C,
        law: &Distribution,
        level: usize,
        kind: ChoiceKind,
    ) -> Result<usize> {
        let i = chooser.choose(law, ChoiceEvent { level, kind })?;
        ensure!(
            i < law.probabilities().len() && law.probabilities()[i] > 0.,
            "chooser selected impossible outcome"
        );
        Ok(i)
    }
    fn stopped(&self, id: u32) -> bool {
        self.eos == Some(id)
    }
    fn fill<C: Choose>(
        &mut self,
        level: usize,
        prefix: &[u32],
        count: usize,
        chooser: &mut C,
    ) -> Result<Vec<Token>> {
        let mut context = prefix.to_vec();
        let mut output = Vec::with_capacity(count);
        while output.len() < count {
            self.stats.rounds[level] += 1;
            if level == 0 {
                let law = self.score(0, &context, &[], true)?.pop().unwrap();
                let id = self.draw(chooser, &law, 0, ChoiceKind::Base)? as u32;
                output.push(Token { id, law });
                context.push(id);
                // The sampled token is pending in this model's cache until the
                // next score/restore. Do not pretend sampling ingested it.
                if self.stopped(id) {
                    break;
                }
                continue;
            }
            let remaining = count - output.len();
            let k = self.windows[level - 1].min(remaining);
            let draft = self.fill(level - 1, &context, k, chooser)?;
            ensure!(
                !draft.is_empty() && draft.len() <= k,
                "empty/oversized inner refill"
            );
            let ids: Vec<_> = draft.iter().map(Token::id).collect();
            ensure!(
                draft.len() == k || self.stopped(*ids.last().unwrap()),
                "short inner refill without EOS"
            );
            let bonus = remaining > draft.len() && !self.stopped(*ids.last().unwrap());
            let rows = self.score(level, &context, &ids, bonus)?;
            let mut rejected = false;
            for (position, candidate) in draft.iter().enumerate() {
                let p = &rows[position];
                let q = &candidate.law;
                let qi = q.probabilities()[candidate.id as usize];
                ensure!(qi > 0., "inner token outside declared proposal support");
                let a = (p.probabilities()[candidate.id as usize] / qi).min(1.);
                let accepted = if a == 1. {
                    true
                } else if a == 0. {
                    false
                } else {
                    let coin = Distribution::from_weights(&[a, 1. - a])?;
                    self.draw(chooser, &coin, level, ChoiceKind::Acceptance)? == 0
                };
                let id = if accepted {
                    self.stats.accepted[level] += 1;
                    candidate.id
                } else {
                    self.stats.rejected[level] += 1;
                    self.stats.discarded_after_rejection[level] += draft.len() - position;
                    let residual = Distribution::from_weights(
                        &p.probabilities()
                            .iter()
                            .zip(q.probabilities())
                            .map(|(p, q)| (p - q).max(0.))
                            .collect::<Vec<_>>(),
                    )?;
                    let correction =
                        self.draw(chooser, &residual, level, ChoiceKind::Correction)? as u32;
                    // context includes only the accepted prefix. The correction
                    // is pending; every lower private cache loses its descendants.
                    for owner in 0..=level {
                        self.backend.restore(owner, &context)?;
                        self.stats.restore_calls[owner] += 1;
                    }
                    rejected = true;
                    correction
                };
                output.push(Token { id, law: p.clone() });
                context.push(id);
                if self.stopped(id) {
                    self.stats.returned_tokens[level] += output.len();
                    return Ok(output);
                }
                if rejected {
                    break;
                }
            }
            if !rejected && bonus && output.len() < count {
                let law = rows.last().unwrap().clone();
                let id = self.draw(chooser, &law, level, ChoiceKind::Bonus)? as u32;
                self.stats.bonus[level] += 1;
                output.push(Token { id, law });
                context.push(id);
                if self.stopped(id) {
                    break;
                }
            }
        }
        self.stats.returned_tokens[level] += output.len();
        Ok(output)
    }
}
