//! Bounded CPU reference for merging proposals from independent workers.
//!
//! This implements token-only target-sampled tree walking, NOT p/q rejection
//! sampling. Freeze the tree before consuming fresh target randomness. Each
//! callback must sample the actual target distribution at the supplied history,
//! including its path-specific filters. Proposal arrival order then affects
//! coverage/cost, not the emitted distribution. No GPU state is committed here.
use anyhow::{Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Policy identifies target weights, tokenizer, filtering and state semantics.
/// Epoch separates requests/restarts/commits, including identical-length ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PrefixStamp {
    epoch: u64,
    position: u64,
    history: [u8; 32],
    policy: [u8; 32],
}
impl PrefixStamp {
    pub fn from_tokens(epoch: u64, tokens: &[u32], policy: [u8; 32]) -> Self {
        let mut stamp = Self {
            epoch,
            position: 0,
            history: Sha256::digest(b"qwen-hybrid-prefix-v1").into(),
            policy,
        };
        stamp.append_history(tokens);
        stamp
    }
    fn append_history(&mut self, tokens: &[u32]) {
        for token in tokens {
            let mut h = Sha256::new();
            h.update(self.history);
            h.update(token.to_le_bytes());
            self.history = h.finalize().into();
        }
        self.position += tokens.len() as u64;
    }
    pub fn advance(&self, tokens: &[u32]) -> Result<Self> {
        ensure!(!tokens.is_empty(), "empty commit");
        ensure!(
            self.position.checked_add(tokens.len() as u64).is_some(),
            "position overflow"
        );
        let mut next = *self;
        next.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("epoch overflow"))?;
        next.append_history(tokens);
        Ok(next)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Producer {
    Cuda,
    Metal,
    Ane,
    Lookup,
}
impl Producer {
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Clone, Debug)]
pub struct CandidatePath {
    pub stamp: PrefixStamp,
    pub producer: Producer,
    /// Relative to the exact stamped prefix; includes all branch ancestors.
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Node {
    pub token: u32,
    pub parent: Option<usize>,
    pub source_mask: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateTree {
    stamp: PrefixStamp,
    vocab: u32,
    nodes: Vec<Node>,
    histories: Vec<Vec<u32>>,
    /// Slot0 = committed prefix; slot i+1 = history after node i.
    children: Vec<BTreeMap<u32, usize>>,
    fingerprint: [u8; 32],
    node_budget: usize,
    depth_budget: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Stop {
    Uncovered,
    Eos,
    Limit,
}

/// A plan for state publication, not evidence that device state is ready.
#[derive(Clone, Debug)]
pub struct Walk {
    tree_fingerprint: [u8; 32],
    tokens: Vec<u32>,
    state_path: Vec<usize>,
    pending: Option<u32>,
    stop: Stop,
}
impl Walk {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    /// Already represented/scored target nodes, in their ancestral order.
    pub fn state_path(&self) -> &[usize] {
        &self.state_path
    }
    /// Emitted from a valid target row but not yet incorporated in target state.
    pub fn pending(&self) -> Option<u32> {
        self.pending
    }
    pub fn stop(&self) -> Stop {
        self.stop
    }
}

impl CandidateTree {
    pub fn merge(
        stamp: PrefixStamp,
        vocab: u32,
        paths: &[CandidatePath],
        node_budget: usize,
        depth_budget: usize,
    ) -> Result<Self> {
        ensure!(vocab > 0, "empty vocabulary");
        ensure!(
            (1..=256).contains(&node_budget) && (1..=64).contains(&depth_budget),
            "research tree bounds"
        );
        ensure!(paths.len() <= 1024, "too many submitted paths");
        let mut prefixes = BTreeMap::<Vec<u32>, u8>::new();
        for path in paths {
            ensure!(
                path.stamp == stamp,
                "stale or mismatched proposal prefix/policy"
            );
            ensure!(
                !path.tokens.is_empty() && path.tokens.len() <= depth_budget,
                "path length bounds"
            );
            ensure!(
                path.tokens.iter().all(|&t| t < vocab),
                "token outside vocabulary"
            );
            for length in 1..=path.tokens.len() {
                *prefixes.entry(path.tokens[..length].to_vec()).or_default() |= path.producer.bit();
                ensure!(prefixes.len() <= node_budget, "tree node budget exceeded");
            }
        }
        // Lexicographic paths give canonical IDs, with ancestors before children.
        // Canonicalization is for descriptors/replay, not a repeatability gate.
        let ids: BTreeMap<_, _> = prefixes
            .keys()
            .cloned()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        let mut nodes = Vec::with_capacity(ids.len());
        let mut histories = Vec::with_capacity(ids.len());
        let mut children = vec![BTreeMap::new(); ids.len() + 1];
        for (history, source_mask) in prefixes {
            let parent = if history.len() == 1 {
                None
            } else {
                Some(ids[&history[..history.len() - 1]])
            };
            let token = *history.last().unwrap();
            children[parent.map_or(0, |p| p + 1)].insert(token, nodes.len());
            nodes.push(Node {
                token,
                parent,
                source_mask,
            });
            histories.push(history);
        }
        let mut h = Sha256::new();
        h.update(b"qwen-hybrid-candidate-tree-v1");
        h.update(serde_json::to_vec(&stamp)?);
        h.update(vocab.to_le_bytes());
        h.update(serde_json::to_vec(&nodes)?);
        Ok(Self {
            stamp,
            vocab,
            nodes,
            histories,
            children,
            fingerprint: h.finalize().into(),
            node_budget,
            depth_budget,
        })
    }
    pub fn stamp(&self) -> PrefixStamp {
        self.stamp
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn history(&self, node: usize) -> Result<&[u32]> {
        self.histories
            .get(node)
            .map(|p| p.as_slice())
            .ok_or_else(|| anyhow::anyhow!("invalid node"))
    }
    pub fn parents(&self) -> Vec<Option<usize>> {
        self.nodes.iter().map(|n| n.parent).collect()
    }
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    /// The callback receives the exact relative context of an available target
    /// row. It must NOT pick a proposal because it arrived sooner. That is a
    /// different sampler. Empty trees provide the ordinary one-token fallback.
    pub fn walk(
        &self,
        current: PrefixStamp,
        max_new: usize,
        eos: Option<u32>,
        mut target_sample: impl FnMut(&[u32]) -> Result<u32>,
    ) -> Result<Walk> {
        ensure!(current == self.stamp, "stale target tree");
        ensure!(max_new > 0 && max_new <= 65, "walk output bound");
        ensure!(eos.is_none_or(|t| t < self.vocab), "invalid EOS");
        let mut out = Walk {
            tree_fingerprint: self.fingerprint,
            tokens: vec![],
            state_path: vec![],
            pending: None,
            stop: Stop::Limit,
        };
        let mut slot = 0;
        for _ in 0..max_new {
            let token = target_sample(&out.tokens)?;
            ensure!(token < self.vocab, "target sampled invalid token");
            out.tokens.push(token);
            if let Some(&child) = self.children[slot].get(&token) {
                out.state_path.push(child);
                slot = child + 1;
            } else {
                out.pending = Some(token);
            }
            if eos == Some(token) {
                out.stop = Stop::Eos;
                break;
            }
            if out.tokens.len() == max_new {
                break;
            }
            if out.pending.is_some() {
                out.stop = Stop::Uncovered;
                break;
            }
        }
        Ok(out)
    }

    /// Retain token-only descendants after a partial commit. Device caches,
    /// sampler probabilities and target features are NOT rebased by this call.
    pub fn rebase(&self, walk: &Walk) -> Result<Self> {
        ensure!(
            walk.tree_fingerprint == self.fingerprint,
            "walk belongs to another tree"
        );
        ensure!(walk.stop != Stop::Eos, "cannot continue after EOS");
        let stamp = self.stamp.advance(&walk.tokens)?;
        let mut paths = vec![];
        for (node, history) in self.nodes.iter().zip(&self.histories) {
            if history.len() <= walk.tokens.len() || !history.starts_with(&walk.tokens) {
                continue;
            }
            for producer in [
                Producer::Cuda,
                Producer::Metal,
                Producer::Ane,
                Producer::Lookup,
            ] {
                if node.source_mask & producer.bit() != 0 {
                    paths.push(CandidatePath {
                        stamp,
                        producer,
                        tokens: history[walk.tokens.len()..].to_vec(),
                    });
                }
            }
        }
        Self::merge(
            stamp,
            self.vocab,
            &paths,
            self.node_budget,
            self.depth_budget,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stamp() -> PrefixStamp {
        PrefixStamp::from_tokens(7, &[3, 4], [1; 32])
    }
    fn path(producer: Producer, tokens: &[u32]) -> CandidatePath {
        CandidatePath {
            stamp: stamp(),
            producer,
            tokens: tokens.to_vec(),
        }
    }
    #[test]
    fn merge_by_complete_prefix_not_token_and_ignore_arrival_order() {
        let paths = [
            path(Producer::Cuda, &[0, 1, 2]),
            path(Producer::Ane, &[2, 1, 2]),
            path(Producer::Metal, &[0, 1, 3]),
        ];
        let first = CandidateTree::merge(stamp(), 8, &paths, 256, 64).unwrap();
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let reordered = order.map(|i| paths[i].clone());
            assert_eq!(
                first,
                CandidateTree::merge(stamp(), 8, &reordered, 256, 64).unwrap()
            );
        }
        assert_eq!(first.nodes().len(), 7);
        assert_ne!(first.nodes()[2].parent, first.nodes()[6].parent);
        assert_eq!(
            first.nodes()[0].source_mask,
            Producer::Cuda.bit() | Producer::Metal.bit()
        );
    }
    #[test]
    fn stamp_binds_history_position_epoch_and_policy_and_streaming_chunks() {
        let full = PrefixStamp::from_tokens(9, &[3, 4, 5, 6, 7], [1; 32]);
        assert_eq!(
            stamp().advance(&[5]).unwrap().advance(&[6, 7]).unwrap(),
            full
        );
        let mut cases = vec![stamp(); 4];
        cases[0].epoch += 1;
        cases[1].position += 1;
        cases[2].history[3] ^= 1;
        cases[3].policy[5] ^= 1;
        for wrong in cases {
            let p = CandidatePath {
                stamp: wrong,
                ..path(Producer::Cuda, &[0])
            };
            assert!(CandidateTree::merge(stamp(), 8, &[p], 256, 64).is_err());
        }
        assert!(stamp().advance(&[]).is_err());
    }
    #[test]
    fn pending_token_and_partial_commit_do_not_retain_sibling_state() {
        let tree = CandidateTree::merge(
            stamp(),
            8,
            &[
                path(Producer::Cuda, &[0, 1, 2]),
                path(Producer::Ane, &[0, 3, 4]),
            ],
            256,
            64,
        )
        .unwrap();
        let w = tree.walk(stamp(), 1, None, |_| Ok(0)).unwrap();
        let tail = tree.rebase(&w).unwrap();
        assert_eq!(tail.nodes().len(), 4);
        assert!(tail.walk(stamp(), 1, None, |_| Ok(1)).is_err());
        let w = tree
            .walk(stamp(), 4, None, |h| Ok([0, 3, 7][h.len()]))
            .unwrap();
        assert_eq!(w.tokens(), &[0, 3, 7]);
        assert_eq!(w.state_path().len(), 2);
        assert_eq!(w.pending(), Some(7));
        assert_eq!(w.stop(), Stop::Uncovered);
        assert!(tree.rebase(&w).unwrap().nodes().is_empty());
        let sibling =
            CandidateTree::merge(stamp(), 8, &[path(Producer::Cuda, &[0, 1, 2])], 256, 64).unwrap();
        assert!(sibling.rebase(&w).is_err());
        let eos = tree
            .walk(stamp(), 4, Some(3), |h| Ok([0, 3][h.len()]))
            .unwrap();
        assert_eq!(eos.stop(), Stop::Eos);
        assert!(tree.rebase(&eos).is_err());
    }
    #[test]
    fn bounded_inputs_and_empty_fallback() {
        assert!(CandidateTree::merge(stamp(), 8, &[path(Producer::Cuda, &[])], 256, 64).is_err());
        assert!(CandidateTree::merge(stamp(), 8, &[path(Producer::Cuda, &[8])], 256, 64).is_err());
        assert!(CandidateTree::merge(stamp(), 8, &[path(Producer::Cuda, &[0, 1])], 1, 64).is_err());
        assert!(
            CandidateTree::merge(stamp(), 8, &[path(Producer::Cuda, &[0, 1])], 256, 1).is_err()
        );
        let tree = CandidateTree::merge(stamp(), 8, &[], 256, 64).unwrap();
        let w = tree.walk(stamp(), 9, None, |_| Ok(7)).unwrap();
        assert_eq!(w.tokens(), &[7]);
        assert!(w.state_path().is_empty());
        assert_eq!(w.pending(), Some(7));
        assert!(tree.walk(stamp(), 1, None, |_| Ok(8)).is_err());
    }
}
