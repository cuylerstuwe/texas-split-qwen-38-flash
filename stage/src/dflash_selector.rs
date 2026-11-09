//! DFlash2 adjacent-candidate lattice and host conditional proposal selection.
//! Mathematical contract follows pinned Sushi dflash.zig:1553-1807 (MIT;
//! NOTICE.txt). New Rust implementation; no MLX runtime is used here.
use crate::{
    bf16_linear::{HEADER_SHA, MODEL, Matrix},
    embedding::{HIDDEN, VOCAB},
    projection::sha,
    rounding::bf16_from_f64,
    speculative::{Distribution, MASS, SparseProposal},
    weights::Shard,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

pub const RANK: usize = 256;
pub const TOPK: usize = 16;
pub const MAX_DRAFTS: usize = 15;
fn f(v: u16) -> f64 {
    bf16::from_bits(v).to_f64()
}
fn round(v: f64) -> f64 {
    bf16_from_f64(v).to_f64()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidates {
    pub vocab: usize,
    pub rows: usize,
    pub width: usize,
    pub ids: Vec<u32>,
    pub unary: Vec<f32>,
}
impl Candidates {
    /// Deterministic low-token-ID tie rule is our explicit proposal policy.
    /// Upstream argpartition ordering/ties need not be identical. Final target
    /// verification must always use the actual resulting proposal law.
    pub fn from_logits(logits: &[u16], vocab: usize, width: usize) -> Result<Self> {
        ensure!(
            (1..=VOCAB).contains(&vocab) && !logits.is_empty() && logits.len() % vocab == 0,
            "candidate logit shape"
        );
        let rows = logits.len() / vocab;
        ensure!(
            (1..=MAX_DRAFTS).contains(&rows) && (1..=TOPK.min(vocab)).contains(&width),
            "candidate bounds"
        );
        ensure!(
            logits.iter().all(|&x| bf16::from_bits(x).is_finite()),
            "candidate logits finite"
        );
        let mut ids = vec![];
        let mut unary = vec![];
        for row in logits.chunks_exact(vocab) {
            let cmp = |a: &usize, b: &usize| f(row[*b]).total_cmp(&f(row[*a])).then(a.cmp(b));
            let mut order = (0..vocab).collect::<Vec<_>>();
            order.select_nth_unstable_by(width - 1, cmp);
            order.truncate(width);
            order.sort_unstable_by(cmp);
            for id in order {
                ids.push(id as u32);
                unary.push(f(row[id]) as f32);
            }
        }
        let out = Self {
            vocab,
            rows,
            width,
            ids,
            unary,
        };
        out.validate()?;
        Ok(out)
    }
    /// R58: candidates from per-row top lists already ranked by BF16-rounded logit (ties to the
    /// lower id) with their BF16 values (MetalModel::head_top), i.e. from_logits' result.
    pub fn from_top(tops: &[Vec<(u32, f32)>], vocab: usize) -> Result<Self> {
        ensure!(!tops.is_empty() && (1..=MAX_DRAFTS).contains(&tops.len()), "candidate rows");
        let width = tops[0].len();
        ensure!((1..=TOPK).contains(&width) && tops.iter().all(|t| t.len() == width), "candidate width");
        let out = Self {
            vocab,
            rows: tops.len(),
            width,
            ids: tops.iter().flatten().map(|x| x.0).collect(),
            unary: tops.iter().flatten().map(|x| x.1).collect(),
        };
        out.validate()?;
        Ok(out)
    }
    /// R58: the same candidates as from_logits on BF16-rounded logits (top `width` per row by
    /// BF16 value, ties to the lower id), from FP32 logits in one pass per row (no sort, no BF16
    /// copy of the vocabulary rows).
    pub fn from_f32_top(logits: &[Vec<f32>], width: usize) -> Result<Self> {
        ensure!(!logits.is_empty() && (1..=MAX_DRAFTS).contains(&logits.len()) && (1..=TOPK).contains(&width), "candidate bounds");
        let vocab = logits[0].len();
        ensure!((1..=VOCAB).contains(&vocab) && logits.iter().all(|r| r.len() == vocab), "candidate logit shape");
        let tops: Vec<Vec<(u32, f64)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = logits
                .iter()
                .map(|row| {
                    sc.spawn(move || {
                        let mut top: Vec<(u32, f64)> = Vec::with_capacity(width + 1);
                        let better = |a: (u32, f64), b: (u32, f64)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
                        for (i, &x) in row.iter().enumerate() {
                            let c = (i as u32, bf16::from_f32(x).to_f64());
                            if top.len() == width && !better(c, top[width - 1]) {
                                continue;
                            }
                            let at = top.iter().position(|&t| better(c, t)).unwrap_or(top.len());
                            top.insert(at, c);
                            top.truncate(width);
                        }
                        top
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("candidate thread")).collect()
        });
        let (mut ids, mut unary) = (vec![], vec![]);
        for top in tops {
            for (id, v) in top {
                ids.push(id);
                unary.push(v as f32);
            }
        }
        let out = Self { vocab, rows: logits.len(), width, ids, unary };
        out.validate()?;
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=VOCAB).contains(&self.vocab)
                && (1..=MAX_DRAFTS).contains(&self.rows)
                && (1..=TOPK.min(self.vocab)).contains(&self.width),
            "candidate bounds"
        );
        ensure!(
            self.ids.len() == self.rows * self.width && self.unary.len() == self.ids.len(),
            "candidate array lengths"
        );
        ensure!(
            self.ids.iter().all(|&i| (i as usize) < self.vocab)
                && self.unary.iter().all(|x| x.is_finite()),
            "candidate IDs/unary values"
        );
        for row in self.ids.chunks_exact(self.width) {
            ensure!(
                row.iter().copied().collect::<BTreeSet<_>>().len() == row.len(),
                "duplicate candidate in one row"
            );
        }
        Ok(())
    }
}

pub struct Codebooks {
    ids: Vec<u32>,
    pred: Vec<u16>,
    succ: Vec<u16>,
    pub provenance: Value,
}
impl Codebooks {
    pub fn load(anchor: u32, c: &Candidates) -> Result<Self> {
        c.validate()?;
        ensure!(
            c.vocab == VOCAB && (anchor as usize) < VOCAB,
            "real selector vocabulary/anchor"
        );
        let ids = std::iter::once(anchor)
            .chain(c.ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "selector checkpoint pin"
        );
        let mut arrays = vec![];
        let mut sources = vec![];
        for name in [
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        ] {
            let t = s.tensors.get(name).context("selector codebook absent")?;
            ensure!(
                t.dtype == "BF16" && t.shape == [VOCAB as u64, RANK as u64],
                "selector codebook shape"
            );
            let mut values = vec![];
            let mut rowpins = vec![];
            for &id in &ids {
                let raw = s.read_range(name, id as u64 * RANK as u64 * 2, RANK as u64 * 2)?;
                rowpins.push(json!({"id":id,"sha256":sha(&raw)}));
                values.extend(
                    raw.chunks_exact(2)
                        .map(|b| u16::from_le_bytes(b.try_into().unwrap())),
                );
            }
            ensure!(
                values.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "selector codebook finite"
            );
            arrays.push(values);
            sources.push(json!({"tensor":name,"header_sha256":s.header_sha256,"row_bytes":RANK*2,"rows":rowpins}));
        }
        let succ = arrays.pop().unwrap();
        let pred = arrays.pop().unwrap();
        Ok(Self {
            ids,
            pred,
            succ,
            provenance: json!({"file":MODEL,"sources":sources,"read_selected_rows_only":true}),
        })
    }
    fn row(&self, id: u32, predecessor: bool) -> Result<&[u16]> {
        let i = self
            .ids
            .binary_search(&id)
            .map_err(|_| anyhow::anyhow!("missing selector token row"))?;
        let source = if predecessor { &self.pred } else { &self.succ };
        Ok(&source[i * RANK..(i + 1) * RANK])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lattice {
    pub candidates: Candidates,
    pub anchor_id: u32,
    pub anchor_edges: Vec<f32>,
    /// [draft row - 1, previous candidate index, next candidate index].
    pub transitions: Vec<f32>,
}
impl Lattice {
    pub fn validate(&self) -> Result<()> {
        let c = &self.candidates;
        c.validate()?;
        ensure!(
            (self.anchor_id as usize) < c.vocab
                && self.anchor_edges.len() == c.width
                && self.transitions.len() == (c.rows - 1) * c.width * c.width,
            "lattice dimensions"
        );
        ensure!(
            self.anchor_edges
                .iter()
                .chain(&self.transitions)
                .all(|x| x.is_finite()),
            "lattice edges finite"
        );
        Ok(())
    }
    fn scores(&self, row: usize, previous: Option<usize>) -> Result<Vec<f32>> {
        self.validate()?;
        let c = &self.candidates;
        ensure!(row < c.rows, "conditional row");
        ensure!(
            (row == 0 && previous.is_none()) || (row > 0 && previous.is_some_and(|i| i < c.width)),
            "conditional predecessor"
        );
        let edge = if row == 0 {
            &self.anchor_edges[..]
        } else {
            let start = ((row - 1) * c.width + previous.unwrap()) * c.width;
            &self.transitions[start..start + c.width]
        };
        let scores = edge
            .iter()
            .zip(&c.unary[row * c.width..(row + 1) * c.width])
            .map(|(&e, &u)| e + u)
            .collect::<Vec<_>>();
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "selector scores finite"
        );
        Ok(scores)
    }
    pub fn conditional(
        &self,
        row: usize,
        previous: Option<usize>,
        temperature: f32,
    ) -> Result<Distribution> {
        ensure!(
            temperature.is_finite() && temperature > 0.,
            "conditional temperature"
        );
        // Preserve the source FP32 unary+edge addition and exponential inputs;
        // normalize the resulting finite weights in FP64 for the explicit law.
        let scores = self.scores(row, previous)?;
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Distribution::from_weights(
            &scores
                .iter()
                .map(|s| f64::from(((s - max) / temperature).exp()))
                .collect::<Vec<_>>(),
        )
    }
    /// Raw array payload; transport framing, epoch/prefix binding and ownership
    /// are deliberately not implied by these bytes.
    pub fn encode_payload(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(self
            .candidates
            .ids
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .chain(
                self.candidates
                    .unary
                    .iter()
                    .chain(&self.anchor_edges)
                    .chain(&self.transitions)
                    .flat_map(|x| x.to_le_bytes()),
            )
            .collect())
    }
    pub fn decode_payload(
        vocab: usize,
        rows: usize,
        width: usize,
        anchor_id: u32,
        raw: &[u8],
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_DRAFTS).contains(&rows) && (1..=TOPK).contains(&width),
            "payload shape bounds"
        );
        let count = rows * width;
        ensure!(
            raw.len() == 4 * (2 * count + width + (rows - 1) * width * width),
            "payload length"
        );
        let ids = raw[..count * 4]
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let values = raw[count * 4..]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let out = Self {
            candidates: Candidates {
                vocab,
                rows,
                width,
                ids,
                unary: values[..count].to_vec(),
            },
            anchor_id,
            anchor_edges: values[count..count + width].to_vec(),
            transitions: values[count + width..].to_vec(),
        };
        out.validate()?;
        Ok(out)
    }
    /// Draw each next token from the SAME integer-mass conditional proposal
    /// returned for verification. Greedy is represented as a one-point q.
    /// `tickets` are independent uniform integers 0..65535 supplied by caller.
    pub fn trace(&self, temperature: f32, tickets: &[u16], eos: Option<u32>) -> Result<Vec<Step>> {
        self.validate()?;
        let c = &self.candidates;
        ensure!(
            temperature.is_finite()
                && temperature >= 0.
                && tickets.len() == c.rows
                && tickets.iter().all(|&t| u32::from(t) < MASS),
            "trace sampling inputs"
        );
        ensure!(eos.is_none_or(|id| (id as usize) < c.vocab), "trace EOS");
        let mut result = vec![];
        let mut previous = None;
        for (row, &ticket) in tickets.iter().enumerate() {
            let distribution = self.conditional(
                row,
                previous,
                if temperature == 0. { 1. } else { temperature },
            )?;
            let p = distribution.probabilities();
            let rowids = &c.ids[row * c.width..(row + 1) * c.width];
            let proposal = if temperature == 0. {
                let scores = self.scores(row, previous)?;
                let mut choice = 0;
                for i in 1..c.width {
                    if scores[i] > scores[choice] {
                        choice = i;
                    }
                }
                SparseProposal::from_masses(c.vocab, vec![(rowids[choice], MASS as u16)])?
            } else {
                SparseProposal::from_sparse_weights(
                    c.vocab,
                    &rowids
                        .iter()
                        .zip(p)
                        .map(|(&id, &p)| (id as usize, p))
                        .collect::<Vec<_>>(),
                )?
            };
            let token = proposal.sample_ticket(ticket)?.token() as u32;
            let choice = rowids
                .iter()
                .position(|&id| id == token)
                .context("sample outside candidate row")?;
            let max_quantization_tv = if temperature == 0. {
                0.
            } else {
                rowids
                    .iter()
                    .zip(p)
                    .map(|(&id, &v)| {
                        let mass = proposal
                            .entries()
                            .iter()
                            .find(|&&(i, _)| i == id)
                            .map_or(0, |x| x.1);
                        (v - f64::from(mass) / f64::from(MASS)).abs()
                    })
                    .sum::<f64>()
                    * 0.5
            };
            result.push(Step {
                row,
                previous_choice: previous,
                token,
                choice,
                proposal,
                reference_probabilities: p.to_vec(),
                quantization_tv: max_quantization_tv,
            });
            previous = Some(choice);
            if eos == Some(token) {
                break;
            }
        }
        Ok(result)
    }
}
pub struct Step {
    pub row: usize,
    pub previous_choice: Option<usize>,
    pub token: u32,
    pub choice: usize,
    pub proposal: SparseProposal,
    pub reference_probabilities: Vec<f64>,
    pub quantization_tv: f64,
}
pub struct Reference {
    pub projected: Vec<u16>,
    pub lattice: Lattice,
    pub provenance: Value,
}
/// The adjacent-candidate lattice from projected hidden rows and the candidates' codebook rows
/// (reference() arithmetic: BF16 product and edge boundaries, FP64 dot products).
fn build_lattice(projected: &[u16], codebooks: &Codebooks, candidates: Candidates, anchor: u32) -> Result<Lattice> {
    let mut anchor_edges = vec![];
    let mut transitions = vec![];
    for row in 0..candidates.rows {
        let predecessors = if row == 0 {
            vec![anchor]
        } else {
            candidates.ids[(row - 1) * candidates.width..row * candidates.width].to_vec()
        };
        for prev in predecessors {
            let pred = codebooks.row(prev, true)?;
            let product = pred
                .iter()
                .zip(&projected[row * RANK..(row + 1) * RANK])
                .map(|(&p, &h)| round(f(p) * f(h)))
                .collect::<Vec<_>>();
            for &id in &candidates.ids[row * candidates.width..(row + 1) * candidates.width] {
                let edge = round(
                    product
                        .iter()
                        .zip(codebooks.row(id, false)?)
                        .map(|(&a, &b)| a * f(b))
                        .sum(),
                ) as f32;
                if row == 0 {
                    anchor_edges.push(edge)
                } else {
                    transitions.push(edge)
                }
            }
        }
    }
    let lattice = Lattice {
        candidates,
        anchor_id: anchor,
        anchor_edges,
        transitions,
    };
    lattice.validate()?;
    Ok(lattice)
}

/// R58: the selector with its weights held across calls. Nothing is redone per call: no checkpoint
/// header parse, no file open per row, no projection reload, no row hashing. The projection is
/// converted to f32 once, and codebook rows come by positioned reads on a persistent handle onto
/// the pinned checkpoint. The arithmetic is reference()'s (the same FP64 dot products, row by row),
/// so the lattice, and with it the proposal law, is identical.
pub struct SelectorCache {
    projection: Vec<f32>,
    file: std::fs::File,
    offsets: [u64; 2],
    provenance: Value,
    /// R58: both codebooks held in memory (2 x VOCAB x RANK BF16, ~254 MB), so a round does not
    /// depend on the page cache: at 64-128K contexts the positioned reads were evicted and the
    /// selector grew 0.9 -> 6.6 ms per round. QWEN_R58_SELECTOR_MEM=0 keeps the reads.
    resident: Option<[Vec<u16>; 2]>,
}
impl SelectorCache {
    pub fn load() -> Result<Self> {
        let p = Matrix::load("candidate_selector.hidden_projection.weight")?;
        ensure!(p.n == RANK && p.k == HIDDEN, "selector projection geometry");
        let s = Shard::open(Path::new(MODEL))?;
        ensure!(
            s.header_sha256 == HEADER_SHA && s.file_bytes == 1_206_140_008,
            "selector checkpoint pin"
        );
        let mut offsets = [0u64; 2];
        for (i, name) in ["candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"].iter().enumerate() {
            let t = s.tensors.get(*name).context("selector codebook absent")?;
            ensure!(
                t.dtype == "BF16" && t.shape == [VOCAB as u64, RANK as u64],
                "selector codebook shape"
            );
            offsets[i] = s.data_start + t.data_offsets[0];
        }
        let file = std::fs::File::open(&s.path)?;
        ensure!(file.metadata()?.len() == s.file_bytes, "selector checkpoint size");
        let resident = if std::env::var("QWEN_R58_SELECTOR_MEM").as_deref() != Ok("0") {
            use std::os::unix::fs::FileExt;
            let mut books = [vec![], vec![]];
            for (k, b) in books.iter_mut().enumerate() {
                let mut raw = vec![0u8; VOCAB * RANK * 2];
                file.read_exact_at(&mut raw, offsets[k])?;
                *b = raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
            }
            Some(books)
        } else {
            None
        };
        Ok(Self {
            projection: p.values.iter().map(|&v| bf16::from_bits(v).to_f32()).collect(),
            file,
            offsets,
            resident,
            provenance: json!({"projection":p.provenance,"file":MODEL,"header_sha256":HEADER_SHA,"cached":true,
                "arithmetic":"reference(): FP64 projection/dot, BF16 boundaries"}),
        })
    }
    pub fn reference(&self, hidden_with_anchor: &[u16], candidates: Candidates, anchor: u32) -> Result<Reference> {
        use std::os::unix::fs::FileExt;
        candidates.validate()?;
        ensure!(
            hidden_with_anchor.len() == (candidates.rows + 1) * HIDDEN,
            "selector hidden rows include anchor"
        );
        ensure!(
            candidates.vocab == VOCAB && (anchor as usize) < VOCAB,
            "real selector vocabulary/anchor"
        );
        let x: Vec<f64> = hidden_with_anchor[HIDDEN..].iter().map(|&v| bf16::from_bits(v).to_f64()).collect();
        // one thread per row; each element is the same sequential FP64 dot product as reference()
        let projected: Vec<u16> = std::thread::scope(|sc| {
            let hs: Vec<_> = x
                .chunks_exact(HIDDEN)
                .map(|xx| {
                    let w = &self.projection;
                    sc.spawn(move || {
                        w.chunks_exact(HIDDEN)
                            .map(|ww| {
                                let mut s = 0.;
                                for (&a, &b) in xx.iter().zip(ww) {
                                    s += a * f64::from(b);
                                }
                                bf16_from_f64(s).to_bits()
                            })
                            .collect::<Vec<u16>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().expect("projection thread")).collect()
        });
        let ids: Vec<u32> = std::iter::once(anchor).chain(candidates.ids.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect();
        let mut arrays = [Vec::with_capacity(ids.len() * RANK), Vec::with_capacity(ids.len() * RANK)];
        let mut raw = vec![0u8; RANK * 2];
        for (k, arr) in arrays.iter_mut().enumerate() {
            for &id in &ids {
                if let Some(books) = &self.resident {
                    let at = id as usize * RANK;
                    arr.extend_from_slice(&books[k][at..at + RANK]);
                    continue;
                }
                self.file.read_exact_at(&mut raw, self.offsets[k] + id as u64 * RANK as u64 * 2)?;
                arr.extend(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
            }
        }
        let [pred, succ] = arrays;
        ensure!(
            pred.iter().chain(&succ).all(|&v| bf16::from_bits(v).is_finite()),
            "selector codebook finite"
        );
        let codebooks = Codebooks { ids, pred, succ, provenance: self.provenance.clone() };
        let lattice = build_lattice(&projected, &codebooks, candidates, anchor)?;
        Ok(Reference { projected, lattice, provenance: self.provenance.clone() })
    }
}

pub fn reference(
    hidden_with_anchor: &[u16],
    candidates: Candidates,
    anchor: u32,
) -> Result<Reference> {
    candidates.validate()?;
    ensure!(
        hidden_with_anchor.len() == (candidates.rows + 1) * HIDDEN,
        "selector hidden rows include anchor"
    );
    let projection = Matrix::load("candidate_selector.hidden_projection.weight")?;
    ensure!(
        projection.n == RANK && projection.k == HIDDEN,
        "selector projection geometry"
    );
    let projected = projection
        .reference(&hidden_with_anchor[HIDDEN..])?
        .into_iter()
        .map(|x| bf16_from_f64(x).to_bits())
        .collect::<Vec<_>>();
    let codebooks = Codebooks::load(anchor, &candidates)?;
    let lattice = build_lattice(&projected, &codebooks, candidates, anchor)?;
    Ok(Reference {
        projected,
        lattice,
        provenance: json!({"projection":projection.provenance,"codebooks":codebooks.provenance,"arithmetic":"FP64 projection/dot; BF16 projection, product and edge boundaries; FP32 unary+edge; conditional exponential FP32 with FP64 normalization","tie_policy":"descending unary, lower token ID on ties; upstream argpartition order is not assumed identical","sampling_policy":"named integer-mass q arm; sample from and verify with the same masses","gpu_opened":false}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Lattice {
        Lattice {
            candidates: Candidates {
                vocab: 8,
                rows: 2,
                width: 2,
                ids: vec![3, 5, 1, 6],
                unary: vec![10., 9., 5., 4.],
            },
            anchor_id: 2,
            anchor_edges: vec![2., 2.],
            transitions: vec![1., 6., 8., -3.],
        }
    }
    #[test]
    fn conditional_path_edges_and_actual_mass_are_preserved() {
        let l = fixture();
        let greedy = l.trace(0., &[0, 0], None).unwrap();
        assert_eq!(greedy.iter().map(|s| s.token).collect::<Vec<_>>(), [3, 6]);
        assert_eq!(greedy[1].proposal.entries(), &[(6, 65535)]);
        let a = l.trace(1., &[0, 0], None).unwrap();
        let b = l.trace(1., &[65534, 0], None).unwrap();
        assert_ne!(a[0].choice, b[0].choice);
        assert_ne!(a[1].proposal.entries(), b[1].proposal.entries());
        for path in [&a, &b] {
            for s in path {
                assert_eq!(
                    s.proposal
                        .entries()
                        .iter()
                        .map(|x| u32::from(x.1))
                        .sum::<u32>(),
                    MASS
                );
                assert!(
                    s.proposal
                        .entries()
                        .iter()
                        .any(|&(id, m)| id == s.token && m > 0)
                );
                assert!(s.quantization_tv < 0.00002);
            }
        }
        let raw = l.encode_payload().unwrap();
        let copy = Lattice::decode_payload(8, 2, 2, 2, &raw).unwrap();
        assert_eq!(raw, copy.encode_payload().unwrap());
        assert_eq!(l.trace(0., &[0, 0], Some(3)).unwrap().len(), 1);
        assert!(l.conditional(0, Some(0), 1.).is_err());
        assert!(l.conditional(1, None, 1.).is_err());
        assert!(Lattice::decode_payload(8, 2, 2, 2, &raw[..raw.len() - 1]).is_err());
        assert!(l.trace(1., &[65535, 0], None).is_err());
        let mut bad = l.clone();
        bad.candidates.ids[1] = bad.candidates.ids[0];
        assert!(bad.validate().is_err());
        let mut bad = l.clone();
        bad.transitions[0] = f32::NAN;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn full_vocabulary_candidate_ties_have_explicit_policy() {
        let x = [1., 2., 2., -2., 3., 0., 1., 0.].map(|x| bf16::from_f32(x).to_bits());
        let c = Candidates::from_logits(&x, 8, 3).unwrap();
        assert_eq!(c.ids, [4, 1, 2]);
        assert!(Candidates::from_logits(&x, 8, 9).is_err());
        assert!(SparseProposal::from_sparse_weights(8, &[(1, 0.), (2, 0.)]).is_err());
        assert!(SparseProposal::from_sparse_weights(8, &[(1, 1.), (1, 2.)]).is_err());
    }
}
