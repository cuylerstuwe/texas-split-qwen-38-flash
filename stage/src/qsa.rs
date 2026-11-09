//! Qwen target quantized sparse-attention indexer semantics and independent CPU
//! oracle. This is not the inference execution backend. Source: pinned Sushi
//! qwen4_qsa.zig/transformer.zig (MIT; NOTICE.txt), text-position branch.
use crate::{dense::Affine8, projection::Pack};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
pub const H: usize = 2560;
pub const D: usize = 128;
pub const HEADS: usize = 4;
pub const QK: usize = (HEADS + 1) * D;
pub const RATIO: usize = 4;
pub const BUDGET: usize = 2048;
pub const TOPK: usize = BUDGET / RATIO;
pub const ROPE: usize = 64;
pub const MAX_ROWS: usize = 16;
pub const MAX_POSITION: usize = 262144;
pub const SENTINEL: u32 = i32::MAX as u32;
fn num(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
fn bits(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
fn rounded(x: f64) -> f64 {
    num(bits(x))
}
fn finite(x: &[u16]) -> bool {
    x.iter().all(|&v| bf16::from_bits(v).is_finite())
}

pub struct Norms {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub provenance: serde_json::Value,
}
impl Norms {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.q.len() == D && self.k.len() == D && finite(&self.q) && finite(&self.k),
            "indexer norm geometry/finite"
        );
        Ok(())
    }
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        ensure!(layer < 48 && layer % 4 == 3, "target QSA layer");
        let mut provenance = vec![];
        let mut read = |kind: &str| -> Result<Vec<u16>> {
            let (t, b, p) = pack.tensor(&format!(
                "language_model.model.layers.{layer}.self_attn.indexer.{kind}_layernorm.weight"
            ))?;
            ensure!(
                t.dtype == "BF16" && t.shape == [D as u64],
                "indexer norm shape"
            );
            provenance.push(p);
            Ok(b.chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect())
        };
        let q = read("q")?;
        let k = read("k")?;
        let n = Self {
            q,
            k,
            provenance: json!(provenance),
        };
        n.validate()?;
        Ok(n)
    }
}
pub struct Weights {
    pub projection: Affine8,
    pub norms: Norms,
    pub layer: usize,
}
impl Weights {
    pub fn load(pack: &Pack, layer: usize) -> Result<Self> {
        let norms = Norms::load(pack, layer)?;
        let projection = Affine8::load(
            pack,
            &format!("language_model.model.layers.{layer}.self_attn.indexer.index_qk_proj"),
            H,
        )?;
        ensure!(projection.n == QK, "indexer projection width");
        Ok(Self {
            projection,
            norms,
            layer,
        })
    }
    pub fn provenance(&self) -> serde_json::Value {
        json!({"layer":self.layer,"projection":self.projection.provenance,"norms":self.norms.provenance})
    }
}
/// Only complete pooled blocks and the raw incomplete tail are committed.
/// Pooled block b rotates at absolute `base+4*b`; visibility uses cache indices.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub base: usize,
    pub len: usize,
    pub pooled: Vec<u16>,
    pub tail: Vec<u16>,
}
impl State {
    pub fn empty(base: usize) -> Result<Self> {
        let s = Self {
            base,
            len: 0,
            pooled: vec![],
            tail: vec![],
        };
        s.validate()?;
        Ok(s)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.base <= MAX_POSITION && self.len <= MAX_POSITION - self.base,
            "indexer position bounds"
        );
        ensure!(
            self.pooled.len() == self.len / RATIO * D
                && self.tail.len() == self.len % RATIO * D
                && finite(&self.pooled)
                && finite(&self.tail),
            "indexer contiguous history/finite"
        );
        Ok(())
    }
}
/// Explicit float32 frequencies/angles, as used by the text-position RoPE.
pub fn frequencies() -> [f32; ROPE / 2] {
    std::array::from_fn(|j| 10_000_000f64.powf(-((2 * j) as f64) / ROPE as f64) as f32)
}
fn norm(x: &[u16], w: &[u16]) -> Vec<u16> {
    let ss = x.iter().map(|&v| num(v) * num(v)).sum::<f64>();
    let inv = (ss / D as f64 + 1e-6).sqrt().recip();
    x.iter()
        .zip(w)
        .map(|(&x, &w)| bits(rounded(num(x) * inv) * num(w)))
        .collect()
}
/// Index queries use fast-rope's one final BF16 store. Pooled keys use BF16
/// cosine/sine tables and separate product/add boundaries in the source chain.
pub fn rotate(x: &[u16], position: usize, pooled: bool) -> Result<Vec<u16>> {
    ensure!(
        x.len() == D && position < MAX_POSITION && finite(x),
        "indexer RoPE input"
    );
    let mut out = x.to_vec();
    let f = frequencies();
    for j in 0..ROPE / 2 {
        let angle = position as f32 * f[j];
        let mut c = angle.cos() as f64;
        let mut s = angle.sin() as f64;
        let a = num(x[j]);
        let b = num(x[j + ROPE / 2]);
        if pooled {
            c = rounded(c);
            s = rounded(s);
            out[j] = bits(rounded(a * c) + rounded(-b * s));
            out[j + ROPE / 2] = bits(rounded(b * c) + rounded(a * s));
        } else {
            out[j] = bits(a * c - b * s);
            out[j + ROPE / 2] = bits(b * c + a * s);
        }
    }
    ensure!(finite(&out), "indexer RoPE finite output");
    Ok(out)
}
pub fn pool(raw: &[u16], norms: &Norms, position: usize) -> Result<Vec<u16>> {
    norms.validate()?;
    ensure!(
        raw.len() == RATIO * D && finite(raw),
        "indexer complete raw block"
    );
    let mean: Vec<_> = (0..D)
        .map(|d| bits((0..RATIO).map(|r| num(raw[r * D + d])).sum::<f64>() / RATIO as f64))
        .collect();
    rotate(&norm(&mean, &norms.k), position, true)
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Selection {
    pub offset: usize,
    pub rows: usize,
    pub total: usize,
    pub ids: Vec<u32>,
    pub counts: Vec<usize>,
}
impl Selection {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&self.rows)
                && self.offset.checked_add(self.rows) == Some(self.total)
                && self.total <= MAX_POSITION
                && self.ids.len() == self.rows * TOPK
                && self.counts.len() == self.rows,
            "indexer selection shape"
        );
        for r in 0..self.rows {
            let n = ((self.offset + r + 1) / RATIO).min(TOPK);
            ensure!(self.counts[r] == n, "indexer visible count");
            let row = &self.ids[r * TOPK..(r + 1) * TOPK];
            ensure!(
                row[..n].windows(2).all(|w| w[0] < w[1])
                    && row[..n]
                        .iter()
                        .all(|&id| (id as usize) < (self.offset + r + 1) / RATIO)
                    && row[n..].iter().all(|&id| id == SENTINEL),
                "sorted visible IDs/sentinels"
            );
        }
        Ok(())
    }
    pub fn allowed(&self, row: usize, key: usize) -> bool {
        if row >= self.rows || key > self.offset + row {
            return false;
        }
        let tail = (self.offset + row + 1) / RATIO * RATIO;
        key >= tail
            || self.ids[row * TOPK..row * TOPK + self.counts[row]]
                .binary_search(&((key / RATIO) as u32))
                .is_ok()
    }
    pub fn tokens(&self, row: usize) -> Result<Vec<usize>> {
        self.validate()?;
        ensure!(row < self.rows, "query row bounds");
        let mut out: Vec<_> = self.ids[row * TOPK..row * TOPK + self.counts[row]]
            .iter()
            .flat_map(|&b| (b as usize * RATIO)..(b as usize + 1) * RATIO)
            .collect();
        out.extend(((self.offset + row + 1) / RATIO * RATIO)..=self.offset + row);
        Ok(out)
    }
}
/// Validate before any addition, multiplication, allocation or device dispatch.
pub fn score_blocks(offset: usize, rows: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows)
            && offset.checked_add(rows).is_some_and(|n| n <= MAX_POSITION),
        "indexer score geometry"
    );
    Ok((offset + rows) / RATIO)
}
pub fn select(scores: &[f32], offset: usize, rows: usize) -> Result<Selection> {
    let nb = score_blocks(offset, rows)?;
    let total = offset + rows;
    ensure!(
        scores.len() == rows * nb && scores.iter().all(|v| v.is_finite() && *v >= 0.),
        "indexer scores finite/nonnegative"
    );
    let mut ids = vec![SENTINEL; rows * TOPK];
    let mut counts = vec![];
    for r in 0..rows {
        let visible = (offset + r + 1) / RATIO;
        let mut order: Vec<_> = (0..visible).collect();
        // partial_cmp intentionally treats +/-zero as equal before the ID tie.
        order.sort_unstable_by(|&a, &b| {
            scores[r * nb + b]
                .partial_cmp(&scores[r * nb + a])
                .unwrap()
                .then(a.cmp(&b))
        });
        order.truncate(TOPK);
        order.sort_unstable();
        counts.push(order.len());
        for (i, b) in order.into_iter().enumerate() {
            ids[r * TOPK + i] = b as u32;
        }
    }
    let s = Selection {
        offset,
        rows,
        total,
        ids,
        counts,
    };
    s.validate()?;
    Ok(s)
}
pub struct Candidate {
    pub offset: usize,
    pub rows: usize,
    pub queries: Vec<u16>,
    pub scores: Vec<f32>,
    pub selection: Selection,
    pub complete: State,
    initial: State,
    raw_new: Vec<u16>,
}
impl Candidate {
    /// Commit a prefix by trimming pooled blocks and retaining only the true
    /// raw leftover. Future tokens never influence a partially accepted block.
    pub fn accepted(&self, n: usize) -> Result<State> {
        ensure!(n <= self.rows, "indexer accepted prefix bounds");
        if n == 0 {
            return Ok(self.initial.clone());
        }
        let len = self.offset + n;
        let mut raw = self.initial.tail.clone();
        raw.extend_from_slice(&self.raw_new[..n * D]);
        let tail_len = len % RATIO * D;
        let tail = raw[raw.len() - tail_len..].to_vec();
        let s = State {
            base: self.initial.base,
            len,
            pooled: self.complete.pooled[..len / RATIO * D].to_vec(),
            tail,
        };
        s.validate()?;
        Ok(s)
    }
}
/// Append already-projected BF16 [rows, 5*128] indexer activations. Always
/// maintain pooling history, even below the sparse-attention crossover.
pub fn reference_projected(norms: &Norms, initial: &State, qk: &[u16]) -> Result<Candidate> {
    norms.validate()?;
    initial.validate()?;
    ensure!(
        !qk.is_empty() && qk.len() % QK == 0 && qk.len() / QK <= MAX_ROWS && finite(qk),
        "indexer projected geometry/finite"
    );
    let rows = qk.len() / QK;
    let len = initial.len + rows;
    ensure!(
        len <= MAX_POSITION - initial.base,
        "indexer appended position"
    );
    let raw_new: Vec<_> = qk
        .chunks_exact(QK)
        .flat_map(|r| r[HEADS * D..].iter().copied())
        .collect();
    let mut raw = initial.tail.clone();
    raw.extend(&raw_new);
    let mut pooled = initial.pooled.clone();
    for block in raw.chunks_exact(RATIO * D) {
        let pos = initial.base + pooled.len() / D * RATIO;
        pooled.extend(pool(block, norms, pos)?);
    }
    let tail = raw[raw.len() / (RATIO * D) * (RATIO * D)..].to_vec();
    let complete = State {
        base: initial.base,
        len,
        pooled,
        tail,
    };
    complete.validate()?;
    let mut queries = vec![];
    for (r, row) in qk.chunks_exact(QK).enumerate() {
        for h in 0..HEADS {
            queries.extend(rotate(
                &norm(&row[h * D..(h + 1) * D], &norms.q),
                initial.base + initial.len + r,
                false,
            )?);
        }
    }
    let nb = len / RATIO;
    let mut scores = vec![0f32; rows * nb];
    for r in 0..rows {
        for b in 0..nb {
            let k = &complete.pooled[b * D..(b + 1) * D];
            let mut sum = 0f32;
            for h in 0..HEADS {
                let q = &queries[(r * HEADS + h) * D..(r * HEADS + h + 1) * D];
                let dot = q.iter().zip(k).map(|(&q, &k)| num(q) * num(k)).sum::<f64>() as f32;
                sum += dot.max(0.);
            }
            scores[r * nb + b] = sum;
        }
    }
    let selection = select(&scores, initial.len, rows)?;
    Ok(Candidate {
        offset: initial.len,
        rows,
        queries,
        scores,
        selection,
        complete,
        initial: initial.clone(),
        raw_new,
    })
}
pub fn reference(
    w: &Weights,
    decoded: &[f32],
    initial: &State,
    input: &[u16],
) -> Result<(Vec<u16>, Candidate)> {
    ensure!(
        !input.is_empty() && input.len() % H == 0 && input.len() / H <= MAX_ROWS,
        "indexer input geometry"
    );
    let projected = w
        .projection
        .reference(decoded, input)?
        .into_iter()
        .map(bits)
        .collect::<Vec<_>>();
    let r = reference_projected(&w.norms, initial, &projected)?;
    Ok((projected, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn norms() -> Norms {
        Norms {
            q: vec![bf16::ONE.to_bits(); D],
            k: vec![bf16::ONE.to_bits(); D],
            provenance: json!(null),
        }
    }
    #[test]
    fn visibility_crossover_incomplete_tail_and_ties() {
        let offset = 2045;
        let rows = 8;
        let nb = (offset + rows) / 4;
        let mut scores = vec![0.; rows * nb];
        for r in 0..rows {
            for b in 0..nb {
                scores[r * nb + b] = if b % 2 == 0 { 0.0 } else { -0.0 };
            }
        }
        let s = select(&scores, offset, rows).unwrap();
        for r in 0..rows {
            let p = offset + r;
            let tokens = s.tokens(r).unwrap();
            assert!(tokens.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(tokens.len(), ((p + 1) / 4).min(512) * 4 + (p + 1) % 4);
            for key in 0..offset + rows {
                assert_eq!(s.allowed(r, key), tokens.contains(&key));
            }
            assert_eq!(s.ids[r * TOPK], 0);
            if p <= 2050 {
                assert_eq!(tokens, (0..=p).collect::<Vec<_>>());
            } else {
                assert!(!tokens.contains(&2048));
            }
        }
        // Make every future complete block huge; early rows still cannot see it.
        for r in 0..rows {
            for b in (offset + r + 1) / 4..nb {
                scores[r * nb + b] = 1e30;
            }
        }
        let future = select(&scores, offset, rows).unwrap();
        assert_eq!(future.ids, s.ids);
        scores[0] = f32::NAN;
        assert!(select(&scores, offset, rows).is_err());
    }
    #[test]
    fn prefix_commit_rebuilds_partial_blocks_without_future_leakage() {
        let n = norms();
        let empty = State::empty(17).unwrap();
        let all: Vec<_> = (0..16 * QK)
            .map(|i| bits(((i * 131 % 257) as f64 - 128.) / 64.))
            .collect();
        let initial = reference_projected(&n, &empty, &all[..3 * QK])
            .unwrap()
            .complete;
        let candidate = reference_projected(&n, &initial, &all[3 * QK..11 * QK]).unwrap();
        for accepted in 0..=8 {
            let committed = candidate.accepted(accepted).unwrap();
            let separate = if accepted == 0 {
                initial.clone()
            } else {
                reference_projected(&n, &initial, &all[3 * QK..(3 + accepted) * QK])
                    .unwrap()
                    .complete
            };
            assert_eq!(committed, separate);
            let resumed =
                reference_projected(&n, &committed, &all[(3 + accepted) * QK..16 * QK]).unwrap();
            let whole = reference_projected(&n, &empty, &all).unwrap();
            assert_eq!(resumed.complete, whole.complete);
            assert_eq!(
                resumed.queries,
                &whole.queries[(3 + accepted) * HEADS * D..]
            );
        }
        assert!(candidate.accepted(9).is_err());
        let mut bad = initial.clone();
        bad.tail.pop();
        assert!(bad.validate().is_err());
        assert!(State::empty(MAX_POSITION + 1).is_err());
    }
    #[test]
    fn scoring_chooses_visible_top_blocks_in_index_order() {
        let nb = 520;
        let scores: Vec<_> = (0..nb).map(|b| b as f32).collect();
        let s = select(&scores, nb * 4 - 1, 1).unwrap();
        assert_eq!(&s.ids[..], &(8..520).collect::<Vec<u32>>());
        assert!(!s.allowed(0, 0));
        assert!(s.allowed(0, 2079));
        assert!(s.tokens(1).is_err());
        assert!(score_blocks(usize::MAX, 1).is_err());
        assert!(score_blocks(MAX_POSITION, 1).is_err());
        assert!(score_blocks(0, 0).is_err());
        assert!(score_blocks(0, MAX_ROWS + 1).is_err());
        assert_eq!(
            score_blocks(MAX_POSITION - MAX_ROWS, MAX_ROWS).unwrap(),
            MAX_POSITION / RATIO
        );
    }
}
