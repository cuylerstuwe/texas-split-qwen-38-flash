//! Qwen's history-only PLE row lookup and source-grounded scalar reference.
//! Runtime table access reads only selected packed rows; no whole-table upload.
pub use crate::hc::{F, H, HC, MAX_ROWS};
use crate::{affine4::Affine4, lowbit_hc::small, projection::sha, weights::Shard};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::Path, time::Instant};
pub const HEADS: usize = 16;
pub const DIM: usize = 160;
pub const GROUP: usize = 32;
pub const HISTORY: usize = 9;
pub const VOCAB: u32 = 248320;
pub const EOS: u32 = 248044;
pub const TABLE_ROWS: usize = 320001536;
const PREFIX: &str = "language_model.model.layers.1.ple";
pub fn num(x: u16) -> f64 {
    bf16::from_bits(x).to_f64()
}
pub fn bits(x: f64) -> u16 {
    crate::rounding::bf16_from_f64(x).to_bits()
}
fn round(x: f64) -> f64 {
    num(bits(x))
}
fn finite(x: &[u16]) -> bool {
    x.iter().all(|&v| bf16::from_bits(v).is_finite())
}

#[derive(Clone)]
pub struct Hash {
    pub multipliers: [i64; 3],
    pub sizes: [i64; HEADS],
    pub offsets: [i64; HEADS],
    pub provenance: Value,
}
impl Hash {
    pub fn load(root: &Path) -> Result<Self> {
        let index = std::fs::read(root.join("model.safetensors.index.json"))?;
        ensure!(
            sha(&index) == "a819797ea340ed7a587c888c9f76177e90a6b482aee77d4be2693cbcf9b56a8e",
            "PLE index pin"
        );
        let index: Value = serde_json::from_slice(&index)?;
        let shard = Shard::open(&root.join("model-trunk-00000.safetensors"))?;
        let mut pins = vec![];
        let mut get = |name: &str, n: usize| -> Result<Vec<i64>> {
            let name = format!("{PREFIX}.ple_embedding.{name}");
            ensure!(
                index["weight_map"][&name] == "model-trunk-00000.safetensors",
                "PLE hash shard"
            );
            let t = shard.tensors.get(&name).context("PLE hash tensor")?;
            ensure!(t.dtype == "I64" && t.shape == [n as u64], "PLE hash shape");
            let raw = shard.read_range(&name, 0, t.bytes())?;
            pins.push(json!({"tensor":name,"payload_sha256":sha(&raw)}));
            Ok(raw
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let multipliers = get("layer_multipliers", 3)?.try_into().unwrap();
        let offsets = get("ngram_heads_offsets", HEADS)?.try_into().unwrap();
        let sizes = get("ngram_heads_vocab_sizes", HEADS)?.try_into().unwrap();
        let h = Self {
            multipliers,
            sizes,
            offsets,
            provenance: json!({"header_sha256":shard.header_sha256,"tensors":pins}),
        };
        h.validate()?;
        Ok(h)
    }
    /// R55: the same hash constants from the target pack (canonical 4bpw view).
    pub fn load_target(pack: &crate::projection::Pack) -> Result<Self> {
        let mut pins = vec![];
        let mut get = |name: &str, n: usize| -> Result<Vec<i64>> {
            let (t, raw, prov) = pack.tensor(&format!("{PREFIX}.ple_embedding.{name}"))?;
            ensure!(t.dtype == "I64" && t.shape == [n as u64], "target PLE hash shape");
            pins.push(prov);
            Ok(raw
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let multipliers = get("layer_multipliers", 3)?.try_into().unwrap();
        let offsets = get("ngram_heads_offsets", HEADS)?.try_into().unwrap();
        let sizes = get("ngram_heads_vocab_sizes", HEADS)?.try_into().unwrap();
        let h = Self { multipliers, sizes, offsets, provenance: json!({"target_tensors":pins}) };
        h.validate()?;
        Ok(h)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.multipliers.iter().all(|&m| m > 0 && m % 2 == 1) && self.offsets[0] == 0,
            "PLE hash multipliers/base"
        );
        for h in 0..HEADS {
            ensure!(
                self.sizes[h] > 0
                    && self.offsets[h] >= 0
                    && self.offsets[h]
                        .checked_add(self.sizes[h])
                        .is_some_and(|v| v <= TABLE_ROWS as i64),
                "PLE hash bounds"
            );
            if h + 1 < HEADS {
                ensure!(
                    self.offsets[h] + self.sizes[h] == self.offsets[h + 1],
                    "PLE hash partitions"
                );
            }
        }
        Ok(())
    }
    /// EOS resets shifted-token visibility, after emitting the EOS row itself.
    pub fn rows(&self, prev: [u32; 2], tokens: &[u32]) -> Result<Vec<u32>> {
        self.validate()?;
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= MAX_ROWS
                && prev.iter().chain(tokens).all(|&t| t < VOCAB),
            "PLE token bounds"
        );
        let all: Vec<_> = prev.into_iter().chain(tokens.iter().copied()).collect();
        let mut last_eos = None;
        let mut rows = Vec::with_capacity(tokens.len() * HEADS);
        for (t, &token) in all.iter().enumerate() {
            if t >= 2 {
                let start = last_eos.map_or(0, |p| p + 1);
                let shifted = |n: usize| if t >= start + n { all[t - n] } else { EOS };
                let mixed = (token as i64).wrapping_mul(self.multipliers[0])
                    ^ (shifted(1) as i64).wrapping_mul(self.multipliers[1]);
                let mixed3 = mixed ^ (shifted(2) as i64).wrapping_mul(self.multipliers[2]);
                for h in 0..HEADS {
                    rows.push(
                        ((if h < 8 { mixed } else { mixed3 }).rem_euclid(self.sizes[h])
                            + self.offsets[h]) as u32,
                    );
                }
            }
            if token == EOS {
                last_eos = Some(t);
            }
        }
        Ok(rows)
    }
}
pub fn accepted_tokens(prev: [u32; 2], tokens: &[u32], accepted: usize) -> Result<[u32; 2]> {
    ensure!(
        accepted <= tokens.len()
            && tokens.len() <= MAX_ROWS
            && prev.iter().chain(tokens).all(|&t| t < VOCAB),
        "PLE accepted tokens"
    );
    let all: Vec<_> = prev
        .into_iter()
        .chain(tokens[..accepted].iter().copied())
        .collect();
    Ok(all[all.len() - 2..].try_into().unwrap())
}
pub struct PackedRows {
    pub ids: Vec<u32>,
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub provenance: Value,
}
impl PackedRows {
    pub fn validate(&self) -> Result<()> {
        let n = self.ids.len();
        ensure!(
            n > 0
                && n % HEADS == 0
                && n <= MAX_ROWS * HEADS
                && self.ids.iter().all(|&i| (i as usize) < TABLE_ROWS),
            "PLE row IDs"
        );
        ensure!(
            self.codes.len() == n * DIM / 2
                && self.scales.len() == n * DIM / GROUP
                && self.biases.len() == self.scales.len()
                && finite(&self.scales)
                && finite(&self.biases),
            "PLE packed rows"
        );
        Ok(())
    }
    /// CPU oracle only; the native CUDA path will unpack these bytes on-card.
    pub fn reference(&self) -> Result<Vec<u16>> {
        self.validate()?;
        Ok((0..self.ids.len() * DIM)
            .map(|i| {
                let q = (self.codes[i / 2] >> (4 * (i % 2))) & 15;
                bits(q as f64 * num(self.scales[i / GROUP]) + num(self.biases[i / GROUP]))
            })
            .collect())
    }
}
/// R56: how a gather reads its unique rows (same bytes, same order in every mode).
/// QWEN_PLE_PARALLEL: unset = `Pool`, `0` = `Serial`, `thread` or `1` = `Threads`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadMode {
    /// one pread after another
    Serial,
    /// R54 G3: one scoped thread per unique row
    Threads,
    /// R56: the persistent io_pool critical lane
    Pool,
}
impl ReadMode {
    pub fn from_env() -> Self {
        match std::env::var("QWEN_PLE_PARALLEL").as_deref() {
            Ok("0") => Self::Serial,
            Ok("thread") | Ok("1") => Self::Threads,
            _ => Self::Pool,
        }
    }
}
pub struct Table {
    file: std::sync::Arc<File>,
    shard: Shard,
}
impl Table {
    pub fn open(root: &Path) -> Result<Self> {
        let shard = Shard::open(&root.join("ngram_table.bin"))?;
        ensure!(
            shard.header_sha256
                == "758e4beb0aa3b7a28826fc177a6ca7d679c8a0cbbe29bdfee8e3e7d4a022ca79"
                && shard.file_bytes == 32000153944,
            "PLE lowbit table pin"
        );
        let file = std::sync::Arc::new(File::open(&shard.path)?);
        Ok(Self { file, shard })
    }
    /// R56: one row's three reads (weight 80 B, scales 10 B, biases 10 B), in that order.
    pub fn row_reads(&self, id: u32) -> Vec<crate::io_pool::Read> {
        [("weight", 80u64), ("scales", 10), ("biases", 10)]
            .iter()
            .map(|&(name, stride)| crate::io_pool::Read {
                file: self.file.clone(),
                at: self.shard.data_start + self.shard.tensors[name].data_offsets[0] + id as u64 * stride,
                len: stride as usize,
            })
            .collect()
    }
    /// R56: warm the page cache for these rows on the background lane (returns at once).
    pub fn prefetch(&self, ids: &[u32]) {
        let mut want: Vec<u32> = ids.iter().copied().filter(|&v| (v as usize) < TABLE_ROWS).collect();
        want.sort_unstable();
        want.dedup();
        crate::io_pool::prefetch(want.into_iter().flat_map(|id| self.row_reads(id)).collect());
    }
    /// Rows in caller order; the read mode comes from QWEN_PLE_PARALLEL (see `ReadMode`).
    pub fn gather(&self, ids: &[u32]) -> Result<PackedRows> {
        self.gather_with(ids, ReadMode::from_env())
    }
    pub fn gather_with(&self, ids: &[u32], mode: ReadMode) -> Result<PackedRows> {
        ensure!(
            !ids.is_empty()
                && ids.len() % HEADS == 0
                && ids.len() <= MAX_ROWS * HEADS
                && ids.iter().all(|&v| (v as usize) < TABLE_ROWS),
            "PLE gather bounds"
        );
        ensure!(
            self.file.metadata()?.len() == self.shard.file_bytes,
            "PLE table size changed"
        );
        let t = Instant::now();
        let mut unique = BTreeMap::<u32, Vec<u8>>::new();
        let mut calls = 0;
        // R54 G3: QWEN_PLE_PARALLEL=1 issues every unique row's reads concurrently (one
        // scoped thread per row) instead of one pread after another. Same bytes, same order.
        // R56: default is the persistent pool (all unique rows' reads concurrently, same bytes);
        // QWEN_PLE_PARALLEL=0 keeps one pread after another, =thread the R54 per-row threads.
        if mode == ReadMode::Pool {
            let mut want: Vec<u32> = ids.to_vec();
            want.sort_unstable();
            want.dedup();
            let groups: Vec<Vec<crate::io_pool::Read>> = want.iter().map(|&id| self.row_reads(id)).collect();
            // every read in flight at once (kernel advisory reads), then the pool copies them out
            for r in groups.iter().flatten() {
                crate::io_pool::advise(&r.file, r.at, r.len);
            }
            let rows = crate::io_pool::read_groups(groups)?;
            for (id, parts) in want.iter().zip(rows) {
                unique.insert(*id, parts.concat());
                calls += 3;
            }
        } else if mode == ReadMode::Threads {
            let mut want: Vec<u32> = ids.to_vec();
            want.sort_unstable();
            want.dedup();
            let rows: Vec<Result<Vec<u8>>> = std::thread::scope(|s| {
                let handles: Vec<_> = want
                    .iter()
                    .map(|&id| {
                        s.spawn(move || -> Result<Vec<u8>> {
                            let mut row = vec![0u8; 100];
                            let mut start = 0usize;
                            for (name, stride) in [("weight", 80u64), ("scales", 10), ("biases", 10)] {
                                let at = self.shard.data_start + self.shard.tensors[name].data_offsets[0] + id as u64 * stride;
                                self.file.read_exact_at(&mut row[start..start + stride as usize], at)?;
                                start += stride as usize;
                            }
                            Ok(row)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("PLE read thread")).collect()
            });
            for (id, row) in want.iter().zip(rows) {
                unique.insert(*id, row?);
                calls += 3;
            }
        }
        for &id in ids {
            if unique.contains_key(&id) {
                continue;
            }
            let mut row = vec![];
            for (name, stride) in [("weight", 80), ("scales", 10), ("biases", 10)] {
                let at = self.shard.data_start
                    + self.shard.tensors[name].data_offsets[0]
                    + id as u64 * stride;
                let start = row.len();
                row.resize(start + stride as usize, 0);
                self.file.read_exact_at(&mut row[start..], at)?;
                calls += 1;
            }
            unique.insert(id, row);
        }
        let mut codes = vec![];
        let mut scales = vec![];
        let mut biases = vec![];
        for id in ids {
            let row = &unique[id];
            codes.extend_from_slice(&row[..80]);
            scales.extend(
                row[80..90]
                    .chunks_exact(2)
                    .map(|v| u16::from_le_bytes(v.try_into().unwrap())),
            );
            biases.extend(
                row[90..100]
                    .chunks_exact(2)
                    .map(|v| u16::from_le_bytes(v.try_into().unwrap())),
            );
        }
        let payload: Vec<_> = codes
            .iter()
            .copied()
            .chain(scales.iter().chain(&biases).flat_map(|v| v.to_le_bytes()))
            .collect();
        crate::io_pool::DRAFTER_GATHER.record(t.elapsed().as_secs_f64() * 1e6);
        let out = PackedRows {
            ids: ids.to_vec(),
            codes,
            scales,
            biases,
            provenance: json!({"table":self.shard.path,"header_sha256":self.shard.header_sha256,"ids":ids,"selected_payload_sha256":sha(&payload),"unique_rows":unique.len(),"file_read_calls":calls,"file_read_bytes":unique.len()*100,"packed_transfer_bytes":ids.len()*100,"host_gather_us":t.elapsed().as_secs_f64()*1e6,"whole_table_mapped":false,"decoded_on_host":false}),
        };
        out.validate()?;
        Ok(out)
    }
}
pub struct Weights {
    pub key: Affine4,
    pub value: Affine4,
    pub norms: [Vec<u16>; 3],
    pub conv: Vec<u16>,
    pub hash: Hash,
    pub provenance: Value,
}
impl Weights {
    pub fn load(root: &Path) -> Result<Self> {
        let key = Affine4::load(root, &format!("{PREFIX}.key_proj"), H)?;
        let value = Affine4::load(root, &format!("{PREFIX}.value_proj"), H)?;
        let mut norms = vec![];
        let mut pins = vec![];
        for n in ["norm_key", "norm_query", "norm_conv"] {
            let (v, p) = small(root, &format!("{PREFIX}.{n}.weight"), &[F as u64])?;
            norms.push(v);
            pins.push(p);
        }
        let (conv, cp) = small(root, &format!("{PREFIX}.conv1d.weight"), &[F as u64, 4, 1])?;
        let hash = Hash::load(root)?;
        let provenance = json!({"key":key.provenance,"value":value.provenance,"norms":pins,"conv":cp,"hash":hash.provenance,"norm_semantics":"exported weights already contain source +1; no additional offset"});
        let w = Self {
            key,
            value,
            norms: norms.try_into().unwrap(),
            conv,
            hash,
            provenance,
        };
        w.validate()?;
        Ok(w)
    }
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        self.value.validate()?;
        self.hash.validate()?;
        ensure!(
            (self.key.k, self.key.n) == (H, F) && (self.value.k, self.value.n) == (H, H),
            "PLE projections"
        );
        ensure!(
            self.norms.iter().all(|v| v.len() == F && finite(v))
                && self.conv.len() == F * 4
                && finite(&self.conv),
            "PLE small weights"
        );
        Ok(())
    }
    pub fn decoded(&self) -> Result<[Vec<f32>; 2]> {
        Ok([self.key.oracle_weights()?, self.value.oracle_weights()?])
    }
}
fn norm(x: &[u16], w: &[u16]) -> Vec<u16> {
    x.chunks_exact(H)
        .enumerate()
        .flat_map(|(h, row)| {
            let inv = (row.iter().map(|&v| num(v).powi(2)).sum::<f64>() / H as f64 + 1e-6)
                .sqrt()
                .recip();
            row.iter()
                .enumerate()
                .map(move |(j, &v)| bits(round(num(v) * inv) * num(w[(h % HC) * H + j])))
        })
        .collect()
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Core {
    pub key_norm: Vec<u16>,
    pub query_norm: Vec<u16>,
    pub dots: Vec<u16>,
    pub scaled: Vec<u16>,
    pub roots: Vec<u16>,
    pub gates: Vec<u16>,
    pub gated: Vec<u16>,
    pub normalized: Vec<u16>,
    pub conv: Vec<u16>,
    pub activated: Vec<u16>,
    pub addend: Vec<u16>,
    pub output: Vec<u16>,
}
pub fn core_reference(
    w: &Weights,
    stream: &[u16],
    key: &[u16],
    value: &[u16],
    initial: &[u16],
) -> Result<Core> {
    w.validate()?;
    ensure!(
        !stream.is_empty()
            && stream.len() % F == 0
            && stream.len() / F <= MAX_ROWS
            && key.len() == stream.len()
            && value.len() == stream.len() / HC
            && initial.len() == HISTORY * F,
        "PLE core geometry"
    );
    ensure!(
        [stream, key, value, initial].iter().all(|v| finite(v)),
        "PLE finite input"
    );
    let rows = stream.len() / F;
    let key_norm = norm(key, &w.norms[0]);
    let query_norm = norm(stream, &w.norms[1]);
    let mut dots = vec![];
    let mut scaled = vec![];
    let mut roots = vec![];
    let mut gates = vec![];
    let mut gated = vec![0; stream.len()];
    for r in 0..rows {
        for h in 0..HC {
            let at = r * F + h * H;
            let dot = bits(
                (0..H)
                    .map(|j| round(num(key_norm[at + j]) * num(query_norm[at + j])))
                    .sum(),
            );
            dots.push(dot);
            let sc = bits(num(dot) * round((H as f64).sqrt().recip()));
            scaled.push(sc);
            let g = num(sc);
            let root = bits(
                round(g.abs().max(round(1e-6)).sqrt()) * if g == 0. { 0. } else { g.signum() },
            );
            roots.push(root);
            let gate = bits(1. / (1. + (-num(root)).exp()));
            gates.push(gate);
            for j in 0..H {
                gated[at + j] = bits(num(gate) * num(value[r * H + j]));
            }
        }
    }
    let normalized = norm(&gated, &w.norms[2]);
    let mut cat = initial.to_vec();
    cat.extend_from_slice(&normalized);
    let mut conv = vec![];
    let mut activated = vec![];
    let mut addend = vec![];
    let mut output = vec![];
    for r in 0..rows {
        for d in 0..F {
            let c = bits(
                (0..4)
                    .map(|j| num(cat[(r + j * 3) * F + d]) * num(w.conv[d * 4 + j]))
                    .sum(),
            );
            conv.push(c);
            let a = bits(num(c) * round(1. / (1. + (-num(c)).exp())));
            activated.push(a);
            let y = bits(num(gated[r * F + d]) + num(a));
            addend.push(y);
            output.push(bits(num(stream[r * F + d]) + num(y)));
        }
    }
    Ok(Core {
        key_norm,
        query_norm,
        dots,
        scaled,
        roots,
        gates,
        gated,
        normalized,
        conv,
        activated,
        addend,
        output,
    })
}
pub fn accepted_conv(initial: &[u16], candidate: &[u16], accepted: usize) -> Result<Vec<u16>> {
    ensure!(
        initial.len() == HISTORY * F
            && candidate.len() % F == 0
            && candidate.len() / F <= MAX_ROWS
            && accepted <= candidate.len() / F
            && finite(initial)
            && finite(candidate),
        "PLE convolution prefix"
    );
    let mut cat = initial.to_vec();
    cat.extend_from_slice(&candidate[..accepted * F]);
    Ok(cat[cat.len() - HISTORY * F..].to_vec())
}
pub struct Reference {
    pub key: Vec<u16>,
    pub value: Vec<u16>,
    pub core: Core,
}
pub fn reference(
    w: &Weights,
    decoded: &[Vec<f32>; 2],
    stream: &[u16],
    embedding: &[u16],
    initial: &[u16],
) -> Result<Reference> {
    ensure!(
        embedding.len() * HC == stream.len(),
        "PLE embedding alignment"
    );
    let key = w
        .key
        .reference(&decoded[0], embedding)?
        .into_iter()
        .map(bits)
        .collect::<Vec<_>>();
    let value = w
        .value
        .reference(&decoded[1], embedding)?
        .into_iter()
        .map(bits)
        .collect::<Vec<_>>();
    let core = core_reference(w, stream, &key, &value, initial)?;
    Ok(Reference { key, value, core })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hash() -> Hash {
        let sizes = [
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        ];
        let mut offsets = [0; HEADS];
        for i in 1..HEADS {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }
        Hash {
            multipliers: [23703573157769, 20109073645365, 8052911324071],
            sizes,
            offsets,
            provenance: json!("source/checkpoint fixed fixture"),
        }
    }
    #[test]
    fn eos_hash_matches_external_vectors_and_every_split() {
        // Independent Python-model vectors retained in Sushi qwen4_exp.zig.
        let want = [
            [
                15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159,
                179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119,
                315720898,
            ],
            [
                12441580, 26378836, 53347667, 75104214, 99467174, 114254887, 126436461, 156012011,
                169119442, 187827161, 214803956, 239809754, 242938905, 266427765, 294337448,
                314484167,
            ],
            [
                10204458, 27984170, 41283776, 68842151, 85621153, 118821647, 129504214, 158727320,
                176298516, 181690702, 206665473, 238343128, 252151767, 267018740, 285543023,
                319927855,
            ],
            [
                18043673, 37626835, 51159316, 78294604, 94015356, 106720349, 136526052, 144330141,
                176817901, 186368539, 203707490, 230017629, 247662678, 266533413, 293096193,
                307951937,
            ],
            [
                10041117, 28960672, 48420531, 71664411, 83016360, 106800418, 122476460, 150044571,
                163654473, 184259024, 206781966, 224776026, 248853488, 273290488, 294849492,
                303242927,
            ],
        ];
        let h = hash();
        let tokens = [5, 7, EOS, 9, 11];
        let all = h.rows([EOS; 2], &tokens).unwrap();
        assert_eq!(all, want.into_iter().flatten().collect::<Vec<u32>>());
        for n in 0..tokens.len() {
            let prev = accepted_tokens([EOS; 2], &tokens, n).unwrap();
            assert_eq!(h.rows(prev, &tokens[n..]).unwrap(), all[n * HEADS..]);
        }
        assert!(h.rows([EOS; 2], &[]).is_err());
        assert!(h.rows([EOS; 2], &[VOCAB]).is_err());
        assert!(accepted_tokens([EOS; 2], &tokens, 6).is_err());
    }
    #[test]
    fn packed_row_words_signed_metadata_and_bf16_ties() {
        let codes: Vec<_> = (0..HEADS * DIM / 2).map(|i| (i * 37) as u8).collect();
        let scales: Vec<_> = (0..HEADS * DIM / GROUP)
            .map(|i| bits(if i % 2 == 0 { 1. / 256. } else { -1. / 128. }))
            .collect();
        let biases: Vec<_> = (0..scales.len())
            .map(|i| bits(if i % 2 == 0 { 1. } else { -0.5 }))
            .collect();
        let p = PackedRows {
            ids: vec![0; HEADS],
            codes,
            scales,
            biases,
            provenance: Value::Null,
        };
        let got = p.reference().unwrap();
        for (i, &v) in got.iter().enumerate() {
            let j = (i / 8) * 4;
            let word = u32::from_le_bytes(p.codes[j..j + 4].try_into().unwrap());
            let code = (word >> (4 * (i % 8))) & 15;
            let raw = code as f32 * bf16::from_bits(p.scales[i / GROUP]).to_f32()
                + bf16::from_bits(p.biases[i / GROUP]).to_f32();
            assert_eq!(v, bf16::from_f32(raw).to_bits());
        }
        let mut bad = p;
        bad.ids[0] = TABLE_ROWS as u32;
        assert!(bad.reference().is_err());
    }
    #[test]
    fn every_conv_prefix_isolated_and_continues() {
        let initial: Vec<_> = (0..HISTORY * F)
            .map(|i| bits((i / F) as f64 - 9.))
            .collect();
        let candidate: Vec<_> = (0..MAX_ROWS * F).map(|i| bits((i / F) as f64)).collect();
        for accepted in 0..=MAX_ROWS {
            let next = accepted_conv(&initial, &candidate, accepted).unwrap();
            for (i, &v) in next.iter().enumerate() {
                assert_eq!(v, bits((accepted + i / F) as f64 - 9.));
            }
            for more in 0..=MAX_ROWS - accepted {
                assert_eq!(
                    accepted_conv(&next, &candidate[accepted * F..], more).unwrap(),
                    accepted_conv(&initial, &candidate, accepted + more).unwrap()
                );
            }
        }
        assert!(accepted_conv(&initial, &candidate, MAX_ROWS + 1).is_err());
        assert!(accepted_conv(&initial[..F], &candidate, 1).is_err());
        assert!(accepted_conv(&initial, &[0x7fc0; F], 1).is_err());
    }
}
