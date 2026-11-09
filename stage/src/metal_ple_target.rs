//! R55 target layer-1 PLE on Metal: host hash + BF16 n-gram row gather from the canonical
//! ngram_table.bin, affine8 key/value projections and the PLE core (kernels/ple_target.metal,
//! ported from ple::core_reference). The 9-row convolution history and the two previous
//! tokens are resident; commit keeps any accepted prefix.
use crate::{
    dense::Affine8,
    hc::{F, H, HC, MAX_ROWS},
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, alloc, dense_scratch_bytes, upload},
    ple::{self, HEADS, HISTORY},
    projection::Pack,
};
use anyhow::{Context, Result, ensure};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{fs::File, os::unix::fs::FileExt, path::Path, ptr::NonNull};

const PREFIX: &str = "language_model.model.layers.1.ple";
const ROW: usize = ple::DIM * 2; // 160 BF16

pub struct Table {
    file: std::sync::Arc<File>,
    data_start: u64,
}
impl Table {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("ngram_table.bin");
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0)?;
        let n = u64::from_le_bytes(len);
        ensure!(n < 4096, "n-gram table header");
        let mut header = vec![0u8; n as usize];
        file.read_exact_at(&mut header, 8)?;
        let h: serde_json::Value = serde_json::from_slice(&header)?;
        ensure!(
            h["weight"]["dtype"] == "BF16"
                && h["weight"]["shape"] == serde_json::json!([ple::TABLE_ROWS, ple::DIM]),
            "target n-gram table is BF16 [{}][{}]",
            ple::TABLE_ROWS,
            ple::DIM
        );
        let data_start = 8 + n;
        ensure!(
            file.metadata()?.len() == data_start + (ple::TABLE_ROWS * ROW) as u64,
            "n-gram table size"
        );
        Ok(Self { file: std::sync::Arc::new(file), data_start })
    }
    pub fn read(&self, id: u32) -> crate::io_pool::Read {
        crate::io_pool::Read { file: self.file.clone(), at: self.data_start + id as u64 * ROW as u64, len: ROW }
    }
    /// R56: warm the page cache for these rows on the background lane (returns at once).
    pub fn prefetch(&self, ids: &[u32]) {
        let mut want: Vec<u32> = ids.iter().copied().filter(|&v| (v as usize) < ple::TABLE_ROWS).collect();
        want.sort_unstable();
        want.dedup();
        crate::io_pool::prefetch(want.into_iter().map(|id| self.read(id)).collect());
    }
    /// Rows in caller order (16 per token: token-major, head-minor); unique rows are read
    /// concurrently.
    pub fn gather(&self, ids: &[u32]) -> Result<Vec<u16>> {
        let t = std::time::Instant::now();
        let mut want: Vec<u32> = ids.to_vec();
        want.sort_unstable();
        want.dedup();
        // R56: every read in flight at once (kernel advisory reads), then the persistent pool's
        // critical lane copies them out (same bytes; no per-call thread spawns).
        let per = want.len().div_ceil(16).max(1);
        let groups: Vec<Vec<crate::io_pool::Read>> = want.chunks(per).map(|c| c.iter().map(|&id| self.read(id)).collect()).collect();
        for r in groups.iter().flatten() {
            crate::io_pool::advise(&r.file, r.at, r.len);
        }
        let rows = crate::io_pool::read_groups(groups)?;
        let map: std::collections::HashMap<u32, Vec<u8>> = want.iter().copied().zip(rows.into_iter().flatten()).collect();
        let mut out = Vec::with_capacity(ids.len() * ple::DIM);
        for id in ids {
            let b = &map[id];
            out.extend(b.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])));
        }
        crate::io_pool::TARGET_GATHER.record(t.elapsed().as_secs_f64() * 1e6);
        Ok(out)
    }
}

struct Pair {
    full: Buffer,
    bf: Buffer,
}
impl Pair {
    fn new(d: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Self> {
        Ok(Self { full: alloc(d, MAX_ROWS * n * 4)?, bf: alloc(d, MAX_ROWS * n * 2)? })
    }
}

pub(crate) struct PleTarget {
    hash: ple::Hash,
    table: Table,
    key: DenseWeight,
    value: DenseWeight,
    norms: [Buffer; 3],
    conv: Buffer,
    history: Buffer,
    emb: Buffer,
    key_out: Pair,
    value_out: Pair,
    kn: Buffer,
    qn: Buffer,
    gated: Buffer,
    gates: Buffer,
    normalized: Buffer,
    out: Buffer,
    sums: [Buffer; 2],
    pipes: [Obj<dyn MTLComputePipelineState>; 3],
    previous: [u32; 2],
    pending: Vec<u32>,
}
fn small(pack: &Pack, name: &str, n: usize) -> Result<Vec<u16>> {
    let (t, b, _) = pack.tensor(&format!("{PREFIX}.{name}"))?;
    ensure!(t.dtype == "BF16" && t.bytes() as usize == n * 2, "PLE small tensor {name}");
    Ok(b.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect())
}
impl PleTarget {
    pub(crate) fn new(d: &ProtocolObject<dyn MTLDevice>, pack: &Pack, root: &Path) -> Result<Self> {
        let key = Affine8::load(pack, &format!("{PREFIX}.key_proj"), H)?;
        let value = Affine8::load(pack, &format!("{PREFIX}.value_proj"), H)?;
        ensure!((key.k, key.n, value.k, value.n) == (H, F, H, H), "target PLE projections");
        let buf = |v: &[u16]| -> Result<Buffer> {
            let b = alloc(d, v.len() * 2)?;
            upload(&b, v)?;
            Ok(b)
        };
        let norms = [
            buf(&small(pack, "norm_key.weight", F)?)?,
            buf(&small(pack, "norm_query.weight", F)?)?,
            buf(&small(pack, "norm_conv.weight", F)?)?,
        ];
        let conv = buf(&small(pack, "conv1d.weight", F * 4)?)?;
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let lib = d
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../kernels/ple_target.metal")),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("PLE compile: {e}"))?;
        let pipe = |name: &str| -> Result<Obj<dyn MTLComputePipelineState>> {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).context("PLE function")?;
            d.newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("PLE pipeline {name}: {e}"))
        };
        let s = Self {
            hash: ple::Hash::load_target(pack)?,
            table: Table::open(root)?,
            key: DenseWeight::new(d, &key)?,
            value: DenseWeight::new(d, &value)?,
            norms,
            conv,
            history: alloc(d, HISTORY * F * 2)?,
            emb: alloc(d, MAX_ROWS * H * 2)?,
            key_out: Pair::new(d, F)?,
            value_out: Pair::new(d, H)?,
            kn: alloc(d, MAX_ROWS * F * 2)?,
            qn: alloc(d, MAX_ROWS * F * 2)?,
            gated: alloc(d, MAX_ROWS * F * 2)?,
            gates: alloc(d, MAX_ROWS * HC * 2)?,
            normalized: alloc(d, MAX_ROWS * F * 2)?,
            out: alloc(d, MAX_ROWS * F * 2)?,
            sums: [
                alloc(d, dense_scratch_bytes(H, F, MAX_ROWS))?,
                alloc(d, dense_scratch_bytes(H, H, MAX_ROWS))?,
            ],
            pipes: [pipe("ple_norm")?, pipe("ple_gate")?, pipe("ple_conv_out")?],
            previous: [ple::EOS; 2],
            pending: vec![],
        };
        Ok(s)
    }
    pub(crate) fn hash(&self) -> &ple::Hash {
        &self.hash
    }
    pub(crate) fn reset(&mut self) -> Result<()> {
        upload(&self.history, &vec![0u16; HISTORY * F])?;
        self.previous = [ple::EOS; 2];
        self.pending.clear();
        Ok(())
    }
    /// Host: hash and gather this forward's rows into the embedding buffer.
    /// R56: warm the page cache for `tokens`' rows, hashed after `previous` (background lane).
    pub(crate) fn prefetch(&self, previous: [u32; 2], tokens: &[u32]) -> Result<()> {
        let ids = self.hash.rows(previous, tokens)?;
        self.table.prefetch(&ids);
        Ok(())
    }
    pub(crate) fn prepare(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= MAX_ROWS, "PLE rows");
        let ids = self.hash.rows(self.previous, tokens)?;
        ensure!(ids.len() == tokens.len() * HEADS, "PLE hash rows");
        let emb = self.table.gather(&ids)?;
        ensure!(emb.len() == tokens.len() * H, "PLE embedding");
        upload(&self.emb, &emb)?;
        self.pending = tokens.to_vec();
        Ok(())
    }
    fn dense_arm(rows: usize) -> Arm {
        // R56: same arm at every row count (row-count-invariant logits).
        let _ = rows;
        Arm::FragmentPacked { splits: 2, residual: true }
    }
    pub(crate) fn encode(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, stream: &Buffer, rows: usize, ops: &DenseOps) -> Result<()> {
        ensure!(rows == self.pending.len(), "PLE prepared rows");
        ops.encode(e, &self.key, &self.emb, &self.key_out.full, &self.key_out.bf, &self.sums[0], rows, 0, Self::dense_arm(rows))?;
        ops.encode(e, &self.value, &self.emb, &self.value_out.full, &self.value_out.bf, &self.sums[1], rows, 0, Self::dense_arm(rows))?;
        crate::metal_ops::barrier(&e);
        let p = [rows as u32, 0, 0, 0];
        let grid = MTLSize { width: HC, height: rows, depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        let bind = |bufs: &[&Buffer], at: usize| unsafe {
            for (i, b) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&p).cast(), 16, at);
        };
        e.setComputePipelineState(&self.pipes[0]);
        bind(&[&self.key_out.bf, &self.norms[0], &self.kn], 3);
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        bind(&[stream, &self.norms[1], &self.qn], 3);
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        crate::metal_ops::barrier(&e);
        e.setComputePipelineState(&self.pipes[1]);
        bind(&[&self.kn, &self.qn, &self.value_out.bf, &self.gated, &self.gates], 5);
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        crate::metal_ops::barrier(&e);
        e.setComputePipelineState(&self.pipes[0]);
        bind(&[&self.gated, &self.norms[2], &self.normalized], 3);
        e.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        crate::metal_ops::barrier(&e);
        e.setComputePipelineState(&self.pipes[2]);
        bind(&[&self.history, &self.normalized, &self.conv, &self.gated, stream, &self.out], 6);
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: (rows * F).div_ceil(256), height: 1, depth: 1 },
            tg,
        );
        crate::metal_ops::barrier(&e);
        Ok(())
    }
    pub(crate) fn output(&self) -> &Buffer {
        &self.out
    }
    /// R58 prefill: (key projection, value projection, [key, query, conv norms], conv weights,
    /// convolution history).
    pub(crate) fn prefill_parts(&self) -> (&DenseWeight, &DenseWeight, &[Buffer; 3], &Buffer, &Buffer) {
        (&self.key, &self.value, &self.norms, &self.conv, &self.history)
    }
    /// R58 prefill, host: the n-gram embedding rows (rows x H) for `tokens` following the
    /// committed tokens (the hash state is not advanced; see prefill_advance).
    pub(crate) fn prefill_embed(&self, tokens: &[u32]) -> Result<Vec<u16>> {
        ensure!(self.pending.is_empty(), "PLE prefill with a forward pending");
        // The hash takes up to 16 tokens after the two before them (its whole state).
        let (mut ids, mut prev) = (Vec::with_capacity(tokens.len() * HEADS), self.previous);
        for piece in tokens.chunks(MAX_ROWS) {
            ids.extend(self.hash.rows(prev, piece)?);
            prev = ple::accepted_tokens(prev, piece, piece.len())?;
        }
        ensure!(ids.len() == tokens.len() * HEADS, "PLE hash rows");
        let emb = self.table.gather(&ids)?;
        ensure!(emb.len() == tokens.len() * H, "PLE embedding");
        Ok(emb)
    }
    /// R58 prefill, host: warm the page cache for all of `tokens`' n-gram rows (background lane;
    /// returns at once), hashed after the committed tokens.
    pub(crate) fn prefill_prefetch(&self, tokens: &[u32]) -> Result<()> {
        let (mut ids, mut prev) = (Vec::with_capacity(tokens.len() * HEADS), self.previous);
        for piece in tokens.chunks(MAX_ROWS) {
            ids.extend(self.hash.rows(prev, piece)?);
            prev = ple::accepted_tokens(prev, piece, piece.len())?;
        }
        self.table.prefetch(&ids);
        Ok(())
    }
    /// R58 prefill, after the chunk completed (its history copy included): the hash keeps the
    /// chunk's last two tokens.
    pub(crate) fn prefill_advance(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(self.pending.is_empty() && !tokens.is_empty(), "PLE prefill advance");
        for piece in tokens.chunks(MAX_ROWS) {
            self.previous = ple::accepted_tokens(self.previous, piece, piece.len())?;
        }
        Ok(())
    }
    /// R58 diagnostic: the convolution history (HISTORY x F BF16).
    pub(crate) fn history_rows(&self) -> &Buffer {
        &self.history
    }
    /// R58 seeding: the two tokens the PLE hash continues from.
    pub(crate) fn previous_tokens(&self) -> [u32; 2] {
        self.previous
    }
    /// Host side after completion: the history keeps the last 9 conv inputs of
    /// [history; accepted rows]; the hash keeps the last two accepted tokens.
    pub(crate) fn commit(&mut self, accepted: usize) -> Result<()> {
        ensure!(accepted <= self.pending.len(), "PLE commit");
        if accepted > 0 {
            unsafe {
                let h = self.history.contents().cast::<u16>().as_ptr();
                let n = self.normalized.contents().cast::<u16>().as_ptr();
                let mut cat = Vec::with_capacity((HISTORY + accepted) * F);
                cat.extend_from_slice(std::slice::from_raw_parts(h, HISTORY * F));
                cat.extend_from_slice(std::slice::from_raw_parts(n, accepted * F));
                std::ptr::copy_nonoverlapping(cat[accepted * F..].as_ptr(), h, HISTORY * F);
            }
            self.previous = ple::accepted_tokens(self.previous, &self.pending, accepted)?;
        }
        self.pending.clear();
        Ok(())
    }
}
