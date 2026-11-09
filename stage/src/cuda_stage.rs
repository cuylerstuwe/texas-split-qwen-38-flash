//! R60 pipeline split: the canonical target's layers [first, 48) as one RTX 5090 stage.
//! The Mac runs the embedding, PLE and layers [0, first); this stage takes the 4-stream HC
//! residual (BF16, rows x 10240) after layer first-1 and returns it after layer 47 (the Mac then
//! runs the final mixer, lm_head and MTP head). Multi-row forwards keep per-row GDN states so a
//! verify can commit any accepted prefix; attention layers append only the accepted rows.
use crate::{
    cuda::{Buffer, Device, Module, PreparedCall},
    cuda_lowbit_attention_layer as al, cuda_lowbit_layer as rl, gdn, hc,
    mcg_moe::Bank,
    projection::Pack,
    qsa, target_attention as a, target_attention_layer, target_layer,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, time::Instant};
const GUARD: usize = 256;
pub enum StageLayer<'m, 'd> {
    Recurrent(rl::Layer<'m, 'd>),
    Attention(al::Layer<'m, 'd>),
}
impl<'d> StageLayer<'_, 'd> {
    fn output(&self) -> &Buffer<'d> {
        match self {
            Self::Recurrent(l) => l.output(),
            Self::Attention(l) => l.output(),
        }
    }
}
pub fn recurrent_policy() -> rl::Policy {
    // all_states: a verify may commit any accepted prefix
    rl::Policy { all_states: true, separate_write: true, ..rl::Policy::canonical() }
}
/// Key splits: the partial kernel walks its keys serially (each step a dependent gather), so few
/// rows get more splits; rows x splits stays within the engine's 16 x 16 scratch.
pub fn attention_policy_for(rows: usize) -> al::Policy {
    let parts = match rows {
        1 => 128,
        2..=4 => 64,
        5..=8 => 32,
        _ => 16,
    };
    // diagnostic: QD_ATTN_PARTS_DIV=n divides the split count (min 8)
    let div: usize = std::env::var("QD_ATTN_PARTS_DIV").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let parts = (parts / div).max(8);
    al::Policy { hc_split: true, separate_write: true, attention_parts: parts, wide: false, expert: rl::Expert::Scalar }
}
pub fn attention_policy() -> al::Policy {
    attention_policy_for(16)
}
pub struct Stage<'m, 'd> {
    d: &'d Device,
    pub first: usize,
    layers: Vec<StageLayer<'m, 'd>>,
    input: Buffer<'d>,
    rows: Cell<usize>,
    position: Cell<usize>,
    pub load_s: f64,
    /// Attention KV tokens per layer (the prefill scratch sizes its QSA score rows by it).
    capacity: usize,
}
fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
impl<'m, 'd> Stage<'m, 'd> {
    /// Load and upload layers [first, 48) of the canonical pack. `capacity`: attention KV tokens.
    pub fn new(
        d: &'d Device,
        m: &'m rl::Modules<'d>,
        attention_module: &'m Module<'d>,
        m8: &'m Module<'d>,
        pack: &Pack,
        first: usize,
        capacity: usize,
    ) -> Result<Self> {
        ensure!((2..48).contains(&first), "stage starts after the PLE layer");
        let t = Instant::now();
        let mut layers = vec![];
        for l in first..48 {
            let bank = Bank::canonical(pack, l, None)?;
            layers.push(if l % 4 == 3 {
                let w = target_attention_layer::Weights::load(pack, l)?;
                StageLayer::Attention(al::Layer::new_canonical(d, m, attention_module, m8, &w, &bank, capacity)?)
            } else {
                let w = target_layer::Weights::load(pack, l)?;
                StageLayer::Recurrent(rl::Layer::new_canonical(d, m, m8, &w, &bank, gdn::MAX_T)?)
            });
        }
        Ok(Self {
            d,
            first,
            layers,
            input: d.alloc(hc::MAX_ROWS * hc::F * 2 + GUARD)?,
            rows: Cell::new(0),
            position: Cell::new(0),
            load_s: t.elapsed().as_secs_f64(),
            capacity,
        })
    }
    pub fn description(&self) -> Value {
        json!({"first":self.first,"layers":self.layers.len(),"load_s":self.load_s,"position":self.position.get(),
            "recurrent_policy":format!("{:?}",recurrent_policy()),"attention_policy":format!("{:?}",attention_policy())})
    }
    pub fn position(&self) -> usize {
        self.position.get()
    }
    /// Rows of the last forward not yet committed (0 when none).
    pub fn pending(&self) -> usize {
        self.rows.get()
    }
    /// Seed layer `layer`'s recurrent state (convolution history, FP32 GDN state).
    pub fn import_recurrent(&self, layer: usize, initial: &gdn::Initial) -> Result<()> {
        match self.layers.get(layer.wrapping_sub(self.first)) {
            // trusted seed: no poison/read-back unless QD_STAGE_SEED_VERIFY=1
            Some(StageLayer::Recurrent(l)) if seed_verify() => l.prepare_initial(initial),
            Some(StageLayer::Recurrent(l)) => l.prepare_initial_lean(initial),
            _ => anyhow::bail!("layer {layer} is not a stage recurrent layer"),
        }
    }
    /// Seed layer `layer`'s attention KV cache and QSA index (both at the same absolute length).
    pub fn import_attention(&self, layer: usize, cache: &a::State, index: &qsa::State) -> Result<()> {
        match self.layers.get(layer.wrapping_sub(self.first)) {
            Some(StageLayer::Attention(l)) if seed_verify() => l.prepare_state(cache, index),
            Some(StageLayer::Attention(l)) => l.prepare_state_lean(cache, index),
            _ => anyhow::bail!("layer {layer} is not a stage attention layer"),
        }
    }
    /// Rotary offset of the rows the next forward/prefill appends (Qwen M-RoPE: text after an image
    /// ropes at row + delta on all three axes; the KV row index is unchanged). 0 for text requests.
    pub fn set_rope_delta(&self, delta: i32) {
        for l in &self.layers {
            if let StageLayer::Attention(at) = l {
                at.set_rope_delta(delta);
            }
        }
    }
    /// After every layer was seeded: the committed sequence length.
    pub fn set_position(&self, position: usize) {
        self.position.set(position);
        self.rows.set(0);
    }
    /// The recurrent and attention state of every stage layer (for the prefix cache).
    pub fn export(&self) -> Result<Vec<Value>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| match l {
                StageLayer::Recurrent(r) => r.initial_lean().map(|s| json!({"layer":self.first+i,"conv":s.conv.len(),"state":s.state.len()})),
                StageLayer::Attention(at) => at.export_state().map(|(c, _)| json!({"layer":self.first+i,"kv_len":c.len})),
            })
            .collect()
    }
    /// Diagnostics after a forward: every layer's output stream (rows x 10240 BF16) and routed
    /// expert ids (rows x 10).
    pub fn trace(&self, rows: usize) -> Result<Vec<(Vec<u16>, Vec<u32>)>> {
        self.layers
            .iter()
            .map(|l| {
                let mut raw = vec![0u8; rows * hc::F * 2];
                l.output().download(&mut raw)?;
                let out = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let ids = match l {
                    StageLayer::Recurrent(r) => r.route_ids(rows)?,
                    StageLayer::Attention(at) => at.route_ids(rows)?,
                };
                Ok((out, ids))
            })
            .collect()
    }
    /// rows x 10240 BF16 HC stream in; the stream after layer 47 out. The rows are positions
    /// [position, position + rows); nothing is committed until the next forward or `commit`.
    /// A previous forward's rows are committed first, in the same submission (no extra sync):
    /// `commit_prev` of them (all when None).
    pub fn forward(&self, x: &[u16], commit_prev: Option<usize>) -> Result<Vec<u16>> {
        let (commits, rows) = self.begin(x, commit_prev)?;
        let calls = self.record_forward(rows)?;
        for c in commits.iter().chain(&calls) {
            c.enqueue()?;
        }
        self.d.sync()?;
        drop(calls);
        drop(commits);
        self.finish(rows)
    }
    /// Graph-replay form, step 1: commit the previous forward (recorded calls to enqueue first;
    /// bookkeeping done), upload `x`, prepare every layer for `rows`. Returns (commit calls, rows).
    pub fn begin(&self, x: &[u16], commit_prev: Option<usize>) -> Result<(Vec<PreparedCall<'_, 'm, 'd>>, usize)> {
        ensure!(!x.is_empty() && x.len() % hc::F == 0 && x.len() / hc::F <= hc::MAX_ROWS, "stage input rows");
        let rows = x.len() / hc::F;
        let mut calls: Vec<PreparedCall<'_, 'm, 'd>> = vec![];
        let prev = self.rows.get();
        if prev > 0 {
            let accepted = commit_prev.unwrap_or(prev);
            ensure!(accepted <= prev, "stage commit beyond the last forward");
            let rp = recurrent_policy();
            for l in &self.layers {
                match l {
                    StageLayer::Recurrent(r) => calls.extend(r.record_commit(accepted, rp)?),
                    StageLayer::Attention(at) => calls.extend(at.record_commit(accepted)?),
                }
            }
            for l in &self.layers {
                match l {
                    StageLayer::Recurrent(r) => r.finish_commit(),
                    StageLayer::Attention(at) => at.finish_commit(accepted)?,
                }
            }
            self.rows.set(0);
            self.position.set(self.position.get() + accepted);
        } else {
            ensure!(commit_prev.is_none_or(|c| c == 0), "stage commit without a forward");
        }
        self.input.upload(&bytes(x))?;
        for l in &self.layers {
            match l {
                StageLayer::Recurrent(r) => r.prepare_resident(rows)?,
                StageLayer::Attention(at) => at.prepare_resident(rows)?,
            }
        }
        Ok((calls, rows))
    }
    /// Graph-replay form, step 2: the forward's calls (the same structure for a given `rows`
    /// at every position, so a captured graph can be re-targeted with Graph::patch_diff).
    pub fn record_forward(&self, rows: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        let (rp, ap) = (recurrent_policy(), attention_policy_for(rows));
        let mut calls: Vec<PreparedCall<'_, 'm, 'd>> = vec![];
        let mut input = &self.input;
        for l in &self.layers {
            let c = match l {
                StageLayer::Recurrent(r) => r.record_from(input, rows, rp)?,
                StageLayer::Attention(at) => at.record_from(input, rows, ap)?,
            };
            calls.extend(c.into_iter().flatten());
            input = l.output();
        }
        Ok(calls)
    }
    /// Graph-replay form, step 3 (after the submission completed): the stream after layer 47.
    pub fn finish(&self, rows: usize) -> Result<Vec<u16>> {
        let mut raw = vec![0u8; rows * hc::F * 2];
        self.layers.last().unwrap().output().download(&mut raw)?;
        let out: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        ensure!(out.iter().all(|&v| bf16::from_bits(v).is_finite()), "stage output finite");
        self.rows.set(rows);
        Ok(out)
    }
    /// Drop the last forward's rows without GPU work (the next seed overwrites every state).
    pub fn discard(&self) -> Result<()> {
        if self.rows.get() > 0 {
            for l in &self.layers {
                match l {
                    StageLayer::Recurrent(r) => r.finish_commit(),
                    StageLayer::Attention(at) => at.finish_commit(0)?,
                }
            }
            self.rows.set(0);
        }
        Ok(())
    }
    /// Keep the first `accepted` rows of the last forward (0..=rows).
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let rows = self.rows.get();
        ensure!(rows > 0 && accepted <= rows, "stage commit after a forward");
        let rp = recurrent_policy();
        let mut calls: Vec<PreparedCall<'_, 'm, 'd>> = vec![];
        for l in &self.layers {
            match l {
                StageLayer::Recurrent(r) => calls.extend(r.record_commit(accepted, rp)?),
                StageLayer::Attention(at) => calls.extend(at.record_commit(accepted)?),
            }
        }
        for c in &calls {
            c.enqueue()?;
        }
        self.d.sync()?;
        drop(calls);
        for l in &self.layers {
            match l {
                StageLayer::Recurrent(r) => r.finish_commit(),
                StageLayer::Attention(at) => at.finish_commit(accepted)?,
            }
        }
        self.rows.set(0);
        self.position.set(self.position.get() + accepted);
        Ok(())
    }
}

/// R60 S4 (prefill v2): chunk scratch plus the two chunk-sized stream buffers a stage prefill
/// ping-pongs between layers.
pub struct Prefill<'m, 'd> {
    pub scratch: rl::PrefillScratch<'m, 'd>,
    streams: [Buffer<'d>; 2],
    sub: usize,
    /// pull-back: rows_cols_copy and, per attention layer, the last raw QSA keys of the last call
    copy: crate::cuda::Kernel<'m, 'd>,
    rings: Vec<Option<Buffer<'d>>>,
    ring_rows: std::cell::Cell<usize>,
}
/// Raw QSA keys a pull-back carries per attention layer (Sushi's ring, QSA_RING_ROWS).
pub const RING_ROWS: usize = 32;
impl<'m, 'd> Stage<'m, 'd> {
    /// R60 S4 (prefill v2): scratch for chunks of up to `rows` rows (attention in `sub`-row
    /// sub-batches). `k4`: kernels/mcg4_tensor.sm120.cubin (cuda_mcg_tensor::module_k4).
    pub fn prefill_scratch(&self, k4: &'m Module<'d>, rows: usize, sub: usize) -> Result<Prefill<'m, 'd>> {
        ensure!(rows > 0 && sub > 0 && sub <= rows, "prefill scratch shape");
        let ns = self
            .layers
            .iter()
            .find_map(|l| match l {
                StageLayer::Attention(a) => Some(a.projection_widths()),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("stage without an attention layer"))?;
        let d = self.d;
        Ok(Prefill {
            scratch: rl::PrefillScratch {
                rows,
                hc: crate::cuda_hc::HcScratch::new(d, rows)?,
                gdn: crate::cuda_gdn::GdnScratch::new(d, rows)?,
                attn: crate::cuda_lowbit_attention::AttnScratch::new(d, rows, sub, self.capacity, ns)?,
                moe: crate::cuda_lowbit_moe::MoeScratch::new(d, rows)?,
                k64: crate::cuda_mcg_moe::K64::new(k4)?,
            },
            streams: [d.alloc(rows * hc::F * 2 + 256)?, d.alloc(rows * hc::F * 2 + 256)?],
            sub,
            copy: k4.kernel("rows_cols_copy")?,
            rings: self
                .layers
                .iter()
                .map(|l| match l {
                    StageLayer::Attention(_) => d.alloc(RING_ROWS * qsa::D * 2 + 256).map(Some),
                    _ => Ok(None),
                })
                .collect::<Result<_>>()?,
            ring_rows: std::cell::Cell::new(0),
        })
    }
    /// R60 S4 (prefill v2): layers [first, 48) over `x` (rows x 10240 BF16 HC stream) at the current
    /// position; every row commits. Returns layer 47's stream for every row (the MTP head's
    /// prefill history needs all of them).
    pub fn prefill(&self, p: &Prefill<'m, 'd>, x: &[u16]) -> Result<Vec<u16>> {
        let mut out = vec![0u8; x.len() * 2];
        self.prefill_raw(p, &bytes(x), &mut out)?;
        let out: Vec<u16> = out.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        ensure!(out.iter().all(|&v| bf16::from_bits(v).is_finite()), "stage prefill output finite");
        Ok(out)
    }
    /// R60 S4 (prefill v2): as `prefill`, streams as little-endian BF16 bytes straight from and into
    /// the caller's memory (the server's shared file; no intermediate copies).
    pub fn prefill_raw(&self, p: &Prefill<'m, 'd>, input: &[u8], output: &mut [u8]) -> Result<()> {
        ensure!(self.rows.get() == 0, "stage prefill with an uncommitted forward");
        ensure!(!input.is_empty() && input.len() % (hc::F * 2) == 0 && output.len() == input.len(), "stage prefill input rows");
        let rows = input.len() / (hc::F * 2);
        ensure!(rows <= p.scratch.rows, "stage prefill chunk exceeds its scratch");
        p.streams[0].upload(input)?;
        for (i, l) in self.layers.iter().enumerate() {
            let (input, output) = (&p.streams[i % 2], &p.streams[(i + 1) % 2]);
            match l {
                StageLayer::Recurrent(r) => {
                    let calls = r.record_prefill(&p.scratch, input, output, rows)?;
                    for c in &calls {
                        c.enqueue()?;
                    }
                }
                StageLayer::Attention(at) => {
                    let views = (0..rows.div_ceil(p.sub))
                        .map(|j| crate::cuda_lowbit_attention::Attention::prefill_views(&p.scratch.attn, j * p.sub, ((j + 1) * p.sub).min(rows)))
                        .collect::<Result<Vec<_>>>()?;
                    let calls = at.record_prefill(&p.scratch, &views, p.sub, input, output, rows)?;
                    for c in &calls {
                        c.enqueue()?;
                    }
                    drop(calls);
                    // the pull-back's raw keys, before the next attention layer reuses the scratch
                    let n = rows.min(RING_ROWS);
                    crate::cuda_lowbit_attention::Attention::record_raw_key_copy(&p.scratch.attn, &p.copy, p.rings[i].as_ref().unwrap(), rows, n)?.enqueue()?;
                    at.finish_prefill(rows)?;
                }
            }
        }
        self.d.sync()?;
        p.streams[self.layers.len() % 2].download(output)?;
        self.position.set(self.position.get() + rows);
        p.ring_rows.set(rows.min(RING_ROWS));
        Ok(())
    }
    /// R60 S4 (prefill v2 pull-back): Sushi's view of layers [first, 48) after the chunk
    /// [from, to) that just ran (the last `prefill_raw` call must have ended at `to` with at least
    /// min(to - from, RING_ROWS) rows). Little-endian blob "QDX1": u32 magic, version 1, first,
    /// layers; u64 from, to; per layer (16-byte aligned) u32 tag, 0 then
    ///   tag 1 recurrent: conv [3, 10240] BF16, state [48, 128, 128] FP32 (after the chunk);
    ///   tag 2 attention: KV positions [from, to) token-major (key codes, key scales, key biases,
    ///     value codes, value scales, value biases), pooled blocks [from/4, to/4) [., 128] BF16,
    ///     u64 n, the last n raw QSA keys [n, 128] BF16 (n = min(to - from, RING_ROWS)).
    pub fn prefill_export(&self, p: &Prefill<'m, 'd>, from: usize, to: usize, out: &mut Vec<u8>) -> Result<()> {
        ensure!(from < to && to == self.position.get(), "prefill export range");
        let n = (to - from).min(RING_ROWS);
        ensure!(p.ring_rows.get() >= n, "prefill export: the last call held {} rows, {n} raw keys needed", p.ring_rows.get());
        out.clear();
        let align = |o: &mut Vec<u8>| o.resize(o.len().div_ceil(16) * 16, 0);
        for v in [0x3158_4451u32, 1, self.first as u32, (48 - self.first) as u32] {
            out.extend(v.to_le_bytes());
        }
        out.extend((from as u64).to_le_bytes());
        out.extend((to as u64).to_le_bytes());
        align(out);
        for (i, l) in self.layers.iter().enumerate() {
            match l {
                StageLayer::Recurrent(r) => {
                    out.extend(1u32.to_le_bytes());
                    out.extend(0u32.to_le_bytes());
                    align(out);
                    let init = r.initial_lean()?;
                    for v in &init.conv {
                        out.extend(v.to_le_bytes());
                    }
                    align(out);
                    for v in &init.state {
                        out.extend(v.to_le_bytes());
                    }
                    align(out);
                }
                StageLayer::Attention(at) => {
                    out.extend(2u32.to_le_bytes());
                    out.extend(0u32.to_le_bytes());
                    align(out);
                    at.export_prefill_rows(from, to, out)?;
                    align(out);
                    out.extend((n as u64).to_le_bytes());
                    align(out);
                    let start = out.len();
                    out.resize(start + n * qsa::D * 2, 0);
                    p.rings[i].as_ref().unwrap().view(0, n * qsa::D * 2)?.download(&mut out[start..])?;
                    align(out);
                }
            }
        }
        Ok(())
    }
    /// R60 S4 (texas-split parking): write the committed state of every stage layer to `w` as a
    /// QDS1 seed blob at the committed position (canon_stage_server's SEED/RESTORE read it back
    /// unchanged): header {u32 0x31534451, u32 1, u32 first, u32 count, u64 position}, then per
    /// layer (16-byte aligned sections) tag 1 recurrent {conv [3*10240] BF16, state [48*128*128]
    /// FP32} or tag 2 attention {u64 len, K codes, K scales, K biases, V codes, V scales, V biases,
    /// pooled blocks, open-group raw keys}. Exact device reads of the committed bytes only, through
    /// a bounded host buffer. Nothing may be pending (commit first). Returns the bytes written.
    pub fn snapshot(&self, w: &mut dyn std::io::Write) -> Result<u64> {
        ensure!(self.rows.get() == 0, "stage snapshot with an uncommitted forward");
        let position = self.position.get();
        struct Out<'w> {
            w: &'w mut dyn std::io::Write,
            at: u64,
        }
        impl Out<'_> {
            fn put(&mut self, b: &[u8]) -> Result<()> {
                self.w.write_all(b)?;
                self.at += b.len() as u64;
                Ok(())
            }
            fn align(&mut self) -> Result<()> {
                let n = ((16 - self.at % 16) % 16) as usize;
                self.put(&[0u8; 16][..n])
            }
        }
        let mut o = Out { w, at: 0 };
        for v in [0x3153_4451u32, 1, self.first as u32, (48 - self.first) as u32] {
            o.put(&v.to_le_bytes())?;
        }
        o.put(&(position as u64).to_le_bytes())?;
        o.align()?;
        const CHUNK: usize = 32 << 20;
        let mut host = vec![0u8; CHUNK];
        for l in &self.layers {
            match l {
                StageLayer::Recurrent(r) => {
                    o.put(&1u32.to_le_bytes())?;
                    o.put(&0u32.to_le_bytes())?;
                    o.align()?;
                    let init = r.initial_lean()?;
                    ensure!(init.conv.len() == 3 * gdn::C && init.state.len() == gdn::STATE, "recurrent snapshot sizes");
                    o.put(&init.conv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                    o.align()?;
                    o.put(&init.state.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                    o.align()?;
                }
                StageLayer::Attention(at) => {
                    o.put(&2u32.to_le_bytes())?;
                    o.put(&0u32.to_le_bytes())?;
                    o.align()?;
                    let (len, ranges) = at.snapshot_ranges()?;
                    ensure!(len == position, "attention snapshot length {len} vs position {position}");
                    o.put(&(len as u64).to_le_bytes())?;
                    o.align()?;
                    for range in ranges.iter().flatten() {
                        let n = range.byte_len();
                        let mut off = 0;
                        while off < n {
                            let k = (n - off).min(CHUNK);
                            range.view(off, k)?.download(&mut host[..k])?;
                            o.put(&host[..k])?;
                            off += k;
                        }
                        o.align()?;
                    }
                }
            }
        }
        o.w.flush()?;
        Ok(o.at)
    }
    /// Every recurrent layer's committed state (exact reads), in layer order. Diagnostic.
    pub fn recurrent_states(&self) -> Result<Vec<gdn::Initial>> {
        self.layers.iter().filter_map(|l| match l {
            StageLayer::Recurrent(r) => Some(r.initial_lean()),
            _ => None,
        }).collect()
    }
    /// Rows per `prefill_raw` call this scratch accepts.
    pub fn prefill_rows(p: &Prefill<'m, 'd>) -> usize {
        p.scratch.rows
    }
}

impl<'m, 'd> Stage<'m, 'd> {
    /// R60 S4 (prefill v2): per-call GPU times of one stage prefill over `x` (R53 completion
    /// stamps on one captured graph of every layer's calls, median of `replays`). The replays
    /// advance the recurrent state; positions are not advanced. Diagnostic only.
    #[cfg(feature = "r53-driver")]
    pub fn prefill_profile(&self, p: &Prefill<'m, 'd>, x: &[u16], replays: usize) -> Result<(Vec<String>, Vec<f64>, f64)> {
        use crate::r53_cal::{read_stamps, stamp_bytes};
        ensure!(self.rows.get() == 0 && !x.is_empty() && x.len() % hc::F == 0, "stage prefill profile input");
        let rows = x.len() / hc::F;
        ensure!(rows <= p.scratch.rows && replays > 0, "stage prefill profile shape");
        p.streams[0].upload(&bytes(x))?;
        let views: Vec<Vec<Vec<Buffer<'d>>>> = self
            .layers
            .iter()
            .map(|l| match l {
                StageLayer::Attention(_) => (0..rows.div_ceil(p.sub))
                    .map(|j| crate::cuda_lowbit_attention::Attention::prefill_views(&p.scratch.attn, j * p.sub, ((j + 1) * p.sub).min(rows)))
                    .collect::<Result<Vec<_>>>(),
                _ => Ok(vec![]),
            })
            .collect::<Result<_>>()?;
        let mut calls = vec![];
        for (i, l) in self.layers.iter().enumerate() {
            let (input, output) = (&p.streams[i % 2], &p.streams[(i + 1) % 2]);
            calls.extend(match l {
                StageLayer::Recurrent(r) => r.record_prefill(&p.scratch, input, output, rows)?,
                StageLayer::Attention(at) => at.record_prefill(&p.scratch, &views[i], p.sub, input, output, rows)?,
            });
        }
        let n = calls.len();
        let names: Vec<String> = calls.iter().map(|c| c.meta().0.to_string()).collect();
        let s = self.d.alloc(stamp_bytes(n))?;
        s.upload(&vec![0u8; stamp_bytes(n)])?;
        self.d.sync()?;
        self.d.r53_stamp(Some(&s), n as u32)?;
        let g = self.d.capture_model_diagnostic(calls);
        self.d.r53_stamp(None, 0)?;
        let g = g?;
        g.launch()?;
        self.d.sync()?;
        read_stamps(&s, n)?;
        let mut per: Vec<Vec<f64>> = vec![vec![]; n];
        let mut spans = vec![];
        for _ in 0..replays {
            g.launch()?;
            self.d.sync()?;
            let st = read_stamps(&s, n)?;
            let start = st[n..].iter().find(|v| v[0] != 0).map(|v| v[1]).ok_or_else(|| anyhow::anyhow!("no chain start stamp"))?;
            ensure!(st[..n].iter().enumerate().all(|(i, v)| v[0] == i as u64 + 1 && v[1] != 0), "prefill profile stamps");
            let mut prev = start as i64;
            for (i, v) in st[..n].iter().enumerate() {
                per[i].push((v[1] as i64 - prev) as f64 / 1e3);
                prev = v[1] as i64;
            }
            spans.push((st[n - 1][1] as i64 - start as i64) as f64 / 1e3);
        }
        drop(g);
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let us: Vec<f64> = per.iter_mut().map(med).collect();
        Ok((names, us, med(&mut spans)))
    }
}

/// QD_STAGE_SEED_VERIFY=1: seed through the verified imports (capacity-wide poison, full read-back).
fn seed_verify() -> bool {
    std::env::var("QD_STAGE_SEED_VERIFY").as_deref() == Ok("1")
}
