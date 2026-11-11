//! Resident packed attention, QSA selection, and prefix-owned KV/index state.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::Linear,
    lowbit_attention::Weights,
    qsa, target_attention as a,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "65f94ed2a8af9a5e2dceb1bbd7e57425da1f0ce9168898a1b390ed1e7ed10631";
const GUARD: usize = 256;
pub const MAX_CAPACITY: usize = a::MAX_REFERENCE_CTX + a::MAX_ROWS;
/// R52: early-launch clones of the nine per-step kernels (candidate arm only).
pub fn r52_attn_enabled() -> bool {
    crate::cuda_affine4::r39_enabled() && std::env::var("QWEN_R52_ATTN").as_deref() == Ok("1")
}
/// R53: r39pdl_attn_preparex (rotary/gate inputs requested before the store loop;
/// same values) in place of R52's early attn_prepare. Needs QWEN_R52_ATTN=1.
pub fn r53_attnx_enabled() -> bool {
    r52_attn_enabled() && std::env::var("QWEN_R53_ATTNX").as_deref() == Ok("1")
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if r53_attnx_enabled() {
        let pin: serde_json::Value = serde_json::from_str(include_str!("../pins/R53-ATTN-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_attention_r53.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r52_attn_enabled() {
        let pin: serde_json::Value = serde_json::from_str(include_str!("../pins/R52-ATTN-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_attention_r52.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_attention.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![255; b.byte_len()])
}
fn raw(b: &Buffer<'_>, n: usize, strict: bool) -> Result<Vec<u8>> {
    ensure!(n + GUARD <= b.byte_len(), "attention read bound");
    let mut out = vec![0; b.byte_len()];
    b.download(&mut out)?;
    let tail = if strict { n } else { b.byte_len() - GUARD };
    ensure!(
        out[tail..].iter().all(|&v| v == 255),
        "attention tail canary"
    );
    out.truncate(n);
    Ok(out)
}
fn bf(b: &Buffer<'_>, n: usize, strict: bool) -> Result<Vec<u16>> {
    let v: Vec<_> = raw(b, n * 2, strict)?
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        v.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "attention BF16 finite/fully written"
    );
    Ok(v)
}
fn fp(b: &Buffer<'_>, n: usize, stats: bool) -> Result<Vec<f32>> {
    let v: Vec<_> = raw(b, n * 4, true)?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if stats {
        ensure!(
            v.chunks_exact(2).all(|p| p[1].is_finite()
                && p[1] >= 0.
                && if p[1] == 0. {
                    p[0] == f32::NEG_INFINITY
                } else {
                    p[0].is_finite()
                }),
            "attention partial mass/max"
        );
    } else {
        ensure!(
            v.iter().all(|v| v.is_finite()),
            "attention FP32 finite/fully written"
        );
    }
    Ok(v)
}
struct Pair<'d> {
    f: Buffer<'d>,
    b: Buffer<'d>,
}
impl<'d> Pair<'d> {
    fn new(d: &'d Device, n: usize) -> Result<Self> {
        Ok(Self {
            f: d.alloc(a::MAX_ROWS * n * 4 + GUARD)?,
            b: d.alloc(a::MAX_ROWS * n * 2 + GUARD)?,
        })
    }
    fn poison(&self) -> Result<()> {
        poison(&self.f)?;
        poison(&self.b)
    }
    fn read(&self, n: usize) -> Result<(Vec<f32>, Vec<u16>)> {
        let f = fp(&self.f, n, false)?;
        let b = bf(&self.b, n, true)?;
        ensure!(
            f.iter()
                .zip(&b)
                .all(|(&f, &b)| bf16::from_f32(f).to_bits() == b),
            "attention projection rounding"
        );
        Ok((f, b))
    }
}
struct Cache<'d> {
    codes: Buffer<'d>,
    scales: Buffer<'d>,
    biases: Buffer<'d>,
}
impl<'d> Cache<'d> {
    fn new(d: &'d Device, capacity: usize) -> Result<Self> {
        Ok(Self {
            codes: d.alloc(capacity * a::KV + GUARD)?,
            scales: d.alloc(capacity * a::META * 2 + GUARD)?,
            biases: d.alloc(capacity * a::META * 2 + GUARD)?,
        })
    }
    fn upload(&self, v: &a::Packed) -> Result<()> {
        for b in [&self.codes, &self.scales, &self.biases] {
            poison(b)?;
        }
        if !v.codes.is_empty() {
            self.codes.upload(&v.codes)?;
            self.scales.upload(bytes(&v.scales))?;
            self.biases.upload(bytes(&v.biases))?;
        }
        Ok(())
    }
    /// R59 incremental refresh: overwrite rows [row0, row0 + v.rows) only (no poisoning).
    fn upload_rows(&self, row0: usize, v: &a::Packed) -> Result<()> {
        if v.codes.is_empty() {
            return Ok(());
        }
        self.codes.upload_at(row0 * a::KV, &v.codes)?;
        self.scales.upload_at(row0 * a::META * 2, bytes(&v.scales))?;
        self.biases.upload_at(row0 * a::META * 2, bytes(&v.biases))
    }
    fn read(&self, len: usize) -> Result<a::Packed> {
        let p = a::Packed {
            codes: raw(&self.codes, len * a::KV, false)?,
            scales: bf(&self.scales, len * a::META, false)?,
            biases: bf(&self.biases, len * a::META, false)?,
        };
        p.validate(len)?;
        Ok(p)
    }
    fn bytes(&self) -> usize {
        self.codes.byte_len() + self.scales.byte_len() + self.biases.byte_len()
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub splits: usize,
    pub projection: usize,
}
#[derive(Clone, Copy)]
struct Position {
    base: usize,
    len: usize,
}
/// R69: one conversation's attention state: KV caches, QSA pooled keys and open-group tail, its bounds and its last
/// forward's bookkeeping. `slot_base` = the row of slot 0's buffers its views start at (0 for slot 0).
struct AttnSeq<'d> {
    /// slot 1..: its own projection outputs (the commit reads the forward's QSA keys); empty for slot 0
    projected: Vec<Pair<'d>>,
    k: Cache<'d>,
    v: Cache<'d>,
    pooled: Buffer<'d>,
    tail: Buffer<'d>,
    next_tail: Buffer<'d>,
    capacity: usize,
    slot_base: usize,
    position: Cell<Option<Position>>,
    rows: Cell<usize>,
    epoch: Cell<u64>,
    valid: Cell<Option<(u64, usize)>>,
}
pub struct Attention<'m, 'd> {
    d: &'d Device,
    linears: Vec<Linear<'m, 'd>>,
    projected: Vec<Pair<'d>>,
    norms: Vec<Buffer<'d>>,
    freq: Buffer<'d>,
    x: Buffer<'d>,
    query: Buffer<'d>,
    key: Buffer<'d>,
    gate: Buffer<'d>,
    index_query: Buffer<'d>,
    scores: Buffer<'d>,
    ids: Buffer<'d>,
    partial: Buffer<'d>,
    stats: Buffer<'d>,
    attention: Buffer<'d>,
    gated: Buffer<'d>,
    kernels: Vec<Kernel<'m, 'd>>,
    /// R52: early-launch clones of kernels 0..9 (attn_prepare .. attn_gate).
    r52k: Option<Vec<Kernel<'m, 'd>>>,
    /// R60: canonical q/k/v/index projections in one affine8_multi4 launch
    multi8: Option<crate::cuda_affine4::Multi8<'m, 'd>>,
    /// R60: attn_part with four-key load batches (kernels/affine8.cu attn8_part)
    part8: Option<Kernel<'m, 'd>>,
    /// R60 S4 (prefill v2): attn8_rows, one block per (KV head, row) sharing each key across the
    /// twelve query heads; QD_ATTN8_ROWS=0 keeps attn8_part in the prefill path
    rows8: Option<Kernel<'m, 'd>>,
    /// R60 S4: attn8_part's decode splits with each key tile dequantized once for the twelve heads
    /// of a KV head (kernels/affine8.cu attn8_rowsp, bit-identical); QD_ATTN8_ROWSP=0: attn8_part
    rowsp8: Option<Kernel<'m, 'd>>,
    /// R60 S4: qsa_select with 1024 threads per row (kernels/affine8.cu qsa_select1k, identical
    /// ids); QD_QSA_SELECT1K=0 keeps the 256-thread kernel
    select8: Option<Kernel<'m, 'd>>,
    /// R60: position-invariant recorded structure at every row count (graph replay + patching)
    stable_all: bool,
    /// R69: one entry per conversation slot (`active` picks the one every call below works on). Slot 1.. hold KV
    /// and pooled keys as views into the top rows of slot 0's buffers (`slot_base`).
    seqs: Vec<AttnSeq<'d>>,
    active: Cell<usize>,
    /// R60 S4: rotary position offset of the rows being appended (rope at base + delta + row).
    /// Qwen's M-RoPE text tokens after an image rope at their row plus the request's decode delta
    /// (all three axes equal); the KV row index is unchanged. 0 for text-only requests.
    rope_delta: Cell<i32>,
}
impl<'m, 'd> Attention<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
        capacity: usize,
    ) -> Result<Self> {
        w.validate()?;
        let linears = w.projections.iter().map(|w| Linear::new(d, affine, w)).collect::<Result<Vec<_>>>()?;
        Self::assemble(d, ops, linears, [&w.norms.q, &w.norms.k, &w.index_norms.q, &w.index_norms.k], capacity, true)
    }
    /// R60 pipeline split: the canonical target's attention layer core (affine8 q/k/v, QSA index
    /// and o projections from `m8`, the affine8 module; attention/QSA kernels unchanged).
    /// Baseline kernels only (no R52/R53 early-launch clones).
    pub fn new_canonical(
        d: &'d Device,
        m8: &'m Module<'d>,
        ops: &'m Module<'d>,
        attn: &crate::target_attention::Weights,
        index: &qsa::Weights,
        capacity: usize,
    ) -> Result<Self> {
        let linears = [&attn.q, &attn.k, &attn.v, &index.projection, &attn.o]
            .into_iter()
            .map(|w| Linear::new_affine8(d, m8, w))
            .collect::<Result<Vec<_>>>()?;
        let mut at = Self::assemble(d, ops, linears, [&attn.norms.q, &attn.norms.k, &index.norms.q, &index.norms.k], capacity, false)?;
        at.multi8 = Some(crate::cuda_affine4::Multi8::load(m8)?);
        at.part8 = Some(m8.kernel("attn8_part")?);
        if std::env::var("QD_ATTN8_ROWS").as_deref() != Ok("0") {
            at.rows8 = Some(m8.kernel("attn8_rows")?);
        }
        if std::env::var("QD_ATTN8_ROWSP").as_deref() != Ok("0") {
            at.rowsp8 = Some(m8.kernel("attn8_rowsp")?);
        }
        if std::env::var("QD_QSA_SELECT1K").as_deref() != Ok("0") {
            at.select8 = Some(m8.kernel("qsa_select1k")?);
        }
        at.stable_all = true;
        Ok(at)
    }
    fn assemble(
        d: &'d Device,
        ops: &'m Module<'d>,
        linears: Vec<Linear<'m, 'd>>,
        norms: [&Vec<u16>; 4],
        capacity: usize,
        fast: bool,
    ) -> Result<Self> {
        ensure!(linears.len() == 5, "attention projections");
        // R60: the canonical stage serves the target's full context (kernels are capacity-agnostic;
        // MAX_CAPACITY is the drafter's CPU-reference qualification bound).
        let bound = if fast { MAX_CAPACITY } else { qsa::MAX_POSITION + a::MAX_ROWS };
        ensure!(
            (a::MAX_ROWS..=bound).contains(&capacity),
            "attention capacity bound"
        );
        let upload = |v: &[u8]| -> Result<Buffer<'d>> {
            let b = d.alloc(v.len())?;
            b.upload(v)?;
            let mut c = vec![0; v.len()];
            b.download(&mut c)?;
            ensure!(c == v, "attention side weight readback");
            Ok(b)
        };
        // R69: allocation order as before (the arena packs by order)
        let projected = [a::QG, a::KV, a::KV, qsa::QK, a::H]
            .into_iter()
            .map(|n| Pair::new(d, n))
            .collect::<Result<_>>()?;
        let norms = norms
            .into_iter()
            .map(|v| upload(bytes(v)))
            .collect::<Result<_>>()?;
        let freq = upload(bytes(&qsa::frequencies()))?;
        let x = d.alloc(a::MAX_ROWS * a::H * 2 + GUARD)?;
        let query = d.alloc(a::MAX_ROWS * a::Q * 2 + GUARD)?;
        let key = d.alloc(a::MAX_ROWS * a::KV * 2 + GUARD)?;
        let gate = d.alloc(a::MAX_ROWS * a::Q * 2 + GUARD)?;
        let pooled = d.alloc(capacity / 4 * qsa::D * 2 + GUARD)?;
        let tail = d.alloc(3 * qsa::D * 2 + GUARD)?;
        let next_tail = d.alloc(3 * qsa::D * 2 + GUARD)?;
        let index_query = d.alloc(a::MAX_ROWS * qsa::HEADS * qsa::D * 2 + GUARD)?;
        let scores = d.alloc(a::MAX_ROWS * (capacity / 4) * 4 + GUARD)?;
        let ids = d.alloc(a::MAX_ROWS * qsa::TOPK * 4 + GUARD)?;
        let k = Cache::new(d, capacity)?;
        let v = Cache::new(d, capacity)?;
        let partial = d.alloc(a::MAX_ROWS * a::Q * 16 * 4 + GUARD)?;
        let stats = d.alloc(a::MAX_ROWS * a::QH * 16 * 2 * 4 + GUARD)?;
        let attention = d.alloc(a::MAX_ROWS * a::Q * 2 + GUARD)?;
        let gated = d.alloc(a::MAX_ROWS * a::Q * 2 + GUARD)?;
        Ok(Self {
            d,
            linears,
            projected,
            norms,
            freq,
            x,
            query,
            key,
            gate,
            index_query,
            scores,
            ids,
            partial,
            stats,
            attention,
            gated,
            seqs: vec![AttnSeq {
                projected: vec![],
                k,
                v,
                pooled,
                tail,
                next_tail,
                capacity,
                slot_base: 0,
                position: Cell::new(None),
                rows: Cell::new(0),
                epoch: Cell::new(0),
                valid: Cell::new(None),
            }],
            active: Cell::new(0),
            kernels: [
                "attn_prepare",
                "attn_pack",
                "qsa_pool",
                "qsa_query",
                "qsa_score",
                "qsa_select",
                "attn_part",
                "attn_merge",
                "attn_gate",
                "qsa_commit_tail",
                "qsa_restore_tail",
            ]
            .into_iter()
            .map(|n| ops.kernel(n))
            .collect::<Result<_>>()?,
            r52k: if fast && r52_attn_enabled() {
                Some(["attn_prepare", "attn_pack", "qsa_pool", "qsa_query", "qsa_score", "qsa_select",
                    "attn_part", "attn_merge", "attn_gate"].into_iter()
                    .map(|n| ops.kernel(&if n == "attn_prepare" && r53_attnx_enabled() { "r39pdl_attn_preparex".to_string() }
                        else { format!("r39pdl_{n}") })).collect::<Result<_>>()?)
            } else { None },
            multi8: None,
            part8: None,
            rows8: None,
            rowsp8: None,
            select8: None,
            stable_all: false,
            rope_delta: Cell::new(0),
        })
    }
    fn sq(&self) -> &AttnSeq<'d> {
        &self.seqs[self.active.get()]
    }
    /// R69: the active slot's projection outputs.
    fn prj(&self) -> &[Pair<'d>] {
        let a = self.active.get();
        if a == 0 { &self.projected } else { &self.seqs[a].projected }
    }
    /// R69: rows the active slot may hold: slot 0 stops below the views of an occupied slot.
    fn limit(&self) -> usize {
        if self.active.get() != 0 {
            return self.sq().capacity;
        }
        self.seqs[1..].iter().filter(|o| o.position.get().is_some()).map(|o| o.slot_base).min().unwrap_or(self.seqs[0].capacity)
    }
    /// R69: add a conversation slot of `rows` rows whose KV and pooled keys are views into the top `rows` rows of slot
    /// 0's buffers (no new KV memory); its QSA tails are its own.
    pub(crate) fn add_slot_view(&mut self, rows: usize) -> Result<usize> {
        let cap = self.seqs[0].capacity;
        let used: usize = self.seqs[1..].iter().map(|o| o.capacity).sum();
        ensure!(rows >= 4 * a::MAX_ROWS && rows % 4 == 0 && rows + used + 4 * a::MAX_ROWS <= cap, "attention slot rows");
        let base = cap - used - rows;
        ensure!(base % 4 == 0, "attention slot base alignment");
        let view = |c: &Cache<'d>| -> Result<Cache<'d>> {
            Ok(Cache {
                codes: c.codes.view(base * a::KV, rows * a::KV + GUARD)?,
                scales: c.scales.view(base * a::META * 2, rows * a::META * 2 + GUARD)?,
                biases: c.biases.view(base * a::META * 2, rows * a::META * 2 + GUARD)?,
            })
        };
        let projected = [a::QG, a::KV, a::KV, qsa::QK, a::H].into_iter().map(|n| Pair::new(self.d, n)).collect::<Result<_>>()?;
        let s0 = &self.seqs[0];
        let seq = AttnSeq {
            projected,
            k: view(&s0.k)?,
            v: view(&s0.v)?,
            pooled: s0.pooled.view(base / 4 * qsa::D * 2, rows / 4 * qsa::D * 2 + GUARD)?,
            tail: self.d.alloc(3 * qsa::D * 2 + GUARD)?,
            next_tail: self.d.alloc(3 * qsa::D * 2 + GUARD)?,
            capacity: rows,
            slot_base: base,
            position: Cell::new(None),
            rows: Cell::new(0),
            epoch: Cell::new(0),
            valid: Cell::new(None),
        };
        self.seqs.push(seq);
        Ok(self.seqs.len() - 1)
    }
    /// R69: the slot every following call works on.
    pub(crate) fn select_slot(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.seqs.len(), "attention slot {slot} of {}", self.seqs.len());
        self.active.set(slot);
        Ok(())
    }
    /// R69: the active slot holds nothing (its rows may be reused; slot 0 may grow into its views again).
    pub(crate) fn drop_slot_state(&self) {
        let sq = self.sq();
        sq.position.set(None);
        sq.rows.set(0);
        sq.valid.set(None);
        sq.epoch.set(sq.epoch.get() + 1);
    }
    /// R69: a seed into slot 1.. must not overlap slot 0's live rows.
    fn slot_seed_check(&self) -> Result<()> {
        if self.active.get() == 0 {
            return Ok(());
        }
        let base = self.sq().slot_base;
        let s0 = &self.seqs[0];
        let live0 = s0.position.get().map_or(0, |p| p.len) + s0.rows.get();
        ensure!(live0 <= base, "attention slot {} views overlap slot 0's rows ({live0} > {base})", self.active.get());
        Ok(())
    }
    /// R60 S4: rotary offset for the next appended rows (see `rope_delta`).
    pub fn set_rope_delta(&self, delta: i32) {
        self.rope_delta.set(delta);
    }
    /// R58: enable the multi-row MMA arm (projection 4) on the attention projections.
    pub(crate) fn enable_rows(&mut self, m: &'m Module<'d>) -> Result<()> {
        for l in &mut self.linears {
            l.enable_rows(m)?;
        }
        Ok(())
    }
    pub fn description(&self) -> Value {
        let extra: usize = [
            &self.freq,
            &self.x,
            &self.query,
            &self.key,
            &self.gate,
            &self.sq().pooled,
            &self.sq().tail,
            &self.sq().next_tail,
            &self.index_query,
            &self.scores,
            &self.ids,
            &self.partial,
            &self.stats,
            &self.attention,
            &self.gated,
        ]
        .into_iter()
        .chain(self.norms.iter())
        .chain(self.prj().iter().flat_map(|p| [&p.f, &p.b]))
        .map(|b| b.byte_len())
        .sum();
        json!({"cubin_sha256":SHA,"capacity":self.sq().capacity,"extra_bytes":extra,"KV_cache_bytes":self.sq().k.bytes()+self.sq().v.bytes(),"projections":self.linears.iter().map(|l|l.description()).collect::<Vec<_>>(),"kernels":self.kernels.iter().map(|k|k.description()).collect::<Vec<_>>(),"host_activation_or_selection_handoffs":0,"state":"append-only candidate KV/pooled cache; metadata prefix and GPU tail gather/restore","selector":"bounded radix8 threshold plus stable ID compaction","scope":"full attention sublayer, no HC/MoE/model sampling"})
    }
    fn invalidate(&self) {
        self.sq().rows.set(0);
        self.sq().valid.set(None);
        self.sq().epoch.set(self.sq().epoch.get() + 1);
    }
    pub fn prepare_state(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.slot_seed_check()?;
        self.invalidate();
        self.sq().position.set(None);
        cache.validate()?;
        index.validate()?;
        ensure!(
            cache.base == index.base && cache.len == index.len && cache.len <= self.limit(),
            "KV/index state alignment/capacity"
        );
        self.sq().k.upload(&cache.k)?;
        self.sq().v.upload(&cache.v)?;
        poison(&self.sq().pooled)?;
        poison(&self.sq().tail)?;
        if !index.pooled.is_empty() {
            self.sq().pooled.upload(bytes(&index.pooled))?;
        }
        if !index.tail.is_empty() {
            self.sq().tail.upload(bytes(&index.tail))?;
        }
        ensure!(
            self.sq().k.read(cache.len)? == cache.k
                && self.sq().v.read(cache.len)? == cache.v
                && bf(&self.sq().pooled, index.pooled.len(), false)? == index.pooled
                && bf(&self.sq().tail, index.tail.len(), true)? == index.tail,
            "attention state upload readback"
        );
        self.sq().position.set(Some(Position {
            base: cache.base,
            len: cache.len,
        }));
        Ok(())
    }
    /// R60 S4: prepare_state for a trusted seed (the pipeline stage). Uploads only the `len`
    /// committed rows of K/V, the pooled index and the raw tail: no capacity-wide poison and no
    /// full-buffer read-back canaries, which cost ~1 GB of eGPU transfers per seed at a 90K-token
    /// capacity whatever the position (~0.2 s per agent turn). Kernels read rows [0, len) and
    /// len % 4 tail keys only.
    pub fn prepare_state_lean(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.slot_seed_check()?;
        self.invalidate();
        self.sq().position.set(None);
        cache.validate()?;
        index.validate()?;
        ensure!(
            cache.base == index.base && cache.len == index.len && cache.len <= self.limit(),
            "KV/index state alignment/capacity"
        );
        self.sq().k.upload_rows(0, &cache.k)?;
        self.sq().v.upload_rows(0, &cache.v)?;
        if !index.pooled.is_empty() {
            self.sq().pooled.upload(bytes(&index.pooled))?;
        }
        if !index.tail.is_empty() {
            self.sq().tail.upload(bytes(&index.tail))?;
        }
        self.sq().position.set(Some(Position {
            base: cache.base,
            len: cache.len,
        }));
        Ok(())
    }
    /// R59 incremental refresh: replace only the rows from `cache.base` on (the target's exact KV
    /// and pooled index for the tokens since q's last seed); earlier rows stay. The window base is
    /// unchanged; `cache.base` must lie inside the held window at a group boundary (multiple of 4
    /// from the base), and no gap may open. QWEN_R59_INCR_VERIFY=1 reads the new rows back.
    pub fn refresh_rows(&self, cache: &a::State, index: &qsa::State) -> Result<()> {
        self.slot_seed_check()?;
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("incremental refresh needs a held window"))?;
        cache.validate()?;
        index.validate()?;
        ensure!(cache.base == index.base && cache.len == index.len, "refresh KV/index alignment");
        ensure!(cache.base >= pos.base && (cache.base - pos.base) % 4 == 0, "refresh start: inside the window, group aligned");
        let row0 = cache.base - pos.base;
        ensure!(row0 <= pos.len && row0 + cache.len <= self.limit(), "refresh rows: no gap, within capacity");
        self.invalidate();
        self.sq().k.upload_rows(row0, &cache.k)?;
        self.sq().v.upload_rows(row0, &cache.v)?;
        if !index.pooled.is_empty() {
            self.sq().pooled.upload_at(row0 / 4 * qsa::D * 2, bytes(&index.pooled))?;
        }
        poison(&self.sq().tail)?;
        if !index.tail.is_empty() {
            self.sq().tail.upload(bytes(&index.tail))?;
        }
        if std::env::var("QWEN_R59_INCR_VERIFY").as_deref() == Ok("1") {
            let k = self.sq().k.read(row0 + cache.len)?;
            ensure!(k.codes[row0 * a::KV..] == cache.k.codes[..], "incremental refresh readback");
        }
        self.sq().position.set(Some(Position { base: pos.base, len: row0 + cache.len }));
        Ok(())
    }
    pub fn prepare(&self, x: &[u16], cache: &a::State, index: &qsa::State) -> Result<()> {
        self.prepare_state(cache, index)?;
        self.prepare_input(x)
    }
    pub fn prepare_input(&self, x: &[u16]) -> Result<()> {
        self.invalidate();
        ensure!(
            !x.is_empty()
                && x.len() % a::H == 0
                && x.len() / a::H <= a::MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "attention input shape/finite"
        );
        self.admit(x.len() / a::H)?;
        poison(&self.x)?;
        self.x.upload(bytes(x))?;
        ensure!(bf(&self.x, x.len(), true)? == x, "attention input readback");
        self.sq().rows.set(x.len() / a::H);
        Ok(())
    }
    fn admit(&self, rows: usize) -> Result<Position> {
        let pos = self
            .sq().position
            .get()
            .ok_or_else(|| anyhow::anyhow!("attention state unavailable"))?;
        ensure!(
            (1..=a::MAX_ROWS).contains(&rows)
                && pos.len + rows <= self.limit()
                && pos.base + pos.len + rows <= qsa::MAX_POSITION,
            "attention append bounds"
        );
        Ok(pos)
    }
    pub fn prepare_resident_input(&self, rows: usize) -> Result<()> {
        self.invalidate();
        self.admit(rows)?;
        self.sq().rows.set(rows);
        Ok(())
    }
    pub fn state(&self) -> Result<(a::State, qsa::State)> {
        let p = self
            .sq().position
            .get()
            .ok_or_else(|| anyhow::anyhow!("attention state unavailable"))?;
        let cache = a::State {
            base: p.base,
            len: p.len,
            k: self.sq().k.read(p.len)?,
            v: self.sq().v.read(p.len)?,
        };
        let index = qsa::State {
            base: p.base,
            len: p.len,
            pooled: bf(&self.sq().pooled, p.len / 4 * qsa::D, false)?,
            tail: bf(&self.sq().tail, p.len % 4 * qsa::D, true)?,
        };
        cache.validate()?;
        index.validate()?;
        Ok((cache, index))
    }
    pub fn output(&self) -> &Buffer<'d> {
        &self.prj()[4].b
    }
    pub fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        let pos = self.admit(rows)?;
        ensure!(
            ([1, 8, 16].contains(&p.splits) || ([32, 64, 128].contains(&p.splits) && rows * p.splits <= 16 * 16))
                && p.projection <= 4
                && x.byte_len() >= rows * a::H * 2,
            "attention record input/policy"
        );
        ensure!(
            self.prj()
                .iter()
                .flat_map(|p| [&p.f, &p.b])
                .all(|b| !x.aliases(b)),
            "attention input/projection alias"
        );
        let o = pos.len as u32;
        // u32 wraparound keeps a negative delta exact: (base + delta) + o + r mod 2^32
        let base = (pos.base as i64 + self.rope_delta.get() as i64) as u32;
        let nb = (pos.len + rows) / 4;
        let new = nb - pos.len / 4;
        let mut calls = vec![];
        if crate::cuda_affine4::ahead_selected() && p.projection == 0 && !self.linears[0].is_affine8() {
            // R38: q/k/v/indexer in one launch, same blocks and outputs.
            let o = &self.prj();
            calls.push(crate::cuda_affine4::record_multi4(
                [&self.linears[0], &self.linears[1], &self.linears[2], &self.linears[3]],
                x,
                [(&o[0].f, &o[0].b), (&o[1].f, &o[1].b), (&o[2].f, &o[2].b), (&o[3].f, &o[3].b)],
                rows,
            )?);
        } else if let (Some(m8), 0) = (&self.multi8, p.projection) {
            let o = &self.prj();
            calls.push(crate::cuda_affine4::record_multi4_affine8(
                m8,
                [&self.linears[0], &self.linears[1], &self.linears[2], &self.linears[3]],
                x,
                [(&o[0].f, &o[0].b), (&o[1].f, &o[1].b), (&o[2].f, &o[2].b), (&o[3].f, &o[3].b)],
                rows,
            )?);
        } else {
            for j in 0..4 {
                calls.push(self.linears[j].record_into(
                    x,
                    &self.prj()[j].f,
                    &self.prj()[j].b,
                    rows,
                    p.projection,
                )?);
            }
        }
        // R56 stable-graph mode (single-row steps): the pool launch is always present (one
        // block; with no completed block it writes only the still-forming one, never read
        // before being overwritten) and the score grid covers the capacity (qsa_score returns
        // for b >= nb): the recorded structure no longer depends on the position.
        // R60: the canonical stage records every row count position-invariantly: the pool grid is
        // the most groups `rows` can complete (extra blocks write still-forming groups, which are
        // recomputed when they complete and are never scored before), the score grid the capacity.
        let stable = (rows == 1 && stable_graph()) || self.stable_all;
        ensure!(!stable || pos.len + rows + 8 <= self.limit(), "stable attention needs capacity slack");
        unsafe {
            if new > 0 || stable {
                calls.push(self.r52_kernel(2).prepare(
                    vec![
                        Arg::Buffer(&self.prj()[3].b),
                        Arg::Buffer(&self.sq().tail),
                        Arg::Buffer(&self.norms[3]),
                        Arg::Buffer(&self.freq),
                        Arg::Buffer(&self.sq().pooled),
                        Arg::U32(o),
                        Arg::U32(base),
                    ],
                    [if stable { new.max(rows.div_ceil(4)).max(1) } else { new } as u32, 1, 1],
                    [32, 1, 1],
                )?);
            }
            calls.push(self.r52_kernel(3).prepare(
                vec![
                    Arg::Buffer(&self.prj()[3].b),
                    Arg::Buffer(&self.norms[2]),
                    Arg::Buffer(&self.freq),
                    Arg::Buffer(&self.index_query),
                    Arg::U32(o),
                    Arg::U32(base),
                ],
                [4, rows as u32, 1],
                [32, 1, 1],
            )?);
            if nb > 0 || stable {
                calls.push(self.r52_kernel(4).prepare(
                    vec![
                        Arg::Buffer(&self.index_query),
                        Arg::Buffer(&self.sq().pooled),
                        Arg::Buffer(&self.scores),
                        Arg::U32(nb as u32),
                    ],
                    [if stable { (self.sq().capacity / 4).div_ceil(4) } else { nb.div_ceil(4) } as u32, rows as u32, 1],
                    [128, 1, 1],
                )?);
            }
            let (select, threads) = match &self.select8 {
                Some(k) => (k, 1024),
                None => (self.r52_kernel(5), 256),
            };
            calls.push(select.prepare(
                vec![
                    Arg::Buffer(&self.scores),
                    Arg::Buffer(&self.ids),
                    Arg::U32(o),
                    Arg::U32(rows as u32),
                ],
                [1, rows as u32, 1],
                [threads, 1, 1],
            )?);
            calls.push(self.r52_kernel(0).prepare(
                vec![
                    Arg::Buffer(&self.prj()[0].b),
                    Arg::Buffer(&self.prj()[1].b),
                    Arg::Buffer(&self.norms[0]),
                    Arg::Buffer(&self.norms[1]),
                    Arg::Buffer(&self.freq),
                    Arg::Buffer(&self.query),
                    Arg::Buffer(&self.key),
                    Arg::Buffer(&self.gate),
                    Arg::U32(o),
                    Arg::U32(base),
                ],
                [26, rows as u32, 1],
                [32, 1, 1],
            )?);
            calls.push(self.r52_kernel(1).prepare(
                vec![
                    Arg::Buffer(&self.key),
                    Arg::Buffer(&self.prj()[2].b),
                    Arg::Buffer(&self.sq().k.codes),
                    Arg::Buffer(&self.sq().k.scales),
                    Arg::Buffer(&self.sq().k.biases),
                    Arg::Buffer(&self.sq().v.codes),
                    Arg::Buffer(&self.sq().v.scales),
                    Arg::Buffer(&self.sq().v.biases),
                    Arg::U32(o),
                ],
                [(rows * 8) as u32, 2, 1],
                [32, 1, 1],
            )?);
            // R60 S4: the dequant-once arm (same ABI; block = KV head x row x split, 12 warps)
            let (part, grid, block) = match (&self.rowsp8, &self.part8) {
                (Some(k), Some(_)) => (k, [2, rows as u32, p.splits as u32], [384, 1, 1]),
                _ => (self.part8.as_ref().unwrap_or(self.r52_kernel(6)), [24, rows as u32, p.splits as u32], [32, 1, 1]),
            };
            calls.push(part.prepare(
                vec![
                    Arg::Buffer(&self.query),
                    Arg::Buffer(&self.sq().k.codes),
                    Arg::Buffer(&self.sq().v.codes),
                    Arg::Buffer(&self.sq().k.scales),
                    Arg::Buffer(&self.sq().k.biases),
                    Arg::Buffer(&self.sq().v.scales),
                    Arg::Buffer(&self.sq().v.biases),
                    Arg::Buffer(&self.ids),
                    Arg::Buffer(&self.partial),
                    Arg::Buffer(&self.stats),
                    Arg::U32(o),
                    Arg::U32(u32::from(pos.len + rows > qsa::BUDGET + 3)),
                    Arg::U32(p.splits as u32),
                ],
                grid,
                block,
            )?);
            calls.push(self.r52_kernel(7).prepare(
                vec![
                    Arg::Buffer(&self.partial),
                    Arg::Buffer(&self.stats),
                    Arg::Buffer(&self.attention),
                    Arg::U32(p.splits as u32),
                ],
                [24, rows as u32, 1],
                [256, 1, 1],
            )?);
            calls.push(self.r52_kernel(8).prepare(
                vec![
                    Arg::Buffer(&self.attention),
                    Arg::Buffer(&self.gate),
                    Arg::Buffer(&self.gated),
                    Arg::U32((rows * a::Q) as u32),
                ],
                [(rows * a::Q).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        calls.push(self.linears[4].record_into(
            &self.gated,
            &self.prj()[4].f,
            &self.prj()[4].b,
            rows,
            p.projection,
        )?);
        Ok(calls)
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        let rows = self.sq().rows.get();
        ensure!(
            rows > 0 && (1..=8).contains(&repetitions),
            "attention plan readiness/bound"
        );
        let mut c = self.record_from(&self.x, rows, p)?;
        let calls = c.len();
        ensure!(calls * repetitions <= 128, "attention graph budget");
        let exec = if graph {
            for _ in 1..repetitions {
                c.extend(self.record_from(&self.x, rows, p)?);
            }
            Executor::Graph(self.d.capture(c)?)
        } else {
            Executor::Prepared(c)
        };
        Ok(Plan {
            owner: self,
            exec,
            rows,
            p,
            repetitions,
            calls,
            epoch: self.sq().epoch.get(),
        })
    }
    pub(crate) fn poison(&self) -> Result<()> {
        self.sq().valid.set(None);
        for p in self.prj() {
            p.poison()?;
        }
        for b in [
            &self.query,
            &self.key,
            &self.gate,
            &self.index_query,
            &self.scores,
            &self.ids,
            &self.partial,
            &self.stats,
            &self.attention,
            &self.gated,
        ] {
            poison(b)?;
        }
        Ok(())
    }
    pub(crate) fn collect(&self, rows: usize, p: Policy) -> Result<Run> {
        let pos = self.admit(rows)?;
        let mut projected = vec![];
        let mut projection_raw = vec![];
        for (j, n) in [a::QG, a::KV, a::KV, qsa::QK].into_iter().enumerate() {
            let (f, b) = self.prj()[j].read(rows * n)?;
            projection_raw.push(f);
            projected.push(b);
        }
        let index_projected = projected.pop().unwrap();
        let v = projected.pop().unwrap();
        let k = projected.pop().unwrap();
        let qg = projected.pop().unwrap();
        let input = a::Input { qg, k, v };
        input.validate()?;
        let prepared = a::Prepared {
            q: bf(&self.query, rows * a::Q, true)?,
            k: bf(&self.key, rows * a::KV, true)?,
            gate: bf(&self.gate, rows * a::Q, true)?,
        };
        let total = pos.len + rows;
        let cache = a::State {
            base: pos.base,
            len: total,
            k: self.sq().k.read(total)?,
            v: self.sq().v.read(total)?,
        };
        cache.validate()?;
        let index_queries = bf(&self.index_query, rows * 4 * qsa::D, true)?;
        let scores = fp(&self.scores, rows * (total / 4), false)?;
        let ids = raw(&self.ids, rows * qsa::TOPK * 4, true)?
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let selection = qsa::Selection {
            offset: pos.len,
            rows,
            total,
            ids,
            counts: (0..rows)
                .map(|r| ((pos.len + r + 1) / 4).min(qsa::TOPK))
                .collect(),
        };
        selection.validate()?;
        let expected = qsa::select(&scores, pos.len, rows)?;
        ensure!(
            expected.ids == selection.ids,
            "CUDA selector conditional exactness"
        );
        let pooled = bf(&self.sq().pooled, total / 4 * qsa::D, false)?;
        let partial = fp(&self.partial, rows * a::Q * p.splits, false)?;
        let stats = fp(&self.stats, rows * a::QH * p.splits * 2, true)?;
        let attention = bf(&self.attention, rows * a::Q, true)?;
        let gated = bf(&self.gated, rows * a::Q, true)?;
        let (output_raw, output) = self.prj()[4].read(rows * a::H)?;
        self.sq().valid.set(Some((self.sq().epoch.get(), rows)));
        Ok(Run {
            projected: input,
            index_projected,
            projection_raw,
            prepared,
            cache,
            index_queries,
            pooled,
            scores,
            selection,
            partial,
            stats,
            attention,
            gated,
            output_raw,
            output,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    /// R56 lean drafting commit: the tail gather/restore kernels as prepared calls.
    pub(crate) fn record_commit(&self, accepted: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        let rows = self.sq().rows.get();
        ensure!(rows > 0 && accepted <= rows, "attention lean prefix");
        let pos = self.admit(rows)?;
        let gather = unsafe {
            self.kernels[9].prepare(
                vec![
                    Arg::Buffer(&self.prj()[3].b),
                    Arg::Buffer(&self.sq().tail),
                    Arg::Buffer(&self.sq().next_tail),
                    Arg::U32(pos.len as u32),
                    Arg::U32(accepted as u32),
                ],
                [3, 1, 1],
                [128, 1, 1],
            )?
        };
        let restore = unsafe {
            self.kernels[10].prepare(
                vec![Arg::Buffer(&self.sq().next_tail), Arg::Buffer(&self.sq().tail)],
                [3, 1, 1],
                [128, 1, 1],
            )?
        };
        Ok(vec![gather, restore])
    }
    /// R58 seeding: the committed KV cache, index pool and raw tail as import structs: the inverse
    /// of prepare_state.
    pub(crate) fn export_state(&self) -> Result<(a::State, qsa::State)> {
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("attention state not prepared"))?;
        let cache = a::State { base: pos.base, len: pos.len, k: self.sq().k.read(pos.len)?, v: self.sq().v.read(pos.len)? };
        let index = qsa::State {
            base: pos.base,
            len: pos.len,
            pooled: if pos.len >= qsa::RATIO { bf(&self.sq().pooled, pos.len / qsa::RATIO * qsa::D, false)? } else { vec![] },
            // the rows past len % RATIO hold stale data after decode steps: no canary check here
            tail: if pos.len % qsa::RATIO > 0 { bf(&self.sq().tail, pos.len % qsa::RATIO * qsa::D, false)? } else { vec![] },
        };
        cache.validate()?;
        index.validate()?;
        Ok((cache, index))
    }
    /// R66d (end-of-turn pull-back): the committed open group's raw keys, [len % 4, 128] BF16 bytes.
    pub(crate) fn open_keys(&self) -> Result<Vec<u8>> {
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("attention state not prepared"))?;
        let mut out = vec![0u8; pos.len % qsa::RATIO * qsa::D * 2];
        if !out.is_empty() {
            self.sq().tail.view(0, out.len())?.download(&mut out)?;
        }
        Ok(out)
    }
    /// R66 (texas-split incremental seed): the committed K codes of rows [from, to), token-major
    /// (`a::KV` bytes per row, the layout seeds upload), for the seed's lineage check.
    pub(crate) fn key_codes(&self, from: usize, to: usize) -> Result<Vec<u8>> {
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("attention state not prepared"))?;
        ensure!(pos.base == 0 && from <= to && to <= pos.len, "key codes [{from}, {to}) outside the held [0, {})", pos.len);
        let mut out = vec![0u8; (to - from) * a::KV];
        if to > from {
            self.sq().k.codes.view(from * a::KV, out.len())?.download(&mut out)?;
        }
        Ok(out)
    }
    /// R60 S4 (texas-split parking): the committed state as device ranges, in the order of the
    /// stage seed's attention section: K codes, K scales, K biases, V codes, V scales, V biases
    /// (rows [0, len)), pooled blocks [0, len/4) and the open group's raw keys [0, len % 4). Exactly
    /// the committed bytes (no capacity-wide read-back); None for an empty range. Returns len.
    pub(crate) fn snapshot_ranges(&self) -> Result<(usize, Vec<Option<Buffer<'d>>>)> {
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("attention state not prepared"))?;
        ensure!(pos.base == 0, "snapshot of a windowed attention state (base {})", pos.base);
        let len = pos.len;
        let range = |b: &Buffer<'d>, n: usize| if n == 0 { Ok(None) } else { b.view(0, n).map(Some) };
        let ranges = vec![
            range(&self.sq().k.codes, len * a::KV)?,
            range(&self.sq().k.scales, len * a::META * 2)?,
            range(&self.sq().k.biases, len * a::META * 2)?,
            range(&self.sq().v.codes, len * a::KV)?,
            range(&self.sq().v.scales, len * a::META * 2)?,
            range(&self.sq().v.biases, len * a::META * 2)?,
            range(&self.sq().pooled, len / qsa::RATIO * qsa::D * 2)?,
            range(&self.sq().tail, len % qsa::RATIO * qsa::D * 2)?,
        ];
        Ok((len, ranges))
    }
    /// R56: the committed raw index tail (the KV cache and pool are append-only by length).
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        vec![&self.sq().tail]
    }
    pub(crate) fn position_meta(&self) -> Option<(usize, usize)> {
        self.sq().position.get().map(|p| (p.base, p.len))
    }
    pub(crate) fn set_position_meta(&self, meta: Option<(usize, usize)>) {
        self.invalidate();
        self.sq().position.set(meta.map(|(base, len)| Position { base, len }));
    }
    pub(crate) fn finish_commit(&self, accepted: usize) -> Result<()> {
        let pos = self.admit(self.sq().rows.get())?;
        self.invalidate();
        self.sq().position.set(Some(Position { base: pos.base, len: pos.len + accepted }));
        Ok(())
    }
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let (epoch, rows) = self
            .sq().valid
            .get()
            .ok_or_else(|| anyhow::anyhow!("attention commit requires completed output"))?;
        ensure!(
            epoch == self.sq().epoch.get() && accepted <= rows,
            "attention commit epoch/prefix"
        );
        let pos = self.admit(rows)?;
        poison(&self.sq().next_tail)?;
        let gather = unsafe {
            self.kernels[9].prepare(
                vec![
                    Arg::Buffer(&self.prj()[3].b),
                    Arg::Buffer(&self.sq().tail),
                    Arg::Buffer(&self.sq().next_tail),
                    Arg::U32(pos.len as u32),
                    Arg::U32(accepted as u32),
                ],
                [3, 1, 1],
                [128, 1, 1],
            )?
        };
        let restore = unsafe {
            self.kernels[10].prepare(
                vec![Arg::Buffer(&self.sq().next_tail), Arg::Buffer(&self.sq().tail)],
                [3, 1, 1],
                [128, 1, 1],
            )?
        };
        self.invalidate();
        self.sq().position.set(None);
        gather.enqueue()?;
        restore.enqueue()?;
        self.d.sync()?;
        self.sq().position.set(Some(Position {
            base: pos.base,
            len: pos.len + accepted,
        }));
        Ok(())
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Attention<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    p: Policy,
    repetitions: usize,
    pub calls: usize,
    epoch: u64,
}
pub struct Run {
    pub projected: a::Input,
    pub index_projected: Vec<u16>,
    pub projection_raw: Vec<Vec<f32>>,
    pub prepared: a::Prepared,
    pub cache: a::State,
    pub index_queries: Vec<u16>,
    pub pooled: Vec<u16>,
    pub scores: Vec<f32>,
    pub selection: qsa::Selection,
    pub partial: Vec<f32>,
    pub stats: Vec<f32>,
    pub attention: Vec<u16>,
    pub gated: Vec<u16>,
    pub output_raw: Vec<f32>,
    pub output: Vec<u16>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        ensure!(
            a.sq().epoch.get() == self.epoch && a.sq().rows.get() == self.rows,
            "attention plan invalidated"
        );
        a.admit(self.rows)?;
        a.poison()?;
        a.d.sync()?;
        let start = Instant::now();
        match &self.exec {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(c) => {
                for _ in 0..self.repetitions {
                    for call in c {
                        call.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6;
        let t = Instant::now();
        a.d.sync()?;
        let wait_us = t.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut r = a.collect(self.rows, self.p)?;
        r.total_us = total_us;
        r.enqueue_us = enqueue_us;
        r.wait_us = wait_us;
        Ok(r)
    }
}

impl<'m, 'd> Attention<'m, 'd> {
    /// R52: the per-step kernel i, or its early-launch clone in the candidate arm.
    fn r52_kernel(&self, i: usize) -> &Kernel<'m, 'd> {
        match &self.r52k {
            Some(v) if i < 9 && crate::cuda_affine4::pdl_selected() => &v[i],
            _ => &self.kernels[i],
        }
    }
}

/// R56: position-invariant recorded structure for single-row graph replay.
pub(crate) fn stable_graph() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN_R56_STABLE_GRAPH").as_deref() == Ok("1"))
}

/// R60 S4 (prefill v2): attention intermediates for chunk-sized row counts, shared by every
/// attention stage layer. Scores, partials and stats cover one row sub-batch (`sub` rows): the
/// per-row QSA score row is capacity/4 floats.
pub struct AttnScratch<'d> {
    rows: usize,
    sub: usize,
    capacity: usize,
    proj: Vec<(Buffer<'d>, Buffer<'d>)>,
    query: Buffer<'d>,
    key: Buffer<'d>,
    gate: Buffer<'d>,
    index_query: Buffer<'d>,
    ids: Buffer<'d>,
    attention: Buffer<'d>,
    gated: Buffer<'d>,
    scores: Buffer<'d>,
    partial: Buffer<'d>,
    stats: Buffer<'d>,
}
impl<'d> AttnScratch<'d> {
    /// `ns`: the five projection widths (q, k, v, index, out) of the stage's attention layers.
    pub fn new(d: &'d Device, rows: usize, sub: usize, capacity: usize, ns: [usize; 5]) -> Result<Self> {
        Ok(Self {
            rows,
            sub,
            capacity,
            proj: ns.iter().map(|&n| Ok((d.alloc(rows * n * 4 + GUARD)?, d.alloc(rows * n * 2 + GUARD)?))).collect::<Result<_>>()?,
            query: d.alloc(rows * a::Q * 2 + GUARD)?,
            key: d.alloc(rows * a::KV * 2 + GUARD)?,
            gate: d.alloc(rows * a::Q * 2 + GUARD)?,
            index_query: d.alloc(rows * qsa::HEADS * qsa::D * 2 + GUARD)?,
            ids: d.alloc(rows * qsa::TOPK * 4 + GUARD)?,
            attention: d.alloc(rows * a::Q * 2 + GUARD)?,
            gated: d.alloc(rows * a::Q * 2 + GUARD)?,
            scores: d.alloc(sub * (capacity / 4) * 4 + GUARD)?,
            partial: d.alloc(sub * a::Q * 4 + GUARD)?,
            stats: d.alloc(sub * a::QH * 2 * 4 + GUARD)?,
        })
    }
    /// The block output (BF16 [rows, H]) of the last recorded prefill.
    pub fn output(&self) -> &Buffer<'d> {
        &self.proj[4].1
    }
}
impl<'m, 'd> Attention<'m, 'd> {
    pub(crate) fn projection_widths(&self) -> [usize; 5] {
        [self.linears[0].n(), self.linears[1].n(), self.linears[2].n(), self.linears[3].n(), self.linears[4].n()]
    }
    fn admit_prefill(&self, rows: usize) -> Result<Position> {
        let pos = self.sq().position.get().ok_or_else(|| anyhow::anyhow!("attention state unavailable"))?;
        ensure!(rows > 0 && pos.len + rows <= self.limit() && pos.base + pos.len + rows <= qsa::MAX_POSITION, "attention prefill append bounds");
        Ok(pos)
    }
    /// R60 S4 (prefill v2): the attention block for chunk rows at the current position: the four
    /// in-projections (64-token tensor tile), the QSA pool append for completed groups, index
    /// queries, q/k norms + rope, the affine8 KV append for every row; then, per row sub-batch,
    /// QSA scores + top-k selection, sparse attention parts and the merge; then the gate and
    /// out_proj. KV and pooled entries are written in place; `record_commit_prefill` updates the
    /// tail and `finish_prefill` advances the position.
    pub(crate) fn record_prefill<'a>(&'a self, s: &'a AttnScratch<'d>, x: &'a Buffer<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        let pos = self.admit_prefill(rows)?;
        ensure!(rows <= s.rows && s.capacity >= self.sq().capacity && x.byte_len() >= rows * a::H * 2, "attention prefill shape");
        ensure!(s.proj.iter().all(|(f, b)| !x.aliases(f) && !x.aliases(b)), "attention prefill alias");
        let m8 = self.multi8.as_ref().ok_or_else(|| anyhow::anyhow!("attention prefill needs the canonical affine8 layer"))?;
        ensure!(self.part8.is_some(), "attention prefill needs attn8_part");
        let (o, base) = (pos.len, (pos.base as i64 + self.rope_delta.get() as i64) as u32 as usize);
        let nb = (o + rows) / 4;
        let new = nb - o / 4;
        let mut calls = vec![crate::cuda_affine4::record_multi4_affine8_rows(
            m8,
            [&self.linears[0], &self.linears[1], &self.linears[2], &self.linears[3]],
            x,
            [(&s.proj[0].0, &s.proj[0].1), (&s.proj[1].0, &s.proj[1].1), (&s.proj[2].0, &s.proj[2].1), (&s.proj[3].0, &s.proj[3].1)],
            rows,
        )?];
        let k = &self.kernels;
        // SAFETY: the decode path's kernel ABIs (kernels/lowbit_attention.cu, affine8.cu attn8_part);
        // whole-chunk buffers are sized for s.rows, sub-batch buffers for s.sub rows.
        unsafe {
            if new > 0 {
                calls.push(k[2].prepare(
                    vec![Arg::Buffer(&s.proj[3].1), Arg::Buffer(&self.sq().tail), Arg::Buffer(&self.norms[3]), Arg::Buffer(&self.freq),
                        Arg::Buffer(&self.sq().pooled), Arg::U32(o as u32), Arg::U32(base as u32)],
                    [new as u32, 1, 1],
                    [32, 1, 1],
                )?);
            }
            calls.push(k[3].prepare(
                vec![Arg::Buffer(&s.proj[3].1), Arg::Buffer(&self.norms[2]), Arg::Buffer(&self.freq), Arg::Buffer(&s.index_query), Arg::U32(o as u32), Arg::U32(base as u32)],
                [4, rows as u32, 1],
                [32, 1, 1],
            )?);
            calls.push(k[0].prepare(
                vec![Arg::Buffer(&s.proj[0].1), Arg::Buffer(&s.proj[1].1), Arg::Buffer(&self.norms[0]), Arg::Buffer(&self.norms[1]), Arg::Buffer(&self.freq),
                    Arg::Buffer(&s.query), Arg::Buffer(&s.key), Arg::Buffer(&s.gate), Arg::U32(o as u32), Arg::U32(base as u32)],
                [26, rows as u32, 1],
                [32, 1, 1],
            )?);
            calls.push(k[1].prepare(
                vec![Arg::Buffer(&s.key), Arg::Buffer(&s.proj[2].1), Arg::Buffer(&self.sq().k.codes), Arg::Buffer(&self.sq().k.scales), Arg::Buffer(&self.sq().k.biases),
                    Arg::Buffer(&self.sq().v.codes), Arg::Buffer(&self.sq().v.scales), Arg::Buffer(&self.sq().v.biases), Arg::U32(o as u32)],
                [(rows * 8) as u32, 2, 1],
                [32, 1, 1],
            )?);
        }
        Ok(calls)
    }
    /// R60 S4 (prefill v2): the per-sub-batch attention calls ([r0, r1) of the chunk), recorded after
    /// `record_prefill`. Views into the chunk buffers are returned with the calls so they outlive
    /// the recording.
    pub(crate) fn record_prefill_attend<'a>(
        &'a self,
        s: &'a AttnScratch<'d>,
        views: &'a [Buffer<'d>],
        r0: usize,
        r1: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        let pos = self.admit_prefill(r1)?;
        let rs = r1 - r0;
        ensure!(rs > 0 && rs <= s.sub && views.len() == 4, "attention prefill sub-batch");
        let [iq, ids, q, att] = [&views[0], &views[1], &views[2], &views[3]];
        let o = pos.len + r0;
        let nb = (pos.len + r1) / 4;
        let part = self.part8.as_ref().ok_or_else(|| anyhow::anyhow!("attention prefill needs attn8_part"))?;
        let k = &self.kernels;
        let mut calls = vec![];
        // SAFETY: as record_prefill; views are bounds-checked slices of the chunk buffers.
        unsafe {
            if nb > 0 {
                calls.push(k[4].prepare(vec![Arg::Buffer(iq), Arg::Buffer(&self.sq().pooled), Arg::Buffer(&s.scores), Arg::U32(nb as u32)], [nb.div_ceil(4) as u32, rs as u32, 1], [128, 1, 1])?);
            }
            calls.push(k[5].prepare(vec![Arg::Buffer(&s.scores), Arg::Buffer(ids), Arg::U32(o as u32), Arg::U32(rs as u32)], [1, rs as u32, 1], [256, 1, 1])?);
            if let Some(rows8) = &self.rows8 {
                calls.push(rows8.prepare(
                    vec![Arg::Buffer(q), Arg::Buffer(&self.sq().k.codes), Arg::Buffer(&self.sq().v.codes), Arg::Buffer(&self.sq().k.scales), Arg::Buffer(&self.sq().k.biases),
                        Arg::Buffer(&self.sq().v.scales), Arg::Buffer(&self.sq().v.biases), Arg::Buffer(ids), Arg::Buffer(&s.partial), Arg::Buffer(&s.stats),
                        Arg::U32(o as u32), Arg::U32(u32::from(o + rs > qsa::BUDGET + 3))],
                    [2, rs as u32, 1],
                    [384, 1, 1],
                )?);
            } else {
                calls.push(part.prepare(
                    vec![Arg::Buffer(q), Arg::Buffer(&self.sq().k.codes), Arg::Buffer(&self.sq().v.codes), Arg::Buffer(&self.sq().k.scales), Arg::Buffer(&self.sq().k.biases),
                        Arg::Buffer(&self.sq().v.scales), Arg::Buffer(&self.sq().v.biases), Arg::Buffer(ids), Arg::Buffer(&s.partial), Arg::Buffer(&s.stats),
                        Arg::U32(o as u32), Arg::U32(u32::from(o + rs > qsa::BUDGET + 3)), Arg::U32(1)],
                    [24, rs as u32, 1],
                    [32, 1, 1],
                )?);
            }
            calls.push(k[7].prepare(vec![Arg::Buffer(&s.partial), Arg::Buffer(&s.stats), Arg::Buffer(att), Arg::U32(1)], [24, rs as u32, 1], [256, 1, 1])?);
        }
        Ok(calls)
    }
    /// R60 S4 (prefill v2): views of the chunk buffers for sub-batch [r0, r1).
    pub(crate) fn prefill_views(s: &AttnScratch<'d>, r0: usize, r1: usize) -> Result<Vec<Buffer<'d>>> {
        let rs = r1 - r0;
        Ok(vec![
            s.index_query.view(r0 * qsa::HEADS * qsa::D * 2, rs * qsa::HEADS * qsa::D * 2)?,
            s.ids.view(r0 * qsa::TOPK * 4, rs * qsa::TOPK * 4)?,
            s.query.view(r0 * a::Q * 2, rs * a::Q * 2)?,
            s.attention.view(r0 * a::Q * 2, rs * a::Q * 2)?,
        ])
    }
    /// R60 S4 (prefill v2): the gate and out_proj for the whole chunk (after every sub-batch).
    pub(crate) fn record_prefill_output<'a>(&'a self, s: &'a AttnScratch<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        // SAFETY: kernels/lowbit_attention.cu attn_gate ABI.
        Ok(vec![
            unsafe {
                self.kernels[8].prepare(
                    vec![Arg::Buffer(&s.attention), Arg::Buffer(&s.gate), Arg::Buffer(&s.gated), Arg::U32((rows * a::Q) as u32)],
                    [(rows * a::Q).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                )?
            },
            self.linears[4].record_rows(&s.gated, &s.proj[4].0, &s.proj[4].1, rows)?,
        ])
    }
    /// R60 S4 (prefill v2): the QSA tail after `rows` accepted chunk rows.
    pub(crate) fn record_commit_prefill<'a>(&'a self, s: &'a AttnScratch<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        let pos = self.admit_prefill(rows)?;
        // SAFETY: kernels/lowbit_attention.cu tail gather / restore ABIs.
        Ok(vec![
            unsafe {
                self.kernels[9].prepare(
                    vec![Arg::Buffer(&s.proj[3].1), Arg::Buffer(&self.sq().tail), Arg::Buffer(&self.sq().next_tail), Arg::U32(pos.len as u32), Arg::U32(rows as u32)],
                    [3, 1, 1],
                    [128, 1, 1],
                )?
            },
            unsafe { self.kernels[10].prepare(vec![Arg::Buffer(&self.sq().next_tail), Arg::Buffer(&self.sq().tail)], [3, 1, 1], [128, 1, 1])? },
        ])
    }
    /// R60 S4 (prefill v2): advance the position past `rows` committed chunk rows.
    pub(crate) fn finish_prefill(&self, rows: usize) -> Result<()> {
        let pos = self.admit_prefill(rows)?;
        self.invalidate();
        self.sq().position.set(Some(Position { base: pos.base, len: pos.len + rows }));
        Ok(())
    }
}

impl<'m, 'd> Attention<'m, 'd> {
    /// R60 S4 (prefill v2 pull-back): append positions [from, to) of the KV cache (token-major:
    /// key codes, key scales, key biases, value codes, value scales, value biases) and the pooled
    /// index blocks [from/4, to/4) to `out`.
    pub(crate) fn export_prefill_rows(&self, from: usize, to: usize, out: &mut Vec<u8>) -> Result<()> {
        ensure!(from < to && to <= self.sq().capacity, "attention export rows");
        let n = to - from;
        for (buf, width) in [(&self.sq().k.codes, a::KV), (&self.sq().k.scales, a::META * 2), (&self.sq().k.biases, a::META * 2),
            (&self.sq().v.codes, a::KV), (&self.sq().v.scales, a::META * 2), (&self.sq().v.biases, a::META * 2)] {
            let at = out.len();
            out.resize(at + n * width, 0);
            buf.view(from * width, n * width)?.download(&mut out[at..])?;
        }
        let (b0, b1) = (from / 4, to / 4);
        if b1 > b0 {
            let at = out.len();
            out.resize(at + (b1 - b0) * qsa::D * 2, 0);
            self.sq().pooled.view(b0 * qsa::D * 2, (b1 - b0) * qsa::D * 2)?.download(&mut out[at..])?;
        }
        Ok(())
    }
    /// R60 S4 (prefill v2 pull-back): copy the raw QSA keys of chunk rows [rows - n, rows) out of
    /// the index projection (640 wide, key at 512) into `ring` ([n, 128] BF16).
    pub(crate) fn record_raw_key_copy<'a>(
        s: &'a AttnScratch<'d>,
        copy: &'a Kernel<'m, 'd>,
        ring: &'a Buffer<'d>,
        rows: usize,
        n: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(n > 0 && n <= rows && ring.byte_len() >= n * qsa::D * 2, "raw key copy");
        // SAFETY: kernels/mcg4_tensor.cu rows_cols_copy ABI; source rows < rows of the chunk.
        unsafe {
            copy.prepare(
                vec![Arg::Buffer(&s.proj[3].1), Arg::Buffer(ring), Arg::U32(640), Arg::U32(512), Arg::U32(qsa::D as u32),
                    Arg::U32((rows - n) as u32), Arg::U32(n as u32)],
                [(n * qsa::D).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )
        }
    }
}
