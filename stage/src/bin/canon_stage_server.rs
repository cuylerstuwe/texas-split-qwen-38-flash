//! R60 S3: the pipeline split's RTX 5090 stage as a server for Sushi.
//! Holds the canonical target's layers [FIRST, 48) (cuda_stage::Stage). Sushi connects over a
//! Unix socket (QD_STAGE_SOCKET) and shares one memory-mapped file (QD_STAGE_SHM) for streams and
//! state seeds. Messages are 32 bytes, little endian:
//!   u32 op, u32 status, u32 rows, u32 aux, u64 position, u64 arg
//! op 1 FORWARD: shm[0..rows*10240*2] holds the BF16 HC stream after layer FIRST-1 for absolute
//!   positions [position, position+rows); the reply leaves the stream after layer 47 in place.
//!   Rows still pending from the previous forward are committed first (full accept / serial).
//!   aux bit 1 (R62): also write the stream after each stage layer set in `arg` (bit l = layer l),
//!   ascending, after the output (DFlash2's target-layer taps past the split point).
//!   aux bit 2 (R64, QD_STAGE_HEAD=1): also run the final mixer, the 8-bit lm_head and a top-20 per row,
//!   written after the output (and taps): per row 20 BF16-valued f32 bit patterns then 20 token ids, u32.
//! op 2 COMMIT: keep `arg` rows of the pending forward (a partial speculative accept). Optional since
//!   R64: a FORWARD with no COMMIT before it commits position - committed rows of the pending forward.
//! op 3 SEED: shm holds a state blob (see `seed`); the stage's committed length becomes `position`.
//! op 4 INFO: reply aux = stage layers, position = committed length.
//! op 5 QUIT.
//! op 6 PREFILL (R60 S4, prefill v2): shm[arg..arg+rows*10240*2] holds the stream after layer FIRST-1
//!   for positions [position, position+rows) (position == the committed length, nothing pending);
//!   every row runs and commits through the chunk-sized prefill path and the stream after layer 47
//!   replaces the input in place. Sushi double-buffers two chunk slots and pipelines them.
//!   aux bit 0: also write the pull-back blob (Stage::prefill_export, "QDX1") at arg + 128 MiB;
//!   the reply's arg holds its length.
//! op 7 PARK (texas-split background parking): shm[0..] holds a NUL-ended absolute file path.
//!   Commits the pending forward up to `position` (the sequence's committed length), then writes
//!   the stage's whole committed state there as a seed blob (Stage::snapshot); the reply's arg
//!   holds its length. The state stays loaded (the next sequence's SEED/PREFILL replaces it).
//! op 8 RESTORE: shm[0..] holds the path of a PARK file; reloads it like SEED (`position` must be
//!   the parked position). The caller removes the file. QD_STAGE_RESTORE_VERIFY=1 re-reads the live
//!   state after the restore and fails the op unless it equals the file byte for byte.
//! Reply: same layout, status 0 = ok, 1 = error (the error text follows in shm[0..], NUL-ended).
//!   canon-stage-server [FIRST] [CAPACITY]
use anyhow::{Context, Result, bail, ensure};
use qwen_hybrid::{
    cuda::Device,
    cuda_affine8, cuda_lowbit_attention,
    cuda_lowbit_layer::Modules,
    cuda_stage::{HEAD_TOPK, Stage, StageHead},
    gdn::{C, Initial, STATE},
    hc::F,
    projection::Pack,
    qsa, target_attention as a,
};
use serde_json::json;
use std::{
    io::{Read, Write},
    os::unix::{io::AsRawFd, net::UnixListener},
    path::Path,
    time::Instant,
};
unsafe extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
}
// 2 GiB: the largest state seed (262160 rows, ~1.56 GB) fits whole; Sushi maps the same size.
const SHM_BYTES: usize = 2 << 30;
const MAGIC: u32 = 0x3153_4451; // "QDS1"

struct Shm {
    ptr: *mut u8,
    _file: std::fs::File,
}
impl Shm {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        file.set_len(SHM_BYTES as u64)?;
        // SAFETY: shared mapping of a file we own, unmapped at process exit.
        let ptr = unsafe { mmap(std::ptr::null_mut(), SHM_BYTES, 1 | 2, 1, file.as_raw_fd(), 0) };
        ensure!(!ptr.is_null() && ptr as isize != -1, "mmap stage shm");
        Ok(Self { ptr: ptr.cast(), _file: file })
    }
    fn bytes(&self) -> &mut [u8] {
        // SAFETY: SHM_BYTES mapped; one client, strictly request/reply.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, SHM_BYTES) }
    }
}
/// Little-endian reader over a seed blob: a 16-byte alignment after every array, none after the
/// scalar words (see `seed`). In memory (the SEED op's shm) or streamed from a PARK file.
trait Src {
    /// `n` bytes, then the cursor moves to the next 16-byte boundary.
    fn take(&mut self, n: usize) -> Result<Vec<u8>>;
    fn word<const N: usize>(&mut self) -> Result<[u8; N]>;
    fn align(&mut self) -> Result<()>;
    fn at(&self) -> usize;
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.word::<4>()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.word::<8>()?))
    }
    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        Ok(self.take(n * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
struct Blob<'a> {
    b: &'a [u8],
    at: usize,
}
impl Src for Blob<'_> {
    fn take(&mut self, n: usize) -> Result<Vec<u8>> {
        ensure!(self.at + n <= self.b.len(), "seed blob truncated");
        let s = self.b[self.at..self.at + n].to_vec();
        self.at = (self.at + n).div_ceil(16) * 16;
        Ok(s)
    }
    fn word<const N: usize>(&mut self) -> Result<[u8; N]> {
        ensure!(self.at + N <= self.b.len(), "seed blob truncated");
        let v = self.b[self.at..self.at + N].try_into()?;
        self.at += N;
        Ok(v)
    }
    fn align(&mut self) -> Result<()> {
        self.at = self.at.div_ceil(16) * 16;
        Ok(())
    }
    fn at(&self) -> usize {
        self.at
    }
}
struct FileSrc {
    r: std::io::BufReader<std::fs::File>,
    at: usize,
}
impl Src for FileSrc {
    fn take(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v).context("park file truncated")?;
        self.at += n;
        self.align()?;
        Ok(v)
    }
    fn word<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut v = [0u8; N];
        self.r.read_exact(&mut v).context("park file truncated")?;
        self.at += N;
        Ok(v)
    }
    fn align(&mut self) -> Result<()> {
        let pad = self.at.div_ceil(16) * 16 - self.at;
        let mut z = [0u8; 16];
        self.r.read_exact(&mut z[..pad]).context("park file truncated")?;
        self.at += pad;
        Ok(())
    }
    fn at(&self) -> usize {
        self.at
    }
}
/// RESTORE self-check (QD_STAGE_RESTORE_VERIFY=1): a fresh snapshot of the live state, compared
/// byte for byte with the park file it was restored from.
struct Compare {
    r: std::io::BufReader<std::fs::File>,
    at: u64,
    mismatch: Option<u64>,
    buf: Vec<u8>,
}
impl std::io::Write for Compare {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if self.buf.len() < b.len() {
            self.buf.resize(b.len(), 0);
        }
        let want = &mut self.buf[..b.len()];
        self.r.read_exact(want)?;
        if self.mismatch.is_none() && want != b {
            let i = want.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
            self.mismatch = Some(self.at + i as u64);
        }
        self.at += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
/// The NUL-ended absolute path a PARK/RESTORE request leaves at the start of shm.
fn park_path(shm: &Shm) -> Result<std::path::PathBuf> {
    let head = &shm.bytes()[..4096];
    let n = head.iter().position(|&c| c == 0).ok_or_else(|| anyhow::anyhow!("park path not NUL-ended"))?;
    let path = std::path::PathBuf::from(std::str::from_utf8(&head[..n])?);
    ensure!(
        path.is_absolute()
            && path.components().all(|c| !matches!(c, std::path::Component::ParentDir))
            && path.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("stage-") && f.ends_with(".bin")),
        "park path {} (absolute, no .., stage-*.bin)",
        path.display()
    );
    Ok(path)
}
/// Seed blob: header {u32 MAGIC, u32 version=1, u32 first, u32 count, u64 position}, then per
/// stage layer (16-byte aligned sections) {u32 tag (1 recurrent, 2 attention), u32 0, ...}:
/// recurrent: conv [3*10240] BF16 (oldest first), state [48*128*128] FP32 ([head][v][k]);
/// attention: u64 len, then K codes [len*512] u8 (token-major [t][kv head][dim]), K scales and
/// biases [len*8] BF16, the same three for V, QSA pooled keys [(len/4)*128] BF16 and the
/// incomplete group's raw index keys [(len%4)*128] BF16.
fn seed<S: Src>(stage: &Stage, b: &mut S, position: usize) -> Result<serde_json::Value> {
    ensure!(b.u32()? == MAGIC && b.u32()? == 1, "seed blob magic/version");
    let (first, count) = (b.u32()? as usize, b.u32()? as usize);
    let pos = b.u64()? as usize;
    ensure!(first == stage.first && first + count == 48 && pos == position, "seed blob geometry");
    b.align()?;
    let mut kv_tokens = 0;
    for l in first..48 {
        let tag = b.u32()?;
        let _ = b.u32()?;
        b.align()?;
        match tag {
            1 => {
                ensure!(l % 4 != 3, "recurrent seed for attention layer {l}");
                let conv = b.u16s(3 * C)?;
                let state = b.f32s(STATE)?;
                stage.import_recurrent(l, &Initial { conv, state })?;
            }
            2 => {
                ensure!(l % 4 == 3, "attention seed for recurrent layer {l}");
                let len = b.u64()? as usize;
                b.align()?;
                ensure!(len == position, "attention seed length {len} vs position {position}");
                let k = a::Packed { codes: b.take(len * a::KV)?, scales: b.u16s(len * a::META)?, biases: b.u16s(len * a::META)? };
                let v = a::Packed { codes: b.take(len * a::KV)?, scales: b.u16s(len * a::META)?, biases: b.u16s(len * a::META)? };
                let pooled = b.u16s(len / 4 * qsa::D)?;
                let tail = b.u16s(len % 4 * qsa::D)?;
                let cache = a::State { base: 0, len, k, v };
                let index = qsa::State { base: 0, len, pooled, tail };
                cache.validate()?;
                index.validate()?;
                stage.import_attention(l, &cache, &index)?;
                kv_tokens += len;
            }
            t => bail!("seed tag {t} at layer {l}"),
        }
    }
    stage.set_position(position);
    Ok(json!({"position":position,"bytes":b.at(),"kv_tokens":kv_tokens}))
}

/// R66 incremental seed ("QDI1"): header {u32 0x31494451, u32 1, u32 first, u32 count, u64 position, u64 base,
/// u64 check_from}, then per layer the SEED sections, an attention layer as {u64 hash, K codes/scales/biases and
/// V codes/scales/biases of rows [base, position), pooled blocks [base/4, position/4), the open group's raw keys}.
/// Rows [0, base) stay as held. `base` is group aligned. The client's `hash` is FNV-1a 64 over that layer's K
/// codes of rows [check_from, base) as Sushi holds them: tokens alone do not prove the held rows were computed
/// the same way (another chunking, or Metal versus CUDA), so a mismatch, or held rows short of `base`, refuses
/// the seed before anything changes and the client sends a full SEED instead.
const MAGIC_INCR: u32 = 0x3149_4451; // "QDI1"
fn fnv64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3))
}
fn seed_incremental<S: Src>(stage: &Stage, b: &mut S, position: usize) -> Result<(bool, serde_json::Value)> {
    ensure!(b.u32()? == MAGIC_INCR && b.u32()? == 1, "incremental seed magic/version");
    let (first, count) = (b.u32()? as usize, b.u32()? as usize);
    let (pos, base, check_from) = (b.u64()? as usize, b.u64()? as usize, b.u64()? as usize);
    ensure!(first == stage.first && first + count == 48 && pos == position && base <= position && base % 4 == 0
        && check_from <= base, "incremental seed geometry");
    b.align()?;
    enum Layer {
        Recurrent(Initial),
        Attention(a::State, qsa::State, u64),
    }
    let len = position - base;
    let mut layers = Vec::with_capacity(count);
    for l in first..48 {
        let tag = b.u32()?;
        let _ = b.u32()?;
        b.align()?;
        match tag {
            1 => {
                ensure!(l % 4 != 3, "recurrent seed for attention layer {l}");
                let conv = b.u16s(3 * C)?;
                let state = b.f32s(STATE)?;
                layers.push(Layer::Recurrent(Initial { conv, state }));
            }
            2 => {
                ensure!(l % 4 == 3, "attention seed for recurrent layer {l}");
                let hash = b.u64()?;
                b.align()?;
                let k = a::Packed { codes: b.take(len * a::KV)?, scales: b.u16s(len * a::META)?, biases: b.u16s(len * a::META)? };
                let v = a::Packed { codes: b.take(len * a::KV)?, scales: b.u16s(len * a::META)?, biases: b.u16s(len * a::META)? };
                let pooled = b.u16s(len / 4 * qsa::D)?;
                let tail = b.u16s(len % 4 * qsa::D)?;
                let cache = a::State { base, len, k, v };
                let index = qsa::State { base, len, pooled, tail };
                cache.validate()?;
                index.validate()?;
                layers.push(Layer::Attention(cache, index, hash));
            }
            t => bail!("incremental seed tag {t} at layer {l}"),
        }
    }
    // the lineage check, before any state changes
    for (l, layer) in (first..48).zip(&layers) {
        if let Layer::Attention(_, _, hash) = layer {
            let refuse = |why: &str| Ok((false, json!({"refused":why,"layer":l,"base":base,"position":position})));
            match stage.attention_len(l) {
                None => return refuse("no held state"),
                Some(held) if held < base => return refuse("held rows short of base"),
                Some(_) => {}
            }
            if fnv64(&stage.attention_key_codes(l, check_from, base)?) != *hash {
                return refuse("lineage hash");
            }
        }
    }
    for (l, layer) in (first..48).zip(layers) {
        match layer {
            Layer::Recurrent(initial) => stage.import_recurrent(l, &initial)?,
            Layer::Attention(cache, index, _) => stage.refresh_attention(l, &cache, &index)?,
        }
    }
    stage.set_position(position);
    Ok((true, json!({"position":position,"base":base,"rows":len,"bytes":b.at()})))
}

fn main() -> Result<()> {
    ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use guarded runner");
    let first: usize = std::env::args().nth(1).map(|a| a.parse()).transpose()?.unwrap_or(28);
    let capacity: usize = std::env::args().nth(2).map(|a| a.parse()).transpose()?.unwrap_or(65536);
    let socket = std::env::var("QD_STAGE_SOCKET").context("QD_STAGE_SOCKET")?;
    let shm_path = std::env::var("QD_STAGE_SHM").context("QD_STAGE_SHM")?;
    // the model directory (a Sushi Qwen3.8-Flash-Next 4bpw pack with the native-rope config.json)
    let model = std::env::var("QWEN_HYBRID_MODEL").unwrap_or_else(|_| qwen_hybrid::metal_model::MODEL.into());
    let pack = Pack::open(Path::new(&model)).with_context(|| format!("model pack {model} (set QWEN_HYBRID_MODEL)"))?;
    let mut d = Device::open()?;
    let m8 = cuda_affine8::module(&d)?;
    let am = cuda_lowbit_attention::module(&d)?;
    let k4 = qwen_hybrid::cuda_mcg_tensor::module_k4(&d)?;
    // R65 (QD_STAGE_SIDE=1): the shared-expert side branches' wait kernels
    #[cfg(feature = "r65-driver")]
    let r65w = if qwen_hybrid::cuda_stage::side_branches() { Some(qwen_hybrid::cuda_stage::r65_wait_module(&d)?) } else { None };
    #[cfg(feature = "r65-driver")]
    let r65k = match &r65w { Some(w) => Some(qwen_hybrid::cuda_stage::r65_wait_kernels(w)?), None => None };
    #[cfg(not(feature = "r65-driver"))]
    let r65k: Option<[qwen_hybrid::cuda::Kernel<'_, '_>; 3]> = None;
    {
        let m = Modules::load(&d)?;
        let stage = Stage::new(&d, &m, &am, &m8, &pack, first, capacity)?;
        if r65k.is_some() {
            eprintln!("canon-stage-server: R65 shared-expert side branches on");
        }
        // R64: the stage head (QD_STAGE_HEAD=1): final mixer + lm_head + GPU top-20 after a forward on request
        let r56 = if std::env::var("QD_STAGE_HEAD").as_deref() == Ok("1") {
            Some(d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/r56_ops.sm120.cubin"),
                "af3efba49e41f2f9218726d9937c9015cb84b9cade4434f5bd393d9b7f0dc1fd")?)
        } else { None };
        let head = match &r56 { Some(ops) => Some(StageHead::new(&d, &m, &m8, ops, &pack)?), None => None };
        let mut head_runs = 0u64;
        // prefill v2 scratch (QD_STAGE_PREFILL_ROWS, default 2048; 0 turns the PREFILL op off)
        let pf_rows: usize = std::env::var("QD_STAGE_PREFILL_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let pf = if pf_rows > 0 { Some(stage.prefill_scratch(&k4, pf_rows, 256)?) } else { None };
        let (mut prefills, mut prefill_rows, mut prefill_us) = (0u64, 0u64, 0f64);
        let shm = Shm::open(Path::new(&shm_path))?;
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket)?;
        let mem = d.memory_info()?;
        // R69: a captured graph stays resident in VRAM (~16.5 MB measured per (slot, rows) graph). With several slots
        // every (slot, rows) pair could want one (26 with four slots) and the free memory left beside the slots does
        // not hold them all, so captures stop at a budget from the free memory at ready (96 MiB kept back, 20 MiB per
        // graph); forwards past it run uncaptured (the same launches, enqueued one by one). One slot: 0.58 GiB free
        // budgets 24 graphs for at most 16 row counts, so nothing changes there.
        let max_graphs: usize = std::env::var("QD_STAGE_MAX_GRAPHS").ok().and_then(|v| v.parse().ok())
            .unwrap_or((mem.free_bytes.saturating_sub(96 << 20) / (20 << 20)) as usize).max(4);
        println!("{}", json!({"kind":"ready","stage":stage.description(),"socket":socket,"shm":shm_path,"capacity":capacity,
            "vram_free_bytes":mem.free_bytes,"vram_managed_bytes":mem.managed_bytes,"slots":stage.slots(),"max_graphs":max_graphs}));
        let (mut conn, _) = listener.accept()?;
        println!("{}", json!({"kind":"connected"}));
        let (mut forwards, mut fw_us, mut rows_total, mut commits, mut partial) = (0u64, 0f64, 0u64, 0u64, 0u64);
        let mut hist = [0u64; 17];
        // R69: one pending commit per conversation slot
        let mut pending_commit: Vec<Option<usize>> = vec![None; stage.slots()];
        let (mut parks, mut restores) = (0u64, 0u64);
        // RESTORE self-check: the live state re-read after the restore must equal the park file
        let restore_verify = std::env::var("QD_STAGE_RESTORE_VERIFY").as_deref() == Ok("1");
        // one captured graph per row count, re-targeted each forward (QD_STAGE_GRAPHS=0: off)
        let use_graphs = std::env::var("QD_STAGE_GRAPHS").as_deref() != Ok("0");
        // R69: keyed by (slot, rows): a slot's buffers differ, and patch_diff re-targets scalars only
        let mut graphs: std::collections::HashMap<(usize, usize), qwen_hybrid::cuda::Graph<'_, '_, '_>> = std::collections::HashMap::new();
        let (mut replays, mut captures) = (0u64, 0u64);
        loop {
            let mut msg = [0u8; 32];
            if conn.read_exact(&mut msg).is_err() {
                break;
            }
            let op = u32::from_le_bytes(msg[0..4].try_into()?);
            let rows = u32::from_le_bytes(msg[8..12].try_into()?) as usize;
            let aux = u32::from_le_bytes(msg[12..16].try_into()?);
            let mut reply_arg: u64 = 0;
            let position = u64::from_le_bytes(msg[16..24].try_into()?) as usize;
            let arg = u64::from_le_bytes(msg[24..32].try_into()?) as usize;
            let t = Instant::now();
            // FORWARD / PREFILL: aux bits 8..31 carry the rows' rotary offset (signed 24-bit; Qwen
            // M-RoPE decode delta after an image, 0 otherwise); bit 0 of a PREFILL is the pull-back flag
            if op == 1 || op == 6 {
                stage.set_rope_delta((aux as i32) >> 8);
            }
            // R69: aux bits 3-5 name the conversation slot of every op (0 unless the stage runs several)
            let slot = ((aux >> 3) & 7) as usize;
            let result: Result<(u32, u64)> = (|| match op {
                _ if slot >= stage.slots() => bail!("slot {slot} of {}", stage.slots()),
                _ if { stage.select(slot)?; false } => unreachable!(),
                1 => {
                    let prev = stage.pending();
                    // R64: with no COMMIT before it, the accepted rows follow from the position (a forward
                    // at P commits P - committed of the pending rows), so a client may skip op 2.
                    let accept = match pending_commit[slot].take() {
                        Some(a) => a,
                        None => {
                            let base = stage.position();
                            ensure!(position >= base && position - base <= prev, "forward at {position}, stage at {base} + {prev} pending");
                            position - base
                        }
                    };
                    if prev > 0 {
                        commits += 1;
                    }
                    ensure!(position == stage.position() + accept, "forward at {position}, stage at {} + {accept}", stage.position());
                    ensure!((1..=16).contains(&rows) && rows <= stage.slot_forward_rows(slot), "forward rows {rows} on slot {slot} (its limit {})", stage.slot_forward_rows(slot));
                    let raw = &shm.bytes()[..rows * F * 2];
                    let x: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    let (commits, rows) = stage.begin(&x, Some(accept))?;
                    for c in &commits {
                        c.enqueue()?;
                    }
                    let fresh = match &r65k { Some(k) => stage.record_forward_side(rows, k)?, None => stage.record_forward(rows)? };
                    let patched = match graphs.get_mut(&(slot, rows)) {
                        Some(g) if use_graphs => g.patch_diff(fresh)?.is_some(),
                        _ => false,
                    };
                    if patched {
                        graphs[&(slot, rows)].launch()?;
                        replays += 1;
                    } else if use_graphs && (graphs.contains_key(&(slot, rows)) || graphs.len() < max_graphs) {
                        let g = d.capture_model_diagnostic(match &r65k { Some(k) => stage.record_forward_side(rows, k)?, None => stage.record_forward(rows)? })?;
                        g.launch()?;
                        graphs.insert((slot, rows), g);
                        captures += 1;
                    } else {
                        for c in &(match &r65k { Some(k) => stage.record_forward_side(rows, k)?, None => stage.record_forward(rows)? }) {
                            c.enqueue()?;
                        }
                    }
                    // R64: aux bit 2 = the head right behind the forward, same stream, one sync
                    let want_head = aux & 4 != 0;
                    ensure!(!want_head || head.is_some(), "stage head requested but QD_STAGE_HEAD is off");
                    if want_head {
                        for c in &head.as_ref().unwrap().record(&stage, rows)? {
                            c.enqueue()?;
                        }
                    }
                    d.sync()?;
                    drop(commits);
                    let out = stage.finish(rows)?;
                    for (dst, v) in shm.bytes()[..rows * F * 2].chunks_exact_mut(2).zip(out) {
                        dst.copy_from_slice(&v.to_le_bytes());
                    }
                    // R62: aux bit 1 = also return the streams after the stage layers set in `arg`
                    // (bit l = layer l), in ascending layer order after the output
                    let mut at = rows * F * 2;
                    if aux & 2 != 0 {
                        for l in first..48 {
                            if arg >> l & 1 == 1 {
                                stage.layer_stream(l, rows, &mut shm.bytes()[at..at + rows * F * 2])?;
                                at += rows * F * 2;
                            }
                        }
                    }
                    if want_head {
                        let n = rows * 2 * HEAD_TOPK * 4;
                        head.as_ref().unwrap().read(rows, &mut shm.bytes()[at..at + n])?;
                        head_runs += 1;
                    }
                    forwards += 1;
                    rows_total += rows as u64;
                    hist[rows] += 1;
                    fw_us += t.elapsed().as_secs_f64() * 1e6;
                    Ok((0, stage.position() as u64))
                }
                2 => {
                    // applied with the next forward, in the same submission
                    ensure!(arg <= stage.pending(), "commit {arg} of {} pending rows", stage.pending());
                    pending_commit[slot] = Some(arg);
                    partial += 1;
                    Ok((0, (stage.position() + arg) as u64))
                }
                3 => {
                    stage.discard()?;
                    pending_commit[slot] = None;
                    let info = seed(&stage, &mut Blob { b: shm.bytes(), at: 0 }, position)?;
                    println!("{}", json!({"kind":"seed","ms":t.elapsed().as_secs_f64()*1e3,"info":info}));
                    Ok((0, stage.position() as u64))
                }
                7 => {
                    // texas-split parking: keep `position` rows of the pending forward, then the
                    // whole committed state to an SSD file
                    let prev = stage.pending();
                    let base = stage.position();
                    ensure!(position >= base && position <= base + prev, "park at {position}, stage at {base} + {prev} pending");
                    let accept = position - base;
                    if let Some(c) = pending_commit[slot].take() {
                        ensure!(c == accept, "park at {position} beside a pending commit of {c}");
                    }
                    if prev > 0 {
                        stage.commit(accept)?;
                    }
                    let path = park_path(&shm)?;
                    let file = {
                        use std::os::unix::fs::OpenOptionsExt;
                        std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path)
                            .with_context(|| format!("park file {}", path.display()))?
                    };
                    let mut w = std::io::BufWriter::with_capacity(8 << 20, file);
                    let bytes = stage.snapshot(&mut w)?;
                    w.into_inner().map_err(|e| anyhow::anyhow!("park file flush: {}", e.error()))?;
                    reply_arg = bytes;
                    parks += 1;
                    println!("{}", json!({"kind":"park","position":position,"accepted":accept,"pending":prev,"bytes":bytes,
                        "ms":t.elapsed().as_secs_f64()*1e3}));
                    Ok((0, stage.position() as u64))
                }
                8 => {
                    stage.discard()?;
                    pending_commit[slot] = None;
                    let path = park_path(&shm)?;
                    let file = std::fs::File::open(&path).with_context(|| format!("park file {}", path.display()))?;
                    let mut src = FileSrc { r: std::io::BufReader::with_capacity(8 << 20, file), at: 0 };
                    let info = seed(&stage, &mut src, position)?;
                    let verified = if restore_verify {
                        let file = std::fs::File::open(&path)?;
                        let len = file.metadata()?.len();
                        let mut cmp = Compare { r: std::io::BufReader::with_capacity(8 << 20, file), at: 0, mismatch: None, buf: vec![] };
                        let n = stage.snapshot(&mut cmp)?;
                        ensure!(cmp.mismatch.is_none() && n == len, "restored stage state differs from its park file (first byte {:?}, {n} of {len} bytes)", cmp.mismatch);
                        Some(n)
                    } else {
                        None
                    };
                    restores += 1;
                    println!("{}", json!({"kind":"restore","ms":t.elapsed().as_secs_f64()*1e3,"info":info,"verified_bytes":verified}));
                    Ok((0, stage.position() as u64))
                }
                9 => {
                    // R66d end-of-turn pull-back: keep `position` rows of the pending forward (as PARK does), then
                    // the committed rows [arg, position) as a QDX1 blob at shm offset 0 (reply arg: its length)
                    let prev = stage.pending();
                    let base = stage.position();
                    ensure!(position >= base && position <= base + prev, "export at {position}, stage at {base} + {prev} pending");
                    let accept = position - base;
                    if let Some(c) = pending_commit[slot].take() {
                        ensure!(c == accept, "export at {position} beside a pending commit of {c}");
                    }
                    if prev > 0 {
                        stage.commit(accept)?;
                    }
                    let mut blob = vec![];
                    stage.export_rows(arg, position, &mut blob)?;
                    ensure!(blob.len() <= SHM_BYTES, "export blob {} bytes", blob.len());
                    shm.bytes()[..blob.len()].copy_from_slice(&blob);
                    reply_arg = blob.len() as u64;
                    println!("{}", json!({"kind":"export","from":arg,"position":position,"accepted":accept,"bytes":blob.len(),
                        "ms":t.elapsed().as_secs_f64()*1e3}));
                    Ok((0, stage.position() as u64))
                }
                10 => {
                    // R66 incremental seed: reply arg 1 applied, 0 refused (nothing changed; the client
                    // sends a full SEED). Pending rows are dropped either way, as SEED does.
                    stage.discard()?;
                    pending_commit[slot] = None;
                    let (applied, info) = seed_incremental(&stage, &mut Blob { b: shm.bytes(), at: 0 }, position)?;
                    println!("{}", json!({"kind":"seed-incremental","applied":applied,"ms":t.elapsed().as_secs_f64()*1e3,"info":info}));
                    reply_arg = u64::from(applied);
                    Ok((0, stage.position() as u64))
                }
                4 => Ok((0, stage.position() as u64)),
                5 => Ok((2, 0)),
                11 => {
                    // R69: drop a slot's conversation (its rows may be reused; slot 0 may grow into its views again)
                    ensure!(slot > 0, "slot 0 is never dropped");
                    pending_commit[slot] = None;
                    stage.discard()?;
                    stage.drop_slot()?;
                    println!("{}", json!({"kind":"drop-slot","slot":slot}));
                    Ok((0, 0))
                }
                6 => {
                    ensure!(slot == 0, "prefill runs on slot 0 only");
                    let pf = pf.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch disabled"))?;
                    ensure!(stage.pending() == 0 && pending_commit[slot].is_none(), "prefill with an uncommitted forward");
                    ensure!(position == stage.position(), "prefill at {position}, stage at {}", stage.position());
                    const EXPORT: usize = 128 << 20;
                    ensure!(rows > 0 && rows * F * 2 <= EXPORT && arg + 2 * EXPORT <= SHM_BYTES, "prefill rows/slot");
                    // balanced pieces: the last one holds at least half a scratch of rows (the
                    // pull-back's raw keys come from it)
                    let cap = Stage::prefill_rows(pf);
                    let pieces = rows.div_ceil(cap);
                    let step = rows.div_ceil(pieces);
                    let from = stage.position();
                    let slot = &mut shm.bytes()[arg..arg + rows * F * 2];
                    let mut out = vec![0u8; step.min(rows) * F * 2];
                    for r0 in (0..rows).step_by(step) {
                        let r1 = (r0 + step).min(rows);
                        let part = &mut out[..(r1 - r0) * F * 2];
                        stage.prefill_raw(pf, &slot[r0 * F * 2..r1 * F * 2], part)?;
                        slot[r0 * F * 2..r1 * F * 2].copy_from_slice(part);
                    }
                    if aux & 1 != 0 {
                        let mut blob = vec![];
                        stage.prefill_export(pf, from, from + rows, &mut blob)?;
                        ensure!(blob.len() <= EXPORT, "pull-back blob {} bytes", blob.len());
                        shm.bytes()[arg + EXPORT..arg + EXPORT + blob.len()].copy_from_slice(&blob);
                        reply_arg = blob.len() as u64;
                    }
                    prefills += 1;
                    prefill_rows += rows as u64;
                    prefill_us += t.elapsed().as_secs_f64() * 1e6;
                    Ok((0, stage.position() as u64))
                }
                o => bail!("unknown op {o}"),
            })();
            let mut reply = [0u8; 32];
            reply[0..4].copy_from_slice(&op.to_le_bytes());
            match result {
                Ok((status, pos)) => {
                    reply[4..8].copy_from_slice(&(status.min(1)).to_le_bytes());
                    reply[12..16].copy_from_slice(&((48 - first) as u32).to_le_bytes());
                    reply[16..24].copy_from_slice(&pos.to_le_bytes());
                    reply[24..32].copy_from_slice(&reply_arg.to_le_bytes());
                    conn.write_all(&reply)?;
                    if status == 2 {
                        break;
                    }
                }
                Err(e) => {
                    let text = format!("{e:#}");
                    println!("{}", json!({"kind":"error","op":op,"error":text}));
                    let n = text.len().min(4000);
                    shm.bytes()[..n].copy_from_slice(&text.as_bytes()[..n]);
                    shm.bytes()[n] = 0;
                    reply[4..8].copy_from_slice(&1u32.to_le_bytes());
                    conn.write_all(&reply)?;
                }
            }
            if forwards > 0 && forwards % 500 == 0 && op == 1 {
                // R69: the allocator's free VRAM rides the periodic line (graph captures and scratch grow with slots)
                let free = d.memory_info().map(|m| m.free_bytes).unwrap_or(0);
                println!("{}", json!({"kind":"stats","forwards":forwards,"rows":rows_total,"forward_us_mean":fw_us/forwards as f64,
                    "commits":commits,"partial_commits":partial,"rows_hist":hist,"graph_replays":replays,"graph_captures":captures,
                    "vram_free_bytes":free}));
            }
        }
        println!("{}", json!({"kind":"park_stats","parks":parks,"restores":restores}));
        println!("{}", json!({"kind":"prefill_stats","prefills":prefills,"rows":prefill_rows,"ms_total":prefill_us/1e3,
            "rows_per_s":if prefill_us > 0. { prefill_rows as f64 / (prefill_us / 1e6) } else { 0. }}));
        println!("{}", json!({"kind":"head_stats","head_runs":head_runs,"head_on":head.is_some()}));
        println!("{}", json!({"kind":"stats","final":true,"forwards":forwards,"rows":rows_total,"forward_us_mean":if forwards>0 {fw_us/forwards as f64} else {0.},
            "commits":commits,"partial_commits":partial,"rows_hist":hist,"graph_replays":replays,"graph_captures":captures,
            "vram_free_bytes":d.memory_info().map(|m| m.free_bytes).unwrap_or(0)}));
        drop(graphs);
    }
    k4.close()?;
    am.close()?;
    m8.close()?;
    #[cfg(feature = "r65-driver")]
    {
        drop(r65k);
        let taken = r65w; // moved here, so its borrow of the device ends in this block
        if let Some(w) = taken {
            w.close()?;
        }
    }
    d.close()?;
    println!("{}", json!({"kind":"summary","passed":true,"cleanup_completed":true}));
    Ok(())
}
