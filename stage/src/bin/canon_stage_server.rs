//! R60 S3: the pipeline split's RTX 5090 stage as a server for Sushi.
//! Holds the canonical target's layers [FIRST, 48) (cuda_stage::Stage). Sushi connects over a
//! Unix socket (QD_STAGE_SOCKET) and shares one memory-mapped file (QD_STAGE_SHM) for streams and
//! state seeds. Messages are 32 bytes, little endian:
//!   u32 op, u32 status, u32 rows, u32 aux, u64 position, u64 arg
//! op 1 FORWARD: shm[0..rows*10240*2] holds the BF16 HC stream after layer FIRST-1 for absolute
//!   positions [position, position+rows); the reply leaves the stream after layer 47 in place.
//!   Rows still pending from the previous forward are committed first (full accept / serial).
//! op 2 COMMIT: keep `arg` rows of the pending forward (a partial speculative accept).
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
    cuda_stage::Stage,
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
    {
        let m = Modules::load(&d)?;
        let stage = Stage::new(&d, &m, &am, &m8, &pack, first, capacity)?;
        // prefill v2 scratch (QD_STAGE_PREFILL_ROWS, default 2048; 0 turns the PREFILL op off)
        let pf_rows: usize = std::env::var("QD_STAGE_PREFILL_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let pf = if pf_rows > 0 { Some(stage.prefill_scratch(&k4, pf_rows, 256)?) } else { None };
        let (mut prefills, mut prefill_rows, mut prefill_us) = (0u64, 0u64, 0f64);
        let shm = Shm::open(Path::new(&shm_path))?;
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket)?;
        let mem = d.memory_info()?;
        println!("{}", json!({"kind":"ready","stage":stage.description(),"socket":socket,"shm":shm_path,"capacity":capacity,
            "vram_free_bytes":mem.free_bytes,"vram_managed_bytes":mem.managed_bytes}));
        let (mut conn, _) = listener.accept()?;
        println!("{}", json!({"kind":"connected"}));
        let (mut forwards, mut fw_us, mut rows_total, mut commits, mut partial) = (0u64, 0f64, 0u64, 0u64, 0u64);
        let mut hist = [0u64; 17];
        let mut pending_commit: Option<usize> = None;
        let (mut parks, mut restores) = (0u64, 0u64);
        // RESTORE self-check: the live state re-read after the restore must equal the park file
        let restore_verify = std::env::var("QD_STAGE_RESTORE_VERIFY").as_deref() == Ok("1");
        // one captured graph per row count, re-targeted each forward (QD_STAGE_GRAPHS=0: off)
        let use_graphs = std::env::var("QD_STAGE_GRAPHS").as_deref() != Ok("0");
        let mut graphs: std::collections::HashMap<usize, qwen_hybrid::cuda::Graph<'_, '_, '_>> = std::collections::HashMap::new();
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
            let result: Result<(u32, u64)> = (|| match op {
                1 => {
                    let prev = stage.pending();
                    let accept = pending_commit.take().unwrap_or(prev);
                    if prev > 0 {
                        commits += 1;
                    }
                    ensure!(position == stage.position() + accept, "forward at {position}, stage at {} + {accept}", stage.position());
                    ensure!((1..=16).contains(&rows), "forward rows");
                    let raw = &shm.bytes()[..rows * F * 2];
                    let x: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    let (commits, rows) = stage.begin(&x, Some(accept))?;
                    for c in &commits {
                        c.enqueue()?;
                    }
                    let fresh = stage.record_forward(rows)?;
                    let patched = match graphs.get_mut(&rows) {
                        Some(g) if use_graphs => g.patch_diff(fresh)?.is_some(),
                        _ => false,
                    };
                    if patched {
                        graphs[&rows].launch()?;
                        replays += 1;
                    } else if use_graphs {
                        let g = d.capture_model_diagnostic(stage.record_forward(rows)?)?;
                        g.launch()?;
                        graphs.insert(rows, g);
                        captures += 1;
                    } else {
                        for c in &stage.record_forward(rows)? {
                            c.enqueue()?;
                        }
                    }
                    d.sync()?;
                    drop(commits);
                    let out = stage.finish(rows)?;
                    for (dst, v) in shm.bytes()[..rows * F * 2].chunks_exact_mut(2).zip(out) {
                        dst.copy_from_slice(&v.to_le_bytes());
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
                    pending_commit = Some(arg);
                    partial += 1;
                    Ok((0, (stage.position() + arg) as u64))
                }
                3 => {
                    stage.discard()?;
                    pending_commit = None;
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
                    if let Some(c) = pending_commit.take() {
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
                    pending_commit = None;
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
                4 => Ok((0, stage.position() as u64)),
                5 => Ok((2, 0)),
                6 => {
                    let pf = pf.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch disabled"))?;
                    ensure!(stage.pending() == 0 && pending_commit.is_none(), "prefill with an uncommitted forward");
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
                println!("{}", json!({"kind":"stats","forwards":forwards,"rows":rows_total,"forward_us_mean":fw_us/forwards as f64,
                    "commits":commits,"partial_commits":partial,"rows_hist":hist,"graph_replays":replays,"graph_captures":captures}));
            }
        }
        println!("{}", json!({"kind":"park_stats","parks":parks,"restores":restores}));
        println!("{}", json!({"kind":"prefill_stats","prefills":prefills,"rows":prefill_rows,"ms_total":prefill_us/1e3,
            "rows_per_s":if prefill_us > 0. { prefill_rows as f64 / (prefill_us / 1e6) } else { 0. }}));
        println!("{}", json!({"kind":"stats","final":true,"forwards":forwards,"rows":rows_total,"forward_us_mean":if forwards>0 {fw_us/forwards as f64} else {0.},
            "commits":commits,"partial_commits":partial,"rows_hist":hist,"graph_replays":replays,"graph_captures":captures}));
        drop(graphs);
    }
    k4.close()?;
    am.close()?;
    m8.close()?;
    d.close()?;
    println!("{}", json!({"kind":"summary","passed":true,"cleanup_completed":true}));
    Ok(())
}
