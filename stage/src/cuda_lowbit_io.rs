//! Resident CUDA embedding -> HC stream and final mixer -> full vocabulary head.
//! Probe plans do not insert transformer layers and are not model generations.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::Packed,
    hc::{F, H, HC, MAX_ROWS, R},
    lowbit_io::{Mixed, Mixer, VOCAB, Vocab},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "d16eec1c9d5f57fb40ace43aec36963d407b2d2588d13e97f4db3a4c18310fed";
const GUARD: usize = 256;
/// R53: QWEN_R53_TAILX=1 loads lowbit_io_r53 (final_normx: norm weights requested
/// with the inputs instead of inside the store loop; same values).
pub fn r53_tailx_enabled() -> bool {
    std::env::var("QWEN_R53_TAILX").as_deref() == Ok("1")
}
fn r53_io_sha() -> String {
    let pin: Value = serde_json::from_str(include_str!("../pins/R53-IO-CUDA-BUILD.json")).expect("R53-IO record");
    pin["cubin_sha256"].as_str().expect("R53-IO cubin_sha256").to_string()
}
/// The cubin this process loads for the IO kernels.
pub fn active_sha() -> String {
    if r53_tailx_enabled() { r53_io_sha() } else { SHA.to_string() }
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if r53_tailx_enabled() {
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_io_r53.sm120.cubin"), &r53_io_sha());
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_io.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![0xff; b.byte_len()])
}
fn read(b: &Buffer<'_>, n: usize) -> Result<Vec<u8>> {
    ensure!(n + GUARD <= b.byte_len(), "IO read bound");
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(raw[n..].iter().all(|&b| b == 0xff), "IO output tail canary");
    raw.truncate(n);
    Ok(raw)
}
fn read_bf(b: &Buffer<'_>, n: usize) -> Result<Vec<u16>> {
    let v: Vec<_> = read(b, 2 * n)?
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        v.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "IO BF16 finite/written"
    );
    Ok(v)
}
fn read_fp(b: &Buffer<'_>, n: usize) -> Result<Vec<f32>> {
    let v: Vec<_> = read(b, 4 * n)?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(v.iter().all(|v| v.is_finite()), "IO FP32 finite/written");
    Ok(v)
}
struct Pair<'d> {
    raw: Buffer<'d>,
    rounded: Buffer<'d>,
}
impl<'d> Pair<'d> {
    fn new(d: &'d Device, n: usize) -> Result<Self> {
        Ok(Self {
            raw: d.alloc(n * 4 + GUARD)?,
            rounded: d.alloc(n * 2 + GUARD)?,
        })
    }
    fn poison(&self) -> Result<()> {
        poison(&self.raw)?;
        poison(&self.rounded)
    }
    fn read(&self, n: usize) -> Result<(Vec<f32>, Vec<u16>)> {
        let r = read_fp(&self.raw, n)?;
        let b = read_bf(&self.rounded, n)?;
        ensure!(
            r.iter()
                .zip(&b)
                .all(|(&v, &b)| bf16::from_f32(v).to_bits() == b),
            "IO projection BF16 rounding"
        );
        Ok((r, b))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Embedding,
    Tail,
    Roundtrip,
}
pub struct Io<'m, 'd> {
    d: &'d Device,
    emb: Packed<'m, 'd>,
    head: Packed<'m, 'd>,
    down: Packed<'m, 'd>,
    up: Packed<'m, 'd>,
    gather: Kernel<'m, 'd>,
    norm: Kernel<'m, 'd>,
    activate: Kernel<'m, 'd>,
    mix: Kernel<'m, 'd>,
    nw: Buffer<'d>,
    ids: Buffer<'d>,
    status: Buffer<'d>,
    stream: Buffer<'d>,
    x: Buffer<'d>,
    normalized: Buffer<'d>,
    lower: Pair<'d>,
    activated: Buffer<'d>,
    upper: Pair<'d>,
    mixed: Buffer<'d>,
    logits: Pair<'d>,
    rows: Cell<usize>,
    epoch: Cell<u64>,
    tail_ready: Cell<bool>,
}
impl<'m, 'd> Io<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        hc: &'m Module<'d>,
        ops: &'m Module<'d>,
        emb: &Vocab,
        head: &Vocab,
        w: &Mixer,
    ) -> Result<Self> {
        w.validate()?;
        let nw = d.alloc(F * 2)?;
        nw.upload(bytes(&w.norm))?;
        let mut check = vec![0; F * 2];
        nw.download(&mut check)?;
        ensure!(check == bytes(&w.norm), "final norm weight readback");
        Ok(Self {
            d,
            emb: Packed::vocabulary(d, affine, emb)?,
            head: Packed::vocabulary(d, affine, head)?,
            down: Packed::new(d, affine, &w.down)?,
            up: Packed::new(d, affine, &w.up)?,
            gather: ops.kernel("vocab_gather")?,
            norm: ops.kernel(if r53_tailx_enabled() { "final_normx" } else { "final_norm" })?,
            activate: ops.kernel("final_activate")?,
            mix: hc.kernel("hc_mix")?,
            nw,
            ids: d.alloc(MAX_ROWS * 4 + GUARD)?,
            status: d.alloc(MAX_ROWS * 4 + GUARD)?,
            stream: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            x: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            normalized: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            lower: Pair::new(d, MAX_ROWS * R)?,
            activated: d.alloc(MAX_ROWS * R * 2 + GUARD)?,
            upper: Pair::new(d, MAX_ROWS * F)?,
            mixed: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            logits: Pair::new(d, MAX_ROWS * VOCAB)?,
            rows: Cell::new(0),
            epoch: Cell::new(0),
            tail_ready: Cell::new(false),
        })
    }
    pub fn description(&self) -> Value {
        let scratch: usize = [
            &self.ids,
            &self.status,
            &self.stream,
            &self.x,
            &self.normalized,
            &self.lower.raw,
            &self.lower.rounded,
            &self.activated,
            &self.upper.raw,
            &self.upper.rounded,
            &self.mixed,
            &self.logits.raw,
            &self.logits.rounded,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":active_sha(),"hc_cubin_sha256":crate::cuda_hc::SHA,
            "rows_max":MAX_ROWS,"vocab":VOCAB,"embedding":self.emb.description(),"head":self.head.description(),"down":self.down.description(),"up":self.up.description(),
            "workspace_bytes":scratch,"norm_weight_bytes":self.nw.byte_len(),"host_intermediates":0,"vocabulary_pruned":false,
            "kernels":[self.gather.description(),self.norm.description(),self.activate.description(),self.mix.description()],
            "scope":"native model endpoints, no transformer layers, not a generation/TPS measurement"})
    }
    pub fn prepare(&self, tokens: &[u32], x: Option<&[u16]>) -> Result<()> {
        self.rows.set(0);
        self.tail_ready.set(false);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= MAX_ROWS
                && tokens.iter().all(|&t| t < (VOCAB as u32)),
            "IO token bounds"
        );
        if let Some(x) = x {
            ensure!(
                x.len() == tokens.len() * F && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "IO tail input shape/finite"
            );
        }
        poison(&self.ids)?;
        poison(&self.x)?;
        self.ids.upload(bytes(tokens))?;
        ensure!(
            read(&self.ids, 4 * tokens.len())? == bytes(tokens),
            "IO ids readback"
        );
        if let Some(x) = x {
            self.x.upload(bytes(x))?;
            ensure!(read_bf(&self.x, x.len())? == x, "IO input readback");
            self.tail_ready.set(true);
        }
        self.rows.set(tokens.len());
        Ok(())
    }
    /// R56 graph-step prepare: only the token ids are uploaded (no poison, no readback). The
    /// graph path downloads only logits/top-k, never these staging buffers' canaries.
    pub fn prepare_lean(&self, tokens: &[u32]) -> Result<()> {
        self.rows.set(0);
        self.tail_ready.set(false);
        self.epoch.set(self.epoch.get().wrapping_add(1));
        ensure!(
            !tokens.is_empty() && tokens.len() <= MAX_ROWS && tokens.iter().all(|&t| t < (VOCAB as u32)),
            "IO token bounds"
        );
        self.ids.upload(bytes(tokens))?;
        self.rows.set(tokens.len());
        Ok(())
    }
    pub fn token_buffer(&self) -> &Buffer<'d> {
        &self.ids
    }
    pub fn stream(&self) -> &Buffer<'d> {
        &self.stream
    }
    pub fn logits(&self) -> &Buffer<'d> {
        &self.logits.rounded
    }
    pub fn mixed(&self) -> &Buffer<'d> {
        &self.mixed
    }
    pub(crate) fn record_start(&self) -> Result<PreparedCall<'_, 'm, 'd>> {
        self.record_embedding(&self.ids, &self.stream, &self.status, self.rows.get())
    }
    /// Device token IDs may come from a later GPU sampler. Status must be checked
    /// before committing a model candidate; malformed IDs never read OOB weights.
    pub fn record_embedding<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        out: &'a Buffer<'d>,
        status: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && ids.byte_len() >= rows * 4
                && out.byte_len() >= rows * F * 2
                && status.byte_len() >= rows * 4
                && !ids.aliases(out)
                && !ids.aliases(status)
                && !out.aliases(status)
                && [&self.emb.w, &self.emb.s, &self.emb.b]
                    .iter()
                    .all(|b| !out.aliases(b) && !status.aliases(b)),
            "IO embedding buffers/alias"
        );
        unsafe {
            self.gather.prepare(
                vec![
                    Arg::Buffer(ids),
                    Arg::Buffer(&self.emb.w),
                    Arg::Buffer(&self.emb.s),
                    Arg::Buffer(&self.emb.b),
                    Arg::Buffer(out),
                    Arg::Buffer(status),
                    Arg::U32(rows as u32),
                ],
                [H.div_ceil(256) as u32, rows as u32, 1],
                [256, 1, 1],
            )
        }
    }
    /// The caller supplies already-flushed HC streams. No pending residual or
    /// recurrent state is hidden at this interface.
    pub fn record_tail<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        arm: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && arm < 4 && x.byte_len() >= rows * F * 2,
            "IO tail shape/arm"
        );
        ensure!(
            [
                &self.normalized,
                &self.lower.raw,
                &self.lower.rounded,
                &self.activated,
                &self.upper.raw,
                &self.upper.rounded,
                &self.mixed,
                &self.logits.raw,
                &self.logits.rounded
            ]
            .iter()
            .all(|b| !x.aliases(b)),
            "IO tail alias"
        );
        let norm = unsafe {
            self.norm.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&self.nw),
                    Arg::Buffer(&self.normalized),
                ],
                [HC as u32, rows as u32, 1],
                [256, 1, 1],
            )?
        };
        let activate = unsafe {
            self.activate.prepare(
                vec![
                    Arg::Buffer(&self.lower.rounded),
                    Arg::Buffer(&self.activated),
                    Arg::U32(rows as u32),
                ],
                [(rows * R).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let mix = unsafe {
            self.mix.prepare(
                vec![
                    Arg::Buffer(&self.upper.rounded),
                    Arg::Buffer(&self.normalized),
                    Arg::Buffer(&self.mixed),
                    Arg::U32(rows as u32),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        Ok(vec![
            norm,
            self.down.record_into(
                &self.normalized,
                &self.lower.raw,
                &self.lower.rounded,
                rows,
                arm,
            )?,
            activate,
            self.up.record_into(
                &self.activated,
                &self.upper.raw,
                &self.upper.rounded,
                rows,
                arm,
            )?,
            mix,
            self.head.record_into(
                &self.mixed,
                &self.logits.raw,
                &self.logits.rounded,
                rows,
                arm,
            )?,
        ])
    }
    pub fn plan(
        &self,
        kind: Kind,
        arm: usize,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        let rows = self.rows.get();
        ensure!(
            rows > 0
                && arm < 4
                && (1..=16).contains(&repetitions)
                && (kind != Kind::Tail || self.tail_ready.get()),
            "IO plan readiness/bounds"
        );
        let make = || -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
            let mut c = vec![];
            if kind != Kind::Tail {
                c.push(self.record_embedding(&self.ids, &self.stream, &self.status, rows)?);
            }
            if kind != Kind::Embedding {
                c.extend(self.record_tail(
                    if kind == Kind::Tail {
                        &self.x
                    } else {
                        &self.stream
                    },
                    rows,
                    arm,
                )?);
            }
            Ok(c)
        };
        let exec = if graph {
            let mut c = vec![];
            for _ in 0..repetitions {
                c.extend(make()?);
            }
            Executor::Graph(self.d.capture(c)?)
        } else {
            Executor::Prepared(make()?)
        };
        Ok(Plan {
            owner: self,
            exec,
            kind,
            rows,
            repetitions,
            epoch: self.epoch.get(),
        })
    }
    pub(crate) fn poison(&self) -> Result<()> {
        for b in [
            &self.status,
            &self.stream,
            &self.normalized,
            &self.activated,
            &self.mixed,
        ] {
            poison(b)?;
        }
        self.lower.poison()?;
        self.upper.poison()?;
        self.logits.poison()
    }
    pub(crate) fn collect(&self, kind: Kind, rows: usize) -> Result<Run> {
        let embedding = if kind != Kind::Tail {
            ensure!(
                read(&self.status, rows * 4)?.iter().all(|&b| b == 0),
                "invalid device embedding token/status"
            );
            Some(read_bf(&self.stream, rows * F)?)
        } else {
            None
        };
        let tail = if kind != Kind::Embedding {
            let normalized = read_bf(&self.normalized, rows * F)?;
            let (down_raw, down) = self.lower.read(rows * R)?;
            let activated = read_bf(&self.activated, rows * R)?;
            let (up_raw, up) = self.upper.read(rows * F)?;
            let mixed = read_bf(&self.mixed, rows * H)?;
            let (logits_raw, logits) = self.logits.read(rows * VOCAB)?;
            Some(Tail {
                stages: Mixed {
                    normalized,
                    down,
                    activated,
                    up,
                    mixed,
                },
                down_raw,
                up_raw,
                logits_raw,
                logits,
            })
        } else {
            None
        };
        Ok(Run {
            embedding,
            tail,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Io<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    kind: Kind,
    rows: usize,
    repetitions: usize,
    epoch: u64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Tail {
    pub stages: Mixed,
    pub down_raw: Vec<f32>,
    pub up_raw: Vec<f32>,
    pub logits_raw: Vec<f32>,
    pub logits: Vec<u16>,
}
pub struct Run {
    pub embedding: Option<Vec<u16>>,
    pub tail: Option<Tail>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        ensure!(
            a.epoch.get() == self.epoch && a.rows.get() == self.rows,
            "stale IO plan"
        );
        a.poison()?;
        a.d.sync()?;
        let start = Instant::now();
        match &self.exec {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(c) => {
                for _ in 0..self.repetitions {
                    for c in c {
                        c.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut out = a.collect(self.kind, self.rows)?;
        out.total_us = total_us;
        out.enqueue_us = enqueue_us;
        out.wait_us = wait_us;
        Ok(out)
    }
}
