//! R10e encoder -> K/V -> normalized/rotated committed context, native CUDA.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_linear::{Arm, Linear, Output},
    dflash_attention::{MAX_ROWS, frequencies},
    dflash_context::{Append, CAPACITY, FEATURES, KV, State, Weights},
    dflash_mlp::{EPS, H},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
const GUARD: usize = 256;
pub const SHA: &str = "cc186318f8a59024a5003e77bcd32ce662bf0f75dfc44482ac8ad921817d6f70";
pub fn load_module(d: &Device) -> Result<Module<'_>> {
    d.module(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/dflash_context.sm120.cubin"
        )),
        SHA,
    )
}
fn bytes<T: Copy>(x: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
fn upload(b: &Buffer<'_>, raw: &[u8]) -> Result<()> {
    b.upload(raw)?;
    let mut check = vec![0; raw.len()];
    b.download(&mut check)?;
    ensure!(check == raw, "context upload readback");
    Ok(())
}
pub struct ContextCache<'m, 'd> {
    d: &'d Device,
    fc: Linear<'m, 'd>,
    k: Linear<'m, 'd>,
    v: Linear<'m, 'd>,
    rms: Kernel<'m, 'd>,
    append: Kernel<'m, 'd>,
    norm: Buffer<'d>,
    kw: Buffer<'d>,
    freq: Buffer<'d>,
    input: Buffer<'d>,
    projected: Buffer<'d>,
    encoded: Buffer<'d>,
    pk: Buffer<'d>,
    pv: Buffer<'d>,
    rotated: Buffer<'d>,
    ck: [Buffer<'d>; 2],
    cv: [Buffer<'d>; 2],
    control: Buffer<'d>,
    state: Cell<State>,
    valid: Cell<bool>,
    bank: Cell<usize>,
    pending: Cell<Option<Append>>,
    ticket: Cell<u64>,
}
impl<'m, 'd> ContextCache<'m, 'd> {
    pub fn new(
        d: &'d Device,
        gemm: &'m Module<'d>,
        ops: &'m Module<'d>,
        ctx: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        let weights = |raw: &[u8]| -> Result<Buffer<'d>> {
            let b = d.alloc(raw.len())?;
            upload(&b, raw)?;
            Ok(b)
        };
        let result = Self {
            d,
            fc: Linear::new(d, gemm, &w.fc)?,
            k: Linear::new(d, gemm, &w.k)?,
            v: Linear::new(d, gemm, &w.v)?,
            rms: ops.kernel("dflash_rms")?,
            append: ctx.kernel("dflash_context_append")?,
            norm: weights(bytes(&w.norm))?,
            kw: weights(bytes(&w.qk.k))?,
            freq: weights(bytes(&frequencies()))?,
            input: d.alloc(MAX_ROWS * FEATURES * 2)?,
            projected: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            encoded: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            pk: d.alloc(MAX_ROWS * KV * 2 + GUARD)?,
            pv: d.alloc(MAX_ROWS * KV * 2 + GUARD)?,
            rotated: d.alloc(MAX_ROWS * KV * 2 + GUARD)?,
            ck: [
                d.alloc(CAPACITY * KV * 2 + GUARD)?,
                d.alloc(CAPACITY * KV * 2 + GUARD)?,
            ],
            cv: [
                d.alloc(CAPACITY * KV * 2 + GUARD)?,
                d.alloc(CAPACITY * KV * 2 + GUARD)?,
            ],
            control: d.alloc(24 + GUARD)?,
            state: Cell::new(State::empty(0)?),
            valid: Cell::new(false),
            bank: Cell::new(0),
            pending: Cell::new(None),
            ticket: Cell::new(0),
        };
        result.restore(0, &[], &[])?;
        Ok(result)
    }
    pub fn description(&self) -> Value {
        json!({"backend":"native Rust/CUDA/libtinynv","context_cubin_sha256":SHA,"dependent_calls":5,
            "encoder_fc":self.fc.description(),"k":self.k.description(),"v":self.v.description(),
            "retained_capacity":CAPACITY,"maximum_append":MAX_ROWS,"cache_layout":"token,KV head,channel; disjoint ping-pong banks",
            "commit":"only after GPU completion and diagnostic validation","committed_features_only":true,
            "context_window_copy_is_linear":true,"shared_encoder_across_layers_not_yet_integrated":true})
    }
    pub fn state(&self) -> State {
        self.state.get()
    }
    pub fn restore(&self, base: usize, k: &[u16], v: &[u16]) -> Result<()> {
        self.pending.set(None);
        self.ticket.set(
            self.ticket
                .get()
                .checked_add(1)
                .context("context ticket overflow")?,
        );
        ensure!(
            k.len() == v.len()
                && k.len() % KV == 0
                && k.len() / KV <= CAPACITY
                && base + k.len() / KV <= 262144
                && k.iter().chain(v).all(|&v| bf16::from_bits(v).is_finite()),
            "context restore bounds/finite"
        );
        self.valid.set(false);
        for (b, x) in [(&self.ck[0], k), (&self.cv[0], v)] {
            let mut raw = vec![0xff; b.byte_len()];
            raw[..x.len() * 2].copy_from_slice(bytes(x));
            upload(b, &raw)?;
        }
        self.bank.set(0);
        self.state.set(State {
            base,
            len: k.len() / KV,
            generation: self
                .state
                .get()
                .generation
                .checked_add(1)
                .context("context generation overflow")?,
        });
        self.valid.set(true);
        Ok(())
    }
    pub fn prepare(&self, first: usize, x: &[u16]) -> Result<f64> {
        self.pending.set(None);
        self.ticket.set(
            self.ticket
                .get()
                .checked_add(1)
                .context("context ticket overflow")?,
        );
        ensure!(self.valid.get(), "context requires successful restore");
        ensure!(
            !x.is_empty()
                && x.len() % FEATURES == 0
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "context features geometry/finite"
        );
        let p = self.state.get().plan(first, x.len() / FEATURES)?;
        let t = Instant::now();
        upload(&self.input, bytes(x))?;
        let mut control = vec![0xff; self.control.byte_len()];
        control[..24].copy_from_slice(bytes(&p.control()));
        upload(&self.control, &control)?;
        self.pending.set(Some(p));
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    fn record(&self, p: Append) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        let bank = self.bank.get();
        let next = 1 - bank;
        let mut calls = vec![self.fc.record_into(
            &self.input,
            &self.projected,
            p.rows,
            Arm::Split(8),
            Output::Bf16,
        )?];
        // Fixed audited ABIs, positive validated rows, separate retained buffers.
        unsafe {
            calls.push(self.rms.prepare(
                vec![
                    Arg::Buffer(&self.projected),
                    Arg::Buffer(&self.norm),
                    Arg::Buffer(&self.encoded),
                    Arg::U32(H as u32),
                    Arg::F32(EPS as f32),
                ],
                [p.rows as u32, 1, 1],
                [256, 1, 1],
            )?);
        }
        calls.push(self.k.record_into(
            &self.encoded,
            &self.pk,
            p.rows,
            Arm::Split(8),
            Output::Bf16,
        )?);
        calls.push(self.v.record_into(
            &self.encoded,
            &self.pv,
            p.rows,
            Arm::Split(8),
            Output::Bf16,
        )?);
        unsafe {
            calls.push(self.append.prepare(
                vec![
                    Arg::Buffer(&self.pk),
                    Arg::Buffer(&self.pv),
                    Arg::Buffer(&self.kw),
                    Arg::Buffer(&self.freq),
                    Arg::Buffer(&self.ck[bank]),
                    Arg::Buffer(&self.cv[bank]),
                    Arg::Buffer(&self.ck[next]),
                    Arg::Buffer(&self.cv[next]),
                    Arg::Buffer(&self.rotated),
                    Arg::Buffer(&self.control),
                ],
                [(p.after.len * 8) as u32, 1, 1],
                [32, 1, 1],
            )?);
        }
        Ok(calls)
    }
    pub fn plan(&self, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        let p = self.pending.get().context("context input not prepared")?;
        let calls = self.record(p)?;
        let executor = if graph {
            Executor::Graph(self.d.capture(calls)?)
        } else {
            Executor::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            p,
            ticket: self.ticket.get(),
            executor,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a ContextCache<'m, 'd>,
    p: Append,
    ticket: u64,
    executor: Executor<'a, 'm, 'd>,
}
pub struct Run {
    pub stages: Vec<(&'static str, Vec<f32>)>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub state: State,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub canary_bytes: usize,
    pub counter_words: usize,
    pub old_context_bytes: usize,
}
fn read(b: &Buffer<'_>, count: usize, guards: &mut usize) -> Result<Vec<u16>> {
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(
        raw[count * 2..].iter().all(|&v| v == 0xff),
        "context trailing canary"
    );
    *guards += raw.len() - count * 2;
    let out: Vec<_> = raw[..count * 2]
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        out.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "context nonfinite/unwritten output"
    );
    Ok(out)
}
impl Plan<'_, '_, '_> {
    /// Repetitions replay the same immutable append, committing it exactly once.
    /// Input/weight upload, poison, graph construction and full diagnostics excluded.
    pub fn run(&self, repetitions: usize, wait_each: bool) -> Result<Run> {
        let c = self.owner;
        let p = self.p;
        ensure!(
            self.ticket == c.ticket.get()
                && c.pending.get().is_some()
                && c.state.get() == p.before
                && (1..=32).contains(&repetitions),
            "stale context plan/repetitions"
        );
        ensure!(
            !wait_each || matches!(&self.executor, Executor::Calls(_)),
            "context graph wait-each"
        );
        c.pending.set(None); // Any failure requires explicit preparation; old bank remains valid.
        let bank = c.bank.get();
        let next = 1 - bank;
        let mut guards = 0;
        let oldk = read(&c.ck[bank], p.before.len * KV, &mut guards)?;
        let oldv = read(&c.cv[bank], p.before.len * KV, &mut guards)?;
        for b in [
            &c.projected,
            &c.encoded,
            &c.pk,
            &c.pv,
            &c.rotated,
            &c.ck[next],
            &c.cv[next],
        ] {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        c.d.sync()?;
        let t = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repetitions {
            match &self.executor {
                Executor::Calls(calls) => {
                    for call in calls {
                        let t = Instant::now();
                        call.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            c.d.sync()?;
                            completion_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    }
                }
                Executor::Graph(g) => {
                    let t = Instant::now();
                    g.launch()?;
                    enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                }
            }
        }
        let wait = Instant::now();
        c.d.sync()?;
        completion_us += wait.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let mut stages = vec![];
        for (name, b, width) in [
            ("projected", &c.projected, H),
            ("encoded", &c.encoded, H),
            ("k", &c.pk, KV),
            ("v", &c.pv, KV),
            ("rotated", &c.rotated, KV),
        ] {
            stages.push((
                name,
                read(b, p.rows * width, &mut guards)?
                    .into_iter()
                    .map(|v| bf16::from_bits(v).to_f32())
                    .collect(),
            ));
        }
        let k = read(&c.ck[next], p.after.len * KV, &mut guards)?;
        let v = read(&c.cv[next], p.after.len * KV, &mut guards)?;
        ensure!(
            oldk == read(&c.ck[bank], p.before.len * KV, &mut guards)?
                && oldv == read(&c.cv[bank], p.before.len * KV, &mut guards)?,
            "context old bank mutated"
        );
        let retained = (p.before.len - p.drop) * KV;
        ensure!(
            k[..retained] == oldk[p.drop * KV..] && v[..retained] == oldv[p.drop * KV..],
            "context retained row relocation"
        );
        let mut ctl = vec![0; c.control.byte_len()];
        c.control.download(&mut ctl)?;
        ensure!(
            ctl[..24] == *bytes(&p.control()) && ctl[24..].iter().all(|&v| v == 0xff),
            "context control mutated"
        );
        guards += GUARD;
        let counter_words =
            c.fc.check_counters()? + c.k.check_counters()? + c.v.check_counters()?;
        c.state.set(p.after);
        c.bank.set(next);
        Ok(Run {
            stages,
            k,
            v,
            state: p.after,
            total_us,
            enqueue_us,
            completion_us,
            canary_bytes: guards,
            counter_words,
            old_context_bytes: p.before.len * KV * 4,
        })
    }
}
