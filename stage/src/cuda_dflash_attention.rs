//! R10c native attention core. Resident context; dynamic controls in fixed graphs.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    dflash_attention::{
        D, Fixture, KH, MAX_CTX, MAX_ROWS, MAX_SPLITS, Norms, QH, Shape, frequencies,
    },
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde_json::json;
use std::{cell::Cell, path::Path, time::Instant};
pub const CUBIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/kernels/dflash_attention.sm120.cubin"
);
pub const SHA: &str = "b22bcaf5b29380013387cef067d2be70a77cfcba85936c8dd942b803308e18fa";
const GUARD: usize = 256;
fn bytes(x: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len() * 2) }
}
fn checked_upload(b: &Buffer<'_>, raw: &[u8]) -> Result<()> {
    if raw.is_empty() {
        return Ok(());
    }
    b.upload(raw)?;
    let mut check = vec![0; raw.len()];
    b.download(&mut check)?;
    ensure!(check == raw, "attention upload/readback");
    Ok(())
}
pub fn load_module(d: &Device) -> Result<Module<'_>> {
    d.module(Path::new(CUBIN), SHA)
}
pub struct Attention<'m, 'd> {
    device: &'d Device,
    rope: Kernel<'m, 'd>,
    part: Kernel<'m, 'd>,
    merge: Kernel<'m, 'd>,
    q: Buffer<'d>,
    k: Buffer<'d>,
    v: Buffer<'d>,
    ck: Buffer<'d>,
    cv: Buffer<'d>,
    qw: Buffer<'d>,
    kw: Buffer<'d>,
    freq: Buffer<'d>,
    control: Buffer<'d>,
    qr: Buffer<'d>,
    kr: Buffer<'d>,
    partial: Buffer<'d>,
    out: Buffer<'d>,
    shape: Cell<Option<Shape>>,
}
impl<'m, 'd> Attention<'m, 'd> {
    pub fn new(d: &'d Device, m: &'m Module<'d>, n: &Norms) -> Result<Self> {
        ensure!(
            n.q.len() == D
                && n.k.len() == D
                && n.q
                    .iter()
                    .chain(&n.k)
                    .all(|&x| bf16::from_bits(x).is_finite()),
            "attention norm weights"
        );
        let this = Self {
            device: d,
            rope: m.kernel("dflash_qk_rope")?,
            part: m.kernel("dflash_attn_part")?,
            merge: m.kernel("dflash_attn_merge")?,
            q: d.alloc(MAX_ROWS * QH * D * 2 + GUARD)?,
            k: d.alloc(MAX_ROWS * KH * D * 2 + GUARD)?,
            v: d.alloc(MAX_ROWS * KH * D * 2 + GUARD)?,
            ck: d.alloc(MAX_CTX * KH * D * 2)?,
            cv: d.alloc(MAX_CTX * KH * D * 2)?,
            qw: d.alloc(D * 2)?,
            kw: d.alloc(D * 2)?,
            freq: d.alloc(64 * 4)?,
            control: d.alloc(5 * 4 + GUARD)?,
            qr: d.alloc(MAX_ROWS * QH * D * 2 + GUARD)?,
            kr: d.alloc(MAX_ROWS * KH * D * 2 + GUARD)?,
            partial: d.alloc(MAX_ROWS * QH * MAX_SPLITS * 130 * 4 + GUARD)?,
            out: d.alloc(MAX_ROWS * QH * D * 2 + GUARD)?,
            shape: Cell::new(None),
        };
        checked_upload(&this.qw, bytes(&n.q))?;
        checked_upload(&this.kw, bytes(&n.k))?;
        checked_upload(
            &this.freq,
            &frequencies()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(this)
    }
    pub fn description(&self) -> serde_json::Value {
        let allocated: usize = [
            &self.q,
            &self.k,
            &self.v,
            &self.ck,
            &self.cv,
            &self.qw,
            &self.kw,
            &self.freq,
            &self.control,
            &self.qr,
            &self.kr,
            &self.partial,
            &self.out,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum();
        json!({"backend":"native Rust/CUDA/libtinynv", "cubin_sha256":SHA,
            "kernels":[self.rope.description(),self.part.description(),self.merge.description()],
            "resident_allocated_bytes":allocated,"weight_readback_verified":true,"dependent_device_calls":3,
            "intermediate_host_transfers":0,"maximum_rows":MAX_ROWS,"maximum_context":MAX_CTX,
            "query_heads":QH,"kv_heads":KH,"head_dim":D,"position_and_shapes_read_from_control_buffer":true,
            "context_already_normalized_rotated":true,"context_append_not_timed":true})
    }
    pub fn prepare(&self, f: &Fixture) -> Result<f64> {
        // Invalidate before validation/upload; a failed preparation cannot replay stale data.
        self.shape.set(None);
        f.validate()?;
        let t = Instant::now();
        for (b, x) in [
            (&self.q, &f.q),
            (&self.k, &f.k),
            (&self.v, &f.v),
            (&self.ck, &f.ck),
            (&self.cv, &f.cv),
        ] {
            checked_upload(b, bytes(x))?;
        }
        let mut raw: Vec<u8> = f
            .shape
            .control()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        raw.resize(5 * 4 + GUARD, 0xff);
        checked_upload(&self.control, &raw)?;
        self.shape.set(Some(f.shape));
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    pub(crate) fn block_buffers(&self) -> (&Buffer<'d>, &Buffer<'d>, &Buffer<'d>) {
        (&self.q, &self.k, &self.v)
    }
    pub(crate) fn output_buffer(&self) -> &Buffer<'d> {
        &self.out
    }
    pub(crate) fn prepared_shape(&self) -> Option<Shape> {
        self.shape.get()
    }
    pub(crate) fn prepare_context(&self, shape: Shape, ck: &[u16], cv: &[u16]) -> Result<()> {
        self.shape.set(None);
        shape.validate()?;
        ensure!(
            ck.len() == shape.ctx * KH * D
                && cv.len() == ck.len()
                && ck.iter().chain(cv).all(|&v| bf16::from_bits(v).is_finite()),
            "attention context size/finite"
        );
        checked_upload(&self.ck, bytes(ck))?;
        checked_upload(&self.cv, bytes(cv))?;
        let mut ctl: Vec<_> = shape
            .control()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctl.resize(20 + GUARD, 0xff);
        checked_upload(&self.control, &ctl)?;
        self.shape.set(Some(shape));
        Ok(())
    }
    pub(crate) fn check_context(&self, ck: &[u16], cv: &[u16]) -> Result<usize> {
        let s = self.shape.get().context("context check unprepared")?;
        ensure!(
            ck.len() == s.ctx * KH * D && cv.len() == ck.len(),
            "context check shape"
        );
        for (b, want) in [(&self.ck, ck), (&self.cv, cv)] {
            if want.is_empty() {
                continue;
            }
            let mut got = vec![0; want.len() * 2];
            b.download(&mut got)?;
            ensure!(got == bytes(want), "context changed during noise block");
        }
        Ok((ck.len() + cv.len()) * 2)
    }
    pub(crate) fn poison(&self) -> Result<()> {
        for b in [&self.qr, &self.kr, &self.out, &self.partial] {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        Ok(())
    }
    pub(crate) fn record(&self, splits: usize) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        ensure!([1, 4, 16].contains(&splits), "attention context partitions");
        // Source ABI audit: fixed grids cover the allocation capacity, all tensors
        // disjoint and retained; only checked five-word control can change rows.
        unsafe {
            Ok(vec![
                self.rope.prepare(
                    vec![
                        Arg::Buffer(&self.q),
                        Arg::Buffer(&self.k),
                        Arg::Buffer(&self.qw),
                        Arg::Buffer(&self.kw),
                        Arg::Buffer(&self.freq),
                        Arg::Buffer(&self.qr),
                        Arg::Buffer(&self.kr),
                        Arg::Buffer(&self.control),
                    ],
                    [(MAX_ROWS * (QH + KH)) as u32, 1, 1],
                    [32, 1, 1],
                )?,
                self.part.prepare(
                    vec![
                        Arg::Buffer(&self.qr),
                        Arg::Buffer(&self.kr),
                        Arg::Buffer(&self.v),
                        Arg::Buffer(&self.ck),
                        Arg::Buffer(&self.cv),
                        Arg::Buffer(&self.partial),
                        Arg::Buffer(&self.control),
                        Arg::U32(splits as u32),
                    ],
                    [(MAX_ROWS * QH) as u32, splits as u32, 1],
                    [32, 1, 1],
                )?,
                self.merge.prepare(
                    vec![
                        Arg::Buffer(&self.partial),
                        Arg::Buffer(&self.out),
                        Arg::Buffer(&self.control),
                        Arg::U32(splits as u32),
                    ],
                    [(MAX_ROWS * QH) as u32, 1, 1],
                    [128, 1, 1],
                )?,
            ])
        }
    }
    pub(crate) fn collect(&self, splits: usize) -> Result<Collected> {
        ensure!([1, 4, 16].contains(&splits), "attention collect splits");
        let a = self;
        let shape = self.shape.get().context("attention collect unprepared")?;
        let mut canary_bytes_checked = 0;
        let mut collect = |b: &Buffer<'_>, active: usize| -> Result<Vec<u8>> {
            let mut raw = vec![0; b.byte_len()];
            b.download(&mut raw)?;
            ensure!(
                raw[active..].iter().all(|&v| v == 0xff),
                "attention output/scratch/control guard"
            );
            canary_bytes_checked += raw.len() - active;
            raw.truncate(active);
            Ok(raw)
        };
        let bf = |raw: Vec<u8>| -> Result<Vec<f32>> {
            let v: Vec<_> = raw
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect();
            ensure!(
                v.iter().all(|v| v.is_finite()),
                "attention nonfinite/unwritten result"
            );
            Ok(v)
        };
        let q = bf(collect(&a.qr, shape.rows * QH * D * 2)?)?;
        let k = bf(collect(&a.kr, shape.rows * KH * D * 2)?)?;
        let out = bf(collect(&a.out, shape.rows * QH * D * 2)?)?;
        let raw = collect(&a.partial, shape.rows * QH * splits * 130 * 4)?;
        let partial: Vec<_> = raw
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let chunk = (shape.ctx + shape.rows).div_ceil(splits);
        let mut empty_partitions = 0;
        for (i, p) in partial.chunks_exact(130).enumerate() {
            let sp = i % splits;
            let row = i / splits / QH;
            let lo = sp * chunk;
            let hi = (shape.ctx + shape.rows).min(lo + chunk);
            let count = (lo..hi).filter(|&key| shape.allowed(row, key)).count();
            if count == 0 {
                ensure!(
                    p[..128].iter().all(|&x| x == 0.)
                        && p[128] == f32::NEG_INFINITY
                        && p[129] == 0.,
                    "attention empty partition"
                );
                empty_partitions += 1;
            } else {
                ensure!(
                    p.iter().all(|x| x.is_finite())
                        && p[129] > 0.
                        && p[129] <= count as f32 + 0.001,
                    "attention nonempty partition"
                );
            }
        }
        let ctl = collect(&a.control, 20)?;
        ensure!(
            ctl == shape
                .control()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            "attention control mutated"
        );
        Ok(Collected {
            q,
            k,
            out,
            canary_bytes_checked,
            empty_partitions,
            partial_values_checked: partial.len(),
        })
    }
    pub fn plan(&self, splits: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            self.shape.get().is_some(),
            "prepare attention before recording"
        );
        let t = Instant::now();
        let calls = self.record(splits)?;
        let execute = if graph {
            Executor::Graph(self.device.capture(calls)?)
        } else {
            Executor::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            execute,
            splits,
            setup_us: t.elapsed().as_secs_f64() * 1e6,
        })
    }
}
pub(crate) struct Collected {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub out: Vec<f32>,
    pub canary_bytes_checked: usize,
    pub empty_partitions: usize,
    pub partial_values_checked: usize,
}
enum Executor<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Attention<'m, 'd>,
    execute: Executor<'a, 'm, 'd>,
    splits: usize,
    pub setup_us: f64,
}
pub struct Run {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub out: Vec<f32>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub canary_bytes_checked: usize,
    pub empty_partitions: usize,
    pub partial_values_checked: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, repetitions: usize, wait_each: bool) -> Result<Run> {
        let a = self.owner;
        a.shape.get().context("attention not prepared")?;
        ensure!(
            (1..=32).contains(&repetitions)
                && (!wait_each || matches!(&self.execute, Executor::Calls(_))),
            "attention repetitions/wait mode"
        );
        a.poison()?;
        a.device.sync()?;
        let t = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repetitions {
            match &self.execute {
                Executor::Calls(calls) => {
                    for c in calls {
                        let t = Instant::now();
                        c.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            a.device.sync()?;
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
        a.device.sync()?;
        completion_us += wait.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let Collected {
            q,
            k,
            out,
            canary_bytes_checked,
            empty_partitions,
            partial_values_checked,
        } = a.collect(self.splits)?;
        Ok(Run {
            q,
            k,
            out,
            total_us,
            enqueue_us,
            completion_us,
            canary_bytes_checked,
            empty_partitions,
            partial_values_checked,
        })
    }
}
