//! Resident CUDA GDN projections, recurrence and exact accepted-prefix state.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::Linear,
    gdn::{C, Core, DK, DV, H, HK, HV, Initial, MAX_T, STATE, V},
    lowbit_gdn::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "bdbaf9ff96bf32915a14e9a337d7a05c889f2039b21f2c6d0e8a3a398ce8d714";
const GUARD: usize = 256;
/// R52: early-launch one-row recurrence that requests its carried state before
/// waiting for the projections (candidate arm only; same outputs).
pub fn r52_gdn_enabled() -> bool {
    crate::cuda_affine4::r39_enabled() && std::env::var("QWEN_R52_GDN").as_deref() == Ok("1")
}
/// R52: early-launch norm/gate (waits at entry) so the output projection can
/// launch, and request its weights, during the recurrence.
pub fn r52_ng_enabled() -> bool {
    crate::cuda_affine4::r39_enabled() && std::env::var("QWEN_R52_NG").as_deref() == Ok("1")
}
/// R53: one-row recurrence with the carried state requested before the
/// convolution/gate prologue, launched normally (no early launch); candidate arm.
pub fn r53_gdnh_enabled() -> bool {
    crate::cuda_affine4::r39_enabled() && std::env::var("QWEN_R53_GDNH").as_deref() == Ok("1")
}
/// R53: early-launch norm/gate with its constant norm weights and z requested
/// before the wait (candidate arm; same outputs). Needs QWEN_R52_NG=1.
pub fn r53_ng5_enabled() -> bool {
    r52_ng_enabled() && std::env::var("QWEN_R53_NG5").as_deref() == Ok("1")
}
/// R53: which one-row recurrence this engine records. QWEN_R53_GDNX=1: conv-window
/// copy with all loads first (gdn_fused1x). QWEN_R53_GDNV=1: also 16-byte state
/// loads (gdn_fused1v). QWEN_R53_GDNL=N: the first N GDN engines built in this
/// process load their state with L2::evict_last (gdn_fused1l), the rest use 1v.
/// Same values in every case.
static R53_GDN_ENGINES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn r53_fused1h_name() -> &'static str {
    let index = R53_GDN_ENGINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(n) = std::env::var("QWEN_R53_GDNL").ok().and_then(|v| v.parse::<usize>().ok()) {
        return if index < n { "gdn_fused1l" } else { "gdn_fused1v" };
    }
    if std::env::var("QWEN_R53_GDNV").as_deref() == Ok("1") { return "gdn_fused1v"; }
    if std::env::var("QWEN_R53_GDNX").as_deref() == Ok("1") { "gdn_fused1x" } else { "gdn_fused1h" }
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    if r53_gdnh_enabled() || r53_ng5_enabled() {
        let pin: serde_json::Value = serde_json::from_str(include_str!("../pins/R53-GDN-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_gdn_r53.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r52_gdn_enabled() || r52_ng_enabled() {
        let pin: serde_json::Value = serde_json::from_str(include_str!("../pins/R52-GDN-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_gdn_r52.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_gdn.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![0xff; b.byte_len()])
}
fn raw(b: &Buffer<'_>, n: usize) -> Result<Vec<u8>> {
    ensure!(n + GUARD <= b.byte_len(), "GDN read bound");
    let mut out = vec![0; b.byte_len()];
    b.download(&mut out)?;
    ensure!(out[n..].iter().all(|&v| v == 0xff), "GDN output tail");
    out.truncate(n);
    Ok(out)
}
fn bf(b: &Buffer<'_>, n: usize) -> Result<Vec<u16>> {
    let out: Vec<_> = raw(b, n * 2)?
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        out.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "GDN BF16 finite/fully written"
    );
    Ok(out)
}
fn fp(b: &Buffer<'_>, n: usize) -> Result<Vec<f32>> {
    let out: Vec<_> = raw(b, n * 4)?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(
        out.iter().all(|v| v.is_finite()),
        "GDN FP32 finite/fully written"
    );
    Ok(out)
}
struct Pair<'d> {
    f: Buffer<'d>,
    b: Buffer<'d>,
}
impl<'d> Pair<'d> {
    fn new(d: &'d Device, n: usize) -> Result<Self> {
        Ok(Self {
            f: d.alloc(MAX_T * n * 4 + GUARD)?,
            b: d.alloc(MAX_T * n * 2 + GUARD)?,
        })
    }
    fn poison(&self) -> Result<()> {
        poison(&self.f)?;
        poison(&self.b)
    }
    fn read(&self, n: usize) -> Result<(Vec<f32>, Vec<u16>)> {
        let f = fp(&self.f, n)?;
        let b = bf(&self.b, n)?;
        ensure!(
            f.iter()
                .zip(&b)
                .all(|(&f, &b)| bf16::from_f32(f).to_bits() == b),
            "GDN projection rounding"
        );
        Ok((f, b))
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub fused: bool,
    pub all_states: bool,
    pub projection: usize,
}
impl Gdn<'_, '_> {
    /// R52 pump: (device address, bytes) of the four same-input projections'
    /// packed codes, scales and biases (the multi4 launch's weights).
    pub(crate) fn multi4_regions(&self) -> Vec<(u64, usize)> {
        let mut v = vec![];
        for l in self.linears.iter().take(4) {
            let (w, s, b, _) = l.packed_weights();
            for x in [w, s, b] { v.push((x.device_addr(), x.byte_len())); }
        }
        v
    }
}
pub struct Gdn<'m, 'd> {
    d: &'d Device,
    linears: Vec<Linear<'m, 'd>>,
    projections: Vec<Pair<'d>>,
    x: Buffer<'d>,
    weights: Vec<Buffer<'d>>,
    initial_conv: Buffer<'d>,
    initial_state: Buffer<'d>,
    next_conv: Buffer<'d>,
    next_state: Buffer<'d>,
    q: Buffer<'d>,
    k: Buffer<'d>,
    v: Buffer<'d>,
    gates: Buffer<'d>,
    y: Buffer<'d>,
    gated: Buffer<'d>,
    states: Buffer<'d>,
    prefix_capacity: usize,
    conv_out: Buffer<'d>,
    /// R56 zero-copy drafting (single-row steps): the core reads its initial state from snapshot
    /// slot 0 of `states` (each thread loads its state elements before storing the snapshot to
    /// the same elements); the FP32 state is never copied; the model commits the convolution
    /// history with copy tables.
    zero_copy: std::cell::Cell<bool>,
    prepare: Kernel<'m, 'd>,
    recur: Kernel<'m, 'd>,
    fused: [Kernel<'m, 'd>; 4],
    r52_fused1: Option<Kernel<'m, 'd>>,
    r53_fused1h: Option<Kernel<'m, 'd>>,
    r52_norm: Option<Kernel<'m, 'd>>,
    norm: Kernel<'m, 'd>,
    /// R60: canonical in-projections (qkv/z/a/b) in one affine8_multi4 launch
    multi8: Option<crate::cuda_affine4::Multi8<'m, 'd>>,
    commit: Kernel<'m, 'd>,
    restore: Kernel<'m, 'd>,
    rows: Cell<usize>,
    has_state: Cell<bool>,
    valid: Cell<Option<(usize, bool)>>,
}
impl<'m, 'd> Gdn<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        Self::new_with_prefix_capacity(d, affine, ops, w, MAX_T)
    }
    /// Reserve only the requested number of intermediate recurrent checkpoints.
    /// Final-only execution still permits MAX_T rows; all-prefix execution must
    /// fit this capacity. Kernel arithmetic and persistent state are unchanged.
    pub fn new_with_prefix_capacity(
        d: &'d Device,
        affine: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
        prefix_capacity: usize,
    ) -> Result<Self> {
        w.validate()?;
        let linears = w.projections.iter().map(|w| Linear::new(d, affine, w)).collect::<Result<Vec<_>>>()?;
        Self::assemble(d, ops, linears, [&w.conv, &w.a_log, &w.dt_bias, &w.norm], prefix_capacity, true)
    }
    /// R60 pipeline split: the canonical target's GDN (affine8 qkv/z/a/b/out from `m8`, the affine8
    /// module; recurrence kernels unchanged). Baseline kernels only: record with projection arm
    /// 0-3 and no R52/R53 fused variants.
    pub fn new_canonical(
        d: &'d Device,
        m8: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &crate::gdn::Weights,
        prefix_capacity: usize,
    ) -> Result<Self> {
        w.validate()?;
        let linears = w.projections.iter().map(|w| Linear::new_affine8(d, m8, w)).collect::<Result<Vec<_>>>()?;
        let mut g = Self::assemble(d, ops, linears, [&w.conv, &w.a_log, &w.dt_bias, &w.norm], prefix_capacity, false)?;
        g.multi8 = Some(crate::cuda_affine4::Multi8::load(m8)?);
        Ok(g)
    }
    fn assemble(
        d: &'d Device,
        ops: &'m Module<'d>,
        linears: Vec<Linear<'m, 'd>>,
        small: [&Vec<u16>; 4],
        prefix_capacity: usize,
        fast: bool,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_T).contains(&prefix_capacity),
            "GDN prefix capacity"
        );
        ensure!(
            linears.len() == 5
                && small[0].len() == C * 4
                && small[1].len() == HV
                && small[2].len() == HV
                && small[3].len() == DV,
            "GDN weight geometry"
        );
        let upload = |v: &[u16]| -> Result<Buffer<'d>> {
            let b = d.alloc(v.len() * 2)?;
            b.upload(bytes(v))?;
            let mut r = vec![0; v.len() * 2];
            b.download(&mut r)?;
            ensure!(r == bytes(v), "GDN weight upload readback");
            Ok(b)
        };
        Ok(Self {
            d,
            linears,
            projections: [C, V, HV, HV, H]
                .into_iter()
                .map(|n| Pair::new(d, n))
                .collect::<Result<_>>()?,
            x: d.alloc(MAX_T * H * 2 + GUARD)?,
            weights: small
                .into_iter()
                .map(|w| upload(w))
                .collect::<Result<_>>()?,
            initial_conv: d.alloc(3 * C * 2 + GUARD)?,
            initial_state: d.alloc(STATE * 4 + GUARD)?,
            next_conv: d.alloc(3 * C * 2 + GUARD)?,
            next_state: d.alloc(STATE * 4 + GUARD)?,
            q: d.alloc(MAX_T * HK * DK * 2 + GUARD)?,
            k: d.alloc(MAX_T * HK * DK * 2 + GUARD)?,
            v: d.alloc(MAX_T * V * 2 + GUARD)?,
            gates: d.alloc(MAX_T * HV * 2 * 4 + GUARD)?,
            y: d.alloc(MAX_T * V * 2 + GUARD)?,
            gated: d.alloc(MAX_T * V * 2 + GUARD)?,
            states: d.alloc(prefix_capacity * STATE * 4 + GUARD)?,
            prefix_capacity,
            zero_copy: std::cell::Cell::new(false),
            conv_out: d.alloc(3 * C * 2 + GUARD)?,
            prepare: ops.kernel("gdn_prepare")?,
            recur: ops.kernel("gdn_recur")?,
            r52_fused1: if fast && r52_gdn_enabled() { Some(ops.kernel("r39pdl_gdn_fused1p")?) } else { None },
            // R53: QWEN_R53_GDNX=1 swaps in gdn_fused1x (conv-window copy with all
            // loads requested first; same values).
            r53_fused1h: if fast && r53_gdnh_enabled() { Some(ops.kernel(r53_fused1h_name())?) } else { None },
            r52_norm: if fast && r53_ng5_enabled() { Some(ops.kernel("r39pdl_gdn_norm_gate5")?) }
                else if fast && r52_ng_enabled() { Some(ops.kernel("r39pdl_gdn_norm_gate")?) } else { None },
            fused: [
                ops.kernel("gdn_fused1")?,
                ops.kernel("gdn_fused4")?,
                ops.kernel("gdn_fused8")?,
                ops.kernel("gdn_fused16")?,
            ],
            norm: ops.kernel("gdn_norm_gate")?,
            multi8: None,
            commit: ops.kernel("gdn_commit")?,
            restore: ops.kernel("gdn_restore")?,
            rows: Cell::new(0),
            has_state: Cell::new(false),
            valid: Cell::new(None),
        })
    }
    pub fn description(&self) -> Value {
        let extra: usize = [
            &self.x,
            &self.initial_conv,
            &self.initial_state,
            &self.next_conv,
            &self.next_state,
            &self.q,
            &self.k,
            &self.v,
            &self.gates,
            &self.y,
            &self.gated,
            &self.states,
            &self.conv_out,
        ]
        .into_iter()
        .chain(self.weights.iter())
        .chain(self.projections.iter().flat_map(|p| [&p.f, &p.b]))
        .map(|b| b.byte_len())
        .sum();
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":SHA,"max_rows":MAX_T,"prefix_capacity":self.prefix_capacity,"state_values":STATE,"extra_bytes":extra,"linears":self.linears.iter().map(|l|l.description()).collect::<Vec<_>>(),
   "kernels":[self.prepare.description(),self.recur.description(),self.fused[0].description(),self.fused[1].description(),self.fused[2].description(),self.fused[3].description(),self.norm.description(),self.commit.description(),self.restore.description()],
   "host_intermediates":0,"state_policy":"all-prefix or final-only FP32 checkpoint; explicit accepted-prefix gather+restore on device"})
    }
    pub fn prepare(&self, x: &[u16], initial: &Initial) -> Result<()> {
        self.prepare_initial(initial)?;
        self.prepare_input(x)
    }
    pub(crate) fn prepare_initial(&self, initial: &Initial) -> Result<()> {
        self.rows.set(0);
        self.valid.set(None);
        self.has_state.set(false);
        initial.validate()?;
        poison(&self.initial_conv)?;
        poison(&self.initial_state)?;
        self.initial_conv.upload(bytes(&initial.conv))?;
        self.initial_state.upload(bytes(&initial.state))?;
        ensure!(
            bf(&self.initial_conv, 3 * C)? == initial.conv
                && fp(&self.initial_state, STATE)? == initial.state,
            "GDN initial upload"
        );
        if self.zero_copy.get() {
            self.states.upload(bytes(&initial.state))?;
        }
        self.has_state.set(true);
        Ok(())
    }
    /// R58 seeding: prepare_initial for a trusted state, with no poison or read-back canaries. In
    /// zero-copy mode the state goes only to the live snapshot slot (what execution and checkpoints
    /// read), otherwise to the initial buffer.
    pub(crate) fn prepare_initial_lean(&self, initial: &Initial) -> Result<()> {
        self.rows.set(0);
        self.valid.set(None);
        self.has_state.set(false);
        ensure!(initial.conv.len() == 3 * C && initial.state.len() == STATE, "GDN initial geometry");
        self.initial_conv.upload(bytes(&initial.conv))?;
        if self.zero_copy.get() {
            self.states.upload(bytes(&initial.state))?;
        } else {
            self.initial_state.upload(bytes(&initial.state))?;
        }
        self.has_state.set(true);
        Ok(())
    }
    /// Keep committed state on the card while supplying the next activation rows.
    pub fn prepare_input(&self, x: &[u16]) -> Result<()> {
        self.rows.set(0);
        self.valid.set(None);
        ensure!(
            self.has_state.get()
                && !x.is_empty()
                && x.len() % H == 0
                && x.len() / H <= MAX_T
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "GDN input/state/finite"
        );
        poison(&self.x)?;
        self.x.upload(bytes(x))?;
        ensure!(bf(&self.x, x.len())? == x, "GDN input readback");
        self.rows.set(x.len() / H);
        Ok(())
    }
    pub(crate) fn prepare_resident_input(&self, rows: usize) -> Result<()> {
        self.rows.set(0);
        self.valid.set(None);
        ensure!(
            self.has_state.get() && (1..=MAX_T).contains(&rows),
            "GDN resident input/state"
        );
        self.rows.set(rows);
        Ok(())
    }
    pub(crate) fn invalidate_output(&self) {
        self.valid.set(None);
    }
    pub fn initial(&self) -> Result<Initial> {
        ensure!(self.has_state.get(), "GDN initial unavailable");
        Ok(Initial {
            conv: bf(&self.initial_conv, 3 * C)?,
            state: fp(&self.initial_state, STATE)?,
        })
    }
    /// R60 S4: the committed conv/recurrent state, read exactly (no guard-tail canary and no
    /// finite check): the stage's lean seeds do not poison the guard bytes, and the prefill
    /// pull-back reads this once per recurrent layer per chunk.
    pub fn initial_lean(&self) -> Result<Initial> {
        ensure!(self.has_state.get(), "GDN initial unavailable");
        let mut conv = vec![0u8; 3 * C * 2];
        let mut state = vec![0u8; STATE * 4];
        self.initial_conv.view(0, conv.len())?.download(&mut conv)?;
        self.initial_state.view(0, state.len())?.download(&mut state)?;
        Ok(Initial {
            conv: conv.chunks_exact(2).map(|v| u16::from_le_bytes(v.try_into().unwrap())).collect(),
            state: state.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect(),
        })
    }
    pub fn output(&self) -> &Buffer<'d> {
        &self.projections[4].b
    }
    fn args<'a>(&'a self, rows: usize, p: Policy) -> Vec<Arg<'a, 'd>> {
        vec![
            Arg::Buffer(&self.projections[0].b),
            Arg::Buffer(&self.projections[1].b),
            Arg::Buffer(&self.projections[2].b),
            Arg::Buffer(&self.projections[3].b),
            Arg::Buffer(&self.initial_conv),
            Arg::Buffer(if self.zero_copy.get() { &self.states } else { &self.initial_state }),
            Arg::Buffer(&self.weights[0]),
            Arg::Buffer(&self.weights[1]),
            Arg::Buffer(&self.weights[2]),
            Arg::Buffer(&self.weights[3]),
            Arg::Buffer(&self.q),
            Arg::Buffer(&self.k),
            Arg::Buffer(&self.v),
            Arg::Buffer(&self.gates),
            Arg::Buffer(&self.y),
            Arg::Buffer(&self.gated),
            Arg::Buffer(&self.states),
            Arg::Buffer(&self.conv_out),
            Arg::U32(rows as u32),
            Arg::U32(u32::from(p.all_states)),
        ]
    }
    pub fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (1..=MAX_T).contains(&rows) && x.byte_len() >= rows * H * 2 && p.projection <= 4,
            "GDN record shape"
        );
        ensure!(
            !p.all_states || rows <= self.prefix_capacity,
            "GDN insufficient prefix capacity"
        );
        let outputs = [
            &self.q,
            &self.k,
            &self.v,
            &self.gates,
            &self.y,
            &self.gated,
            &self.states,
            &self.conv_out,
        ];
        ensure!(
            outputs
                .into_iter()
                .chain(self.projections.iter().flat_map(|p| [&p.f, &p.b]))
                .all(|b| !x.aliases(b)),
            "GDN input/output alias"
        );
        let arm = p.projection;
        let mut calls = Vec::new();
        if crate::cuda_affine4::ahead_selected() && arm == 0 && !self.linears[0].is_affine8() {
            // R38: qkv/z/a/b in one launch, same blocks and outputs.
            let o = &self.projections;
            calls.push(crate::cuda_affine4::record_multi4(
                [&self.linears[0], &self.linears[1], &self.linears[2], &self.linears[3]],
                x,
                [(&o[0].f, &o[0].b), (&o[1].f, &o[1].b), (&o[2].f, &o[2].b), (&o[3].f, &o[3].b)],
                rows,
            )?);
        } else if let (Some(m8), 0) = (&self.multi8, arm) {
            let o = &self.projections;
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
                    &self.projections[j].f,
                    &self.projections[j].b,
                    rows,
                    arm,
                )?);
            }
        }
        if p.fused {
            let k = if rows == 1 {
                0
            } else if rows <= 4 {
                1
            } else if rows <= 8 {
                2
            } else {
                3
            };
            let kernel = match (&self.r52_fused1, &self.r53_fused1h) {
                (Some(f), _) if k == 0 && crate::cuda_affine4::pdl_selected() => f,
                (_, Some(f)) if k == 0 && crate::cuda_affine4::pdl_selected() => f,
                _ => &self.fused[k],
            };
            calls.push(unsafe {
                kernel.prepare(self.args(rows, p), [(HV * 4) as u32, 1, 1], [128, 1, 1])?
            });
        } else {
            calls.push(unsafe {
                self.prepare.prepare(
                    self.args(rows, p),
                    [HV as u32, rows as u32, 1],
                    [128, 1, 1],
                )?
            });
            calls.push(unsafe {
                self.recur
                    .prepare(self.args(rows, p), [(HV * 4) as u32, 1, 1], [128, 1, 1])?
            });
        }
        let norm = match &self.r52_norm {
            Some(k) if crate::cuda_affine4::pdl_selected() => k,
            _ => &self.norm,
        };
        calls.push(unsafe {
            norm.prepare(self.args(rows, p), [HV as u32, rows as u32, 1], [32, 1, 1])?
        });
        calls.push(self.linears[4].record_into(
            &self.gated,
            &self.projections[4].f,
            &self.projections[4].b,
            rows,
            arm,
        )?);
        Ok(calls)
    }
    pub(crate) fn poison(&self) -> Result<()> {
        self.valid.set(None);
        for p in &self.projections {
            p.poison()?;
        }
        for b in [
            &self.q,
            &self.k,
            &self.v,
            &self.gates,
            &self.y,
            &self.gated,
            &self.states,
            &self.conv_out,
        ] {
            poison(b)?;
        }
        Ok(())
    }
    pub(crate) fn collect(&self, t: usize, p: Policy) -> Result<Run> {
        ensure!(
            (1..=MAX_T).contains(&t) && (!p.all_states || t <= self.prefix_capacity),
            "GDN collect prefix capacity"
        );
        let mut projected = Vec::new();
        let mut projection_raw = Vec::new();
        for (j, n) in [C, V, HV, HV].into_iter().enumerate() {
            let (f, b) = self.projections[j].read(t * n)?;
            projected.push(b);
            projection_raw.push(f);
        }
        let (output_raw, output) = self.projections[4].read(t * H)?;
        let core = Core {
            y: bf(&self.y, t * V)?,
            gated: bf(&self.gated, t * V)?,
            states: fp(&self.states, if p.all_states { t * STATE } else { STATE })?,
            conv: bf(&self.conv_out, 3 * C)?,
        };
        if p.fused {
            for b in [&self.q, &self.k, &self.v, &self.gates] {
                raw(b, 0)?;
            }
        } else {
            bf(&self.q, t * HK * DK)?;
            bf(&self.k, t * HK * DK)?;
            bf(&self.v, t * V)?;
            let g = fp(&self.gates, t * HV * 2)?;
            ensure!(g.iter().all(|v| (0.0..=1.0).contains(v)), "GDN gate bounds");
        }
        self.valid.set(Some((t, p.all_states)));
        Ok(Run {
            projected,
            projection_raw,
            core,
            output,
            output_raw,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        let rows = self.rows.get();
        ensure!(
            rows > 0 && self.has_state.get() && (1..=8).contains(&repetitions),
            "GDN plan bounds"
        );
        let mut c = self.record_from(&self.x, rows, p)?;
        let n = c.len();
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
            rows,
            p,
            exec,
            repetitions,
            calls: n,
        })
    }
    /// R56 lean drafting commit: the commit's gather/restore kernels as prepared calls,
    /// without the diagnostic poison uploads or a sync; finish_commit updates host state
    /// once they have executed.
    pub(crate) fn record_commit(&self, accepted: usize, rows: usize, all: bool) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        ensure!(rows > 0 && accepted <= rows && (all || accepted == 0 || accepted == rows), "GDN lean prefix");
        if self.zero_copy.get() {
            ensure!(accepted == 1 && rows == 1, "zero-copy GDN commits accept the single drafted row");
            return Ok(vec![]);
        }
        let c = unsafe {
            self.commit.prepare(
                vec![
                    Arg::Buffer(&self.projections[0].b),
                    Arg::Buffer(&self.initial_conv),
                    Arg::Buffer(&self.initial_state),
                    Arg::Buffer(&self.states),
                    Arg::Buffer(&self.next_conv),
                    Arg::Buffer(&self.next_state),
                    Arg::U32(accepted as u32),
                    Arg::U32(rows as u32),
                    Arg::U32(u32::from(all)),
                ],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let restore = unsafe {
            self.restore.prepare(
                vec![
                    Arg::Buffer(&self.next_conv),
                    Arg::Buffer(&self.next_state),
                    Arg::Buffer(&self.initial_conv),
                    Arg::Buffer(&self.initial_state),
                ],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        Ok(vec![c, restore])
    }
    /// R56: the committed recurrent state (convolution history, FP32 state).
    pub(crate) fn state_buffers(&self) -> Vec<&Buffer<'d>> {
        if self.zero_copy.get() {
            return vec![&self.initial_conv, &self.states];
        }
        vec![&self.initial_conv, &self.initial_state]
    }
    pub(crate) fn set_zero_copy(&self, on: bool) {
        self.zero_copy.set(on);
    }
    pub(crate) fn zero_copy(&self) -> bool {
        self.zero_copy.get()
    }
    /// R56 zero-copy commit of one accepted row as (dst, src, bytes) device copies: gather
    /// next_conv = [conv_in rows 1..3, the row's pre-convolution channels] (gdn_commit with
    /// accepted 1), then restore conv_in = next_conv in a later launch.
    /// R58 batched prefill, zero-copy: after `rows` (>= 3) committed rows the convolution history
    /// is the last three pre-convolution rows, copied straight into conv_in (dst, src, bytes).
    /// R58: enable the multi-row MMA arm (projection 4) on the five projections.
    pub(crate) fn enable_rows(&mut self, m: &'m crate::cuda::Module<'d>) -> Result<()> {
        for l in &mut self.linears {
            l.enable_rows(m)?;
        }
        Ok(())
    }
    pub(crate) fn conv_copy_rows(&self, rows: usize) -> Result<(u64, u64, u64)> {
        ensure!((3..=MAX_T).contains(&rows), "zero-copy conv rows");
        let row = (C * 2) as u64;
        Ok((self.initial_conv.device_ptr(), self.projections[0].b.device_ptr() + (rows as u64 - 3) * row, 3 * row))
    }
    pub(crate) fn conv_copies(&self) -> ([(u64, u64, u64); 2], (u64, u64, u64)) {
        let row = (C * 2) as u64;
        let (next, init, qkv) = (self.next_conv.device_ptr(), self.initial_conv.device_ptr(), self.projections[0].b.device_ptr());
        ([(next, init + row, 2 * row), (next + 2 * row, qkv, row)], (init, next, 3 * row))
    }
    pub(crate) fn finish_commit(&self) {
        self.valid.set(None);
        self.rows.set(0);
        self.has_state.set(true);
    }
    /// No state roundtrip: gather a prefix checkpoint and convolution tail, then
    /// restore them to the next invocation's inputs. The caller owns token commit.
    pub fn commit(&self, accepted: usize) -> Result<()> {
        let (rows, all) = self
            .valid
            .get()
            .ok_or_else(|| anyhow::anyhow!("GDN commit requires completed output"))?;
        ensure!(
            accepted <= rows && (all || accepted == 0 || accepted == rows),
            "GDN unavailable prefix checkpoint"
        );
        self.valid.set(None);
        poison(&self.next_conv)?;
        poison(&self.next_state)?;
        let c = unsafe {
            self.commit.prepare(
                vec![
                    Arg::Buffer(&self.projections[0].b),
                    Arg::Buffer(&self.initial_conv),
                    Arg::Buffer(&self.initial_state),
                    Arg::Buffer(&self.states),
                    Arg::Buffer(&self.next_conv),
                    Arg::Buffer(&self.next_state),
                    Arg::U32(accepted as u32),
                    Arg::U32(rows as u32),
                    Arg::U32(u32::from(all)),
                ],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let restore = unsafe {
            self.restore.prepare(
                vec![
                    Arg::Buffer(&self.next_conv),
                    Arg::Buffer(&self.next_state),
                    Arg::Buffer(&self.initial_conv),
                    Arg::Buffer(&self.initial_state),
                ],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        self.rows.set(0);
        self.has_state.set(false);
        c.enqueue()?;
        restore.enqueue()?;
        self.d.sync()?;
        self.has_state.set(true);
        Ok(())
    }
}
enum Executor<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Gdn<'m, 'd>,
    rows: usize,
    p: Policy,
    exec: Executor<'a, 'm, 'd>,
    repetitions: usize,
    pub calls: usize,
}
pub struct Run {
    pub projected: Vec<Vec<u16>>,
    pub projection_raw: Vec<Vec<f32>>,
    pub core: Core,
    pub output: Vec<u16>,
    pub output_raw: Vec<f32>,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        let t = self.rows;
        ensure!(
            a.rows.get() == t && a.has_state.get(),
            "GDN plan invalidated"
        );
        a.poison()?;
        a.d.sync()?;
        let start = Instant::now();
        match &self.exec {
            Executor::Graph(g) => g.launch()?,
            Executor::Prepared(c) => {
                for _ in 0..self.repetitions {
                    for k in c {
                        k.enqueue()?;
                    }
                }
            }
        }
        let enqueue_us = start.elapsed().as_secs_f64() * 1e6;
        let wait = Instant::now();
        a.d.sync()?;
        let wait_us = wait.elapsed().as_secs_f64() * 1e6;
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        let mut r = a.collect(t, self.p)?;
        r.total_us = total_us;
        r.enqueue_us = enqueue_us;
        r.wait_us = wait_us;
        Ok(r)
    }
}

/// R60 S4 (prefill v2): GDN intermediates for chunk-sized row counts, shared by every recurrent
/// stage layer.
pub struct GdnScratch<'d> {
    rows: usize,
    /// qkv [rows, C], z [rows, V], a [rows, HV], b [rows, HV], out [rows, H]: FP32 + BF16
    proj: Vec<(Buffer<'d>, Buffer<'d>)>,
    q: Buffer<'d>,
    k: Buffer<'d>,
    v: Buffer<'d>,
    gates: Buffer<'d>,
    y: Buffer<'d>,
    gated: Buffer<'d>,
    states: Buffer<'d>,
    conv_out: Buffer<'d>,
}
impl<'d> GdnScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        let mut proj = vec![];
        for n in [C, V, HV, HV, H] {
            proj.push((d.alloc(rows * n * 4 + GUARD)?, d.alloc(rows * n * 2 + GUARD)?));
        }
        Ok(Self {
            rows,
            proj,
            q: d.alloc(rows * HK * DK * 2 + GUARD)?,
            k: d.alloc(rows * HK * DK * 2 + GUARD)?,
            v: d.alloc(rows * V * 2 + GUARD)?,
            gates: d.alloc(rows * HV * 2 * 4 + GUARD)?,
            y: d.alloc(rows * V * 2 + GUARD)?,
            gated: d.alloc(rows * V * 2 + GUARD)?,
            states: d.alloc(STATE * 4 + GUARD)?,
            conv_out: d.alloc(3 * C * 2 + GUARD)?,
        })
    }
    /// The block output (BF16 [rows, H]) of the last recorded prefill.
    pub fn output(&self) -> &Buffer<'d> {
        &self.proj[4].1
    }
}
impl<'m, 'd> Gdn<'m, 'd> {
    fn args_prefill<'a>(&'a self, s: &'a GdnScratch<'d>, rows: usize) -> Vec<Arg<'a, 'd>> {
        vec![
            Arg::Buffer(&s.proj[0].1),
            Arg::Buffer(&s.proj[1].1),
            Arg::Buffer(&s.proj[2].1),
            Arg::Buffer(&s.proj[3].1),
            Arg::Buffer(&self.initial_conv),
            Arg::Buffer(&self.initial_state),
            Arg::Buffer(&self.weights[0]),
            Arg::Buffer(&self.weights[1]),
            Arg::Buffer(&self.weights[2]),
            Arg::Buffer(&self.weights[3]),
            Arg::Buffer(&s.q),
            Arg::Buffer(&s.k),
            Arg::Buffer(&s.v),
            Arg::Buffer(&s.gates),
            Arg::Buffer(&s.y),
            Arg::Buffer(&s.gated),
            Arg::Buffer(&s.states),
            Arg::Buffer(&s.conv_out),
            Arg::U32(rows as u32),
            Arg::U32(0),
        ]
    }
    /// R60 S4 (prefill v2): the GDN block for chunk rows: the four in-projections on the 64-token
    /// tensor tile, gdn_prepare / gdn_recur (sequential over the rows, final state only) /
    /// gdn_norm_gate, then out_proj. The output lands in `s.output()`. Nothing commits until
    /// `record_commit_prefill`.
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a GdnScratch<'d>,
        x: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && x.byte_len() >= rows * H * 2 && !self.zero_copy.get(), "GDN prefill shape");
        ensure!(s.proj.iter().all(|(f, b)| !x.aliases(f) && !x.aliases(b)), "GDN prefill alias");
        let m8 = self.multi8.as_ref().ok_or_else(|| anyhow::anyhow!("GDN prefill needs the canonical affine8 layer"))?;
        let mut calls = vec![crate::cuda_affine4::record_multi4_affine8_rows(
            m8,
            [&self.linears[0], &self.linears[1], &self.linears[2], &self.linears[3]],
            x,
            [(&s.proj[0].0, &s.proj[0].1), (&s.proj[1].0, &s.proj[1].1), (&s.proj[2].0, &s.proj[2].1), (&s.proj[3].0, &s.proj[3].1)],
            rows,
        )?];
        // SAFETY: the decode path's kernel ABIs (kernels/lowbit_gdn.cu); buffers sized for s.rows.
        calls.push(unsafe { self.prepare.prepare(self.args_prefill(s, rows), [HV as u32, rows as u32, 1], [128, 1, 1])? });
        calls.push(unsafe { self.recur.prepare(self.args_prefill(s, rows), [(HV * 4) as u32, 1, 1], [128, 1, 1])? });
        calls.push(unsafe { self.norm.prepare(self.args_prefill(s, rows), [HV as u32, rows as u32, 1], [32, 1, 1])? });
        calls.push(self.linears[4].record_rows(&s.gated, &s.proj[4].0, &s.proj[4].1, rows)?);
        Ok(calls)
    }
    /// R60 S4 (prefill v2): commit every row of the last prefill (final state, last three qkv rows
    /// as the convolution window).
    pub(crate) fn record_commit_prefill<'a>(&'a self, s: &'a GdnScratch<'d>, rows: usize) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && !self.zero_copy.get(), "GDN prefill commit");
        // SAFETY: kernels/lowbit_gdn.cu gdn_commit / gdn_restore ABIs.
        let c = unsafe {
            self.commit.prepare(
                vec![
                    Arg::Buffer(&s.proj[0].1), Arg::Buffer(&self.initial_conv), Arg::Buffer(&self.initial_state), Arg::Buffer(&s.states),
                    Arg::Buffer(&self.next_conv), Arg::Buffer(&self.next_state), Arg::U32(rows as u32), Arg::U32(rows as u32), Arg::U32(0),
                ],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        let r = unsafe {
            self.restore.prepare(
                vec![Arg::Buffer(&self.next_conv), Arg::Buffer(&self.next_state), Arg::Buffer(&self.initial_conv), Arg::Buffer(&self.initial_state)],
                [STATE.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        };
        Ok(vec![c, r])
    }
}
