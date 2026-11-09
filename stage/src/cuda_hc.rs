//! Native CUDA lowbit hyper-connections, with resident intermediates.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_affine4::{Linear, Packed},
    lowbit_hc::{F, H, HC, MAX_ROWS, R, Read, Weights},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "fc20fb32cf26eec16ad89b9be0244ca298f1ec8c21550e57024563a417022b11";
const GUARD: usize = 256;
/// R38 (Opus, direct): unchanged lowbit_hc.cu plus the two-launch HC site
/// (hc_r38.cu) in one module. Loaded only with QWEN_AFFINE4_R38=1.
pub const R38_SHA: &str = "2eb88e8be9f9fb368d0577a5797f43b032fb0a760ab0ae7c243c5817f19ba455";
pub fn r39_hc_enabled() -> bool { crate::cuda_affine4::r39_enabled() && std::env::var("QWEN_R39_HC_PDL").as_deref()==Ok("1") }
pub static FAST_SIGMOID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn r42_enabled() -> bool { std::env::var("QWEN_R42_HC_FAST").as_deref()==Ok("1") }
pub fn r43_enabled() -> bool { std::env::var("QWEN_R43_HC_SPLIT").as_deref()==Ok("1") }
pub static SPLIT_INPUT_STREAMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// R52: pump the next launch's constant weights into L2 from this site's front
/// launch (candidate arm only). Loads the R39 HC module plus r39pdl_hc_front2p.
pub fn r52_pump_enabled() -> bool { r39_hc_enabled() && std::env::var("QWEN_R52_PUMP").as_deref()==Ok("1") }
/// R52: norm+down that normalizes once per block (candidate arm; same outputs).
pub fn r52_hcn_enabled() -> bool { r39_hc_enabled() && std::env::var("QWEN_R52_HCN").as_deref()==Ok("1") }
/// R52: front launch with eight coordinates per block (320 blocks; same outputs).
pub fn r52_hcf8_enabled() -> bool { r39_hc_enabled() && std::env::var("QWEN_R52_HCF8").as_deref()==Ok("1") }
/// R53: the per-layer stream write after the MoE as an early-launch clone
/// (waits at entry; same body). Candidate arm only. Loads lowbit_hc_r53.
pub fn r53_hcw_enabled() -> bool { r39_hc_enabled() && std::env::var("QWEN_R53_HCW").as_deref()==Ok("1") }
/// R53: norm+down whose injection blocks request their constant weights before
/// the wait (candidate arm; same outputs). Loads lowbit_hc_r53.
pub fn r53_hcn5_enabled() -> bool { r39_hc_enabled() && std::env::var("QWEN_R53_HCN5").as_deref()==Ok("1") }
/// R53: r39pdl_hc_front4, four coordinates per block (640 blocks, under the
/// 1,000-block early-launch boundary step); same values as front2. Negative:
/// +195 us/step (=1) and +17 us (=trig); front2's size gates the early-launch cascade.
pub fn r53_hcf4_enabled() -> bool { r39_hc_enabled() && matches!(std::env::var("QWEN_R53_HCF4").as_deref(), Ok("1") | Ok("trig")) }
/// R53: front4 that triggers its successor after its wait (QWEN_R53_HCF4=trig; needs
/// QWEN_R53_TRIG=1, driver bit 128), holding the early-launch cascade as front2 does.
pub fn r53_hcf4_trig() -> bool { r39_hc_enabled() && std::env::var("QWEN_R53_HCF4").as_deref()==Ok("trig") }
pub fn module(d: &Device) -> Result<Module<'_>> {
    if r53_hcw_enabled() || r53_hcn5_enabled() || r53_hcf4_enabled() {
        let pin: Value=serde_json::from_str(include_str!("../pins/R53-HC-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r53.sm120.cubin"),pin["cubin_sha256"].as_str().unwrap());
    }
    if r52_pump_enabled() || r52_hcn_enabled() || r52_hcf8_enabled() {
        let pin: Value=serde_json::from_str(include_str!("../pins/R52-HC-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r52.sm120.cubin"),pin["cubin_sha256"].as_str().unwrap());
    }
    if r43_enabled() {
        ensure!(crate::cuda_affine4::r38_enabled(),"R43 requires R38");
        let pin: Value=serde_json::from_str(include_str!("../pins/R43-HC-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r43.sm120.cubin"),pin["cubin_sha256"].as_str().unwrap());
    }
    if r42_enabled() {
        ensure!(crate::cuda_affine4::r38_enabled(),"R42 requires R38");
        let pin: Value=serde_json::from_str(include_str!("../pins/R42-HC-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r42.sm120.cubin"),pin["cubin_sha256"].as_str().unwrap());
    }
    if r39_hc_enabled() {
        let pin: Value=serde_json::from_str(include_str!("../pins/R39-HC-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r39.sm120.cubin"),pin["cubin_sha256"].as_str().unwrap());
    }
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/lowbit_hc.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn poison(b: &Buffer<'_>) -> Result<()> {
    b.upload(&vec![0xff; b.byte_len()])
}
fn read_raw(b: &Buffer<'_>, count: usize) -> Result<Vec<u8>> {
    ensure!(count + GUARD <= b.byte_len(), "HC read bound");
    let mut raw = vec![0; b.byte_len()];
    b.download(&mut raw)?;
    ensure!(raw[count..].iter().all(|&v| v == 0xff), "HC tail canary");
    raw.truncate(count);
    Ok(raw)
}
fn read_bf(b: &Buffer<'_>, n: usize) -> Result<Vec<u16>> {
    let v: Vec<_> = read_raw(b, n * 2)?
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        v.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "HC BF16 finite/fully written"
    );
    Ok(v)
}
fn read_f32(b: &Buffer<'_>, n: usize) -> Result<Vec<f32>> {
    let v: Vec<_> = read_raw(b, n * 4)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        v.iter().all(|v| v.is_finite()),
        "HC FP32 finite/fully written"
    );
    Ok(v)
}
struct Pair<'d> {
    fp32: Buffer<'d>,
    bf16: Buffer<'d>,
}
impl<'d> Pair<'d> {
    fn new(d: &'d Device, n: usize) -> Result<Self> {
        Ok(Self {
            fp32: d.alloc(n * 4 + GUARD)?,
            bf16: d.alloc(n * 2 + GUARD)?,
        })
    }
    fn poison(&self) -> Result<()> {
        poison(&self.fp32)?;
        poison(&self.bf16)
    }
    fn read(&self, n: usize) -> Result<(Vec<f32>, Vec<u16>)> {
        let f = read_f32(&self.fp32, n)?;
        let b = read_bf(&self.bf16, n)?;
        ensure!(
            f.iter()
                .zip(&b)
                .all(|(&f, &b)| bf16::from_f32(f).to_bits() == b),
            "HC projection rounding"
        );
        Ok((f, b))
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub split_down: bool,
    pub separate_write: bool,
}
/// Experimental block-local projection/gate/mix fusion. Ordinary HC unchanged.
pub struct UpMix<'m, 'd> {
    kernels: [Kernel<'m, 'd>; 2],
    reduce: Option<Kernel<'m, 'd>>,
    replicated: Option<Kernel<'m, 'd>>,
}
impl<'m, 'd> UpMix<'m, 'd> {
    pub fn new(m: &'m Module<'d>) -> Result<Self> {
        Ok(Self {
            kernels: [m.kernel("hc_upmix1")?, m.kernel("hc_upmix4")?],
            reduce: None,
            replicated: None,
        })
    }
    pub fn with_reduce(m: &'m Module<'d>, fold: &'m Module<'d>, replicated: bool) -> Result<Self> {
        let mut this = Self::new(m)?;
        this.reduce = Some(fold.kernel("hc_reduce_activate")?);
        if replicated {
            this.replicated = Some(fold.kernel("hc_replicated_front")?);
        }
        Ok(this)
    }
    pub fn description(&self) -> Value {
        json!({"kernels":self.kernels.iter().map(|k|k.description()).collect::<Vec<_>>(),
            "reduce_activate":self.reduce.as_ref().map(|k|k.description()),
            "replicated_front":self.replicated.as_ref().map(|k|k.description()),
            "scope":"four stream warps share producer results inside one block; no inter-block waits"})
    }
}
pub struct Hc<'m, 'd> {
    d: &'d Device,
    down: Linear<'m, 'd>,
    up: Packed<'m, 'd>,
    norm: Kernel<'m, 'd>,
    activate: Kernel<'m, 'd>,
    mix: Kernel<'m, 'd>,
    write: Kernel<'m, 'd>,
    r53_write: Option<Kernel<'m, 'd>>,
    /// R53 g4w2h: the stream write that also clears the MoE counter block after its wait.
    r53_write_z: Option<Kernel<'m, 'd>>,
    r53nd5: Option<Kernel<'m, 'd>>,
    r38: Option<[Kernel<'m, 'd>; 5]>,
    /// R60: the canonical (affine8) HC site in two launches (kernels/affine8.cu hc8_*)
    hc8: Option<[Kernel<'m, 'd>; 2]>,
    /// R60 S4: the multi-row forms (hc8r_*: one block per column block for all rows, weights
    /// loaded once; bit-identical). Opt-in (QD_HC8_ROWS=1): slower than the per-row launches at
    /// every row count measured (4 rows: 680 + 462 vs 429 + 392 us per step; latency-bound, the
    /// row loop serializes what per-row blocks spread across SMs).
    hc8r: Option<[Kernel<'m, 'd>; 2]>,
    r39: Option<[Kernel<'m, 'd>; 2]>,
    r42: Option<[Kernel<'m, 'd>; 2]>,
    r43: Option<[Kernel<'m, 'd>; 4]>,
    r52p: Option<Kernel<'m, 'd>>,
    r52nd: Option<Kernel<'m, 'd>>,
    r52f8: Option<Kernel<'m, 'd>>,
    r53f4: Option<Kernel<'m, 'd>>,
    pump: Cell<Option<(u64, u32)>>,
    dpart: Buffer<'d>,
    nw: Buffer<'d>,
    iw: Buffer<'d>,
    x: Buffer<'d>,
    pending_block: Buffer<'d>,
    pending_inject: Buffer<'d>,
    normalized: Buffer<'d>,
    parts: Buffer<'d>,
    lower: Pair<'d>,
    activated: Buffer<'d>,
    upper: Pair<'d>,
    mixed: Buffer<'d>,
    inject: Buffer<'d>,
    next: Buffer<'d>,
    rows: Cell<usize>,
    pending: Cell<bool>,
}
impl<'m, 'd> Hc<'m, 'd> {
    pub fn new(
        d: &'d Device,
        affine: &'m Module<'d>,
        split: &'m Module<'d>,
        ops: &'m Module<'d>,
        w: &Weights,
    ) -> Result<Self> {
        w.validate()?;
        Self::assemble(d, ops, Linear::new_split(d, affine, split, &w.down)?, Packed::new(d, affine, &w.up)?, &w.norm, &w.inject, true)
    }
    /// R60 pipeline split: the canonical target's HC (affine8 down/up from `m8`, the affine8
    /// module). Baseline kernels only: no split-K down projection, no R38-R53 fused variants;
    /// record with Policy { split_down: false, .. }.
    pub fn new_canonical(d: &'d Device, m8: &'m Module<'d>, ops: &'m Module<'d>, w: &crate::hc::Weights) -> Result<Self> {
        w.validate()?;
        let mut hc = Self::assemble(d, ops, Linear::new_affine8_split(d, m8, &w.down)?, Packed::new_affine8(d, m8, &w.up)?, &w.norm, &w.inject, false)?;
        hc.hc8 = Some([m8.kernel("hc8_norm_down3")?, m8.kernel("hc8_front2")?]);
        if std::env::var("QD_HC8_ROWS").as_deref() == Ok("1") {
            hc.hc8r = Some([m8.kernel("hc8r_norm_down3")?, m8.kernel("hc8r_front2")?]);
        }
        Ok(hc)
    }
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        d: &'d Device,
        ops: &'m Module<'d>,
        down: Linear<'m, 'd>,
        up: Packed<'m, 'd>,
        norm: &[u16],
        inject: &[u16],
        fast: bool,
    ) -> Result<Self> {
        let upload = |v: &[u16]| -> Result<Buffer<'d>> {
            let b = d.alloc(v.len() * 2)?;
            b.upload(bytes(v))?;
            let mut check = vec![0; v.len() * 2];
            b.download(&mut check)?;
            ensure!(check == bytes(v), "HC side-weight readback");
            Ok(b)
        };
        Ok(Self {
            d,
            down,
            // This composite already owns activation and output buffers. Keep
            // only packed weights here instead of another standalone MAX64 set.
            up,
            norm: ops.kernel("hc_norm")?,
            activate: ops.kernel("hc_activate")?,
            mix: ops.kernel("hc_mix")?,
            write: ops.kernel("hc_write")?,
            r53_write: if fast && r53_hcw_enabled() { Some(ops.kernel("r39pdl_hc_write")?) } else { None },
            r53_write_z: if fast && r53_hcw_enabled() && std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2h") {
                Some(ops.kernel("r39pdl_hc_write_z")?)
            } else { None },
            r53nd5: if fast && r53_hcn5_enabled() { Some(ops.kernel("r39pdl_hc_norm_down5")?) } else { None },
            r39: if fast && r39_hc_enabled() { Some([ops.kernel("r39pdl_hc_norm_down3")?,ops.kernel("r39pdl_hc_front2")?]) } else { None },
            r42: if fast && r42_enabled() {Some([ops.kernel("hc_fast_front2")?,ops.kernel("r39pdl_hc_fast_front2")?])} else {None},
            r52p: if fast && r52_pump_enabled() { Some(ops.kernel("r39pdl_hc_front2p")?) } else { None },
            r52nd: if fast && r52_hcn_enabled() { Some(ops.kernel("r39pdl_hc_norm_down4")?) } else { None },
            r52f8: if fast && r52_hcf8_enabled() { Some(ops.kernel("r39pdl_hc_front8")?) } else { None },
            r53f4: if fast && r53_hcf4_enabled() {
                Some(ops.kernel(if r53_hcf4_trig() { "r39pdl_hc_front4_trig" } else { "r39pdl_hc_front4" })?)
            } else { None },
            pump: Cell::new(None),
            hc8: None,
            hc8r: None,
            r43: if fast && r43_enabled() {Some([ops.kernel("hc_splitpending_norm_down3")?,ops.kernel("r39pdl_hc_splitpending_norm_down3")?,
                ops.kernel("hc_splitpending_norm_down3_streamed")?,ops.kernel("r39pdl_hc_splitpending_norm_down3_streamed")?])} else {None},
            r38: if fast && crate::cuda_affine4::r38_enabled() {
                Some([
                    ops.kernel("hc_r38_norm_down")?,
                    ops.kernel("hc_r38_front")?,
                    ops.kernel("hc_r38_norm_down2")?,
                    ops.kernel("hc_r38_front2")?,
                    ops.kernel("hc_r38_norm_down3")?,
                ])
            } else {
                None
            },
            nw: upload(norm)?,
            iw: upload(inject)?,
            x: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            pending_block: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            pending_inject: d.alloc(MAX_ROWS * HC * 2 + GUARD)?,
            normalized: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            parts: d.alloc(MAX_ROWS * HC * HC * 4 + GUARD)?,
            dpart: d.alloc(MAX_ROWS * HC * 8 * HC * 4 + GUARD)?,
            lower: Pair::new(d, MAX_ROWS * R)?,
            activated: d.alloc(MAX_ROWS * R * 2 + GUARD)?,
            upper: Pair::new(d, MAX_ROWS * F)?,
            mixed: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            inject: d.alloc(MAX_ROWS * HC * 2 + GUARD)?,
            next: d.alloc(MAX_ROWS * F * 2 + GUARD)?,
            rows: Cell::new(0),
            pending: Cell::new(false),
        })
    }
    pub fn description(&self) -> Value {
        let bytes: usize = [
            &self.nw,
            &self.iw,
            &self.x,
            &self.pending_block,
            &self.pending_inject,
            &self.normalized,
            &self.parts,
            &self.lower.fp32,
            &self.lower.bf16,
            &self.activated,
            &self.upper.fp32,
            &self.upper.bf16,
            &self.mixed,
            &self.inject,
            &self.next,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum();
        let pin=if self.r43.is_some() {Some(include_str!("../pins/R43-HC-CUDA-BUILD.json"))}
            else if self.r42.is_some() {Some(include_str!("../pins/R42-HC-CUDA-BUILD.json"))}
            else if self.r39.is_some() {Some(include_str!("../pins/R39-HC-CUDA-BUILD.json"))} else {None};
        let cubin=pin.map(|p|serde_json::from_str::<Value>(p).expect("embedded build pin")["cubin_sha256"].clone())
            .unwrap_or_else(||json!(if self.r38.is_some(){R38_SHA}else{SHA}));
        json!({"backend":"native CUDA/libtinynv","cubin_sha256":cubin,"hidden":H,"streams":HC,"max_rows":MAX_ROWS,
            "extra_bytes":bytes,"down":self.down.description(),"up":self.up.description(),"host_intermediates":0,
            "kernels":[self.norm.description(),self.activate.description(),self.mix.description(),self.write.description()],
            "r38_kernels":self.r38.as_ref().map(|a|a.iter().map(|k|k.description()).collect::<Vec<_>>()),
            "r39_kernels":self.r39.as_ref().map(|a|a.iter().map(|k|k.description()).collect::<Vec<_>>()),
            "r42_kernels":self.r42.as_ref().map(|a|a.iter().map(|k|k.description()).collect::<Vec<_>>()),
            "r43_kernels":self.r43.as_ref().map(|a|a.iter().map(|k|k.description()).collect::<Vec<_>>()),
            "arithmetic":"BF16 semantic boundaries, FP32 reductions; canonical packed affine4; optional fused pending residual update"})
    }
    pub fn prepare(&self, x: &[u16], pending: Option<(&[u16], &[u16])>) -> Result<()> {
        self.rows.set(0);
        ensure!(
            !x.is_empty()
                && x.len() % F == 0
                && x.len() / F <= MAX_ROWS
                && x.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "HC input shape/finite"
        );
        let rows = x.len() / F;
        if let Some((b, i)) = pending {
            ensure!(
                b.len() == rows * H
                    && i.len() == rows * HC
                    && b.iter().chain(i).all(|&v| bf16::from_bits(v).is_finite()),
                "HC pending shape/finite"
            );
        }
        for b in [&self.x, &self.pending_block, &self.pending_inject] {
            poison(b)?;
        }
        self.x.upload(bytes(x))?;
        if let Some((b, i)) = pending {
            self.pending_block.upload(bytes(b))?;
            self.pending_inject.upload(bytes(i))?;
        }
        ensure!(read_bf(&self.x, x.len())? == x, "HC input readback");
        if let Some((b, i)) = pending {
            ensure!(
                read_bf(&self.pending_block, b.len())? == b
                    && read_bf(&self.pending_inject, i.len())? == i,
                "HC pending readback"
            );
        }
        self.pending.set(pending.is_some());
        self.rows.set(rows);
        Ok(())
    }
    pub fn mixed(&self) -> &Buffer<'d> {
        &self.mixed
    }
    pub fn injection(&self) -> &Buffer<'d> {
        &self.inject
    }
    pub fn updated_stream(&self) -> &Buffer<'d> {
        &self.next
    }
    /// Records into resident scratch/output buffers. Used later by layer composition.
    pub fn record_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        pending: Option<(&'a Buffer<'d>, &'a Buffer<'d>)>,
        rows: usize,
        p: Policy,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        self.record_from_impl(x,pending,rows,p,false)
    }
    /// R43: consume four FP32 projection partials and preserve the BF16 reduce
    /// boundary locally. Single-GPU split-K; no model parallelism is involved.
    pub fn record_from_split4<'a>(
        &'a self, x:&'a Buffer<'d>, partials:&'a Buffer<'d>, inject:&'a Buffer<'d>,
        rows:usize, p:Policy,
    ) -> Result<Vec<PreparedCall<'a,'m,'d>>> {
        ensure!(self.r43.is_some() && crate::cuda_affine4::ahead_selected()
            && p.split_down && !p.separate_write && rows<8 && rows>0,
            "R43 requires qualified two-launch HC, rows1..7, resident partials");
        ensure!(partials.byte_len()>=rows*4*H*4,"R43 partial buffer length");
        self.record_from_impl(x,Some((partials,inject)),rows,p,true)
    }
    fn record_from_impl<'a>(
        &'a self,x:&'a Buffer<'d>,pending:Option<(&'a Buffer<'d>,&'a Buffer<'d>)>,
        rows:usize,p:Policy,split_pending:bool,
    ) -> Result<Vec<PreparedCall<'a,'m,'d>>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && x.byte_len() >= rows * F * 2,
            "HC record input"
        );
        let outputs = [
            &self.normalized,
            &self.parts,
            &self.lower.fp32,
            &self.lower.bf16,
            &self.activated,
            &self.upper.fp32,
            &self.upper.bf16,
            &self.mixed,
            &self.inject,
            &self.next,
        ];
        ensure!(
            outputs.iter().all(|b| !x.aliases(b)),
            "HC input/output alias"
        );
        if let Some((b, i)) = pending {
            ensure!(
                b.byte_len() >= rows * H * 2
                    && i.byte_len() >= rows * HC * 2
                    && outputs.iter().all(|o| !b.aliases(o) && !i.aliases(o)),
                "HC pending buffers/alias"
            );
        }
        let (b, i) = pending.unwrap_or((&self.pending_block, &self.pending_inject));
        // R60: canonical fused site (norm + split-4 down partials, then replicated front + up + mix);
        // the pending residual write folds into the norm launch.
        if let (Some([nd, fr]), true) = (&self.hc8, p.split_down && rows < 8 && !split_pending) {
            let (dw, ds, db, dgroup) = self.down.packed_weights();
            let partials = self.down.split_partials()?;
            if let Some([ndr, frr]) = &self.hc8r {
                // SAFETY: hc8_* ABIs plus a trailing row count; grids without the row dimension.
                return Ok(vec![
                    unsafe {
                        ndr.prepare(
                            vec![
                                Arg::Buffer(x), Arg::Buffer(&self.nw), Arg::Buffer(&self.iw), Arg::Buffer(b), Arg::Buffer(i),
                                Arg::Buffer(dw), Arg::Buffer(ds), Arg::Buffer(db), Arg::Buffer(&self.normalized),
                                Arg::Buffer(&self.dpart), Arg::Buffer(&self.next), Arg::Buffer(partials),
                                Arg::U32(dgroup as u32), Arg::U32(u32::from(pending.is_some())), Arg::U32(rows as u32),
                            ],
                            [(R / 4 + 8) as u32, 1, HC as u32],
                            [64, 1, 1],
                        )?
                    },
                    unsafe {
                        frr.prepare(
                            vec![
                                Arg::Buffer(partials), Arg::Buffer(&self.dpart), Arg::Buffer(&self.parts),
                                Arg::Buffer(&self.up.w), Arg::Buffer(&self.up.s), Arg::Buffer(&self.up.b),
                                Arg::Buffer(&self.normalized), Arg::Buffer(&self.lower.fp32), Arg::Buffer(&self.lower.bf16),
                                Arg::Buffer(&self.activated), Arg::Buffer(&self.inject), Arg::Buffer(&self.upper.fp32),
                                Arg::Buffer(&self.upper.bf16), Arg::Buffer(&self.mixed), Arg::U32(self.up.group_size() as u32),
                                Arg::U32(rows as u32),
                            ],
                            [(H / 2) as u32, 1, 1],
                            [128, 1, 1],
                        )?
                    },
                ]);
            }
            // SAFETY: fixed kernel ABIs (as hc_r38_norm_down3/front2); buffers sized at construction.
            return Ok(vec![
                unsafe {
                    nd.prepare(
                        vec![
                            Arg::Buffer(x), Arg::Buffer(&self.nw), Arg::Buffer(&self.iw), Arg::Buffer(b), Arg::Buffer(i),
                            Arg::Buffer(dw), Arg::Buffer(ds), Arg::Buffer(db), Arg::Buffer(&self.normalized),
                            Arg::Buffer(&self.dpart), Arg::Buffer(&self.next), Arg::Buffer(partials),
                            Arg::U32(dgroup as u32), Arg::U32(u32::from(pending.is_some())),
                        ],
                        [(R / 4 + 8) as u32, rows as u32, HC as u32],
                        [64, 1, 1],
                    )?
                },
                unsafe {
                    fr.prepare(
                        vec![
                            Arg::Buffer(partials), Arg::Buffer(&self.dpart), Arg::Buffer(&self.parts),
                            Arg::Buffer(&self.up.w), Arg::Buffer(&self.up.s), Arg::Buffer(&self.up.b),
                            Arg::Buffer(&self.normalized), Arg::Buffer(&self.lower.fp32), Arg::Buffer(&self.lower.bf16),
                            Arg::Buffer(&self.activated), Arg::Buffer(&self.inject), Arg::Buffer(&self.upper.fp32),
                            Arg::Buffer(&self.upper.bf16), Arg::Buffer(&self.mixed), Arg::U32(self.up.group_size() as u32),
                        ],
                        [(H / 2) as u32, rows as u32, 1],
                        [128, 1, 1],
                    )?
                },
            ]);
        }
        if let Some([norm_down1, front1, norm_down2, front2, norm_down3]) = &self.r38 {
            if crate::cuda_affine4::ahead_selected() && p.split_down && rows < 8 {
                // QWEN_R38_HC_V2: 3 (default) = 8 norm blocks per stream, one
                // wave, all weight loads first; 2 = the same without the early
                // loads; 0 = v1 (one norm block per stream).
                let ver = match std::env::var("QWEN_R38_HC_V2").as_deref() {
                    Ok("0") => 1,
                    Ok("2") => 2,
                    _ => 3,
                };
                let v2 = ver >= 2;
                ensure!(!split_pending || ver==3,"R43 requires HC v3");
                let (norm_down, front) = match ver {
                    1 => (norm_down1, front1),
                    2 => (norm_down2, front2),
                    _ => (norm_down3, front2),
                };
                let pdl_hc=(crate::cuda_affine4::pdl_selected()
                    || std::env::var("QWEN_R39_HC_ALWAYS").as_deref() == Ok("1")) && ver==3 && self.r39.is_some();
                let (norm_down,front) = if pdl_hc {
                    if let Some([a,b])=&self.r39 { (a,b) } else { (norm_down,front) }
                } else { (norm_down,front) };
                let front=if FAST_SIGMOID.load(std::sync::atomic::Ordering::Relaxed) && ver==3 {
                    &self.r42.as_ref().ok_or_else(||anyhow::anyhow!("R42 fast kernel not loaded"))?[usize::from(pdl_hc)]
                } else {front};
                // R52 pump (candidate arm): same front launch plus asynchronous
                // L2 fills of the next launch's weights.
                let pumping = crate::cuda_affine4::pdl_selected() && pdl_hc && self.r52p.is_some() && self.pump.get().is_some();
                let norm_down = match (&self.r52nd, &self.r53nd5) {
                    (Some(k), _) if crate::cuda_affine4::pdl_selected() && pdl_hc && !split_pending => k,
                    (_, Some(k)) if crate::cuda_affine4::pdl_selected() && pdl_hc && !split_pending => k,
                    _ => norm_down,
                };
                let front = if pumping { self.r52p.as_ref().unwrap() } else { front };
                let front8 = !pumping && crate::cuda_affine4::pdl_selected() && pdl_hc && v2 && self.r52f8.is_some();
                let front = if front8 { self.r52f8.as_ref().unwrap() } else { front };
                let front4 = !pumping && !front8 && crate::cuda_affine4::pdl_selected() && pdl_hc && v2 && self.r53f4.is_some();
                let front = if front4 { self.r53f4.as_ref().unwrap() } else { front };
                let norm_down=if split_pending {
                    &self.r43.as_ref().ok_or_else(||anyhow::anyhow!("R43 kernel missing"))?[
                        usize::from(pdl_hc)+2*usize::from(SPLIT_INPUT_STREAMED.load(std::sync::atomic::Ordering::Relaxed))]
                } else {norm_down};
                // R38 two-launch site: norm (+pending write) folded into the
                // split4 down partials; reduce/activate/up/mix in one launch.
                let (dw, ds, db, dgroup) = self.down.packed_weights();
                let partials = self.down.split_partials()?;
                return Ok(vec![
                    unsafe {
                        norm_down.prepare(
                            vec![
                                Arg::Buffer(x),
                                Arg::Buffer(&self.nw),
                                Arg::Buffer(&self.iw),
                                Arg::Buffer(b),
                                Arg::Buffer(i),
                                Arg::Buffer(dw),
                                Arg::Buffer(ds),
                                Arg::Buffer(db),
                                Arg::Buffer(&self.normalized),
                                Arg::Buffer(if v2 { &self.dpart } else { &self.parts }),
                                Arg::Buffer(&self.next),
                                Arg::Buffer(partials),
                                Arg::U32(dgroup as u32),
                                Arg::U32(u32::from(pending.is_some())),
                            ],
                            [(R / 4 + if v2 { 8 } else { 1 }) as u32, rows as u32, HC as u32],
                            [64, 1, 1],
                        )?
                    },
                    unsafe {
                        front.prepare(
                            {
                                let mut a = vec![Arg::Buffer(partials)];
                                if v2 {
                                    a.push(Arg::Buffer(&self.dpart));
                                }
                                a.extend([
                                Arg::Buffer(&self.parts),
                                Arg::Buffer(&self.up.w),
                                Arg::Buffer(&self.up.s),
                                Arg::Buffer(&self.up.b),
                                Arg::Buffer(&self.normalized),
                                Arg::Buffer(&self.lower.fp32),
                                Arg::Buffer(&self.lower.bf16),
                                Arg::Buffer(&self.activated),
                                Arg::Buffer(&self.inject),
                                Arg::Buffer(&self.upper.fp32),
                                Arg::Buffer(&self.upper.bf16),
                                Arg::Buffer(&self.mixed),
                                Arg::U32(self.up.group_size() as u32),
                                ]);
                                if pumping {
                                    let (addr, n16) = self.pump.get().unwrap();
                                    a.push(Arg::U64(addr));
                                    a.push(Arg::U32(n16));
                                }
                                a
                            },
                            [(H / if front8 { 8 } else if front4 { 4 } else { 2 }) as u32, rows as u32, 1],
                            [128, 1, 1],
                        )?
                    },
                ]);
            }
        }
        ensure!(!split_pending,"R43 cannot fall back to a BF16-input HC path");
        let separated = pending.is_some() && p.separate_write;
        let mut calls = Vec::new();
        if separated {
            calls.push(unsafe {
                self.write.prepare(
                    vec![
                        Arg::Buffer(x),
                        Arg::Buffer(b),
                        Arg::Buffer(i),
                        Arg::Buffer(&self.next),
                        Arg::U32(rows as u32),
                    ],
                    [(rows * F).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                )?
            });
        }
        calls.push(unsafe {
            self.norm.prepare(
                vec![
                    Arg::Buffer(if separated { &self.next } else { x }),
                    Arg::Buffer(&self.nw),
                    Arg::Buffer(&self.iw),
                    Arg::Buffer(b),
                    Arg::Buffer(i),
                    Arg::Buffer(&self.normalized),
                    Arg::Buffer(&self.parts),
                    Arg::Buffer(&self.next),
                    Arg::U32(u32::from(pending.is_some() && !separated)),
                ],
                [HC as u32, rows as u32, 1],
                [256, 1, 1],
            )?
        });
        if p.split_down && self.down.has_split() {
            calls.extend(self.down.record_split_into(
                &self.normalized,
                &self.lower.fp32,
                &self.lower.bf16,
                rows,
                if rows < 8 { 1 } else { 4 },
                4,
            )?);
        } else {
            calls.push(self.down.record_into(
                &self.normalized,
                &self.lower.fp32,
                &self.lower.bf16,
                rows,
                if rows < 8 { 0 } else { 3 },
            )?);
        }
        calls.push(unsafe {
            self.activate.prepare(
                vec![
                    Arg::Buffer(&self.lower.bf16),
                    Arg::Buffer(&self.parts),
                    Arg::Buffer(&self.activated),
                    Arg::Buffer(&self.inject),
                    Arg::U32(rows as u32),
                ],
                [(rows * (R + HC)).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        });
        calls.push(self.up.record_into(
            &self.activated,
            &self.upper.fp32,
            &self.upper.bf16,
            rows,
            if rows < 8 { 0 } else { 3 },
        )?);
        calls.push(unsafe {
            self.mix.prepare(
                vec![
                    Arg::Buffer(&self.upper.bf16),
                    Arg::Buffer(&self.normalized),
                    Arg::Buffer(&self.mixed),
                    Arg::U32(rows as u32),
                ],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        });
        Ok(calls)
    }
    pub(crate) fn poison(&self, p: Policy) -> Result<()> {
        for b in [
            &self.normalized,
            &self.parts,
            &self.activated,
            &self.mixed,
            &self.inject,
            &self.next,
        ] {
            poison(b)?;
        }
        self.lower.poison()?;
        self.upper.poison()?;
        if p.split_down && self.down.has_split() {
            self.down.poison_split()?;
        }
        Ok(())
    }
    /// Component-probe hooks; never part of the timed inference path.
    pub fn poison_diagnostic(&self,p:Policy)->Result<()> { self.poison(p) }
    pub fn collect_diagnostic(&self,x:&Buffer<'d>,rows:usize,pending:bool,p:Policy)->Result<Run> {
        ensure!((1..=MAX_ROWS).contains(&rows) && x.byte_len()>=rows*F*2,"HC diagnostic shape");
        self.collect_from(x,rows,pending,p)
    }
    pub(crate) fn collect_from(
        &self,
        x: &Buffer<'d>,
        rows: usize,
        pending: bool,
        p: Policy,
    ) -> Result<Run> {
        let normalized = read_bf(&self.normalized, rows * F)?;
        let (down_raw, down) = self.lower.read(rows * R)?;
        let activated = read_bf(&self.activated, rows * R)?;
        let (up_raw, up) = self.upper.read(rows * F)?;
        let mixed = read_bf(&self.mixed, rows * H)?;
        let inject = read_bf(&self.inject, rows * HC)?;
        let stream = if pending {
            read_bf(&self.next, rows * F)?
        } else {
            read_raw(&self.next, 0)?;
            read_bf(x, rows * F)?
        };
        let injection_partials = read_f32(&self.parts, rows * HC * HC)?;
        let partial_values_checked = if p.split_down && self.down.has_split() {
            self.down.check_split(rows, 4)?
        } else {
            0
        };
        Ok(Run {
            stages: Read {
                normalized,
                down,
                activated,
                up,
                mixed,
                inject,
            },
            stream,
            down_raw,
            up_raw,
            injection_partials,
            partial_values_checked,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn record_write_from<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        block: &'a Buffer<'d>,
        inject: &'a Buffer<'d>,
        out: &'a Buffer<'d>,
        rows: usize,
        reset: Option<&'a Buffer<'d>>,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && x.byte_len() >= rows * F * 2
                && block.byte_len() >= rows * H * 2
                && inject.byte_len() >= rows * HC * 2
                && out.byte_len() >= rows * F * 2,
            "HC write shapes"
        );
        ensure!(
            !out.aliases(x) && !out.aliases(block) && !out.aliases(inject),
            "HC write alias"
        );
        // R53 g4w2h: the MoE before this write leaves its counters for this launch to clear.
        if let Some(z) = &self.r53_write_z {
            if crate::cuda_affine4::pdl_selected() {
                let c = reset.ok_or_else(|| anyhow::anyhow!("g4w2h stream write without the MoE counter block"))?;
                ensure!(c.byte_len() >= 4096, "MoE counter block size");
                return unsafe {
                    z.prepare(
                        vec![Arg::Buffer(x), Arg::Buffer(block), Arg::Buffer(inject), Arg::Buffer(out), Arg::U32(rows as u32), Arg::Buffer(c)],
                        [(rows * F).div_ceil(256) as u32, 1, 1],
                        [256, 1, 1],
                    )
                };
            }
        }
        unsafe {
            (match &self.r53_write { Some(w) if crate::cuda_affine4::pdl_selected() => w, _ => &self.write }).prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(block),
                    Arg::Buffer(inject),
                    Arg::Buffer(out),
                    Arg::U32(rows as u32),
                ],
                [(rows * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )
        }
    }
    pub fn plan(&self, p: Policy, repetitions: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        self.plan_impl(p, repetitions, graph, None)
    }
    pub fn plan_up_mix<'a>(
        &'a self,
        p: Policy,
        repetitions: usize,
        graph: bool,
        fused: &'a UpMix<'m, 'd>,
    ) -> Result<Plan<'a, 'm, 'd>> {
        self.plan_impl(p, repetitions, graph, Some(fused))
    }
    fn plan_impl<'a>(
        &'a self,
        p: Policy,
        repetitions: usize,
        graph: bool,
        fused: Option<&'a UpMix<'m, 'd>>,
    ) -> Result<Plan<'a, 'm, 'd>> {
        let rows = self.rows.get();
        ensure!(
            rows > 0 && (1..=16).contains(&repetitions),
            "HC plan bounds"
        );
        if let Some(f) = fused {
            ensure!(
                f.reduce.is_none() || p.split_down,
                "fused reduction requires split4 down"
            );
            ensure!(
                f.replicated.is_none() || rows == 1,
                "replicated front is a one-row experiment"
            );
        }
        let make = || {
            let mut calls = self.record_from(
                &self.x,
                self.pending
                    .get()
                    .then_some((&self.pending_block, &self.pending_inject)),
                rows,
                p,
            )?;
            if let Some(fused) = fused {
                // Replace the last two known producer/consumer calls. Keep all
                // diagnostic buffers and BF16 boundaries for a clean comparison.
                calls.truncate(calls.len() - 2);
                if let Some(kernel) = &fused.replicated {
                    // Each output block privately reduces/activates the small
                    // 320-element front. Only block0 publishes diagnostics.
                    // No block waits on another block in this kernel.
                    calls.truncate(calls.len() - 2);
                    calls.push(unsafe {
                        kernel.prepare(
                            vec![
                                Arg::Buffer(self.down.split_partials()?),
                                Arg::Buffer(&self.parts),
                                Arg::Buffer(&self.up.w),
                                Arg::Buffer(&self.up.s),
                                Arg::Buffer(&self.up.b),
                                Arg::Buffer(&self.normalized),
                                Arg::Buffer(&self.lower.fp32),
                                Arg::Buffer(&self.lower.bf16),
                                Arg::Buffer(&self.activated),
                                Arg::Buffer(&self.inject),
                                Arg::Buffer(&self.upper.fp32),
                                Arg::Buffer(&self.upper.bf16),
                                Arg::Buffer(&self.mixed),
                                Arg::U32(self.up.group_size() as u32),
                            ],
                            [(H / 2) as u32, 1, 1],
                            [128, 1, 1],
                        )?
                    });
                    return Ok::<_, anyhow::Error>(calls);
                }
                if let Some(kernel) = &fused.reduce {
                    calls.truncate(calls.len() - 2);
                    calls.push(unsafe {
                        kernel.prepare(
                            vec![
                                Arg::Buffer(self.down.split_partials()?),
                                Arg::Buffer(&self.parts),
                                Arg::Buffer(&self.lower.fp32),
                                Arg::Buffer(&self.lower.bf16),
                                Arg::Buffer(&self.activated),
                                Arg::Buffer(&self.inject),
                                Arg::U32(rows as u32),
                            ],
                            [(rows * (R + HC)).div_ceil(256) as u32, 1, 1],
                            [256, 1, 1],
                        )?
                    });
                }
                let nv = if rows < 8 { 1 } else { 4 };
                calls.push(unsafe {
                    fused.kernels[usize::from(nv == 4)].prepare(
                        vec![
                            Arg::Buffer(&self.activated),
                            Arg::Buffer(&self.up.w),
                            Arg::Buffer(&self.up.s),
                            Arg::Buffer(&self.up.b),
                            Arg::Buffer(&self.normalized),
                            Arg::Buffer(&self.upper.fp32),
                            Arg::Buffer(&self.upper.bf16),
                            Arg::Buffer(&self.mixed),
                            Arg::U32(self.up.group_size() as u32),
                            Arg::U32(rows as u32),
                        ],
                        [(H / 2) as u32, rows.div_ceil(nv) as u32, 1],
                        [128, 1, 1],
                    )?
                });
            }
            Ok::<_, anyhow::Error>(calls)
        };
        let mut calls = make()?;
        let count = calls.len();
        let exec = if graph {
            for _ in 1..repetitions {
                calls.extend(make()?);
            }
            Executor::Graph(self.d.capture(calls)?)
        } else {
            Executor::Prepared(calls)
        };
        Ok(Plan {
            owner: self,
            exec,
            rows,
            pending: self.pending.get(),
            policy: p,
            repetitions,
            calls_per_read: count,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Hc<'m, 'd>,
    exec: Executor<'a, 'm, 'd>,
    rows: usize,
    pending: bool,
    policy: Policy,
    repetitions: usize,
    pub calls_per_read: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Run {
    pub stages: Read,
    pub stream: Vec<u16>,
    pub down_raw: Vec<f32>,
    pub up_raw: Vec<f32>,
    pub injection_partials: Vec<f32>,
    pub partial_values_checked: usize,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        let a = self.owner;
        let rows = self.rows;
        ensure!(
            a.rows.get() == rows && a.pending.get() == self.pending,
            "HC plan invalidated"
        );
        a.poison(self.policy)?;
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
        let mut r = a.collect_from(&a.x, rows, self.pending, self.policy)?;
        r.total_us = total_us;
        r.enqueue_us = enqueue_us;
        r.wait_us = wait_us;
        Ok(r)
    }
}

impl Hc<'_, '_> {
    /// R52: constant weights (device address, bytes; 16-byte aligned) that this
    /// site's front launch warms into L2 for the following launch.
    pub(crate) fn set_pump(&self, region: (u64, usize)) {
        if region.0 % 16 == 0 && region.1 % 16 == 0 && region.1 / 16 <= u32::MAX as usize {
            self.pump.set(Some((region.0, (region.1 / 16) as u32)));
        }
    }
}

/// R60 S4 (prefill v2): HC intermediates for chunk-sized row counts, shared by every HC site of
/// every stage layer (the sites run one after another in one stream).
pub struct HcScratch<'d> {
    rows: usize,
    normalized: Buffer<'d>,
    parts: Buffer<'d>,
    lower: Pair<'d>,
    activated: Buffer<'d>,
    upper: Pair<'d>,
    mixed: Buffer<'d>,
    inject: Buffer<'d>,
    next: Buffer<'d>,
}
impl<'d> HcScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        Ok(Self {
            rows,
            normalized: d.alloc(rows * F * 2 + GUARD)?,
            parts: d.alloc(rows * HC * HC * 4 + GUARD)?,
            lower: Pair::new(d, rows * R)?,
            activated: d.alloc(rows * R * 2 + GUARD)?,
            upper: Pair::new(d, rows * F)?,
            mixed: d.alloc(rows * H * 2 + GUARD)?,
            inject: d.alloc(rows * HC * 2 + GUARD)?,
            next: d.alloc(rows * F * 2 + GUARD)?,
        })
    }
    pub fn mixed(&self) -> &Buffer<'d> {
        &self.mixed
    }
    pub fn injection(&self) -> &Buffer<'d> {
        &self.inject
    }
    pub fn updated_stream(&self) -> &Buffer<'d> {
        &self.next
    }
}
impl<'m, 'd> Hc<'m, 'd> {
    /// R60 S4 (prefill v2): the generic HC site (hc_norm with the pending residual folded in,
    /// down projection, activation, up projection, mix) for chunk rows, intermediates in `s`.
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a HcScratch<'d>,
        x: &'a Buffer<'d>,
        pending: Option<(&'a Buffer<'d>, &'a Buffer<'d>)>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && x.byte_len() >= rows * F * 2, "HC prefill shape");
        let outs = [&s.normalized, &s.parts, &s.lower.fp32, &s.lower.bf16, &s.activated, &s.upper.fp32, &s.upper.bf16, &s.mixed, &s.next];
        ensure!(outs.iter().all(|b| !x.aliases(b)), "HC prefill input/output alias");
        if let Some((b, i)) = pending {
            ensure!(b.byte_len() >= rows * H * 2 && i.byte_len() >= rows * HC * 2 && !outs.iter().any(|o| b.aliases(o)), "HC prefill pending");
        }
        let (b, i) = pending.unwrap_or((&self.pending_block, &self.pending_inject));
        let mut calls = Vec::new();
        // SAFETY: the decode path's kernel ABIs (kernels/lowbit_hc.cu); buffers checked above.
        calls.push(unsafe {
            self.norm.prepare(
                vec![
                    Arg::Buffer(x), Arg::Buffer(&self.nw), Arg::Buffer(&self.iw), Arg::Buffer(b), Arg::Buffer(i),
                    Arg::Buffer(&s.normalized), Arg::Buffer(&s.parts), Arg::Buffer(&s.next), Arg::U32(u32::from(pending.is_some())),
                ],
                [HC as u32, rows as u32, 1],
                [256, 1, 1],
            )?
        });
        calls.push(self.down.record_rows(&s.normalized, &s.lower.fp32, &s.lower.bf16, rows)?);
        calls.push(unsafe {
            self.activate.prepare(
                vec![Arg::Buffer(&s.lower.bf16), Arg::Buffer(&s.parts), Arg::Buffer(&s.activated), Arg::Buffer(&s.inject), Arg::U32(rows as u32)],
                [(rows * (R + HC)).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        });
        calls.push(self.up.record_rows(&s.activated, &s.upper.fp32, &s.upper.bf16, rows)?);
        calls.push(unsafe {
            self.mix.prepare(
                vec![Arg::Buffer(&s.upper.bf16), Arg::Buffer(&s.normalized), Arg::Buffer(&s.mixed), Arg::U32(rows as u32)],
                [(rows * H).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )?
        });
        Ok(calls)
    }
    /// R60 S4 (prefill v2): the residual write (hc_write) for chunk rows.
    pub(crate) fn record_write_prefill<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        block: &'a Buffer<'d>,
        inject: &'a Buffer<'d>,
        out: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            x.byte_len() >= rows * F * 2 && block.byte_len() >= rows * H * 2 && inject.byte_len() >= rows * HC * 2
                && out.byte_len() >= rows * F * 2 && !out.aliases(x) && !out.aliases(block) && !out.aliases(inject),
            "HC prefill write"
        );
        // SAFETY: kernels/lowbit_hc.cu hc_write ABI; buffers checked above.
        unsafe {
            self.write.prepare(
                vec![Arg::Buffer(x), Arg::Buffer(block), Arg::Buffer(inject), Arg::Buffer(out), Arg::U32(rows as u32)],
                [(rows * F).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
            )
        }
    }
}
