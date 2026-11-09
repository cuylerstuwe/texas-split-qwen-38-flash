//! R58 prefill track, component 2: the routed experts of one layer for a prefill chunk of T tokens,
//! on the full 512-expert bank. Slots (token, expert) are sorted by expert and cut into blocks of up
//! to 32 rows that share an expert; every block runs R55's exl3_mpp kernel (EXL3 K4 / MCG trellis
//! decoded straight into matrix fragments, adapted from Sushi's NAX reference), so each expert tile
//! is decoded once per block instead of once per slot as on the decode path. Split-K is off: the
//! blocks give enough parallelism. Gather, sign and Hadamard transforms, the SiLU gate and the
//! weighted combine are R55's exl3_moe.metal kernels. Each output row's arithmetic does not depend
//! on which other rows share its block, so the result is independent of how slots are blocked.
use crate::moe::{FullExpertBank, H, I, TOPK};
use crate::metal_profile::StageClock;
use crate::projection::ScaledProjection;
use anyhow::{Context, Result, ensure};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use std::ptr::NonNull;

type Obj<T> = Retained<ProtocolObject<T>>;
type Buffer = Obj<dyn MTLBuffer>;
type Pipeline = Obj<dyn MTLComputePipelineState>;
pub const EXPERTS: usize = 512;

struct Weight {
    trellis: Buffer,
    suh: Buffer,
    svh: Buffer,
}
fn alloc(d: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer> {
    d.newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared).context("Metal allocation")
}
fn upload<T: Copy>(b: &ProtocolObject<dyn MTLBuffer>, data: &[T]) -> Result<()> {
    ensure!(std::mem::size_of_val(data) <= b.length(), "Metal upload bound");
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), b.contents().cast::<u8>().as_ptr(), std::mem::size_of_val(data)) };
    Ok(())
}
fn weight(d: &ProtocolObject<dyn MTLDevice>, p: &ScaledProjection, k: usize, n: usize) -> Result<Weight> {
    ensure!(
        p.raw.k == k && p.raw.n == n && p.raw.experts == EXPERTS && p.raw.packed.len() == EXPERTS * k * n / 2
            && p.suh.len() == EXPERTS * k && p.svh.len() == EXPERTS * n,
        "prefill MoE weight shape"
    );
    let w = Weight { trellis: alloc(d, p.raw.packed.len())?, suh: alloc(d, p.suh.len() * 2)?, svh: alloc(d, p.svh.len() * 2)? };
    upload(&w.trellis, &p.raw.packed)?;
    upload(&w.suh, &p.suh)?;
    upload(&w.svh, &p.svh)?;
    Ok(w)
}
fn dispatch(e: &ProtocolObject<dyn MTLComputeCommandEncoder>, x: usize, y: usize, z: usize, threads: usize) {
    e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: x, height: y, depth: z }, MTLSize { width: threads, height: 1, depth: 1 });
}
fn bind(e: &ProtocolObject<dyn MTLComputeCommandEncoder>, buffers: &[&ProtocolObject<dyn MTLBuffer>], params: &[u32; 8]) {
    unsafe {
        for (i, b) in buffers.iter().enumerate() {
            e.setBuffer_offset_atIndex(Some(b), 0, i);
        }
        e.setBytes_length_atIndex(NonNull::from(params).cast(), 32, 7);
    }
}

/// The sorted slot plan for a chunk: order (sorted slot -> original slot), inverse, sorted expert
/// ids, and same-expert blocks [start, rows, expert, 0] of at most `block_rows` rows.
pub struct Plan {
    pub order: Vec<u32>,
    pub inverse: Vec<u32>,
    pub ids: Vec<u32>,
    pub blocks: Vec<[u32; 4]>,
}
pub fn plan(ids: &[u32], block_rows: usize) -> Result<Plan> {
    ensure!((1..=32).contains(&block_rows) && ids.len() % TOPK == 0, "plan geometry");
    let mut order: Vec<u32> = (0..ids.len() as u32).collect();
    order.sort_unstable_by_key(|&s| (ids[s as usize], s));
    let sorted: Vec<u32> = order.iter().map(|&s| ids[s as usize]).collect();
    let mut inverse = vec![0u32; order.len()];
    for (i, &s) in order.iter().enumerate() {
        inverse[s as usize] = i as u32;
    }
    let mut blocks = vec![];
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && end - start < block_rows && sorted[end] == sorted[start] {
            end += 1;
        }
        blocks.push([start as u32, (end - start) as u32, sorted[start], 0]);
        start = end;
    }
    Ok(Plan { order, inverse, ids: sorted, blocks })
}

pub struct MoePrefill {
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    mpp: Pipeline,
    prepare: Pipeline,
    mid: Pipeline,
    finish: Pipeline,
    gate: Weight,
    up: Weight,
    down: Weight,
    max_rows: usize,
    x: Buffer,
    ids: Buffer,
    order: Buffer,
    inverse: Buffer,
    scores: Buffer,
    blocks: Buffer,
    pg: Buffer,
    pu: Buffer,
    rg: Buffer,
    ru: Buffer,
    md: Buffer,
    rd: Buffer,
    out: Buffer,
    out_bf: Buffer,
}
impl MoePrefill {
    pub fn new(bank: &FullExpertBank, max_rows: usize) -> Result<Self> {
        ensure!((1..=8192).contains(&max_rows), "prefill MoE rows");
        let device = MTLCreateSystemDefaultDevice().context("Metal device")?;
        let queue = device.newCommandQueue().context("Metal queue")?;
        let options = MTLCompileOptions::new();
        options.setMathMode(crate::metal_ops::math_mode());
        let src = format!("{}\n{}", include_str!("../kernels/exl3_mpp.metal"), include_str!("../kernels/exl3_moe.metal"));
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&options))
            .map_err(|e| anyhow::anyhow!("prefill MoE MSL: {e}"))?;
        let pipe = |name: &str, threads: usize| -> Result<Pipeline> {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).context("prefill MoE function")?;
            let p = device.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow::anyhow!("prefill MoE pipeline {name}: {e}"))?;
            ensure!(p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= threads, "prefill MoE geometry");
            Ok(p)
        };
        let (mpp, prepare, mid, finish) = (pipe("exl3_mpp", 128)?, pipe("moe_prepare_pair", 32)?, pipe("moe_mid", 32)?, pipe("moe_finish", TOPK * 32)?);
        let (gate, up, down) = (weight(&device, &bank.gate, H, I)?, weight(&device, &bank.up, H, I)?, weight(&device, &bank.down, I, H)?);
        let slots = max_rows * TOPK;
        Ok(Self {
            x: alloc(&device, max_rows * H * 4)?,
            ids: alloc(&device, slots * 4)?,
            order: alloc(&device, slots * 4)?,
            inverse: alloc(&device, slots * 4)?,
            scores: alloc(&device, slots * 4)?,
            blocks: alloc(&device, slots * 16)?,
            pg: alloc(&device, slots * H * 2)?,
            pu: alloc(&device, slots * H * 2)?,
            rg: alloc(&device, slots * I * 4)?,
            ru: alloc(&device, slots * I * 4)?,
            md: alloc(&device, slots * I * 2)?,
            rd: alloc(&device, slots * H * 4)?,
            out: alloc(&device, max_rows * H * 4)?,
            out_bf: alloc(&device, max_rows * H * 2)?,
            device,
            queue,
            mpp,
            prepare,
            mid,
            finish,
            gate,
            up,
            down,
            max_rows,
        })
    }
    fn project(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, x: &Buffer, w: &Weight, out: &Buffer, k: usize, n: usize, blocks: usize) {
        let p = [k as u32, n as u32, 1u32, 0];
        unsafe {
            for (i, b) in [(0usize, x), (1, &w.trellis), (2, &self.ids), (3, out), (5, &self.blocks)] {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&p).cast(), 16, 4);
        }
        e.setComputePipelineState(&self.mpp);
        dispatch(e, n / 128, blocks, 1, 128);
    }
    /// Encode the routed experts for `rows` tokens whose plan is uploaded (see stage()).
    fn encode(&self, e: &ProtocolObject<dyn MTLComputeCommandEncoder>, rows: usize, blocks: usize) {
        let slots = rows * TOPK;
        let p = [H as u32, I as u32, TOPK as u32, 1, 1, 0, 0, 0];
        e.setComputePipelineState(&self.prepare);
        bind(e, &[&self.x, &self.gate.suh, &self.up.suh, &self.ids, &self.order, &self.pg, &self.pu], &p);
        dispatch(e, H / 128, slots, 1, 32);
        crate::metal_ops::barrier(e);
        self.project(e, &self.pg, &self.gate, &self.rg, H, I, blocks);
        self.project(e, &self.pu, &self.up, &self.ru, H, I, blocks);
        crate::metal_ops::barrier(e);
        e.setComputePipelineState(&self.mid);
        bind(e, &[&self.rg, &self.ru, &self.gate.svh, &self.up.svh, &self.down.suh, &self.ids, &self.md], &p);
        dispatch(e, I / 128, slots, 1, 32);
        crate::metal_ops::barrier(e);
        self.project(e, &self.md, &self.down, &self.rd, I, H, blocks);
        crate::metal_ops::barrier(e);
        e.setComputePipelineState(&self.finish);
        bind(e, &[&self.rd, &self.down.svh, &self.ids, &self.inverse, &self.scores, &self.out, &self.out_bf], &p);
        dispatch(e, H / 128, rows, 1, TOPK * 32);
    }
    /// Upload one chunk's input rows (FP32 [rows][H]), routes and their plan.
    fn stage(&self, x: &[f32], scores: &[f32], plan: &Plan) -> Result<()> {
        let rows = x.len() / H;
        ensure!(rows >= 1 && rows <= self.max_rows && x.len() == rows * H && scores.len() == rows * TOPK && plan.order.len() == rows * TOPK, "prefill MoE inputs");
        upload(&self.x, x)?;
        upload(&self.ids, &plan.ids)?;
        upload(&self.order, &plan.order)?;
        upload(&self.inverse, &plan.inverse)?;
        upload(&self.scores, scores)?;
        upload(&self.blocks, &plan.blocks)?;
        Ok(())
    }
    /// One chunk: (BF16 output bits [rows][H], median GPU us over reps).
    pub fn run(&self, x: &[f32], ids: &[u32], scores: &[f32], block_rows: usize, reps: usize) -> Result<(Vec<u16>, f64)> {
        let rows = x.len() / H;
        let plan = plan(ids, block_rows)?;
        self.stage(x, scores, &plan)?;
        let mut clock = StageClock::new(&self.queue)?;
        for _ in 0..reps.max(1) {
            clock.stage("moe", |e| {
                self.encode(e, rows, plan.blocks.len());
                Ok(())
            })?;
        }
        let mut t: Vec<f64> = clock.finish()?.into_iter().map(|s| s.gpu_us).collect();
        if t.len() > 1 {
            t.remove(0);
        }
        t.sort_by(f64::total_cmp);
        let out = unsafe { std::slice::from_raw_parts(self.out_bf.contents().cast::<u16>().as_ptr(), rows * H).to_vec() };
        Ok((out, t[t.len() / 2]))
    }
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// R58 component screen on one layer's real 512-expert bank. Routes are uniform random distinct
/// top-10 per token (the conservative case for block sizes; real routing concentrates on popular
/// experts), softmax-like scores; inputs roughly unit scale. For each T: GPU time per chunk, then
/// the same rows re-run in 16-row sub-chunks must give bitwise-identical outputs (blocking
/// independence) for T up to `check_cap`.
pub fn screen(bank: &FullExpertBank, t_list: &[usize], reps: usize, check_cap: usize) -> Result<Vec<Value>> {
    ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use the guarded runner");
    let tmax = *t_list.iter().max().context("lengths")?;
    let m = MoePrefill::new(bank, tmax)?;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let x: Vec<f32> = (0..tmax * H).map(|_| ((rng.u() - 0.5) * 3.4) as f32).collect();
    let mut ids = Vec::with_capacity(tmax * TOPK);
    let mut scores = Vec::with_capacity(tmax * TOPK);
    for _ in 0..tmax {
        let mut row: Vec<u32> = vec![];
        while row.len() < TOPK {
            let e = (rng.u() * EXPERTS as f64) as u32 % EXPERTS as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        let w: Vec<f64> = (0..TOPK).map(|_| (rng.u() * 3.).exp()).collect();
        let z: f64 = w.iter().sum();
        ids.extend(row);
        scores.extend(w.iter().map(|v| (v / z) as f32));
    }
    let mut out = vec![];
    for &t in t_list {
        let (y, us) = m.run(&x[..t * H], &ids[..t * TOPK], &scores[..t * TOPK], 32, reps)?;
        let p = plan(&ids[..t * TOPK], 32)?;
        let mut check = json!(null);
        if t <= check_cap && t > 16 {
            let mut same = true;
            for c in 0..t.div_ceil(16) {
                let (a, b) = (c * 16, ((c + 1) * 16).min(t));
                let (yc, _) = m.run(&x[a * H..b * H], &ids[a * TOPK..b * TOPK], &scores[a * TOPK..b * TOPK], 32, 1)?;
                same &= yc == y[a * H..b * H];
            }
            check = json!({"bitwise_equal_to_16_row_chunks":same});
        }
        ensure!(y.iter().all(|&b| half::bf16::from_bits(b).is_finite()), "prefill MoE output nonfinite");
        out.push(json!({"kind":"moe_prefill","tokens":t,"blocks":p.blocks.len(),"mean_rows_per_block":(t * TOPK) as f64 / p.blocks.len() as f64,
            "used_experts":p.blocks.iter().map(|b| b[2]).collect::<std::collections::BTreeSet<_>>().len(),
            "gpu_us":us,"us_per_token":us / t as f64,"check":check}));
    }
    Ok(out)
}
