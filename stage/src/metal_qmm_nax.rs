//! R58 prefill track, component 1: MLX's affine quantized matmul y = x . W^T (qmm_t) on the M5
//! GPU's matrix units (NAX), for the dense affine8 projections at prefill row counts. The kernel is
//! MLX's (MIT License, Copyright (c) 2023-2024 Apple Inc.; kernels/r58_qmm_nax.in.metal, inlined
//! by tools/inline_metal.py into kernels/mlx_qmm_nax.metal); the dispatch follows MLX's
//! qmm_nax(): 64-wide N tiles, 32- or 64-row M tiles, 128 threads (32 x WN 2 x WM 2), K % 64 == 0.
//! Our affine8 weights already use MLX's layout (four 8-bit codes per u32, little-endian, BF16
//! scales/biases per group of 64), so the model's weight buffers bind unchanged.
use crate::{
    dense::Affine8,
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, alloc, upload},
    metal_profile::StageClock,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use std::ptr::NonNull;

/// The instantiations, for group 64 then group 128 (both occur in the canonical pack: GDN/out/
/// shared-expert/head projections use 128, attention q and the HC weights 64), each in the order
/// (BM 64, N aligned), (64, not), (32, aligned), (32, not).
fn names() -> Vec<String> {
    let mut v = vec![];
    for gs in [64, 128] {
        for (bm, al) in [(64, true), (64, false), (32, true), (32, false)] {
            v.push(format!("affine_qmm_t_nax_bfloat16_t_gs_{gs}_b_8_bm{bm}_bn64_bk64_wm2_wn2_alN_{al}_batch_0"));
        }
    }
    v
}

pub struct QmmNax {
    pipes: Vec<Obj<dyn MTLComputePipelineState>>,
}
impl QmmNax {
    pub fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let opt = MTLCompileOptions::new();
        opt.setMathMode(crate::metal_ops::math_mode());
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("../kernels/mlx_qmm_nax.metal")), Some(&opt))
            .map_err(|e| anyhow::anyhow!("MLX qmm NAX MSL: {e}"))?;
        let mut pipes = vec![];
        for name in names() {
            let name = name.as_str();
            let f = lib.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("function {name}"))?;
            let p = d.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow::anyhow!("pipeline {name}: {e}"))?;
            ensure!(p.maxTotalThreadsPerThreadgroup() >= 128, "qmm NAX threadgroup size");
            pipes.push(p);
        }
        Ok(Self { pipes })
    }
    /// y[m][n] (BF16) = x[m][k] (BF16) . W^T for affine8 W [n][k] with groups of `group` (64 or
    /// 128) along k. Offsets are bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: [&Buffer; 3],
        group: usize,
        x: &Buffer,
        x_offset: usize,
        y: &Buffer,
        y_offset: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(m > 0 && n > 0 && k % 64 == 0 && k % group == 0 && k <= i32::MAX as usize && n <= i32::MAX as usize && m <= i32::MAX as usize, "qmm NAX geometry");
        ensure!(x.length() >= x_offset + m * k * 2 && y.length() >= y_offset + m * n * 2, "qmm NAX buffer bounds");
        let bm = if m <= 32 { 32 } else { 64 };
        let g = match group {
            64 => 0,
            128 => 4,
            _ => anyhow::bail!("qmm NAX group {group}"),
        };
        let idx = g + if bm == 64 { 0 } else { 2 } + if n % 64 == 0 { 0 } else { 1 };
        let (kk, nn, mm) = (k as i32, n as i32, m as i32);
        // The batch arguments a non-batched instantiation declares but never reads.
        let zero_i32 = [0i32; 4];
        let zero_i64 = [0i64; 4];
        unsafe {
            e.setComputePipelineState(&self.pipes[idx]);
            e.setBuffer_offset_atIndex(Some(w[0]), 0, 0);
            e.setBuffer_offset_atIndex(Some(w[1]), 0, 1);
            e.setBuffer_offset_atIndex(Some(w[2]), 0, 2);
            e.setBuffer_offset_atIndex(Some(x), x_offset, 3);
            e.setBuffer_offset_atIndex(Some(y), y_offset, 4);
            e.setBytes_length_atIndex(NonNull::from(&kk).cast(), 4, 5);
            e.setBytes_length_atIndex(NonNull::from(&nn).cast(), 4, 6);
            e.setBytes_length_atIndex(NonNull::from(&mm).cast(), 4, 7);
            for i in [8usize, 11] {
                e.setBytes_length_atIndex(NonNull::from(&zero_i32).cast(), 16, i);
            }
            for i in [9usize, 12] {
                e.setBytes_length_atIndex(NonNull::from(&zero_i32).cast(), 16, i);
            }
            for i in [10usize, 13, 14, 15] {
                e.setBytes_length_atIndex(NonNull::from(&zero_i64).cast(), 32, i);
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(64), height: m.div_ceil(bm), depth: 1 },
                MTLSize { width: 32, height: 2, depth: 2 },
            );
        }
        Ok(())
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// R58 component screen on one real affine8 projection: MLX's NAX GEMM at prefill row counts
/// against the FP64 reference (first rows) and, as the speed baseline, our 16-row fragment arm
/// (FragmentPacked, the target's dense path) run in 16-row pieces. Warm GPU, one timestamped
/// encoder per repetition. Activations are deterministic, roughly unit scale.
pub fn screen(w: &Affine8, label: &str, rows_list: &[usize], reps: usize) -> Result<Vec<Value>> {
    ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use the guarded runner");
    ensure!((2..=32).contains(&reps) && [64, 128].contains(&w.group), "screen setup");
    let device = MTLCreateSystemDefaultDevice().context("screen device")?;
    let queue = device.newCommandQueue().context("screen queue")?;
    let nax = QmmNax::new(&device)?;
    let dw = DenseWeight::new(&device, w)?;
    let ops = DenseOps::new(&device)?;
    let max_rows = *rows_list.iter().max().context("rows")?;
    ensure!(max_rows <= 8192, "screen rows");
    let x = alloc(&device, max_rows * w.k * 2)?;
    let y_nax = alloc(&device, max_rows * w.n * 2)?;
    let (y, ybf) = (alloc(&device, 16 * w.n * 4)?, alloc(&device, 16 * w.n * 2)?);
    let sums = alloc(&device, (16 * w.n * 32 * 4).max(16 * (w.k / w.group) * 4))?;
    let mut state = 0x2545F4914F6CDD1Du64;
    let xs: Vec<u16> = (0..max_rows * w.k)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bf16::from_f64(((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.4).to_bits()
        })
        .collect();
    upload(&x, &xs)?;
    let decoded = w.oracle_weights()?;
    let read = |b: &Buffer, n: usize| -> Vec<u16> { unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), n).to_vec() } };
    let flop = |rows: usize| 2. * rows as f64 * w.n as f64 * w.k as f64;
    let weight_bytes = (w.n * w.k + 4 * w.n * (w.k / w.group)) as f64;
    // Baseline: the fragment arm at 16 rows (its maximum), per-row cost.
    let mut clock = StageClock::new(&queue)?;
    for _ in 0..reps {
        clock.stage("fragment16", |e| ops.encode(e, &dw, &x, &y, &ybf, &sums, 16, 0, Arm::FragmentPacked { splits: 2, residual: true }))?;
    }
    let base: Vec<f64> = clock.finish()?.into_iter().skip(1).map(|s| s.gpu_us).collect();
    let base16 = median(base);
    let frag = read(&ybf, 16 * w.n);
    let mut out = vec![json!({"kind":"qmm_baseline","matrix":label,"k":w.k,"n":w.n,"arm":"fragment_packed_s2","rows":16,"gpu_us":base16,
        "us_per_row":base16 / 16.,"tflops":flop(16) / base16 / 1e6,"weight_gb_s":weight_bytes / base16 / 1e3})];
    for &rows in rows_list {
        let mut clock = StageClock::new(&queue)?;
        for _ in 0..reps {
            clock.stage(format!("nax{rows}"), |e| nax.encode(e, [&dw.original()[0], &dw.original()[1], &dw.original()[2]], w.group, &x, 0, &y_nax, 0, rows, w.n, w.k))?;
        }
        let times: Vec<f64> = clock.finish()?.into_iter().skip(1).map(|s| s.gpu_us).collect();
        let t = median(times);
        let got = read(&y_nax, rows * w.n);
        ensure!(got.iter().all(|&v| bf16::from_bits(v).is_finite()), "NAX output nonfinite");
        // Accuracy: the first rows against the FP64 reference; against the fragment arm too.
        let check = rows.min(8);
        let reference = w.reference(&decoded, &xs[..check * w.k])?;
        let (mut num, mut den, mut max_abs) = (0f64, 0f64, 0f64);
        for (a, r) in got[..check * w.n].iter().zip(&reference) {
            let a = bf16::from_bits(*a).to_f64();
            num += (a - r) * (a - r);
            den += r * r;
            max_abs = max_abs.max((a - r).abs());
        }
        let (mut fnum, mut fden) = (0f64, 0f64);
        for (r, (a, b)) in reference.iter().zip(got.iter().zip(&frag)) {
            let (a, b) = (bf16::from_bits(*a).to_f64(), bf16::from_bits(*b).to_f64());
            fnum += (b - r) * (b - r);
            fden += r * r;
            let _ = a;
        }
        let pieces = rows.div_ceil(16) as f64;
        out.push(json!({"kind":"qmm_nax","matrix":label,"k":w.k,"n":w.n,"group":w.group,"rows":rows,"gpu_us":t,"us_per_row":t / rows as f64,
            "tflops":flop(rows) / t / 1e6,"weight_gb_s":weight_bytes / t / 1e3,
            "fragment16_pieces_us":base16 * pieces,"speedup_vs_fragment16":base16 * pieces / t,
            "rel_l2_vs_fp64":(num / den.max(1e-30)).sqrt(),"max_abs_vs_fp64":max_abs,
            "fragment_rel_l2_vs_fp64":(fnum / fden.max(1e-30)).sqrt(),"checked_rows":check}));
    }
    Ok(out)
}
