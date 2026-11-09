//! R58 prefill track, component 3: MLX's chunked gated delta rule on the M5 GPU's matrix units
//! (kernels/r58_gdn_nax.in.metal -> kernels/mlx_gdn_nax.metal; MIT, Apple), with FP32 gates. One
//! dispatch runs a whole prefill chunk of T tokens for all 48 value heads and leaves the final
//! state in our layout ([Hv][Dv][Dk] FP32), so a decode step can continue from it. Dispatch as
//! MLX's gated_delta_update(): 32 x (Dv/16) x Hv threads in groups of 32 x 4.
use crate::metal_ops::{Buffer, Obj, alloc, upload};
use crate::metal_profile::StageClock;
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use std::ptr::NonNull;

pub const HK: usize = 16;
pub const HV: usize = 48;
pub const D: usize = 128;
const NAME: &str = "gated_delta_fused_nax_bf16_f32gates_128_128_16_48_16";

pub struct GdnNax {
    pipe: Obj<dyn MTLComputePipelineState>,
}
impl GdnNax {
    pub fn new(d: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let opt = MTLCompileOptions::new();
        opt.setMathMode(crate::metal_ops::math_mode());
        let lib = d
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("../kernels/mlx_gdn_nax.metal")), Some(&opt))
            .map_err(|e| anyhow::anyhow!("MLX gated delta NAX MSL: {e}"))?;
        let f = lib.newFunctionWithName(&NSString::from_str(NAME)).context("gated delta function")?;
        let pipe = d.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow::anyhow!("gated delta pipeline: {e}"))?;
        ensure!(pipe.maxTotalThreadsPerThreadgroup() >= 128, "gated delta threadgroup size");
        Ok(Self { pipe })
    }
    /// q, k [t][HK][D] and v, y [t][HV][D] BF16; g, beta [t][HV] FP32 (g = per-step decay);
    /// state_in / state_out [HV][D][D] FP32; they may alias (each SIMD group loads its own
    /// 16 x D slice at the start and stores only that slice at the end).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        state_in: &Buffer,
        g: &Buffer,
        beta: &Buffer,
        y: &Buffer,
        state_out: &Buffer,
        t: usize,
    ) -> Result<()> {
        ensure!(t > 0 && t <= i32::MAX as usize, "gated delta length");
        ensure!(
            q.length() >= t * HK * D * 2 && k.length() >= t * HK * D * 2 && v.length() >= t * HV * D * 2 && y.length() >= t * HV * D * 2
                && g.length() >= t * HV * 4 && beta.length() >= t * HV * 4 && state_in.length() >= HV * D * D * 4 && state_out.length() >= HV * D * D * 4,
            "gated delta buffer bounds"
        );
        let tt = t as i32;
        unsafe {
            e.setComputePipelineState(&self.pipe);
            for (i, b) in [q, k, v, state_in, g, beta, y, state_out].into_iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&tt).cast(), 4, 8);
            e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: 32, height: D / 16, depth: HV }, MTLSize { width: 32, height: 4, depth: 1 });
        }
        Ok(())
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
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Sequential gated delta rule for one value head, in f64 or (emulated) f32 arithmetic:
/// S *= g; d = beta (v - S k); S += d k^T; y = S q. Returns (y [t][D], final S [D][D]).
fn reference(h: usize, t: usize, q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], s0: &[f32], f32_math: bool) -> (Vec<f64>, Vec<f64>) {
    let hk = h / (HV / HK);
    let r = |x: f64| if f32_math { x as f32 as f64 } else { x };
    let mut s: Vec<f64> = s0[h * D * D..(h + 1) * D * D].iter().map(|&x| x as f64).collect();
    let mut y = vec![0f64; t * D];
    for tt in 0..t {
        let (qv, kv) = (&q[(tt * HK + hk) * D..][..D], &k[(tt * HK + hk) * D..][..D]);
        let vv = &v[(tt * HV + h) * D..][..D];
        let (gg, bb) = (g[tt * HV + h] as f64, beta[tt * HV + h] as f64);
        for dv in 0..D {
            let row = &mut s[dv * D..(dv + 1) * D];
            let mut mem = 0f64;
            for dk in 0..D {
                row[dk] = r(row[dk] * gg);
                mem = r(mem + row[dk] * kv[dk] as f64);
            }
            let delta = r(bb * (vv[dv] as f64 - mem));
            let mut out = 0f64;
            for dk in 0..D {
                row[dk] = r(row[dk] + delta * kv[dk] as f64);
                out = r(out + row[dk] * qv[dk] as f64);
            }
            y[tt * D + dv] = out;
        }
    }
    (y, s)
}
fn rel_l2(a: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, r) in a {
        num += (x - r) * (x - r);
        den += r * r;
    }
    (num / den.max(1e-300)).sqrt()
}

/// R58 component screen: MLX's chunked gated delta (NAX, FP32 gates) on realistic synthetic inputs
/// (L2-normalized q/k as our prepare kernel produces them, decays 0.90..0.999 per head, beta
/// 0.05..0.95, a nonzero initial state) against an FP64 sequential reference on four heads, plus
/// the error an FP32 sequential recurrence (our decode kernel's arithmetic) has; GPU time per T.
pub fn screen(t_list: &[usize], reps: usize, ref_cap: usize) -> Result<Vec<Value>> {
    ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use the guarded runner");
    ensure!((2..=32).contains(&reps), "screen repetitions");
    let device = MTLCreateSystemDefaultDevice().context("screen device")?;
    let queue = device.newCommandQueue().context("screen queue")?;
    let nax = GdnNax::new(&device)?;
    let tmax = *t_list.iter().max().context("lengths")?;
    ensure!(tmax <= 16384, "screen length");
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let norm_rows = |rng: &mut Rng, rows: usize, scale: f64| -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * D);
        for _ in 0..rows {
            let raw: Vec<f64> = (0..D).map(|_| rng.u() * 2. - 1.).collect();
            let n = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            out.extend(raw.iter().map(|x| bf16::from_f64(x / n * scale).to_f32()));
        }
        out
    };
    let q = norm_rows(&mut rng, tmax * HK, 1. / (D as f64).sqrt());
    let k = norm_rows(&mut rng, tmax * HK, 1.);
    let v: Vec<f32> = (0..tmax * HV * D).map(|_| bf16::from_f64((rng.u() * 2. - 1.) * 1.5).to_f32()).collect();
    let base: Vec<f64> = (0..HV).map(|h| 1. - 0.1 * (0.01f64).powf(h as f64 / (HV - 1) as f64)).collect(); // 0.90 .. 0.999
    let g: Vec<f32> = (0..tmax * HV).map(|i| base[i % HV].powf(0.5 + rng.u()) as f32).collect();
    let beta: Vec<f32> = (0..tmax * HV).map(|_| (0.05 + 0.9 * rng.u()) as f32).collect();
    let s0: Vec<f32> = (0..HV * D * D).map(|_| ((rng.u() * 2. - 1.) * 0.05) as f32).collect();
    let bf = |x: &[f32]| -> Vec<u16> { x.iter().map(|&v| bf16::from_f32(v).to_bits()).collect() };
    let (bq, bk, bv) = (alloc(&device, q.len() * 2)?, alloc(&device, k.len() * 2)?, alloc(&device, v.len() * 2)?);
    upload(&bq, &bf(&q))?;
    upload(&bk, &bf(&k))?;
    upload(&bv, &bf(&v))?;
    let (bg, bb, bs0) = (alloc(&device, g.len() * 4)?, alloc(&device, beta.len() * 4)?, alloc(&device, s0.len() * 4)?);
    upload(&bg, &g)?;
    upload(&bb, &beta)?;
    upload(&bs0, &s0)?;
    let by = alloc(&device, tmax * HV * D * 2)?;
    let bs1 = alloc(&device, HV * D * D * 4)?;
    let mut out = vec![];
    for &t in t_list {
        let mut clock = StageClock::new(&queue)?;
        for _ in 0..reps {
            clock.stage(format!("gdn{t}"), |e| nax.encode(e, &bq, &bk, &bv, &bs0, &bg, &bb, &by, &bs1, t))?;
        }
        let times: Vec<f64> = clock.finish()?.into_iter().skip(1).map(|s| s.gpu_us).collect();
        let tm = median(times);
        let y: Vec<f32> = unsafe { std::slice::from_raw_parts(by.contents().cast::<u16>().as_ptr(), t * HV * D) }
            .iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let s1: Vec<f32> = unsafe { std::slice::from_raw_parts(bs1.contents().cast::<f32>().as_ptr(), HV * D * D).to_vec() };
        ensure!(y.iter().chain(&s1).all(|x| x.is_finite()), "gated delta output nonfinite");
        let mut acc = json!(null);
        if t <= ref_cap {
            let heads = [0usize, 7, 23, 47];
            let (mut ey, mut es, mut ey32, mut es32) = (vec![], vec![], vec![], vec![]);
            for &h in &heads {
                let (yr, sr) = reference(h, t, &q, &k, &v, &g, &beta, &s0, false);
                let (y32, s32) = reference(h, t, &q, &k, &v, &g, &beta, &s0, true);
                for tt in 0..t {
                    for dv in 0..D {
                        ey.push((y[(tt * HV + h) * D + dv] as f64, yr[tt * D + dv]));
                        ey32.push((bf16::from_f64(y32[tt * D + dv]).to_f64(), yr[tt * D + dv]));
                    }
                }
                for i in 0..D * D {
                    es.push((s1[h * D * D + i] as f64, sr[i]));
                    es32.push((s32[i], sr[i]));
                }
            }
            acc = json!({"heads":heads,"y_rel_l2_vs_fp64":rel_l2(ey.into_iter()),"state_rel_l2_vs_fp64":rel_l2(es.into_iter()),
                "fp32_sequential_y_rel_l2":rel_l2(ey32.into_iter()),"fp32_sequential_state_rel_l2":rel_l2(es32.into_iter())});
        }
        out.push(json!({"kind":"gdn_nax","tokens":t,"gpu_us":tm,"us_per_token":tm / t as f64,"accuracy":acc}));
    }
    Ok(out)
}
