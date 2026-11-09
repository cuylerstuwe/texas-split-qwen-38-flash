//! R55 dense affine8 screen: existing QMV and fragment arms versus the multi-row RMV
//! kernels on one real projection, 1-16 rows, warm GPU (one command buffer per arm and
//! row count, one timestamped encoder per projection). RMV outputs must equal QMV's bits.
use crate::{
    dense::Affine8,
    metal_ops::{Arm, Buffer, DenseOps, DenseWeight, Obj, alloc, upload},
    metal_profile::StageClock,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::{Value, json};
use std::ptr::NonNull;

pub struct Rmv {
    pipes: Vec<(String, (usize, usize, usize), Obj<dyn MTLComputePipelineState>)>,
}
impl Rmv {
    pub(crate) fn new(d: &objc2::runtime::ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let lib = d
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../kernels/affine8_rmv.metal")),
                Some(&options),
            )
            .map_err(|e| anyhow::anyhow!("rmv compile: {e}"))?;
        let mut pipes = vec![];
        for (r, c, s) in [
            (1, 2, 2), (1, 4, 2), (1, 8, 2), (1, 4, 4), (1, 8, 4),
            (2, 4, 2), (2, 8, 2), (2, 4, 4),
            (4, 4, 2), (4, 8, 2), (4, 4, 4),
            (8, 2, 2), (8, 4, 2), (8, 8, 2), (8, 4, 4),
        ] {
            let name = format!("affine8_rmv_r{r}_c{c}_s{s}");
            let f = lib.newFunctionWithName(&NSString::from_str(&name)).context("rmv function")?;
            let p = d
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("rmv pipeline {name}: {e}"))?;
            ensure!(p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= 32 * s, "rmv geometry");
            pipes.push((name, (r, c, s), p));
        }
        Ok(Self { pipes })
    }
    /// Encode y = x W^T for `rows` rows with the named variant; rows above the variant's R
    /// run as extra grid rows (each re-reads the weights).
    pub(crate) fn encode(
        &self,
        e: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        variant: usize,
        w: &DenseWeight,
        x: &Buffer,
        y: &Buffer,
        ybf: &Buffer,
        rows: usize,
    ) {
        let (_, (r, c, s), p) = &self.pipes[variant];
        e.setComputePipelineState(p);
        let params = [w.k as u32, w.n as u32, w.group as u32, rows as u32, 1, 0];
        unsafe {
            let [codes, scales, biases] = w.original();
            for (i, b) in [x, codes, scales, biases, y, ybf].into_iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(NonNull::from(&params).cast(), 24, 6);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: w.n.div_ceil(c * s), height: rows.div_ceil(*r), depth: 1 },
            MTLSize { width: 32 * s, height: 1, depth: 1 },
        );
    }
    pub fn variants(&self) -> Vec<(String, (usize, usize, usize))> {
        self.pipes.iter().map(|(n, g, _)| (n.clone(), *g)).collect()
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub fn screen(w: &Affine8, label: &str, rows_list: &[usize], reps: usize) -> Result<Vec<Value>> {
    ensure!(std::env::var("QWEN_HYBRID_GUARDED").as_deref() == Ok("1"), "use the guarded runner");
    ensure!((2..=64).contains(&reps), "screen repetitions");
    let device = MTLCreateSystemDefaultDevice().context("screen device")?;
    let queue = device.newCommandQueue().context("screen queue")?;
    let dw = DenseWeight::new(&device, w)?;
    let ops = DenseOps::new(&device)?;
    let rmv = Rmv::new(&device)?;
    let max_rows = *rows_list.iter().max().context("rows")?;
    ensure!(max_rows <= 16, "screen rows");
    let x = alloc(&device, max_rows * w.k * 2)?;
    let y = alloc(&device, max_rows * w.n * 4)?;
    let ybf = alloc(&device, max_rows * w.n * 2)?;
    let sums = alloc(&device, (max_rows * w.n * 32 * 4).max(max_rows * (w.k / w.group) * 4))?;
    // Deterministic activations, roughly unit scale like normalized hidden states.
    let mut state = 0x2545F4914F6CDD1Du64;
    let xs: Vec<u16> = (0..max_rows * w.k)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            bf16::from_f64((u - 0.5) * 3.4).to_bits()
        })
        .collect();
    upload(&x, &xs)?;
    let bytes = (w.n * w.k + 4 * w.n * (w.k / w.group)) as f64;
    let read = |b: &Buffer, n: usize| -> Vec<u16> {
        unsafe { std::slice::from_raw_parts(b.contents().cast::<u16>().as_ptr(), n).to_vec() }
    };
    let mut out = vec![];
    for &rows in rows_list {
        let (ops, dw, x, y, ybf, sums) = (&ops, &dw, &x, &y, &ybf, &sums);
        let mut arms: Vec<(String, Box<dyn Fn(&objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<()>>)> = vec![];
        arms.push(("qmv".into(), Box::new(|e| ops.encode(e, &dw, &x, &y, &ybf, &sums, rows, 0, Arm::Qmv4))));
        for splits in [2usize, 8] {
            arms.push((
                format!("fragment_packed_f16x2_s{splits}"),
                Box::new(move |e| {
                    ops.encode(e, &dw, &x, &y, &ybf, &sums, rows, 0, Arm::FragmentPacked { splits, residual: true })
                }),
            ));
        }
        for (i, (name, (r, _, _))) in rmv.variants().into_iter().enumerate() {
            // The variant whose R matches the row count (R=8 covers 16 rows in two passes).
            if r != rows.next_power_of_two().min(8) {
                continue;
            }
            let rmv = &rmv;
            arms.push((name, Box::new(move |e| {
                rmv.encode(e, i, dw, x, y, ybf, rows);
                Ok(())
            })));
        }
        let mut reference = None;
        for (name, f) in &arms {
            let mut clock = StageClock::new(&queue)?;
            for _ in 0..reps {
                clock.stage(name.clone(), |e| f(e))?;
            }
            let times: Vec<f64> = clock.finish()?.into_iter().skip(1).map(|s| s.gpu_us).collect();
            let got = read(&ybf, rows * w.n);
            ensure!(got.iter().all(|&v| bf16::from_bits(v).is_finite()), "{name} nonfinite");
            let exact = match &reference {
                None => {
                    reference = Some(got.clone());
                    true
                }
                Some(r) => r == &got,
            };
            let max_rel = reference.as_ref().map(|r| {
                let (mut num, mut den) = (0f64, 0f64);
                for (a, b) in r.iter().zip(&got) {
                    let (a, b) = (bf16::from_bits(*a).to_f64(), bf16::from_bits(*b).to_f64());
                    num += (a - b) * (a - b);
                    den += a * a;
                }
                (num / den.max(1e-30)).sqrt()
            });
            let t = median(times.clone());
            out.push(json!({"kind":"dense_screen","matrix":label,"k":w.k,"n":w.n,"group":w.group,"rows":rows,"arm":name,
                "gpu_us":t,"min_us":times.iter().cloned().fold(f64::INFINITY,f64::min),"gb_s":bytes/t/1e3,
                "bits_equal_qmv":exact,"rel_l2_vs_qmv":max_rel}));
        }
    }
    Ok(out)
}
