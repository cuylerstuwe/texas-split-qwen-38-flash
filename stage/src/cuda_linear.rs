//! Native BF16 linear operations using pinned, audited tinycublas cubin symbols.
//! No CUBLAS host wrapper/global handles, allocator or runtime heuristics.
use crate::{
    bf16_linear::{MAX_ROWS, Matrix},
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
use std::{cell::Cell, path::Path, time::Instant};

pub const CUBIN: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/macuda/gemm.sm_120.cubin");
pub const CUBIN_SHA: &str = "4ee3b384e349b89bfe3d28bb9b65d99f9fff3b8a00b2afa9550422b7bb99daaa";
pub const SOURCE_SHA: &str = "bbb21c8d5c6717618c997c252f10ecf44563664cd47fd91a28d34bba3739f40d";
const GUARD: usize = 256;
const MAX_SPLITS: usize = 16;
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // Used only for initialized u16/f32 scalars on this little-endian host.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
pub fn load_module(device: &Device) -> Result<Module<'_>> {
    device.module(Path::new(CUBIN), CUBIN_SHA)
}

#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Small,
    Large,
    Split(usize),
}
impl Arm {
    pub const ALL: [Self; 6] = [
        Self::Small,
        Self::Large,
        Self::Split(2),
        Self::Split(4),
        Self::Split(8),
        Self::Split(16),
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Small => "tc64",
            Self::Large => "tc128",
            Self::Split(2) => "tc64_split2",
            Self::Split(4) => "tc64_split4",
            Self::Split(8) => "tc64_split8",
            Self::Split(16) => "tc64_split16",
            _ => "invalid_split",
        }
    }
    fn splits(self) -> usize {
        if let Self::Split(s) = self { s } else { 1 }
    }
}
#[derive(Clone, Copy)]
pub enum Output {
    F32,
    Bf16,
}
impl Output {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
    fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
    fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::Bf16 => 2,
        }
    }
}
pub struct Run {
    pub values: Vec<f32>,
    pub launch_us: f64,
    pub completion_us: f64,
    pub total_us: f64,
    pub counter_words_checked: usize,
}
pub struct Linear<'m, 'd> {
    device: &'d Device,
    small: Kernel<'m, 'd>,
    large: Kernel<'m, 'd>,
    split: Kernel<'m, 'd>,
    w: Buffer<'d>,
    x: Buffer<'d>,
    y: Buffer<'d>,
    scratch: Buffer<'d>,
    counters: Buffer<'d>,
    k: usize,
    n: usize,
    output_bytes: usize,
    counter_bytes: usize,
    allocated_bytes: usize,
    prepared_rows: Cell<usize>,
}
impl<'m, 'd> Linear<'m, 'd> {
    pub fn new(device: &'d Device, module: &'m Module<'d>, weight: &Matrix) -> Result<Self> {
        weight.validate()?;
        let tiles = weight.n.div_ceil(64) * MAX_ROWS.div_ceil(64);
        let output_bytes = MAX_ROWS * weight.n * 4 + GUARD;
        let scratch_bytes = MAX_SPLITS * tiles * 64 * 64 * 4;
        let counter_bytes = tiles * 4 + GUARD;
        let w = device.alloc(weight.values.len() * 2)?;
        w.upload(bytes(&weight.values))?;
        // Full weight readback settles transfer and verifies actual device contents.
        let mut check = vec![0; weight.values.len() * 2];
        w.download(&mut check)?;
        ensure!(check == bytes(&weight.values), "CUDA BF16 weight upload");
        let counters = device.alloc(counter_bytes)?;
        counters.upload(&vec![0; counter_bytes])?;
        Ok(Self {
            device,
            small: module.kernel("tinyblas_gemm_bf16_tc2s_tn")?,
            large: module.kernel("tinyblas_gemm_bf16_tc2_tn")?,
            split: module.kernel("tinyblas_gemm_bf16_tc2k_tn")?,
            w,
            x: device.alloc(MAX_ROWS * weight.k * 2)?,
            y: device.alloc(output_bytes)?,
            scratch: device.alloc(scratch_bytes)?,
            counters,
            k: weight.k,
            n: weight.n,
            output_bytes,
            counter_bytes,
            prepared_rows: Cell::new(0),
            allocated_bytes: weight.values.len() * 2
                + MAX_ROWS * weight.k * 2
                + output_bytes
                + scratch_bytes
                + counter_bytes,
        })
    }
    pub fn description(&self) -> serde_json::Value {
        json!({"backend":"native CUDA/libtinynv","cubin":CUBIN,"cubin_sha256":CUBIN_SHA,
            "source_sha256":SOURCE_SHA,"k":self.k,"n":self.n,"allocated_bytes":self.allocated_bytes,
            "weight_dtype":"BF16","accumulation":"FP32","host_wrapper_used":false,
            "weight_readback_checked_bytes":self.k*self.n*2,
            "kernels":{"small":self.small.description(),"large":self.large.description(),"split":self.split.description()},
            "timing":"host enqueue + stream completion; NOT GPU timestamps","single_stream_private_scratch":true})
    }
    pub fn prepare(&self, input: &[u16]) -> Result<(usize, f64)> {
        ensure!(
            !input.is_empty() && input.len() % self.k == 0 && input.len() / self.k <= MAX_ROWS,
            "CUDA linear input geometry"
        );
        ensure!(
            input.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "CUDA linear input finite"
        );
        let t = Instant::now();
        self.prepared_rows.set(0);
        self.x.upload(bytes(input))?;
        let mut read = vec![0; input.len() * 2];
        self.x.download(&mut read)?;
        ensure!(read == bytes(input), "CUDA resident input upload");
        self.prepared_rows.set(input.len() / self.k);
        Ok((input.len() / self.k, t.elapsed().as_secs_f64() * 1e6))
    }
    pub fn run(
        &self,
        rows: usize,
        arm: Arm,
        output: Output,
        repetitions: usize,
        wait_each: bool,
    ) -> Result<Run> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && rows == self.prepared_rows.get()
                && (1..=64).contains(&repetitions),
            "CUDA linear run bounds"
        );
        ensure!(
            [1, 2, 4, 8, 16].contains(&arm.splits()),
            "CUDA linear split bound"
        );
        self.measure(rows, output, || {
            let mut launch_us = 0.;
            let mut completion_us = 0.;
            for _ in 0..repetitions {
                let t = Instant::now();
                self.encode(rows, arm, output)?;
                launch_us += t.elapsed().as_secs_f64() * 1e6;
                if wait_each {
                    let t = Instant::now();
                    self.device.sync()?;
                    completion_us += t.elapsed().as_secs_f64() * 1e6;
                }
            }
            Ok((launch_us, completion_us))
        })
    }
    fn measure(
        &self,
        rows: usize,
        output: Output,
        work: impl FnOnce() -> Result<(f64, f64)>,
    ) -> Result<Run> {
        ensure!(
            rows == self.prepared_rows.get() && rows > 0,
            "CUDA prepared input rows changed"
        );
        // Outside resident timing. Poison trailing storage as well as live output.
        self.y.upload(&vec![0xff; self.output_bytes])?;
        self.device.sync()?;
        let total = Instant::now();
        let (launch_us, mut completion_us) = work()?;
        let t = Instant::now();
        self.device.sync()?;
        completion_us += t.elapsed().as_secs_f64() * 1e6;
        let total_us = total.elapsed().as_secs_f64() * 1e6;
        let mut raw = vec![0; self.output_bytes];
        self.y.download(&mut raw)?;
        let active = rows * self.n * output.bytes();
        ensure!(
            raw[active..].iter().all(|&v| v == 0xff),
            "CUDA trailing output canary"
        );
        let values: Vec<_> = match output {
            Output::F32 => raw[..active]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Output::Bf16 => raw[..active]
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
        };
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "CUDA output unwritten/nonfinite"
        );
        let mut cnt = vec![0xff; self.counter_bytes];
        self.counters.download(&mut cnt)?;
        ensure!(
            cnt.iter().all(|&v| v == 0),
            "CUDA split counters not reset or guard overwritten"
        );
        Ok(Run {
            values,
            launch_us,
            completion_us,
            total_us,
            counter_words_checked: cnt.len() / 4,
        })
    }
    fn encode(&self, rows: usize, arm: Arm, output: Output) -> Result<()> {
        self.call(rows, arm, output)?.enqueue()
    }
    fn call(&self, rows: usize, arm: Arm, output: Output) -> Result<PreparedCall<'_, 'm, 'd>> {
        self.record_into(&self.x, &self.y, rows, arm, output)
    }
    /// Composable resident operation. Buffers are retained by the prepared call;
    /// callers order producer/consumer calls on this device's one owned stream.
    pub(crate) fn record_into<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        dest: &'a Buffer<'d>,
        rows: usize,
        arm: Arm,
        output: Output,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && [1, 2, 4, 8, 16].contains(&arm.splits()),
            "CUDA linear record geometry"
        );
        ensure!(
            input.byte_len() >= rows * self.k * 2
                && dest.byte_len() >= rows * self.n * output.bytes()
                && !input.aliases(dest)
                && !dest.aliases(&self.w)
                && !input.aliases(&self.scratch)
                && !dest.aliases(&self.scratch)
                && !input.aliases(&self.counters)
                && !dest.aliases(&self.counters),
            "CUDA connected linear buffers"
        );
        // Y[T,N] = X[T,K] W[N,K]^T becomes column-major C[N,T]=op(W)[N,K]*X[K,T].
        // K%8 and aligned allocation guarantee all cp.async vector loads are in bounds.
        let mut args = vec![
            Arg::Buffer(&self.w),
            Arg::Buffer(input),
            Arg::Buffer(dest),
            Arg::U64(0),
            Arg::U64(0),
            Arg::U64(0),
            Arg::U32(self.n as u32),
            Arg::U32(rows as u32),
            Arg::U32(self.k as u32),
            Arg::U32(self.k as u32),
            Arg::U32(self.k as u32),
            Arg::U32(self.n as u32),
            Arg::F32(1.),
            Arg::F32(0.),
            Arg::U32(output.code()),
        ];
        let (kernel, tile, threads) = match arm {
            Arm::Small => (&self.small, 64, 128),
            Arm::Large => (&self.large, 128, 256),
            Arm::Split(s) => {
                args.extend([
                    Arg::Buffer(&self.scratch),
                    Arg::Buffer(&self.counters),
                    Arg::U32(s as u32),
                ]);
                (&self.split, 64, 128)
            }
        };
        // Exact pinned ABI, positive bounded shape, disjoint owned buffers; each
        // split has a complete64x64 FP32 scratch tile. Same stream serializes reuse.
        unsafe {
            kernel.prepare(
                args,
                [
                    self.n.div_ceil(tile) as u32,
                    rows.div_ceil(tile) as u32,
                    arm.splits() as u32,
                ],
                [threads, 1, 1],
            )
        }
    }
    pub(crate) fn check_counters(&self) -> Result<usize> {
        let mut c = vec![0xff; self.counter_bytes];
        self.counters.download(&mut c)?;
        ensure!(
            c.iter().all(|&v| v == 0),
            "connected linear counters not reset/guard changed"
        );
        Ok(c.len() / 4)
    }
    pub fn plan(
        &self,
        rows: usize,
        arm: Arm,
        output: Output,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            (1..=64).contains(&repetitions) && rows == self.prepared_rows.get(),
            "CUDA linear plan bounds"
        );
        let executor = if graph {
            let calls = (0..repetitions)
                .map(|_| self.call(rows, arm, output))
                .collect::<Result<Vec<_>>>()?;
            Executor::Graph(self.device.capture(calls)?)
        } else {
            Executor::Prepared(self.call(rows, arm, output)?)
        };
        Ok(Plan {
            owner: self,
            executor,
            rows,
            output,
            repetitions,
        })
    }
}
enum Executor<'a, 'm, 'd> {
    Prepared(PreparedCall<'a, 'm, 'd>),
    Graph(Graph<'a, 'm, 'd>),
}
/// Retains this operator's buffers/module. Changed input of the same shape is
/// supported; upload orders and completes before replay. Linear is not Send/Sync.
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Linear<'m, 'd>,
    executor: Executor<'a, 'm, 'd>,
    rows: usize,
    output: Output,
    repetitions: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self) -> Result<Run> {
        self.owner.measure(self.rows, self.output, || {
            let t = Instant::now();
            match &self.executor {
                Executor::Prepared(c) => {
                    for _ in 0..self.repetitions {
                        c.enqueue()?;
                    }
                }
                Executor::Graph(g) => g.launch()?,
            }
            Ok((t.elapsed().as_secs_f64() * 1e6, 0.))
        })
    }
}
