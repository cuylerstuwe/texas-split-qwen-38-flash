//! Connected K1.75 routed experts; six dependent device calls, no host boundary.
use crate::{
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_mcg_grouped::Grouped,
    mcg_moe::Bank,
    moe::{H, I, Routes, TOPK},
};
use anyhow::{Result, ensure};
use half::{bf16, f16};
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const OPS_SHA: &str = "570bfb1ad55d75c84e3efd7a2ede0807810d5d5505d9bc52a6ab79a09aa91978";
pub const PROJECT_SHA: &str = "8c36f6afc2375274c2ae76a7ad07e029280df6e400ee341b8cc712cd5c2b36b8";
pub const MAX_ROWS: usize = 16;
const GUARD: usize = 256;
const SLOTS: usize = MAX_ROWS * TOPK;
pub const NAMES: [&str; 8] = [
    "prepared_gate",
    "prepared_up",
    "raw_gate",
    "raw_up",
    "middle",
    "raw_down",
    "output",
    "bf16_output",
];
fn u16_bytes(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn u32_bytes(x: &[u32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn f32_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
/// R38 (Opus, direct): unchanged mcg_moe.cu plus load-ahead mid and finish(+add).
pub const R38_OPS_SHA: &str = "83e427c7863f4a951dafd767fa5c6b5d453813f360198b156dddf2e8a9b879bc";
pub fn modules(d: &Device) -> Result<(Module<'_>, Module<'_>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    Ok((
        d.module(&root.join("mcg_project.sm120.cubin"), PROJECT_SHA)?,
        if crate::cuda_affine4::r38_enabled() {
            d.module(&root.join("mcg_moe_r38.sm120.cubin"), R38_OPS_SHA)?
        } else {
            d.module(&root.join("mcg_moe.sm120.cubin"), OPS_SHA)?
        },
    ))
}
pub struct Moe<'m, 'd> {
    device: &'d Device,
    project: Kernel<'m, 'd>,
    prepare_bf16: Option<Kernel<'m, 'd>>,
    full_ids: bool,
    prepare: Kernel<'m, 'd>,
    middle: Kernel<'m, 'd>,
    finish: Kernel<'m, 'd>,
    r38: Option<[Kernel<'m, 'd>; 2]>,
    weights: Vec<Buffer<'d>>,
    input: Buffer<'d>,
    ids: Buffer<'d>,
    scores: Buffer<'d>,
    bufs: Vec<Buffer<'d>>,
    experts: usize,
    /// MCG halfwords per 16x16 tile: 28 (K1.75) or 64 (canonical K4, scalar path only)
    halfwords: usize,
    rows: Cell<usize>,
    grouping: Option<Grouped<'m, 'd>>,
    tensor: Option<crate::cuda_mcg_tensor::Tensor<'m, 'd>>,
    /// R60 S4: load-ahead K4 scalar projections (mcg4_tensor mcg_project4v [8 steps ahead],
    /// mcg_project4v4 [4 steps ahead, 12 blocks/SM]), same ABI/geometry.
    project4: Option<[Kernel<'m, 'd>; 3]>,
}
impl<'m, 'd> Moe<'m, 'd> {
    pub fn new(
        d: &'d Device,
        project: &'m Module<'d>,
        ops: &'m Module<'d>,
        bank: &Bank,
    ) -> Result<Self> {
        let mut weights = vec![];
        // K1.75 (28 halfwords per 16x16 tile) or, for the R60 split, the canonical K4 (64)
        let halfwords = bank.gate.raw.halfwords;
        for (p, k, n) in [(&bank.gate, H, I), (&bank.up, H, I), (&bank.down, I, H)] {
            ensure!(
                p.raw.k == k
                    && p.raw.n == n
                    && [28, 64].contains(&p.raw.halfwords)
                    && p.raw.halfwords == halfwords
                    && p.raw.experts == bank.original_ids.len()
                    && p.suh.len() == p.raw.experts * k
                    && p.svh.len() == p.raw.experts * n
                    && p.raw.packed.len() == p.raw.experts * k * n / 256 * halfwords * 2,
                "CUDA lowbit bank geometry"
            );
            for bytes in [p.raw.packed.clone(), u16_bytes(&p.suh), u16_bytes(&p.svh)] {
                let b = d.alloc(bytes.len())?;
                b.upload(&bytes)?;
                let mut check = vec![0; bytes.len()];
                b.download(&mut check)?;
                ensure!(check == bytes, "lowbit weight transfer");
                weights.push(b);
            }
        }
        let capacities = [
            SLOTS * H * 2,
            SLOTS * H * 2,
            SLOTS * I * 4 * 4,
            SLOTS * I * 4 * 4,
            SLOTS * I * 2,
            SLOTS * H * 4 * 4,
            MAX_ROWS * H * 4,
            MAX_ROWS * H * 2,
        ];
        Ok(Self {
            device: d,
            prepare_bf16: None,
            full_ids: bank.original_ids.iter().copied().eq(0..512),
            project: project.kernel("mcg_project")?,
            prepare: ops.kernel("mcg_moe_prepare")?,
            middle: ops.kernel("mcg_moe_mid")?,
            finish: ops.kernel("mcg_moe_finish")?,
            r38: if crate::cuda_affine4::r38_enabled() {
                Some([ops.kernel("mcg_moe_mid_ahead")?, ops.kernel("mcg_moe_finish_add")?])
            } else {
                None
            },
            weights,
            input: d.alloc(MAX_ROWS * H * 4)?,
            ids: d.alloc(SLOTS * 4)?,
            scores: d.alloc(SLOTS * 4)?,
            bufs: capacities
                .iter()
                .map(|&n| d.alloc(n + GUARD))
                .collect::<Result<_>>()?,
            experts: bank.original_ids.len(),
            halfwords,
            rows: Cell::new(0),
            grouping: None,
            tensor: None,
            project4: None,
        })
    }
    pub(crate) fn new_full(
        d: &'d Device,
        project: &'m Module<'d>,
        ops: &'m Module<'d>,
        join: &'m Module<'d>,
        bank: &Bank,
    ) -> Result<Self> {
        ensure!(
            bank.original_ids.iter().copied().eq(0..512),
            "original512 ID layout required"
        );
        let mut a = Self::new(d, project, ops, bank)?;
        a.prepare_bf16 = Some(join.kernel("moe_prepare_bf16")?);
        Ok(a)
    }
    pub(crate) fn enable_tensor(&mut self, module: &'m Module<'d>) -> Result<()> {
        ensure!(
            self.full_ids && self.grouping.is_none() && self.halfwords == 28,
            "tensor requires full original K1.75 bank"
        );
        self.tensor = Some(crate::cuda_mcg_tensor::Tensor::new(self.device, module)?);
        Ok(())
    }
    /// R60 S4: route the scalar decode projections through mcg_project4v (bit-identical,
    /// load-ahead) unless QD_MCG_PROJECT_V2=0.
    pub fn enable_project_v2(&mut self, k4: &'m Module<'d>) -> Result<()> {
        if self.halfwords == 64 && std::env::var("QD_MCG_PROJECT_V2").as_deref() != Ok("0") {
            self.project4 = Some([k4.kernel("mcg_project4v")?, k4.kernel("mcg_project4v4")?, k4.kernel("mcg_project4v10")?]);
        }
        Ok(())
    }
    /// R60: the K4 tensor-core kernels (cuda_mcg_tensor::module_k4) for a canonical bank.
    pub fn enable_tensor_k4(&mut self, module: &'m Module<'d>) -> Result<()> {
        ensure!(self.full_ids && self.grouping.is_none() && self.halfwords == 64, "K4 tensor requires the full canonical bank");
        self.tensor = Some(crate::cuda_mcg_tensor::Tensor::new_k4(self.device, module)?);
        Ok(())
    }
    /// R60 probe: like `plan`, through the tensor-core kernels (`arm`), on the uploaded routes.
    pub fn plan_tensor(&self, splits: usize, graph: bool, arm: crate::cuda_mcg_tensor::Arm) -> Result<Plan<'_, 'm, 'd>> {
        let rows = self.rows.get();
        ensure!(rows > 0 && [1, 2, 4].contains(&splits) && self.tensor.is_some(), "tensor plan shape");
        let start = Instant::now();
        let calls = self.record(&self.input, &self.ids, &self.scores, &self.prepare, rows, splits, 0, Some(arm))?;
        let execute = if graph { Execute::Graph(self.device.capture(calls)?) } else { Execute::Calls(calls) };
        Ok(Plan { owner: self, execute, rows, splits, batch: 0, setup_us: start.elapsed().as_secs_f64() * 1e6 })
    }
    pub(crate) fn tensor_check(
        &self,
        ids: &[u32],
        arm: crate::cuda_mcg_tensor::Arm,
    ) -> Result<usize> {
        self.tensor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tensor module absent"))?
            .check(ids, arm)
    }
    /// R38 routed path: prepare, gate+up in one MMA launch, load-ahead mid,
    /// down MMA, and finish that also adds the shared-expert output.
    /// R38 island: packed weights [0..9) and work buffers [0..8).
    pub(crate) fn island_parts(&self) -> (&[Buffer<'d>], &[Buffer<'d>]) {
        (&self.weights, &self.bufs)
    }
    pub(crate) fn record_routed_fused<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        routes: crate::cuda_router::DeviceRoutes<'a, 'd>,
        splits: usize,
        shared: &'a Buffer<'d>,
        added: &'a Buffer<'d>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            self.full_ids && self.experts == 512 && self.grouping.is_none(),
            "full route producer requires original512 bank and ungrouped execution"
        );
        let rows = routes.rows;
        ensure!(
            (1..=MAX_ROWS).contains(&rows)
                && [1, 2, 4].contains(&splits)
                && x.byte_len() >= rows * H * 2
                && shared.byte_len() >= rows * H * 2
                && added.byte_len() >= rows * H * 2,
            "R38 routed geometry"
        );
        let [middle, finish] = self
            .r38
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("R38 routed kernels not loaded"))?;
        let prepare = self
            .prepare_bf16
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("BF16 prepare absent"))?;
        let t = self
            .tensor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tensor module absent"))?;
        let (ids, scores) = (routes.ids, routes.scores);
        let slots = rows * TOPK;
        let w = &self.weights;
        let b = &self.bufs;
        let mut calls = vec![];
        unsafe {
            calls.push(prepare.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&w[1]),
                    Arg::Buffer(&w[4]),
                    Arg::Buffer(ids),
                    Arg::Buffer(&b[0]),
                    Arg::Buffer(&b[1]),
                ],
                [(H / 128) as u32, slots as u32, 1],
                [32, 1, 1],
            )?);
            calls.push(t.project_pair(ids, &b[0], &w[0], &b[2], &b[1], &w[3], &b[3], H, I, slots, splits)?);
            calls.push(middle.prepare(
                vec![
                    Arg::Buffer(&b[2]),
                    Arg::Buffer(&b[3]),
                    Arg::Buffer(&w[2]),
                    Arg::Buffer(&w[5]),
                    Arg::Buffer(&w[7]),
                    Arg::Buffer(ids),
                    Arg::Buffer(&b[4]),
                    Arg::U32(splits as u32),
                ],
                [(I / 128) as u32, slots as u32, 1],
                [32, 1, 1],
            )?);
            calls.push(t.project(ids, &b[4], &w[6], &b[5], I, H, slots, splits, crate::cuda_mcg_tensor::Arm::Mma)?);
            calls.push(finish.prepare(
                vec![
                    Arg::Buffer(&b[5]),
                    Arg::Buffer(&w[8]),
                    Arg::Buffer(ids),
                    Arg::Buffer(scores),
                    Arg::Buffer(&b[6]),
                    Arg::Buffer(&b[7]),
                    Arg::U32(splits as u32),
                    Arg::Buffer(shared),
                    Arg::Buffer(added),
                ],
                [(H / 128) as u32, rows as u32, 1],
                [320, 1, 1],
            )?);
        }
        Ok(calls)
    }
    pub(crate) fn record_routed<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        routes: crate::cuda_router::DeviceRoutes<'a, 'd>,
        splits: usize,
        bf16: bool,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            self.full_ids && self.experts == 512 && self.grouping.is_none(),
            "full route producer requires original512 bank and ungrouped execution"
        );
        ensure!(
            x.byte_len() >= routes.rows * H * if bf16 { 2 } else { 4 },
            "expert input capacity"
        );
        let prepare = if bf16 {
            self.prepare_bf16
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("BF16 prepare absent"))?
        } else {
            &self.prepare
        };
        self.record(
            x,
            routes.ids,
            routes.scores,
            prepare,
            routes.rows,
            splits,
            0,
            tensor_arm,
        )
    }
    pub(crate) fn output(&self) -> &Buffer<'d> {
        &self.bufs[7]
    }
    pub(crate) fn poison(&self) -> Result<()> {
        if let Some(t) = &self.tensor {
            t.poison()?;
        }
        for b in &self.bufs {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        Ok(())
    }
    pub fn new_grouped(
        d: &'d Device,
        project: &'m Module<'d>,
        ops: &'m Module<'d>,
        grouped: &'m Module<'d>,
        bank: &Bank,
    ) -> Result<Self> {
        let mut this = Self::new(d, project, ops, bank)?;
        ensure!(this.halfwords == 28, "grouped MCG kernels are K1.75 only");
        this.grouping = Some(Grouped::new(d, grouped, bank.original_ids.len())?);
        Ok(this)
    }
    pub fn description(&self) -> Value {
        json!({"backend":"native Rust/CUDA/libtinynv","project_cubin_sha256":PROJECT_SHA,
            "ops_cubin_sha256":OPS_SHA,"dependent_calls_baseline":6,"dependent_calls_grouped":7,"grouped":self.grouping.as_ref().map(|g|g.description()),"tensor":self.tensor.as_ref().map(|t|t.description()),"intermediate_host_transfers":0,
            "weight_readback_verified":true,"experts":self.experts,"max_token_rows":MAX_ROWS,
            "kernels":[self.prepare.description(),self.project.description(),self.middle.description(),self.finish.description()],
            "allocated_bytes":self.weights.iter().chain(&self.bufs).map(|b|b.byte_len()).sum::<usize>()+self.input.byte_len()+self.ids.byte_len()+self.scores.byte_len()})
    }
    pub fn prepare_input(&self, x: &[f32], r: &Routes) -> Result<()> {
        self.rows.set(0);
        r.validate()?;
        ensure!(
            r.experts == self.experts
                && r.rows <= MAX_ROWS
                && x.len() == r.rows * H
                && x.iter().all(|v| v.is_finite()),
            "lowbit input geometry"
        );
        for (b, v) in [
            (&self.input, f32_bytes(x)),
            (&self.ids, u32_bytes(&r.ids)),
            (&self.scores, f32_bytes(&r.scores)),
        ] {
            b.upload(&v)?;
            let mut read = vec![0; v.len()];
            b.download(&mut read)?;
            ensure!(read == v, "lowbit input/route transport");
        }
        self.rows.set(r.rows);
        Ok(())
    }
    pub fn plan(&self, splits: usize, graph: bool) -> Result<Plan<'_, 'm, 'd>> {
        self.plan_arm(splits, graph, 0)
    }
    pub fn plan_arm(&self, splits: usize, graph: bool, batch: usize) -> Result<Plan<'_, 'm, 'd>> {
        ensure!([0, 1, 2, 4].contains(&batch), "grouped batch arm");
        let rows = self.rows.get();
        ensure!(rows > 0 && [1, 2, 4].contains(&splits), "lowbit plan shape");
        let start = Instant::now();
        let calls = self.record(
            &self.input,
            &self.ids,
            &self.scores,
            &self.prepare,
            rows,
            splits,
            batch,
            None,
        )?;
        let execute = if graph {
            Execute::Graph(self.device.capture(calls)?)
        } else {
            Execute::Calls(calls)
        };
        Ok(Plan {
            owner: self,
            execute,
            rows,
            splits,
            batch,
            setup_us: start.elapsed().as_secs_f64() * 1e6,
        })
    }
    fn record<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        ids: &'a Buffer<'d>,
        scores: &'a Buffer<'d>,
        prepare: &'a Kernel<'m, 'd>,
        rows: usize,
        splits: usize,
        batch: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && [1, 2, 4].contains(&splits),
            "expert record geometry"
        );
        let slots = (rows * TOPK) as u32;
        let w = &self.weights;
        let b = &self.bufs;
        let mut calls = vec![];
        if let Some(arm) = tensor_arm {
            ensure!(
                batch == 0 && self.full_ids,
                "mutually exclusive grouping paths"
            );
            let t = self
                .tensor
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("tensor module absent"))?;
            if arm.grouped() {
                calls.push(t.group(ids, slots as usize)?);
            }
        }
        if batch > 0 {
            let g = self
                .grouping
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("grouped module not loaded"))?;
            calls.push(g.record(ids, slots as usize, batch)?);
        }
        // Audited fixed model dimensions. Buffers are disjoint and retained by
        // this plan; route IDs either validated on CPU with completed upload
        // readback, or produced by the qualified original512 router.
        unsafe {
            calls.push(prepare.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(&w[1]),
                    Arg::Buffer(&w[4]),
                    Arg::Buffer(ids),
                    Arg::Buffer(&b[0]),
                    Arg::Buffer(&b[1]),
                ],
                [(H / 128) as u32, slots, 1],
                [32, 1, 1],
            )?);
            for (x, t, y) in [(0, 0, 2), (1, 3, 3)] {
                calls.push(self.record_projection(
                    ids,
                    &b[x],
                    &w[t],
                    &b[y],
                    H,
                    I,
                    slots as usize,
                    splits,
                    batch,
                    tensor_arm,
                )?);
            }
            calls.push(self.middle.prepare(
                vec![
                    Arg::Buffer(&b[2]),
                    Arg::Buffer(&b[3]),
                    Arg::Buffer(&w[2]),
                    Arg::Buffer(&w[5]),
                    Arg::Buffer(&w[7]),
                    Arg::Buffer(ids),
                    Arg::Buffer(&b[4]),
                    Arg::U32(splits as u32),
                ],
                [(I / 128) as u32, slots, 1],
                [32, 1, 1],
            )?);
            calls.push(self.record_projection(
                ids,
                &b[4],
                &w[6],
                &b[5],
                I,
                H,
                slots as usize,
                splits,
                batch,
                tensor_arm,
            )?);
            calls.push(self.finish.prepare(
                vec![
                    Arg::Buffer(&b[5]),
                    Arg::Buffer(&w[8]),
                    Arg::Buffer(ids),
                    Arg::Buffer(scores),
                    Arg::Buffer(&b[6]),
                    Arg::Buffer(&b[7]),
                    Arg::U32(splits as u32),
                ],
                [(H / 128) as u32, rows as u32, 1],
                [320, 1, 1],
            )?);
        }
        Ok(calls)
    }
    fn record_projection<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        x: &'a Buffer<'d>,
        w: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        k: usize,
        n: usize,
        slots: usize,
        splits: usize,
        batch: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        if let Some(arm) = tensor_arm {
            return self
                .tensor
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("tensor module absent"))?
                .project(ids, x, w, y, k, n, slots, splits, arm);
        }
        if batch > 0 {
            return self
                .grouping
                .as_ref()
                .unwrap()
                .project(x, w, y, k, n, slots, splits, batch);
        }
        // Original R16 scalar path, same types/geometry and pinned cubin (R60 S4: or the
        // bit-identical load-ahead K4 kernel when enabled).
        unsafe {
            // steps per warp: (k/16 per split) over four warps; <= 4: the 4-ahead kernel
            let per_warp = (k / 16).div_ceil(splits).div_ceil(4);
            let kernel = match &self.project4 {
                Some([v8, v4, v10]) => {
                    if per_warp <= 4 && std::env::var("QD_MCG_PROJECT_V4").as_deref() != Ok("0") {
                        v4
                    } else if per_warp > 8 && per_warp <= 10 && std::env::var("QD_MCG_PROJECT_V10").as_deref() != Ok("0") {
                        v10
                    } else {
                        v8
                    }
                }
                None => &self.project,
            };
            kernel.prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(w),
                    Arg::Buffer(ids),
                    Arg::Buffer(y),
                    Arg::U32(k as u32),
                    Arg::U32(n as u32),
                    Arg::U32(splits as u32),
                    Arg::U32(self.halfwords as u32),
                ],
                [(n / 16) as u32, slots as u32, splits as u32],
                [128, 1, 1],
            )
        }
    }
    pub(crate) fn collect(&self, rows: usize, splits: usize) -> Result<(Vec<Vec<f32>>, usize)> {
        let slots = rows * TOPK;
        let counts = [
            slots * H,
            slots * H,
            slots * I * splits,
            slots * I * splits,
            slots * I,
            slots * H * splits,
            rows * H,
            rows * H,
        ];
        let mut out = vec![];
        let mut checked = 0;
        for (i, b) in self.bufs.iter().enumerate() {
            let width = if [0, 1, 4, 7].contains(&i) { 2 } else { 4 };
            let active = counts[i] * width;
            let mut raw = vec![0; b.byte_len()];
            b.download(&mut raw)?;
            ensure!(
                raw[active..].iter().all(|&v| v == 0xff),
                "lowbit {} canary",
                NAMES[i]
            );
            checked += raw.len() - active;
            let values: Vec<f32> = if width == 2 {
                raw[..active]
                    .chunks_exact(2)
                    .map(|v| {
                        let bits = u16::from_le_bytes(v.try_into().unwrap());
                        if i == 7 {
                            bf16::from_bits(bits).to_f32()
                        } else {
                            f16::from_bits(bits).to_f32()
                        }
                    })
                    .collect()
            } else {
                raw[..active]
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect()
            };
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "lowbit {} nonfinite/unwritten",
                NAMES[i]
            );
            if [2, 3, 5].contains(&i) {
                let dim = if i == 5 { H } else { I };
                let mut summed = vec![0.; slots * dim];
                for r in 0..slots {
                    for s in 0..splits {
                        for c in 0..dim {
                            summed[r * dim + c] += values[(r * splits + s) * dim + c];
                        }
                    }
                }
                out.push(summed);
            } else {
                out.push(values);
            }
        }
        Ok((out, checked))
    }
}
enum Execute<'a, 'm, 'd> {
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
    Graph(Graph<'a, 'm, 'd>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Moe<'m, 'd>,
    execute: Execute<'a, 'm, 'd>,
    rows: usize,
    splits: usize,
    batch: usize,
    pub setup_us: f64,
}
pub struct Run {
    pub stages: Vec<Vec<f32>>,
    pub host_us: f64,
    pub enqueue_us: f64,
    pub completion_us: f64,
    pub canary_bytes: usize,
    pub groups_checked: usize,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, repeats: usize, wait_each: bool) -> Result<Run> {
        ensure!(
            self.rows == self.owner.rows.get() && (1..=16).contains(&repeats),
            "lowbit plan/repetitions"
        );
        ensure!(
            !wait_each || matches!(self.execute, Execute::Calls(_)),
            "graph wait-each"
        );
        let m = self.owner;
        for b in &m.bufs {
            b.upload(&vec![0xff; b.byte_len()])?;
        }
        if self.batch > 0 {
            m.grouping.as_ref().unwrap().poison()?;
        }
        m.device.sync()?;
        let start = Instant::now();
        let mut enqueue_us = 0.;
        let mut completion_us = 0.;
        for _ in 0..repeats {
            match &self.execute {
                Execute::Graph(g) => {
                    let t = Instant::now();
                    g.launch()?;
                    enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                }
                Execute::Calls(calls) => {
                    for c in calls {
                        let t = Instant::now();
                        c.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            m.device.sync()?;
                            completion_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    }
                }
            }
        }
        let t = Instant::now();
        m.device.sync()?;
        completion_us += t.elapsed().as_secs_f64() * 1e6;
        let host_us = start.elapsed().as_secs_f64() * 1e6;
        let (stages, canary_bytes) = m.collect(self.rows, self.splits)?;
        let groups_checked = if self.batch > 0 {
            m.grouping
                .as_ref()
                .unwrap()
                .check(&m.ids, self.rows * TOPK, self.batch)?
        } else {
            0
        };
        Ok(Run {
            stages,
            host_us,
            enqueue_us,
            completion_us,
            canary_bytes,
            groups_checked,
        })
    }
}

/// R60 S4 (prefill v2): the expert-grouped K4 kernels (kernels/mcg4_tensor.cu).
pub struct K64<'m, 'd> {
    group: Kernel<'m, 'd>,
    mma: Kernel<'m, 'd>,
    fast: bool,
    one: bool,
}
impl<'m, 'd> K64<'m, 'd> {
    pub fn new(k4: &'m Module<'d>) -> Result<Self> {
        // parallel grouping unless QD_K64_FAST=0. Projection: QD_K64_KERNEL = s (mcg_mma64s, branchy
        // two-tile), t (one tile per warp), u (tile count fixed per task) or v (u, software-pipelined:
        // the default, ~3x s at 2K-row chunks, bit-identical to s and u); QD_K64_ONE_TILE=1 = t.
        let fast = std::env::var("QD_K64_FAST").as_deref() != Ok("0");
        let pick = std::env::var("QD_K64_KERNEL").unwrap_or_else(|_| {
            if std::env::var("QD_K64_ONE_TILE").as_deref() == Ok("1") { "t".into() } else { "v".into() }
        });
        let name = match pick.as_str() {
            "s" => "mcg_mma64s",
            "t" => "mcg_mma64t",
            "u" => "mcg_mma64u",
            "v" => "mcg_mma64v",
            other => anyhow::bail!("QD_K64_KERNEL={other}: expected s, t, u or v"),
        };
        Ok(Self {
            group: k4.kernel(if fast { "mcg_group512_64f" } else { "mcg_group512_64" })?,
            mma: k4.kernel(name)?,
            fast,
            one: pick == "t",
        })
    }
}
/// R60 S4 (prefill v2): routed-expert intermediates for chunk-sized row counts (slots = rows x 10).
pub struct RoutedScratch<'d> {
    rows: usize,
    b: Vec<Buffer<'d>>,
    desc: Buffer<'d>,
    count: Buffer<'d>,
}
impl<'d> RoutedScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        let slots = rows * TOPK;
        let sizes = [slots * H * 2, slots * H * 2, slots * I * 4, slots * I * 4, slots * I * 2, slots * H * 4, rows * H * 4, rows * H * 2];
        Ok(Self {
            rows,
            b: sizes.iter().map(|&n| d.alloc(n + GUARD)).collect::<Result<_>>()?,
            desc: d.alloc((512 + slots / 64 + 1) * 66 * 4 + GUARD)?,
            count: d.alloc(16 + GUARD)?,
        })
    }
    /// BF16 [rows, H] routed output of the last recorded prefill.
    pub fn output(&self) -> &Buffer<'d> {
        &self.b[7]
    }
}
impl<'m, 'd> Moe<'m, 'd> {
    /// R60 S4 (prefill v2): the routed experts for chunk rows: per-slot prepare, one 64-slot
    /// expert grouping for all three projections (mcg_mma64s, inputs per slot), mid, finish.
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a RoutedScratch<'d>,
        k: &'a K64<'m, 'd>,
        x: &'a Buffer<'d>,
        ids: &'a Buffer<'d>,
        scores: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows && x.byte_len() >= rows * H * 2, "routed prefill shape");
        ensure!(self.full_ids && self.experts == 512 && self.halfwords == 64, "routed prefill needs the full canonical K4 bank");
        let prepare = self.prepare_bf16.as_ref().ok_or_else(|| anyhow::anyhow!("BF16 prepare absent"))?;
        let slots = rows * TOPK;
        let (w, b) = (&self.weights, &s.b);
        let groups_max = 512 + slots / 64 + 1;
        let tiles_per_block = if k.one { 4 } else { 8 };
        // mcg_mma64u/v: whole 8-step chunks and whole 8-tile parts (the canonical H x I banks)
        ensure!(H % 128 == 0 && I % 128 == 0 && (H / 16) % 8 == 0 && (I / 16) % 8 == 0, "K4 prefill shapes");
        let grid = |n: usize| ((170 * 8) as usize).min(groups_max * (n / 16).div_ceil(tiles_per_block)) as u32;
        let project = |xin: &'a Buffer<'d>, wt: &'a Buffer<'d>, y: &'a Buffer<'d>, kk: usize, n: usize| -> Result<PreparedCall<'a, 'm, 'd>> {
            // SAFETY: kernels/mcg4_tensor.cu mcg_mma64s ABI; buffers sized for s.rows.
            unsafe {
                k.mma.prepare(
                    vec![Arg::Buffer(xin), Arg::Buffer(wt), Arg::Buffer(&s.desc), Arg::Buffer(&s.count), Arg::Buffer(y),
                        Arg::U32(kk as u32), Arg::U32(n as u32), Arg::U32(1), Arg::U32(1)],
                    [grid(n), 1, 1],
                    [128, 1, 1],
                )
            }
        };
        // SAFETY: the decode path's kernel ABIs (kernels/moe_join.cu, mcg_moe.cu, mcg4_tensor.cu).
        Ok(vec![
            unsafe { prepare.prepare(vec![Arg::Buffer(x), Arg::Buffer(&w[1]), Arg::Buffer(&w[4]), Arg::Buffer(ids), Arg::Buffer(&b[0]), Arg::Buffer(&b[1])], [(H / 128) as u32, slots as u32, 1], [32, 1, 1])? },
            unsafe { k.group.prepare(vec![Arg::Buffer(ids), Arg::Buffer(&s.desc), Arg::Buffer(&s.count), Arg::U32(slots as u32), Arg::U32(512)], [1, 1, 1], [if k.fast { 1024 } else { 512 }, 1, 1])? },
            project(&b[0], &w[0], &b[2], H, I)?,
            project(&b[1], &w[3], &b[3], H, I)?,
            unsafe {
                self.middle.prepare(
                    vec![Arg::Buffer(&b[2]), Arg::Buffer(&b[3]), Arg::Buffer(&w[2]), Arg::Buffer(&w[5]), Arg::Buffer(&w[7]), Arg::Buffer(ids), Arg::Buffer(&b[4]), Arg::U32(1)],
                    [(I / 128) as u32, slots as u32, 1],
                    [32, 1, 1],
                )?
            },
            project(&b[4], &w[6], &b[5], I, H)?,
            unsafe {
                self.finish.prepare(
                    vec![Arg::Buffer(&b[5]), Arg::Buffer(&w[8]), Arg::Buffer(ids), Arg::Buffer(scores), Arg::Buffer(&b[6]), Arg::Buffer(&b[7]), Arg::U32(1)],
                    [(H / 128) as u32, rows as u32, 1],
                    [320, 1, 1],
                )?
            },
        ])
    }
}
