//! Device-owned expert grouping plus row-shared MCG decoding. No host route plan.
use crate::cuda::{Arg, Buffer, Device, Kernel, Module, PreparedCall};
use crate::moe::{H, I, TOPK};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub const SHA: &str = "e30ed1fd78216d3ab10736b75539f826b3ea995c5ba8cdd206e49c0ef22b8043";
const SLOTS: usize = 16 * TOPK;
const GUARD: usize = 256;
const GRID_CAP: usize = 65536;
pub fn module(d: &Device) -> Result<Module<'_>> {
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/mcg_grouped.sm120.cubin"),
        SHA,
    )
}
pub struct Grouped<'m, 'd> {
    grouping: Kernel<'m, 'd>,
    project: [Kernel<'m, 'd>; 3],
    desc: Buffer<'d>,
    count: Buffer<'d>,
    experts: usize,
}
impl<'m, 'd> Grouped<'m, 'd> {
    pub fn new(d: &'d Device, module: &'m Module<'d>, experts: usize) -> Result<Self> {
        ensure!((TOPK..=128).contains(&experts), "grouping expert bound");
        Ok(Self {
            grouping: module.kernel("mcg_group_rows")?,
            project: [
                module.kernel("mcg_project_grouped_b1")?,
                module.kernel("mcg_project_grouped_b2")?,
                module.kernel("mcg_project_grouped_b4")?,
            ],
            desc: d.alloc(SLOTS * 6 * 4 + GUARD)?,
            count: d.alloc(4 + GUARD)?,
            experts,
        })
    }
    pub fn description(&self) -> Value {
        json!({"cubin_sha256":SHA,"route_plan":"GPU only, one128-thread block, no atomics",
            "descriptor":"expert,count,slot0..3, unused slots=u32::MAX",
            "maximum_experts":128,"maximum_slots":SLOTS,"batch_arms":[1,2,4],
            "project_grid_cap":GRID_CAP,"grouping_kernel":self.grouping.description(),"project_kernels":self.project.iter().map(|k|k.description()).collect::<Vec<_>>(),"width_specialized":true,
            "allocated_bytes":self.desc.byte_len()+self.count.byte_len()})
    }
    pub fn record<'a>(
        &'a self,
        ids: &'a Buffer<'d>,
        slots: usize,
        batch: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            slots > 0
                && slots <= SLOTS
                && ids.byte_len() >= slots * 4
                && [1, 2, 4].contains(&batch),
            "grouping shape"
        );
        // Single block always writes128 shared counters. Owned descriptors hold
        // at most one group per input slot. Unique top10 routes bound reuse<=16.
        unsafe {
            self.grouping.prepare(
                vec![
                    Arg::Buffer(ids),
                    Arg::Buffer(&self.desc),
                    Arg::Buffer(&self.count),
                    Arg::U32(slots as u32),
                    Arg::U32(self.experts as u32),
                    Arg::U32(batch as u32),
                ],
                [1, 1, 1],
                [128, 1, 1],
            )
        }
    }
    pub fn project<'a>(
        &'a self,
        x: &'a Buffer<'d>,
        weights: &'a Buffer<'d>,
        y: &'a Buffer<'d>,
        k: usize,
        n: usize,
        slots: usize,
        splits: usize,
        batch: usize,
    ) -> Result<PreparedCall<'a, 'm, 'd>> {
        ensure!(
            [(H, I), (I, H)].contains(&(k, n))
                && slots > 0
                && slots <= SLOTS
                && [1, 2, 4].contains(&splits)
                && [1, 2, 4].contains(&batch),
            "grouped projection geometry"
        );
        ensure!(
            x.byte_len() >= slots * k * 2
                && y.byte_len() >= slots * n * splits * 4
                && weights.byte_len() == self.experts * k * n / 256 * 56
                && !x.aliases(y)
                && !weights.aliases(y),
            "grouped projection buffers"
        );
        let blocks = (slots * (n / 16) * splits).min(GRID_CAP) as u32;
        // Pinned kernel consumes only the device-generated active descriptor
        // count, loops over a bounded persistent grid, writes original slots.
        unsafe {
            self.project[match batch {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => unreachable!(),
            }]
            .prepare(
                vec![
                    Arg::Buffer(x),
                    Arg::Buffer(weights),
                    Arg::Buffer(&self.desc),
                    Arg::Buffer(&self.count),
                    Arg::Buffer(y),
                    Arg::U32(k as u32),
                    Arg::U32(n as u32),
                    Arg::U32(splits as u32),
                    Arg::U32(batch as u32),
                ],
                [blocks, 1, 1],
                [128, 1, 1],
            )
        }
    }
    pub fn poison(&self) -> Result<()> {
        self.desc.upload(&vec![0xff; self.desc.byte_len()])?;
        self.count.upload(&vec![0xff; self.count.byte_len()])
    }
    pub fn check(&self, ids: &Buffer<'d>, slots: usize, batch: usize) -> Result<usize> {
        let mut count = vec![0; self.count.byte_len()];
        self.count.download(&mut count)?;
        ensure!(count[4..].iter().all(|&v| v == 0xff), "group count canary");
        let groups = u32::from_le_bytes(count[..4].try_into().unwrap()) as usize;
        ensure!(groups > 0 && groups <= slots, "group count invalid");
        let mut desc = vec![0; self.desc.byte_len()];
        self.desc.download(&mut desc)?;
        ensure!(
            desc[groups * 24..].iter().all(|&v| v == 0xff),
            "group descriptor canary"
        );
        let decode = |raw: &[u8]| {
            raw.chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut rawids = vec![0; slots * 4];
        ids.download(&mut rawids)?;
        let routes = decode(&rawids);
        let mut expected = vec![];
        // Independent host grouping is a post-timing oracle only. Device
        // execution never consumes this list or a host-reported group count.
        for e in 0..self.experts {
            let list: Vec<u32> = routes
                .iter()
                .enumerate()
                .filter_map(|(s, &id)| (id == e as u32).then_some(s as u32))
                .collect();
            for chunk in list.chunks(batch) {
                expected.extend([e as u32, chunk.len() as u32]);
                expected.extend_from_slice(chunk);
                expected.extend(std::iter::repeat_n(u32::MAX, 4 - chunk.len()));
            }
        }
        ensure!(
            decode(&desc[..groups * 24]) == expected,
            "GPU grouping/slot ownership mismatch"
        );
        Ok(groups)
    }
}
