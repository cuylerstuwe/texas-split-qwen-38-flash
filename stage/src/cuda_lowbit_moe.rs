//! R20b: router -> original512 packed experts + shared branch -> BF16 sum.
//! One device stream owns all dependencies. Readbacks are diagnostics after
//! completion, never a route/addressing handoff on the execution critical path.
use crate::{
    bf16_linear::Matrix,
    cuda::{Arg, Buffer, Device, Graph, Kernel, Module, PreparedCall},
    cuda_mcg_moe::Moe,
    cuda_router::{self, Projection, Router},
    cuda_shared::{self, Policy, Shared},
    lowbit_router::{H, MAX_ROWS, Scores},
    lowbit_shared::Weights,
    mcg_moe::Bank,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{cell::Cell, path::Path, time::Instant};
pub const SHA: &str = "e1d186354ed32797726ebc1f146042b412be2dcb7571c03df8fc2c5da6f6ff52";
const GUARD: usize = 256;
/// R38 (Opus, direct): unchanged moe_join.cu plus the one-launch MoE island
/// (moe_island.cu). Loaded only with QWEN_AFFINE4_R38=1.
pub const R38_SHA: &str = "c192cd5a608d0836bfc6e152d8611598319f7405b12822bfc9e737fbcf4829ab";
const ISLAND_WORDS: usize = 256;
fn r39_moe_enabled() -> bool {
    crate::cuda_affine4::r39_enabled()
        && std::env::var("QWEN_R39_MOE_PDL").as_deref() == Ok("1")
}
/// R52 (Opus, direct): one-launch MoE back half (kernels/moe_r52.cu). Opt-in with
/// QWEN_R52_MOE=g4d4|g2d4 on top of the R39 MoE module; the candidate (PDL) arm
/// uses r39pdl F0 + the R52 launch. QWEN_R52_BASE_PDL=1 gives the baseline arm
/// the R39 PDL MoE path, so the A/B isolates R52 against the current best.
fn r52_variant() -> Option<&'static str> {
    if !r39_moe_enabled() { return None; }
    let trace = std::env::var("QWEN_R52_TRACE").as_deref() == Ok("1");
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2p") && !trace {
        return Some("moe_r52_back_g4_vpre");
    }
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2e") && !trace {
        return Some("r39pdl_moe_r52_back_g4_v");
    }
    // R53: R52's g4_v back half launched with gate/up items only (800 blocks); the
    // shared expert runs in the router launch (r39pdl_moe_r53f0s). Loads moe_join_r53.
    if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2t" | "g4w2q")) && !trace {
        return Some("r39pdl_moe_r52_back_g4_v");
    }
    // R53: g4w2t whose gate/up launch starts on the selection flag (overlap form).
    // g4w2q/g4w2v: g4w2t/g4w2u whose router blocks preload the shared down weights (F0q).
    if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2u" | "g4w2v")) && !trace {
        return Some("r39pdl_moe_r53_back_ovl");
    }
    // R53: trace twins of g4w2v (F0q, overlap gate/up, ds2i) for the in-model timeline.
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2v") && trace {
        return Some("r39pdl_moe_r53_trace_ovl");
    }
    // R53: g4w2e with the SRED and final-sum load-order fixes (r53_back_fx, ds2x).
    if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2x" | "g4w2w")) {
        return Some(if trace { "r39pdl_moe_r53_trace_fx" } else { "r39pdl_moe_r53_back_fx" });
    }
    // R53: g4w2x whose shared items request their shared-down weights first (y), and
    // the same on the trimmed one-wave back half (z).
    // R53: g4w2m with the SRED and final-sum load-order fixes.
    // g4w2a: g4w2n's back half with the parallel-poll down launch (ds2ma).
    // g4w2h: the same back half with the tagged-tail down launch that leaves its counter
    // block to the layer's stream write (r39pdl_hc_write_z). Read
    // research/R53-G4W2H-SAFETY.txt before using it: guarded, but it must run through layers.
    // g4w2g: the same back half with the tagged-tail down launch (ds2mg).
    if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2n" | "g4w2a" | "g4w2g" | "g4w2h")) {
        return Some(if trace { "r39pdl_moe_r53_trace_g4_vmx" } else { "r39pdl_moe_r53_back_g4_vmx" });
    }
    // R53: g4w2n whose shared items scale and publish their own rows of the shared
    // output (no serial shared finish); the final sums wait on the items' count.
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2r") {
        return Some(if trace { "r39pdl_moe_r53_trace_g4_vmr" } else { "r39pdl_moe_r53_back_g4_vmr" });
    }
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2y") {
        return Some(if trace { "r39pdl_moe_r53_trace_fy" } else { "r39pdl_moe_r53_back_fy" });
    }
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2z") {
        return Some(if trace { "r39pdl_moe_r53_trace_fz" } else { "r39pdl_moe_r53_back_fz" });
    }
    // R53: g4w2e with the deep gate/up item (QWEN_R53_GD = tiles of weight requests ahead).
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2d") && !trace {
        return Some(match std::env::var("QWEN_R53_GD").as_deref() {
            Ok("2") => "r39pdl_moe_r53_back_gd2", Ok("4") => "r39pdl_moe_r53_back_gd4",
            Ok("6") => "r39pdl_moe_r53_back_gd6", Ok("10") => "r39pdl_moe_r53_back_gd10",
            _ => "r39pdl_moe_r53_back_gd3",
        });
    }
    // R53: early-launch trace twin of g4w2e's back half (in-model block timeline).
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2e") && trace {
        return Some("r39pdl_moe_r53_trace_g4_v");
    }
    // R53: g4w2e without the gate/up mid tails; the down launch computes the
    // activation pieces it reads (r53_down_split2m). Loads moe_join_r53.
    if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2m" | "g4w2s" | "g4w2f")) {
        return Some(if trace { "moe_r53_trace_g4_vm" } else { "r39pdl_moe_r53_back_g4_vm" });
    }
    // R53: g4w2e with the back half sized for six blocks per SM (one wave).
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2o") {
        return Some(if trace { "moe_r53_trace_g4_vo" } else { "r39pdl_moe_r53_back_g4_vo" });
    }
    if std::env::var("QWEN_R52_MOE").as_deref() == Ok("ovl") {
        return Some(if trace { "r39pdl_moe_r52ot_g4_d4" } else { "r39pdl_moe_r52o_g4_d4" });
    }
    let name = match std::env::var("QWEN_R52_MOE").as_deref() {
        Ok("g4d4") => "g4_d4",
        Ok("g4d8") => "g4_d8",
        Ok("g4d4p") => "g4_d4p",
        Ok("g4d8p") => "g4_d8p",
        Ok("g4d4s") => "g4_d4s",
        Ok("g4d4y") => "g4_d4",
        Ok("g4d4t") => "g4_d4t",
        Ok("g4d4u") => "g4_d4u",
        Ok("g4v") => "g4_v",
        Ok("g4w") => "g4_v",
        Ok("g4wl") => "g4_v",
        Ok("g4w2") => "g4_v",
        _ => return None,
    };
    let full = format!("moe_r52_{}_{name}", if trace { "trace" } else { "back" });
    Some(Box::leak(full.into_boxed_str()))
}
/// R53: QWEN_R53_DOWNTRIG=entry|main|exit selects the down launch's trigger twin
/// (*_trig: it decides when its successor launches; driver bit 128 via QWEN_R53_TRIG=1).
fn r53_down_trig(base: &str) -> String {
    match std::env::var("QWEN_R53_DOWNTRIG").as_deref() {
        Ok(p @ ("entry" | "main" | "exit")) => format!("{base}_{p}_trig"),
        _ => base.to_string(),
    }
}
fn r52_tiles() -> (u32, u32) {
    match std::env::var("QWEN_R52_MOE").as_deref() { Ok("g4d8" | "g4d8p") => (4, 8), _ => (4, 4) }
}
/// R52: run the R52 MoE path in BOTH A/B arms (to isolate a different change).
fn r52_both() -> bool {
    r52_variant().is_some() && std::env::var("QWEN_R52_BOTH").as_deref() == Ok("1")
}
fn r52_base_pdl() -> bool {
    r52_variant().is_some() && std::env::var("QWEN_R52_BASE_PDL").as_deref() == Ok("1")
}
fn r53_moe() -> bool {
    r52_variant().is_some() && (matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2o" | "g4w2m" | "g4w2s" | "g4w2f"))
        || r52_variant() == Some("r39pdl_moe_r53_trace_g4_v")
        || matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2t" | "g4w2u" | "g4w2q" | "g4w2v" | "g4w2d" | "g4w2x" | "g4w2y" | "g4w2z" | "g4w2w" | "g4w2n" | "g4w2r" | "g4w2a" | "g4w2g" | "g4w2h")))
}
pub fn module(d: &Device) -> Result<Module<'_>> {
    // R53a: the same R52/R53 MoE source with acquire/release fences in place of
    // sequentially consistent ones (moe_join_r53a); kernels and values unchanged.
    if (r53_moe() || r52_variant().is_some()) && std::env::var("QWEN_R53_ACQREL").as_deref() == Ok("1") {
        ensure!(crate::cuda_affine4::r38_enabled(), "R53a MoE requires R38");
        let pin: Value = serde_json::from_str(include_str!("../pins/R53A-MOE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join_r53a.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r53_moe() {
        ensure!(crate::cuda_affine4::r38_enabled(), "R53 MoE requires R38");
        let pin: Value = serde_json::from_str(include_str!("../pins/R53-MOE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join_r53.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r52_variant().is_some() {
        ensure!(crate::cuda_affine4::r38_enabled(), "R52 MoE requires R38");
        let pin: Value = serde_json::from_str(include_str!("../pins/R52-MOE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join_r52.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if r39_moe_enabled() {
        ensure!(crate::cuda_affine4::r38_enabled(), "R39 MoE requires R38");
        let pin: Value = serde_json::from_str(include_str!("../pins/R39-MOE-CUDA-BUILD.json"))?;
        return d.module(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join_r39.sm120.cubin"),
            pin["cubin_sha256"].as_str().unwrap());
    }
    if crate::cuda_affine4::r38_enabled() {
        return d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join_r38.sm120.cubin"),
            R38_SHA,
        );
    }
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/moe_join.sm120.cubin"),
        SHA,
    )
}
fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
pub struct Engine<'m, 'd> {
    d: &'d Device,
    router: Router<'m, 'd>,
    routed: Moe<'m, 'd>,
    shared: Shared<'m, 'd>,
    add: Kernel<'m, 'd>,
    island: Option<(Kernel<'m, 'd>, Buffer<'d>, Buffer<'d>, u32)>,
    island_last_stage: Cell<u32>,
    staged: Option<Vec<Kernel<'m, 'd>>>,
    pdl_staged: Option<Vec<Kernel<'m, 'd>>>,
    r52: Option<Kernel<'m, 'd>>,
    /// R52 overlap: F0 clone that releases route/activation flags.
    r52_f0: Option<Kernel<'m, 'd>>,
    /// R52 split down kernel (variant g4v), early-launch eligible.
    r52_down: Option<Kernel<'m, 'd>>,
    /// R52 multi4 pump: pumping entries and the (table, entries, units) to warm.
    r52_pumpk: Option<[Kernel<'m, 'd>; 2]>,
    r52_pump: std::cell::OnceCell<(Buffer<'d>, u32, u32)>,
    r52_trace: Option<Buffer<'d>>,
    /// R52 counters (own buffer: 1024 words, zeroed) and FP32 finish scratch.
    r52_bufs: Option<(Buffer<'d>, Buffer<'d>)>,
    /// R53 g4w2g: per-layer tagged words of the down tails (split pairs, then slots).
    r53_tags: Option<Buffer<'d>>,
    x: Buffer<'d>,
    x32: Buffer<'d>,
    output: Buffer<'d>,
    rows: Cell<usize>,
}
impl<'m, 'd> Engine<'m, 'd> {
    /// R60 pipeline split: the canonical target's MLP: BF16 router, canonical EXL3 K4 routed
    /// experts (scalar MCG projection path, halfwords 64) and the affine8 shared expert.
    /// Baseline kernels only: no island, staged, PDL or R52/R53 variants.
    #[allow(clippy::too_many_arguments)]
    pub fn new_canonical(
        d: &'d Device,
        router_ops: &'m Module<'d>,
        tensor: &'m Module<'d>,
        project: &'m Module<'d>,
        moe_ops: &'m Module<'d>,
        m8: &'m Module<'d>,
        shared_ops: &'m Module<'d>,
        join: &'m Module<'d>,
        k4: &'m Module<'d>,
        w: &crate::target_mlp::Weights,
        bank: &Bank,
    ) -> Result<Self> {
        ensure!(bank.gate.raw.halfwords == 64 && bank.original_ids.len() == 512, "canonical full K4 bank");
        let mut routed = Moe::new_full(d, project, moe_ops, join, bank)?;
        routed.enable_project_v2(k4)?;
        Ok(Self {
            d,
            router: Router::new(d, router_ops, tensor, &w.router)?,
            routed,
            shared: Shared::new_canonical(d, m8, shared_ops, w)?,
            add: join.kernel("moe_add_bf16")?,
            island: None,
            island_last_stage: Cell::new(6),
            staged: None,
            pdl_staged: None,
            r52: None,
            r52_f0: None,
            r52_down: None,
            r52_pumpk: None,
            r52_pump: std::cell::OnceCell::new(),
            r52_trace: None,
            r52_bufs: None,
            r53_tags: None,
            x: d.alloc(MAX_ROWS * H * 2)?,
            x32: d.alloc(MAX_ROWS * H * 4)?,
            output: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            rows: Cell::new(0),
        })
    }
    pub fn new(
        d: &'d Device,
        router_ops: &'m Module<'d>,
        tensor: &'m Module<'d>,
        project: &'m Module<'d>,
        moe_ops: &'m Module<'d>,
        affine: &'m Module<'d>,
        split: &'m Module<'d>,
        shared_ops: &'m Module<'d>,
        join: &'m Module<'d>,
        router: &Matrix,
        bank: &Bank,
        shared: &Weights,
    ) -> Result<Self> {
        Ok(Self {
            d,
            router: Router::new(d, router_ops, tensor, router)?,
            routed: Moe::new_full(d, project, moe_ops, join, bank)?,
            shared: Shared::new(d, affine, split, shared_ops, shared)?,
            add: join.kernel("moe_add_bf16")?,
            island: if crate::cuda_affine4::r38_enabled() {
                let k = join.kernel("moe_island_r38")?;
                let (regs, smem) = k.resources();
                let by_regs = 65536 / (regs.max(1) * 128);
                let by_smem = (99 * 1024) / (smem + 1024);
                let per_sm = by_regs.min(by_smem).min(16).max(1);
                // Every block must be resident. Default: 2 blocks on each of
                // 160 SMs (5090 has 170); QWEN_R38_ISLAND_GRID overrides for probes.
                let grid = std::env::var("QWEN_R38_ISLAND_GRID")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or((160 * per_sm.min(2)) as u32)
                    .clamp(32, 1024);
                let counters = d.alloc(ISLAND_WORDS * 4)?;
                counters.upload(&vec![0u8; ISLAND_WORDS * 4])?;
                let status = d.alloc(4 + GUARD)?;
                status.upload(&vec![0u8; 4 + GUARD])?;
                Some((k, counters, status, grid))
            } else {
                None
            },
            island_last_stage: Cell::new(6),
            staged: if crate::cuda_affine4::r38_enabled() {
                // 0..=6: the island's stages; 7..=9: stage 3 with 2/4/8
                // output tiles per block; 10..=12: stage 5 likewise; 13..=16:
                // the fused-tail launches F0, F1, F2 (M=2), F2 (M=4).
                Some(
                    (0..7)
                        .map(|k| format!("moe_stage{k}_r38"))
                        .chain([3, 5].into_iter().flat_map(|s| {
                            [2, 4, 8].into_iter().map(move |m| format!("moe_stage{s}_m{m}_r38"))
                        }))
                        .chain(["moe_f0_r38", "moe_f1_r38", "moe_f2_m2_r38", "moe_f2_m4_r38"].map(String::from))
                        // 17..=20: deep-load stage 3 (M2 D10, M4 D5), stage 5 (M4 D3, M8 D3).
                        .chain(
                            ["moe_stage3_deep2_r38", "moe_stage3_deep4_r38", "moe_stage5_deep4_r38", "moe_stage5_deep8_r38"]
                                .map(String::from),
                        )
                        // 21: F0 with the one-trip router projection and tree selection.
                        .chain(["moe_f0b_r38".to_string()])
                        .map(|name| join.kernel(&name))
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            },
            pdl_staged: if r39_moe_enabled() {
                Some(["moe_f0_r38", "moe_f1_r38", "moe_f2_m2_r38", "moe_stage5_m4_r38", "moe_stage6_r38"]
                    .into_iter().map(|name| join.kernel(&format!("r39pdl_{name}")))
                    .collect::<Result<Vec<_>>>()?)
            } else { None },
            r52: match r52_variant() { Some(name) => Some(join.kernel(name)?), None => None },
            r52_f0: {
                let trace = std::env::var("QWEN_R52_TRACE").as_deref() == Ok("1");
                match r52_variant() {
                    Some("r39pdl_moe_r53_trace_g4_v" | "r39pdl_moe_r53_trace_g4_vmx" | "r39pdl_moe_r53_trace_g4_vmr") if std::env::var("QWEN_R52_F0Z").as_deref() == Ok("1") =>
                        Some(join.kernel("r39pdl_moe_r53_f0zt")?),
                    Some(_) if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2t" | "g4w2u")) => Some(join.kernel("r39pdl_moe_r53f0s")?),
                    // g4w2w: F0z whose shared gate/up blocks prefetch the shared down weights into L2.
                    Some(_) if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2w") && !trace && std::env::var("QWEN_R52_F0Z").as_deref() == Ok("1") =>
                        Some(join.kernel("r39pdl_moe_r53f0zs")?),
                    Some(_) if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2x" | "g4w2y" | "g4w2z" | "g4w2w")) && std::env::var("QWEN_R52_F0Z").as_deref() == Ok("1") =>
                        Some(join.kernel(if trace { "r39pdl_moe_r53_f0zt" } else { "r39pdl_moe_r52f0z" })?),
                    Some(_) if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w2d") && !trace
                        && std::env::var("QWEN_R52_F0Z").as_deref() == Ok("1") => Some(join.kernel("r39pdl_moe_r52f0z")?),
                    Some(_) if matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2q" | "g4w2v")) =>
                        Some(join.kernel(if trace { "r39pdl_moe_r53f0qt" } else { "r39pdl_moe_r53f0q" })?),
                    Some(name) if (name.ends_with("_g4_v") || name.ends_with("_g4_vm") || name.ends_with("_g4_vmx") || name.ends_with("_g4_vmr") || name.ends_with("_g4_vo") || name.ends_with("_g4_vpre")) && !trace
                        && std::env::var("QWEN_R52_F0Z").as_deref() == Ok("1") => Some(join.kernel(
                            if std::env::var("QWEN_R52_SPF").as_deref() == Ok("1") { "r39pdl_moe_r52f0zp" } else { "r39pdl_moe_r52f0z" })?),
                    Some(name) if name.ends_with("_d4s") || name.ends_with("_d4t") || name.ends_with("_d4u") || name.ends_with("_g4_v") || name.ends_with("_g4_vm") || name.ends_with("_g4_vo") || name.ends_with("_g4_vpre") => Some(join.kernel(if trace { "r39pdl_moe_r52f0xt" } else { "r39pdl_moe_r52f0x" })?),
                    Some(_) if std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4d4y") =>
                        Some(join.kernel(if trace { "r39pdl_moe_r52f0yt" } else { "r39pdl_moe_r52f0y" })?),
                    Some(_) if trace => Some(join.kernel("r39pdl_moe_r52f0t")?),
                    Some(name) if name.starts_with("r39pdl_") => Some(join.kernel("r39pdl_moe_r52f0")?),
                    _ => None,
                }
            },
            r53_tags: if r52_variant().is_some() && matches!(std::env::var("QWEN_R52_MOE").as_deref(), Ok("g4w2g" | "g4w2h")) {
                // 10 slots x 20 pieces x 2 splits x 128 + 20 pieces x 10 slots x 128 words of 8 bytes.
                let b = d.alloc(76800 * 8)?;
                b.upload(&vec![0u8; 76800 * 8])?;
                Some(b)
            } else { None },
            r52_bufs: if r52_variant().is_some() {
                let c = d.alloc(4096)?;
                c.upload(&vec![0u8; 4096])?;
                let f = d.alloc(10 * 2560 * 4)?;
                f.upload(&vec![0xffu8; 10 * 2560 * 4])?;
                Some((c, f))
            } else { None },
            r52_pumpk: if std::env::var("QWEN_R52_MPUMP").as_deref() == Ok("1")
                && std::env::var("QWEN_R52_MOE").as_deref() == Ok("g4w") && r52_variant().is_some() {
                Some([join.kernel("moe_r52_back_g4_vp")?, join.kernel("r39pdl_moe_r52dsp")?])
            } else { None },
            r52_pump: std::cell::OnceCell::new(),
            r52_down: {
                let trace = std::env::var("QWEN_R52_TRACE").as_deref() == Ok("1");
                match std::env::var("QWEN_R52_MOE").as_deref() {
                    Ok("g4v") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r52dt" } else { "r39pdl_moe_r52d" })?),
                    Ok("g4w") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r52dst" } else { "r39pdl_moe_r52ds" })?),
                    Ok("g4wl") if r52_variant().is_some() => Some(join.kernel("moe_r52ds_late")?),
                    Ok("g4w2") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r52ds2t" } else { "r39pdl_moe_r52ds2" })?),
                    Ok("g4w2p") if r52_variant().is_some() => Some(join.kernel("r39pdl_moe_r52ds2")?),
                    Ok("g4w2e") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r52ds2t" } else { "r39pdl_moe_r52ds2" })?),
                    Ok("g4w2d") if r52_variant().is_some() => Some(join.kernel("r39pdl_moe_r52ds2")?),
                    Ok("g4w2h") if r52_variant().is_some() => {
                        // Its counters are cleared by the stream write that follows (r39pdl_hc_write_z).
                        ensure!(crate::cuda_hc::r53_hcw_enabled(), "QWEN_R52_MOE=g4w2h needs QWEN_R53_HCW=1 (its stream write clears the MoE counters)");
                        // Only the candidate arm's stream write clears them, so the baseline arm must not run it.
                        ensure!(!r52_both(), "QWEN_R52_MOE=g4w2h is candidate-arm only (QWEN_R52_BOTH would leave the baseline arm's counters uncleared)");
                        Some(join.kernel(if trace { "r39pdl_moe_r53ds2mht" } else { "r39pdl_moe_r53ds2mh" })?)
                    }
                    Ok("g4w2g") if r52_variant().is_some() =>
                        Some(join.kernel(if trace { "r39pdl_moe_r53ds2mgt" } else { "r39pdl_moe_r53ds2mg" })?),
                    Ok("g4w2a") if r52_variant().is_some() =>
                        Some(join.kernel(if trace { "r39pdl_moe_r53ds2mat" } else { "r39pdl_moe_r53ds2ma" })?),
                    Ok("g4w2r") if r52_variant().is_some() =>
                        Some(join.kernel(if trace { "r39pdl_moe_r53ds2mrt" } else { "r39pdl_moe_r53ds2mr" })?),
                    Ok("g4w2n") if r52_variant().is_some() =>
                        Some(join.kernel(&if trace { "r39pdl_moe_r53ds2mxt".to_string() } else { r53_down_trig("r39pdl_moe_r53ds2mx") })?),
                    Ok("g4w2x" | "g4w2y" | "g4w2z" | "g4w2w") if r52_variant().is_some() =>
                        Some(join.kernel(&if trace { "r39pdl_moe_r53ds2xt".to_string() } else { r53_down_trig("r39pdl_moe_r53ds2x") })?),
                    Ok("g4w2o") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r53ds2ot" } else { "r39pdl_moe_r53ds2o" })?),
                    Ok("g4w2t" | "g4w2q") if r52_variant().is_some() => Some(join.kernel("r39pdl_moe_r53ds2o")?),
                    Ok("g4w2u" | "g4w2v") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r53ds2it" } else { "r39pdl_moe_r53ds2i" })?),
                    Ok("g4w2m") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r53ds2mt" } else { "r39pdl_moe_r53ds2m" })?),
                    // R53: g4w2m with the down weights' second half staged in shared memory before the wait.
                    Ok("g4w2f") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r53ds2ft" } else { "r39pdl_moe_r53ds2f" })?),
                    // R53: the shared expert also moves to the down launch; gate/up is 800 blocks.
                    Ok("g4w2s") if r52_variant().is_some() => Some(join.kernel(if trace { "r39pdl_moe_r53ds2st" } else { "r39pdl_moe_r53ds2s" })?),
                    _ => None,
                }
            },
            r52_trace: match r52_variant() {
                Some(name) if name.contains("_trace_") || name.contains("_r52ot_") => {
                    let b = d.alloc(4400 * 6 * 8)?;
                    b.upload(&vec![0u8; 4400 * 6 * 8])?;
                    Some(b)
                }
                _ => None,
            },
            x: d.alloc(MAX_ROWS * H * 2)?,
            x32: d.alloc(MAX_ROWS * H * 4)?,
            output: d.alloc(MAX_ROWS * H * 2 + GUARD)?,
            rows: Cell::new(0),
        })
    }
    /// R38 diagnostics: record with one external input (router, experts and
    /// shared expert all read it, as in the model), collect, poison, stop the
    /// island after a stage, and read the island's timeout status.
    pub fn r38_record_external<'a>(
        &'a self,
        input: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        self.record_inputs(input, input, input, rows, p, router, scores, splits, true, tensor_arm)
    }
    pub fn r38_collect(
        &self,
        rows: usize,
        p: Policy,
        splits: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Run> {
        self.collect(rows, p, splits, tensor_arm)
    }
    pub fn r38_poison(&self, p: Policy) -> Result<()> {
        self.poison(p)
    }
    pub fn r38_set_last_stage(&self, stage: u32) {
        self.island_last_stage.set(stage.min(6));
    }
    pub fn r38_island_status(&self) -> Result<Option<(u32, u32)>> {
        let Some((_, _, status, grid)) = &self.island else { return Ok(None) };
        let mut raw = [0u8; 4];
        let mut full = vec![0u8; status.byte_len()];
        status.download(&mut full)?;
        raw.copy_from_slice(&full[..4]);
        Ok(Some((u32::from_le_bytes(raw), *grid)))
    }
    pub fn enable_tensor(&mut self, module: &'m Module<'d>) -> Result<()> {
        ensure!(
            self.rows.get() == 0,
            "enable experimental kernels before preparing input"
        );
        self.routed.enable_tensor(module)
    }
    pub fn description(&self) -> Value {
        let join_sha = if self.r52.is_some() {
            serde_json::from_str::<Value>(include_str!("../pins/R52-MOE-CUDA-BUILD.json"))
                .unwrap()["cubin_sha256"].as_str().unwrap().to_owned()
        } else if self.pdl_staged.is_some() {
            serde_json::from_str::<Value>(include_str!("../pins/R39-MOE-CUDA-BUILD.json"))
                .unwrap()["cubin_sha256"].as_str().unwrap().to_owned()
        } else if self.staged.is_some() { R38_SHA.to_owned() } else { SHA.to_owned() };
        json!({"join_cubin_sha256":join_sha,"router":self.router.description(),
            "routed":self.routed.description(),"shared":self.shared.description(),
            "add":self.add.description(),"full_original_experts":512,"host_route_handoffs":0,
            "distinct_router_expert_inputs":true,"grouping":"explicit plan arm; no host route handoff",
            "extra_bytes":self.x.byte_len()+self.x32.byte_len()+self.output.byte_len(),
            "scope":"one MoE sublayer; A/B component scratch retained; no HC/state/head/generation"})
    }
    pub fn prepare(&self, router: &[u16], expert: &[u16]) -> Result<()> {
        self.rows.set(0);
        ensure!(
            !expert.is_empty()
                && expert.len() == router.len()
                && expert.len() % H == 0
                && expert.len() / H <= MAX_ROWS
                && expert.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "composed inputs finite/shape"
        );
        self.router.prepare_input(router)?;
        let fp: Vec<_> = expert
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        for (b, raw) in [(&self.x, bytes(expert)), (&self.x32, bytes(&fp))] {
            b.upload(raw)?;
            let mut check = vec![0; raw.len()];
            b.download(&mut check)?;
            ensure!(check == raw, "composed input readback");
        }
        self.rows.set(expert.len() / H);
        Ok(())
    }
    fn record_inputs<'a>(
        &'a self,
        rx: &'a Buffer<'d>,
        ex: &'a Buffer<'d>,
        sx: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        bf16: bool,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        // R38 MoE island: correct (bit-identical) but slower than the fused
        // multi-launch path at the residency this driver allows (32 KB carveout,
        // 4 blocks/SM). Opt-in only, via QWEN_R38_MOE_ISLAND=1.
        let island_on = std::env::var("QWEN_R38_MOE_ISLAND").as_deref() == Ok("1");
        // R38 staged MoE: the island's stages as ordinary full-grid launches
        // (shared expert work rides along with the routed stages).
        // Default (QWEN_R38_MOE_STAGED unset or 2): five launches, with the
        // router selection, shared reduce/activate and per-slot mid run as
        // tails by the last block of their producers (moe_fused.cu), 2 output
        // tiles per gate/up block and 4 per down block. =1: seven launches.
        // =0: the earlier R38 multi-launch path. The island wins if requested.
        let staged_mode = match std::env::var("QWEN_R38_MOE_STAGED").as_deref() {
            Ok("0") => 0,
            Ok("1") => 1,
            _ => 2,
        };
        let staged_on = staged_mode > 0 && !island_on;
        if let (true, Some((island, counters, status, grid))) =
            (island_on || staged_on, &self.island)
        {
            let shared_ok = p.combined
                && matches!(p.projection, cuda_shared::Projection::Split { width: 1, splits: 4 })
                && matches!(p.down, cuda_shared::Projection::Original(0))
                && p.arithmetic == crate::lowbit_shared::Arithmetic::StagedBf16;
            if crate::cuda_affine4::ahead_selected()
                && bf16
                && rows == 1
                && splits == 4
                && shared_ok
                && matches!(router, Projection::Qmv)
                && scores == Scores::StagedBf16
                && tensor_arm == Some(crate::cuda_mcg_tensor::Arm::Mma)
                && std::ptr::eq(rx, ex)
                && std::ptr::eq(ex, sx)
            {
                let [rw, raw, logits, ids, scores_b] = self.router.island_parts();
                let (sh, [ggroup, dgroup]) = self.shared.island_parts()?;
                let (w, b) = self.routed.island_parts();
                let mut args = vec![
                    Arg::Buffer(rx),
                    Arg::Buffer(rw),
                    Arg::Buffer(raw),
                    Arg::Buffer(logits),
                    Arg::Buffer(ids),
                    Arg::Buffer(scores_b),
                    Arg::U32(0),
                    Arg::Buffer(sh[0]),
                    Arg::Buffer(sh[1]),
                    Arg::Buffer(sh[2]),
                    Arg::U32(ggroup as u32),
                    Arg::Buffer(sh[3]),
                    Arg::Buffer(sh[4]),
                    Arg::Buffer(sh[5]),
                    Arg::Buffer(sh[6]),
                    Arg::Buffer(sh[7]),
                    Arg::Buffer(sh[8]),
                    Arg::Buffer(sh[9]),
                    Arg::U32(dgroup as u32),
                    Arg::Buffer(sh[10]),
                    Arg::Buffer(sh[11]),
                    Arg::Buffer(sh[12]),
                    Arg::Buffer(sh[13]),
                    Arg::Buffer(sh[14]),
                    Arg::Buffer(sh[15]),
                    Arg::Buffer(sh[16]),
                    Arg::U32(0),
                ];
                for buf in [
                    &w[1], &w[4], &b[0], &b[1], &w[0], &w[3], &b[2], &b[3], &w[2], &w[5], &w[7],
                    &b[4], &w[6], &b[5], &w[8], &b[6], &b[7],
                ] {
                    args.push(Arg::Buffer(buf));
                }
                args.push(Arg::Buffer(&self.output));
                if staged_on {
                    let stages = self
                        .staged
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("R38 staged MoE kernels not loaded"))?;
                    let prefetch = std::env::var("QWEN_R38_MOE_PF").as_deref() == Ok("1");
                    // Work items per stage (moe_stage.cu); stage 2 may add
                    // prefetch-only blocks past its 55 items.
                    // Output tiles per block for the two MMA stages (1 = island's).
                    let tiles = |var: &str, default: u32| -> Result<u32> {
                        let m = std::env::var(var)
                            .ok()
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(default);
                        ensure!([1, 2, 4, 8].contains(&m), "{var} must be 1, 2, 4 or 8");
                        Ok(m)
                    };
                    let (m3, m5) = (
                        tiles("QWEN_R38_MOE_M3", if staged_mode == 2 { 2 } else { 4 })?,
                        tiles("QWEN_R38_MOE_M5", 4)?,
                    );
                    let pick = |stage: usize, m: u32| -> usize {
                        match m {
                            1 => stage,
                            _ => (if stage == 3 { 7 } else { 10 }) + m.trailing_zeros() as usize - 1,
                        }
                    };
                    let grids = [
                        768u32,
                        11,
                        if prefetch { 55 + 115 } else { 55 },
                        if m3 == 1 { 3520 } else { 320 + (40 / m3) * 20 * 4 },
                        14,
                        if m5 == 1 { 6400 } else { (160 / m5) * 10 * 4 },
                        20,
                    ];
                    if staged_mode == 2 {
                        ensure!([2, 4].contains(&m3), "fused MoE stage 3 tiles must be 2 or 4");
                        let f2 = if m3 == 2 { 15 } else { 16 };
                        // Default F0 (validated at 48 layers, r38-model-ab48-09). F0b
                        // (one-trip router, tree selection) is opt-in with
                        // QWEN_R38_MOE_F0=b: identical outputs, no measured gain.
                        let f0 = if std::env::var("QWEN_R38_MOE_F0").as_deref() == Ok("b") { 21 } else { 13 };
                        let plan: [(usize, u32, bool); 5] = [
                            (f0, 768, true),
                            (14, 370, true),
                            (f2, 1 + (40 / m3) * 20 * 4, true),
                            (pick(5, m5), grids[5], false),
                            (6, 20, false),
                        ];
                        let mut calls = Vec::with_capacity(5);
                        let pdl = (crate::cuda_affine4::pdl_selected() || r52_base_pdl()) && self.pdl_staged.is_some();
                        if let (true, Some(back)) = (crate::cuda_affine4::pdl_selected() || r52_both(), &self.r52) {
                            ensure!(f0 == 13 && m3 == 2 && m5 == 4, "R52 MoE requires the R39 defaults");
                            let (g, d) = r52_tiles();
                            let variant = std::env::var("QWEN_R52_MOE").unwrap_or_default();
                            let split_down = matches!(variant.as_str(), "g4v" | "g4w" | "g4wl" | "g4w2" | "g4w2p" | "g4w2e" | "g4w2o" | "g4w2m" | "g4w2s" | "g4w2f" | "g4w2t" | "g4w2u" | "g4w2q" | "g4w2v" | "g4w2d" | "g4w2x" | "g4w2y" | "g4w2z" | "g4w2w" | "g4w2n" | "g4w2r" | "g4w2a" | "g4w2g" | "g4w2h");
                            let shared_blocks = if matches!(variant.as_str(), "g4w2s" | "g4w2t" | "g4w2u" | "g4w2q" | "g4w2v") { 0 } else if variant == "g4d4t" || variant == "g4d4u" || split_down { 160 } else { 320 };
                            // g4d4u is persistent: one block per first-wave item; g4v/g4w have no down items.
                            let grid = if variant == "g4d4u" || split_down { shared_blocks + 1600 * 2 / g } else { shared_blocks + 1600 * 2 / g + 1600 * 4 / d };
                            let down_grid = if variant == "g4w" || variant == "g4wl" { 800u32 } else if matches!(variant.as_str(), "g4w2" | "g4w2p" | "g4w2e" | "g4w2o" | "g4w2m" | "g4w2s" | "g4w2f" | "g4w2t" | "g4w2u" | "g4w2q" | "g4w2v" | "g4w2d" | "g4w2x" | "g4w2y" | "g4w2z" | "g4w2w" | "g4w2n" | "g4w2r" | "g4w2a" | "g4w2g" | "g4w2h") { 400 } else { 200 };
                            let (rc, fin) = self.r52_bufs.as_ref().unwrap();
                            // Candidate arm with a pump table: pumping gate/up and down entries.
                            let pumping = match (&self.r52_pumpk, self.r52_pump.get()) {
                                (Some(k), Some(t)) if crate::cuda_affine4::pdl_selected() => Some((k, t)),
                                _ => None,
                            };
                            let mut a = args.clone();
                            a.push(Arg::Buffer(counters));
                            if let Some(f0) = &self.r52_f0 {
                                a.push(Arg::Buffer(rc));
                                if let Some(trace) = &self.r52_trace { a.push(Arg::Buffer(trace)); }
                                calls.push(unsafe { f0.prepare(a, [768, 1, 1], [128, 1, 1])? });
                            } else {
                                calls.push(unsafe { self.pdl_staged.as_ref().unwrap()[0].prepare(a, [768, 1, 1], [128, 1, 1])? });
                            }
                            let mut a = args.clone();
                            a.push(Arg::Buffer(rc));
                            a.push(Arg::Buffer(fin));
                            if let Some(trace) = &self.r52_trace { a.push(Arg::Buffer(trace)); }
                            if let Some((k, (table, entries, units))) = pumping {
                                ensure!(self.r52_trace.is_none(), "R52 pump has no traced entries");
                                a.extend([Arg::Buffer(table), Arg::U32(*entries), Arg::U32(*units)]);
                                let down_args = a.clone();
                                calls.push(unsafe { k[0].prepare(a, [grid, 1, 1], [128, 1, 1])? });
                                calls.push(unsafe { k[1].prepare(down_args, [down_grid, 1, 1], [128, 1, 1])? });
                                return Ok(calls);
                            }
                            let mut down_args = a.clone();
                            if let Some(tags) = &self.r53_tags { down_args.push(Arg::Buffer(tags)); }
                            calls.push(unsafe { back.prepare(a, [grid, 1, 1], [128, 1, 1])? });
                            if let Some(down) = &self.r52_down {
                                calls.push(unsafe { down.prepare(down_args, [down_grid, 1, 1], [128, 1, 1])? });
                            }
                            return Ok(calls);
                        }
                        ensure!(!pdl || (f0 == 13 && m3 == 2 && m5 == 4),
                            "R39 MoE entry-wait experiment requires F0a, M3=2, M5=4");
                        for (step, (index, grid, counted)) in plan.into_iter().enumerate() {
                            let mut a = args.clone();
                            a.push(if counted { Arg::Buffer(counters) } else { Arg::U32(0) });
                            let kernel = if pdl { &self.pdl_staged.as_ref().unwrap()[step] } else { &stages[index] };
                            calls.push(unsafe { kernel.prepare(a, [grid, 1, 1], [128, 1, 1])? });
                        }
                        return Ok(calls);
                    }
                    // QWEN_R38_MOE_D3=2|4 and QWEN_R38_MOE_D5=4|8: deep-load
                    // variants (moe_stage_m.cu ms_mma_deep) for stages 3 and 5.
                    let d3 = std::env::var("QWEN_R38_MOE_D3").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                    let d5 = std::env::var("QWEN_R38_MOE_D5").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                    ensure!([0, 2, 4].contains(&d3) && [0, 4, 8].contains(&d5), "deep MoE variant");
                    let mut calls = Vec::with_capacity(7);
                    for k in 0..7usize {
                        let (index, grid) = match k {
                            3 if d3 == 2 => (17, 320 + 20 * 20 * 4),
                            3 if d3 == 4 => (18, 320 + 10 * 20 * 4),
                            5 if d5 == 4 => (19, 40 * 10 * 4),
                            5 if d5 == 8 => (20, 20 * 10 * 4),
                            3 => (pick(3, m3), grids[3]),
                            5 => (pick(5, m5), grids[5]),
                            _ => (k, grids[k]),
                        };
                        let mut a = args.clone();
                        a.push(Arg::U32(u32::from(prefetch)));
                        calls.push(unsafe { stages[index].prepare(a, [grid, 1, 1], [128, 1, 1])? });
                    }
                    return Ok(calls);
                }
                args.extend([
                    Arg::U32(self.island_last_stage.get()),
                    Arg::Buffer(counters),
                    Arg::Buffer(status),
                    Arg::U32(2_000_000),
                ]);
                return Ok(vec![unsafe { island.prepare(args, [*grid, 1, 1], [128, 1, 1])? }]);
            }
        }
        let (mut c, r) = self.router.record_from(rx, rows, router, scores)?;
        if crate::cuda_affine4::ahead_selected()
            && bf16
            && tensor_arm == Some(crate::cuda_mcg_tensor::Arm::Mma)
        {
            // R38: shared expert first, then the routed path whose finish also
            // performs moe_add_bf16 into this engine's output.
            c.extend(self.shared.record_from(sx, rows, p)?);
            c.extend(self.routed.record_routed_fused(
                ex,
                r,
                splits,
                self.shared.output(),
                &self.output,
            )?);
            return Ok(c);
        }
        c.extend(self.routed.record_routed(ex, r, splits, bf16, tensor_arm)?);
        c.extend(self.shared.record_from(sx, rows, p)?);
        // All producer and consumer calls borrow this engine's private buffers.
        // Grid and count cover exactly active rows; IDs come only from Router.
        unsafe {
            c.push(self.add.prepare(
                vec![
                    Arg::Buffer(self.routed.output()),
                    Arg::Buffer(self.shared.output()),
                    Arg::Buffer(&self.output),
                    Arg::U32((rows * H) as u32),
                ],
                [(rows * H).div_ceil(128) as u32, 1, 1],
                [128, 1, 1],
            )?);
        }
        Ok(c)
    }
    fn record(
        &self,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        bf16: bool,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'_, 'm, 'd>>> {
        self.record_inputs(
            self.router_input(),
            if bf16 { &self.x } else { &self.x32 },
            &self.x,
            rows,
            p,
            router,
            scores,
            splits,
            bf16,
            tensor_arm,
        )
    }
    fn router_input(&self) -> &Buffer<'d> {
        self.router.input_buffer()
    }
    pub fn record_from<'a>(
        &'a self,
        router_input: &'a Buffer<'d>,
        expert_input: &'a Buffer<'d>,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(
            !router_input.aliases(&self.output) && !expert_input.aliases(&self.output),
            "MoE input/output alias"
        );
        self.record_inputs(
            router_input,
            expert_input,
            expert_input,
            rows,
            p,
            router,
            scores,
            splits,
            true,
            tensor_arm,
        )
    }
    pub fn output(&self) -> &Buffer<'d> {
        &self.output
    }
    /// R52 multi4 pump: upload the region table this engine's MoE launches warm.
    pub(crate) fn set_pump_table(&self, regions: &[(u64, usize)]) -> Result<()> {
        if self.r52_pumpk.is_none() { return Ok(()); }
        ensure!(!regions.is_empty() && regions.len() <= 64, "pump table size");
        let mut raw = vec![];
        let mut cum = 0u32;
        for &(addr, bytes) in regions {
            ensure!(addr % 16 == 0, "pump region alignment");
            let lines = bytes.div_ceil(128) as u32;
            for v in [addr as u32, (addr >> 32) as u32, lines, cum] { raw.extend(v.to_le_bytes()); }
            cum += lines;
        }
        let b = self.d.alloc(raw.len())?;
        b.upload(&raw)?;
        self.r52_pump.set((b, regions.len() as u32, cum)).map_err(|_| anyhow::anyhow!("pump table already set"))
    }
    /// R58 S3: zero the output rows (the guard tail is left alone), so a layer whose MLP is
    /// skipped adds nothing in its HC write.
    pub(crate) fn zero_output(&self) -> Result<()> {
        self.output.upload(&vec![0u8; MAX_ROWS * H * 2])
    }
    /// R58 sibling screen: the expert ids the router selected for the last `rows` rows it ran
    /// ([row][TOPK]; a host read of the router's id buffer, no launch).
    pub(crate) fn route_ids(&self, rows: usize) -> Result<Vec<u32>> {
        ensure!((1..=MAX_ROWS).contains(&rows), "route id rows");
        let ids = self.router.island_parts()[3];
        let n = rows * crate::moe::TOPK;
        ensure!(ids.byte_len() >= n * 4, "route id buffer");
        let mut raw = vec![0u8; ids.byte_len()];
        ids.download(&mut raw)?;
        Ok(raw[..n * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    /// R52 pump: (device address, bytes) of the router's BF16 weights, the
    /// first weights this engine reads.
    pub(crate) fn router_weight_region(&self) -> (u64, usize) {
        let w = self.router.island_parts()[0];
        (w.device_addr(), w.byte_len())
    }
    /// R53 g4w2h's down launch leaves its counter block to the layer's stream write
    /// (r39pdl_hc_write_z). A standalone MoE run has no such write: its second execution
    /// would read stale counters (and the expert ids before F0 wrote them), so standalone
    /// runners refuse it.
    pub fn r53_refuse_standalone() -> Result<()> {
        ensure!(std::env::var("QWEN_R52_MOE").as_deref() != Ok("g4w2h"),
            "QWEN_R52_MOE=g4w2h runs only inside a layer, whose stream write clears its counters");
        Ok(())
    }
    /// R53 g4w2h: the counter block the following stream write must clear (None for every
    /// other variant, whose down launch clears its own counters).
    pub(crate) fn r53_counter_reset(&self) -> Option<&Buffer<'d>> {
        if std::env::var("QWEN_R52_MOE").as_deref() != Ok("g4w2h") { return None; }
        self.r52_bufs.as_ref().map(|(c, _)| c)
    }
    /// R52/R53 sticky spin-timeout word of the counter block (0 = no timeout; otherwise
    /// 1 + the counter index that timed out, 1 + 1000 for a tagged-word wait).
    pub fn r52_status(&self) -> Result<Option<u32>> {
        let Some((c, _)) = &self.r52_bufs else { return Ok(None) };
        let mut raw = vec![0u8; c.byte_len()];
        c.download(&mut raw)?;
        Ok(Some(u32::from_le_bytes(raw[68 * 4..68 * 4 + 4].try_into().unwrap())))
    }
    /// R52 diagnostic trace (traced entries only): six words per ticket.
    pub fn r52_trace(&self) -> Result<Option<(u32, Vec<u64>)>> {
        let Some(b) = &self.r52_trace else { return Ok(None) };
        let mut raw = vec![0u8; b.byte_len()];
        b.download(&mut raw)?;
        let (g, d) = r52_tiles();
        let v = std::env::var("QWEN_R52_MOE").unwrap_or_default();
        let shared_blocks = if v == "g4w2s" { 0 } else if matches!(v.as_str(), "g4d4t" | "g4d4u" | "g4v" | "g4w" | "g4wl" | "g4w2" | "g4w2e" | "g4w2o" | "g4w2m" | "g4w2f") { 160 } else { 320 };
        let downs = if matches!(v.as_str(), "g4v" | "g4w" | "g4wl" | "g4w2" | "g4w2e" | "g4w2o" | "g4w2m" | "g4w2s" | "g4w2f") { 0 } else { 1600 * 4 / d };
        Ok(Some((shared_blocks + 1600 * 2 / g + downs, raw.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())))
    }
    /// R52: the one-launch back half's counter words (ticket, exit, counts, flags,
    /// sticky status at +4); all zero between launches when nothing timed out.
    pub fn r52_counters(&self) -> Result<Vec<u32>> {
        let Some((counters, _)) = &self.r52_bufs else { return Ok(vec![]) };
        let mut raw = vec![0u8; counters.byte_len()];
        counters.download(&mut raw)?;
        Ok(raw[64 * 4..408 * 4].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    pub(crate) fn poison(&self, p: Policy) -> Result<()> {
        self.router.poison(true)?;
        self.routed.poison()?;
        self.shared.poison(p)?;
        self.output.upload(&vec![0xff; self.output.byte_len()])?;
        Ok(())
    }
    pub(crate) fn collect(
        &self,
        rows: usize,
        p: Policy,
        splits: usize,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Run> {
        // A nonfinite route or intermediate fails here, before any output can
        // escape the Run. This component has no committed model/KV state.
        let router = self.router.collect(rows)?;
        let descriptor_words_checked = if let Some(arm) = tensor_arm {
            self.routed.tensor_check(&router.routing.ids, arm)?
        } else {
            0
        };
        let (routed, mut canary_bytes) = self.routed.collect(rows, splits)?;
        let shared = self.shared.collect(rows, p)?;
        let mut raw = vec![0; self.output.byte_len()];
        self.output.download(&mut raw)?;
        let active = rows * H * 2;
        ensure!(
            raw[active..].iter().all(|&b| b == 0xff),
            "composed final canary"
        );
        canary_bytes += raw.len() - active;
        let output: Vec<_> = raw[..active]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            output.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "composed finite output"
        );
        for (i, &v) in output.iter().enumerate() {
            let expected =
                bf16::from_f32(routed[7][i] + bf16::from_bits(shared.stages.output[i]).to_f32());
            ensure!(expected.to_bits() == v, "BF16 sum conditional bit check");
        }
        Ok(Run {
            router,
            routed,
            shared,
            output,
            canary_bytes,
            descriptor_words_checked,
            total_us: 0.,
            enqueue_us: 0.,
            wait_us: 0.,
        })
    }
    pub fn plan(
        &self,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        bf16: bool,
        repetitions: usize,
        graph: bool,
    ) -> Result<Plan<'_, 'm, 'd>> {
        self.plan_tensor(
            rows,
            p,
            router,
            scores,
            splits,
            bf16,
            repetitions,
            graph,
            None,
        )
    }
    pub fn plan_tensor(
        &self,
        rows: usize,
        p: Policy,
        router: Projection,
        scores: Scores,
        splits: usize,
        bf16: bool,
        repetitions: usize,
        graph: bool,
        tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
    ) -> Result<Plan<'_, 'm, 'd>> {
        ensure!(
            rows > 0 && rows == self.rows.get() && (1..=8).contains(&repetitions),
            "composed plan input/bounds"
        );
        let mut c = self.record(rows, p, router, scores, splits, bf16, tensor_arm)?;
        let calls = c.len();
        let execute = if graph {
            for _ in 1..repetitions {
                c.extend(self.record(rows, p, router, scores, splits, bf16, tensor_arm)?);
            }
            Execute::Graph(self.d.capture(c)?)
        } else {
            Execute::Calls(c)
        };
        Ok(Plan {
            owner: self,
            execute,
            rows,
            p,
            splits,
            repetitions,
            calls,
            tensor_arm,
        })
    }
}
enum Execute<'a, 'm, 'd> {
    Graph(Graph<'a, 'm, 'd>),
    Calls(Vec<PreparedCall<'a, 'm, 'd>>),
}
pub struct Plan<'a, 'm, 'd> {
    owner: &'a Engine<'m, 'd>,
    execute: Execute<'a, 'm, 'd>,
    rows: usize,
    p: Policy,
    splits: usize,
    repetitions: usize,
    pub calls: usize,
    tensor_arm: Option<crate::cuda_mcg_tensor::Arm>,
}
pub struct Run {
    pub router: cuda_router::Run,
    pub routed: Vec<Vec<f32>>,
    pub shared: cuda_shared::Run,
    pub output: Vec<u16>,
    pub canary_bytes: usize,
    pub descriptor_words_checked: usize,
    pub total_us: f64,
    pub enqueue_us: f64,
    pub wait_us: f64,
}
impl Plan<'_, '_, '_> {
    pub fn run(&self, wait_each: bool) -> Result<Run> {
        let a = self.owner;
        ensure!(a.rows.get() == self.rows, "stale composed plan rows");
        ensure!(
            !wait_each || matches!(self.execute, Execute::Calls(_)),
            "graph wait-each"
        );
        a.poison(self.p)?;
        a.d.sync()?;
        let t = Instant::now();
        let mut wait_us = 0.;
        let mut enqueue_us = 0.;
        match &self.execute {
            Execute::Graph(g) => {
                let t = Instant::now();
                g.launch()?;
                enqueue_us += t.elapsed().as_secs_f64() * 1e6;
            }
            Execute::Calls(c) => {
                for _ in 0..self.repetitions {
                    for call in c {
                        let t = Instant::now();
                        call.enqueue()?;
                        enqueue_us += t.elapsed().as_secs_f64() * 1e6;
                        if wait_each {
                            let t = Instant::now();
                            a.d.sync()?;
                            wait_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    }
                }
            }
        }
        let w = Instant::now();
        a.d.sync()?;
        wait_us += w.elapsed().as_secs_f64() * 1e6;
        let total_us = t.elapsed().as_secs_f64() * 1e6;
        let mut r = a.collect(self.rows, self.p, self.splits, self.tensor_arm)?;
        r.total_us = total_us;
        r.enqueue_us = enqueue_us;
        r.wait_us = wait_us;
        Ok(r)
    }
}

/// R60 S4 (prefill v2): one MoE block's intermediates for chunk-sized row counts, shared by every
/// stage layer.
pub struct MoeScratch<'d> {
    rows: usize,
    pub router: crate::cuda_router::RouterScratch<'d>,
    pub routed: crate::cuda_mcg_moe::RoutedScratch<'d>,
    pub shared: crate::cuda_shared::SharedScratch<'d>,
    output: Buffer<'d>,
}
impl<'d> MoeScratch<'d> {
    pub fn new(d: &'d Device, rows: usize) -> Result<Self> {
        Ok(Self {
            rows,
            router: crate::cuda_router::RouterScratch::new(d, rows)?,
            routed: crate::cuda_mcg_moe::RoutedScratch::new(d, rows)?,
            shared: crate::cuda_shared::SharedScratch::new(d, rows)?,
            output: d.alloc(rows * crate::moe::H * 2 + GUARD)?,
        })
    }
    /// BF16 [rows, H]: routed + shared of the last recorded prefill.
    pub fn output(&self) -> &Buffer<'d> {
        &self.output
    }
}
impl<'m, 'd> Engine<'m, 'd> {
    /// R60 S4 (prefill v2): router, routed experts (expert-grouped K4), shared expert and the
    /// join for chunk rows; x is the MLP input (BF16 [rows, H]).
    pub(crate) fn record_prefill<'a>(
        &'a self,
        s: &'a MoeScratch<'d>,
        k: &'a crate::cuda_mcg_moe::K64<'m, 'd>,
        x: &'a Buffer<'d>,
        rows: usize,
    ) -> Result<Vec<PreparedCall<'a, 'm, 'd>>> {
        ensure!(rows > 0 && rows <= s.rows, "MoE prefill rows");
        let h = crate::moe::H;
        let mut calls = self.router.record_prefill(&s.router, x, rows)?;
        calls.extend(self.routed.record_prefill(&s.routed, k, x, &s.router.ids, &s.router.scores, rows)?);
        calls.extend(self.shared.record_prefill(&s.shared, x, rows)?);
        // SAFETY: kernels/moe_join.cu moe_add_bf16 ABI (as the decode join).
        calls.push(unsafe {
            self.add.prepare(
                vec![Arg::Buffer(s.routed.output()), Arg::Buffer(s.shared.output()), Arg::Buffer(&s.output), Arg::U32((rows * h) as u32)],
                [(rows * h).div_ceil(128) as u32, 1, 1],
                [128, 1, 1],
            )?
        });
        Ok(calls)
    }
}
