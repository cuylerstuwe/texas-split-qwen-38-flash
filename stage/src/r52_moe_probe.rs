//! R52 (Opus, direct): bounded one-layer MoE timing probe. Loads one GDN layer's
//! router, shared expert and all 512 routed experts, records the current R39
//! (wait-equipped) MoE path and the R52 one-launch back half for several distinct
//! inputs (different routes, so most expert weights are cold), checks that both
//! arms give identical outputs for every input, then replays one graph per arm,
//! interleaved. Optional flush reads between MoE calls make every weight cold.
//! No model state, sampling or generation claim.
use crate::{
    cuda::{Arg, Buffer, Device},
    cuda_affine4,
    cuda_lowbit_layer as rl,
    cuda_lowbit_moe::Engine,
    cuda_router::Projection,
    lowbit_layer,
    lowbit_router::{H, Scores},
    mcg_moe::Bank,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::Path, sync::atomic::Ordering::Relaxed, time::Instant};

fn policy() -> rl::Policy {
    rl::Policy {
        hc_split: true,
        separate_write: true,
        gdn_fused: true,
        all_states: false,
        wide: false,
        expert: rl::Expert::Mma,
    }
}
fn set_arm(candidate: bool) {
    cuda_affine4::AHEAD.store(true, Relaxed);
    cuda_affine4::PDL.store(candidate, Relaxed);
}
fn clear_arm() {
    cuda_affine4::AHEAD.store(false, Relaxed);
    cuda_affine4::PDL.store(false, Relaxed);
}
fn input(seed: u32) -> Vec<u16> {
    let mut s = 0x9e3779b9u32 ^ seed.wrapping_mul(0x85ebca6b);
    (0..H)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = ((s & 0xffff) as f32 - 32768.) / 16384.;
            half::bf16::from_f32(v).to_bits()
        })
        .collect()
}
pub fn run(d: &Device, root: &Path, layer: usize, inputs: usize, replays: usize, flush_mib: usize) -> Result<Value> {
    crate::cuda_lowbit_moe::Engine::r53_refuse_standalone()?;
    ensure!((1..=16).contains(&inputs) && (2..=64).contains(&replays) && flush_mib <= 256, "probe bounds");
    let m = rl::Modules::load(d)?;
    let w = lowbit_layer::Weights::load(root, layer)?;
    let bank = Bank::full_packed(root, layer)?;
    let mut engine = Engine::new(
        d, &m.router, &m.linear, &m.project, &m.moe, &m.affine, &m.split, &m.shared, &m.join,
        &w.router, &bank, &w.shared,
    )?;
    engine.enable_tensor(&m.tensor)?;
    let xs: Vec<Vec<u16>> = (0..inputs).map(|i| input(i as u32 + 1)).collect();
    engine.prepare(&xs[0], &xs[0])?;
    let bufs = xs
        .iter()
        .map(|x| {
            let b = d.alloc(H * 2 + 256)?;
            let mut raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            raw.extend([0u8; 256]);
            b.upload(&raw)?;
            Ok(b)
        })
        .collect::<Result<Vec<Buffer<'_>>>>()?;
    let p = policy();
    let record = |i: usize| {
        engine.r38_record_external(&bufs[i], 1, p.shared(), Projection::Qmv, Scores::StagedBf16, 4, p.expert.arm())
    };
    // Correctness: every input, both arms, outputs compared byte for byte.
    let mut identical = 0;
    let mut route_check = vec![];
    for i in 0..inputs {
        let mut outs = vec![];
        for cand in [false, true] {
            set_arm(cand);
            let calls = record(i);
            clear_arm();
            let calls = calls?;
            let n = calls.len();
            let g = d.capture(calls)?;
            g.launch()?;
            d.sync()?;
            drop(g);
            let mut raw = vec![0u8; engine.output().byte_len()];
            engine.output().download(&mut raw)?;
            outs.push((n, raw));
        }
        ensure!(outs[0].1 == outs[1].1, "R52 MoE output differs from R39 for input {i}");
        identical += 1;
        route_check.push(json!({"input":i,"calls":[outs[0].0,outs[1].0]}));
    }
    let status = engine.r52_counters()?;
    ensure!(status.iter().all(|&v| v == 0), "R52 counters not reset/status set: {status:?}");
    // Timing graphs: inputs MoE calls back to back (optional flush before each).
    let flush = if flush_mib > 0 {
        let probe = d.module(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/l2_probe.sm120.cubin"),
            "30e15c12d6a278bbf11d3dda45e0a98d79c5dd3632411030d80af67dae7134e8",
        )?;
        let f = d.alloc(flush_mib << 20)?;
        f.upload(&vec![0x5au8; flush_mib << 20])?;
        let sink = d.alloc(4096)?;
        Some((probe, f, sink))
    } else {
        None
    };
    let read = match &flush { Some((probe, _, _)) => Some(probe.kernel("l2p_read")?), None => None };
    let flush_call = || -> Result<Option<crate::cuda::PreparedCall<'_, '_, '_>>> {
        match (&flush, &read) {
            (Some((_, f, sink)), Some(k)) => Ok(Some(unsafe {
                k.prepare(
                    vec![Arg::Buffer(f), Arg::U64(((flush_mib << 20) / 16) as u64), Arg::Buffer(sink)],
                    [170 * 8, 1, 1],
                    [256, 1, 1],
                )?
            })),
            _ => Ok(None),
        }
    };
    // A plain (non-early-launch) separator between consecutive MoE calls of this
    // one engine: the overlapped back half must not launch beside the previous
    // call's tail, which still owns this engine's counters. In the model each
    // layer has its own engine, so this cannot arise there.
    let sep_module = d.module(
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/macuda/copy1d.sm_120.cubin")),
        "03a644f4a55d702f0aa2bf969fa0e716fcf89e2d948f42f0bbcddc7e0ede4c58",
    )?;
    let sep_kernel = sep_module.kernel("tinycudart_copy1d")?;
    let (sep_a, sep_b) = (d.alloc(4096)?, d.alloc(4096)?);
    sep_a.upload(&[0x11u8; 4096])?;
    let sep_call = || unsafe {
        sep_kernel.prepare(vec![Arg::Buffer(&sep_b), Arg::Buffer(&sep_a), Arg::U64(256)], [1, 1, 1], [256, 1, 1])
    };
    let mut graphs = vec![];
    let mut counts = vec![];
    for arm in 0..3 {
        // arm 0: R39, arm 1: R52, arm 2: flush only (when flushing)
        if arm == 2 && flush.is_none() { break; }
        let mut calls = vec![];
        for i in 0..inputs {
            if let Some(c) = flush_call()? { calls.push(c); } else { calls.push(sep_call()?); }
            if arm < 2 {
                set_arm(arm == 1);
                let r = record(i);
                clear_arm();
                calls.extend(r?);
            }
        }
        counts.push(calls.len());
        graphs.push(d.capture(calls)?);
    }
    let mut times = vec![vec![]; graphs.len()];
    for g in &graphs { g.launch()?; d.sync()?; }
    for r in 0..replays {
        for k in 0..graphs.len() {
            let k = if r % 2 == 0 { k } else { graphs.len() - 1 - k };
            d.sync()?;
            let t = Instant::now();
            graphs[k].launch()?;
            d.sync()?;
            times[k].push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    drop(graphs);
    // Trace pass (traced entries only): candidate arm alone, optional flush first,
    // then optionally an L2 warming probe over the router weights
    // (QWEN_R52_PUMPTEST: 0 cp.async, 1 real loads, 2 launch only).
    let pump_mode: Option<u32> = std::env::var("QWEN_R52_PUMPTEST").ok().and_then(|v| v.parse().ok());
    let pump_kernel = m.join.kernel("r52_pump_probe")?;
    let (raddr, rbytes) = engine.router_weight_region();
    let mut traces = vec![];
    for i in 0..inputs.min(4) {
        let mut calls = vec![];
        if let Some(c) = flush_call()? { calls.push(c); }
        if let Some(mode) = pump_mode {
            calls.push(unsafe { pump_kernel.prepare(vec![Arg::U64(raddr), Arg::U32((rbytes / 16) as u32), Arg::U32(mode), Arg::U64(0)], [1280, 1, 1], [128, 1, 1])? });
        }
        set_arm(true);
        let r = record(inputs - 1 - i);
        clear_arm();
        calls.extend(r?);
        let g = d.capture(calls)?;
        g.launch()?;
        d.sync()?;
        drop(g);
        if let Some((t, words)) = engine.r52_trace()? {
            let mut s = summarize(t as usize, &words);
            // R53: raw rows of the first trace (QWEN_R53_RAWTRACE=1): back tickets
            // [start, t1, t2, end, smid, kind+16*tail], then down rows at word 21120.
            if i == 0 && std::env::var("QWEN_R53_RAWTRACE").as_deref() == Ok("1") {
                let back: Vec<&[u64]> = (0..t as usize).map(|q| &words[q * 6..q * 6 + 6]).collect();
                let down: Vec<&[u64]> = (0..800).map(|b| &words[21120 + b * 6..21120 + b * 6 + 6]).filter(|r| r[5] % 16 == 5).collect();
                s["raw_back"] = json!(back);
                s["raw_down"] = json!(down);
            }
            traces.push(s);
        }
    }
    let status = engine.r52_counters()?;
    ensure!(status.iter().all(|&v| v == 0), "R52 counters after timing: {status:?}");
    let med = |v: &Vec<f64>| { let mut s = v.clone(); s.sort_by(f64::total_cmp); s[s.len() / 2] };
    let meds: Vec<f64> = times.iter().map(med).collect();
    let flush_us = if meds.len() == 3 { meds[2] } else { 0. };
    Ok(json!({"kind":"r52_moe_probe","layer":layer,"inputs":inputs,"replays":replays,"flush_mib":flush_mib,
        "outputs_identical_inputs":identical,"calls_per_graph":counts,"routes":route_check,
        "median_graph_us":meds,"r39_per_moe_us":(meds[0]-flush_us)/inputs as f64,
        "r52_per_moe_us":(meds[1]-flush_us)/inputs as f64,"times_us":times,
        "variant":std::env::var("QWEN_R52_MOE").ok(),"traces":traces,
        "scope":"one layer MoE sublayer, host launch+sync per graph divided by calls; no model/generation claim"}))
}

/// Per-kind timeline summary of one traced launch (microseconds from the first start).
fn summarize(t: usize, w: &[u64]) -> Value {
    let rows: Vec<&[u64]> = (0..t).map(|q| &w[q * 6..q * 6 + 6]).collect();
    let t0 = rows.iter().map(|r| r[0]).min().unwrap_or(0);
    let us = |v: u64| (v.saturating_sub(t0)) as f64 / 1000.;
    let pct = |mut v: Vec<f64>| -> Value {
        if v.is_empty() { return json!(null); }
        v.sort_by(f64::total_cmp);
        let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        json!([at(0.), at(0.1), at(0.5), at(0.9), at(1.)])
    };
    let mut kinds = vec![];
    for kind in 0..3u64 {
        let k: Vec<&&[u64]> = rows.iter().filter(|r| r[5] % 16 == kind).collect();
        let name = ["shared_down", "gate_up", "down"][kind as usize];
        kinds.push(json!({"kind":name,"blocks":k.len(),
            "start_us":pct(k.iter().map(|r| us(r[0])).collect()),
            "phase1_us":pct(k.iter().map(|r| (r[1].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "main_us":pct(k.iter().map(|r| (r[2].saturating_sub(r[1])) as f64 / 1000.).collect()),
            "duration_us":pct(k.iter().map(|r| (r[3].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "end_us":pct(k.iter().map(|r| us(r[3])).collect()),
            "tails":k.iter().filter(|r| r[5] >= 16).count(),
            "tail_end_us":pct(k.iter().filter(|r| r[5] >= 16).map(|r| us(r[3])).collect())}));
    }
    // F0 rows (traced F0 only) live after the back half's 2720 tickets.
    let f0: Vec<&[u64]> = (0..768).map(|b| &w[16320 + b * 6..16320 + b * 6 + 6]).filter(|r| r[5] % 16 >= 3).collect();
    let f0_base = f0.iter().map(|r| r[0]).min();
    let us0 = |v: u64| match f0_base { Some(b) => (v as f64 - b as f64) / 1000., None => 0. };
    for kind in 3..5u64 {
        let k: Vec<&&[u64]> = f0.iter().filter(|r| r[5] % 16 == kind).collect();
        if k.is_empty() { continue; }
        let name = ["router", "shared_gate_up"][(kind - 3) as usize];
        kinds.push(json!({"kind":name,"blocks":k.len(),
            "start_us":pct(k.iter().map(|r| us0(r[0])).collect()),
            "phase1_us":pct(k.iter().map(|r| (r[1].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "main_us":pct(k.iter().map(|r| (r[2].saturating_sub(r[1])) as f64 / 1000.).collect()),
            "duration_us":pct(k.iter().map(|r| (r[3].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "end_us":pct(k.iter().map(|r| us0(r[3])).collect()),
            "tails":k.iter().filter(|r| r[5] >= 16).count(),
            "tail_end_us":pct(k.iter().filter(|r| r[5] >= 16).map(|r| us0(r[3])).collect())}));
    }
    // Split down kernel rows (variant g4v) after the F0 area.
    let dk: Vec<&[u64]> = (0..800).map(|b| &w[21120 + b * 6..21120 + b * 6 + 6]).filter(|r| r[5] % 16 == 5).collect();
    if !dk.is_empty() {
        kinds.push(json!({"kind":"down_piece","blocks":dk.len(),
            "start_us":pct(dk.iter().map(|r| us(r[0])).collect()),
            "phase1_us":pct(dk.iter().map(|r| (r[1].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "main_us":pct(dk.iter().map(|r| (r[2].saturating_sub(r[1])) as f64 / 1000.).collect()),
            "duration_us":pct(dk.iter().map(|r| (r[3].saturating_sub(r[0])) as f64 / 1000.).collect()),
            "end_us":pct(dk.iter().map(|r| us(r[3])).collect()),
            "tails":dk.iter().filter(|r| r[5] >= 16).count(),
            "tail_end_us":pct(dk.iter().filter(|r| r[5] >= 16).map(|r| us(r[3])).collect())}));
    }
    // Per-SM view of the first wave: gate/up and shared blocks per SM, and the
    // latest gate/up end on each SM, grouped by how many gate/up blocks it ran.
    let mut per_sm: std::collections::BTreeMap<u64, (u32, u32, f64)> = std::collections::BTreeMap::new();
    for r in &rows {
        let e = per_sm.entry(r[4]).or_insert((0, 0, 0.));
        match r[5] % 16 { 1 => { e.0 += 1; e.2 = e.2.max(us(r[3])); } 0 => e.1 += 1, _ => {} }
    }
    let mut by_count: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    for (_, (g, _, end)) in &per_sm { by_count.entry(*g).or_default().push(*end); }
    let by_count: Vec<Value> = by_count.into_iter().map(|(g, v)| json!({"gate_up_blocks":g,"sms":v.len(),"latest_end_us":pct(v)})).collect();
    let back_from_f0 = f0_base.map(|b| (t0 as f64 - b as f64) / 1000.);
    let sel: Vec<f64> = match f0_base { Some(_) => (0..4).map(|i| us0(w[16320 + 768 * 6 + i])).collect(), None => vec![] };
    let span = rows.iter().chain(dk.iter()).map(|r| us(r[3])).fold(0., f64::max);
    let sms: std::collections::BTreeSet<u64> = rows.iter().map(|r| r[4]).collect();
    json!({"span_us":span,"per_sm_gate_up":by_count,"back_start_after_f0_start_us":back_from_f0,"select_stamps_us":sel,"sms":sms.len(),"kinds":kinds,"percentiles":"min,p10,p50,p90,max"})
}
