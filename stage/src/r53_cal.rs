//! R53 calibration (Opus, direct, 2026-11-06): opt-in instrumentation for the
//! resident whole-graph replay. Null twins replay the model's exact launch
//! geometry and dependency edges with empty kernels (launch/dependency cost
//! without work). Driver stamps (isolated r53nv archive only) record every
//! kernel's completion clock and every chain's start. A clock probe reports SM
//! MHz. No model kernel, argument or output changes.
use crate::cuda::{Arg, Buffer, Device, Kernel, Module};
use anyhow::{Result, ensure};
use serde_json::json;
use std::path::Path;

pub const SHA: &str = "8101ff403b1f8dab82b96303faeb78eb5976ceae3419d3ced4a47675bed23371";

pub fn module(d: &Device) -> Result<Module<'_>> {
    d.module(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/r53_cal.sm120.cubin"),
        SHA,
    )
}

fn flag(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty() && v != "0")
}

/// QWEN_R53_NULL=match|serial|both: null replays with the same early-launch
/// edges ("match") and/or with every edge serialized ("serial").
pub fn null_variants() -> Vec<&'static str> {
    match flag("QWEN_R53_NULL").as_deref() {
        Some("1" | "match") => vec!["match"],
        Some("serial") => vec!["serial"],
        Some("both") => vec!["match", "serial"],
        _ => vec![],
    }
}
/// QWEN_R53_STAMP=1, only in a build against the r53nv archive.
pub fn stamp_enabled() -> bool {
    cfg!(feature = "r53-driver") && flag("QWEN_R53_STAMP").is_some()
}
/// QWEN_R53_CLOCK=1: SM clock probe before and after the timed replays.
pub fn clock_enabled() -> bool {
    flag("QWEN_R53_CLOCK").is_some()
}
pub fn module_needed() -> bool {
    !null_variants().is_empty() || clock_enabled()
}
pub fn active() -> bool {
    module_needed() || stamp_enabled()
}

/// QWEN_R53_DUP=name[,name]: in the candidate arm, launch each matching call twice
/// in a row (idempotent kernels only; the A/B output check catches others). The
/// second copy runs with the first copy's weights hot in L2. Calls with more than
/// QWEN_R53_DUP_MAX_BLOCKS blocks (default 10000, so not the vocabulary head) are
/// left single.
pub fn dup_names() -> Vec<String> {
    flag("QWEN_R53_DUP")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}
pub fn dup_max_blocks() -> u32 {
    std::env::var("QWEN_R53_DUP_MAX_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000)
}

pub fn stamp_bytes(calls: usize) -> usize {
    16 * (calls + 128)
}

/// Download one replay's stamps and clear the region for the next replay.
pub fn read_stamps(s: &Buffer<'_>, calls: usize) -> Result<Vec<[u64; 2]>> {
    let mut raw = vec![0u8; stamp_bytes(calls)];
    s.download(&mut raw)?;
    s.upload(&vec![0u8; stamp_bytes(calls)])?;
    Ok(raw
        .chunks_exact(16)
        .map(|c| {
            [
                u64::from_le_bytes(c[..8].try_into().unwrap()),
                u64::from_le_bytes(c[8..].try_into().unwrap()),
            ]
        })
        .collect())
}

/// One JSON line per timed group: host time per replay, and when stamps exist,
/// the GPU span (first chain's front-end start to the last completion), each
/// chain's start and each call's median completion, all relative to the first
/// chain's start.
pub fn report(
    what: &str,
    candidate: bool,
    calls: usize,
    stamps: &[Vec<[u64; 2]>],
    host_us: &[f64],
) -> Result<()> {
    let mut sorted = host_us.to_vec();
    sorted.sort_by(f64::total_cmp);
    let host_median = if sorted.is_empty() {
        0.
    } else {
        (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.
    };
    let mut spans = vec![];
    let mut starts_all = vec![];
    let mut ends: Vec<Vec<i64>> = vec![vec![]; calls];
    let mut missing = 0usize;
    for r in stamps {
        ensure!(r.len() == calls + 128, "R53 stamp region size");
        let starts: Vec<[u64; 2]> = r[calls..].iter().copied().filter(|s| s[0] != 0).collect();
        ensure!(!starts.is_empty(), "R53: no chain start stamps");
        ensure!(
            starts.windows(2).all(|w| w[1][0] == w[0][0] + 1),
            "R53: chain start payloads are not consecutive"
        );
        let t0 = starts[0][1] as i64;
        let mut last = i64::MIN;
        for (i, s) in r[..calls].iter().enumerate() {
            if s[0] != i as u64 + 1 || s[1] == 0 {
                missing += 1;
                continue;
            }
            let e = s[1] as i64 - t0;
            ends[i].push(e);
            last = last.max(e);
        }
        spans.push(last as f64 / 1e3);
        starts_all.push(
            starts
                .iter()
                .map(|s| (s[1] as i64 - t0) as f64 / 1e3)
                .collect::<Vec<_>>(),
        );
    }
    let end_median: Vec<i64> = ends
        .iter_mut()
        .map(|v| {
            if v.is_empty() {
                -1
            } else {
                v.sort_unstable();
                v[v.len() / 2]
            }
        })
        .collect();
    let mut span_sorted = spans.clone();
    span_sorted.sort_by(f64::total_cmp);
    let span_median = if span_sorted.is_empty() {
        0.
    } else {
        (span_sorted[(span_sorted.len() - 1) / 2] + span_sorted[span_sorted.len() / 2]) / 2.
    };
    println!(
        "{}",
        json!({"kind":"r53_replays","what":what,"candidate":candidate,"calls":calls,
            "replays":host_us.len(),"host_us":host_us,"host_median_us":host_median,
            "stamped":!stamps.is_empty(),"missing_stamps":missing,
            "gpu_span_us":spans,"gpu_span_median_us":span_median,
            "chain_start_us":starts_all,"end_ns_median":if stamps.is_empty() {vec![]} else {end_median}})
    );
    Ok(())
}

/// Per-call labels, printed once per arm and call count.
pub fn report_meta(candidate: bool, meta: &[(String, [u32; 3], [u32; 3])]) {
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<(bool, usize)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if seen.contains(&(candidate, meta.len())) {
        return;
    }
    seen.push((candidate, meta.len()));
    println!(
        "{}",
        json!({"kind":"r53_meta","candidate":candidate,"calls":meta.len(),
            "names":meta.iter().map(|m| m.0.as_str()).collect::<Vec<_>>(),
            "grid":meta.iter().map(|m| m.1).collect::<Vec<_>>(),
            "block":meta.iter().map(|m| m.2).collect::<Vec<_>>()})
    );
}

/// SM clock: 340 one-warp blocks each spin 200 us of globaltimer and count SM
/// cycles; reports min/median/max MHz over the SMs that ran a block.
pub fn clock(d: &Device, k: &Kernel<'_, '_>, when: &str, candidate: bool) -> Result<()> {
    const BLOCKS: usize = 340;
    let out = d.alloc(BLOCKS * 32)?;
    out.upload(&vec![0u8; BLOCKS * 32])?;
    unsafe {
        k.launch(
            &[Arg::Buffer(&out), Arg::U32(200_000)],
            [BLOCKS as u32, 1, 1],
            [32, 1, 1],
        )?;
    }
    d.sync()?;
    let mut raw = vec![0u8; BLOCKS * 32];
    out.download(&mut raw)?;
    let mut mhz = vec![];
    let mut sms = std::collections::BTreeSet::new();
    for c in raw.chunks_exact(32) {
        let w = |i: usize| u64::from_le_bytes(c[8 * i..8 * i + 8].try_into().unwrap());
        let dt = w(1).saturating_sub(w(0));
        if dt > 0 {
            mhz.push(w(2) as f64 * 1e3 / dt as f64);
            sms.insert(w(3));
        }
    }
    ensure!(!mhz.is_empty(), "R53 clock probe returned nothing");
    mhz.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"kind":"r53_clock","when":when,"candidate":candidate,"blocks":mhz.len(),
            "sms":sms.len(),"mhz_min":mhz[0],"mhz_median":mhz[mhz.len()/2],"mhz_max":mhz[mhz.len()-1]})
    );
    Ok(())
}
