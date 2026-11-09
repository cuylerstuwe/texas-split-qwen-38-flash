# Sushi 🍣 1.2.0-dev2 — GLM benchmarks

Version: **1.2.0-dev2**. Runtime commit: `6731e2db5cace6c2c090dac0305d54a90ec10fb9`. Date: 2026-10-06 (Asia/Bangkok).

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 60 s idle between servers. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k,128k --runs 3`. Scenario warmups are discarded; the table contains llmprobe medians. Server flags: `--kv-quant 8`, default context, prefix cache and vision; automatic A4 g64 DFlash2. Two draft nodes plus root, with default prompt lookup enabled.

The packs use MCG W12 routed experts at K2.25 and K2.5 respectively, an A6 g128 trunk, and A4 g64 assistants. Their tokenizer, chat template, generation defaults and assistant configuration are identical.

Binary SHA-256: `25265f7a4b320fff502f7d09cab572a1a4fa0816b9a33fd0355354abe6531de0`. Binary mtime: 2026-10-06T10:33:48.615872+07:00.

| Sushi 🍣 1.2.0-dev2 | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **GLM 5.3 Flash-2.3bpw** | Prefill | 865 | 842 | 805 | 798 | 788 | 718 | 624 |
|  | Decode | 47.3 | 49.0 | 48.0 | 42.8 | 45.5 | 47.0 | 43.5 |
| **GLM 5.3 Flash-2.5bpw** | Prefill | 852 | 818 | 788 | 774 | 779 | 706 | 609 |
|  | Decode | 44.7 | 47.1 | 45.8 | 44.2 | 44.7 | 41.6 | 38.3 |

Standalone scenario medians (the short code prompt, predictable text and novel prose):

| Model | Runs | Decode tok/s | Prefill tok/s | Spec mode | Predictable tok/s | Novel tok/s | Tok/step |
|---|---:|---:|---:|---|---:|---:|---:|
| GLM-5.3-Flash-Sushi-2.3bpw | 3 | 51.1 | 850.0 | dflash | 66.6 | 41.7 | 3.69 |
| GLM-5.3-Flash-Sushi-2.5bpw | 3 | 55.4 | 830.0 | dflash | 63.9 | 38.6 | 3.56 |

| Model report | Measured runs | Speculative mode |
|---|---:|---|
| [GLM-5.3-Flash-Sushi-2.3bpw](GLM-5.3-Flash-Sushi-2.3bpw.md) | 3 | dflash |
| [GLM-5.3-Flash-Sushi-2.5bpw](GLM-5.3-Flash-Sushi-2.5bpw.md) | 3 | dflash |

The [prior dev baseline](../v1.2.0-dev/summary.md) is inherited without rerunning its binary. These are throughput measurements, not long-context answer-quality results. Per-model reports retain probe caveats; raw reports, server logs and thermal records are indexed in the private measurement ledger.

The sustained-load checks reported **−13.3%** decode drift for 2.3bpw and **−10.3%** for 2.5bpw. Both servers ran with thinking enabled and the same 192-token decode budget. The probe’s reasoning-detection banner differs because 2.3bpw answered its simple detection question without a reasoning segment; that banner does not change these timed decode budgets.

## Pack comparison

Both packs engage the same optimized three-row verification and drafting paths: expert reuse/tiling, MLA value
reuse and fused normalization, followed by the same A4 g64 drafter optimizations. Their routed weight precision
selects the normal K2.25 or K2.5 specialization; neither pack misses an optimized path.

The larger stock decode gaps at 64K and 128K coincide mainly with fewer accepted tokens per step on 2.5bpw.
These are medians of the three code requests' server profiles; the ceiling requests are excluded.

| Context | 2.3bpw verify ms | 2.5bpw verify ms | 2.3bpw draft ms | 2.5bpw draft ms | 2.3bpw tok/step | 2.5bpw tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 64K | 47.41 | 48.83 | 4.99 | 5.04 | 2.53 | 2.31 |
| 128K | 50.58 | 52.87 | 5.09 | 5.20 | 2.49 | 2.31 |

The verification difference is about 3.0% at 64K and 4.5% at 128K, smaller than the corresponding 11.5% and 12.0%
decode gaps. The completed stock ladders above are the reported benchmarks. An optional repeat with matched
cache-busting prefixes was stopped after confirming the shared code paths and is not used for a paired claim.
