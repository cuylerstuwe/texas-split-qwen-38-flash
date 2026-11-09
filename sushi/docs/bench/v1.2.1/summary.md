# Sushi 🍣 1.2.1 benchmarks

Version: **1.2.1**. Date: 2026-10-08 (Asia/Bangkok). Release commit tree: `975a7694` plus the version and CHANGELOG edit.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 3 minutes idle before each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`. Scenario warmups are discarded; the table holds llmprobe medians. Server flags: `--kv-quant 8`, default context and vision, `--mtp` on Qwen and MiMo; GLM uses its automatically detected DFlash2 assistant.

| Sushi 🍣 1.2.1 | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | 1,932 | 2,116 | 2,180 | 2,165 | 2,149 | 2,123 | 2,003 |
|  | Decode | 92.1 | 95.5 | 91.9 | 90.4 | 85.3 | 85.7 | 70.5 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 1,954 | 2,312 | 2,305 | 2,340 | 2,395 | 2,275 | 2,115 |
|  | Decode | 78.5 | 89.4 | 84.3 | 86.1 | 82.1 | 81.0 | 73.4 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 1,221 | 1,209 | 1,205 | 1,143 | 1,037 | 901 | 712 |
|  | Decode | 49.1 | 54.4 | 59.3 | 62.4 | 64.9 | 56.2 | 51.5 |
| **GLM 5.3 Flash-2.4bpw** | Prefill | 922 | 822 | 809 | 805 | 807 | 756 | 673 |
|  | Decode | 48.0 | 46.2 | 46.9 | 45.9 | 45.7 | 47.2 | 45.5 |

| Model run | Measured runs | Speculative mode | Commit | Binary SHA-256 |
|---|---:|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 2 | mtp | `975a7694` | `a1b30feb07e2a4c9` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 2 | mtp | `975a7694` | `a1b30feb07e2a4c9` |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp | `975a7694` | `a1b30feb07e2a4c9` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | 2 | dflash | `975a7694` | `a1b30feb07e2a4c9` |

All four ran in one pass on the release tree `975a7694`, 3 minutes idle between servers. A Qwen 4bpw 128K-only recheck in a fresh boot read 82.2 tok/s decode (2 runs) against the 73.4 above: speculative cells vary across boots. These throughput measurements do not establish long-context answer quality. Raw reports and server logs are indexed in the private measurement ledger.
