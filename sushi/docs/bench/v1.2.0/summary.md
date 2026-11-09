# Sushi 🍣 1.2.0 benchmarks

Version: **1.2.0**. Date: 2026-10-07 (Asia/Bangkok). Release commit tree: `e26fd471` plus docs and test edits.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 3 minutes idle before each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`. Scenario warmups are discarded; the table holds llmprobe medians. Server flags: `--kv-quant 8`, default context and vision, `--mtp` on Qwen and MiMo; GLM uses its automatically detected DFlash2 assistant.

| Sushi 🍣 1.2.0 | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | 1,894 | 2,126 | 2,171 | 2,112 | 2,144 | 2,060 | 1,975 |
|  | Decode | 90.4 | 80.7 | 93.7 | 84.2 | 79.4 | 79.9 | 80.6 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 2,119 | 2,362 | 2,433 | 2,420 | 2,363 | 2,262 | 2,196 |
|  | Decode | 81.9 | 82.1 | 91.8 | 83.7 | 88.2 | 84.1 | 80.9 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 1,218 | 1,222 | 1,197 | 1,126 | 1,033 | 851 | 621 |
|  | Decode | 47.2 | 55.3 | 59.1 | 55.8 | 58.6 | 53.3 | 46.1 |
| **GLM 5.3 Flash-2.4bpw** | Prefill | 863 | 831 | 818 | 810 | 790 | 712 | 613 |
|  | Decode | 47.0 | 47.9 | 49.5 | 49.0 | 43.4 | 44.9 | 42.1 |

| Model run | Measured runs | Speculative mode | Commit | Binary SHA-256 |
|---|---:|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 2 | mtp | `177c526f` | `fb0070b38432ecce` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 2 | mtp | `177c526f` | `fb0070b38432ecce` |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp | `177c526f` | `fb0070b38432ecce` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | 2 | dflash | `e26fd471` | `ecd387ad520fa8d7` |

Qwen and MiMo ran on `177c526f`; the later commits change only GLM kernels, tests and docs, so their paths are the release binary's. GLM ran on the release tree `e26fd471`. These throughput measurements do not establish long-context answer quality. Raw reports and server logs are indexed in the private measurement ledger.
