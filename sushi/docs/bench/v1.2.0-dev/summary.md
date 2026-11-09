# Sushi 🍣 1.2.0-dev benchmarks

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 60 s idle after stopping each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k,128k --runs <n>`. GLM was already running with 3 measured runs when the request changed; subsequent Qwen and MiMo runs use 2. Scenario warmups are discarded; the table contains llmprobe medians. Server flags: `--kv-quant 8`, default context and vision, `--mtp` on Qwen and MiMo; GLM uses its automatically detected DFlash2 assistant.

Binary SHA-256: `d45b324457338d7a25ff0ae901224e0f3215db2f210e6708a47fa31f6cc26ad5`. Binary mtime: 2026-10-05T19:37:36.643203+07:00.

| Sushi 🍣 1.2.0-dev | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **GLM 5.3 Flash-2.5bpw** | Prefill | 870 | 843 | 822 | 814 | 807 | 738 | 655 |
|  | Decode | 43.1 | 40.9 | 42.6 | 42.2 | 40.5 | 40.1 | 38.5 |
| **Qwen 3.8 Flash-Next-2bpw** | Prefill | 2,013 | 2,202 | 2,242 | 2,141 | 2,135 | 2,146 | 2,061 |
|  | Decode | 93.8 | 100.3 | 95.9 | 98.6 | 84.0 | 89.7 | 86.0 |
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | 1,913 | 2,059 | 2,129 | 2,105 | 2,096 | 2,051 | 1,976 |
|  | Decode | 90.7 | 88.3 | 87.4 | 88.2 | 86.4 | 89.4 | 85.6 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 2,048 | 2,327 | 2,361 | 2,366 | 2,279 | 2,245 | 2,198 |
|  | Decode | 80.2 | 73.7 | 88.3 | 85.4 | 81.9 | 81.1 | 80.5 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 1,216 | 1,190 | 1,192 | 1,122 | 1,039 | 898 | 691 |
|  | Decode | 55.0 | 57.2 | 58.8 | 57.8 | 64.8 | 58.1 | 50.9 |

| Model run | Measured runs | Speculative mode |
|---|---:|---|
| [GLM-5.3-Flash-Sushi-2.5bpw](GLM-5.3-Flash-Sushi-2.5bpw.md) | 3 | dflash |
| [Qwen3.8-Flash-Next-Sushi-2bpw](Qwen3.8-Flash-Next-Sushi-2bpw.md) | 2 | mtp |
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 2 | mtp |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 2 | mtp |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp |

The v1.1.0 Qwen 4bpw release cell is retained as the inherited reference; no old binary was rerun and no speedup claim is made across the changed run counts. These throughput measurements do not establish long-context answer quality. Raw reports and server logs are indexed in the private measurement ledger.
