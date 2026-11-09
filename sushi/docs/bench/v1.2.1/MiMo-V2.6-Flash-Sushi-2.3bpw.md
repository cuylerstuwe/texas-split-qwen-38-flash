# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.2.1

Version: **1.2.1**. Commit: `975a7694`. Binary SHA-256: `a1b30feb07e2a4c96620b8c73db9be6c49c602aa908ccd40528ec955656108e5`. Date: 2026-10-08 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 61.8 | 58.9 | 60.4 | 58.9 | 61.8 |
| Prefill tok/s | 1258.3 | 1238.5 | 1248.4 | 1238.5 | 1258.3 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2091 | 2 | 1221 | 49.1 | 1.7 | 2.82 |
| 4k | 4090 | 2 | 1209 | 54.4 | 3.4 | 2.83 |
| 8k | 8270 | 2 | 1205 | 59.3 | 6.9 | 3.01 |
| 16k | 16330 | 2 | 1143 | 62.4 | 14.3 | 2.85 |
| 32k | 32728 | 2 | 1037 | 64.9 | 31.6 | 2.99 |
| 64k | 65631 | 2 | 901 | 56.2 | 72.8 | 2.40 |
| 128k | 131045 | 2 | 712 | 51.5 | 184.0 | 3.04 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.33 |
| Speculative verdict | effective |
| Prefix cache speedup | 11.2× |
| Cached / prompt tokens | 1535 / 1536 |
| Batch streams | 4 |
| Single stream tok/s | 36.4 |
| Aggregate tok/s | 75.0 |
| Batch efficiency | 0.51 |
| Sustained initial / final tok/s | 60.4 / 58.1 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
