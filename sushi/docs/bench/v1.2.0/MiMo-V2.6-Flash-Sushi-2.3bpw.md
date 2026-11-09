# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.2.0

Version: **1.2.0**. Commit: `177c526f`. Binary SHA-256: `fb0070b38432ecce93cf5f3b444030d1a533c94e9d5bd877c9388197b2691771`. Date: 2026-10-07 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 54.5 | 40.3 | 47.4 | 40.3 | 54.5 |
| Prefill tok/s | 1282.8 | 1274.3 | 1278.5 | 1274.3 | 1282.8 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2091 | 2 | 1218 | 47.2 | 1.7 | 2.24 |
| 4k | 4090 | 2 | 1222 | 55.3 | 3.3 | 2.92 |
| 8k | 8270 | 2 | 1197 | 59.1 | 6.9 | 2.62 |
| 16k | 16330 | 2 | 1126 | 55.8 | 14.5 | 2.87 |
| 32k | 32728 | 2 | 1033 | 58.6 | 31.7 | 2.91 |
| 64k | 65631 | 2 | 851 | 53.3 | 77.1 | 2.23 |
| 128k | 131045 | 2 | 621 | 46.1 | 211.4 | 2.63 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.82 |
| Speculative verdict | effective |
| Prefix cache speedup | 10.4× |
| Cached / prompt tokens | 1535 / 1536 |
| Batch streams | 4 |
| Single stream tok/s | 24.3 |
| Aggregate tok/s | 78.0 |
| Batch efficiency | 0.80 |
| Sustained initial / final tok/s | 47.4 / 46.2 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
