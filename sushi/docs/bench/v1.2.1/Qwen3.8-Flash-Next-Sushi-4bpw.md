# Qwen3.8-Flash-Next-Sushi-4bpw — 1.2.1

Version: **1.2.1**. Commit: `975a7694`. Binary SHA-256: `a1b30feb07e2a4c96620b8c73db9be6c49c602aa908ccd40528ec955656108e5`. Date: 2026-10-08 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 89.0 | 96.7 | 92.9 | 89.0 | 96.7 |
| Prefill tok/s | 2520.6 | 2479.2 | 2499.9 | 2479.2 | 2520.6 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2112 | 2 | 1954 | 78.5 | 1.1 | 2.56 |
| 4k | 4081 | 2 | 2312 | 89.4 | 1.8 | 3.30 |
| 8k | 8270 | 2 | 2305 | 84.3 | 3.6 | 3.31 |
| 16k | 16276 | 2 | 2340 | 86.1 | 7.0 | 2.75 |
| 32k | 32892 | 2 | 2395 | 82.1 | 13.7 | 3.10 |
| 64k | 65479 | 2 | 2275 | 81.0 | 28.8 | 2.65 |
| 128k | 131152 | 2 | 2115 | 73.4 | 62.0 | 3.70 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.67 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.0× |
| Cached / prompt tokens | 1508 / 1539 |
| Batch streams | 4 |
| Single stream tok/s | 63.5 |
| Aggregate tok/s | 89.7 |
| Batch efficiency | 0.35 |
| Sustained initial / final tok/s | 92.9 / 75.3 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
