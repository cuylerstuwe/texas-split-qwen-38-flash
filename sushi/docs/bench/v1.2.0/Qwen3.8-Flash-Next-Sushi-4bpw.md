# Qwen3.8-Flash-Next-Sushi-4bpw — 1.2.0

Version: **1.2.0**. Commit: `177c526f`. Binary SHA-256: `fb0070b38432ecce93cf5f3b444030d1a533c94e9d5bd877c9388197b2691771`. Date: 2026-10-07 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 77.6 | 92.3 | 85.0 | 77.6 | 92.3 |
| Prefill tok/s | 2565.1 | 2509.4 | 2537.2 | 2509.4 | 2565.1 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2115 | 2 | 2119 | 81.9 | 1.0 | 2.69 |
| 4k | 4079 | 2 | 2362 | 82.1 | 1.7 | 2.76 |
| 8k | 8274 | 2 | 2433 | 91.8 | 3.4 | 3.24 |
| 16k | 16270 | 2 | 2420 | 83.7 | 6.7 | 3.08 |
| 32k | 32897 | 2 | 2363 | 88.2 | 13.9 | 3.31 |
| 64k | 65480 | 2 | 2262 | 84.1 | 28.9 | 3.01 |
| 128k | 131151 | 2 | 2196 | 80.9 | 59.7 | 3.71 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.84 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.5× |
| Cached / prompt tokens | 1507 / 1538 |
| Batch streams | 4 |
| Single stream tok/s | 67.9 |
| Aggregate tok/s | 91.1 |
| Batch efficiency | 0.34 |
| Sustained initial / final tok/s | 85.0 / 78.8 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
