# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.2.0

Version: **1.2.0**. Commit: `177c526f`. Binary SHA-256: `fb0070b38432ecce93cf5f3b444030d1a533c94e9d5bd877c9388197b2691771`. Date: 2026-10-07 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 93.1 | 91.1 | 92.1 | 91.1 | 93.1 |
| Prefill tok/s | 2273.8 | 2212.9 | 2243.4 | 2212.9 | 2273.8 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2112 | 2 | 1894 | 90.4 | 1.1 | 2.91 |
| 4k | 4081 | 2 | 2126 | 80.7 | 1.9 | 2.49 |
| 8k | 8270 | 2 | 2171 | 93.7 | 3.8 | 2.89 |
| 16k | 16276 | 2 | 2112 | 84.2 | 7.7 | 2.94 |
| 32k | 32892 | 2 | 2144 | 79.4 | 15.3 | 3.10 |
| 64k | 65479 | 2 | 2060 | 79.9 | 31.8 | 3.18 |
| 128k | 131152 | 2 | 1975 | 80.6 | 66.4 | 2.90 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.92 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.9× |
| Cached / prompt tokens | 1508 / 1539 |
| Batch streams | 4 |
| Single stream tok/s | 69.0 |
| Aggregate tok/s | 94.4 |
| Batch efficiency | 0.34 |
| Sustained initial / final tok/s | 92.1 / 92.7 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
