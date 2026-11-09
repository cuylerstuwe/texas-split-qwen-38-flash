# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.2.0-dev

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 92.9 | 97.6 | · | 95.3 | 92.9 | 97.6 |
| Prefill tok/s | 2251.4 | 2164.8 | · | 2208.1 | 2164.8 | 2251.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2109 | 2 | 1913 | 90.7 | 2.71 | 120.9 | 4.54 |
| 4k | 4082 | 2 | 2059 | 88.3 | 3.32 | 118.5 | 4.52 |
| 8k | 8271 | 2 | 2129 | 87.4 | 2.71 | 120.5 | 4.71 |
| 16k | 16273 | 2 | 2105 | 88.2 | 2.98 | 114.3 | 3.74 |
| 32k | 32895 | 2 | 2096 | 86.4 | 3.48 | 122.0 | 5.75 |
| 64k | 65478 | 2 | 2051 | 89.4 | 2.91 | 107.4 | 3.92 |
| 128k | 131151 | 2 | 1976 | 85.6 | 2.92 | 95.5 | 2.98 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 148.1 |
| Novel decode tok/s | 75.8 |
| Speculative ratio | 1.95× |
| Speculative verdict | effective |
| Tokens per decode step | 6.00 |
| Prefix cache speedup | 5.6× |
| Cached / prompt tokens | 1509 / 1540 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 65.4 |
| Aggregate tok/s | 94.2 |
| Batch efficiency | 0.36 |
| Batch verdict | partial |
| Sustained initial / final tok/s | 95.3 / 81.6 |
| Sustained drift | -14.4% |
| Sustained verdict | degraded |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
- 2k: answer never used RETRY_BUDGET_MS — context may be unread.
- 4k: answer never used RETRY_BUDGET_MS — context may be unread.
- 8k: answer never used RETRY_BUDGET_MS — context may be unread.
- 16k: answer never used RETRY_BUDGET_MS — context may be unread.
- 32k: answer never used RETRY_BUDGET_MS — context may be unread.
- 64k: answer never used RETRY_BUDGET_MS — context may be unread.
- 128k: answer never used RETRY_BUDGET_MS — context may be unread.
