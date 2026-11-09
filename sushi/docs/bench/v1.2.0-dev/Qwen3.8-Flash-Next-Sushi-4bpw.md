# Qwen3.8-Flash-Next-Sushi-4bpw — 1.2.0-dev

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 92.1 | 93.4 | · | 92.7 | 92.1 | 93.4 |
| Prefill tok/s | 2413.3 | 2361.8 | · | 2387.5 | 2361.8 | 2413.3 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2109 | 2 | 2048 | 80.2 | 3.26 | 115.8 | 5.06 |
| 4k | 4082 | 2 | 2327 | 73.7 | 2.95 | 108.0 | 4.58 |
| 8k | 8271 | 2 | 2361 | 88.3 | 3.18 | 112.4 | 4.81 |
| 16k | 16273 | 2 | 2366 | 85.4 | 3.20 | 113.2 | 4.86 |
| 32k | 32895 | 2 | 2279 | 81.9 | 2.63 | 115.3 | 4.75 |
| 64k | 65478 | 2 | 2245 | 81.1 | 3.18 | 105.4 | 4.38 |
| 128k | 131151 | 2 | 2198 | 80.5 | 2.85 | 106.8 | 4.81 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 129.6 |
| Novel decode tok/s | 77.4 |
| Speculative ratio | 1.67× |
| Speculative verdict | effective |
| Tokens per decode step | 5.00 |
| Prefix cache speedup | 5.6× |
| Cached / prompt tokens | 1509 / 1540 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 67.4 |
| Aggregate tok/s | 88.6 |
| Batch efficiency | 0.33 |
| Batch verdict | serialized |
| Sustained initial / final tok/s | 92.7 / 87.1 |
| Sustained drift | -6.0% |
| Sustained verdict | steady |

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
