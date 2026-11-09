# Qwen3.8-Flash-Next-Sushi-2bpw — 1.2.0-dev

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 104.9 | 112.6 | · | 108.8 | 104.9 | 112.6 |
| Prefill tok/s | 2389.2 | 2381.0 | · | 2385.1 | 2381.0 | 2389.2 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2112 | 2 | 2013 | 93.8 | 2.84 | 131.6 | 4.55 |
| 4k | 4081 | 2 | 2202 | 100.3 | 3.21 | 123.7 | 3.54 |
| 8k | 8270 | 2 | 2242 | 95.9 | 3.15 | 122.0 | 3.80 |
| 16k | 16276 | 2 | 2141 | 98.6 | 3.00 | 122.0 | 3.81 |
| 32k | 32892 | 2 | 2135 | 84.0 | 3.31 | 125.3 | 5.41 |
| 64k | 65479 | 2 | 2146 | 89.7 | 3.76 | 117.2 | 4.31 |
| 128k | 131152 | 2 | 2061 | 86.0 | 2.78 | 118.5 | 5.65 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 151.4 |
| Novel decode tok/s | 83.9 |
| Speculative ratio | 1.81× |
| Speculative verdict | effective |
| Tokens per decode step | 5.58 |
| Prefix cache speedup | 6.1× |
| Cached / prompt tokens | 1508 / 1539 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 69.2 |
| Aggregate tok/s | 102.3 |
| Batch efficiency | 0.37 |
| Batch verdict | partial |
| Sustained initial / final tok/s | 108.8 / 101.9 |
| Sustained drift | -6.3% |
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
