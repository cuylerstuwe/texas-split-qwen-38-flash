# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.2.0-dev

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 37.1 | 48.5 | · | 42.8 | 37.1 | 48.5 |
| Prefill tok/s | 1249.5 | 1274.4 | · | 1262.0 | 1249.5 | 1274.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2091 | 2 | 1216 | 55.0 | 2.38 | 84.3 | 3.76 |
| 4k | 4090 | 2 | 1190 | 57.2 | 2.79 | 85.7 | 3.69 |
| 8k | 8270 | 2 | 1192 | 58.8 | 2.34 | 79.4 | 3.26 |
| 16k | 16330 | 2 | 1122 | 57.8 | 2.91 | 71.8 | 3.50 |
| 32k | 32728 | 2 | 1039 | 64.8 | 3.05 | 80.7 | 3.92 |
| 64k | 65631 | 2 | 898 | 58.1 | 2.87 | 74.2 | 3.88 |
| 128k | 131045 | 2 | 691 | 50.9 | 2.87 | 62.7 | 3.92 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 53.7 |
| Novel decode tok/s | 49.2 |
| Speculative ratio | 1.09× |
| Speculative verdict | marginal |
| Tokens per decode step | 3.21 |
| Prefix cache speedup | 9.8× |
| Cached / prompt tokens | 1535 / 1536 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 31.9 |
| Aggregate tok/s | 76.4 |
| Batch efficiency | 0.60 |
| Batch verdict | partial |
| Sustained initial / final tok/s | 42.8 / 55.3 |
| Sustained drift | 29.2% |
| Sustained verdict | improved |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
- 4k: answer never used RETRY_BUDGET_MS — context may be unread.
- 8k: answer never used RETRY_BUDGET_MS — context may be unread.
