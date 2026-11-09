# GLM-5.3-Flash-Sushi-2.5bpw — 1.2.0-dev2

Version: **1.2.0-dev2**. Commit: `6731e2db5cace6c2c090dac0305d54a90ec10fb9`. Date: 2026-10-06 (Asia/Bangkok).

Measured runs: **3**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.5bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.5bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 3`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 55.4 | 55.8 | 52.6 | 55.4 | 52.6 | 55.8 |
| Prefill tok/s | 838.2 | 828.9 | 830.0 | 830.0 | 828.9 | 838.2 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2074 | 3 | 852 | 44.7 | 2.29 | 58.1 | 2.95 |
| 4k | 4089 | 3 | 818 | 47.1 | 2.46 | 57.1 | 2.95 |
| 8k | 8268 | 3 | 788 | 45.8 | 2.43 | 56.5 | 2.95 |
| 16k | 16312 | 3 | 774 | 44.2 | 2.37 | 56.5 | 3.00 |
| 32k | 32770 | 3 | 779 | 44.7 | 2.40 | 56.3 | 3.00 |
| 64k | 65665 | 3 | 706 | 41.6 | 2.31 | 53.5 | 2.95 |
| 128k | 131077 | 3 | 609 | 38.3 | 2.31 | 50.6 | 2.95 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 63.9 |
| Novel decode tok/s | 38.6 |
| Speculative ratio | 1.65× |
| Speculative verdict | effective |
| Tokens per decode step | 3.56 |
| Prefix cache speedup | 6.2× |
| Cached / prompt tokens | 1388 / 1420 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 35.2 |
| Aggregate tok/s | 49.6 |
| Batch efficiency | 0.35 |
| Batch verdict | partial |
| Sustained initial / final tok/s | 55.4 / 49.7 |
| Sustained drift | -10.3% |
| Sustained verdict | degraded |

## Probe notes

- custom setup: 3 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs.
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
- 4k: answer never used RETRY_BUDGET_MS — context may be unread.
- 8k: answer never used RETRY_BUDGET_MS — context may be unread.
- 16k: answer never used RETRY_BUDGET_MS — context may be unread.
- 32k: answer never used RETRY_BUDGET_MS — context may be unread.
- 64k: answer never used RETRY_BUDGET_MS — context may be unread.
- 128k: answer never used RETRY_BUDGET_MS — context may be unread.
