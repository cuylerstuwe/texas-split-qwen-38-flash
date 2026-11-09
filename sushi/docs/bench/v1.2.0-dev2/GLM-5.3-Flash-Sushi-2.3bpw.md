# GLM-5.3-Flash-Sushi-2.3bpw — 1.2.0-dev2

Version: **1.2.0-dev2**. Commit: `6731e2db5cace6c2c090dac0305d54a90ec10fb9`. Date: 2026-10-06 (Asia/Bangkok).

Measured runs: **3**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 3`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 49.9 | 51.1 | 51.4 | 51.1 | 49.9 | 51.4 |
| Prefill tok/s | 864.6 | 850.0 | 847.0 | 850.0 | 847.0 | 864.6 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2078 | 3 | 865 | 47.3 | 2.34 | 58.5 | 2.91 |
| 4k | 4087 | 3 | 842 | 49.0 | 2.40 | 59.7 | 2.95 |
| 8k | 8267 | 3 | 805 | 48.0 | 2.43 | 58.5 | 2.95 |
| 16k | 16311 | 3 | 798 | 42.8 | 2.21 | 57.3 | 2.95 |
| 32k | 32777 | 3 | 788 | 45.5 | 2.40 | 56.7 | 3.00 |
| 64k | 65657 | 3 | 718 | 47.0 | 2.53 | 54.2 | 2.95 |
| 128k | 131088 | 3 | 624 | 43.5 | 2.49 | 52.4 | 3.00 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 66.6 |
| Novel decode tok/s | 41.7 |
| Speculative ratio | 1.60× |
| Speculative verdict | effective |
| Tokens per decode step | 3.69 |
| Prefix cache speedup | 6.4× |
| Cached / prompt tokens | 1388 / 1419 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 36.8 |
| Aggregate tok/s | 49.3 |
| Batch efficiency | 0.33 |
| Batch verdict | serialized |
| Sustained initial / final tok/s | 51.1 / 44.3 |
| Sustained drift | -13.3% |
| Sustained verdict | degraded |

## Probe notes

- custom setup: 3 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs.
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
- 2k: answer never used RETRY_BUDGET_MS — context may be unread.
- 4k: answer never used RETRY_BUDGET_MS — context may be unread.
- 16k: answer never used RETRY_BUDGET_MS — context may be unread.
- 128k: answer never used RETRY_BUDGET_MS — context may be unread.
