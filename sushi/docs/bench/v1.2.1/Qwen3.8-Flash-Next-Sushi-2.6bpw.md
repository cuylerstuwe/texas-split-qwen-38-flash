# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.2.1

Version: **1.2.1**. Commit: `975a7694`. Binary SHA-256: `a1b30feb07e2a4c96620b8c73db9be6c49c602aa908ccd40528ec955656108e5`. Date: 2026-10-08 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 88.2 | 91.5 | 89.9 | 88.2 | 91.5 |
| Prefill tok/s | 2265.2 | 2225.0 | 2245.1 | 2225.0 | 2265.2 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2115 | 2 | 1932 | 92.1 | 1.1 | 3.00 |
| 4k | 4079 | 2 | 2116 | 95.5 | 1.9 | 2.83 |
| 8k | 8274 | 2 | 2180 | 91.9 | 3.8 | 2.92 |
| 16k | 16270 | 2 | 2165 | 90.4 | 7.5 | 3.25 |
| 32k | 32897 | 2 | 2149 | 85.3 | 15.3 | 3.50 |
| 64k | 65480 | 2 | 2123 | 85.7 | 30.8 | 3.11 |
| 128k | 131151 | 2 | 2003 | 70.5 | 65.5 | 3.21 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.34 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.9× |
| Cached / prompt tokens | 1507 / 1538 |
| Batch streams | 4 |
| Single stream tok/s | 69.1 |
| Aggregate tok/s | 96.9 |
| Batch efficiency | 0.35 |
| Sustained initial / final tok/s | 89.9 / 87.2 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
