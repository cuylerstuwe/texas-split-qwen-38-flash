# GLM-5.3-Flash-Sushi-2.4bpw — 1.2.1

Version: **1.2.1**. Commit: `975a7694`. Binary SHA-256: `a1b30feb07e2a4c96620b8c73db9be6c49c602aa908ccd40528ec955656108e5`. Date: 2026-10-08 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.4bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 50.9 | 56.5 | 53.7 | 50.9 | 56.5 |
| Prefill tok/s | 856.7 | 838.9 | 847.8 | 838.9 | 856.7 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2079 | 2 | 922 | 48.0 | 2.3 | 2.36 |
| 4k | 4090 | 2 | 822 | 46.2 | 5.0 | 2.29 |
| 8k | 8266 | 2 | 809 | 46.9 | 10.2 | 2.36 |
| 16k | 16308 | 2 | 805 | 45.9 | 20.3 | 2.32 |
| 32k | 32781 | 2 | 807 | 45.7 | 40.6 | 2.30 |
| 64k | 65652 | 2 | 756 | 47.2 | 86.9 | 2.38 |
| 128k | 131092 | 2 | 673 | 45.5 | 194.8 | 2.40 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 3.56 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.1× |
| Cached / prompt tokens | 1384 / 1418 |
| Batch streams | 4 |
| Single stream tok/s | 36.4 |
| Aggregate tok/s | 49.9 |
| Batch efficiency | 0.34 |
| Sustained initial / final tok/s | 53.7 / 49.9 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs.
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
