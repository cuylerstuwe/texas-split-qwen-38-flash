# GLM-5.3-Flash-Sushi-2.4bpw — 1.2.0

Version: **1.2.0**. Commit: `e26fd471`. Binary SHA-256: `ecd387ad520fa8d7c25a3931934292eaf3e77b80dda8bae53c0c3930c5196456`. Date: 2026-10-07 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.4bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 55.9 | 53.3 | 54.6 | 53.3 | 55.9 |
| Prefill tok/s | 870.3 | 854.9 | 862.6 | 854.9 | 870.3 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2078 | 2 | 863 | 47.0 | 2.4 | 2.27 |
| 4k | 4087 | 2 | 831 | 47.9 | 4.9 | 2.33 |
| 8k | 8267 | 2 | 818 | 49.5 | 10.1 | 2.45 |
| 16k | 16311 | 2 | 810 | 49.0 | 20.1 | 2.47 |
| 32k | 32776 | 2 | 790 | 43.4 | 41.5 | 2.25 |
| 64k | 65660 | 2 | 712 | 44.9 | 92.3 | 2.40 |
| 128k | 131085 | 2 | 613 | 42.1 | 213.9 | 2.45 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 3.69 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.3× |
| Cached / prompt tokens | 1388 / 1419 |
| Batch streams | 4 |
| Single stream tok/s | 37.9 |
| Aggregate tok/s | 50.0 |
| Batch efficiency | 0.33 |
| Sustained initial / final tok/s | 54.6 / 53.2 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs.
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
