# GLM-5.3-Flash-Sushi-2.5bpw — 1.2.0-dev

Version: **1.2.0-dev**. Commit: `73a9659c38f4818f399bdd2cc32309e348a3077c`. Date: 2026-10-05 (Asia/Bangkok).

Measured runs: **3**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, binary stamp, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.5bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.5bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 3`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 44.7 | 45.7 | 42.7 | 44.7 | 42.7 | 45.7 |
| Prefill tok/s | 864.1 | 848.1 | 845.5 | 848.1 | 845.5 | 864.1 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | Tok/step | Speculative ceiling tok/s | Ceiling tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | 2074 | 3 | 870 | 43.1 | 2.49 | 51.9 | 2.95 |
| 4k | 4089 | 3 | 843 | 40.9 | 2.37 | 50.8 | 2.91 |
| 8k | 8268 | 3 | 822 | 42.6 | 2.49 | 51.2 | 3.00 |
| 16k | 16312 | 3 | 814 | 42.2 | 2.53 | 51.2 | 3.00 |
| 32k | 32770 | 3 | 807 | 40.5 | 2.43 | 50.3 | 3.00 |
| 64k | 65665 | 3 | 738 | 40.1 | 2.46 | 49.5 | 3.00 |
| 128k | 131077 | 3 | 655 | 38.5 | 2.43 | 47.5 | 2.95 |

llmprobe saves aggregate context medians rather than individual rung sample rates. Per-request timings remain in the raw server logs.

## Additional benchmark results

| Metric | Value |
|---|---|
| Predictable decode tok/s | 58.5 |
| Novel decode tok/s | 34.8 |
| Speculative ratio | 1.68× |
| Speculative verdict | effective |
| Tokens per decode step | 3.56 |
| Prefix cache speedup | 6.1× |
| Cached / prompt tokens | 1388 / 1420 |
| Prefix cache verdict | active |
| Batch streams | 4 |
| Single stream tok/s | 33.5 |
| Aggregate tok/s | 47.0 |
| Batch efficiency | 0.35 |
| Batch verdict | partial |
| Sustained initial / final tok/s | 44.7 / 42.1 |
| Sustained drift | -5.8% |
| Sustained verdict | steady |

## Probe notes

- custom setup: 3 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs.
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort.
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work.
- 2k: answer never used RETRY_BUDGET_MS — context may be unread.
- 4k: answer never used RETRY_BUDGET_MS — context may be unread.
- 8k: answer never used RETRY_BUDGET_MS — context may be unread.
- 16k: answer never used RETRY_BUDGET_MS — context may be unread.
- 32k: answer never used RETRY_BUDGET_MS — context may be unread.

## Grok harness configuration

Add this entry to `~/.grok/config.toml`, then select it with
`grok --model sushi-glm53-2.5bpw --reasoning-effort high` while Sushi serves this pack on port 12345.
GLM accepts `low`, `high` and `max`; the explicit effort avoids a global Qwen `xhigh` setting.

```toml
[model."sushi-glm53-2.5bpw"]
model = "GLM-5.3-Flash-Sushi-2.5bpw"
base_url = "http://127.0.0.1:12345/v1"
name = "GLM 5.3 Flash [T:I:V · sushi 2.5bpw]"
description = "Local sushi GLM-5.3-Flash-Sushi-2.5bpw, 1M ctx, DFlash2"
api_backend = "chat_completions"
api_key = "sushi"
context_window = 1048576
max_completion_tokens = 65536
supports_reasoning_effort = true
reasoning_effort = "high"

[[model."sushi-glm53-2.5bpw".reasoning_efforts]]
id = "low"
value = "low"
label = "Low"
default = false

[[model."sushi-glm53-2.5bpw".reasoning_efforts]]
id = "high"
value = "high"
label = "High"
default = true

[[model."sushi-glm53-2.5bpw".reasoning_efforts]]
id = "max"
value = "max"
label = "Max"
default = false
```

The context window is the model's advertised limit; Sushi checks actual memory admission per request.
This harness configuration was added after the llmprobe benchmark and does not change its measurement settings.
