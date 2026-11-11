<img width="330" alt="image" src="https://github.com/user-attachments/assets/9532d96b-f52e-4451-8ab0-751e06acb89a" />

# texas-split
### Qwen3.8-Flash-Next across an M5 Max and an RTX 5090

**texas-split runs [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on an Apple Silicon
Mac with an NVIDIA RTX 5090 attached over Thunderbolt, splitting the model between them.** The Mac runs
[Sushi](https://github.com/beamivalice/sushi) (included here as a fork) on the
[Sushi 4bpw pack](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (EXL3 4-bit routed experts,
8-bit affine dense weights). Transformer layers 28–47 run on the 5090, which also runs the LM head for
speculative-decoding verify rounds; the rest of the model, the MTP draft head and sampling stay on the Mac. One OpenAI-compatible HTTP endpoint serves it, with tools, reasoning, images,
video and the full 262,144-token context.

No NVIDIA macOS driver is involved. The GPU side uses [macuda](https://github.com/Davinchy/macuda)'s userspace
driver (`libtinynv`, a C port of tinygrad's NVIDIA driver) through tinygrad's signed TinyGPU DriverKit extension.

| | Mac only (stock Sushi) | texas-split |
|---|---|---|
| Decode, agentic coding sessions | ~79–84 tok/s | **~118–124 tok/s** |
| Total task time, same sessions | 1.00× | **1.52–1.59× faster** (pooled 1.57×) |
| Prefill, cold 54K-token prompt (time to first token) | 27.5 s (~2.0K tok/s) | **14.6–16.6 s (~3.3–3.7K tok/s), 1.7–1.9× faster** |
| Several agent sessions at once (opt-in), against the same sessions one after another | not measured | 1.07× with two, 1.16× with four |
| Several requests at once, decode only with short prompts, total | not measured | 120 tok/s with two, 132 with three or four |
| Prefill, cold 64K / 192K-token prompts | not measured at these lengths | 3.1–3.4K / 2.8–3.3K tok/s (earlier build) |
| Quality vs. a teacher model (KL, short panel) | .00458 | .00446 (same level) |

Measured on an M5 Max (128 GB) with an RTX 5090, macOS 27.2, replaying two recorded agentic coding
sessions request by request with the thinking sampler (three seeds; stock Sushi measured with the same
harness and seeds; the texas-split column is the 2026-11-11 build, and runs of one build on different days
differ by a few percent). Speedups come from the 5090 taking 20 of the 48 layers' work and the verify rounds'
LM head; prefill is pipelined between the two machines and limited by the Mac's 28 layers (the 5090 runs
its 20 at ~5.6–6.3K tok/s). Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/VALIDATION.md](docs/VALIDATION.md).

**Updates 2026-11-11:** a fix for replies that failed after a 5090 failure in a long conversation; opt-in
serving of several requests at once (up to four on the 5090 together); follow-up turns keep their place on the
5090 instead of re-sending or re-prefilling state; warm turns of image sessions prefill through the 5090.

**Later on 2026-11-11, a correction and its fix:** the first multi-request numbers (120 and 132 tok/s) were decode
with short prompts. On agent-shaped sessions (20-77K-token contexts, about 40% of the time in prefill) that build
was *slower* than running the same sessions one after another (0.93× with two, about 0.88× with four). Three
exact changes fix it: every request now prefills through the 5090 (only the first one did), requests on the 5090
alternate their rounds so none waits on it at the end of a tick, and a request's rows on the 5090 are sized for
its conversation instead of a fixed share. Side by side is now 1.07× the serial run with two sessions and 1.16×
with four. One request at a time is unchanged (identical replies, equal timing). See
[CHANGELOG.md](CHANGELOG.md); existing installs: `git pull`, `./install.sh stage sushi pins`,
`bin/texas-split restart`.

**Updates 2026-11-10:** the verify rounds' LM head moved to the 5090, two prefill speedups (a bf16 weight
cache and better pipelining of the last chunks), and later the same day ~4% more decode speed: the 5090 runs
each layer's shared expert beside the routed experts, the Mac's GPU stays clocked up while it waits for the
5090, and the split uses Sushi's windowed draft-depth controller. See [CHANGELOG.md](CHANGELOG.md);
existing installs: `git pull`, `./install.sh driver stage sushi pins`, `bin/texas-split restart`.

## What you need

- An **Apple Silicon Mac with 128 GB** of unified memory, macOS 26.2 or later (tested: M5 Max, macOS 27.2).
  The engine needs 82 GiB reclaimable memory when it starts.
- An **RTX 5090** (GB202, 32 GB) in a Thunderbolt eGPU enclosure. At the full context the stage uses all
  but ~0.6 GB of its memory.
- **~175 GB free disk** for the model download (63.7 GiB of weights plus a 95.4 GiB n-gram table that
  stays on the SSD), plus a few GB for builds.
- Xcode command line tools, Homebrew (`webp`, `hf`, Python 3.12+), Rust 1.85+. The installer checks
  everything and fetches the pinned Zig toolchain itself.
- One manual step: approving TinyGPU's system extension in System Settings.

## Install

```sh
git clone --recursive https://github.com/cuylerstuwe/texas-split-qwen-38-flash
cd texas-split-qwen-38-flash
./install.sh check        # reports what is missing; changes nothing
./install.sh              # builds everything, downloads the model (asks first), writes the config
bin/texas-split start     # launchd agent; ~40 s (stage server, then Sushi)
host/tests/smoke.py       # end-to-end check of the endpoint
```

[INSTALL.md](INSTALL.md) walks through every step, what each one needs and how to fix failures.
Coding agents: read [AGENTS.md](AGENTS.md) first.

## Use

The endpoint is `http://127.0.0.1:8003/v1` (OpenAI chat/completions API; model id `texas`,
`texas-split` or `Qwen3.8-Flash-Next-Sushi-4bpw`).

```sh
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"texas","messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

- Thinking is on by default (Qwen's recommended sampling: T=1.0, top-p 0.95, top-k 20). Set
  `"reasoning_effort": "none"` for non-thinking mode.
- **Background requests**: send `X-LLM-Priority: background` (or `X-LocalAI-Mode: background`). While
  any foreground request is queued or running, the background request is paused at a step boundary and
  its state on both the Mac and the 5090 is parked to SSD; it resumes exactly where it stopped when the
  last foreground request finishes. One background request at a time; background requests are text-only.
- **Several requests at once** (off by default): `bin/texas-split concurrency N` (1–16), then
  `bin/texas-split restart`. Up to four requests then run through the 5090 together, prefill and decode, and
  further ones run on the Mac beside them. Two agent sessions side by side finish about 7% sooner than one after
  the other, four about 14% sooner; each request gets its share of the throughput, and while one request prefills
  a long prompt the others pause. The conversations on the 5090 share its 262,144 rows (tokens); one that does
  not fit runs on the Mac. With 1, the default, requests run one at a time exactly as before. Limits are in
  [CHANGELOG.md](CHANGELOG.md).
- `bin/texas-split status | health | logs [--follow] | stop | restart | concurrency [N]`.
- If the 5090 fails (unplugged, crashed, hung), the requests in flight continue on the Mac alone after a
  pause (the Mac re-runs their context at ~2.3K tokens/s: about 7 s for a 16K-token conversation), and the
  split comes back by itself when the service is idle.

## Repository layout

| Path | What it is |
|---|---|
| `install.sh`, `bin/texas-split` | installer and CLI |
| `host/` | the HTTP host and supervisor (`texas_split.py`), priority controller, tests |
| `sushi/` | Sushi v1.3.0 with this project's changes: the pipeline split, 5090 stage client, background parking, the native-priority overlay, and exact prefill/decode speedups ([sushi/FORK.md](sushi/FORK.md)) |
| `stage/` | the 5090 stage server (Rust + CUDA kernels; prebuilt cubins included) |
| `driver/` | macuda's `libtinynv` with this project's patches |
| `third_party/macuda` | macuda (submodule, pinned): the GPU preflight and launch tooling |
| `docs/` | architecture, validation results and evidence |

## Status

This is a single-machine research deployment that has been qualified on one hardware setup. Not covered:
sustained multi-hour load, sleep/wake with the stage running, a physically disconnected card (simulated
by killing or freezing the stage server), recovery of parked requests after a crash or reboot. Serving several
requests at once is newer than the rest: its exactness, caching, priority and failure handling were tested, its
end-to-end smoke test was not adapted yet.

## Credits and licenses

Built on [Sushi](https://github.com/beamivalice/sushi) (MIT), [macuda](https://github.com/Davinchy/macuda)
(MIT), [tinygrad](https://github.com/tinygrad/tinygrad) (MIT), [MLX](https://github.com/ml-explore/mlx) (MIT)
and the Qwen3.8-Flash-Next weights (Qwen community license; downloaded at install, not included). The
vendored and adapted components keep their licenses: see [NOTICE](NOTICE).
