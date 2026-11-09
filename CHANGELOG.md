# Changelog

## 2026-11-09

Existing installs: `git pull`, `./install.sh sushi pins`, `bin/texas-split restart`.

### Fixed: MTP verify rounds on the split ran layer 1 with a zero n-gram embedding

This model adds an n-gram per-layer embedding (PLE) at trunk layer 1, which Sushi gathers on the host from
the token ids. A forward built on token ids that are still lazy (an MTP verify on its drafted tokens, and
the pipelined 1-row decode) defers that gather: the embedding starts as zeros and is filled in later. The
split sends layers 0–27's output to the RTX 5090 in the middle of the forward, and that send ran before
the fill. So since the split first went live, every verify round on the split, and with it the sampled
output, used a slightly wrong model: a wrong distribution, not noise.

- Fixed in `sushi/src/qwen4_forward.zig`: the embedding is filled before the rows go to the stage.
- Greedy MTP continuations on the split now follow a Mac-only run for 498 and 743 characters on two long
  prompts (20K and 54K tokens), against 37 and 136 before: about as long as cross-device rounding allows.
- No speed cost.
- The earlier quality numbers (KL vs. a teacher, in the README) measured one-token teacher-forced forwards,
  which never took this path, so they stand for what they measured. They did not cover MTP verify rounds,
  which is why this went unnoticed.

### Faster, with unchanged outputs

Details and switches in [sushi/FORK.md](sushi/FORK.md) §4–5. The host sets them all, and `0` disables
any one. Measured on an M5 Max with an RTX 5090.

- **Decode.**
  - Bit-identical kernel and scheduling changes: 4-row router/gate kernels, a load-hoisted HC kernel,
    verify layers handed to the GPU while the CPU builds the rest, lazy verify captures, and draft steps
    started as soon as they are built.
  - Drafter-side changes, which do not alter the output distribution: a vocab-prefix coarse draft head,
    and a cheaper confidence for the adaptive controller's chain-extension gate.
  - Together roughly −10% time per token at fixed draft depth, plus +2% decode throughput under the
    adaptive controller.
- **Prefill.** Split-prefill chunk shaping cuts the time the two machines wait on each other from 4.8–4.9
  to 2.8–2.9 s per pair of agentic sessions. Only chunk placement changes: each run is deterministic, and
  the numbers move less than switching the chunk width does.

### Host

- `texas-split stop` signals the host itself and waits for its teardown (Sushi, the stage server's card
  put-down, the GPU lock) before unloading the launchd job. `launchctl bootout` alone kills the job about
  20 s after its SIGTERM, which could leave the GPU lock behind with a dead owner.
- `host/tests/fault.py`: `FAULT_NONCE=<anything>` changes the prompt's first tokens, so `fault.py prefill`
  can be rerun cold despite the prefix cache.

### Validation

Priority suite 8/8 exact (with SSD parking), smoke 10/10, `fault.py term` and a cold `fault.py prefill`.
See [docs/VALIDATION.md](docs/VALIDATION.md).
