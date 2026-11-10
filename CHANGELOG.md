# Changelog

## 2026-11-10, later: memory guard fix

Existing installs already on the 2026-11-10 release below: `git pull`, `bin/texas-split restart` (host only;
nothing to rebuild).

### Fixed: the memory guard could stop an idle engine

The host stops the engine when memory gets tight (and answers 503 until `bin/texas-split restart`). One of
its signals is compression growth: more than 16 GiB of compressor above its recent low. It could trip on
the engine's own memory:

- About 61–63 s after Sushi goes idle, macOS unwires its Metal memory: wired falls from ~70–77 GiB to
  ~5–6 GiB, and Sushi rewires it within 1–2 s of the next request. The kernel may then compress those pages
  when another app needs room.
- That is the engine giving memory back, but the guard counted compression growth until 5 minutes of idle.
  On our machine the compressor grew 17.7 GiB in one 6 s step at ~97 s idle (swap flat, ~29 GiB still
  reclaimable), the guard tripped, and the service stayed down until restarted.

Now (`MemoryGuard` in `host/texas_split.py`):

- While the engine is idle and wired memory is below half the most seen while it worked, compression growth
  is not counted. The host logs one line per idle period when it would have tripped.
- Unchanged: the 12 GiB reclaimable floor, the 256 MiB swap-growth check (not reset by unwiring), the
  5-minute idle reset and the critical-thermal stop.
- Compression growth counts again from the moment a request arrives.
- `memory.jsonl` rows in the run directory carry `engine_unwired`.

`host/tests/test_guard.py` checks the rule without a running service. Synthetic cases cover trips while
working or still wired, the reclaimable, swap and thermal trips while unwired, the 5-minute rule, and a short
request that ends before Sushi rewires. Pass run directories to replay their `memory.jsonl` through the old
and new rules.

## 2026-11-10

Existing installs: `git pull`, `./install.sh stage sushi pins`, `bin/texas-split restart` (the stage server
changed too this time).

### Faster decode: the verify's head on the 5090

For an MTP verify round, the RTX 5090 now also runs the final mixer, the 8-bit LM head (248,320 × 2560) and
a top-20 per row right after layer 47, and returns those 20 candidates with the stream. The Mac takes them as
the verify's logits and computes the acceptance law on them, instead of running its own head over the whole
vocabulary. Details in [sushi/FORK.md](sushi/FORK.md) §6.

- Exact where it engages: a sampler that reads at most the top 20 candidates (top-k ≤ 20 or greedy) with no
  presence/repeat penalty, logit bias or think penalty. That is the thinking-mode default (T=1.0,
  top-p 0.95, top-k 20); other requests keep the Mac's head, as does a round whose top 20 contains a
  reserved special token.
- The acceptance law is the same, so outputs follow the same distribution. Its random draws are taken over
  the 20 candidates, so a seeded sampled run's text differs from older builds (greedy text does not).
- One decode round is ~0.8 ms shorter (the Mac's post-verify work 2.2 → 0.6 ms).
- The head needs 0.71 GB on the card. At the full 262K context it fits with the stage's prefill pieces at
  1,536 rows (the host sets `QD_STAGE_PREFILL_ROWS=1536`; time to first token is unchanged); ~0.6 GB stays
  free.

### Faster prefill

- **Dequant cache** (bit-identical): the Mac keeps each 8-bit dense weight's bf16 copy instead of
  rebuilding it for every 4,096-row chunk. ~4.1 GB more resident memory once the first long prompt has
  run. Time to first token −5% per pair of agentic sessions, −10% on a cold 54K-token prompt.
- **Whole-tail chunk planning**: the last two chunk widths of a split prefill are planned together, so each
  of the Mac's chunks covers the 5090's work on the previous one and the final chunk is short. Interior
  checkpoint boundaries there are skipped (the end-of-prompt snapshot always lands). This removes waits of
  up to 0.7 s per follow-up turn (e.g. a checkpoint boundary that split a 4,720-row prompt into 4,557 + 163
  rows, the small chunk then waiting 703 ms for the 5090). Six-turn agentic pair: summed time to first token
  33.6–34.7 → 31.4 s. Chunk placement only: end-of-prefill logprobs move by about as much as switching the
  chunk width does.

### Host

- `QD_STAGE_HEAD=1` and `QD_STAGE_PREFILL_ROWS=1536` for the stage; `SUSHI_R64_STAGE_HEAD`,
  `SUSHI_R64_SPARSE_ACCEPT`, `SUSHI_STAGE_INFER_COMMIT`, `SUSHI_R64_DQ_CACHE_MB=6144` and
  `SUSHI_R64_SPLIT_PLAN` for Sushi. Each can be set to `0`.
- `host/tests/priority.py`: `TEXAS_SPLIT_PRIORITY_EXTRA` adds request fields (JSON) to every case, e.g.
  `{"presence_penalty": 0}` to run the non-thinking cases through the stage head.

### Validation

Priority suite 8/8 exact with SSD parking after each change, including a run with presence 0 on every case
so all 2,894 verify forwards used the stage head; smoke 10/10; `fault.py term`; a cold `fault.py prefill`.
See [docs/VALIDATION.md](docs/VALIDATION.md).

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
