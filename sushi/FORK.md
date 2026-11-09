# This Sushi fork

This directory is [Sushi](https://github.com/beamivalice/sushi) at **v1.3.0** (commit
`8e4c0af1f5d239dcbed3d58728259432c86f1d96`, MIT) with the changes below. `FORK.patch` is the complete diff
from v1.3.0. Sushi's own documentation in this directory (README.md, docs/, CLAUDE.md) describes upstream
Sushi; for building and running texas-split use the repository root's README.md and INSTALL.md.

Build (what `install.sh mlx sushi` does): the MLX runtime libraries from the v1.3.0 release bundle in
`lib/mlx/lib`, the pinned MLX headers (`lib/mlx-src`, `lib/mlxc-src`), Zig 0.17.0 from
`scripts/fetch-zig.sh`, then

```sh
xcrun clang++ -std=c++20 -O2 -mmacosx-version-min=26.2 -Ilib/mlx-src -Ilib/mlxc-src -Inative -c native/stage_split.cpp -o native/stage_split.o
xcrun clang++ -std=c++20 -O2 -mmacosx-version-min=26.2 -Ilib/mlx-src -Ilib/mlxc-src -Inative -c native/texas_checkpoint.cpp -o native/texas_checkpoint.o
.zig-toolchain/zig build -Doptimize=ReleaseFast --prefix "$PWD/zig-out"
```

## 1. Pipeline split: layers 28–47 on an RTX 5090 stage

- `native/stage_split.cpp` — the client of the stage server (`../stage`): a Unix socket carrying 32-byte
  messages and a 2 GiB shared file for activation streams and state. Seeds the stage from Sushi's caches,
  sends decode/verify forwards (the HC residual stream after layer 27, BF16) and receives the stream after
  layer 47, infers partial MTP accepts from positions, pipelines prefill chunks two deep with an optional
  pull-back of the stage's layer state into Sushi's caches. Every socket wait is bounded
  (`SUSHI_STAGE_TIMEOUT_MS`); any failure gives the stage up for the rest of the process
  (`SUSHI_STAGE_DEAD_FILE` receives the reason). Sequences are named by owner (the slot's KV cache) so a
  parked background request's stage state can be saved (op 7 PARK) and restored (op 8 RESTORE); a
  sequence that decoded on the stage is "resident" and never continues from Sushi's stale caches.
- `src/qwen4_forward.zig` — the forward hook: at layer `SUSHI_STAGE_FIRST` the rows go to the stage
  (decode, or a split-prefill chunk) and come back after layer 47; seeds or restores the stage when it
  does not hold the sequence; image/video (M-RoPE) prompts prefill on the Mac and decode on the stage with
  the request's rope delta.
- `src/generate.zig` — the split prefill loop (pipelined chunks, pull-back imports, two-step SSM
  checkpoints), and landing the in-flight chunk before a background request parks.
- `src/scheduler.zig` — fatal-stage fallback: a stage error re-runs the interrupted request on the Mac
  (a prefill restarts; a decoding stream re-prefills prompt + streamed tokens and continues); the stage
  half of background parking.
- `src/kld.zig`, `src/transformer.zig` — the split in KL-divergence measurements; small hooks.
- Environment (set by the host): `SUSHI_STAGE_SOCKET`, `SUSHI_STAGE_SHM`, `SUSHI_STAGE_FIRST`,
  `SUSHI_STAGE_DEAD_FILE`, `SUSHI_STAGE_TIMEOUT_MS`; switches `SUSHI_STAGE_DECODE=0`,
  `SUSHI_STAGE_MROPE=0` (images/video entirely on the Mac), `SUSHI_STAGE_PULLBACK=0`. Without
  `SUSHI_STAGE_SOCKET` this build behaves as Sushi v1.3.0.

## 2. Native foreground/background priority with exact SSD parking

- `src/texas_priority.zig`, `native/texas_ipc.c`, `native/texas_checkpoint.cpp` (+ `_test.cpp`) — a
  private shared-memory control page from the host, and lossless parking of a background request's MLX
  allocations to SSD (identity, offsets and aliases preserved; SHA-256 checked on restore) at a prefill
  chunk or decode boundary, while foreground requests run on the same inference thread.
- `src/scheduler.zig`, `src/server.zig`, `src/generate.zig` — admission (foreground first; one background,
  never batched with foreground work), the park/restore boundaries, `X-LocalAI-Mode: background`.
- `docs/server-texas-priority.md`, `docs/envvar.md` — the contract; `TEXAS_PRIORITY_IPC` and
  `TEXAS_PRIORITY_CHECKPOINTS` enable it (set by the host). Without them this build behaves as Sushi v1.3.0.
- The parking helper depends on the exact MLX 0.32.3 library of the Sushi v1.3.0 release bundle (it reads
  MLX's random-number state by symbol name); a different MLX build fails closed.

`build.zig` links the two native objects.

## 3. Bug fix: fill the deferred PLE leaf before the stage send

- `src/qwen4_forward.zig` — this model's trunk layer 1 adds an n-gram per-layer embedding (PLE) that
  Sushi gathers on the host from the token ids. A forward built on lazy token ids (an MTP verify on its
  drafts, the pipelined 1-row decode) defers that gather: the leaf is created zero-filled and filled later.
  The stage send evaluates layers `[0, SUSHI_STAGE_FIRST)`, and it ran before the fill, so every MTP verify
  on the split computed layer 1 with a zero n-gram embedding: a wrong verify distribution, not noise. The
  leaf is now filled at the stage boundary (`plePreEval`, the same gather, earlier). Greedy MTP
  continuations on the split now follow a Mac-only run for ~120–180 tokens (498 / 743 characters on a 20K
  and a 54K prompt) instead of 9–33 (37 / 136). Always on; no speed cost.

## 4. Split-prefill chunk shaping

- `src/generate.zig` `splitChunkEndFull` — reshapes the split prefill's chunk ends, which removes most of
  the time the pipeline spends waiting for the stage:
  - a first chunk that would end on an SSM checkpoint boundary within `SUSHI_R63_SPLIT_FRAG` rows (default
    1024) merges into the next chunk (that interior checkpoint is skipped; the end-of-prompt snapshot
    still lands);
  - a chunk ending within `SUSHI_R63_SPLIT_FRAG` rows before a boundary extends to it (no sliver chunk);
  - within 1.5 chunks of the end, each chunk is sized so the Mac's layers cover the stage's work on the
    previous chunk, and a leftover under `SUSHI_R63_SPLIT_LAST` rows (default 384) rides with the final
    chunk;
  - a shaped chunk never grows past 6144 rows (the stage's per-chunk slot holds 6553).
- Chunk placement only: each run is deterministic, and the numbers move by less than switching the chunk
  width does (any prefix-restore point moves them the same way). Measured on agentic two-session replays:
  pipeline stalls 4.8–4.9 → 2.8–2.9 s per session pair. `0` for either variable restores the plain policy.

## 5. Decode-round speedups

Each is exact (bit-identical outputs in fixed-depth MTP runs) unless noted; each switch unset or `0` keeps
the old path. The host sets all of them.

- `SUSHI_MTP_DENSE_ROWS=1` — Sushi's own opt-in 4-row router/gate kernels.
- `src/qwen4_hc.zig` — HC up kernel U2: the same arithmetic with every load issued first; on by default
  (`SUSHI_HC_U2=0` off). With dense rows: −2–3% per token.
- `src/mtp.zig`, `src/transformer.zig` — `SUSHI_MTP_DRAFT_HEAD_ROWS=N`: the coarse MTP draft head scans
  only vocab ids `[0, N)` (rounded to 1024; BPE ids are roughly in merge order, so the prefix holds the
  common tokens). Proposal side only: acceptance stays exact, so the output distribution is unchanged.
  −5% per token at `N=98304`. `mtpCoarsePairs` takes any 8-row multiple; `SUSHI_R63_COARSE_CHECK=1`
  self-checks the coarse readouts.
- `SUSHI_R64_VERIFY_LADDER=<stride>` (`src/qwen4_forward.zig`) — for split verify forwards (2–16 rows),
  `mlx_async_eval` at layer 0 and every `stride` layers hands finished layers to the GPU while the CPU
  builds the rest (the PLE leaf is filled before any point past layer 0). −0.5 ms per round at stride 4.
- `SUSHI_R64_VERIFY_NOSETTLE=1` (`src/generate.zig`, `ForwardCtx.capture_settle` in `src/transformer.zig`)
  — the verify's hidden-state captures are evaluated in the round's batched eval instead of two blocking
  evals right after the forward.
- `SUSHI_R64_CHAIN_LADDER=1` (`src/generate.zig`) — each built MTP draft step starts on the GPU while the
  CPU builds the next. With the previous switch: −0.5 ms per round.
- `SUSHI_R64_SHORTLIST_CONF=1` (`src/generate.zig`) — when the adaptive draft controller considers
  extending a chain, its gate needs ln p(draft) for the first steps; that came from a full 248K-row head
  readout per draft step and now comes from the 32-candidate rerank shortlist. Drafter side only (the gate
  only sizes the draft). +2% decode throughput under the adaptive controller (three seeds).
