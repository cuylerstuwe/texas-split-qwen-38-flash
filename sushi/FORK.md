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
