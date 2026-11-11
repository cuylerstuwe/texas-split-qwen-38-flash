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

`build.zig` links the two native objects (and, since §8, `lib/gpu_warm.m`).

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

## 6. The head on the 5090: stage head, sparse acceptance, commit inference

With `SUSHI_R64_STAGE_HEAD=1` (`src/generate.zig`, `src/qwen4_forward.zig`, `native/stage_split.cpp`) an MTP
verify forward on the split asks the stage (op 1 aux bit 2, `QD_STAGE_HEAD=1` on the stage) to run the final
HC mixer, the 8-bit lm_head (248,320 × 2560) and a top-20 per row right after layer 47. The verify logits are
those 20 values per row (−inf elsewhere) and the Mac skips its own mixer + head for them.

- Engaged only where it is exact: a sampler that reads at most the top 20 (top-k ≤ 20, or greedy) with no
  presence/repeat penalty, logit bias or think penalty — Qwen's thinking defaults. MTP already excludes
  logprobs, grammars and forced calls. A row whose top 20 holds a suppressed special falls back to the
  Mac head for that forward.
- `SUSHI_R64_SPARSE_ACCEPT=1`: the acceptance graph runs on those rows (`sparseAcceptGraph`): the same
  filter as `probsAllPositions` (temperature, top-k, the f32 nucleus with its exclusive scan, a precise
  softmax) and the same accept/residual/bonus law, over 20 candidates instead of the vocabulary. The law
  is unchanged; the categorical draws differ, so a seeded sampled run's text differs from older builds
  (greedy text does not).
- `SUSHI_STAGE_INFER_COMMIT=1`: no separate COMMIT round trip; the stage infers a partial accept from the
  next FORWARD's position (and PARK commits the same way).
- The stage needs `QD_STAGE_PREFILL_ROWS=1536` at the 262K capacity for the head's 0.71 GB to fit.
- −0.78 ms per decode round in the production configuration (the Mac's post-verify eval 2.22 → 0.58 ms).

## 7. Prefill: dequant cache and whole-tail chunk planning

- `SUSHI_R64_DQ_CACHE_MB=<MiB>` (`src/transformer.zig`, `prefillDqGemm`): keep each 8-bit dense weight's
  dequantized bf16 copy for later prefill chunks instead of re-dequantizing it for every chunk (keyed by
  the packed weight's and scales' buffer addresses, shape and dtype; each entry holds a reference to the
  packed weight). Same operands for the same GEMM: bit-identical. ~4.1 GB for the split's 28 Mac layers;
  time to first token −5% per agentic session pair (−10% on a cold 54K prompt).
- `SUSHI_R64_SPLIT_PLAN=1` (`src/generate.zig`, `splitPlanTail`): within the last two chunk widths of a
  split prefill, the chunks are planned together: the current chunk is sized so that it, then the chunk
  that just covers the stage's work on it, and so on down to a ~384-row final chunk, sum to the rows left.
  Interior checkpoint boundaries in that stretch are skipped (the end-of-prompt snapshot always lands).
  Removes the stalls the per-chunk rules of §4 left (a boundary cutting a 4,557 + 163-row pair; a small
  remainder after a big chunk; a 1.3–1.4K-row final chunk draining). Chunk placement only: end-of-prefill
  logprobs move as much as switching the chunk width does.

## 8. Mac GPU warm-up while the stage runs

- `SUSHI_R65_GPU_WARM=1` (`native/stage_split.cpp`, `lib/gpu_warm.m`, `build.zig`): while a split decode
  forward's reply is awaited, the stage client waits in 0.5 ms `select()` slices and commits a 256-thread
  kernel on a Metal queue of its own between them. An idle Apple GPU clocks down within a few ms (on an M5
  Max the same work after a 3–12 ms gap ran ~40% slower than back to back), and the Mac idles ~6.5 ms per
  decode round while the 5090 runs layers 28–47, so the post-stage work used to start on a slow GPU. The tick
  touches only its own 1 KB buffer: outputs are unchanged. Post-stage eval 0.57 → 0.31 ms per round; prefill
  and time to first token unchanged. `2` alternates the warm-up per 32-round trace window
  (`[r65-warm] window warm=N` with `SUSHI_MTP_TRACE=1`) for in-process A/Bs. A socket fd at or above
  `FD_SETSIZE` takes the plain `poll()` wait without the warm-up.

## 9. Instrumentation and switches that are off

- Decode PLE gather timing (`src/qwen4_exp.zig`, `notePleGather`): printed as `[r64-ple]` with
  `SUSHI_MTP_TRACE=1`. `SUSHI_R64_PLE_RESIDENCY=1` counts how many rows sit in resident pages.
- Measured and left off (each exact): `SUSHI_R64_SSD_KEEPALIVE_MS` (an uncached 4 KiB read every N ms
  while generating), `SUSHI_R64_PLE_WALK` / `SUSHI_R64_PLE_HYBRID` / `SUSHI_R64_PLE_LEAN` (other ways to
  gather decode-sized PLE rows), `SUSHI_R64_DEFER_CAPTURES` (verify captures outside the accept batch),
  `SUSHI_R64_CPU_ACCEPT` (the sparse acceptance on the host). None paid on this machine.

## 10. Stage state across turns: incremental seed, end-of-turn pull-back, kept checkpoints

- **Incremental seed** (`SUSHI_STAGE_ISEED=1`, set by the host; `native/stage_split.cpp`, `src/qwen4_forward.zig`
  `stageNotePrompt`). The stage keeps the rows of the sequence it last held. The client notes each request's
  prompt and restore point and tracks how many rows the stage holds exactly as Sushi does (rows seeded from
  Sushi, or split-prefill chunks pulled back; decode rows never count). A seed then sends rows [base, position)
  and the recurrent states only (stage op 10, ~50 MB instead of ~450 MB at 70K tokens), with base = that count
  capped at the restore point. The stage hashes its K codes of the 16 rows below base against Sushi's and refuses
  before changing anything on a mismatch; the client then seeds in full. `SUSHI_STAGE_ISEED_TRACE=1` logs the
  bookkeeping.
- **End-of-turn pull-back** (`SUSHI_STAGE_PULLBACK_END=1`, set by the host; `stagePullbackTurnEnd`). After a
  split turn the client commits the last forward's rows and asks the stage for the decode-time rows and
  recurrent states (stage op 9); Sushi imports them and checkpoints at the trunk's end, so the next turn resumes
  after the reply and the cache entry can persist to SSD whole. The MTP head has to reach the same position (its
  deferred stash applied, the last pair appended) or the next turn could not adopt it; the checkpoint goes in
  only where it does.
- **Kept checkpoints** (`SUSHI_HOT_KEEP_CPS=1`, set by the host; `src/prefix_cache.zig`). A hot prefix-cache
  entry keeps every recurrent-state checkpoint (at most 16, thinned) instead of the newest two, so a request
  that diverges mid-history restores from the nearest one. Restore points only: the restore path is Sushi's own.

## 11. Image sessions: text rows prefill through the stage

`SUSHI_STAGE_MROPE_TEXT=1` (set by the host; `src/generate.zig`, `src/qwen4_forward.zig` `mropeTextDelta`). A
request with an image anywhere in its prompt has M-RoPE positions and used to prefill entirely on the Mac. When
every row a turn prefills is text at one rotary offset (checked from the 4-row QSA group the first new row
completes), the split prefill sends those rows to the stage with that offset in the PREFILL message's aux bits,
as decode rows already do, and the MTP head's split-path history gets the request's M-RoPE context. Image rows and
cold image prompts stay on the Mac.

## 12. Several requests at once: stage slots and interleaved rounds

Active only with `--max-concurrent` above 1 (the host's `TEXAS_SPLIT_MAX_CONCURRENT`); with 1 none of this runs.

- **Stage client** (`native/stage_split.cpp`): up to four sequences (`SUSHI_STAGE_SLOTS`), one per stage slot,
  each with its own position, token mirror and head buffer; the slot rides in bits 3-5 of a message's aux word.
  Forwards can be sent and received separately (`stage_split_forward_send` / `_recv`), one outstanding per slot;
  every write to the shared file first drains the other slots' outstanding replies, and no stage call may go out
  from inside another call's wait.
- **Forward** (`src/qwen4_forward.zig`): with `ctx.stage_async_verify` a verify forward returns after the send at
  the stage boundary and `forwardQwen4Resume` finishes it after the receive (`src/generate.zig`:
  `mtpRoundVerifyPre` / `mtpRoundVerifyPost`).
- **Scheduler** (`src/scheduler.zig`): `stageHolderClaim` places a request on slot 0, on a free extra slot it
  fits (`SUSHI_STAGE_SLOT_ROWS`, with `SUSHI_STAGE_SLOT_RESERVE` rows of headroom for the reply), or on the Mac.
  `runDecodeTick` collects the slot holders: two or more run through `runSplitGroup` (every holder's first half
  in turn, the Mac-side requests, then the second halves); one holder runs alone with the Mac-side requests
  decoding inside its stage wait (`SUSHI_STAGE_MS_OVERLAP`, on). Draft counts are capped by the slot's verify
  rows (`SUSHI_STAGE_SLOT_VERIFY_ROWS`). `stageSlotGuard` moves an extra slot's request to the Mac before a
  request crosses a slot boundary (its rows are pulled back first), `stageHolderRelease` passes a freed slot to
  the Mac-side request with the most tokens left, and a request left on a narrower slot moves to slot 0.
  Extra-slot holders sit a tick out while slot 0's request has split-prefill chunks on the stage (one stream of
  replies on the socket). A background request takes slot 0 only when nothing else is on the stage, gives it up
  while parked and claims it again on resume.
- Switches for experiments: `SUSHI_STAGE_MS_HOLDER=0` (every request on the Mac), `SUSHI_STAGE_MS_TRACE=1` (where
  a multi-request tick's time goes), `SUSHI_STAGE_FORCE_SLOT1=1` (every holder on slot 1 of a two-slot stage).

## 13. Stage-failure recovery with decode ticks inside a re-run

`src/scheduler.zig`. A decoding request interrupted by a stage failure re-runs from its tokens on the Mac, which
is a prefill, and decode ticks run between a prefill's chunks. Those ticks used to take the request being re-run
(no generator until its prefill ends) and failed it, so a re-run longer than one chunk ended in
`generation failed`. Now `stageMarkRecover` marks the requests to re-run (one request at a time: every request of
the failed tick; several: the ones on stage slots or resident on the stage), `interleaveDecodeTick` leaves marked
requests out, and `stageRecoverMarked` re-runs them one after another. A stage error inside a tick hosted by
another request's prefill marks them for the next pass instead of failing every stream of that tick. Requests
decoding on the Mac keep their state and carry on. The reset before a re-run clears the MTP head's live state only
when it is the request's own (`headStateIsSlots`).

## 14. Several requests at once on long prompts: prefill on every slot, rounds per tick, slot rows per request

Active only with `--max-concurrent` above 1. Section 12 was measured on decode with short prompts; with long
prompts side by side it lost time to running the same requests one after another (docs/VALIDATION.md).

- **Split prefill on every slot** (`native/stage_split.cpp`, `src/generate.zig`): the prefill pipeline works on
  the request's own stage slot (op 6 carries the slot; the chunks in flight remember theirs), where a request on an
  extra slot used to prefill every layer on the Mac and seed the slot afterwards. `SUSHI_STAGE_SLOT_PREFILL=0`
  restores that. In `runDecodeTick` every slot holder now sits a tick out while any request's prefill has chunks
  on the stage (one stream of replies on the socket).
- **Several rounds per tick** (`src/scheduler.zig`, `runSplitGroup`): after the first halves, each holder's second
  half is followed at once by its next first half, so only the tick's last second half waits for the 5090.
  `SUSHI_STAGE_GROUP_ROUNDS` (default 4, at most 6 so a slot cannot grow past the guard's margin between two
  checks; 1 = one round per tick). A tick hosted inside a prefill runs one round; so does a tick with a foreground
  request waiting for admission; a tick ends early once a holder has ended. `=alt` alternates 1 and 4 every 128
  rounds and `SUSHI_STAGE_MS_TRACE=1` then prints each arm's time per round.
- **Slot rows per request** (`stageExtraClaim`, `stageRegionPick` / `stageRegionFind` / `stageRegionSet`; stage op
  12 through `stage_split_slot_view`): an extra slot's rows are set when a request claims it: its tokens, a reply
  reserve that halves from `SUSHI_STAGE_SLOT_RESERVE` until the request fits, and the guard margin, on a 4,096-row
  grid, placed at the highest free rows above slot 0's request. `stageMoveOrEvictExtra`: a request that outgrows
  its rows gets more at the same base when the rows above are free, else moves to other free rows (its state is
  pulled back and its next forward seeds the new rows), else continues on the Mac. Slots and rows freed inside a
  stage wait pass to Mac-side requests at the next tick's start (`stagePromoteFree`), since setting rows is a
  stage call. Rows a finished conversation left on slot 0 are forgotten by the stage when they would block a
  slot's rows. `SUSHI_STAGE_SLOT_DYNAMIC=0` keeps the equal shares of section 12.
- **Ended slots are freed before the next claim** (`stageDropPending`): a slot freed inside a stage wait is only
  marked, and the stage call that frees its rows ran at the next tick's start. When the last request of a burst
  ended on an extra slot there was no next tick: the stage kept counting its rows, and a later request on slot 0
  whose prompt reached them was refused mid-prefill, which gave the stage up. The pending frees now also run before
  every slot claim.
- Also: a request whose extra slot was taken back during a prefill on the Mac no longer keeps the stage without a
  slot; a move to slot 0 needs 256 tokens left to generate; the stage client reads the server's own time per op
  from a reply's bytes 8-12 (the shipped server leaves them zero; a measurement build filled them: transport =
  send-to-reply minus that, 16-27 us per forward, so polling the socket instead of sleeping has nothing to win).
