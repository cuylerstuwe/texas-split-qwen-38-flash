# Architecture

## The model and the split

Qwen3.8-Flash-Next has 48 transformer layers over a 2560-wide hidden state carried as four
hyper-connection streams (10,240 BF16 values per token between layers). Every fourth layer (3, 7, …, 47) is
full attention with QSA block-sparse selection (a lightweight indexer picks up to 2,048 key blocks); the other
36 are gated-delta-net (linear attention) layers with a recurrent state. Each layer has a 512-expert MoE
(top-10) plus a shared expert. Around them: token and n-gram embeddings, a per-layer-embedding block at
layer 2, the final mixer and LM head, a one-layer MTP draft head, and a vision tower for images and video.

texas-split cuts the stack after layer 27:

```
            Apple Silicon Mac (Sushi, MLX/Metal)                    RTX 5090 (stage server, CUDA)
 tokens ─► embeddings, n-gram table, layers 0–27 ──HC stream──► layers 28–47 (5 attention, 15 GDN, MoE)
 sample ◄─ final mixer, LM head, MTP draft head ◄──HC stream───┤
 accept ◄─ (MTP verify rounds) ◄──── top-20 per row ──────────── final mixer, 8-bit LM head, top-20
```

Each row crossing the cut is 20 KB each way over Thunderbolt. For an MTP verify round the 5090 also runs the
final mixer, the 8-bit LM head and a top-20 per row right after layer 47, and returns those 20 candidates
(160 bytes a row) with the stream; the Mac decides acceptance on them instead of running its own head for
those rows ([sushi/FORK.md](../sushi/FORK.md) §6). The 5090 holds layers 28–47's weights, the head, and their
per-sequence state (attention KV in 8-bit affine form, the QSA index, the recurrent states), sized for
262,160 positions; about 0.6 GB of the card stays free. Sushi keeps every layer's weights resident as well,
which is what lets it carry on alone when the 5090 fails.

## Sushi side (`sushi/`)

At layer 28 the forward hook flushes the hyper-connection stream, sends the rows to the stage and resumes
after layer 47 with the stage's output (`src/qwen4_forward.zig`, `native/stage_split.cpp`).

- **Decode** — Sushi decodes with its MTP head: each round verifies 1 + k drafted tokens, so a stage
  forward carries 1–16 rows. When fewer drafts are accepted, the next forward's position tells the stage
  how many rows to keep (a COMMIT folded into the next submission). The stage replays one captured CUDA
  graph per row count, re-targeted with the new parameters each time. On the Mac, a round's work starts
  in pieces: finished layers of the verify go to the GPU while the CPU builds the rest of the graph, each
  draft step starts as soon as it is built, and the verify's captured hidden states are read in the
  round's one batched eval ([sushi/FORK.md](../sushi/FORK.md) §5). The n-gram embedding a verify gathers
  late is filled before the rows go to the stage (§3). With the stage head (§6), the verify's logits are the
  5090's top-20 per row, which is exact for a sampler that reads at most 20 candidates (the thinking
  defaults; other requests keep the Mac's head).
- **Prefill** — prompt chunks (4,096 rows, reshaped near checkpoint boundaries and at the end) are pipelined
  two deep: while the 5090 runs chunk *n* (in 1,536-row pieces), the Mac runs layers 0–27 of chunk *n+1*.
  With each chunk the stage returns its layer state for those rows (the "pull-back"), so Sushi's caches stay
  complete. The prefix cache, MTP history and SSM checkpoints work as without the split, and a later request
  can seed the stage from them. The stage prefills at ~5.6–6.3K tok/s; the Mac's layers 0–27 are the
  bottleneck (~3.5–3.8K tok/s on cold 20K–54K-token prompts). The Mac keeps each dense weight's dequantized
  bf16 copy instead of rebuilding it for every chunk (§7). The last two chunk widths of a prompt are planned
  together, so each Mac chunk covers the stage's work on the previous one and the final chunk is short
  (§4, §7).
- **Seeding** — when the stage does not hold the request's sequence (a new request with a prefix-cache hit,
  say), Sushi uploads its exact committed state for layers 28–47 (about 5.9 KB per position plus 48 MB of
  recurrent state; up to 1.56 GB at full context, through a 2 GiB shared file).
- **Images and video** — vision tokens have 3-D M-RoPE positions, so a prompt with an image or video
  prefills entirely on the Mac; its decode rows are text and go to the stage with the request's rope delta
  (text after an image ropes at position + delta on all three axes).

## Stage side (`stage/`, `driver/`)

`canon-stage-server` runs layers 28–47 with CUDA kernels written for this model on `sm_120` (EXL3 4-bit MoE
projections with tensor cores, dequantize-once attention over 8-bit KV, QSA block selection, gated-delta-net
recurrence with per-row states for partial accepts). It talks to the card through macuda's `libtinynv`
(userspace driver, GSP firmware boot, hardware queues) and tinygrad's TinyGPU DriverKit extension, with a
few driver additions (`driver/README.md`).

## Background parking

The host and Sushi implement foreground/background priority. Holds come from foreground requests on the
host and from leased `/v1/scheduler` controls (for example, a router's preemption adapter); the host
publishes them on a private shared-memory page that Sushi's inference thread polls. When a hold arrives,
the background request stops at its next boundary (a decode round, or a prefill chunk, after landing the
chunk already on the 5090). Then:

1. **The 5090 half parks first** (stage op 7): pending verify rows are committed up to the sequence's
   committed length, then every committed byte of layers 28–47's state is written to
   `stage-request-N.bin` as a seed-format snapshot (all attention KV rows, scales, biases, pooled index
   blocks, open-group keys; all recurrent conv and SSM states).
2. **Sushi parks its own state**: every request-owned MLX allocation goes to `request-N.bin`, with identity,
   offsets and aliases preserved and a SHA-256 per allocation. The RNG state is saved too.
3. Foreground requests run, one at a time, through the split. The hold stays while any foreground request
   is queued or running; it is released when the last one finishes, is cancelled or loses its client.
4. **Restore**: Sushi's allocations come back (checksums verified); the stage half is restored by the
   request's next stage call (op 8), which reloads the snapshot and re-reads the live state to compare it
   byte for byte with the file.

Sushi names each sequence the stage holds (by its KV cache), so another request's first stage call
re-seeds, and the parked one restores instead of re-seeding from Sushi's caches (which lack the rows it
decoded on the stage). A sequence that decoded on the stage is *resident*: if the stage is gone when it
resumes, it re-runs from its tokens on the Mac rather than continuing from stale caches.

Measured: a 192,008-token background request parks its 5090 half (1.15 GB) in 0.38 s and Sushi's half
(2.1 GB) alongside, and restores in 0.58 s; foreground requests start within 1.5 s of the hold. Short
requests switch in 0.2–0.5 s.

## Failure handling

Every stage call has a deadline (20 s by default). Any failure (the stage process dies, a socket error, a
timeout, an error the stage reports, a failed restore check) makes Sushi give the stage up for the rest of
its life and re-run the interrupted request on the Mac: a prefill restarts, a decoding stream re-prefills
its prompt plus the tokens already sent and keeps streaming (the client sees a pause). The host notices,
reports `mode: mac-only`, lets the stage server exit by itself, and when the service has been idle for two
minutes restarts the stack to bring the split back (backing off up to an hour if the card stays
unavailable). A stage that cannot start at all also leaves the service running Mac-only.

## Host (`host/texas_split.py`)

One Python process: the OpenAI-compatible endpoint and request normalization (Qwen's sampling defaults by
thinking mode), the priority controller (`priority_control.py`), and supervision of the stage server and
Sushi. It starts the stage server through macuda's preflight launcher, starts Sushi, and restarts both with
backoff if Sushi crashes. On shutdown it stops Sushi first, so the stage server exits by itself when its
socket closes; signals come only after 90 s. `texas-split stop` signals the host itself and waits for this
teardown before unloading the launchd job (`launchctl bootout` alone kills the job about 20 s after its
SIGTERM, which could leave the GPU lock behind). It keeps the memory/thermal guard (stops the engine below 12
GiB reclaimable memory, on swap or compression growth, or at critical thermal pressure), holds the display
awake around the GPU's cold boot, keeps the Mac awake while running, and pins every binary it runs by
SHA-256.
