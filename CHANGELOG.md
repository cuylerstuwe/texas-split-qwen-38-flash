# Changelog

## 2026-11-13: a reached thinking budget lost the 5090 (fixed)

Existing installs: `git pull`, `./install.sh sushi pins`, `bin/texas-split restart` (Sushi changed; the stage server
and the driver did not).

A request with a thinking budget (`reasoning_budget_tokens`, or a `reasoning_effort` that maps to one) whose thought
reached the budget made Sushi give the 5090 up. The host logged `split lost: Sushi gave up on the stage: split decode
failed in the scheduler`, every request continued on the Mac, and the stack restarted when idle. The budget was then
ignored for that request: its reply was thought up to `max_tokens`, with no answer. One such request was enough,
with one request at a time too. It has been there since the first release; nothing had used a thinking budget until
a client did, and the install this is developed on then lost the 5090 18 times in a day.

- **Cause.** At the budget Sushi commits an early-stop line and the think closer through the model as one
  multi-token forward (`Generator.commitForcedTokens`; loop recovery inside a thought uses it too). That forward does
  not pass the decode entry points that route layers 28-47 to the stage, so a request whose state for those layers
  lives on the 5090 failed the forward's check and the scheduler treated it as a stage failure. The bound had
  already been marked as fired, so the re-run on the Mac never applied it.
- **Fix** (`stageBringBack` in `sushi/src/scheduler.zig`). Before the forced commit the request's rows of layers
  28-47 come back into Sushi's caches (the end-of-turn pull-back, stage op 9, ~22 ms). The forced forward runs on
  the Mac, and the request's next forward seeds the stage again incrementally (13-15 ms, also at a 19K-token
  context). The bound is marked fired only once the state is back, so a request that a real stage failure re-runs
  on the Mac still meets its bound there. Requests without a budget execute nothing new.
- **Unchanged**, and upstream Sushi's behaviour: after a reached budget the rest of that reply decodes without MTP
  drafts (about 65-75 tok/s alone).
- `smoke.py` has an eleventh case, `think_budget`: a greedy request with a 48-token budget must stop its thought at
  the budget, answer, and leave the host in the mode it was in.

### Validation

On the previous build one request with a 64-token budget reproduces it: the split is lost and the reply is 465
thinking tokens with no answer. On this build, with one request at a time 5 of 5 replies and with four at once 9 of
9 (three budget requests on slots beside a plain one; a 19K-token prompt and its second turn) end with the thought
closed at the budget and an answer, the host in split mode throughout; a greedy request repeated gives the same
thought and answer. The parking suite is 8/8 exact with one request at a time and with four, smoke passes in both,
and the greedy four-turn conversation is identical to the previous build's. On the installed service afterwards:
the budget test 9 of 9, then 17 hours of ordinary use in which 9 more requests reached their budgets, no split
lost to this; smoke 11/11 with the new case. Details in [docs/VALIDATION.md](docs/VALIDATION.md).

### Known, not fixed here

With several requests at once, a parked background request that resumes while a foreground request holds slot 0
gives the 5090 up ("resumed background without slot 0"; its request re-runs on the Mac and the stack restarts when
idle). Seen twice in three days of use.

## 2026-11-11, second update: several requests at once now pays off on long prompts (a correction)

Existing installs: `git pull`, `./install.sh stage sushi pins`, `bin/texas-split restart` (the stage server and
Sushi changed; the driver did not).

Everything here is active only above `concurrency 1`. One request at a time is unchanged: against the previous
build, a greedy four-turn conversation and 30 seeded sampled replies (thinking sampler) are identical, and cold and
warm timings are equal in an interleaved run.

### The correction

The multi-request figures in the first update below (120 tok/s with two requests, 132 with three or four) were
measured on decode with ~500-token prompts. Agent sessions are different: long prompts, and about 40% of the time
in prefill. Replaying two recorded agent sessions side by side (A: three turns adding 20K, 8K and 5K prompt tokens;
B: 54K, 10K, 10K; 1,024 tokens generated per turn, thinking sampler) against the same sessions one after another
on the same host, that build was slower:

| | first update | this update |
|---|---|---|
| two sessions side by side, against one after the other | 0.93× (96.1 s against 89.3 s) | **1.07×** (87.8 s against 94.0 s) |
| four sessions (A, A, B, B), against two serial pairs | about 0.88× (200.5 s against ~175 s) | **1.16×** (162.5 s against ~188 s) |

Single runs with one seed; the columns were measured at different times of one day on a Mac that was also in
interactive use (the serial pair itself took 86-94 s), so compare within a column.

### What was wrong, and the three changes (all exact)

- **A request on an extra slot prefilled on the Mac alone**: all 48 layers at ~2.3K tok/s instead of the split's
  ~3.6K, while the requests already decoding got one decode round per 4,096-token chunk. The stage's prefill now
  runs on any slot (its layers already worked on the selected slot; the server only refused the op off slot 0)
  and Sushi's prefill pipeline works on the request's own slot. While any request's prefill has chunks on the
  5090, the requests decoding there pause (one stream of replies on the socket; the Mac is the shared resource
  either way). `SUSHI_STAGE_SLOT_PREFILL=0` in `TEXAS_SPLIT_SUSHI_ENV` restores the old path.
- **Every tick ended with the last request waiting for the 5090**: 5-8 ms of a ~40 ms tick with two requests at
  20-54K contexts. A tick now runs several rounds (`SUSHI_STAGE_GROUP_ROUNDS`, default 4): each request's second
  half is followed at once by its next first half, so only the tick's last second half waits. Measured with both
  arms alternating inside one process: 39.8 -> 34.4 ms per round with two requests, +15.8% rounds per second.
- **Extra slots were fixed equal shares of the stage's rows** (65,536 each with four slots), so a conversation
  past its share ran on the Mac. A slot's rows are now set when a request claims it (stage op 12, SLOT VIEW): its
  tokens, a reply reserve that shrinks from 16,384 rows until the request fits, and a margin, taken from whatever
  rows are free. A request that outgrows its rows gets more in place when the rows above are free, else moves to
  other free rows (its state is pulled back and seeded again), else continues on the Mac. The stage refuses rows
  that overlap another conversation's, at the op and at every seed. `SUSHI_STAGE_SLOT_DYNAMIC=0` keeps the fixed
  shares. The conversations on the 5090 together still share its 262,144 rows.

With four requests on the 5090 at 20-77K contexts a round takes ~69 ms: 15 ms per request for the drafts, the
Mac's layers 0-27 and the send, ~2.4 ms per request for the rest. That is the same Mac time per round as a request
running alone, so what is left is the Mac's verify forward itself; the 5090 is busy about 35% of the time.

### Fixed: a long request after a burst could lose the 5090

A slot freed while another request waited on the 5090 was only marked, and its rows were freed on the stage at
the next decode tick. When the last request of a burst ended on an extra slot there was no next tick: the stage
went on counting its rows, and a later request on slot 0 whose prompt reached them was refused mid-prefill, which
made Sushi give the 5090 up and re-run the request on the Mac (it needed a prompt longer than roughly 262K tokens
minus the leftover conversation). Present since the first update. The pending frees now also run before every
slot claim. Reproduced with a small row limit (two requests, then a 51.6K-token one: 40.5 s to first token on the
Mac) and fixed (the same sequence prefills through the 5090 in 14-16 s).

Also fixed: a request whose extra slot was taken back during a prefill on the Mac kept permission to use the 5090
without a slot.

### Host and tests

- The launch memory check (82 GiB reclaimable) re-reads for up to 45 s before it fails. The host's own restart
  after a 5090 failure stops the engine and launches again at once; a reading taken two seconds after the stop,
  before the kernel had reclaimed the engine's memory, left the service in `fault` until restarted by hand.
- `/health` and `bin/texas-split concurrency` report `slot_regions: "dynamic"` above concurrency 1.
- `smoke.py`: `queue_concurrent` checks that a second request starts while the first runs when several requests
  are allowed (it expected one at a time and failed by design). `tools_stream` accepts either offered tool and asks
  again, twice at most, when a reply makes no tool call. That is the model under the thinking sampler, not the
  engine: over 300 requests each with plain decode, adaptive MTP and fixed-depth MTP, 5.0-5.7% of replies called
  `run_shell` before `read_file` and 1.0% made no call at all (they reason about a constraint that is not in the
  prompt), the same in all three.

### Tried and dropped

Polling the stage socket instead of sleeping on it. The round trip adds only 16-27 us to the stage's own time per
forward (measured with a server build that reports that time in its reply; the stage client reads the field, which the
shipped server leaves at zero), so there was nothing
to win.

### Validation

With one request at a time and with four at once: the parking suite 8/8 exact, smoke 10/10. Prefill through slots
1-3: four staggered greedy requests, cold, byte-identical to their solo runs; a four-turn conversation forced onto
slot 1 identical to the reference. Slot rows under pressure (a small row limit): one slot grew in place and one
moved mid-reply, every reply (up to 8,000 tokens) identical to its solo run. 5090 failures with several requests:
four killed mid-reply, a hang with three, and the 5090 dying while a second request prefilled a 56K-token prompt
on an extra slot: every stream paused and continued exactly. After the install: `fault.py term`, `prefill` and
`long` clean, the host restoring the split by itself each time. Details in
[docs/VALIDATION.md](docs/VALIDATION.md).

## 2026-11-11, first update: several requests at once (opt-in), the cache across turns, a failure-recovery fix

Existing installs: `git pull`, `./install.sh stage sushi pins`, `bin/texas-split restart` (the stage server and
Sushi changed; the driver did not).

With one request at a time, which stays the default, replies are identical to the previous build's (a greedy
four-turn conversation) and decode speed is the same. The agentic replay on this build (two recorded sessions, the
thinking sampler, 1,024 tokens per turn, three seeds): total task **1.565×** faster than stock Sushi pooled
(1.52–1.59× by seed; 250.9 s against 260.6 s for the previous release), decode 120.5 tok/s, a cold 54K-token
prompt's first token in 14.6–16.6 s. Part of the gain over the previous release is the end-of-turn pull-back
below (+2.3% in a same-session A/B); the rest is within what one build varies between sessions.

### Fixed: a 5090 failure mid-reply in a long conversation ended the request with an error

When the 5090 path fails, Sushi re-runs the interrupted request on the Mac from its tokens and keeps streaming.
That re-run is a prefill, and decode ticks run between a prefill's chunks. They picked up the very request being
re-run, which has no generator until its prefill ends, and failed it: any conversation longer than one prefill
chunk (4,096 tokens) ended in `generation failed` instead of a pause. `fault.py term` uses a short prompt and
never showed it. The scheduler now marks the requests to re-run and keeps them out of those ticks.
`host/tests/fault.py long` covers it: the stage killed 13 s into the reply of a 35.8K-token conversation, a 15.8 s
pause, and the reply (a copy task) continued exactly.

### Several requests at once (opt-in)

```sh
bin/texas-split concurrency 4     # 1-16; writes TEXAS_SPLIT_MAX_CONCURRENT to texas-split.env
bin/texas-split restart
```

Above 1, up to four requests decode through the 5090 at once and any further ones run every layer on the Mac
beside them (Sushi's continuous batching). The stage holds up to four conversations in slots: slot 0 is the
stage's whole state as before, each extra slot a view into the top of slot 0's KV buffers with its own recurrent
state. A decode round is split in two: every slot holder's first half (drafts, layers 0-27 on the Mac, the rows
sent to the 5090) runs in turn, then the second halves (receive, accept, sample), so each request's 5090 time
hides behind the Mac's work on the others. A request alone runs exactly the single-request path.

Total throughput at the thinking sampler (seeded, ~500-token prompts, 1,000-token replies, tok/s over the wall,
mean of two seeds):

| requests at once | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| up to four on the 5090 (this release) | 102.6 | 120.1 | 131.6 | 131.7 | 122.9 |
| one on the 5090, the rest on the Mac | 101.7 | 98.7 | 95.9 | 98.5 | 114.3 |

Each request is slower than it would be alone (four share ~132 tok/s); the total is what rises. A slot holder's
reply is byte-identical to the same request run alone (greedy, same prefill path). Each conversation keeps its own
prefix-cache entry: two 65K-token conversations at concurrency 2 reach the first token of warm turns in 0.7-1.2 s
with the whole prompt restored. Background requests behave as before: one at a time, alone on slot 0, parked
exactly while any foreground request runs.

These totals are decode with short prompts; on agent-shaped sessions this build lost time to running the sessions
one after another (see the second update above, which also removes the first three limits below).

Limits: a request on an extra slot prefills on the Mac, and extra-slot holders pause while slot 0's request
prefills through the 5090; extra slots have a fixed share of the stage's rows (131,080 each with two slots, 87,384
with three, 65,536 with four) and a request that outgrows its slot moves to the Mac (its state is pulled back
first); with three or four slots the later slots verify 4 rows per round instead of 9 and prefill pieces shrink to
1,024 or 768 rows, which is what fits the card; `smoke.py`'s `queue_concurrent` check expects one request at a time
and fails by design above 1. If the 5090 fails, only the requests on it re-run on the Mac, one after another; the
ones already on the Mac keep streaming.

### The cache across turns

- **Incremental stage seed** (`SUSHI_STAGE_ISEED`, stage op 10). Every turn re-sent layers 28-47's whole state to
  the 5090 (~450 MB, ~100 ms at 70K tokens) although the stage still held those rows. A request whose prompt
  shares rows with what the stage holds now sends only the rows after them and the recurrent states (~50 MB). The
  stage hashes its rows just below that point against Sushi's and refuses on a mismatch (then the full seed runs).
  Warm seeds 13-15 ms instead of 45-53 ms at 24K tokens; every reply identical to the full-seed build's.
- **End-of-turn pull-back** (`SUSHI_STAGE_PULLBACK_END`, stage op 9). A reply's rows of layers 28-47 lived only on
  the 5090, so the next turn prefilled the previous reply again and the conversation's cache entry could not be
  written to SSD. After each split turn those rows and the recurrent states are copied back (~48 MB, ~25 ms, after
  the response) and a checkpoint is taken at the reply's end. Agentic replay, three seeds, same binaries: total
  task 1.458x -> 1.491x stock Sushi, time to first token lower on every seed.
- **Hot cache entries keep their checkpoints** (`SUSHI_HOT_KEEP_CPS`). A request that diverges from a cached
  conversation mid-history (an agent's compaction prompt, a pruned tool result) restarts from the nearest
  checkpoint instead of from 0. The split's idle cache allowance is 2 GB.

### Image sessions: warm turns prefill through the 5090

A prompt with an image used to prefill entirely on the Mac, on every turn of the session. When every row a turn
prefills is text at one rotary offset (no new image), those rows now go through the stage with that offset
(`SUSHI_STAGE_MROPE_TEXT`). Image rows and cold image prompts stay on the Mac. Four images and a 19K-token cold
turn, then ~7K-token text turns: time to first token 3.58-3.71 s -> 2.46-2.62 s.

### Host and tests

- The TinyGPU server is started in a session of its own before the stage step. Started by the step itself (after
  a reboot), it lived in the stage's process group, so every stop waited 90 s and then killed it.
- `TEXAS_SPLIT_STAGE_BIN`, `TEXAS_SPLIT_SUSHI_BIN` and `TEXAS_SPLIT_STAGE_ENV` (a JSON object of `QD_`/`QWEN_`
  variables) select other builds and stage settings for experiments; `/health` reports `concurrency`.
- `smoke.py` sends a 192-px image. With the old 96-px one (9 image tokens) the thinking sampler names one colour
  for both shapes in about 5% of replies (8 of 150, in every condition tried; greedy is always right), which made
  the check fail now and then. At 192 px: 0 of 30.

### Validation

Priority suite 8/8 exact with SSD parking, at one request at a time and at `TEXAS_SPLIT_MAX_CONCURRENT=4`;
`smoke.py` 10/10; `fault.py term`, `prefill` and `long`; stage-failure scenarios with several requests in flight.
Details in [docs/VALIDATION.md](docs/VALIDATION.md).

## 2026-11-10, third update: faster decode (shared expert beside the routed experts, warm Mac GPU, windowed draft depth)

Existing installs: `git pull`, `./install.sh driver stage sushi pins`, `bin/texas-split restart` (the driver, the
stage server and Sushi all changed).

Agentic replay (two recorded sessions, the thinking sampler, 1,024 tokens per turn, three seeds): total task
1.466× → **1.507×** faster than stock Sushi (pooled; 267.8 → 260.6 s), decode 114.7 → **119.5 tok/s**, time to
first token unchanged. Details and the qualification in [docs/VALIDATION.md](docs/VALIDATION.md).

### The shared expert beside the routed experts (stage, driver)

Each MoE layer's shared expert reads only the layer's input, yet ran after the router and the routed experts.
With `QD_STAGE_SIDE=1` (set by the host) the stage records it as a side branch: the driver's QMD launch latches
order the launches (`driver/R65-SIDE-EDGES.patch`, `tinynv_r65_edge`), a launch starts early only when everything
it reads finished before its predecessor started, and the three joins are copies of existing kernels with
`griddepcontrol.wait` at entry (`stage/kernels/r65_wait.cu`), because a latch only holds a kernel that executes
it. Bit-identical outputs; decode forwards (1–7 rows) 0.24–0.36 ms faster on the 5090. Forwards of 8+ rows and
prefill are unchanged. The stage now builds with the `r65-driver` feature.

### The Mac GPU stays warm while the 5090 runs (Sushi)

An idle Apple GPU clocks down within a few milliseconds, and the Mac idles ~6.5 ms per decode round while the
5090 runs layers 28–47, so the Mac's work after each reply started on a slow GPU. With `SUSHI_R65_GPU_WARM=1` (set
by the host) the stage client commits a 256-thread kernel on a Metal queue of its own every 0.5 ms while it waits
(`sushi/lib/gpu_warm.m`). It touches only its own buffer, so outputs are unchanged; the post-reply work fell from
0.57 to 0.31 ms per round. See [sushi/FORK.md](sushi/FORK.md) §8.

### Windowed draft depth on the split (host)

Sushi's adaptive planner picks each round's draft depth from measured round costs. On the split it handed the two
speedups above back: it drafted 5.5% shallower once rounds got cheaper, and the replay tied. The host now sets
`SUSHI_MTP_ADAPTIVE=0`, `SUSHI_MTP_COST_TABLE=0` and `SUSHI_MTP_ADAPTIVE_SERIAL=0` in split mode: Sushi's windowed
controller, 3 drafts per round, fewer only on a cold stretch. On the same sessions it beats the planner by itself
(1.500× vs 1.466× stock) and keeps the speedups (1.507×). It changes only how many tokens are drafted
(speculative sampling keeps the target distribution at any depth), and the generated text no longer depends on
timing: a seed reproduces its text. The Mac-only fallback keeps the planner. To go back, set
`TEXAS_SPLIT_SUSHI_ENV={"SUSHI_MTP_ADAPTIVE": "1", "SUSHI_MTP_COST_TABLE": "1", "SUSHI_MTP_ADAPTIVE_SERIAL": "1"}`
in your config and restart.

## 2026-11-10, second update: memory guard fix

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
