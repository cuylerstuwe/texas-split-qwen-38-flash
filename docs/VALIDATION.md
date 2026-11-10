# Validation

Hardware: Apple M5 Max (128 GB), macOS 27.2; RTX 5090 over Thunderbolt. Model: the Sushi 4bpw pack at
revision `f25c0c4`. Evidence files are in [evidence/](evidence/); the tests are in `host/tests/`.

## Speed and quality

- **Total task time** on recorded agentic coding sessions, replayed request by request (prefill + decode +
  queueing, the thinking sampler, 1,024 tokens per turn, three seeds): 1.42×, 1.47× and 1.51× faster than
  stock Sushi on the same pack (pooled 1.46×, 392.7 → 268.1 s) with the 2026-11-10 build; the first
  release measured 1.28×, 1.33× and 1.34× (pooled 1.32×). Decode ~108–122 tok/s (first release ~98–111)
  vs ~79–84 tok/s. The stock runs are the first release's (same harness and seeds; stock Sushi has not
  changed).
- **Prefill** (cold prompts, nothing cached):

  | Prompt | Stock Sushi, Mac only | texas-split |
  |---|---|---|
  | 54K tokens, first turn of a recorded agent session (time to first token, same prompt) | 27.5 s (~2.0K tok/s) | 16.8–17.2 s (~3.2K tok/s): 1.6× (first release 18.2 s, 1.52×) |
  | 63,596 tokens (two runs) | not run | 3,443 and 3,101 tok/s |
  | 192,008 tokens (two runs) | not run | 2,814 and 3,289 tok/s |
  | 8K / 32K tokens, separate runs with prefix caching off | 2,048–2,714 / 2,105–2,539 tok/s | — |

  Across the multi-turn sessions (later turns mostly cached), summed time to first token fell from 15.3 to
  10.3–10.6 s in one session and from 38.7 to 24.6–25.3 s in the other (first release: 11.4 and 25.8 s).
  The 5090 runs its 20 layers' prefill at ~5,600–6,300 tok/s, so the Mac's 28 layers set the pace: the
  split alone can reach about 1.6× (the Mac's share of stock prefill time is ~62–65%); the bf16 weight
  cache speeds up the Mac's layers themselves. Run-to-run spread is about ±10%
  (heat, and whether the 95 GiB n-gram table is in the page cache). Two 63.6K runs that were parked
  mid-prefill by the priority tests (2,692 and 2,943 tok/s, pause included) are left out above.
- **Quality**: mean KL divergence to a teacher model on a fixed prompt panel: 0.004463 split vs 0.004581
  Mac-only (short panel). Split vs Mac-only directly: KL 0.00052 per token, the two disagreeing on the top
  token 35 times in 5,765. Paired by prompt, split minus Mac-only is −0.000117 (bootstrap 95% CI
  −0.00022 to −0.00002): the split's floating-point differences cost no fidelity.
- **Images and video** (top-5 KL over the greedy prefix shared with the Mac-only run, `vision_logprobs.py`,
  `vision_sizes.py`): small image 120/120 identical tokens, KL 0.00036; video (6 frames) 0.00077; images
  from 128 to 896 px and five 600–832 px images: identical for 40 of 40 tokens in most cases, KL
  0.0003–0.0013. Text control: identical.

## Background parking is exact

Exactness can only be judged with deterministic decoding. Sushi chooses its MTP draft width from measured
round timings, so the verify row counts, and with them the stage's floating-point results, vary from run to
run: three uninterrupted greedy runs all differed from about the 50th token. With the timing-adaptive width
off (`SUSHI_MTP_ADAPTIVE=0 SUSHI_MTP_COST_TABLE=0 SUSHI_MTP_ADAPTIVE_SERIAL=0`), four runs were
bit-identical, cold and warm. In that mode, `host/tests/priority.py` holds a background request mid-flight,
runs a foreground request, releases it, and compares the background's final output with an uninterrupted
reference ([evidence/priority-suite.json](evidence/priority-suite.json),
[evidence/priority-192k.json](evidence/priority-192k.json)):

| Case | Result |
|---|---|
| Greedy, held mid-decode by a scheduler lease | exact; switch 0.17 s; Sushi park 311 MB, 5090 park 48.7 MB |
| Seeded sampling (T = 1), held by a foreground request | exact |
| Held mid-prefill (63,596-token prompt, parked at row 4,096) | exact (cold and warm prefill also identical) |
| Two queued foreground requests during one hold | exact; frozen until the last finished |
| The foreground request's client disconnects while the background is parked | exact; resumed 0.27 s later |
| Streaming with thinking and a tool call | exact content, reasoning and tool arguments |
| 192,008-token background held mid-decode | exact; 5090 park 1.154 GB in 375 ms, restore 583 ms; Sushi park 2.12 GB; switch 1.52 s |
| Admission rules | second background 429, background with media 400, bad priority 400, unknown model 404, unauthenticated scheduler 401 |

Every 5090 restore re-read its state and matched the park file byte for byte.

## Failure handling

- `host/tests/fault.py term`: the stage server gets SIGTERM 4 s into a stream → 0.5 s pause, the stream
  finished normally on the Mac; the split came back when idle.
- `fault.py prefill`: SIGTERM 3 s into a 124,320-token split prefill → re-run on the Mac, correct answer.
- `fault.py stop`: the stage server frozen (SIGSTOP) → the 20 s stage timeout fired, the stream finished.
- `host/tests/fault_parked.py`: the stage server killed while a background request was parked → the
  foreground request re-ran on the Mac, then the background request re-ran from its 68 streamed tokens on
  the Mac and finished all 700.
- A prefix-cache hit of 191,977 tokens seeds 1.15 GB into the stage (the shared file is 2 GiB; an oversized
  seed fails into the Mac re-run instead of crashing).

## End to end

`host/tests/smoke.py` (10/10): model listing; streaming chat with reasoning; non-thinking; streaming tool
calls (including parallel calls) and a tool-result turn; a non-streaming tool call; an image; a
35,577-token prompt (correct answer); a concurrent stream waits for the first; a queued client that leaves
is dropped; a cancelled stream frees the engine.

## 2026-11-09 update (see CHANGELOG.md)

- **Verify fix.** The forward now fills the deferred n-gram embedding before the stage send. Greedy MTP
  continuations (256 tokens, cold 20K- and 54K-token agentic prompts) match a Mac-only run for 498 / 743
  characters, against 37 / 136 before the fix.
- **Bit-identical changes** (kernels, verify and draft scheduling) were checked in the fixed-depth MTP
  mode, where every run is deterministic: whole two-session replays and greedy continuations came out
  byte-identical with each change on and off. Per-round costs from the same runs: −0.5 ms (verify layers
  handed to the GPU early, stride 4), −0.5 ms more (lazy captures plus early draft steps).
- **Drafter-side changes** cannot change the output distribution (acceptance is exact). The vocab-prefix
  draft head drafted the same tokens on the same text at lower cost. The extension-gate confidence
  measured +2.0% decode throughput under the adaptive controller (three seeds: 103.3 → 107.9,
  109.7 → 110.9, 107.9 → 108.5 tok/s).
- **Chunk shaping.** Same-prompt runs are deterministic. Against the plain policy, end-of-prefill and
  greedy-decode logprobs (top 20) differ by about 1e-3 mean KL per position at 20–54K tokens, with greedy
  text identical for 24/24 tokens. Switching the chunk width from 4096 to 2048 rows moves them more (2.5e-3
  to 4.3e-3). Pipeline stalls on the agentic replays: 4.8–4.9 → 2.8–2.9 s per session pair.
- **Qualification of the final build.**
  - Priority suite 8/8 exact in the fixed-depth mode, with SSD parking.
  - smoke.py 10/10.
  - `fault.py term`: the stream finished on the Mac.
  - `FAULT_NONCE=x fault.py prefill`: the stage was killed 4 s into a cold 124K-token split prefill, and
    the Mac finished it with the correct answer.
  - The split came back when idle.
- **Disk.** Sushi parks a background request only when the SSD has the checkpoint size plus 32 GiB free.
  With less, the pause holds the request in RAM instead: resume is still exact, `park_error` shows in the
  priority status, and the engine log says `checkpoint SSD reserve unavailable`.

## 2026-11-10 update (see CHANGELOG.md)

- **Stage head.** Greedy continuations (A and B prompts, 512 tokens) are byte-identical with the head on and
  off; 341 of 341 verify forwards used it. Over 2,164 sampled forwards, 6 fell back to the Mac's head (a
  reserved special in the top 20). Production configuration, adaptive drafting, the agentic replay, two
  seeds in counterbalanced order: a decode round 27.63 → 26.85 ms, the Mac's post-verify work 2.22 →
  0.58 ms; 3,780 of 3,780 verify forwards used the head. At the 262,160-row capacity the card has
  0.58 GiB free with 1,536-row prefill pieces (with 2,048 the first decode of a full-context sequence ran
  out of memory); time to first token with 1,536-row pieces was not worse (counterbalanced runs).
- **Dequant cache.** Fixed-depth replays of both sessions, two seeds: all twelve turns byte-identical with
  the cache on and off. Summed time to first token per pair 38.3 / 37.3 → 36.4 / 35.4 s (cold 54K turn
  18.3 → 16.6 s). Minimum available memory during the runs 3–4 GiB lower (24.8 GiB).
- **Whole-tail chunk planning.** Time to first token per turn (single runs, the same prompts):

  | turn | before | after |
  |---|---|---|
  | A0, cold 20K | 5.40–5.42 s | 5.41 s |
  | A1, +7.9K | 2.60 s | 2.53 s |
  | A2, +4.7K | 2.14 s | 1.71 s |
  | B0, cold 54K | 15.2–16.4 s | 14.7 s |
  | B1, +10.1K | 4.04–4.15 s | 3.41 s |
  | B2, +10.1K | 4.02–4.11 s | 3.65 s |

  Numerics (cold A0 and B0, then 24 greedy tokens with top-20 logprobs, speculation off), before vs after:
  A0 greedy 11/24, mean KL 3.4e-3 (max 2.2e-2); B0 24/24, 2.4e-3 (3.6e-2). Switching the chunk width from
  4,096 to 2,048 rows gives 11/24 (the same near-tie token), 2.5e-3 (1.4e-2) and 24/24, 4.3e-3 (4.1e-2).
  The most likely next token at the end of prefill is unchanged in both.
- **Replay benchmark** (the README's numbers; same harness and seeds as the first release's stock
  comparison, the production configuration, adaptive drafting): seeds 61 / 62 / 63 total task 1.42× /
  1.47× / 1.51× (session A 1.33–1.50×, B 1.48–1.52×), decode 108.4 / 111.8 / 121.8 tok/s, cold 54K first
  turn 16.80 / 17.20 / 16.98 s.
- **Qualification of the final build** (all three changes): priority suite 8/8 exact with SSD parking
  ([evidence/priority-suite.json](evidence/priority-suite.json)); with `{"presence_penalty": 0}` added to
  every case, 2,894 of 2,894 verify forwards used the stage head and the suite was again 8/8 exact;
  smoke.py 10/10; `fault.py term`; `FAULT_NONCE=x fault.py prefill` (the stage killed 4 s into a cold
  124K-token prefill; the Mac finished it correctly in 66.9 s).

## Not covered

Sustained multi-hour load, sleep/wake with the stage running, a physically disconnected card (simulated by
killing and freezing the stage server), recovery of parked requests after a crash or reboot, other GPUs or
Macs.
