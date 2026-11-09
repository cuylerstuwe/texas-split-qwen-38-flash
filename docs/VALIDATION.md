# Validation

Hardware: Apple M5 Max (128 GB), macOS 27.2; RTX 5090 over Thunderbolt. Model: the Sushi 4bpw pack at
revision `f25c0c4`. Evidence files are in [evidence/](evidence/); the tests are in `host/tests/`.

## Speed and quality

- **Total task time** on recorded agentic coding sessions, replayed request by request (prefill + decode +
  queueing, the thinking sampler): 1.28×, 1.33× and 1.34× faster than stock Sushi on the same pack (pooled
  1.32×). Decode ~98–111 tok/s vs ~79–84 tok/s.
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

## Not covered

Sustained multi-hour load, sleep/wake with the stage running, a physically disconnected card (simulated by
killing and freezing the stage server), recovery of parked requests after a crash or reboot, other GPUs or
Macs.
