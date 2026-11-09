# Instructions for coding agents

You are probably here to install and run texas-split for a user. The whole procedure is `install.sh`;
[INSTALL.md](INSTALL.md) explains each step. Do not consult other projects' documentation (Sushi, macuda,
tinygrad): everything this deployment needs is in this repository or fetched by `install.sh` at pinned
versions.

## Install

1. `./install.sh check`. For each `[MISS]` line, install the tool it names (`brew install webp hf
   python@3.13`, `rustup`, `xcode-select --install`). Ask the user before installing system software if
   your instructions require it.
2. TinyGPU: `./install.sh tinygpu`, then the user must run
   `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install` and approve the system extension in System
   Settings. You cannot approve it; ask the user and wait. Re-run `./install.sh check` to confirm.
3. The model download is ~171 GB. Confirm free disk space and the download location
   (`TEXAS_SPLIT_MODEL_DOWNLOAD`) with the user; reuse an existing download of
   `beamster/Qwen3.8-Flash-Next-Sushi-4bpw` (revision `f25c0c4...`) by pointing `TEXAS_SPLIT_MODEL_DOWNLOAD`
   at it: `install.sh` verifies it against `host/model-manifest.json` instead of downloading.
4. `YES=1 ./install.sh` (or without `YES=1` to answer prompts). Each step is idempotent; rerun single steps
   (`./install.sh driver stage ...`) after fixing a failure.
5. `bin/texas-split start`, wait for `"status": "ready"` (about 40 s), then `host/tests/smoke.py`. All ten
   checks should pass. `bin/texas-split status` shows `"mode": "split"` when the RTX 5090 is in use
   (`"mac-only"` means the stage did not start: see INSTALL.md Troubleshooting).

## Operating rules

- Stop the service (`bin/texas-split stop`) before anything else uses the GPU, the eGPU, or ~100 GB of
  unified memory. It holds the card and ~70 GB of memory while running.
- Never send SIGKILL to `canon-stage-server` or the host; use `bin/texas-split stop` (SIGTERM-based, waits
  for the card to be put down).
- After rebuilding anything, run `./install.sh pins`; the host refuses to start on a pin mismatch.
- The private scheduler key (`~/.config/texas-split/control-key`) and `texas-split.env` are local
  configuration: never commit or publish them.
- Exact-resume tests (`host/tests/priority.py`) need deterministic decoding: start the host with
  `TEXAS_SPLIT_SUSHI_ENV='{"SUSHI_MTP_ADAPTIVE":"0","SUSHI_MTP_COST_TABLE":"0","SUSHI_MTP_ADAPTIVE_SERIAL":"0"}'
  bin/texas-split serve` (Sushi's adaptive draft width is timing-based, so normal runs differ run to run).
