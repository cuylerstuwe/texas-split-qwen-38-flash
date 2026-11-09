# Installing texas-split

Everything below is what `./install.sh` does, step by step, so that a person or a coding agent can run it,
understand a failure and fix it. Run `./install.sh check` at any time: it reports what is present and what
is missing, and changes nothing.

## 0. Hardware and system

- Apple Silicon Mac, **128 GB** unified memory, macOS **26.2+** (tested: M5 Max, macOS 27.2, Xcode SDK 27.2).
  The host refuses to start the engine with less than 82 GiB of reclaimable memory, and stops it if
  reclaimable memory falls below 12 GiB. Quit large apps first.
- **RTX 5090** (PCI device id `0x2b85`) in a Thunderbolt eGPU enclosure, connected and powered.
  `TEXAS_SPLIT_GPU_DEVID` overrides the device id; other GPUs are untested and the stage's kernels are built
  for `sm_120`.
- **Disk**: ~171 GB for the model plus ~5 GB of builds. The model can live on an external SSD
  (`TEXAS_SPLIT_MODEL_DOWNLOAD`); its 95 GiB n-gram table is read from disk during inference, so use a fast SSD.

## 1. Tools

| Tool | Why | Install |
|---|---|---|
| Xcode command line tools | clang, the macOS SDK | `xcode-select --install` |
| Homebrew | the packages below | https://brew.sh |
| `webp` ≥ 1.6 | linked by Sushi's build | `brew install webp` |
| Python ≥ 3.12 | the host and its tests | `brew install python@3.13` |
| Rust ≥ 1.85 | the stage server | https://rustup.rs |
| `hf` | the model download | `brew install hf` |

Zig is not a prerequisite: `sushi/scripts/fetch-zig.sh` stages the pinned, checksummed Zig 0.17.0 inside
`sushi/.zig-toolchain`. No CUDA toolkit is needed: the CUDA kernels ship prebuilt (`stage/kernels/*.cubin`,
each pinned by SHA-256 and checked when loaded).

## 2. Submodules — `./install.sh submodules`

`git clone --recursive` fetches them; otherwise this step does:

- `sushi/lib/mlx-src`, `sushi/lib/mlxc-src`: MLX and mlx-c at the commits Sushi v1.3.0 pins (headers for the build).
- `third_party/macuda`: macuda at `e797e31` (GPU preflight, TinyGPU server control, the stage launcher
  `tools/nv_shim_step.sh`). Nothing in it is built.

## 3. TinyGPU — `./install.sh tinygpu`

tinygrad's signed TinyGPU.app provides the DriverKit extension and the Unix-socket server through which
`libtinynv` reaches the card. The step downloads the pinned release (SHA-256 checked) into `/Applications`
(it asks first). Then **you** must install and approve the extension:

```sh
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install
```

and allow it in System Settings → General → Login Items & Extensions (Driver Extensions). `./install.sh check`
reports whether the extension is running. An agent cannot do this approval; ask the user.

## 4. Driver — `./install.sh driver`

Builds `driver/build/libtinynv.a`: macuda's userspace NVIDIA driver (`libtinynv` at macuda `e797e31`) with
this project's patches (graph latch launches, recorded-graph parameter patching, calibration stamps; see
`driver/README.md`). It first fetches, pinned by URL and SHA-256 into `driver/libtinynv/third_party/`:

- NVIDIA `open-gpu-kernel-modules` headers (used under their MIT license),
- the GB202 GSP firmware 570.144 from linux-firmware (NVIDIA's redistributable firmware license).

The firmware path is compiled in; `TINYNV_FW_DIR` overrides it (the host sets it).

## 5. Stage server — `./install.sh stage`

`cargo build --release --features cuda,metal,r53-driver --bin canon-stage-server` in `stage/`, linking the
archive from step 4 (`QWEN_TINYNV_ARCHIVE`). The binary embeds the absolute path of `stage/kernels`: if you
move the repository, rebuild (`./install.sh stage pins`).

## 6. MLX runtime — `./install.sh mlx`

Downloads the upstream Sushi v1.3.0 macOS bundle (pinned URL and SHA-256) and keeps its `lib/` as
`runtime/mlx-lib`: MLX 0.32.3 built with the M5 NAX kernels, `mlx.metallib`, mlx-c, webp. These exact
libraries are required: the background-parking helper reads MLX's private random-number state by symbol
name from this build. It then points `sushi/lib/mlx/{lib,include}` at those libraries and the submodule
headers.

## 7. Sushi — `./install.sh sushi`

Fetches Zig, compiles the two native objects (`sushi/native/stage_split.cpp`, the 5090 stage client, and
`texas_checkpoint.cpp`, the SSD parking helper) with `clang++ -std=c++20`, then
`zig build -Doptimize=ReleaseFast` → `sushi/zig-out/bin/sushi`.

## 8. Host helpers — `./install.sh host`

`host/texas_ipc.dylib` (the private IPC page shared with Sushi; same source as Sushi's) and
`runtime/memory_caps.dylib` (caps MLX's memory and wired limits at 108 GiB inside the engine, injected with
`DYLD_INSERT_LIBRARIES`).

## 9. Model — `./install.sh model`

Downloads `beamster/Qwen3.8-Flash-Next-Sushi-4bpw` at revision `f25c0c4acdc44289a9cc6dab7f0eb788d1125ded`
(the validated one; README images excluded) to `TEXAS_SPLIT_MODEL_DOWNLOAD` (default
`~/Models/beamster/Qwen3.8-Flash-Next-Sushi-4bpw`), after asking. An existing download there is checked against
`host/model-manifest.json` (every file's name and size) and reused. Then `host/make_model_view.py` builds
`models/Qwen3.8-Flash-Next-Sushi-4bpw/`: symlinks to the download plus a `config.json` with the model's native
rope (`rope_type: default`, 262,144 positions) instead of the pack's YaRN ×4. The directory name is the
model id the API reports, so keep it.

## 10. Config — `./install.sh config`

Writes `texas-split.env` (kept if present) and generates the private scheduler key
`~/.config/texas-split/control-key` (mode 0600). Settings, all optional (`TEXAS_SPLIT_*` environment
variables override the file):

| Key | Default | Meaning |
|---|---|---|
| `TEXAS_SPLIT_PORT` / `TEXAS_SPLIT_ENGINE_PORT` | 8003 / 8014 | host / Sushi ports (loopback) |
| `TEXAS_SPLIT_MODEL_DIR` | `models/Qwen3.8-Flash-Next-Sushi-4bpw` | the model directory from step 9 |
| `TEXAS_SPLIT_CONTROL_KEY` | `~/.config/texas-split/control-key` | Bearer key for `/v1/scheduler` |
| `TEXAS_SPLIT_CACHE` | `~/Library/Caches/Texas-Split` | prefix cache (20 GB disk tier), parking checkpoints, run logs |
| `TEXAS_SPLIT_LOG` | `~/Library/Logs/Texas-Split/server.log` | host log (launchd) |
| `TEXAS_SPLIT_LABEL` | `com.texas-split.server` | launchd label (`~/Library/LaunchAgents/<label>.plist`) |
| `TEXAS_SPLIT_GPU_DEVID` | `0x2b85` | PCI device id of the card |
| `TEXAS_SPLIT_STAGE_TIMEOUT_MS` | 20000 | a stage call slower than this counts as a failure |
| `TEXAS_SPLIT_DISPLAY_HOLD` | `boot` | hold the display awake around the GPU's cold boot (`always`: whole life) |
| `TEXAS_SPLIT_VERIFY_PARKS` | 1 | re-read and byte-compare the stage state after every restore |
| `TEXAS_SPLIT_SUSHI_ENV` | — | extra `SUSHI_*` variables for the engine (JSON object), for experiments |

## 11. Pins — `./install.sh pins`

Records SHA-256 of the binaries and helpers in `runtime/PINS.json`; the host refuses to start if any changes
(after a rebuild, run `./install.sh pins` again).

## 12. Run

```sh
bin/texas-split start        # installs ~/Library/LaunchAgents/com.texas-split.server.plist and starts it
bin/texas-split status       # JSON; "status": "ready", "mode": "split" when up
host/tests/smoke.py          # chat, reasoning, tools, an image, a 35K-token prompt, queueing, cancellation
bin/texas-split logs --follow
bin/texas-split stop
```

Or run it in the foreground without launchd: `bin/texas-split serve`. Startup: the stage server boots the
card (~27 s, the first time after power-on may take longer), then Sushi loads (~10 s).
`bin/texas-split uninstall` stops it and removes the launchd agent.

## Troubleshooting

- **`check` says TinyGPU's extension is not running**: run the `TinyGPU install` command above and approve
  it; after macOS updates you may need to approve again.
- **The host log says `stage unavailable, running every layer on the Mac`**: the service still works (Mac
  only). Look at the run directory named in the log (`stage.log`, and the macuda log it points to). Common
  causes: the card is not connected or powered, the extension is not approved, the card is in a bad state
  (power-cycle the enclosure and replug). `sh third_party/macuda/tools/preflight.sh` must end in
  `VERDICT: OK`. The host retries the split when idle (2 minutes, backing off to an hour).
- **`preflight ABORT` in the stage log** (`third_party/macuda/logs/shim-probe-*.log` names the reason): macuda's
  read-only preflight refuses to touch the card. `[new DART error data]` means the Mac's IOMMU recorded a
  fault from the card. That record survives a card reset and clears only when the card is re-enumerated:
  replug or power-cycle the enclosure (or reboot). If it reappears after the card works again, macuda's
  `sh third_party/macuda/tools/dart-stale.sh "reason"` records that exact fault data as known, so later
  preflights only abort on a new fault (`--clear` forgets it). `[a new dext instance since the record]`:
  the extension restarted; see macuda's `docs/` for its recovery procedure. To share an existing macuda
  checkout and its records, set `TEXAS_SPLIT_MACUDA` to it.
- **`reclaimable memory ... < 82 GiB`**: free memory (quit apps) and `bin/texas-split restart`.
- **`Pinned file changed since install`**: something was rebuilt; `./install.sh pins`.
- **Moved the repository**: `./install.sh driver stage host pins` (paths are compiled in).
- **The model check fails**: an incomplete download; rerun `./install.sh model` (`hf download` resumes).
- **Never `kill -9` the stage server** (`canon-stage-server`): on SIGTERM or a normal exit the driver puts
  the card down cleanly; SIGKILL skips that. `bin/texas-split stop` does the right thing.
- Engine logs: `<run dir>/engine.log` and `sushi-native.log`; stage: `<run dir>/stage.log`. Run directories
  are under `~/Library/Caches/Texas-Split/runs/`.
