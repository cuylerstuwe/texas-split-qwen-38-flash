#!/bin/sh
# install.sh — build and set up texas-split on an Apple Silicon Mac with an RTX 5090 eGPU.
#
#   ./install.sh check            report what is present and what is missing; installs and builds nothing
#   ./install.sh                  every step below, in order (asks before installing anything outside this
#                                 repository; YES=1 answers yes)
#   ./install.sh STEP [STEP...]   only those steps:
#       submodules  pinned MLX / mlx-c headers (sushi/lib) and macuda (third_party/macuda)
#       tinygpu     tinygrad's signed TinyGPU.app (the DriverKit extension that reaches the card)
#       driver      NVIDIA headers + GSP firmware (pinned), then the patched libtinynv (driver/)
#       stage       the 5090 stage server (stage/, Rust)
#       mlx         the MLX runtime libraries Sushi runs on (the upstream Sushi v1.3.0 macOS bundle)
#       sushi       Sushi with the split, parking and the native-priority overlay (sushi/, Zig)
#       host        the host's native helpers (IPC page library, MLX memory caps)
#       model       download the model at the pinned revision (~159 GiB) and build its model directory
#       config      texas-split.env and the private control key
#       pins        record SHA-256 pins of everything the host runs
#       service     install and start the launchd agent (bin/texas-split start)
#
# Settings (environment, or texas-split.env after the first run):
#   TEXAS_SPLIT_MODEL_DOWNLOAD   where the pack is (or goes); default ~/Models/beamster/Qwen3.8-Flash-Next-Sushi-4bpw
#   TEXAS_SPLIT_PYTHON           Python 3.12+ for the host; default: the first python3.13/3.12 found
#
# What this cannot do for you: approve the TinyGPU system extension (System Settings asks you), connect the
# card, or replug it if it wedges. See INSTALL.md.
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

MODEL_REPO=beamster/Qwen3.8-Flash-Next-Sushi-4bpw
MODEL_REV=f25c0c4acdc44289a9cc6dab7f0eb788d1125ded
MODEL_NAME=Qwen3.8-Flash-Next-Sushi-4bpw
SUSHI_BUNDLE_URL=https://github.com/beamivalice/sushi/releases/download/v1.3.0/sushi-bin-macos-arm64.tar.gz
SUSHI_BUNDLE_SHA=96764e49d2d0c340a5b7f1cd70a392e75d42bb931763152365d102334ac5f5c8
TINYGPU_URL=https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdf217f255da7101d91e8323/TinyGPU.zip
TINYGPU_SHA=0c47285e2232643210555cf30ce08289b9e55da261c300e0c82e8448a359a21f
DRIVER_BUILD_ID=e797e31-r65side
MIN_MACOS=26.2

ENV_FILE=$ROOT/texas-split.env
getenv_file() { # the last KEY=value of texas-split.env, or nothing (never fails: set -e)
  if [ -f "$ENV_FILE" ]; then sed -n "s/^$1=//p" "$ENV_FILE" | tail -1 | sed "s/^[\"']//; s/[\"']\$//"; fi
  return 0
}
MODEL_DOWNLOAD=${TEXAS_SPLIT_MODEL_DOWNLOAD:-$(getenv_file TEXAS_SPLIT_MODEL_DOWNLOAD)}
MODEL_DOWNLOAD=${MODEL_DOWNLOAD:-$HOME/Models/beamster/$MODEL_NAME}
MODEL_DIR=$ROOT/models/$MODEL_NAME
PYTHON=${TEXAS_SPLIT_PYTHON:-$(getenv_file TEXAS_SPLIT_PYTHON)}

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ok]    %s\n' "$*"; }
miss() { printf '  [MISS]  %s\n' "$*"; MISSING=$((MISSING + 1)); }
warn() { printf '  [warn]  %s\n' "$*"; }
die()  { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
ask()  { # ask "question" -> 0 when yes
  [ "${YES:-0}" = 1 ] && return 0
  [ -t 0 ] || [ -r /dev/tty ] || return 1
  printf '  ?  %s [y/N] ' "$1"; read -r a < /dev/tty; case "$a" in y|Y|yes) return 0;; *) return 1;; esac
}
vercmp_ge() { # vercmp_ge A B -> A >= B (dotted versions)
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" = "$2" ]
}
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
find_python() {
  [ -n "$PYTHON" ] && { echo "$PYTHON"; return; }
  for p in /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 python3.13 python3.12 python3; do
    c=$(command -v "$p" 2>/dev/null) || continue
    "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null && { echo "$c"; return; }
  done
}

# ------------------------------------------------------------------------------------------------ check
step_check() {
  MISSING=0
  say "texas-split prerequisites"
  [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] && ok "Apple Silicon macOS" || miss "an Apple Silicon Mac (this is $(uname -sm))"
  v=$(sw_vers -productVersion 2>/dev/null || echo 0)
  vercmp_ge "$v" "$MIN_MACOS" && ok "macOS $v" || miss "macOS $MIN_MACOS or later (this is $v)"
  mem=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  if [ "$mem" -ge 128 ]; then ok "${mem} GiB unified memory"
  else warn "${mem} GiB unified memory: the host needs 82 GiB reclaimable when it starts (tested on 128 GiB)"; fi
  xcrun --find clang >/dev/null 2>&1 && ok "Xcode command line tools ($(xcrun --show-sdk-version 2>/dev/null) SDK)" || miss "Xcode command line tools: xcode-select --install"
  for t in git curl shasum unzip make; do command -v $t >/dev/null 2>&1 || miss "$t"; done
  command -v brew >/dev/null 2>&1 && ok "Homebrew" || miss "Homebrew (https://brew.sh): webp, hf and Python come from it"
  if command -v brew >/dev/null 2>&1; then
    wv=$(brew list --versions webp 2>/dev/null | awk '{print $2}')
    [ -n "$wv" ] && vercmp_ge "$wv" 1.6.0 && ok "webp $wv (Sushi's build links it)" || miss "webp >= 1.6.0: brew install webp"
  fi
  py=$(find_python)
  [ -n "$py" ] && ok "Python 3.12+ ($py)" || miss "Python 3.12 or later: brew install python@3.13"
  if command -v cargo >/dev/null 2>&1; then
    rv=$(rustc --version | awk '{print $2}')
    vercmp_ge "$rv" 1.85.0 && ok "Rust $rv" || miss "Rust 1.85 or later (this is $rv): rustup update"
  else miss "Rust 1.85 or later: https://rustup.rs"; fi
  command -v hf >/dev/null 2>&1 && ok "Hugging Face CLI (hf)" || miss "the Hugging Face CLI for the model download: brew install hf"
  if [ -d /Applications/TinyGPU.app ]; then
    ok "TinyGPU.app"
    pgrep -qf 'org\.tinygrad\.tinygpu\.driver2' && ok "TinyGPU's DriverKit extension is running" \
      || miss "TinyGPU's extension is not running: /Applications/TinyGPU.app/Contents/MacOS/TinyGPU install, then approve it in System Settings"
  else miss "TinyGPU.app (tinygrad's signed DriverKit extension): ./install.sh tinygpu"; fi
  [ -f third_party/macuda/tools/nv_shim_step.sh ] && ok "submodules" || miss "submodules: ./install.sh submodules"
  if [ -f "$MODEL_DOWNLOAD/config.json" ]; then ok "model download at $MODEL_DOWNLOAD (verified by ./install.sh model)"
  else
    d=$(dirname "$MODEL_DOWNLOAD"); while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    free=$(df -g "$d" | awk 'NR==2{print $4}')
    [ "$free" -ge 175 ] && ok "${free} GB free for the ~171 GB model download at $MODEL_DOWNLOAD" \
      || miss "~171 GB free for the model at $MODEL_DOWNLOAD (${free} GB free; set TEXAS_SPLIT_MODEL_DOWNLOAD)"
  fi
  if [ -x third_party/macuda/tools/preflight.sh ] || [ -f third_party/macuda/tools/preflight.sh ]; then
    if pgrep -qf 'org\.tinygrad\.tinygpu\.driver2'; then
      sh third_party/macuda/tools/preflight.sh 2>/dev/null | tail -1 | grep -q "VERDICT: OK" && ok "the RTX 5090 passes macuda's read-only preflight" \
        || warn "macuda's preflight does not pass (card not connected or not idle?): sh third_party/macuda/tools/preflight.sh"
    fi
  fi
  [ "$MISSING" -eq 0 ] && say "all prerequisites present" || say "$MISSING missing; see INSTALL.md"
  return 0
}

# ------------------------------------------------------------------------------------------------ steps
step_submodules() {
  say "==> submodules (MLX and mlx-c headers at Sushi v1.3.0's pins, macuda e797e31)"
  git submodule update --init sushi/lib/mlx-src sushi/lib/mlxc-src third_party/macuda
}

step_tinygpu() {
  say "==> TinyGPU.app"
  if [ ! -d /Applications/TinyGPU.app ]; then
    ask "download tinygrad's signed TinyGPU.app (pinned) and unzip it to /Applications?" || die "TinyGPU.app is required"
    tmp=$(mktemp -d); curl -fL --progress-bar "$TINYGPU_URL" -o "$tmp/TinyGPU.zip"
    [ "$(sha_of "$tmp/TinyGPU.zip")" = "$TINYGPU_SHA" ] || die "TinyGPU.zip checksum mismatch"
    unzip -q "$tmp/TinyGPU.zip" -d /Applications && rm -rf "$tmp"
  fi
  if pgrep -qf 'org\.tinygrad\.tinygpu\.driver2'; then ok "TinyGPU's DriverKit extension is running"
  else
    say "  Install and approve the extension (macOS asks you in System Settings > General > Login Items & Extensions):"
    say "    /Applications/TinyGPU.app/Contents/MacOS/TinyGPU install"
  fi
}

step_driver() {
  say "==> libtinynv (macuda's userspace NVIDIA driver at e797e31 + this repository's patches)"
  sh driver/libtinynv/tools/fetch_nv_headers.sh
  sh driver/libtinynv/tools/fetch_firmware.sh
  sdk=$(sh driver/build/sdk.sh)
  make -s -C driver/libtinynv BUILD="$ROOT/driver/build" CC="cc -isysroot $sdk" BUILD_ID=$DRIVER_BUILD_ID "$ROOT/driver/build/libtinynv.a"
  ok "driver/build/libtinynv.a ($(strings driver/build/libtinynv.a | grep -m1 -E '^e797e31'))"
}

step_stage() {
  say "==> stage server (stage/: layers 28-47 on the RTX 5090)"
  [ -f driver/build/libtinynv.a ] || die "driver/build/libtinynv.a is missing: ./install.sh driver"
  ( cd stage && QWEN_TINYNV_ARCHIVE="$ROOT/driver/build/libtinynv.a" cargo build --release --features cuda,metal,r65-driver --bin canon-stage-server )
  ok "stage/target/release/canon-stage-server"
}

step_mlx() {
  say "==> MLX runtime libraries (upstream Sushi v1.3.0 macOS bundle: MLX 0.32.3 with NAX kernels, mlx-c, webp)"
  if [ ! -f runtime/mlx-lib/libmlx.dylib ]; then
    mkdir -p runtime/.download
    f=runtime/.download/sushi-bin-macos-arm64.tar.gz
    [ -f "$f" ] && [ "$(sha_of "$f")" = "$SUSHI_BUNDLE_SHA" ] || curl -fL --progress-bar "$SUSHI_BUNDLE_URL" -o "$f"
    [ "$(sha_of "$f")" = "$SUSHI_BUNDLE_SHA" ] || die "Sushi bundle checksum mismatch"
    rm -rf runtime/.bundle && mkdir -p runtime/.bundle && tar -xzf "$f" -C runtime/.bundle
    lib=$(find runtime/.bundle -type d -name lib | head -1)
    [ -f "$lib/libmlx.dylib" ] && [ -f "$lib/mlx.metallib" ] || die "unexpected bundle layout"
    rm -rf runtime/mlx-lib && cp -R "$lib" runtime/mlx-lib && rm -rf runtime/.bundle
  fi
  # Sushi's build looks for lib/mlx/{lib,include}: the bundle's libraries and the pinned submodule headers
  mkdir -p sushi/lib/mlx/include
  ln -sfn ../../../runtime/mlx-lib sushi/lib/mlx/lib
  ln -sfn ../../mlx-src/mlx sushi/lib/mlx/include/mlx
  [ -f sushi/lib/mlx-src/mlx/array.h ] || die "MLX headers missing: ./install.sh submodules"
  # the stamp sushi/scripts/build-mlx.sh writes: the bundle was built at exactly these pins
  mlx=$(git -C sushi/lib/mlx-src rev-parse --short=12 HEAD); mlxc=$(git -C sushi/lib/mlxc-src rev-parse --short=12 HEAD)
  patch=$(shasum sushi/patches/mlxc-gather-qmm-global-scale.patch | cut -c1-12)
  echo "mlx=$mlx mlxc=$mlxc patch=$patch target=26.2" > sushi/lib/mlx/.version
  ok "runtime/mlx-lib ($(ls runtime/mlx-lib | tr '\n' ' '))"
}

step_sushi() {
  say "==> Sushi (sushi/: the split, background parking, native-priority overlay)"
  [ -e sushi/lib/mlx/lib/libmlx.dylib ] || die "MLX runtime libraries missing: ./install.sh mlx"
  ( cd sushi && sh scripts/fetch-zig.sh )
  cxx="xcrun clang++ -std=c++20 -O2 -mmacosx-version-min=26.2 -Isushi/lib/mlx-src -Isushi/lib/mlxc-src -Isushi/native"
  $cxx -c sushi/native/stage_split.cpp -o sushi/native/stage_split.o
  $cxx -c sushi/native/texas_checkpoint.cpp -o sushi/native/texas_checkpoint.o
  ( cd sushi && .zig-toolchain/zig build -Doptimize=ReleaseFast --prefix "$ROOT/sushi/zig-out" )
  ok "sushi/zig-out/bin/sushi"
}

step_host() {
  say "==> host helpers"
  [ -f runtime/mlx-lib/libmlxc.dylib ] || die "MLX runtime libraries missing: ./install.sh mlx"
  xcrun clang -dynamiclib -O2 -std=c11 -mmacosx-version-min=26.2 -o host/texas_ipc.dylib sushi/native/texas_ipc.c
  xcrun clang -dynamiclib -O2 -mmacosx-version-min=26.2 -install_name @rpath/memory_caps.dylib \
    -o runtime/memory_caps.dylib host/native/memory_caps.c -Lruntime/mlx-lib -lmlxc
  ok "host/texas_ipc.dylib, runtime/memory_caps.dylib"
}

step_model() {
  say "==> model: $MODEL_REPO@$(echo $MODEL_REV | cut -c1-10) at $MODEL_DOWNLOAD"
  py=$(find_python); [ -n "$py" ] || die "Python 3.12+ is required"
  if ! "$py" host/make_model_view.py --check "$MODEL_DOWNLOAD" >/dev/null 2>&1; then
    command -v hf >/dev/null 2>&1 || die "the Hugging Face CLI is required: brew install hf"
    ask "download the ~171 GB model to $MODEL_DOWNLOAD?" || die "the model is required"
    mkdir -p "$MODEL_DOWNLOAD"
    hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DOWNLOAD" --exclude 'assets/*'
  fi
  "$py" host/make_model_view.py "$MODEL_DOWNLOAD" "$MODEL_DIR"
}

step_config() {
  say "==> configuration"
  py=$(find_python); [ -n "$py" ] || die "Python 3.12+ is required"
  if [ ! -f "$ENV_FILE" ]; then
    umask 077
    cat > "$ENV_FILE" <<EOF
# texas-split settings (KEY=VALUE; TEXAS_SPLIT_* environment variables override these)
TEXAS_SPLIT_PYTHON=$py
TEXAS_SPLIT_MODEL_DOWNLOAD=$MODEL_DOWNLOAD
TEXAS_SPLIT_MODEL_DIR=$MODEL_DIR
TEXAS_SPLIT_PORT=8003
TEXAS_SPLIT_ENGINE_PORT=8014
TEXAS_SPLIT_CONTROL_KEY=$HOME/.config/texas-split/control-key
EOF
    ok "wrote texas-split.env"
  else ok "texas-split.env exists (kept)"; fi
  key=$(sed -n 's/^TEXAS_SPLIT_CONTROL_KEY=//p' "$ENV_FILE" | tail -1); key=${key:-$HOME/.config/texas-split/control-key}
  if [ ! -f "$key" ]; then
    mkdir -p "$(dirname "$key")"; chmod 700 "$(dirname "$key")"
    ( umask 077; "$py" -c 'import secrets; print(secrets.token_hex(32))' > "$key" )
    ok "generated the private scheduler control key at $key (mode 0600)"
  else ok "control key $key exists (kept)"; fi
}

step_pins() {
  say "==> pins"
  py=$(find_python); [ -n "$py" ] || die "Python 3.12+ is required"
  "$py" host/texas_split.py write-pins >/dev/null && "$py" host/texas_split.py check
}

step_service() {
  say "==> service"
  bin/texas-split start
}

case "${1:-all}" in
  check) step_check ;;
  all)
    step_check
    for s in submodules tinygpu driver stage mlx sushi host model config pins; do "step_$s"; done
    say ""
    say "Built. Start it with:  bin/texas-split start      (launchd agent; ~40 s: stage server, then Sushi)"
    say "Then check it with:    host/tests/smoke.py"
    ;;
  *) for s in "$@"; do
       case "$s" in
         check|submodules|tinygpu|driver|stage|mlx|sushi|host|model|config|pins|service) "step_$s" ;;
         *) die "unknown step $s (see the header of install.sh)" ;;
       esac
     done ;;
esac
