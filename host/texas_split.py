#!/usr/bin/env python3
"""texas-split host: Qwen3.8-Flash-Next (Sushi 4bpw pack) served by Sushi on an Apple Silicon Mac with
layers 28-47 on an RTX 5090 eGPU (a pipeline split), behind one OpenAI-compatible endpoint.
See README.md and INSTALL.md at the repository root.

  texas_split.py serve                 the host (run by launchd, or by hand)
  texas_split.py start|stop|restart    lifecycle through launchd (installs the agent on start)
  texas_split.py status|health|logs [--follow]
  texas_split.py install-plist|uninstall|write-pins|check
  texas_split.py concurrency [N]       show, or set, how many requests generate at once (1-16; default 1)

Configuration: texas-split.env at the repository root (written by install.sh; KEY=VALUE lines) and
TEXAS_SPLIT_* environment variables, which win. The host listens on 127.0.0.1:8003 by default and owns:
- the stage server (stage/: canon-stage-server, layers 28-47 on the 5090, launched through macuda's
  nv_shim_step.sh preflight) and the Sushi engine (sushi/) on 127.0.0.1:8014, started in that order;
- the API: model ids, request normalization and limits, X-LocalAI-Mode / X-LLM-Priority
  foreground/background with exact-progress parking (priority_control.py: leased holds on
  /v1/scheduler with a private control key, a private IPC page Sushi polls, SSD checkpoints; the
  5090 stage's half of a parked request parks to SSD beside Sushi's), one generation at a time in
  Sushi by default (later ones wait there, or here while the engine starts or restarts), or up to
  TEXAS_SPLIT_MAX_CONCURRENT at once: up to four of them decode through the 5090 on conversation
  slots, interleaved, and the rest run every layer on the Mac beside them;
- supervision: crashed engines restart with backoff; a stage that cannot start leaves the service
  running every layer on the Mac (mode mac-only) instead of failing;
- shutdown that never signals the stage server while it puts the card down: Sushi stops first,
  the stage server exits by itself when its socket closes, signals only after 90 s;
- a memory/thermal guard (12 GiB reclaimable floor, swap/compression growth, critical thermals): a
  trip stops the engine and the host answers 503 until restarted. Compression growth is not counted
  while the idle engine has released its wiring (macOS unwires Sushi's Metal memory ~60 s after its
  last request; compressing those pages is the engine giving memory back): see MemoryGuard.
"""
import argparse, fcntl, hashlib, hmac, http.client, json, os, plistlib, re, select, signal, socket, subprocess, sys, threading, time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mac_probes import _read_kernel, notify_thermal_pressure_reader, level_name  # noqa: E402
from priority_control import Controller, IPC  # noqa: E402

REPO = ROOT.parent                        # host/ sits in the repository root


def load_config():
    """KEY=VALUE lines from texas-split.env (TEXAS_SPLIT_CONFIG names another file); TEXAS_SPLIT_*
    environment variables win."""
    cfg = {}
    path = Path(os.environ.get('TEXAS_SPLIT_CONFIG', REPO / 'texas-split.env')).expanduser()
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                cfg[key.strip()] = value.strip().strip('"').strip("'")
    cfg.update({k: v for k, v in os.environ.items() if k.startswith('TEXAS_SPLIT_')})
    return cfg


CFG = load_config()


def set_config(key, value):
    """KEY=VALUE into the configuration file load_config reads: the key's line is replaced, or one is appended."""
    path = Path(os.environ.get('TEXAS_SPLIT_CONFIG', REPO / 'texas-split.env')).expanduser()
    lines = path.read_text().splitlines() if path.is_file() else []
    out, done = [], False
    for line in lines:
        if line.strip().startswith(key + '='):
            if not done:
                out.append(f'{key}={value}')
                done = True
        else:
            out.append(line)
    if not done:
        out.append(f'{key}={value}')
    path.write_text('\n'.join(out) + '\n')
    return path


def cfg_path(key, default):
    return Path(os.path.expanduser(CFG.get(key) or str(default)))


VERSION = 'texas-split-20261108'
# the model directory: the Sushi pack with the native-rope config.json (install.sh builds this view)
MODEL = cfg_path('TEXAS_SPLIT_MODEL_DIR', REPO / 'models/Qwen3.8-Flash-Next-Sushi-4bpw')
MODEL_ID = MODEL.name
ALIAS = 'texas-split'
ALLOWED_MODELS = {'texas', MODEL_ID, ALIAS}
BACKEND = VERSION + '-native-priority'    # X-Texas-Backend
SUSHI = cfg_path('TEXAS_SPLIT_SUSHI_BIN', REPO / 'sushi/zig-out/bin/sushi')   # another build: experiments only (unpinned)
LIBS = REPO / 'runtime/mlx-lib'           # MLX runtime libraries (install.sh)
CAPS = REPO / 'runtime/memory_caps.dylib'
STAGE_DIR = REPO / 'stage'
STAGE_BIN = cfg_path('TEXAS_SPLIT_STAGE_BIN', STAGE_DIR / 'target/release/canon-stage-server')   # likewise
FIRMWARE = REPO / 'driver/libtinynv/third_party/firmware'
MACUDA = cfg_path('TEXAS_SPLIT_MACUDA', REPO / 'third_party/macuda')
CACHE = cfg_path('TEXAS_SPLIT_CACHE', Path.home() / 'Library/Caches/Texas-Split')
# one GPU owner at a time (this host holds the lock for its lifetime)
LOCKS = [CACHE / 'gpu.lock']
STAGE_FIRST = 28
CONTEXT = 262144
STAGE_CAPACITY = 262160   # qsa::MAX_POSITION + MAX_ROWS: the stage's bound, enough for the 262144 context
PORT = int(CFG.get('TEXAS_SPLIT_PORT') or 8003)
ENGINE_PORT = int(CFG.get('TEXAS_SPLIT_ENGINE_PORT') or 8014)
FOREIGN_PORTS = ()
GPU_DEVID = CFG.get('TEXAS_SPLIT_GPU_DEVID') or '0x2b85'   # RTX 5090 (GB202)
CONTROL_KEY = cfg_path('TEXAS_SPLIT_CONTROL_KEY', Path.home() / '.config/texas-split/control-key')
MAX_SLOTS = 8                             # at most 8 requests in flight


def cfg_int(key, default):
    try:
        return int(CFG.get(key) or default)
    except ValueError:
        return default


# Requests generating at once on the split: TEXAS_SPLIT_MAX_CONCURRENT (1-16; `texas-split concurrency N` writes it to
# texas-split.env). 1, the default, is one generation at a time. Above that, up to STAGE_SLOTS requests decode
# through the 5090 at once, interleaved (each one's 5090 half runs while the Mac works on the others'), and any
# further ones run every layer on the Mac beside them (Sushi's continuous batching).
MAX_CONCURRENT = max(1, min(16, cfg_int('TEXAS_SPLIT_MAX_CONCURRENT', 1)))
MAX_SLOTS = max(MAX_SLOTS, MAX_CONCURRENT + 8)
# Conversation slots on the 5090 stage: 1 when one request runs at a time, else up to 4 (TEXAS_SPLIT_SLOTS). Slot 0
# is the stage's whole state as before; extra slot k holds STAGE_SLOT_ROWS rows carved from the top of slot 0's
# 262,160 (slot 0 keeps what the occupied extra slots do not take; Sushi moves an extra slot's request to the Mac
# before a request crosses a boundary). What fits the card beside the stage (free VRAM at ready): 2 slots with 9
# verify rows each and 1536-row prefill pieces 0.40 GiB; 3 slots (9, 9, 4 rows) with 1024-row pieces 0.47 GiB;
# 4 slots (9, 9, 4, 4) with 768-row pieces 0.39 GiB. One slot: 16 rows, 1536-row pieces, 0.58 GiB, as always.
STAGE_SLOTS = 1 if MAX_CONCURRENT == 1 else max(1, min(4, MAX_CONCURRENT, cfg_int('TEXAS_SPLIT_SLOTS', 4)))
# Rows per extra slot: an equal share of the stage's rows by default (2 slots: 131,080; 3: 87,384; 4: 65,536), so
# two long agent conversations both fit. TEXAS_SPLIT_SLOT_ROWS overrides.
STAGE_SLOT_ROWS = max(4096, min(STAGE_CAPACITY // 2, cfg_int('TEXAS_SPLIT_SLOT_ROWS', STAGE_CAPACITY // max(2, STAGE_SLOTS)))) // 8 * 8
STAGE_SLOT_LAYOUT = {1: (None, '1536'), 2: ('9', '1536'), 3: ('9,4', '1024'), 4: ('9,4,4', '768')}[STAGE_SLOTS]
BACKGROUND_MEDIA = {'image', 'image_url', 'input_image', 'video_url', 'input_audio', 'audio_url', 'video', 'audio'}
GIB = 1024 ** 3
LABEL = CFG.get('TEXAS_SPLIT_LABEL') or 'com.texas-split.server'
DOMAIN = f'gui/{os.getuid()}'
PLIST = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
LOG = cfg_path('TEXAS_SPLIT_LOG', Path.home() / 'Library/Logs/Texas-Split/server.log')
PYTHON = CFG.get('TEXAS_SPLIT_PYTHON') or sys.executable
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer',
       'transfer-encoding', 'upgrade', 'content-length'}
EXPOSED = {'/health', '/v1/models', '/v1/chat/completions', '/v1/completions', '/tokenize', '/v1/tokenize',
           '/detokenize', '/v1/detokenize', '/metrics.json'}
GENERATION = {'/v1/chat/completions', '/v1/completions'}
MAX_BODY = {'foreground': 64 * 1024 ** 2, 'background': 4 * 1024 ** 2}
# The cold GSP boot is when every recorded eGPU drop-off happened (display asleep): hold the display
# awake from stage launch until this long after it is ready ('always' holds it for the stage's life).
DISPLAY_HOLD = CFG.get('TEXAS_SPLIT_DISPLAY_HOLD') or 'boot'
DISPLAY_AFTER_READY = 120


_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def encode(value):
    return json.dumps(value, separators=(',', ':')).encode()


# --- request normalization -----------------------------------------------------------------------
def preserve_reasoning_policy(body):
    if body.get('reasoning_effort') == 'auto':
        body.pop('reasoning_effort')
    body.setdefault('reasoning_budget_tokens', -1)
    return body


def recommended_chat_defaults(body):
    nested = body.get('chat_template_kwargs')
    nested = nested if isinstance(nested, dict) else {}
    enabled = body.get('enable_thinking', nested.get('enable_thinking'))
    effort = body.get('reasoning_effort')
    signals = ([enabled] if isinstance(enabled, bool) else [])
    if isinstance(effort, str):
        signals.append(effort not in {'none', 'off'})
    thinking = any(signals) if signals else True
    defaults = dict(temperature=1.0 if thinking else 0.7, top_p=0.95 if thinking else 0.80, top_k=20, min_p=0.0,
                    presence_penalty=0.0 if thinking else 1.5, repetition_penalty=1.0)
    for name, value in defaults.items():
        body.setdefault(name, value)
    return body


def adapt_model_metadata(value):
    if isinstance(value, dict):
        if isinstance(value.get('reasoning_effort_budgets'), dict):
            value['reasoning_effort_budgets'] = {k: (0 if k in {'off', 'none'} else None)
                                                 for k in value['reasoning_effort_budgets']}
        for child in value.values():
            adapt_model_metadata(child)
    elif isinstance(value, list):
        for child in value:
            adapt_model_metadata(child)
    return value


# --- pins -------------------------------------------------------------------------------------------
PIN_FILES = [SUSHI, CAPS, STAGE_BIN, ROOT / 'mac_probes.py', ROOT / 'priority_control.py', ROOT / 'texas_ipc.dylib',
             MACUDA / 'tools/nv_shim_step.sh']
PINS = REPO / 'runtime/PINS.json'         # written by install.sh (texas_split.py write-pins)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_pins():
    if not PINS.exists():
        raise RuntimeError(f'{PINS} is missing: run install.sh (or texas_split.py write-pins after a build)')
    pins = json.loads(PINS.read_text())
    for path, expected in pins['files'].items():
        if sha256(REPO / path) != expected:   # absolute paths (a macuda outside the repo) stay absolute
            raise RuntimeError('Pinned file changed since install (rebuild, then write-pins): ' + path)
    for name, expected in pins.get('libs', {}).items():
        if sha256(LIBS / name) != expected:
            raise RuntimeError('Release runtime library changed: ' + name)
    return pins


def write_pins():
    pins = dict(version=VERSION, written=time.strftime('%Y-%m-%dT%H:%M:%S'),
                files={(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)): sha256(p) for p in PIN_FILES},
                libs={p.name: sha256(p) for p in sorted(LIBS.iterdir()) if p.is_file()})
    PINS.parent.mkdir(parents=True, exist_ok=True)
    PINS.write_text(json.dumps(pins, indent=2) + '\n')
    return pins


# --- small helpers ------------------------------------------------------------------------------------
def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def group_alive(proc):
    # reap the leader first: an unreaped (zombie) leader makes killpg(pgid, 0) fail with EPERM
    proc.poll()
    try:
        os.killpg(proc.pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def client_closed(sock):
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
    except BlockingIOError:
        return False
    except (OSError, ValueError):
        return True


def http_json(port, path, timeout=3):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r:
        return r.status, json.load(r)


class Unavailable(Exception):
    pass


class ClientGone(Exception):
    pass


class SplitController(Controller):
    """The leased-hold controller (priority_control.py); only the reported backend differs."""
    def snapshot(self):
        value = super().snapshot()
        value['backend'] = BACKEND
        return value


# --- the memory/thermal guard -----------------------------------------------------------------------------
class MemoryGuard:
    """The memory/thermal limits, applied to one kernel reading every 0.5 s. Pure (no I/O), so
    tests/test_guard.py can drive it with synthetic readings or replay a run's memory.jsonl.

    A trip stops the engine: reclaimable memory below 12 GiB; compressor occupancy more than 16 GiB above
    its low; swap more than 256 MiB above its low; critical thermal pressure. The lows follow the current
    reading after 300 s idle (growth while the engine sits unused is not its doing).

    Compression growth also stops counting once the idle engine has released its wiring. About a minute
    after Sushi goes idle, macOS unwires its Metal memory (~70-77 GiB wired while it works, ~5-6 GiB some
    61-63 s after the last request; it rewires within 1-2 s of the next one). From then on the kernel may
    compress those pages to make room for other apps: memory the engine gives back, not pressure it puts
    on the Mac. Before 2026-11-10 that could trip the guard (+17.7 GiB of compressor in one 6 s step at
    ~97 s idle, swap flat, ~29 GiB still reclaimable), and the service then stayed down until restarted.
    So while the engine is idle and its wired memory is below half its level while working, the
    compression low follows the reading too. The reclaimable floor, the swap check and the thermal check
    are unchanged."""

    RECLAIMABLE_FLOOR = 12 * GIB
    COMPRESSION_GROWTH = 16 * GIB
    SWAP_GROWTH = 256 * 1024 ** 2
    IDLE_RESET_S = 300

    def __init__(self, baseline, now):
        self.compressor_low = baseline.compressor_bytes
        self.swap_low = baseline.swap_used_bytes
        self.last_busy = now
        # the most wired memory seen while the engine worked (never lowered: a short request may end before
        # Sushi has finished rewiring, ~1-2 s, so its own peak would understate the level)
        self.busy_wired = baseline.wired_bytes
        self.unwired = False                     # idle, and wired below half of busy_wired
        self.unwired_low = None                  # compressor low since the engine released its wiring
        self.noted = False                       # the uncounted-growth note: once per idle period
        self.note = None

    def update(self, reading, busy, thermal, now):
        """One reading; returns the fault that stops the engine, or None. Afterwards self.unwired says
        whether the engine counted as unwired, and self.note holds a log line (at most once per idle
        period) when compression grew past the limit while that was so."""
        self.note = None
        if busy:
            self.busy_wired = max(self.busy_wired, reading.wired_bytes)
            self.last_busy = now
            self.noted = False
        unwired = not busy and reading.wired_bytes < self.busy_wired / 2
        compressor, swap = reading.compressor_bytes, reading.swap_used_bytes
        if unwired and compressor is not None:
            if not self.unwired or self.unwired_low is None:
                self.unwired_low = self.compressor_low if self.compressor_low is not None else compressor
            self.unwired_low = min(self.unwired_low, compressor)
            if compressor - self.unwired_low > self.COMPRESSION_GROWTH and not self.noted:
                self.noted = True
                self.note = (f'guard: compression grew {(compressor - self.unwired_low) / GIB:.1f} GiB while the '
                             f'engine was idle and unwired (its own pages); not counted')
        elif not unwired:
            self.unwired_low = None
        self.unwired = unwired
        if now - self.last_busy > self.IDLE_RESET_S:
            self.compressor_low, self.swap_low = compressor, swap
        elif unwired:
            self.compressor_low = compressor
        if compressor is not None:
            self.compressor_low = compressor if self.compressor_low is None else min(self.compressor_low, compressor)
        if swap is not None:
            self.swap_low = swap if self.swap_low is None else min(self.swap_low, swap)
        if reading.available_bytes < self.RECLAIMABLE_FLOOR:
            return 'reclaimable memory below 12 GiB'
        if compressor is not None and compressor - self.compressor_low > self.COMPRESSION_GROWTH:
            return 'compression growth above 16 GiB'
        if swap is not None and swap - self.swap_low > self.SWAP_GROWTH:
            return 'swap growth above 256 MiB'
        if thermal in {'trapping', 'sleeping'}:
            return 'critical thermal pressure'
        return None


# --- the host ---------------------------------------------------------------------------------------------
class Host:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Condition()
        self.state = 'starting'          # starting | ready | restarting | fault | stopping
        self.mode = 'split'              # split | mac-only
        self.mode_reason = None
        self.fault = None
        self.inflight = {'foreground': 0, 'background': 0}   # generation requests past admission
        self.waiting = 0                 # requests waiting for the engine to be ready
        self.slots = threading.BoundedSemaphore(MAX_SLOTS)
        self.served = 0
        self.started = time.time()
        self.stage = None
        self.engine = None
        self.stage_info = {}
        self.restarts = []
        self.stopping = threading.Event()
        self.held = []
        self.display = None
        self.thermal = notify_thermal_pressure_reader()
        self.run_dir = args.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.want_split = not args.mac_only
        self.restore_at = None           # monotonic time of the next attempt to bring the split back
        self.restore_delay = 120
        # Native priority: the private control key, an exclusive checkpoint
        # directory, the private IPC page Sushi polls, and the leased-hold controller.
        if CONTROL_KEY.stat().st_mode & 0o077:
            raise RuntimeError('Private scheduler key must have mode 0600')
        self.control_key = CONTROL_KEY.read_text().strip()
        if len(self.control_key) < 32:
            raise RuntimeError('Invalid private scheduler key')
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.checkpoint_dir, 0o700)
        self.checkpoint_lock = (self.checkpoint_dir / '.owner.lock').open('a+b')
        os.chmod(self.checkpoint_dir / '.owner.lock', 0o600)
        fcntl.flock(self.checkpoint_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.clean_checkpoints()
        self.ipc = IPC(self.run_dir / 'control.ipc')
        self.controller = SplitController(self.ipc)
        self.standin = threading.Thread(target=self.stand_in, name='ipc-stand-in', daemon=True)
        self.standin.start()

    # ---- native priority ----
    def clean_checkpoints(self):
        """Sushi's request-N.bin and the stage's stage-request-N.bin parks of an engine that is gone."""
        for path in self.checkpoint_dir.iterdir():
            if re.fullmatch(r'(stage-)?request-[0-9]+\.bin', path.name):
                st = path.lstat()
                if path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
                    raise RuntimeError('Unexpected checkpoint ownership or permissions')
                path.unlink()

    def reset_priority(self):
        """A (re)started engine has no background stack: clear the old engine's page fields and parks."""
        for field in ('state', 'tokens', 'prompt', 'phase', 'park_error', 'restore_wait', 'freed', 'bytes'):
            self.ipc.set(field, 0)
        self.clean_checkpoints()

    def stand_in(self):
        """While no engine serves (starting, restarting, faulted), no background stack exists: answer
        the page's drain handshake for it, so a router hold never times out on a missing engine."""
        while not self.stopping.wait(0.02):
            if self.state != 'ready':
                self.ipc.set('state', 0)
                self.ipc.set('ack', self.ipc.get('seq'))

    def close_priority(self):
        self.controller.close()
        self.ipc.close()
        try:
            self.clean_checkpoints()
        except (OSError, RuntimeError) as e:
            log(f'checkpoint cleanup: {e!r}')
        self.checkpoint_lock.close()

    # ---- admission ----
    def wait_ready(self, conn):
        """Until the engine is ready (requests wait while it starts or restarts), the service faults or
        stops (Unavailable), or the client leaves (ClientGone)."""
        with self.lock:
            if self.state == 'ready':
                return
            self.waiting += 1
            log(f'request waiting for the engine (state {self.state})')
            try:
                while True:
                    if self.state in ('fault', 'stopping'):
                        raise Unavailable(self.fault or 'texas-split is stopping')
                    if self.state == 'ready':
                        return
                    self.lock.wait(0.1)
                    if client_closed(conn):
                        log('request: client left while waiting for the engine')
                        raise ClientGone()
            finally:
                self.waiting -= 1

    def begin(self, mode):
        with self.lock:
            self.inflight[mode] += 1

    def end(self, mode):
        with self.lock:
            self.inflight[mode] -= 1
            self.served += 1
            self.lock.notify_all()

    def set_state(self, state, fault=None):
        with self.lock:
            self.state = state
            if fault:
                self.fault = fault
            self.lock.notify_all()
        log(f'state {state}' + (f': {fault}' if fault else '') + f' (mode {self.mode})')

    def health(self):
        with self.lock:
            ready = self.state == 'ready'
            return ready, dict(backend=BACKEND, version=VERSION, status=self.state, ready=ready, mode=self.mode,
                               mode_reason=self.mode_reason, fault=self.fault, waiting=self.waiting,
                               foreground=self.inflight['foreground'], background=self.inflight['background'],
                               served=self.served,
                               uptime_s=round(time.time() - self.started, 1),
                               engine_pid=self.engine.pid if self.engine else None,
                               stage_pid=self.stage.pid if self.stage else None,
                               stage=self.stage_info or None, restarts=len(self.restarts),
                               model=MODEL_ID, context=CONTEXT, endpoint=f'http://127.0.0.1:{PORT}/v1',
                               concurrency=dict(max=MAX_CONCURRENT, stage_slots=STAGE_SLOTS, slot_rows=STAGE_SLOT_ROWS))

    # ---- locks ----
    def clean_old_runs(self):
        """Stage shm files of earlier runs (their stage servers are gone: this host holds the GPU lock)."""
        for shm in self.run_dir.parent.glob('*/stage.shm'):
            if shm.parent != self.run_dir:
                try:
                    shm.unlink()
                except OSError as e:
                    log(f'old stage shm {shm}: {e!r}')

    def acquire_locks(self):
        for lock in LOCKS:
            owner = lock / 'owner.json'
            if lock.exists():
                try:
                    info = json.loads(owner.read_text())
                except (OSError, ValueError):
                    info = {}
                pid = info.get('pid')
                if isinstance(pid, int) and not pid_alive(pid):
                    log(f'removing stale lock {lock} (owner {info.get("name")} pid {pid} is gone)')
                    try:
                        owner.unlink()
                    except FileNotFoundError:
                        pass
                    lock.rmdir()
                elif info.get('name') == ALIAS and pid == os.getpid():
                    self.held.append(lock)
                    continue
                else:
                    raise RuntimeError(f'{lock} is held by {info.get("name")} (pid {pid}); stop it first')
            lock.mkdir()
            self.held.append(lock)
            owner.write_text(json.dumps({'pid': os.getpid(), 'name': ALIAS, 'started': time.time()}))

    def release_locks(self):
        for lock in reversed(self.held):
            try:
                (lock / 'owner.json').unlink()
                lock.rmdir()
            except OSError as e:
                log(f'lock release {lock}: {e!r}')
        self.held = []

    # ---- display (eGPU cold-boot policy) ----
    def hold_display(self):
        if self.display is None or self.display.poll() is not None:
            self.display = subprocess.Popen(['/usr/bin/caffeinate', '-d', '-w', str(os.getpid())],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['/usr/bin/caffeinate', '-u', '-t', '2'], stdin=subprocess.DEVNULL, timeout=15)

    def release_display(self):
        if self.display is not None and self.display.poll() is None:
            self.display.terminate()
            try:
                self.display.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.display.kill()
        self.display = None

    # ---- launch ----
    def preflight(self):
        for port in FOREIGN_PORTS + (ENGINE_PORT,):
            if port_open(port):
                raise RuntimeError(f'port {port} already has a listener (an experiment?); refusing contention')
        reading = _read_kernel()
        if reading is None:
            raise RuntimeError('memory probe unavailable')
        if reading.available_bytes < 82 * GIB:
            raise RuntimeError(f'reclaimable memory {reading.available_bytes / GIB:.1f} GiB < 82 GiB')
        return reading

    def launch_stage(self):
        # a Unix socket path must fit sun_path (104 bytes on macOS): a short private directory, not the run dir
        sock_dir = Path(f'/tmp/texas-split-{os.getuid()}')
        sock_dir.mkdir(mode=0o700, exist_ok=True)
        st = sock_dir.lstat()
        if sock_dir.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise RuntimeError(f'{sock_dir} must be a private directory owned by this user (mode 0700)')
        sock, shm = str(sock_dir / f'stage-{os.getpid()}.sock'), str(self.run_dir / 'stage.shm')
        for p in (sock, shm):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TINYNV_', 'TINYCUDART_', 'TINYCUBLAS_', 'TEXAS_', 'SUSHI_', 'QD_', 'DYLD_'))}
        for key in ('DRY', 'QUIESCE', 'GMALLOC', 'MACUDA_SHARED_ROOT', 'DART_STALE', 'DEXT_SEEN'):
            env.pop(key, None)
        env.update(EGPU_ROOT=str(MACUDA), EGPU_GPU_DEVID=GPU_DEVID, TINYNV_GRAPH_RESIDENT='1',
                   TINYNV_DOWNLOAD_VIA_COMPUTE='1', TINYNV_QMD_MEMBAR='sys', TINYNV_INLINE_UPLOAD='0',
                   QWEN_HYBRID_GUARDED='1', QD_STAGE_SOCKET=sock, QD_STAGE_SHM=shm,
                   QWEN_HYBRID_MODEL=str(MODEL), TINYNV_FW_DIR=str(FIRMWARE),
                   # every restore of a parked background re-reads the live stage state and must equal
                   # its park file byte for byte (a mismatch fails the restore: the request re-runs
                   # Mac-only); TEXAS_SPLIT_VERIFY_PARKS=0 skips the re-read
                   QD_STAGE_RESTORE_VERIFY=CFG.get('TEXAS_SPLIT_VERIFY_PARKS') or '1',
                   # the stage head (final mixer + 8-bit lm_head + top-20 on the 5090; sushi/FORK.md section 6).
                   # Its 0.71 GB fits beside the 262K-row state only with 1536-row prefill pieces (2048 leaves
                   # too little for the first decode); time to first token is unchanged. QD_STAGE_HEAD='0'
                   # (with SUSHI_R64_STAGE_HEAD='0') disables it.
                   QD_STAGE_HEAD='1', QD_STAGE_PREFILL_ROWS=STAGE_SLOT_LAYOUT[1],
                   # each MoE block's shared expert runs as a side branch beside the router chain and the routed
                   # experts (driver launch latches, stage/README.md); bit-identical, 1-7 row forwards 0.24-0.36 ms
                   # faster, prefill unchanged. '0' disables it.
                   QD_STAGE_SIDE='1')
        if STAGE_SLOTS > 1:
            # conversation slots (see STAGE_SLOTS)
            env.update(QD_STAGE_SLOTS=str(STAGE_SLOTS), QD_STAGE_SLOT_ROWS=str(STAGE_SLOT_ROWS),
                       QD_STAGE_SLOT_VERIFY_ROWS=STAGE_SLOT_LAYOUT[0])
        # experiments only: extra stage variables as a JSON object (QD_ / QWEN_ names)
        extra = json.loads(CFG.get('TEXAS_SPLIT_STAGE_ENV') or '{}')
        env.update({k: str(v) for k, v in extra.items() if k.startswith(('QD_', 'QWEN_'))})
        self.hold_display()
        slog = open(self.run_dir / 'stage.log', 'ab', buffering=0)
        self.ensure_tinygpu_server(env, slog)
        self.stage = subprocess.Popen(['sh', str(MACUDA / 'tools/nv_shim_step.sh'), 'local', 'probe', str(STAGE_BIN),
                                       str(STAGE_FIRST), str(STAGE_CAPACITY)], cwd=STAGE_DIR, env=env, stdout=slog,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        log(f'stage server starting (pid {self.stage.pid}, layers {STAGE_FIRST}-47, capacity {STAGE_CAPACITY})')
        self.stage_sock = sock
        t0 = time.time()
        # nv_shim_step.sh keeps the step's own output in macuda's shim log (echoed only when the step
        # ends); the socket appearing is the readiness signal, the shim log's ready line the details.
        while time.time() - t0 < 300:
            if self.stopping.is_set():
                raise RuntimeError('stopping')
            if self.stage.poll() is not None:
                raise RuntimeError(f'stage server exited during startup ({self.stage.returncode}); see stage.log')
            if os.path.exists(sock):
                self.stage_info = dict(first=STAGE_FIRST, capacity=STAGE_CAPACITY, ready_s=round(time.time() - t0, 1))
                try:
                    header = (self.run_dir / 'stage.log').read_text(errors='replace')
                    shim = re.findall(r' log (\S+shim-probe-\S+\.log)', header)
                    if shim:
                        self.stage_info['shim_log'] = shim[-1]
                        for line in Path(shim[-1]).read_text(errors='replace').splitlines():
                            line = line.strip()
                            if line.startswith('{') and '"kind":"ready"' in line:
                                free = json.loads(line).get('vram_free_bytes')
                                if free:
                                    self.stage_info['vram_free_gib'] = round(free / GIB, 2)
                except (OSError, ValueError):
                    pass
                log(f'stage server ready in {time.time() - t0:.0f} s ({self.stage_info.get("vram_free_gib")} GiB VRAM free)')
                return sock, shm
            time.sleep(1)
        raise RuntimeError('stage server not ready within 300 s')

    def stage_quit(self):
        """A stage server whose client never connected (or has gone) waits in accept(): ask it to
        quit through its own protocol (op 5) so it puts the card down on its normal exit path."""
        sock = getattr(self, 'stage_sock', None)
        if not sock or not os.path.exists(sock):
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
                c.settimeout(5)
                c.connect(sock)
                c.sendall((5).to_bytes(4, 'little') + bytes(28))
                c.recv(32)
            log('stage server asked to quit (op 5)')
            return True
        except OSError:
            return False

    def launch_engine(self, stage_env):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TINYNV_', 'TINYCUDART_', 'TINYCUBLAS_', 'TEXAS_', 'SUSHI_', 'QD_', 'DYLD_'))}
        cache = self.args.cache_dir
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        env.update(MLX_ENABLE_TF32='0', SUSHI_PREFIX_CACHE_DIR=str(cache), DYLD_INSERT_LIBRARIES=str(CAPS),
                   DYLD_LIBRARY_PATH=str(LIBS), TEXAS_PRIORITY_IPC=str(self.ipc.path),
                   TEXAS_PRIORITY_CHECKPOINTS=str(self.checkpoint_dir))
        env.update(stage_env)
        # Exact or noise-level speedups (sushi/FORK.md sections 4-7; each switch '0' restores the old path):
        # - bit-identical 4-row router/gate kernels, and a coarse MTP draft head over vocab ids [0, 98304)
        #   (proposal side only: acceptance stays exact, so the output distribution is unchanged);
        env.update(SUSHI_MTP_DENSE_ROWS='1', SUSHI_MTP_DRAFT_HEAD_ROWS='98304')
        # - split-prefill chunk shaping: no sliver chunks at checkpoint boundaries, and a tail sized so the
        #   Mac covers the 5090's previous chunk (chunk placement only, like any prefix-restore point);
        env.update(SUSHI_R63_SPLIT_FRAG='1024', SUSHI_R63_SPLIT_LAST='384')
        # - MTP verify rounds: hand finished Mac layers to the GPU while the CPU builds the rest, keep the
        #   verify's hidden captures lazy, start each draft step as soon as it is built (bit-identical);
        env.update(SUSHI_R64_VERIFY_LADDER='4', SUSHI_R64_VERIFY_NOSETTLE='1', SUSHI_R64_CHAIN_LADDER='1')
        # - the adaptive draft controller's extension gate reads its confidence from the 32-candidate rerank
        #   shortlist instead of a full 248K-row head per draft step (drafter side only).
        env.update(SUSHI_R64_SHORTLIST_CONF='1')
        # - the verify's head on the 5090 with the acceptance law computed on its top-20 rows (the same law,
        #   so the same output distribution; engaged only for top-k <= 20 with no penalty, logit bias or think
        #   penalty, i.e. the thinking defaults), and no separate COMMIT round trip (sushi/FORK.md section 6);
        env.update(SUSHI_R64_STAGE_HEAD='1', SUSHI_R64_SPARSE_ACCEPT='1', SUSHI_STAGE_INFER_COMMIT='1')
        # - prefill keeps each 8-bit weight's dequantized bf16 copy (~4.1 GB; bit-identical), and plans the
        #   last two chunk widths of a split prefill as one covered tail (chunk placement only; section 7).
        env.update(SUSHI_R64_DQ_CACHE_MB='6144', SUSHI_R64_SPLIT_PLAN='1')
        # - keep the Mac GPU's clocks up while a decode forward waits for the 5090: a tiny kernel on a Metal queue
        #   of its own every 0.5 ms (touches only its own buffer; outputs unchanged; sushi/FORK.md section 8).
        env.update(SUSHI_R65_GPU_WARM='1')
        if stage_env:
            # On the split, the windowed MTP depth controller (depth 3, stepped down only on a cold window) instead of
            # the adaptive planner: the planner's measured round-cost table cannot resolve the split's costs, and it
            # answered cheaper rounds with shallower drafts. Depth policy only (speculative sampling keeps the
            # target distribution at any depth); the text also stops depending on timing. Mac-only keeps the planner.
            env.update(SUSHI_MTP_ADAPTIVE='0', SUSHI_MTP_COST_TABLE='0', SUSHI_MTP_ADAPTIVE_SERIAL='0')
            # Hot prefix-cache entries keep every recurrent-state checkpoint in RAM (capped at 16 and thinned) instead
            # of only the newest two. The others were meant to live in the SSD record, but a split entry whose reply
            # rows are still on the 5090 cannot persist, so a request that diverged mid-history (an agent's
            # compaction prompt, a pruned tool result) prefilled from 0. Restore points only. '0': stock retention.
            env.update(SUSHI_HOT_KEEP_CPS='1')
            # Incremental stage seed (stage op 10): the stage keeps the rows of the sequence it last held; a request
            # whose prompt shares them seeds only rows [base, position) and the recurrent states (~50 MB) instead of
            # every row (~450 MB, ~100 ms at 70K tokens). The stage hashes its rows below base against Sushi's before
            # accepting; a refusal falls back to the full seed. '0': always the full seed.
            env.update(SUSHI_STAGE_ISEED='1')
            # End-of-turn pull-back (stage op 9): after a split turn the reply's rows of layers 28-47 and the
            # recurrent states are copied back from the 5090 and a checkpoint is taken at the reply's end, so the
            # next turn resumes after the reply instead of prefilling it again, and the entry can persist. '0': off.
            env.update(SUSHI_STAGE_PULLBACK_END='1')
            # An image-bearing request whose uncached rows are all text at one rotary offset (a warm turn with no
            # new image) prefills through the stage with that offset, as decode rows already do; image rows and
            # cold image prompts stay on the Mac. Warm image-session turns reach the first token ~30% sooner. '0': off.
            env.update(SUSHI_STAGE_MROPE_TEXT='1')
            if STAGE_SLOTS > 1:
                # the stage's conversation slots, as launched above
                env.update(SUSHI_STAGE_SLOTS=str(STAGE_SLOTS), SUSHI_STAGE_SLOT_ROWS=str(STAGE_SLOT_ROWS),
                           SUSHI_STAGE_SLOT_VERIFY_ROWS=STAGE_SLOT_LAYOUT[0], SUSHI_STAGE_CAPACITY=str(STAGE_CAPACITY))
        # qualification only: extra Sushi variables as a JSON object (e.g. a fixed MTP width)
        extra = json.loads(CFG.get('TEXAS_SPLIT_SUSHI_ENV') or '{}')
        env.update({k: str(v) for k, v in extra.items() if k.startswith('SUSHI_')})
        self.reset_priority()
        if stage_env:
            dead = self.run_dir / 'stage-dead'
            try:
                dead.unlink()
            except FileNotFoundError:
                pass
            env.update(SUSHI_STAGE_DEAD_FILE=str(dead),
                       SUSHI_STAGE_TIMEOUT_MS=CFG.get('TEXAS_SPLIT_STAGE_TIMEOUT_MS') or '20000')
        cmd = [str(SUSHI), 'serve', '--model', str(MODEL), '--host', '127.0.0.1', '--port', str(ENGINE_PORT),
               '--ctx-size', str(CONTEXT),
               '--config-overrides', '{"text_config":{"rope_parameters":{"rope_type":"default"},"max_position_embeddings":262144}}',
               '--max-tokens', '131072', '--temp', '1', '--top-p', '0.95', '--top-k', '20',
               '--mtp', '--kv-quant', '8', '--qwen-gdn', '32', '--mtp-head-kv-quant', '--think', 'xhigh',
               '--max-resident-models', '1', '--max-resident-mem', '108GB',
               # the split's idle allowance is 2 GB (1 GB Mac-only): its entries also carry their checkpoints (~110 MB
               # each), and an idle conversation that overflows the allowance cannot spill to SSD, it is dropped
               '--max-concurrent', str(MAX_CONCURRENT if stage_env else 1), '--prefix-cache-entries', '4',
               '--prefix-cache-mem', '2GB' if stage_env else '1GB',
               '--prefix-cache-disk', '20GB', '--metrics', '--no-update-check',
               '--log-file', str(self.run_dir / 'sushi-native.log')]
        elog = open(self.run_dir / 'engine.log', 'ab', buffering=0)
        self.engine = subprocess.Popen(cmd, env=env, stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
        (self.run_dir / 'launch.json').write_text(json.dumps(dict(started=time.time(), pid=self.engine.pid, command=cmd,
                                                                  stage=stage_env, mode=self.mode), indent=2) + '\n')
        log(f'Sushi starting (pid {self.engine.pid}, mode {self.mode})')
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.stopping.is_set():
                raise RuntimeError('stopping')
            if self.engine.poll() is not None:
                raise RuntimeError(f'Sushi exited during startup ({self.engine.returncode}); see engine.log')
            if self.stage is not None and self.stage.poll() is not None:
                raise RuntimeError('stage server exited while Sushi was loading')
            try:
                status, data = http_json(ENGINE_PORT, '/v1/models')
                models = data.get('data') or []
                if status == 200 and models and models[0].get('loaded', True):
                    log(f'Sushi ready in {time.time() - t0:.0f} s')
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError('Sushi not ready within 900 s')

    def launch(self):
        self.preflight()
        stage_env = {}
        self.stage_info = {}
        if self.want_split:
            try:
                sock, shm = self.launch_stage()
                stage_env = dict(SUSHI_STAGE_SOCKET=sock, SUSHI_STAGE_SHM=shm, SUSHI_STAGE_FIRST=str(STAGE_FIRST))
                self.mode, self.mode_reason = 'split', None
                self.restore_at, self.restore_delay = None, 120
            except Exception as e:
                if self.stopping.is_set():
                    raise
                log(f'stage unavailable, running every layer on the Mac: {e}')
                self.stop_stage()
                self.mode, self.mode_reason = 'mac-only', f'stage failed to start: {e}'
                self.schedule_restore()
        else:
            self.mode, self.mode_reason = 'mac-only', 'started with --mac-only'
        self.launch_engine(stage_env)
        if DISPLAY_HOLD != 'always':
            timer = threading.Timer(DISPLAY_AFTER_READY, self.release_display)
            timer.daemon = True
            timer.start()

    def schedule_restore(self):
        """The split is down: try to bring it back later (2 min, doubling to an hour)."""
        if not self.want_split:
            return
        self.restore_at = time.monotonic() + self.restore_delay
        log(f'will try to restore the split when idle, in {self.restore_delay // 60} min or later')
        self.restore_delay = min(3600, self.restore_delay * 2)

    # ---- teardown: Sushi first; the stage server exits by itself when its socket closes ----
    def stop_engine(self):
        p, self.engine = self.engine, None
        if p is not None and group_alive(p):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            t_end = time.time() + 30
            while time.time() < t_end and group_alive(p):
                time.sleep(0.25)
            if group_alive(p):
                log('Sushi did not exit after SIGTERM; SIGKILL')
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.wait(timeout=10)

    def ensure_tinygpu_server(self, env, slog):
        """The TinyGPU server outlives stage servers. With none listening (after a reboot or a replug) the stage step
        starts one itself, inside its own process group: that group then never empties when the stage server exits, so
        every stop waits out its 90 s and signals the group, killing the TinyGPU server each time. Start it here
        instead, in a session of its own, after the same read-only preflight the step runs; the step's own `ensure`
        then finds it listening. Any failure here leaves the step to do it as before."""
        try:
            pf = subprocess.run(['sh', str(MACUDA / 'tools/preflight.sh')], env=env, capture_output=True, text=True,
                                timeout=120)
            verdict = (pf.stdout.strip().splitlines() or [''])[-1]
            if 'VERDICT: OK' not in verdict:
                return
            r = subprocess.run(['sh', str(MACUDA / 'tools/tinygpu-server.sh'), 'ensure'], env=env, capture_output=True,
                               text=True, timeout=60, start_new_session=True)
            slog.write(('host: ' + (r.stdout.strip().splitlines() or ['tinygpu-server ensure: no output'])[-1] + '\n').encode())
        except (OSError, subprocess.SubprocessError) as e:
            log(f'tinygpu server pre-start skipped: {e}')

    def stop_stage(self):
        s, self.stage = self.stage, None
        if s is None:
            return
        # it exits by itself once Sushi's connection closes; an idle one never connected is asked to quit
        t_end = time.time() + 10
        while time.time() < t_end and group_alive(s):
            time.sleep(0.5)
        if group_alive(s):
            self.stage_quit()
        t_end = time.time() + 90
        while time.time() < t_end and group_alive(s):
            time.sleep(0.5)
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            if not group_alive(s):
                break
            log(f'stage server still running after its socket closed; {signal.Signals(sig).name}')
            try:
                os.killpg(s.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            t_end = time.time() + wait
            while time.time() < t_end and group_alive(s):
                time.sleep(0.5)
        log(f'stage server stopped ({s.returncode})')
        # the shm file keeps every page a seed or prefill ever touched (up to ~1.6 GB): drop it with the stage
        for path in (self.run_dir / 'stage.shm', Path(getattr(self, 'stage_sock', None) or self.run_dir / 'stage.sock')):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def teardown(self):
        self.stop_engine()
        self.stop_stage()
        if DISPLAY_HOLD == 'always':
            self.release_display()

    # ---- supervision ----
    def watch(self):
        """Returns why the running stack must stop: 'stop', 'guard', 'engine', or 'stage'."""
        baseline = _read_kernel()
        if baseline is None:
            self.fault = 'memory probe unavailable'
            return 'guard'
        guard = MemoryGuard(baseline, time.monotonic())
        with (self.run_dir / 'memory.jsonl').open('a', buffering=1) as mem:
            while not self.stopping.wait(0.5):
                reading = _read_kernel()
                if reading is None:
                    self.fault = 'memory probe unavailable'
                    return 'guard'
                thermal = level_name(self.thermal())
                with self.lock:
                    busy = sum(self.inflight.values()) > 0 or self.waiting > 0
                fault = guard.update(reading, busy, thermal, time.monotonic())
                if guard.note:
                    log(guard.note)
                mem.write(json.dumps(reading.to_dict() | dict(time=time.time(), thermal=thermal, busy=busy,
                                                              engine_unwired=guard.unwired)) + '\n')
                if fault:
                    self.fault = fault
                    return 'guard'
                if self.engine is None or self.engine.poll() is not None:
                    return 'engine'
                if self.mode == 'split':
                    dead = self.run_dir / 'stage-dead'
                    lost = None
                    if dead.exists():
                        lost = 'Sushi gave up on the stage: ' + dead.read_text(errors='replace').strip()
                    elif self.stage is None or not group_alive(self.stage):
                        # the stage server runs under nv_shim_step.sh in its own process group: the
                        # group, not the wrapper, is what is alive
                        lost = f'stage server exited ({None if self.stage is None else self.stage.returncode})'
                    if lost:
                        # Sushi carries the request it was serving over to the Mac by itself (it re-runs
                        # it Mac-only); the stage server is let go of in the background.
                        self.mode, self.mode_reason = 'mac-only', lost
                        log(f'split lost: {lost}; Sushi continues on the Mac')
                        threading.Thread(target=self.stop_stage, name='stage-stop', daemon=True).start()
                        self.schedule_restore()
                elif (self.restore_at is not None and time.monotonic() >= self.restore_at
                      and time.monotonic() - guard.last_busy > 120):
                    return 'restore'
        return 'stop'

    def supervise(self):
        try:
            self.acquire_locks()
        except Exception as e:
            self.set_state('fault', f'locks: {e}')
            return
        self.clean_old_runs()
        failures = 0
        while not self.stopping.is_set():
            try:
                self.launch()
                failures = 0
                self.set_state('ready')
            except Exception as e:
                self.teardown()
                if self.stopping.is_set():
                    break
                failures += 1
                log(f'launch failed ({failures}): {e}')
                if failures >= 4 or 'refusing contention' in str(e) or 'reclaimable memory' in str(e):
                    self.set_state('fault', f'launch failed: {e}')
                    break
                self.set_state('restarting')
                self.stopping.wait(min(300, 15 * 2 ** failures))
                continue
            why = self.watch()
            if why == 'stop':
                break
            if why == 'guard':
                log(f'guard tripped: {self.fault}; stopping the engine')
                self.set_state('fault', self.fault)
                self.teardown()
                break
            if why == 'restore':
                log('idle in mac-only mode: restarting the stack to bring the split back')
                self.restore_at = None
                self.set_state('restarting')
                self.teardown()
                continue
            self.restarts.append(time.time())
            self.restarts = [t for t in self.restarts if time.time() - t < 3600]
            reason = 'Sushi exited' if why == 'engine' else 'stage server exited'
            log(f'{reason}; restarting the stack ({len(self.restarts)} restart(s) in the last hour)')
            self.set_state('restarting')
            self.teardown()
            if len(self.restarts) > 6:
                self.set_state('fault', f'{reason} repeatedly ({len(self.restarts)} restarts in an hour)')
                break
            self.stopping.wait(5)
        self.teardown()
        self.release_locks()
        self.release_display()

    def shutdown(self):
        if self.stopping.is_set():
            return
        self.stopping.set()
        with self.lock:
            if self.state != 'fault':
                self.state = 'stopping'
            self.lock.notify_all()
        log('stopping')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'texas-split'

    def log_message(self, fmt, *args):
        pass

    def json(self, status, value):
        data = encode(value)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def error(self, status, message, code):
        self.json(status, {'error': {'message': message, 'type': 'invalid_request_error' if status < 500 else 'server_error',
                                     'code': code}})

    def do_GET(self):
        self.proxy()

    def do_POST(self):
        self.proxy()

    def models(self, host):
        # Sushi's listing with the effort-budget adaptation (the model id stays MODEL_ID)
        if host.state != 'ready':
            return self.error(503, f'Texas (texas-split) is {host.state}' + (f': {host.fault}' if host.fault else ''),
                              'backend_unavailable')
        try:
            status, data = http_json(ENGINE_PORT, '/v1/models')
        except (OSError, ValueError) as e:
            return self.error(502, 'Sushi transport failed: ' + type(e).__name__, 'upstream_error')
        return self.json(status, adapt_model_metadata(data))

    def health(self, host):
        # Sushi's own health when it serves, plus this host's state
        ready, value = host.health()
        if ready:
            try:
                status, data = http_json(ENGINE_PORT, '/health')
                if isinstance(data, dict):
                    engine_status = data.pop('status', None)
                    value = data | value | dict(engine_status=engine_status)
                return self.json(status, value)
            except (OSError, ValueError):
                pass
        return self.json(200 if ready else 503, value)

    def scheduler(self, host):
        # leased native holds (e.g. a router's preemption control URL), authenticated with the control key
        if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + host.control_key):
            return self.error(401, 'Unauthorized', 'unauthorized')
        if self.command == 'GET':
            snap = host.controller.snapshot()
            _, h = host.health()
            snap['split'] = dict(mode=h['mode'], mode_reason=h['mode_reason'], status=h['status'])
            return self.json(200, snap)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                return self.error(413, 'Control body must be at most 4096 bytes', 'invalid_body_size')
            self.connection.settimeout(30)
            result = host.controller.control(json.loads(self.rfile.read(length)))
            self.json(200, result)
        except (ValueError, TypeError) as e:
            self.error(400, str(e), 'invalid_control')
        except TimeoutError as e:
            self.error(503, str(e), 'control_timeout')

    def proxy(self):
        host = self.server.host
        path = self.path.split('?')[0]
        if path == '/v1/scheduler':
            return self.scheduler(host)
        if path not in EXPOSED:
            return self.error(404, 'Endpoint is not exposed by Texas', 'endpoint_not_found')
        mode = self.headers.get('X-LocalAI-Mode', self.headers.get('X-LLM-Priority', 'foreground')).strip().lower()
        if mode not in {'foreground', 'background'}:
            return self.error(400, 'Priority must be foreground or background', 'invalid_priority')
        if path == '/health':
            return self.health(host)
        if host.state in ('fault', 'stopping'):
            return self.error(503, 'Texas is unavailable: ' + str(host.fault or host.state), 'backend_unavailable')
        if path == '/v1/models':
            return self.models(host)
        if not host.slots.acquire(blocking=False):
            return self.error(429, 'Texas request queue is full', 'capacity_exceeded')
        generation = path in GENERATION and self.command == 'POST'
        conn = ticket = watcher = None
        counted = False
        finished = threading.Event()
        response_started = False
        try:
            body = None
            if self.command == 'POST':
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_BODY[mode]:
                    return self.error(413, 'Request body exceeds the bounded JSON limit', 'invalid_body_size')
                self.connection.settimeout(60)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    return self.error(400, 'Incomplete request body', 'invalid_request')
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    return self.error(400, 'Expected JSON object', 'invalid_request')
                if obj.get('model', MODEL_ID) not in ALLOWED_MODELS:
                    return self.error(404, 'Unknown Texas model', 'model_not_found')
                if mode == 'background' and generation:
                    def media(v):
                        if isinstance(v, dict):
                            return v.get('type') in BACKGROUND_MEDIA or any(media(x) for x in v.values())
                        return isinstance(v, list) and any(media(x) for x in v)
                    if media(obj):
                        return self.error(400, 'Texas background mode accepts text only', 'background_media_unsupported')
                    if obj.get('echo') and obj.get('logprobs') is not None:
                        return self.error(400, 'Background prompt scoring is not qualified', 'background_scoring_unsupported')
                obj['model'] = MODEL_ID
                if generation:
                    preserve_reasoning_policy(obj)
                    if path == '/v1/chat/completions':
                        recommended_chat_defaults(obj)
                if path in {'/tokenize', '/v1/tokenize'} and 'content' not in obj:
                    for field in ('text', 'prompt'):
                        if isinstance(obj.get(field), str):
                            obj['content'] = obj.pop(field)
                            break
                body = encode(obj)
            self.connection.settimeout(None)
            if generation and mode == 'background':
                with host.lock:
                    if host.inflight['background']:
                        raise BlockingIOError('Texas permits one native background request')
            host.wait_ready(self.connection)
            if generation:
                # foreground: hold any background stack and wait until Sushi parked it at a boundary
                ticket = host.controller.enter(mode)
                host.begin(mode)
                counted = True
            conn = http.client.HTTPConnection('127.0.0.1', ENGINE_PORT, timeout=1800)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP | {'host', 'authorization', 'x-api-key', 'x-localai-mode', 'x-llm-priority'}}
            headers['X-LocalAI-Mode'] = mode
            if body is not None:
                headers['Content-Type'] = 'application/json'
                headers['Content-Length'] = str(len(body))
            upstream_path = {'/v1/tokenize': '/tokenize', '/v1/detokenize': '/detokenize'}.get(path, path)
            conn.request(self.command, upstream_path, body=body, headers=headers)
            if generation:
                # a paused stream may send nothing for a long time: notice a departing client on our
                # own and close upstream, so Sushi cancels the slot (queued, running or parked)
                upstream = conn.sock

                def watch_disconnect():
                    while not finished.wait(0.05):
                        if client_closed(self.connection):
                            try:
                                upstream.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            return
                watcher = threading.Thread(target=watch_disconnect, daemon=True)
                watcher.start()
            response = conn.getresponse()
            self.send_response(response.status)
            response_started = True
            for k, v in response.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header('X-Texas-Backend', BACKEND)
            self.send_header('X-Texas-Split-Mode', host.mode)
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            while block := response.read1(65536):
                self.wfile.write(block)
                self.wfile.flush()
        except ClientGone:
            pass
        except BlockingIOError as e:
            if not response_started:
                self.error(429, str(e), 'background_capacity_exceeded')
        except TimeoutError as e:
            if not response_started:
                self.error(503, str(e), 'control_timeout')
        except Unavailable as e:
            if not response_started:
                self.error(503, f'Texas is unavailable: {e}', 'backend_unavailable')
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, TypeError, RecursionError) as e:
            if not response_started:
                self.error(400, str(e), 'invalid_request')
        except (OSError, http.client.HTTPException) as e:
            if not response_started:
                self.error(502, 'Sushi transport failed: ' + type(e).__name__, 'upstream_error')
        finally:
            finished.set()
            if conn:
                conn.close()
            if watcher:
                watcher.join(0.2)
            if ticket:
                host.controller.leave(ticket)
            if counted:
                host.end(mode)
            host.slots.release()


def serve(args):
    os.umask(0o077)
    verify_pins()
    for port in (PORT, ENGINE_PORT) + FOREIGN_PORTS:
        if port_open(port):
            raise RuntimeError(f'Port {port} already has a listener; refusing overlapping deployment')
    # absolute: the stage server and Sushi run in other working directories
    args.run_dir = args.run_dir.expanduser().resolve() / (time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}')
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    host = Host(args)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    server.daemon_threads = True
    server.host = host
    signal.signal(signal.SIGTERM, lambda *_: host.shutdown())
    signal.signal(signal.SIGINT, lambda *_: host.shutdown())
    # a server holds the Mac awake (idle sleep always, system sleep on AC) for exactly its own life
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f'{VERSION} listening on 127.0.0.1:{PORT}; run dir {host.run_dir}')
    supervisor = threading.Thread(target=host.supervise, name='supervisor')
    supervisor.start()
    threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.25), daemon=True).start()
    # until launchd (or the CLI) stops us; a guard fault keeps answering 503 with its reason
    while not host.stopping.wait(0.5):
        pass
    supervisor.join()
    server.shutdown()
    host.close_priority()
    if awake.poll() is None:
        awake.terminate()
    log('stopped')
    return 0


# --- lifecycle CLI ------------------------------------------------------------------------------------------
def call(*args, check=True):
    return subprocess.run(args, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def loaded(label=LABEL):
    return call('launchctl', 'print', f'{DOMAIN}/{label}', check=False).returncode == 0


def job_pid(label=LABEL):
    r = call('launchctl', 'print', f'{DOMAIN}/{label}', check=False)
    m = re.search(r'^\s*pid = (\d+)', r.stdout, re.M) if r.returncode == 0 else None
    return int(m.group(1)) if m else None


def host_health():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except ValueError:
            return None
    except (OSError, ValueError):
        return None


def install_plist(mac_only=False):
    verify_pins()
    LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
    args = [PYTHON, str(ROOT / 'texas_split.py'), 'serve',
            '--run-dir', str(CACHE / 'runs'), '--cache-dir', str(CACHE / 'prefix-cache'),
            '--checkpoint-dir', str(CACHE / 'priority-checkpoints')]
    if mac_only:
        args.append('--mac-only')
    value = dict(Label=LABEL, ProgramArguments=args, RunAtLoad=True, KeepAlive=dict(SuccessfulExit=False),
                 ThrottleInterval=30, ExitTimeOut=300, ProcessType='Interactive',
                 EnvironmentVariables=dict(PATH='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin', PYTHONUNBUFFERED='1'),
                 StandardOutPath=str(LOG), StandardErrorPath=str(LOG))
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLIST.with_suffix('.tmp')
    tmp.write_bytes(plistlib.dumps(value))
    tmp.chmod(0o600)
    tmp.replace(PLIST)


def cli_status():
    h = host_health()
    print(json.dumps(dict(backend='split', label=LABEL, launchd_loaded=loaded(), endpoint=f'http://127.0.0.1:{PORT}/v1',
                          host=h)))
    return bool(h and h.get('ready'))


def cli_start(mac_only=False, wait=True):
    if not loaded() and (port_open(PORT) or port_open(ENGINE_PORT)):
        raise RuntimeError(f'an unmanaged listener holds {PORT}/{ENGINE_PORT}; refusing to start')
    if loaded():
        h = host_health()
        if h and h.get('status') in ('ready', 'starting', 'restarting'):
            return cli_status()
        call('launchctl', 'bootout', f'{DOMAIN}/{LABEL}', check=False)
    install_plist(mac_only)
    call('launchctl', 'enable', f'{DOMAIN}/{LABEL}')
    call('launchctl', 'bootstrap', DOMAIN, str(PLIST))
    if not wait:
        return cli_status()
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        h = host_health()
        if h and h.get('ready'):
            return cli_status()
        if h and h.get('status') == 'fault':
            cli_status()
            raise RuntimeError(f'texas-split faulted: {h.get("fault")}; see {LOG}')
        time.sleep(1)
    raise TimeoutError(f'texas-split did not become ready; inspect {LOG}')


def cli_stop():
    if loaded():
        # Stop the host ourselves and wait for it first: its teardown (Sushi, then the stage server's card
        # put-down, then the GPU lock) can outlast `launchctl bootout`, which kills the job ~20 s after its
        # SIGTERM whatever ExitTimeOut says, leaving the lock behind with a dead owner. A clean exit (0) is
        # not restarted (KeepAlive: SuccessfulExit false); bootout then only unloads the job.
        pid = job_pid()
        if pid and pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 330
            while pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.5)
        call('launchctl', 'bootout', f'{DOMAIN}/{LABEL}', check=False)
        deadline = time.monotonic() + 330
        while (loaded() or port_open(PORT) or port_open(ENGINE_PORT)) and time.monotonic() < deadline:
            time.sleep(0.5)
        if loaded() or port_open(PORT) or port_open(ENGINE_PORT):
            raise RuntimeError('texas-split is still stopping')
    elif port_open(ENGINE_PORT):
        raise RuntimeError(f'an unmanaged listener holds {ENGINE_PORT}; refusing broad process termination')
    print('texas-split stopped.')


def main():
    p = argparse.ArgumentParser(prog='texas-split')
    p.add_argument('action', choices=['serve', 'start', 'stop', 'restart', 'status', 'health', 'logs', 'install-plist', 'uninstall',
                                      'write-pins', 'check', 'concurrency'])
    p.add_argument('value', nargs='?', help='concurrency: requests generating at once (1-16); omit to show')
    p.add_argument('--run-dir', type=Path, default=CACHE / 'runs')
    p.add_argument('--cache-dir', type=Path, default=CACHE / 'prefix-cache')
    p.add_argument('--checkpoint-dir', type=Path, default=CACHE / 'priority-checkpoints')
    p.add_argument('--mac-only', action='store_true', help='run every layer on the Mac (no stage server)')
    p.add_argument('--follow', action='store_true')
    p.add_argument('--no-wait', action='store_true')
    a = p.parse_args()
    if a.action == 'serve':
        return serve(a)
    if a.action == 'concurrency':
        if a.value is not None:
            n = int(a.value)
            if not 1 <= n <= 16:
                print('concurrency: 1 to 16', file=sys.stderr)
                return 2
            path = set_config('TEXAS_SPLIT_MAX_CONCURRENT', n)
            print(f'TEXAS_SPLIT_MAX_CONCURRENT={n} written to {path}; applies at the next `texas-split restart`')
        else:
            # what the configuration says now; a running service keeps what it started with (see /health)
            print(json.dumps(dict(configured=dict(max_concurrent=MAX_CONCURRENT, stage_slots=STAGE_SLOTS,
                                                  slot_rows=STAGE_SLOT_ROWS)), indent=1))
        return 0
    if a.action == 'write-pins':
        print(json.dumps(write_pins(), indent=1))
        return 0
    if a.action == 'check':
        verify_pins()
        print(json.dumps(dict(pins='passed', version=VERSION)))
        return 0
    if a.action == 'install-plist':
        install_plist(a.mac_only)
        return 0
    if a.action == 'uninstall':
        cli_stop()
        PLIST.unlink(missing_ok=True)
        print(f'removed {PLIST}')
        return 0
    if a.action == 'start':
        return 0 if cli_start(a.mac_only, not a.no_wait) else 1
    if a.action == 'stop':
        cli_stop()
        return 0
    if a.action == 'restart':
        cli_stop()
        return 0 if cli_start(a.mac_only, not a.no_wait) else 1
    if a.action == 'status':
        return 0 if cli_status() else 1
    if a.action == 'health':
        print(json.dumps(host_health()))
        return 0
    if a.action == 'logs':
        os.execvp('tail', ['tail', '-n', '200'] + (['-f'] if a.follow else []) + [str(LOG)])
    return 0


if __name__ == '__main__':
    sys.exit(main())
