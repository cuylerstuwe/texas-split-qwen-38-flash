// R60 pipeline split (isolated experiment): Sushi's side of the RTX 5090 stage, which runs the
// canonical target's layers [first, 48) in qwen-hybrid-rs canon-stage-server. One Unix socket
// (SUSHI_STAGE_SOCKET) carries 32-byte messages; one shared file (SUSHI_STAGE_SHM) carries the
// BF16 HC stream rows and state seeds. Enabled only when both variables are set.
//
// Commits are inferred from positions: a forward at P with the previous forward covering
// [base, base+rows) commits P-base rows first (all of them after a full accept or a serial step,
// fewer after a speculative partial accept whose trunk Sushi already rolled back). A normal full
// forward (prefill) invalidates the stage; the next split forward re-seeds it from Sushi's caches.
//
// Sequences (texas-split background parking): the stage holds one sequence's state, named by its
// owner (Sushi's slot: the address of its KV cache). A different owner's first stage call re-seeds
// (or restores). A sequence that decoded on the stage is "resident": its layers [first, 48) exist
// only on the stage, so it must never continue Mac-only from Sushi's caches (the forward errors and
// the scheduler re-runs it from its tokens). Parking (op 7) writes the stage's committed state of
// the owner to an SSD file; its next stage call restores it (op 8) instead of re-seeding.
#include <mlx/c/mlx.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int gpu_warm_init(void);   // lib/gpu_warm.m (R65)
extern "C" void gpu_warm_tick(void);

namespace {
// 2 GiB: the largest state seed (262160 rows: 5 attention layers x 1152 B/row + 15 recurrent states,
// ~1.56 GB) fits whole; canon-stage-server maps the same size.
constexpr size_t kShmBytes = size_t(2) << 30;
constexpr uint32_t kMagic = 0x31534451;  // "QDS1"
constexpr int kF = 10240;
int g_fd = -1;
int g_first = -2;  // -2 unopened, -1 disabled
uint8_t* g_shm = nullptr;
// R69: the stage may hold two conversations (slots, QD_STAGE_SLOTS=2 on the server); each slot's client state below.
// Slot 1 serves the one owner the scheduler assigned it; every other owner is slot 0's, as before slots existed.
struct Seq {
  bool valid = false;    // the stage holds this sequence's state
  uint64_t base = 0;     // first position of the last forward
  uint64_t rows = 0;     // rows of the last forward (uncommitted)
  int head_rows = 0;
  uint32_t head_buf[16 * 40];
  uint64_t owner = 0;    // the sequence whose state this slot holds (valid when valid)
  uint64_t mirror = 0;   // R66: rows known equal to Sushi's caches
  uint32_t* tok = nullptr;
  uint64_t tok_n = 0, tok_cap = 0;
};
constexpr int kSlots = 4;
Seq g_seq[kSlots];
int g_cur = 0;                         // the slot the current call works on (stage_split_bind / the owner's slot)
uint64_t g_slot_owner[kSlots] = {};    // [k >= 1]: the owner assigned slot k (0: none)
#define g_valid (g_seq[g_cur].valid)
#define g_base (g_seq[g_cur].base)
#define g_rows (g_seq[g_cur].rows)
#define g_head_rows (g_seq[g_cur].head_rows)
#define g_head_buf (g_seq[g_cur].head_buf)
#define g_owner (g_seq[g_cur].owner)
#define g_mirror (g_seq[g_cur].mirror)
#define g_tok (g_seq[g_cur].tok)
#define g_tok_n (g_seq[g_cur].tok_n)
#define g_tok_cap (g_seq[g_cur].tok_cap)
int slot_of(uint64_t owner) {
  if (owner)
    for (int k = 1; k < kSlots; k++)
      if (owner == g_slot_owner[k]) return k;
  return 0;
}
size_t g_at = 0;        // seed blob cursor
uint64_t g_forwards = 0, g_seeds = 0, g_partials = 0;
double g_rpc_us = 0, g_copy_us = 0;
// R72: the server's own time for the last op (reply bytes 8..12, us; 0 from a server that does not report it) and its
// sum over forwards: send-to-reply minus this is the transport (measured 2026-11-11: 16-27 us per forward, so a
// spin handoff on either side of the socket has nothing to win).
uint32_t g_reply_us = 0;
double g_stage_us = 0;
// R64 stage head: request the final mixer + lm_head + top-20 with the next forward (op 1 aux bit 2); the reply's
// per-row (20 BF16-valued f32 bit patterns, 20 token ids) are copied out of the shared file at receive time.
int g_want_head = 0;
// R64 (SUSHI_STAGE_INFER_COMMIT=1, a stage server that infers commits from positions): no op 2 round trip.
static int infer_commit() {
  static int v = -1;
  if (v < 0) { const char* e = getenv("SUSHI_STAGE_INFER_COMMIT"); v = (e && e[0] == '1') ? 1 : 0; }
  return v;
}
// R60 S4 (prefill v2): pipelined chunk prefill through the stage (op 6). Two shm slots after the
// forward/seed area; replies arrive in send order. While chunks are in flight no other op may run.
constexpr size_t kSlotBase = size_t(256) << 20;
constexpr size_t kSlotStride = size_t(256) << 20;  // per slot: stream area, then the pull-back blob
constexpr size_t kSlotBytes = size_t(128) << 20;   // stream area (the blob starts here)
uint64_t g_pf_export_len = 0;
int g_pf_export_slot = -1;
int g_pullback = -1;
int g_inflight = 0, g_pf_head = 0;
int g_pf_cur = 0;  // R71: the conversation slot the chunks in flight belong to
uint32_t g_pf_rows[2] = {0, 0};
uint64_t g_pf_pos[2] = {0, 0};
uint64_t g_pf_next = 0, g_pf_chunks = 0;
double g_pf_send_us = 0, g_pf_wait_us = 0;

double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Fatal-stage fallback: a stage that fails, closes or stops answering is dead for the rest of this
// process. Every split entry point then refuses, stage_split_first() reports the split disabled
// (every layer runs on the Mac), and the scheduler re-runs the request it was serving Mac-only.
// SUSHI_STAGE_TIMEOUT_MS bounds each socket wait (default 60 s; a prefill reply gets twice that);
// SUSHI_STAGE_DEAD_FILE receives the reason, for the host.
bool g_dead = false;
// R66 incremental seed (SUSHI_STAGE_ISEED=1): the stage keeps the rows of the sequence it last held. g_tok is the
// prompt those rows belong to (noted at each request's start), g_mirror how many of them are known equal to
// Sushi's caches (seeded from them, or computed by a split prefill and pulled back; decode rows live only on the
// stage and never count). A seed then sends rows [base, position) only; the stage hashes its K rows below base
// against Sushi's before it accepts (op 10), so a wrong belief here costs a full seed, never wrong state.
uint64_t g_iseeds = 0, g_irefused = 0;
constexpr uint32_t kMagicIncr = 0x31494451;  // "QDI1"
int iseed_trace() {
  static int v = -1;
  if (v < 0) { const char* e = getenv("SUSHI_STAGE_ISEED_TRACE"); v = (e && e[0] == '1') ? 1 : 0; }
  return v;
}
#define MIRROR_TRACE(...) do { if (iseed_trace()) fprintf(stderr, __VA_ARGS__); } while (0)
int g_timeout_ms = -1;
uint64_t g_reply_arg = 0;  // the last reply's arg word
uint64_t g_resident[16] = {};
struct Park {
  uint64_t owner, position;
  char path[1024];
};
Park g_parks[8] = {};
uint64_t g_parked = 0, g_restored = 0;
bool resident_has(uint64_t o) {
  for (uint64_t r : g_resident)
    if (o && r == o) return true;
  return false;
}
void resident_add(uint64_t o) {
  if (!o || resident_has(o)) return;
  for (uint64_t& r : g_resident)
    if (!r) { r = o; return; }
  fprintf(stderr, "[stage] residency table full\n");
}
void resident_del(uint64_t o) {
  for (uint64_t& r : g_resident)
    if (r == o) r = 0;
}
Park* park_find(uint64_t o) {
  for (Park& e : g_parks)
    if (o && e.owner == o) return &e;
  return nullptr;
}
void park_drop(Park* e) {
  if (!e || !e->owner) return;
  unlink(e->path);
  e->owner = 0;
  e->position = 0;
  e->path[0] = 0;
}
int timeout_ms() {
  if (g_timeout_ms < 0) {
    const char* t = getenv("SUSHI_STAGE_TIMEOUT_MS");
    g_timeout_ms = t ? atoi(t) : 60000;
    if (g_timeout_ms < 100) g_timeout_ms = 60000;
  }
  return g_timeout_ms;
}
void mark_dead(const char* why) {
  if (g_dead) return;
  g_dead = true;
  for (Seq& q : g_seq) {
    q.valid = false;
    q.mirror = 0;
  }
  g_inflight = 0;
  for (Park& e : g_parks) park_drop(&e);  // a re-run from tokens replaces every parked state
  if (g_fd >= 0) {
    shutdown(g_fd, SHUT_RDWR);
    close(g_fd);
    g_fd = -1;
  }
  fprintf(stderr, "[stage] DEAD: %s; every layer runs on the Mac from now on\n", why);
  if (const char* f = getenv("SUSHI_STAGE_DEAD_FILE")) {
    if (FILE* o = fopen(f, "w")) {
      fprintf(o, "%s\n", why);
      fclose(o);
    }
  }
}
// R65 (SUSHI_R65_GPU_WARM=1; 2 = alternate per 32-round trace window, starting off): while a decode forward's
// reply is awaited (the Mac GPU idles ~6.5 ms per round), a 256-thread kernel is committed on a queue of its own
// every 0.5 ms (lib/gpu_warm.m), so the GPU does not clock down before the post-stage work.
int g_warm_mode = -1, g_warm_on = 0;
bool g_warm_read = false;
uint64_t g_warm_ticks = 0;
int warm_mode() {
  if (g_warm_mode < 0) {
    const char* e = getenv("SUSHI_R65_GPU_WARM");
    g_warm_mode = (e && (e[0] == '1' || e[0] == '2')) ? e[0] - '0' : 0;
    if (g_warm_mode && gpu_warm_init() != 0) {
      fprintf(stderr, "[stage] GPU warm-up unavailable (Metal setup failed); off\n");
      g_warm_mode = 0;
    }
    g_warm_on = g_warm_mode == 1;
  }
  return g_warm_mode;
}
bool io(void* p, size_t n, bool wr, int limit_ms = -1) {
  if (g_fd < 0) return false;
  auto* b = static_cast<uint8_t*>(p);
  const double deadline = now_us() + 1000.0 * (limit_ms > 0 ? limit_ms : timeout_ms());
  while (n) {
    const int left = int((deadline - now_us()) / 1000.0);
    if (left <= 0) return false;
    // R65: wait in 0.5 ms slices, ticking the GPU between them (select() needs fd < FD_SETSIZE; above that the
    // plain poll() wait below runs instead, without the warm-up)
    if (!wr && g_warm_read && g_fd < FD_SETSIZE) {
      bool ready = false;
      while (!ready) {
        const double rem = deadline - now_us();
        if (rem <= 0) return false;
        fd_set rf;
        FD_ZERO(&rf);
        FD_SET(g_fd, &rf);
        timeval tv{0, int(rem < 500 ? rem : 500)};
        const int r = select(g_fd + 1, &rf, nullptr, nullptr, &tv);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return false;
        if (r > 0) ready = true;
        else { gpu_warm_tick(); g_warm_ticks++; }
      }
      ssize_t k = read(g_fd, b, n);
      if (k < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (k <= 0) return false;
      b += k;
      n -= size_t(k);
      continue;
    }
    pollfd pfd{g_fd, short(wr ? POLLOUT : POLLIN), 0};
    const int r = poll(&pfd, 1, left);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    ssize_t k = wr ? write(g_fd, b, n) : read(g_fd, b, n);
    if (k < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (k <= 0) return false;
    b += k;
    n -= size_t(k);
  }
  return true;
}
// R69: a forward split in two (interleaved slots): sent now, received later. At most one forward is outstanding on
// the socket; any other call first receives it into its slot's stash (`drain_outstanding`).
struct PendingFwd {
  bool outstanding = false;  // sent, reply not read yet
  bool stashed = false;      // reply read, rows copied out of shm
  uint32_t rows = 0;
  uint64_t position = 0;
  int want_head = 0;
  mlx_dtype dtype = MLX_BFLOAT16;
  double t1 = 0;
  std::vector<uint16_t> out;
  uint32_t head[16 * 40];
};
PendingFwd g_pend[kSlots];
int g_out_slot = -1;  // the slot whose forward is outstanding
bool read_reply(uint32_t op);
// Receive the outstanding forward (if any) into its slot's stash. 0 ok, -1 the stage failed.
int drain_outstanding() {
  if (g_out_slot < 0) return 0;
  PendingFwd& p = g_pend[g_out_slot];
  const int slot = g_out_slot;
  g_out_slot = -1;
  p.outstanding = false;
  if (!read_reply(1)) return -1;
  p.out.resize(size_t(p.rows) * kF);
  memcpy(p.out.data(), g_shm, size_t(p.rows) * kF * 2);
  if (p.want_head) memcpy(p.head, g_shm + size_t(p.rows) * kF * 2, size_t(p.rows) * 40 * 4);
  p.stashed = true;
  (void)slot;
  return 0;
}
// R67 multi-stream: a callback run once between a forward's send (op 1) and its reply, while the 5090 computes:
// the scheduler decodes the Mac-only requests there. Nothing in it may talk to the stage (refused below).
typedef void (*stage_wait_hook_fn)(void);
stage_wait_hook_fn g_wait_hook = nullptr;
bool g_in_hook = false;
double g_hook_us = 0, g_last_hook_us = 0;
uint64_t g_hooks = 0;
int rpc(uint32_t op, uint32_t rows, uint64_t position, uint64_t arg, uint32_t aux = 0) {
  if (g_dead) return -1;
  if (drain_outstanding()) return -1;
  if (g_in_hook) {
    fprintf(stderr, "[stage] op %u refused: a forward's reply is outstanding\n", op);
    return -1;
  }
  if (g_inflight) {
    fprintf(stderr, "[stage] op %u with %d prefill chunks in flight\n", op, g_inflight);
    return -1;
  }
  uint8_t m[32] = {};
  aux |= uint32_t(g_cur) << 3;  // R69: the slot
  memcpy(m, &op, 4);
  memcpy(m + 8, &rows, 4);
  memcpy(m + 12, &aux, 4);
  memcpy(m + 16, &position, 8);
  memcpy(m + 24, &arg, 8);
  bool sent = io(m, 32, true);
  g_last_hook_us = 0;
  if (sent && op == 1 && g_wait_hook) {
    const stage_wait_hook_fn f = g_wait_hook;
    g_wait_hook = nullptr;  // once per registration
    g_in_hook = true;
    const int cur = g_cur;  // R69: the hook's requests must not leave another slot selected
    const double th = now_us();
    f();
    g_cur = cur;
    g_in_hook = false;
    g_last_hook_us = now_us() - th;
    g_hook_us += g_last_hook_us;
    g_hooks++;
  }
  if (!sent) {
    mark_dead("socket failure on send");
    return -1;
  }
  return read_reply(op) ? 0 : -1;
}
// The reply of the op just sent (or outstanding): status and reply arg. false: the stage failed (given up).
bool read_reply(uint32_t op) {
  uint8_t m[32] = {};
  if (!io(m, 32, false)) {
    char why[96];
    snprintf(why, sizeof why, "socket failure or timeout on op %u", op);
    mark_dead(why);
    return false;
  }
  uint32_t status;
  memcpy(&status, m + 4, 4);
  memcpy(&g_reply_us, m + 8, 4);
  memcpy(&g_reply_arg, m + 24, 8);
  if (status) {
    char why[480];
    snprintf(why, sizeof why, "op %u failed: %.400s", op, reinterpret_cast<const char*>(g_shm));
    mark_dead(why);
    return false;
  }
  return true;
}
void align16() { g_at = (g_at + 15) & ~size_t(15); }
bool g_put_overflow = false;  // a seed blob outgrew shm: the seed fails instead of writing past it
void put(const void* p, size_t n) {
  if (g_put_overflow || g_at + n > kShmBytes) {
    g_put_overflow = true;
    return;
  }
  memcpy(g_shm + g_at, p, n);
  g_at += n;
}
// A contiguous evaluated copy of `a` (optionally cast); caller frees.
int materialize(mlx_array* out, mlx_array a, int cast, mlx_dtype dt, mlx_stream s) {
  mlx_array c = mlx_array_new();
  if (cast) {
    mlx_array t = mlx_array_new();
    if (mlx_astype(&t, a, dt, s)) return -1;
    if (mlx_contiguous(&c, t, false, s)) return -1;
    mlx_array_free(t);
  } else if (mlx_contiguous(&c, a, false, s)) {
    return -1;
  }
  if (mlx_array_eval(c)) return -1;
  *out = c;
  return 0;
}
// [1, H, T, W] -> [T, H, W] (rows 0..len of axis 2), appended to the blob.
int put_token_major(mlx_array a, uint64_t len, mlx_stream s) {
  const int* sh = mlx_array_shape(a);
  if (mlx_array_ndim(a) != 4 || uint64_t(sh[2]) < len) return -1;
  mlx_array cut = mlx_array_new(), tr = mlx_array_new(), c = mlx_array_new();
  const int start[4] = {0, 0, 0, 0}, stop[4] = {sh[0], sh[1], int(len), sh[3]}, strides[4] = {1, 1, 1, 1};
  const int axes[4] = {0, 2, 1, 3};
  if (mlx_slice(&cut, a, start, 4, stop, 4, strides, 4, s)) return -1;
  if (mlx_transpose_axes(&tr, cut, axes, 4, s)) return -1;
  if (materialize(&c, tr, 0, MLX_UINT8, s)) return -1;
  const size_t bytes = mlx_array_size(c) * mlx_array_itemsize(c);
  const void* data = nullptr;
  switch (mlx_array_dtype(c)) {
    case MLX_UINT32: data = mlx_array_data_uint32(c); break;
    case MLX_BFLOAT16: data = mlx_array_data_bfloat16(c); break;
    case MLX_UINT8: data = mlx_array_data_uint8(c); break;
    default: fprintf(stderr, "[stage] KV dtype %d\n", int(mlx_array_dtype(c))); return -1;
  }
  put(data, bytes);
  align16();
  mlx_array_free(cut);
  mlx_array_free(tr);
  mlx_array_free(c);
  return 0;
}
// R66: rows [from, to) of axis 2 of [1, H, T, W] as a contiguous evaluated token-major [to-from, H, W] copy
// (caller frees), and its bytes.
int token_major_rows(mlx_array* out, mlx_array a, uint64_t from, uint64_t to, mlx_stream s) {
  const int* sh = mlx_array_shape(a);
  if (mlx_array_ndim(a) != 4 || from > to || uint64_t(sh[2]) < to) return -1;
  mlx_array cut = mlx_array_new(), tr = mlx_array_new();
  const int start[4] = {0, 0, int(from), 0}, stop[4] = {sh[0], sh[1], int(to), sh[3]}, strides[4] = {1, 1, 1, 1};
  const int axes[4] = {0, 2, 1, 3};
  int rc = mlx_slice(&cut, a, start, 4, stop, 4, strides, 4, s) || mlx_transpose_axes(&tr, cut, axes, 4, s) ||
           materialize(out, tr, 0, MLX_UINT8, s);
  mlx_array_free(cut);
  mlx_array_free(tr);
  return rc ? -1 : 0;
}
const uint8_t* array_bytes(mlx_array c, size_t* n) {
  *n = mlx_array_size(c) * mlx_array_itemsize(c);
  if (*n == 0) return nullptr;
  switch (mlx_array_dtype(c)) {
    case MLX_UINT32: return reinterpret_cast<const uint8_t*>(mlx_array_data_uint32(c));
    case MLX_BFLOAT16: return reinterpret_cast<const uint8_t*>(mlx_array_data_bfloat16(c));
    case MLX_UINT8: return mlx_array_data_uint8(c);
    default: fprintf(stderr, "[stage] KV dtype %d\n", int(mlx_array_dtype(c))); *n = 0; return nullptr;
  }
}
int put_token_major_range(mlx_array a, uint64_t from, uint64_t to, mlx_stream s) {
  mlx_array c = mlx_array_new();
  if (token_major_rows(&c, a, from, to, s)) return -1;
  size_t n = 0;
  const uint8_t* p = array_bytes(c, &n);
  if (n && !p) { mlx_array_free(c); return -1; }
  if (n) put(p, n);
  align16();
  mlx_array_free(c);
  return 0;
}
uint64_t fnv64(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}
}  // namespace

extern "C" {
// R65 GPU warm-up switch, for the trace windows: mode (0 off, 1 on, 2 alternating), the current arm, its setter
// and the ticks committed so far.
int stage_split_warm_mode(void) { return warm_mode(); }
int stage_split_warm_get(void) { return g_warm_on; }
void stage_split_warm_set(int on) { g_warm_on = on ? 1 : 0; }
unsigned long long stage_split_warm_ticks(void) { return (unsigned long long)g_warm_ticks; }
// Layer index where the stage begins, or -1 when the split is disabled (lazy open).
int stage_split_first(void) {
  if (g_dead) return -1;
  if (g_first != -2) return g_first;
  g_first = -1;
  const char* sock = getenv("SUSHI_STAGE_SOCKET");
  const char* shm = getenv("SUSHI_STAGE_SHM");
  if (!sock || !shm) return -1;
  const char* f = getenv("SUSHI_STAGE_FIRST");
  int fd = open(shm, O_RDWR);
  if (fd < 0) { fprintf(stderr, "[stage] cannot open %s\n", shm); return -1; }
  void* p = mmap(nullptr, kShmBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr, "[stage] mmap failed\n"); return -1; }
  g_shm = static_cast<uint8_t*>(p);
  g_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock, sizeof(addr.sun_path) - 1);
  // a write to a stage that died must fail, not raise SIGPIPE in the server
  int one = 1;
  setsockopt(g_fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
  if (connect(g_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    fprintf(stderr, "[stage] cannot connect %s\n", sock);
    close(g_fd);
    g_fd = -1;
    return -1;
  }
  g_first = f ? atoi(f) : 28;
  fprintf(stderr, "[stage] connected: layers %d..47 on the RTX 5090\n", g_first);
  return g_first;
}
// The stage no longer mirrors Sushi's caches (a normal full forward ran).
void stage_split_invalidate(void) { g_valid = false; }
// R67 multi-stream: a Mac-only forward of `owner` leaves the stage behind only if the stage holds that sequence; other
// streams' forwards (they run every layer on the Mac beside the stage holder) leave its state alone.
void stage_split_invalidate_owner(uint64_t owner) {
  for (Seq& q : g_seq)
    if (owner && owner == q.owner) q.valid = false;
}
int stage_split_valid_at(uint64_t position) {
  return !g_dead && g_valid && position >= g_base && position <= g_base + g_rows;
}
// 1 once the stage has been given up on (see mark_dead).
int stage_split_dead(void) { return g_dead ? 1 : 0; }
// The scheduler gives up on the stage after any split error it cannot attribute to the socket.
void stage_split_give_up(const char* why) { mark_dead(why); }
// The sequence the next stage calls serve. A different one than the stage holds re-seeds (or
// restores) first; switching with a prefill chunk in flight would mix two sequences' replies.
void stage_split_bind(uint64_t owner) {
  g_cur = slot_of(owner);
  if (owner == g_owner) return;
  if (g_inflight) {
    mark_dead("sequence switch with a prefill chunk in flight");
    return;
  }
  // R66: the mirror is about content, not the slot: a request restored into another slot still shares the rows
  // (note_prompt matches tokens, op 10 hashes the rows), so an owner switch keeps it.
  MIRROR_TRACE("[stage] mirror: owner switch %llx -> %llx, mirror %llu kept\n", (unsigned long long)g_owner, (unsigned long long)owner, (unsigned long long)g_mirror);
  g_owner = owner;
  g_valid = false;
}
// 1 when the sequence's layers [first, 48) live on the stage (it decoded or prefilled there).
int stage_split_resident(uint64_t owner) { return resident_has(owner) ? 1 : 0; }
// Park the sequence's stage state at its committed length `position` into `path` (op 7). 1 parked,
// 0 nothing to park (the stage holds no state of it), -1 the stage failed (given up).
int stage_split_park(uint64_t owner, uint64_t position, const char* path) {
  if (g_dead) return -1;
  if (drain_outstanding()) return -1;  // R69: before shm is written
  g_cur = slot_of(owner);
  if (!g_valid || g_owner != owner) return 0;
  if (g_inflight) { fprintf(stderr, "[stage] park with a prefill chunk in flight\n"); return -1; }
  if (position < g_base || position > g_base + g_rows) {
    fprintf(stderr, "[stage] park at %llu outside the last forward [%llu, %llu]\n", (unsigned long long)position,
            (unsigned long long)g_base, (unsigned long long)(g_base + g_rows));
    return -1;
  }
  const size_t n = strlen(path);
  if (n + 1 > sizeof(Park::path)) return -1;
  Park* e = park_find(owner);
  if (!e)
    for (Park& f : g_parks)
      if (!f.owner) { e = &f; break; }
  if (!e) { fprintf(stderr, "[stage] park table full\n"); return -1; }
  const double t = now_us();
  memcpy(g_shm, path, n + 1);
  if (rpc(7, 0, position, 0)) return -1;
  e->owner = owner;
  e->position = position;
  memcpy(e->path, path, n + 1);
  g_base = position;
  g_rows = 0;
  g_parked++;
  fprintf(stderr, "[stage] parked at %llu (%.1f MB, %.1f ms)\n", (unsigned long long)position, g_reply_arg / 1e6,
          (now_us() - t) / 1e3);
  return 1;
}
// Restore the sequence's parked stage state if it was parked at `position` (op 8; the file is removed
// either way). 1 restored, 0 nothing parked there (the caller seeds), -1 the stage failed.
int stage_split_unpark(uint64_t owner, uint64_t position) {
  if (drain_outstanding()) return -1;  // R69: before shm is written
  g_cur = slot_of(owner);
  Park* e = park_find(owner);
  if (!e) return 0;
  if (g_dead) { park_drop(e); return -1; }
  if (e->position != position) {
    fprintf(stderr, "[stage] parked at %llu, resumed at %llu: dropped\n", (unsigned long long)e->position,
            (unsigned long long)position);
    park_drop(e);
    return 0;
  }
  const double t = now_us();
  memcpy(g_shm, e->path, strlen(e->path) + 1);
  const int rc = rpc(8, 0, position, 0);
  park_drop(e);
  if (rc) return -1;
  g_owner = owner;
  g_valid = true;
  g_base = position;
  g_rows = 0;
  g_mirror = 0;  // R66: a parked sequence's rows are not the noted prompt's
  resident_add(owner);
  g_restored++;
  fprintf(stderr, "[stage] restored at %llu (%.1f ms)\n", (unsigned long long)position, (now_us() - t) / 1e3);
  return 1;
}
// The sequence is finished or re-runs from its tokens: drop its parked state and residency.
void stage_split_forget(uint64_t owner) {
  park_drop(park_find(owner));
  resident_del(owner);
}
// Seed: begin / per layer / end (sends op 3). conv [1,3,10240], ssm [1,48,128,128].
int stage_split_seed_begin(uint64_t position) {
  if (g_dead) return -1;
  if (drain_outstanding()) return -1;  // R69: before shm is written
  g_at = 0;
  g_put_overflow = false;
  const uint32_t hdr[4] = {kMagic, 1, uint32_t(g_first), uint32_t(48 - g_first)};
  put(hdr, 16);
  put(&position, 8);
  align16();
  return 0;
}
int stage_split_seed_recurrent(mlx_array conv, mlx_array ssm, mlx_stream s) {
  const uint32_t tag[2] = {1, 0};
  put(tag, 8);
  align16();
  mlx_array c = mlx_array_new(), st = mlx_array_new();
  if (materialize(&c, conv, 1, MLX_BFLOAT16, s) || materialize(&st, ssm, 1, MLX_FLOAT32, s)) return -1;
  if (mlx_array_size(c) != size_t(3 * kF) || mlx_array_size(st) != size_t(48 * 128 * 128)) {
    fprintf(stderr, "[stage] recurrent seed sizes %zu %zu\n", mlx_array_size(c), mlx_array_size(st));
    return -1;
  }
  put(mlx_array_data_bfloat16(c), mlx_array_size(c) * 2);
  align16();
  put(mlx_array_data_float32(st), mlx_array_size(st) * 4);
  align16();
  mlx_array_free(c);
  mlx_array_free(st);
  return 0;
}
// keys/values [1,2,T,64] uint32 codes, scales/biases [1,2,T,4] BF16 (T >= len),
// pooled [1,len/4,128] BF16, raw [1,R,128] BF16 (the last len%4 rows are the open group).
int stage_split_seed_attention(mlx_array keys, mlx_array ks, mlx_array kb, mlx_array values, mlx_array vs,
                               mlx_array vb, mlx_array pooled, mlx_array raw, uint64_t len, mlx_stream s) {
  const uint32_t tag[2] = {2, 0};
  put(tag, 8);
  align16();
  put(&len, 8);
  align16();
  for (mlx_array a : {keys, ks, kb, values, vs, vb})
    if (put_token_major(a, len, s)) { fprintf(stderr, "[stage] KV seed layout\n"); return -1; }
  const uint64_t groups = len / 4, open = len % 4;
  if (groups) {
    mlx_array p = mlx_array_new();
    if (materialize(&p, pooled, 1, MLX_BFLOAT16, s)) return -1;
    if (mlx_array_size(p) < groups * 128) { fprintf(stderr, "[stage] pooled %zu < %llu\n", mlx_array_size(p), (unsigned long long)groups * 128); return -1; }
    put(mlx_array_data_bfloat16(p), groups * 128 * 2);
    align16();
    mlx_array_free(p);
  }
  if (open) {
    mlx_array r = mlx_array_new();
    if (materialize(&r, raw, 1, MLX_BFLOAT16, s)) return -1;
    const size_t rows = mlx_array_size(r) / 128;
    if (rows < open) { fprintf(stderr, "[stage] raw keys %zu < %llu\n", rows, (unsigned long long)open); return -1; }
    put(mlx_array_data_bfloat16(r) + (rows - open) * 128, open * 128 * 2);
    align16();
    mlx_array_free(r);
  }
  return 0;
}
int stage_split_seed_end(uint64_t position) {
  if (g_put_overflow) {
    fprintf(stderr, "[stage] seed at %llu does not fit the %zu-byte shm\n", (unsigned long long)position, kShmBytes);
    return -1;
  }
  const double t = now_us();
  if (rpc(3, 0, position, g_at)) return -1;
  g_valid = true;
  g_base = position;
  g_rows = 0;
  g_seeds++;
  g_mirror = g_tok_n >= position ? position : 0;
  MIRROR_TRACE("[stage] mirror: full seed at %llu -> %llu\n", (unsigned long long)position, (unsigned long long)g_mirror);
  fprintf(stderr, "[stage] seeded at %llu (%.1f MB, %.1f ms)\n", (unsigned long long)position, g_at / 1e6, (now_us() - t) / 1e3);
  return 0;
}
// R66: a new request's prompt (`n` tokens) and its restore point. The rows the stage holds stay usable for it up
// to the common prefix, but only below `restored`: Sushi recomputes every row from there (a chunk shape of its own,
// so the stage's copies of those rows no longer hash the same).
void stage_split_note_prompt(const uint32_t* toks, uint64_t n, uint64_t restored) {
  uint64_t lim = g_mirror < g_tok_n ? g_mirror : g_tok_n;
  if (lim > n) lim = n;
  if (lim > restored) lim = restored;
  uint64_t cp = 0;
  while (cp < lim && g_tok[cp] == toks[cp]) cp++;
  MIRROR_TRACE("[stage] mirror: prompt %llu tokens, restored %llu, held %llu of %llu noted -> %llu\n", (unsigned long long)n,
               (unsigned long long)restored, (unsigned long long)g_mirror, (unsigned long long)g_tok_n, (unsigned long long)cp);
  g_mirror = cp;
  if (n > g_tok_cap) {
    void* p = realloc(g_tok, size_t(n) * sizeof(uint32_t));
    if (!p) {
      g_mirror = 0;
      g_tok_n = 0;
      return;
    }
    g_tok = static_cast<uint32_t*>(p);
    g_tok_cap = n;
  }
  if (n) memcpy(g_tok, toks, size_t(n) * sizeof(uint32_t));
  g_tok_n = n;
}
// R66: rows [0, mirror) of the noted prompt that the stage holds exactly as Sushi's caches do.
uint64_t stage_split_mirror(void) {
  MIRROR_TRACE("[stage] mirror: asked, %llu (valid %d base %llu rows %llu)\n", (unsigned long long)g_mirror, int(g_valid),
               (unsigned long long)g_base, (unsigned long long)g_rows);
  return g_dead ? 0 : g_mirror;
}
// R66 incremental seed: begin / per layer (recurrent layers: stage_split_seed_recurrent) / end (op 10).
// Rows [base, position) and every recurrent state; `check_from` <= base starts the K rows the stage hashes.
int stage_split_iseed_begin(uint64_t base, uint64_t check_from, uint64_t position) {
  if (drain_outstanding()) return -1;  // R69: before shm is written
  if (g_dead) return -1;
  g_at = 0;
  g_put_overflow = false;
  const uint32_t hdr[4] = {kMagicIncr, 1, uint32_t(g_first), uint32_t(48 - g_first)};
  put(hdr, 16);
  put(&position, 8);
  put(&base, 8);
  put(&check_from, 8);
  align16();
  return 0;
}
int stage_split_iseed_attention(mlx_array keys, mlx_array ks, mlx_array kb, mlx_array values, mlx_array vs,
                                mlx_array vb, mlx_array pooled, mlx_array raw, uint64_t base, uint64_t check_from,
                                uint64_t position, mlx_stream s) {
  if (base % 4 || check_from > base || base > position) return -1;
  const uint32_t tag[2] = {2, 0};
  put(tag, 8);
  align16();
  mlx_array h = mlx_array_new();
  if (token_major_rows(&h, keys, check_from, base, s)) { fprintf(stderr, "[stage] incremental seed hash rows\n"); return -1; }
  size_t hn = 0;
  const uint8_t* hp = array_bytes(h, &hn);
  const uint64_t hash = fnv64(hp ? hp : reinterpret_cast<const uint8_t*>(""), hp ? hn : 0);
  mlx_array_free(h);
  put(&hash, 8);
  align16();
  for (mlx_array a : {keys, ks, kb, values, vs, vb})
    if (put_token_major_range(a, base, position, s)) { fprintf(stderr, "[stage] incremental seed KV layout\n"); return -1; }
  const uint64_t b0 = base / 4, b1 = position / 4, open = position % 4;
  if (b1 > b0) {
    mlx_array p = mlx_array_new();
    if (materialize(&p, pooled, 1, MLX_BFLOAT16, s)) return -1;
    if (mlx_array_size(p) < b1 * 128) { fprintf(stderr, "[stage] pooled %zu < %llu\n", mlx_array_size(p), (unsigned long long)b1 * 128); return -1; }
    put(mlx_array_data_bfloat16(p) + b0 * 128, (b1 - b0) * 128 * 2);
    align16();
    mlx_array_free(p);
  }
  if (open) {
    mlx_array r = mlx_array_new();
    if (materialize(&r, raw, 1, MLX_BFLOAT16, s)) return -1;
    const size_t rows = mlx_array_size(r) / 128;
    if (rows < open) { fprintf(stderr, "[stage] raw keys %zu < %llu\n", rows, (unsigned long long)open); return -1; }
    put(mlx_array_data_bfloat16(r) + (rows - open) * 128, open * 128 * 2);
    align16();
    mlx_array_free(r);
  }
  return 0;
}
// R66d end-of-turn pull-back (op 9): commit the last forward's rows up to `position` (as PARK does) and export the
// owner's committed rows [from, position) as a QDX1 blob in shm (valid until the next stage call). 0 exported, 1
// nothing to export (the stage holds no state of this sequence there), -1 the stage failed (given up).
int stage_split_export(uint64_t owner, uint64_t from, uint64_t position, const uint8_t** data, uint64_t* len) {
  if (g_dead) return -1;
  g_cur = slot_of(owner);
  if (!g_valid || g_owner != owner || g_inflight || from >= position) return 1;
  if (position < g_base || position > g_base + g_rows) return 1;
  const double t = now_us();
  if (rpc(9, 0, position, from)) return -1;
  g_base = position;
  g_rows = 0;
  *data = g_shm;
  *len = g_reply_arg;
  if (g_mirror == from) g_mirror = position;  // the caller imports these rows: Sushi then holds them as the stage does
  fprintf(stderr, "[stage] pulled back [%llu, %llu) (%.1f MB, %.1f ms)\n", (unsigned long long)from,
          (unsigned long long)position, g_reply_arg / 1e6, (now_us() - t) / 1e3);
  return 0;
}
// R66d: the caller could not import an export; the rows past `n` are not Sushi's.
void stage_split_mirror_cap(uint64_t n) {
  if (g_mirror > n) g_mirror = n;
}
// 1 applied, 0 refused (the caller sends a full seed), -1 the stage failed (given up).
int stage_split_iseed_end(uint64_t position) {
  if (g_put_overflow) return 0;
  const double t = now_us();
  if (rpc(10, 0, position, g_at)) return -1;
  if (g_reply_arg != 1) {
    g_irefused++;
    g_mirror = 0;
    fprintf(stderr, "[stage] incremental seed at %llu refused (%llu so far); full seed\n", (unsigned long long)position,
            (unsigned long long)g_irefused);
    return 0;
  }
  g_valid = true;
  g_base = position;
  g_rows = 0;
  g_mirror = position;
  g_iseeds++;
  fprintf(stderr, "[stage] seeded incrementally at %llu (%.1f MB, %.1f ms; %llu incremental, %llu refused)\n",
          (unsigned long long)position, g_at / 1e6, (now_us() - t) / 1e3, (unsigned long long)g_iseeds,
          (unsigned long long)g_irefused);
  return 1;
}
// h: the HC stream [1, rows, 10240] after layer first-1 at absolute positions [position, ...).
// out: the stream after layer 47, in h's dtype.
// rope_delta: the rows' rotary offset (Qwen M-RoPE decode delta after an image; 0 for text), sent in
// aux bits 8..31 as a signed 24-bit value. The KV row index stays `position`.
int stage_split_forward(mlx_array h, mlx_array* out, uint64_t position, int32_t rope_delta, mlx_stream s) {
  if (g_dead) return -1;
  if (drain_outstanding()) return -1;  // R69: before shm is written
  if (!stage_split_valid_at(position)) return -2;  // caller re-seeds
  if (g_mirror > position) {  // R66: rows from here on are the stage's own
    MIRROR_TRACE("[stage] mirror: forward at %llu clamps %llu\n", (unsigned long long)position, (unsigned long long)g_mirror);
    g_mirror = position;
  }
  const double t0 = now_us();
  mlx_array c = mlx_array_new();
  if (materialize(&c, h, 1, MLX_BFLOAT16, s)) return -1;
  const int* sh = mlx_array_shape(c);
  const int nd = int(mlx_array_ndim(c));
  const uint32_t rows = uint32_t(mlx_array_size(c) / kF);
  if (rows < 1 || rows > 16 || sh[nd - 1] != kF) { fprintf(stderr, "[stage] forward rows %u\n", rows); return -1; }
  memcpy(g_shm, mlx_array_data_bfloat16(c), size_t(rows) * kF * 2);
  const double t1 = now_us();
  // infer the commit of the previous forward
  if (g_rows && position < g_base + g_rows) {
    if (!infer_commit() && rpc(2, 0, 0, position - g_base)) return -1;
    g_partials++;
  }
  if (rope_delta < -(1 << 23) || rope_delta >= (1 << 23)) { fprintf(stderr, "[stage] rope delta %d out of range\n", rope_delta); return -1; }
  const int want_head = g_want_head;
  g_head_rows = 0;
  g_warm_read = warm_mode() && g_warm_on;  // R65: tick the Mac GPU while this forward's reply is awaited
  const int fr = rpc(1, rows, position, 0, ((uint32_t(rope_delta) & 0xFFFFFFu) << 8) | (want_head ? 4u : 0u));
  g_warm_read = false;
  if (fr) return -1;
  const double t2 = now_us();
  g_base = position;
  g_rows = rows;
  resident_add(g_owner);
  if (want_head) {
    memcpy(g_head_buf, g_shm + size_t(rows) * kF * 2, size_t(rows) * 40 * 4);
    g_head_rows = int(rows);
  }
  int oshape[3] = {1, int(rows), kF};
  mlx_array ob = mlx_array_new_data(g_shm, oshape, 3, MLX_BFLOAT16);
  const mlx_dtype want = mlx_array_dtype(h);
  if (want != MLX_BFLOAT16) {
    if (mlx_astype(out, ob, want, s)) return -1;
    mlx_array_free(ob);
  } else {
    *out = ob;
  }
  mlx_array_free(c);
  g_forwards++;
  g_copy_us += (t1 - t0) + (now_us() - t2);
  g_rpc_us += t2 - t1 - g_last_hook_us;
  g_stage_us += double(g_reply_us);
  if (g_forwards % 1000 == 0)
    fprintf(stderr, "[stage] %llu forwards: rpc %.0f us (the stage's own time %.0f us), copies %.0f us avg; %llu partial commits, %llu seeds; %llu overlapped (%.1f ms Mac work each)\n",
            (unsigned long long)g_forwards, g_rpc_us / g_forwards, g_stage_us / g_forwards, g_copy_us / g_forwards,
            (unsigned long long)g_partials, (unsigned long long)g_seeds, (unsigned long long)g_hooks,
            g_hooks ? g_hook_us / g_hooks / 1e3 : 0.0);
  return 0;
}
void stage_split_set_head(int on) { g_want_head = on; }
// R69: the first half of stage_split_forward for interleaved slots: the rows go to the stage and the call returns;
// stage_split_forward_recv (same slot) receives them. 0 sent, -1 failed (given up), -2 the caller re-seeds.
int stage_split_forward_send(mlx_array h, uint64_t position, int32_t rope_delta, mlx_stream s) {
  if (g_dead) return -1;
  if (g_inflight) { fprintf(stderr, "[stage] forward send with prefill chunks in flight\n"); return -1; }
  if (!stage_split_valid_at(position)) return -2;
  if (g_mirror > position) g_mirror = position;
  PendingFwd& p = g_pend[g_cur];
  if (p.outstanding || p.stashed) { fprintf(stderr, "[stage] slot %d: a forward is already pending\n", g_cur); return -1; }
  if (drain_outstanding()) return -1;  // the other slot's reply, before shm is reused
  const double t0 = now_us();
  mlx_array c = mlx_array_new();
  if (materialize(&c, h, 1, MLX_BFLOAT16, s)) return -1;
  const int* sh = mlx_array_shape(c);
  const int nd = int(mlx_array_ndim(c));
  const uint32_t rows = uint32_t(mlx_array_size(c) / kF);
  if (rows < 1 || rows > 16 || sh[nd - 1] != kF) { fprintf(stderr, "[stage] forward rows %u\n", rows); mlx_array_free(c); return -1; }
  // the previous forward's commit, inferred or explicit (before this slot's shm write: op 2 is a full rpc)
  if (g_rows && position < g_base + g_rows) {
    if (!infer_commit() && rpc(2, 0, 0, position - g_base)) { mlx_array_free(c); return -1; }
    g_partials++;
  }
  memcpy(g_shm, mlx_array_data_bfloat16(c), size_t(rows) * kF * 2);
  mlx_array_free(c);
  if (rope_delta < -(1 << 23) || rope_delta >= (1 << 23)) { fprintf(stderr, "[stage] rope delta %d out of range\n", rope_delta); return -1; }
  const int want_head = g_want_head;
  g_head_rows = 0;
  uint8_t m[32] = {};
  const uint32_t op = 1;
  const uint32_t aux = ((uint32_t(rope_delta) & 0xFFFFFFu) << 8) | (want_head ? 4u : 0u) | (uint32_t(g_cur) << 3);
  memcpy(m, &op, 4);
  memcpy(m + 8, &rows, 4);
  memcpy(m + 12, &aux, 4);
  memcpy(m + 16, &position, 8);
  if (!io(m, 32, true)) { mark_dead("socket failure on forward send"); return -1; }
  p.outstanding = true;
  p.stashed = false;
  p.rows = rows;
  p.position = position;
  p.want_head = want_head;
  p.dtype = mlx_array_dtype(h);
  p.t1 = now_us();
  g_out_slot = g_cur;
  g_copy_us += p.t1 - t0;
  return 0;
}
// R69: the second half: the slot's forward output (stashed, or received now). 0 ok, -1 failed (given up).
int stage_split_forward_recv(mlx_array* out, mlx_stream s) {
  if (g_dead) return -1;
  PendingFwd& p = g_pend[g_cur];
  if (!p.outstanding && !p.stashed) { fprintf(stderr, "[stage] slot %d: no forward to receive\n", g_cur); return -1; }
  if (p.outstanding) {
    if (g_out_slot != g_cur) { fprintf(stderr, "[stage] slot %d: outstanding forward mismatch\n", g_cur); return -1; }
    if (drain_outstanding()) return -1;
  }
  const double t2 = now_us();
  g_rpc_us += t2 - p.t1;
  p.stashed = false;
  g_base = p.position;
  g_rows = p.rows;
  resident_add(g_owner);
  if (p.want_head) {
    memcpy(g_head_buf, p.head, size_t(p.rows) * 40 * 4);
    g_head_rows = int(p.rows);
  }
  int oshape[3] = {1, int(p.rows), kF};
  mlx_array ob = mlx_array_new_data(p.out.data(), oshape, 3, MLX_BFLOAT16);
  if (p.dtype != MLX_BFLOAT16) {
    if (mlx_astype(out, ob, p.dtype, s)) return -1;
    mlx_array_free(ob);
  } else {
    *out = ob;
  }
  g_forwards++;
  g_copy_us += now_us() - t2;
  return 0;
}
// R69: slot k >= 1 serves `owner`, or (slot 0) `owner` leaves whichever slot it had. A slot's client state restarts for
// a new owner (its first forward seeds); the server's slot is dropped separately (stage_split_drop_slot).
void stage_split_assign_slot(uint64_t owner, int slot) {
  for (int k = 1; k < kSlots; k++)
    if (g_slot_owner[k] == owner && k != slot) g_slot_owner[k] = 0;
  if (slot < 1 || slot >= kSlots) return;
  g_slot_owner[slot] = owner;
  Seq& q = g_seq[slot];
  if (q.owner != owner) {
    q.owner = owner;
    q.valid = false;
    q.rows = 0;
    q.mirror = 0;
    q.tok_n = 0;
  }
}
int stage_split_slot_of(uint64_t owner) { return slot_of(owner); }
// R69: the slot later calls work on (note_prompt reads the slot's noted prompt) without an owner switch.
void stage_split_select_owner(uint64_t owner) { g_cur = slot_of(owner); }
// R69: free the server's slot k >= 1 (op 11); its owner, if any, re-seeds or re-runs at its next forward.
// 0 done, -1 the stage failed (given up).
int stage_split_drop_slot(int slot) {
  if (g_dead) return -1;
  if (slot < 1 || slot >= kSlots) return 0;
  const int prev = g_cur;
  g_cur = slot;
  const int rc = rpc(11, 0, 0, 0);
  g_seq[slot].valid = false;
  g_seq[slot].rows = 0;
  g_seq[slot].owner = 0;
  g_slot_owner[slot] = 0;
  g_pend[slot].outstanding = false;
  g_pend[slot].stashed = false;
  g_cur = prev;
  return rc ? -1 : 0;
}
// R71: the server's slot k >= 1 views rows [base, base + rows) of slot 0's attention buffers from now on (op 12).
// keep = 0: the slot is dropped first (its owner, if any, re-seeds at its next forward); keep = 1: it keeps its
// conversation and only its row count changes (same base). floor0 >= 0: the rows slot 0's request holds at most; if
// the server's slot 0 holds more (a finished conversation's, which would block the view) they are forgotten first.
// 0 done, -1 the stage failed (given up).
int stage_split_slot_view(int slot, uint64_t base, uint64_t rows, int keep, int64_t floor0) {
  if (g_dead) return -1;
  if (slot < 1 || slot >= kSlots) return -1;
  const int prev = g_cur;
  g_cur = slot;
  const uint32_t aux = (keep ? 1u : 0u) | (floor0 >= 0 ? 2u : 0u);
  const int rc = rpc(12, floor0 >= 0 ? uint32_t(floor0) : 0u, base, rows, aux);
  if (!keep) {
    g_seq[slot].valid = false;
    g_seq[slot].rows = 0;
    g_seq[slot].mirror = 0;
    g_seq[slot].tok_n = 0;
    g_pend[slot].outstanding = false;
    g_pend[slot].stashed = false;
  }
  if (!rc && (g_reply_arg & 1)) {
    // slot 0's leftover rows are gone: its next owner seeds in full
    g_seq[0].valid = false;
    g_seq[0].rows = 0;
    g_seq[0].mirror = 0;
    g_seq[0].tok_n = 0;
    fprintf(stderr, "[stage] slot 0: a finished conversation's rows forgotten (slot %d takes rows from %llu)\n", slot, (unsigned long long)base);
  }
  g_cur = prev;
  return rc ? -1 : 0;
}
void stage_split_set_wait_hook(stage_wait_hook_fn f) { g_wait_hook = f; }
// The last forward's head rows (per row 20 BF16-valued f32 bit patterns then 20 token ids); -1 unless it
// carried them for exactly `rows` rows.
int stage_split_head_topk(uint32_t* dst, int rows) {
  if (rows < 1 || rows > 16 || g_head_rows != rows) return -1;
  memcpy(dst, g_head_buf, size_t(rows) * 40 * 4);
  return 0;
}
// R60 S4 (prefill v2): the empty state at position 0 (a request with no cached prefix): zero
// convolution windows and recurrent states, no KV or index rows. Layer types follow the canonical
// stack (full attention at layer % 4 == 3), which the server checks tag by tag.
int stage_split_seed_zero(void) {
  if (stage_split_seed_begin(0)) return -1;
  for (int l = g_first; l < 48; l++) {
    if (l % 4 == 3) {
      const uint32_t tag[2] = {2, 0};
      put(tag, 8);
      align16();
      const uint64_t len = 0;
      put(&len, 8);
      align16();
    } else {
      const uint32_t tag[2] = {1, 0};
      put(tag, 8);
      align16();
      if (g_at + size_t(3) * kF * 2 + size_t(48) * 128 * 128 * 4 + 32 > kShmBytes) return -1;
      memset(g_shm + g_at, 0, size_t(3) * kF * 2);
      g_at += size_t(3) * kF * 2;
      align16();
      memset(g_shm + g_at, 0, size_t(48) * 128 * 128 * 4);
      g_at += size_t(48) * 128 * 128 * 4;
      align16();
    }
  }
  return stage_split_seed_end(0);
}
// R60 S4 (prefill v2). A new request's prefill starts here: the stage holds no state of it yet,
// so its first chunk re-seeds (from Sushi's caches at the chunk's position).
void stage_split_prefill_begin(uint64_t owner) {
  stage_split_bind(owner);
  park_drop(park_find(owner));  // a new prefill of this sequence: any parked state is stale
  if (!g_inflight) g_valid = false;
}
// Whether a prefill chunk at `position` can go to the stage without a seed first.
int stage_split_prefill_ready(uint64_t position) {
  // R71: on the current slot, whichever it is (slot 0 only before); chunks in flight belong to one slot
  if (g_inflight) return g_cur == g_pf_cur && position == g_pf_next;
  return g_valid && g_rows == 0 && position == g_base;
}
int stage_split_prefill_inflight(void) { return g_inflight; }
// h: the HC stream [1, rows, 10240] after layer first-1 at positions [position, position+rows).
// Copied into the next free slot; op 6 is sent without waiting (stage_split_prefill_recv reads the
// reply, oldest first). At most two chunks in flight.
// R68: `rope_delta` = the rows' rotary offset (text rows of an M-RoPE request after an image; 0 otherwise).
int stage_split_prefill_send(mlx_array h, uint64_t position, int32_t rope_delta, mlx_stream s) {
  if (g_dead) return -1;
  if (drain_outstanding()) return -1;  // R69
  if (rope_delta < -(1 << 23) || rope_delta >= (1 << 23)) { fprintf(stderr, "[stage] prefill rope delta %d out of range\n", rope_delta); return -1; }
  if (g_inflight >= 2) { fprintf(stderr, "[stage] prefill: two chunks already in flight\n"); return -1; }
  if (!stage_split_prefill_ready(position)) { fprintf(stderr, "[stage] prefill at %llu not ready\n", (unsigned long long)position); return -1; }
  const double t0 = now_us();
  mlx_array c = mlx_array_new();
  if (materialize(&c, h, 1, MLX_BFLOAT16, s)) return -1;
  const uint64_t rows = mlx_array_size(c) / kF;
  if (rows < 1 || rows * kF * 2 > kSlotBytes) { fprintf(stderr, "[stage] prefill rows %llu\n", (unsigned long long)rows); return -1; }
  const int slot = (g_pf_head + g_inflight) % 2;
  const uint64_t off = kSlotBase + uint64_t(slot) * kSlotStride;
  if (g_pullback < 0) {
    const char* v = getenv("SUSHI_STAGE_PULLBACK");
    g_pullback = !(v && strcmp(v, "0") == 0);
  }
  const uint32_t aux = uint32_t(g_pullback) | (uint32_t(g_cur) << 3) | ((uint32_t(rope_delta) & 0xFFFFFFu) << 8);
  memcpy(g_shm + off, mlx_array_data_bfloat16(c), size_t(rows) * kF * 2);
  mlx_array_free(c);
  uint8_t m[32] = {};
  const uint32_t op = 6, r = uint32_t(rows);
  memcpy(m, &op, 4);
  memcpy(m + 8, &r, 4);
  memcpy(m + 12, &aux, 4);
  memcpy(m + 16, &position, 8);
  memcpy(m + 24, &off, 8);
  if (!io(m, 32, true)) { mark_dead("socket failure or timeout on prefill send"); return -1; }
  g_pf_rows[slot] = r;
  g_pf_pos[slot] = position;
  g_pf_cur = g_cur;
  g_inflight++;
  g_pf_next = position + rows;
  g_pf_send_us += now_us() - t0;
  return 0;
}
// The oldest in-flight chunk's stream after layer 47: [1, rows, 10240] in dtype `dt`.
int stage_split_prefill_recv(mlx_array* out, mlx_dtype dt, mlx_stream s) {
  if (g_dead || !g_inflight) return -1;
  g_cur = g_pf_cur;  // R71: the reply is for the slot that sent the chunk
  const double t0 = now_us();
  uint8_t m[32];
  if (!io(m, 32, false, 2 * timeout_ms())) { mark_dead("socket failure or timeout on prefill reply"); return -1; }
  uint32_t status;
  memcpy(&status, m + 4, 4);
  const int slot = g_pf_head;
  g_pf_head = (g_pf_head + 1) % 2;
  g_inflight--;
  if (status) {
    char why[480];
    snprintf(why, sizeof why, "prefill failed: %.400s", reinterpret_cast<const char*>(g_shm));
    mark_dead(why);
    return -1;
  }
  const uint32_t rows = g_pf_rows[slot];
  memcpy(&g_pf_export_len, m + 24, 8);
  g_pf_export_slot = slot;
  int oshape[3] = {1, int(rows), kF};
  mlx_array ob = mlx_array_new_data(g_shm + kSlotBase + uint64_t(slot) * kSlotStride, oshape, 3, MLX_BFLOAT16);
  if (dt != MLX_BFLOAT16) {
    if (mlx_astype(out, ob, dt, s)) return -1;
    mlx_array_free(ob);
  } else {
    *out = ob;
  }
  g_valid = true;
  g_base = g_pf_pos[slot] + rows;
  g_rows = 0;
  // R66: a pulled-back chunk is in Sushi's caches exactly as on the stage
  if (g_pullback == 1 && g_mirror == g_pf_pos[slot]) g_mirror = g_base;
  else MIRROR_TRACE("[stage] mirror: chunk [%llu, %llu) not mirrored (pullback %d, mirror %llu)\n", (unsigned long long)g_pf_pos[slot],
                    (unsigned long long)g_base, g_pullback, (unsigned long long)g_mirror);
  resident_add(g_owner);
  g_pf_chunks++;
  g_pf_wait_us += now_us() - t0;
  if (g_pf_chunks % 8 == 0)
    fprintf(stderr, "[stage] %llu prefill chunks: send %.1f ms, wait %.1f ms avg\n", (unsigned long long)g_pf_chunks,
            g_pf_send_us / g_pf_chunks / 1e3, g_pf_wait_us / g_pf_chunks / 1e3);
  return 0;
}
// The pull-back blob of the last received chunk (valid until the next chunk goes to its slot).
int stage_split_prefill_export(const uint8_t** data, uint64_t* len) {
  if (g_pf_export_slot < 0 || !g_pf_export_len) return -1;
  *data = g_shm + kSlotBase + uint64_t(g_pf_export_slot) * kSlotStride + kSlotBytes;
  *len = g_pf_export_len;
  return 0;
}
int stage_split_pullback_enabled(void) {
  if (g_pullback < 0) {
    const char* v = getenv("SUSHI_STAGE_PULLBACK");
    g_pullback = !(v && strcmp(v, "0") == 0);
  }
  return g_pullback;
}
}
