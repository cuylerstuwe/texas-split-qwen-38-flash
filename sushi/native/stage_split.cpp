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
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
// 2 GiB: the largest state seed (262160 rows: 5 attention layers x 1152 B/row + 15 recurrent states,
// ~1.56 GB) fits whole; canon-stage-server maps the same size.
constexpr size_t kShmBytes = size_t(2) << 30;
constexpr uint32_t kMagic = 0x31534451;  // "QDS1"
constexpr int kF = 10240;
int g_fd = -1;
int g_first = -2;  // -2 unopened, -1 disabled
uint8_t* g_shm = nullptr;
bool g_valid = false;   // the stage holds this sequence's state
uint64_t g_base = 0;    // first position of the last forward
uint64_t g_rows = 0;    // rows of the last forward (uncommitted)
size_t g_at = 0;        // seed blob cursor
uint64_t g_forwards = 0, g_seeds = 0, g_partials = 0;
double g_rpc_us = 0, g_copy_us = 0;
// R60 S4 (prefill v2): pipelined chunk prefill through the stage (op 6). Two shm slots after the
// forward/seed area; replies arrive in send order. While chunks are in flight no other op may run.
constexpr size_t kSlotBase = size_t(256) << 20;
constexpr size_t kSlotStride = size_t(256) << 20;  // per slot: stream area, then the pull-back blob
constexpr size_t kSlotBytes = size_t(128) << 20;   // stream area (the blob starts here)
uint64_t g_pf_export_len = 0;
int g_pf_export_slot = -1;
int g_pullback = -1;
int g_inflight = 0, g_pf_head = 0;
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
int g_timeout_ms = -1;
uint64_t g_reply_arg = 0;  // the last reply's arg word
uint64_t g_owner = 0;      // the sequence whose state the stage holds (valid when g_valid)
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
  g_valid = false;
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
bool io(void* p, size_t n, bool wr, int limit_ms = -1) {
  if (g_fd < 0) return false;
  auto* b = static_cast<uint8_t*>(p);
  const double deadline = now_us() + 1000.0 * (limit_ms > 0 ? limit_ms : timeout_ms());
  while (n) {
    const int left = int((deadline - now_us()) / 1000.0);
    if (left <= 0) return false;
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
int rpc(uint32_t op, uint32_t rows, uint64_t position, uint64_t arg, uint32_t aux = 0) {
  if (g_dead) return -1;
  if (g_inflight) {
    fprintf(stderr, "[stage] op %u with %d prefill chunks in flight\n", op, g_inflight);
    return -1;
  }
  uint8_t m[32] = {};
  memcpy(m, &op, 4);
  memcpy(m + 8, &rows, 4);
  memcpy(m + 12, &aux, 4);
  memcpy(m + 16, &position, 8);
  memcpy(m + 24, &arg, 8);
  if (!io(m, 32, true) || !io(m, 32, false)) {
    char why[96];
    snprintf(why, sizeof why, "socket failure or timeout on op %u", op);
    mark_dead(why);
    return -1;
  }
  uint32_t status;
  memcpy(&status, m + 4, 4);
  memcpy(&g_reply_arg, m + 24, 8);
  if (status) {
    char why[480];
    snprintf(why, sizeof why, "op %u failed: %.400s", op, reinterpret_cast<const char*>(g_shm));
    mark_dead(why);
    return -1;
  }
  return 0;
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
}  // namespace

extern "C" {
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
  if (owner == g_owner) return;
  if (g_inflight) {
    mark_dead("sequence switch with a prefill chunk in flight");
    return;
  }
  g_owner = owner;
  g_valid = false;
}
// 1 when the sequence's layers [first, 48) live on the stage (it decoded or prefilled there).
int stage_split_resident(uint64_t owner) { return resident_has(owner) ? 1 : 0; }
// Park the sequence's stage state at its committed length `position` into `path` (op 7). 1 parked,
// 0 nothing to park (the stage holds no state of it), -1 the stage failed (given up).
int stage_split_park(uint64_t owner, uint64_t position, const char* path) {
  if (g_dead) return -1;
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
  fprintf(stderr, "[stage] seeded at %llu (%.1f MB, %.1f ms)\n", (unsigned long long)position, g_at / 1e6, (now_us() - t) / 1e3);
  return 0;
}
// h: the HC stream [1, rows, 10240] after layer first-1 at absolute positions [position, ...).
// out: the stream after layer 47, in h's dtype.
// rope_delta: the rows' rotary offset (Qwen M-RoPE decode delta after an image; 0 for text), sent in
// aux bits 8..31 as a signed 24-bit value. The KV row index stays `position`.
int stage_split_forward(mlx_array h, mlx_array* out, uint64_t position, int32_t rope_delta, mlx_stream s) {
  if (g_dead) return -1;
  if (!stage_split_valid_at(position)) return -2;  // caller re-seeds
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
    if (rpc(2, 0, 0, position - g_base)) return -1;
    g_partials++;
  }
  if (rope_delta < -(1 << 23) || rope_delta >= (1 << 23)) { fprintf(stderr, "[stage] rope delta %d out of range\n", rope_delta); return -1; }
  if (rpc(1, rows, position, 0, (uint32_t(rope_delta) & 0xFFFFFFu) << 8)) return -1;
  const double t2 = now_us();
  g_base = position;
  g_rows = rows;
  resident_add(g_owner);
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
  g_rpc_us += t2 - t1;
  if (g_forwards % 1000 == 0)
    fprintf(stderr, "[stage] %llu forwards: rpc %.0f us, copies %.0f us avg; %llu partial commits, %llu seeds\n",
            (unsigned long long)g_forwards, g_rpc_us / g_forwards, g_copy_us / g_forwards,
            (unsigned long long)g_partials, (unsigned long long)g_seeds);
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
  if (g_inflight) return position == g_pf_next;
  return g_valid && g_rows == 0 && position == g_base;
}
int stage_split_prefill_inflight(void) { return g_inflight; }
// h: the HC stream [1, rows, 10240] after layer first-1 at positions [position, position+rows).
// Copied into the next free slot; op 6 is sent without waiting (stage_split_prefill_recv reads the
// reply, oldest first). At most two chunks in flight.
int stage_split_prefill_send(mlx_array h, uint64_t position, mlx_stream s) {
  if (g_dead) return -1;
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
  const uint32_t aux = uint32_t(g_pullback);
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
  g_inflight++;
  g_pf_next = position + rows;
  g_pf_send_us += now_us() - t0;
  return 0;
}
// The oldest in-flight chunk's stream after layer 47: [1, rows, 10240] in dtype `dt`.
int stage_split_prefill_recv(mlx_array* out, mlx_dtype dt, mlx_stream s) {
  if (g_dead || !g_inflight) return -1;
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
