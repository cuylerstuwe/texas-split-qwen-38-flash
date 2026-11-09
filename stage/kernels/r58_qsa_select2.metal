// R58: a two-stage QSA top-512 selection for long contexts (bit-identical to qsa_select11).
// Stage 1: several threadgroups per row, each runs qsa_select11's radix algorithm on a contiguous
// chunk of the row's valid blocks and writes that chunk's top-512 global ids in ascending order
// (SENTINEL padded). Stage 2: one threadgroup per row runs the same algorithm over the chunks'
// candidates, which are in ascending global id order, and writes the final ids. Every element of
// the global top-512 (score descending, id ascending on ties) is in its chunk's top-512, so the
// threshold, the tie choice (lowest ids) and the ascending output order are qsa_select11's.
// The algorithm text below is qsa_select11's, under different load/bounds macros.
struct Sel2Params { uint chunk; uint chunks; uint pad0; uint pad1; };
kernel void qsa_select11_chunk(const device float* scores [[buffer(0)]],
    device int* cand [[buffer(1)]], constant QsaParams& p [[buffer(2)]], constant Sel2Params& q [[buffer(3)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TGS=256, DIGITS=11, K=512;
constexpr uint TGN   = (uint)TGS;
constexpr uint SIMDW = 32u;
constexpr uint NSIMD = TGN / SIMDW;
constexpr uint BINS  = (DIGITS == 8) ? 256u : 2048u;
constexpr uint NLEV  = (DIGITS == 8) ? 4u : 3u;
constexpr uint KTOP  = (uint)K;
constexpr int  SENTINEL = 2147483647;
static_assert(NSIMD <= SIMDW);

threadgroup metal::atomic_uint hist[BINS];
threadgroup uint sgs[2u * NSIMD];
threadgroup uint sh[4];

const uint row  = threadgroup_position_in_grid.y;
const uint cb   = threadgroup_position_in_grid.x;
const uint tid  = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg   = simdgroup_index_in_threadgroup;

const uint nb = (p.offset+p.rows)/4;
const int  vbi = int((p.offset+row+1)/4);
const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
const device float* sc = scores + (ulong)row * (ulong)nb;
device int* outp = cand + ((ulong)row * (ulong)q.chunks + (ulong)cb) * (ulong)KTOP;
const uint lo = metal::min(cb * q.chunk, vb), hi = metal::min(lo + q.chunk, vb);
if (hi - lo <= KTOP) {
  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (lo + i < hi) ? int(lo + i) : SENTINEL;
  return;
}
#define SEL_LO lo
#define SEL_HI hi
#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = int(i); u = sushi_qsa_ord(sc[i]); ok = 1u; } }
uint pref = 0u;
uint fixed = 0u;
uint need = KTOP;
uint T = 0u;
uint need_eq = 0u;

for (uint lv = 0u; lv < NLEV; ++lv) {
  const uint width = (DIGITS == 8) ? 8u : ((lv == 2u) ? 10u : 11u);
  const uint nbins = 1u << width;
  const uint shift = 32u - fixed - width;
  const uint hi_b  = (fixed == 0u) ? 31u : (32u - fixed);

  for (uint b = tid; b < nbins; b += TGN) metal::atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
  if (tid == 0u) { sh[0] = 0u; sh[1] = 0u; sh[2] = 0u; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint last_d = 0xFFFFFFFFu;
  uint run = 0u;
  for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
    const uint i = base + tid;
    uint u = 0u;
    int out_idx = SENTINEL;
    uint ok = 0u;
    SEL_LOAD(i, u, out_idx, ok)
    if (ok != 0u) {
      if (!(fixed != 0u && (u >> hi_b) != pref)) {
        const uint d = (u >> shift) & (nbins - 1u);
        if (d == last_d) { run += 1u; }
        else {
          if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
          last_d = d;
          run = 1u;
        }
      }
    }
  }
  if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (sg == 0u) {
    const uint chunk = nbins / SIMDW;
    const uint base  = nbins - (lane + 1u) * chunk;
    uint tot = 0u;
    for (uint j = 0u; j < chunk; ++j) tot += metal::atomic_load_explicit(&hist[base + j], metal::memory_order_relaxed);
    const uint pre = metal::simd_prefix_exclusive_sum(tot);
    if (pre < need && need <= pre + tot) {
      uint acc = pre;
      for (uint jj = chunk; jj > 0u; --jj) {
        const uint bidx = base + jj - 1u;
        const uint c = metal::atomic_load_explicit(&hist[bidx], metal::memory_order_relaxed);
        if (acc + c >= need) { sh[0] = bidx; sh[1] = acc; sh[2] = c; break; }
        acc += c;
      }
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint d_sel   = sh[0];
  const uint above_w = sh[1];
  const uint cnt_d   = sh[2];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  need = need - above_w;
  pref = (pref << width) | d_sel;
  fixed += width;
  if (cnt_d == need || fixed >= 32u) {
    T = (fixed >= 32u) ? pref : (pref << (32u - fixed));
    need_eq = need;
    break;
  }
}

for (uint base = 0u; base < KTOP; base += TGN) {
  const uint i = base + tid;
  if (i < KTOP) outp[i] = SENTINEL;
}
threadgroup_barrier(metal::mem_flags::mem_device);

uint run_gt = 0u;
uint run_eq = 0u;
for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
  const uint i = base + tid;
  uint gtf = 0u;
  uint e = 0u;
  int out_idx = SENTINEL;
  uint u = 0u;
  uint ok = 0u;
  SEL_LOAD(i, u, out_idx, ok)
  if (ok != 0u) {
    gtf = (u > T) ? 1u : 0u;
    e = (u == T) ? 1u : 0u;
  }
  const uint pg = metal::simd_prefix_exclusive_sum(gtf);
  const uint pe = metal::simd_prefix_exclusive_sum(e);
  const uint sg_g = metal::simd_sum(gtf);
  const uint sg_e = metal::simd_sum(e);
  if (lane == 0u) { sgs[sg] = sg_g; sgs[NSIMD + sg] = sg_e; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint off_g = 0u, off_e = 0u, tot_g = 0u, tot_e = 0u;
  {
    const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    const uint b = (lane < NSIMD) ? sgs[NSIMD + lane] : 0u;
    const uint pa = metal::simd_prefix_exclusive_sum(a);
    const uint pb = metal::simd_prefix_exclusive_sum(b);
    off_g = metal::simd_shuffle(pa, sg);
    off_e = metal::simd_shuffle(pb, sg);
    tot_g = metal::simd_sum(a);
    tot_e = metal::simd_sum(b);
  }
  const uint gb = run_gt + off_g + pg;
  const uint eb = run_eq + off_e + pe;
  if (ok != 0u) {
    if (gtf != 0u) {
      const uint pos = gb + metal::min(eb, need_eq);
      if (pos < KTOP) outp[pos] = out_idx;
    } else if (e != 0u && eb < need_eq) {
      const uint pos = gb + eb;
      if (pos < KTOP) outp[pos] = out_idx;
    }
  }
  run_gt += tot_g;
  run_eq += tot_e;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (run_gt + metal::min(run_eq, need_eq) >= KTOP) break;
}

#undef SEL_LO
#undef SEL_HI
#undef SEL_LOAD
}

kernel void qsa_select11_merge(const device float* scores [[buffer(0)]],
    device int* ids [[buffer(1)]], constant QsaParams& p [[buffer(2)]], constant Sel2Params& q [[buffer(3)]],
    const device int* cand [[buffer(4)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TGS=256, DIGITS=11, K=512;
constexpr uint TGN   = (uint)TGS;
constexpr uint SIMDW = 32u;
constexpr uint NSIMD = TGN / SIMDW;
constexpr uint BINS  = (DIGITS == 8) ? 256u : 2048u;
constexpr uint NLEV  = (DIGITS == 8) ? 4u : 3u;
constexpr uint KTOP  = (uint)K;
constexpr int  SENTINEL = 2147483647;
static_assert(NSIMD <= SIMDW);

threadgroup metal::atomic_uint hist[BINS];
threadgroup uint sgs[2u * NSIMD];
threadgroup uint sh[4];

const uint row  = threadgroup_position_in_grid.y;
const uint tid  = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg   = simdgroup_index_in_threadgroup;

const uint nb = (p.offset+p.rows)/4;
const int  vbi = int((p.offset+row+1)/4);
const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
const device float* sc = scores + (ulong)row * (ulong)nb;
const device int* cr = cand + (ulong)row * (ulong)q.chunks * (ulong)KTOP;
device int* outp = ids + (ulong)row * (ulong)KTOP;
if (vb <= KTOP) {
  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (i < vb) ? int(i) : SENTINEL;
  return;
}
#define SEL_LO 0u
#define SEL_HI (q.chunks * KTOP)
#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { const int c_ = cr[i]; if (c_ != SENTINEL) { out_idx = c_; u = sushi_qsa_ord(sc[c_]); ok = 1u; } } }
uint pref = 0u;
uint fixed = 0u;
uint need = KTOP;
uint T = 0u;
uint need_eq = 0u;

for (uint lv = 0u; lv < NLEV; ++lv) {
  const uint width = (DIGITS == 8) ? 8u : ((lv == 2u) ? 10u : 11u);
  const uint nbins = 1u << width;
  const uint shift = 32u - fixed - width;
  const uint hi_b  = (fixed == 0u) ? 31u : (32u - fixed);

  for (uint b = tid; b < nbins; b += TGN) metal::atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
  if (tid == 0u) { sh[0] = 0u; sh[1] = 0u; sh[2] = 0u; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint last_d = 0xFFFFFFFFu;
  uint run = 0u;
  for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
    const uint i = base + tid;
    uint u = 0u;
    int out_idx = SENTINEL;
    uint ok = 0u;
    SEL_LOAD(i, u, out_idx, ok)
    if (ok != 0u) {
      if (!(fixed != 0u && (u >> hi_b) != pref)) {
        const uint d = (u >> shift) & (nbins - 1u);
        if (d == last_d) { run += 1u; }
        else {
          if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
          last_d = d;
          run = 1u;
        }
      }
    }
  }
  if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (sg == 0u) {
    const uint chunk = nbins / SIMDW;
    const uint base  = nbins - (lane + 1u) * chunk;
    uint tot = 0u;
    for (uint j = 0u; j < chunk; ++j) tot += metal::atomic_load_explicit(&hist[base + j], metal::memory_order_relaxed);
    const uint pre = metal::simd_prefix_exclusive_sum(tot);
    if (pre < need && need <= pre + tot) {
      uint acc = pre;
      for (uint jj = chunk; jj > 0u; --jj) {
        const uint bidx = base + jj - 1u;
        const uint c = metal::atomic_load_explicit(&hist[bidx], metal::memory_order_relaxed);
        if (acc + c >= need) { sh[0] = bidx; sh[1] = acc; sh[2] = c; break; }
        acc += c;
      }
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint d_sel   = sh[0];
  const uint above_w = sh[1];
  const uint cnt_d   = sh[2];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  need = need - above_w;
  pref = (pref << width) | d_sel;
  fixed += width;
  if (cnt_d == need || fixed >= 32u) {
    T = (fixed >= 32u) ? pref : (pref << (32u - fixed));
    need_eq = need;
    break;
  }
}

for (uint base = 0u; base < KTOP; base += TGN) {
  const uint i = base + tid;
  if (i < KTOP) outp[i] = SENTINEL;
}
threadgroup_barrier(metal::mem_flags::mem_device);

uint run_gt = 0u;
uint run_eq = 0u;
for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
  const uint i = base + tid;
  uint gtf = 0u;
  uint e = 0u;
  int out_idx = SENTINEL;
  uint u = 0u;
  uint ok = 0u;
  SEL_LOAD(i, u, out_idx, ok)
  if (ok != 0u) {
    gtf = (u > T) ? 1u : 0u;
    e = (u == T) ? 1u : 0u;
  }
  const uint pg = metal::simd_prefix_exclusive_sum(gtf);
  const uint pe = metal::simd_prefix_exclusive_sum(e);
  const uint sg_g = metal::simd_sum(gtf);
  const uint sg_e = metal::simd_sum(e);
  if (lane == 0u) { sgs[sg] = sg_g; sgs[NSIMD + sg] = sg_e; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint off_g = 0u, off_e = 0u, tot_g = 0u, tot_e = 0u;
  {
    const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    const uint b = (lane < NSIMD) ? sgs[NSIMD + lane] : 0u;
    const uint pa = metal::simd_prefix_exclusive_sum(a);
    const uint pb = metal::simd_prefix_exclusive_sum(b);
    off_g = metal::simd_shuffle(pa, sg);
    off_e = metal::simd_shuffle(pb, sg);
    tot_g = metal::simd_sum(a);
    tot_e = metal::simd_sum(b);
  }
  const uint gb = run_gt + off_g + pg;
  const uint eb = run_eq + off_e + pe;
  if (ok != 0u) {
    if (gtf != 0u) {
      const uint pos = gb + metal::min(eb, need_eq);
      if (pos < KTOP) outp[pos] = out_idx;
    } else if (e != 0u && eb < need_eq) {
      const uint pos = gb + eb;
      if (pos < KTOP) outp[pos] = out_idx;
    }
  }
  run_gt += tot_g;
  run_eq += tot_e;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (run_gt + metal::min(run_eq, need_eq) >= KTOP) break;
}

#undef SEL_LO
#undef SEL_HI
#undef SEL_LOAD
}
