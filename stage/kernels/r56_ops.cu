// R56 drafter step helpers (new; no third-party source).
// r56_topk_partial/r56_topk_final: top-20 BF16 logits per row, larger value first and, for equal
// values, the lower token id first (the host sampler's order); NaN never selected. Two launches:
// ceil(n / 2048) blocks scan disjoint chunks into a 20-candidate list each, one block per row
// merges them (up to 4096 candidates).
// r56_copy_table: device-to-device copies listed in a table {dst, src, bytes}, one launch.
typedef unsigned int uint;
typedef unsigned short bf;
typedef unsigned long long u64;
#define K 20
#define THREADS 256
#define NEG_INF __int_as_float(0xff800000)
__device__ __forceinline__ bool better(float a, uint ia, float b, uint ib) {
  return a > b || (a == b && ia < ib);
}
// Block-wide argmax over one (value, id) per thread; returns the winner to every thread.
__device__ void block_best(float v, uint id, float* sv, uint* si, float& bv, uint& bi) {
  uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (uint o = 16; o; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o);
    uint oi = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(ov, oi, v, id)) { v = ov; id = oi; }
  }
  if (lane == 0) { sv[warp] = v; si[warp] = id; }
  __syncthreads();
  if (warp == 0) {
    v = lane < THREADS / 32 ? sv[lane] : NEG_INF;
    id = lane < THREADS / 32 ? si[lane] : 0xffffffffu;
    for (uint o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, v, o);
      uint oi = __shfl_xor_sync(0xffffffffu, id, o);
      if (better(ov, oi, v, id)) { v = ov; id = oi; }
    }
    if (lane == 0) { sv[0] = v; si[0] = id; }
  }
  __syncthreads();
  bv = sv[0]; bi = si[0];
  __syncthreads();
}
#define PER 8   // values per thread in the partial pass (chunk = THREADS * PER)
extern "C" __global__ void __launch_bounds__(THREADS) r56_topk_partial(const bf* logits,
    uint n, uint row_stride, float* cand_v, uint* cand_i) {
  __shared__ float sv[THREADS / 32]; __shared__ uint si[THREADS / 32];
  uint row = blockIdx.y, base = blockIdx.x * THREADS * PER;
  float v[PER]; uint id[PER];
  for (uint j = 0; j < PER; j++) {
    uint t = base + j * THREADS + threadIdx.x;
    float x = t < n ? __uint_as_float((uint)logits[(u64)row * row_stride + t] << 16) : NEG_INF;
    v[j] = isnan(x) ? NEG_INF : x; id[j] = t < n ? t : 0xffffffffu;
  }
  for (uint r = 0; r < K; r++) {
    float bv = NEG_INF; uint bi = 0xffffffffu;
    for (uint j = 0; j < PER; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; }
    float wv; uint wi;
    block_best(bv, bi, sv, si, wv, wi);
    for (uint j = 0; j < PER; j++) if (id[j] == wi) v[j] = NEG_INF, id[j] = 0xffffffffu;
    if (threadIdx.x == 0) {
      u64 at = ((u64)row * gridDim.x + blockIdx.x) * K + r;
      cand_v[at] = wv; cand_i[at] = wi;
    }
  }
}
// out row r: K value bit patterns (float) then K token ids, best first.
extern "C" __global__ void __launch_bounds__(THREADS) r56_topk_final(const float* cand_v, const uint* cand_i,
    uint candidates, uint* out) {
  __shared__ float sv[THREADS / 32]; __shared__ uint si[THREADS / 32];
  uint row = blockIdx.y;
  const uint P = 16;   // up to THREADS * 16 = 4096 candidates (TOPK_BLOCKS * 20)
  float v[P]; uint id[P];
  for (uint j = 0; j < P; j++) {
    uint c = j * THREADS + threadIdx.x;
    bool ok = c < candidates;
    v[j] = ok ? cand_v[(u64)row * candidates + c] : NEG_INF;
    id[j] = ok ? cand_i[(u64)row * candidates + c] : 0xffffffffu;
  }
  for (uint r = 0; r < K; r++) {
    float bv = NEG_INF; uint bi = 0xffffffffu;
    for (uint j = 0; j < P; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; }
    float wv; uint wi;
    block_best(bv, bi, sv, si, wv, wi);
    for (uint j = 0; j < P; j++) if (id[j] == wi) v[j] = NEG_INF, id[j] = 0xffffffffu;
    if (threadIdx.x == 0) { out[row * 2 * K + r] = __float_as_uint(wv); out[row * 2 * K + K + r] = wi; }
  }
}
struct CopyEntry { u64 dst, src, bytes; };
// grid (blocks_per_entry, entries); 16-byte units when every address and size allows it.
extern "C" __global__ void __launch_bounds__(THREADS) r56_copy_table(const CopyEntry* table) {
  CopyEntry e = table[blockIdx.y];
  unsigned char* dst = (unsigned char*)e.dst; const unsigned char* src = (const unsigned char*)e.src;
  u64 stride = (u64)gridDim.x * THREADS, start = (u64)blockIdx.x * THREADS + threadIdx.x;
  if (((e.dst | e.src | e.bytes) & 15) == 0) {
    u64 units = e.bytes / 16;
    for (u64 u = start; u < units; u += stride) ((uint4*)dst)[u] = ((const uint4*)src)[u];
  } else {
    for (u64 u = start; u < e.bytes; u += stride) dst[u] = src[u];
  }
}
