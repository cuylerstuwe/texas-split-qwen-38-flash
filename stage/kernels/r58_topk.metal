// R58: the target's per-row top-K over its vocabulary logits on the GPU, appended to the forward
// (or head) command buffer, so the host reads K candidates per row instead of scanning 248K.
// Same rule as MetalModel::top_rows_by on the host: key = the logit (or its BF16 rounding with
// `bf16`), NaN -> -inf; order by key descending, ties to the lower id. Two passes: each block of
// PER_BLOCK ids keeps its own top-K (K rounds of a block-wide argmax), then one group per row
// merges the blocks' candidates the same way. The global top-K is a subset of the union of the
// blocks' top-K, so the result equals the host's exactly.
#include <metal_stdlib>
using namespace metal;
constant uint K = 20;
constant uint PER_BLOCK = 2048;
constant uint THREADS = 256;
constant uint PER_THREAD = PER_BLOCK / THREADS;
struct TkParams { uint n; uint tile; uint rows; uint bf16; uint blocks; uint vocab; uint pad0; uint pad1; };

inline float tk_key(float x, uint bf) {
    if (isnan(x)) return -INFINITY;
    return bf ? float(bfloat(x)) : x;
}
inline bool tk_better(float a, uint ia, float b, uint ib) { return a > b || (a == b && ia < ib); }

// Block-wide argmax of (v, id) over THREADS threads; every thread gets the winner.
inline void tk_block_best(float v, uint id, threadgroup float* sv, threadgroup uint* si,
                          uint lane, uint sg, thread float& bv, thread uint& bi) {
    for (ushort o = 16; o; o >>= 1) {
        float ov = simd_shuffle_xor(v, o);
        uint oi = simd_shuffle_xor(id, o);
        if (tk_better(ov, oi, v, id)) { v = ov; id = oi; }
    }
    if (lane == 0) { sv[sg] = v; si[sg] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = sv[0]; bi = si[0];
    for (uint s = 1; s < THREADS / 32; s++) {
        if (tk_better(sv[s], si[s], bv, bi)) { bv = sv[s]; bi = si[s]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void r58_topk_partial(const device float* logits [[buffer(0)]],
                             device float* cv [[buffer(1)]], device uint* ci [[buffer(2)]],
                             constant TkParams& p [[buffer(3)]],
                             uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[THREADS / 32];
    threadgroup uint si[THREADS / 32];
    uint b = g.x, r = g.y;
    float v[PER_THREAD];
    uint id[PER_THREAD];
    for (uint i = 0; i < PER_THREAD; i++) {
        uint gid = b * PER_BLOCK + i * THREADS + tid;
        v[i] = -INFINITY;
        id[i] = 0xffffffffu;
        if (gid < p.vocab) {
            // the head's logits are tiled: tile t holds ids [t*tile, +wn) as [n rows][wn]
            uint first = gid / p.tile * p.tile, wn = min(p.tile, p.vocab - first);
            v[i] = tk_key(logits[first * p.n + r * wn + (gid - first)], p.bf16);
            id[i] = gid;
        }
    }
    for (uint k = 0; k < K; k++) {
        float mv = -INFINITY; uint mi = 0xffffffffu;
        for (uint i = 0; i < PER_THREAD; i++) if (tk_better(v[i], id[i], mv, mi)) { mv = v[i]; mi = id[i]; }
        float bv; uint bi;
        tk_block_best(mv, mi, sv, si, lane, sg, bv, bi);
        for (uint i = 0; i < PER_THREAD; i++) if (id[i] == bi) { v[i] = -INFINITY; id[i] = 0xffffffffu; }
        if (tid == 0) { cv[(r * p.blocks + b) * K + k] = bv; ci[(r * p.blocks + b) * K + k] = bi; }
    }
}

kernel void r58_topk_merge(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                           device float* ov [[buffer(2)]], device uint* oi [[buffer(3)]],
                           constant TkParams& p [[buffer(4)]],
                           uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[THREADS / 32];
    threadgroup uint si[THREADS / 32];
    uint r = g, total = p.blocks * K;
    // each thread scans its strided share of the row's candidates every round, skipping the ids
    // already taken (block candidates are distinct vocabulary ids)
    uint taken[K];
    for (uint i = 0; i < K; i++) taken[i] = 0xffffffffu;
    for (uint k = 0; k < K; k++) {
        float mv = -INFINITY; uint mi = 0xffffffffu;
        for (uint j = tid; j < total; j += THREADS) {
            float x = cv[r * total + j];
            uint xi = ci[r * total + j];
            bool used = false;
            for (uint t = 0; t < k; t++) used = used || (taken[t] == xi);
            if (!used && tk_better(x, xi, mv, mi)) { mv = x; mi = xi; }
        }
        float bv; uint bi;
        tk_block_best(mv, mi, sv, si, lane, sg, bv, bi);
        taken[k] = bi;
        if (tid == 0) { ov[r * K + k] = bv; oi[r * K + k] = bi; }
    }
}
