// R59 barrier-cost screen (no model work). A chain of N dependent phases over a ping-pong pair:
// each phase's threadgroup g reads its neighbour's slice (written by the previous phase) and
// writes its own. Two ways to separate phases:
//   - r59_phase: one dispatch per phase, separated by encoder barriers (today's model layout);
//   - r59_persistent: one dispatch, phases separated by an in-kernel grid barrier (atomic arrive +
//     bounded spin). A spin that runs out of budget counts a timeout and continues: never a hang.
#include <metal_stdlib>
using namespace metal;

constant uint SLICE = 256;

static inline float phase_work(float v, uint p, uint work) {
    float acc = v;
    for (uint k = 0; k < work; k++) {
        acc = fma(acc, 0.999f, float(p & 7) * 0.001f);
    }
    return acc;
}

kernel void r59_phase(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint* prm [[buffer(2)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint ntg [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    const uint p = prm[0], work = prm[1];
    const uint nb = (tg + 1) % ntg;
    dst[tg * SLICE + tid] = phase_work(src[nb * SLICE + tid], p, work);
}

kernel void r59_persistent(device float* a [[buffer(0)]],
                           device float* b [[buffer(1)]],
                           constant uint* prm [[buffer(2)]],
                           device atomic_uint* bar [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint ntg [[threadgroups_per_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    const uint phases = prm[0], work = prm[1], limit = prm[2];
    const uint nb = (tg + 1) % ntg;
    for (uint p = 0; p < phases; p++) {
        device float* src = (p & 1) ? b : a;
        device float* dst = (p & 1) ? a : b;
        dst[tg * SLICE + tid] = phase_work(src[nb * SLICE + tid], p, work);
        // grid barrier: generation counter in bar[1], arrivals in bar[0], timeouts in bar[2]
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            const uint gen = atomic_load_explicit(&bar[1], memory_order_relaxed);
            const uint arrived = atomic_fetch_add_explicit(&bar[0], 1u, memory_order_relaxed) + 1u;
            if (arrived == ntg) {
                atomic_store_explicit(&bar[0], 0u, memory_order_relaxed);
                atomic_fetch_add_explicit(&bar[1], 1u, memory_order_relaxed);
            } else if (atomic_load_explicit(&bar[3], memory_order_relaxed) == 0u) {
                // bounded spin; after the first timeout (bar[3] set) no barrier waits again, so a
                // grid that is not co-resident costs one timeout, not one per phase
                uint spins = 0;
                while (atomic_load_explicit(&bar[1], memory_order_relaxed) == gen && spins < limit) {
                    spins++;
                    if ((spins & 1023u) == 0u && atomic_load_explicit(&bar[3], memory_order_relaxed) != 0u) {
                        break;
                    }
                }
                if (spins >= limit) {
                    atomic_fetch_add_explicit(&bar[2], 1u, memory_order_relaxed);
                    atomic_store_explicit(&bar[3], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
