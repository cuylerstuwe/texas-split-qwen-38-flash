// R58 prefill helpers (new code).
#include <metal_stdlib>
using namespace metal;
// gdn_prepare writes the gates interleaved [rows][HV][2] = (decay, beta); MLX's chunked gated
// delta reads them as two arrays [rows][HV].
kernel void r58_split_gates(const device float* gates [[buffer(0)]], device float* g [[buffer(1)]],
                            device float* beta [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    g[i] = gates[2 * i];
    beta[i] = gates[2 * i + 1];
}
// Word copy (the new convolution history into the committed one, after gdn_prepare read it).
kernel void r58_copy_words(const device uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

// Router logits for prefill: y[t][512] (FP32) = x[t][2560] . W^T with BF16 operands on the M5
// matrix units, FP32 accumulation (not relaxed; only the summation order differs from the
// decode path's FP32 FMA kernel). Each SIMD group: 32 rows x 32 experts; four per threadgroup
// cover 128 experts. Fragment origins follow the decode path's affine8_fragment kernel.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline uint2 r58_frag_origin(uint lane) {
    uint qid = lane >> 2;
    return uint2(((qid & 2u) | (lane & 1u)) * 4u, (qid & 4u) | ((lane >> 1) & 3u));
}
template<typename L> inline void r58_left(thread L& left, const device bfloat* x, uint row, uint k0, uint2 o, uint count) {
    using b4 = vec<bfloat, 4>;
    constexpr uint K = 2560;
    ulong at = ulong(row + o.y) * K + k0 + o.x;
    b4 a = o.y < count ? *(const device b4*)(x + at) : b4(0);
    b4 b = o.y + 8u < count ? *(const device b4*)(x + at + 8ul * K) : b4(0);
    for (uint i = 0; i < 4; i++) { left[i] = a[i]; left[4 + i] = b[i]; }
}
template<typename D> inline void r58_store(const thread D& dst, device float* y, uint row, uint base, uint2 o, uint count) {
    for (uint tile = 0; tile < 2; tile++) for (uint c = 0; c < 4; c++) {
        uint col = base + tile * 16u + o.x + c;
        for (uint high = 0; high < 2; high++) if (o.y + 8u * high < count)
            y[ulong(row + o.y + 8u * high) * 512u + col] = dst[tile * 8u + high * 4u + c];
    }
}
kernel void r58_router_mma(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& rows [[buffer(3)]],
                           uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    using b4 = vec<bfloat, 4>;
    constexpr uint K = 2560;
    uint lane = tid & 31u, sg = tid >> 5, base = grid.x * 128u + sg * 32u, row = grid.y * 32u;
    if (row >= rows) return;
    uint count = min(32u, rows - row), hi = count > 16u ? count - 16u : 0u;
    uint2 o = r58_frag_origin(lane);
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto left = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto right = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto dst = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    auto dst_hi = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    for (uint i = 0; i < dst.get_capacity(); i++) { dst[i] = 0.f; dst_hi[i] = 0.f; }
    for (uint k0 = 0; k0 < K; k0 += 16u) {
        for (uint nr = 0; nr < 4; nr++) {
            b4 v = *(const device b4*)(w + ulong(base + o.y + nr * 8u) * K + k0 + o.x);
            for (uint i = 0; i < 4; i++) right[nr * 4u + i] = v[i];
        }
        r58_left(left, x, row, k0, o, count);
        op.run(left, right, dst);
        if (hi) { r58_left(left, x, row + 16u, k0, o, hi); op.run(left, right, dst_hi); }
    }
    r58_store(dst, y, row, base, o, count);
    if (hi) r58_store(dst_hi, y, row + 16u, base, o, hi);
}
