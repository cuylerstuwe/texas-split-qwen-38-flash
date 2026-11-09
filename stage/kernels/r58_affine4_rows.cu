// R58 multi-row packed affine4 projection for the CUDA drafter's batched prefill. Derived from
// R51's affine4_mma_rows (R41 lineage, this project; mma.m16n8k16 fragment mapping per the NVIDIA
// PTX documentation): the ORIGINAL row-major packed weights (eight 4-bit codes per u32, BF16
// scale/bias per group) are decoded straight into BF16 MMA operands (one rounding of the affine
// weight, as R41's tiled BF16 arm), FP32 accumulation, FP32 + BF16 outputs. No tiled or expanded
// weight copy exists. Each warp computes 16 output channels x 16 input rows: the weight fragment
// is decoded once per k-step and feeds two n8 MMAs (rows 0-7 and 8-15 of the block's row group).
// W warps split K and reduce through shared memory. y/ybf are [rows][n], as the decode kernels.
__device__ __forceinline__ unsigned short r58_out(float x) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ unsigned r58_pair(unsigned packed, unsigned shift, unsigned scale, unsigned bias) {
  // 128+q is exactly representable in BF16; a packed subtract recovers both codes, then one
  // packed BF16 FMA forms scale*q+bias (R51's am_pair_bf).
  const unsigned v = packed >> shift;
  unsigned q = (v & 15u) | ((v & 240u) << 12) | 0x43004300u;
  asm("add.rn.bf16x2 %0, %0, %1;" : "+r"(q) : "r"(0xc300c300u));
  unsigned out;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(out) : "r"(q), "r"(scale | (scale << 16)), "r"(bias | (bias << 16)));
  return out;
}
template <unsigned W>
__device__ void r58_rows(const unsigned short* x, const unsigned* w, const unsigned short* s, const unsigned short* b,
                         float* y, unsigned short* ybf, unsigned k, unsigned n, unsigned group, unsigned rows, float* smem) {
  const unsigned t = threadIdx.x, lane = t & 31u, warp = t >> 5;
  const unsigned g = lane >> 2, c = (lane & 3u) * 2;
  const unsigned ch0 = blockIdx.x * 16, row0 = blockIdx.y * 16;
  const unsigned it = k / 16, ng = k / group, words = k / 8;
  float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  for (unsigned tk = warp; tk < it; tk += W) {
    const unsigned kk = tk * 16;
    unsigned aa[4];
#pragma unroll
    for (unsigned j = 0; j < 2; j++) {
      const unsigned r = ch0 + g + j * 8;
      const bool ok = r < n;
      const unsigned sc = ok ? s[r * ng + kk / group] : 0u, bi = ok ? b[r * ng + kk / group] : 0u;
      const unsigned p0 = ok ? w[(size_t)r * words + kk / 8] : 0u, p1 = ok ? w[(size_t)r * words + kk / 8 + 1] : 0u;
      aa[j] = r58_pair(p0, c * 4, sc, bi);
      aa[j + 2] = r58_pair(p1, c * 4, sc, bi);
    }
#pragma unroll
    for (unsigned h = 0; h < 2; h++) {
      const unsigned xr = row0 + h * 8 + g;
      const bool ok = xr < rows;
      const unsigned b0 = ok ? *reinterpret_cast<const unsigned*>(x + (size_t)xr * k + kk + c) : 0u;
      const unsigned b1 = ok ? *reinterpret_cast<const unsigned*>(x + (size_t)xr * k + kk + c + 8) : 0u;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                   : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(b0), "r"(b1));
    }
  }
  // C fragment: element j at output channel g + (j>>1)*8, input row c + (j&1) of each n8 half.
#pragma unroll
  for (unsigned h = 0; h < 2; h++)
#pragma unroll
    for (unsigned j = 0; j < 4; j++) {
      const unsigned oc = g + (j >> 1) * 8, ir = h * 8 + c + (j & 1);
      smem[warp * 256 + ir * 16 + oc] = acc[h][j];
    }
  __syncthreads();
  for (unsigned i = t; i < 256; i += W * 32) {
    const unsigned nr = row0 + i / 16, nc = ch0 + i % 16;
    if (nr < rows && nc < n) {
      float sum = 0.f;
#pragma unroll
      for (unsigned j = 0; j < W; j++) sum += smem[j * 256 + i];
      const size_t at = (size_t)nr * n + nc;
      y[at] = sum;
      ybf[at] = r58_out(sum);
    }
  }
}
#define R58_ROWS(NAME, W)                                                                            \
  extern "C" __global__ void NAME(const unsigned short* x, const unsigned* w, const unsigned short* s, \
                                  const unsigned short* b, float* y, unsigned short* ybf, unsigned k,  \
                                  unsigned n, unsigned group, unsigned rows) {                          \
    __shared__ float smem[W * 256];                                                                  \
    r58_rows<W>(x, w, s, b, y, ybf, k, n, group, rows, smem);                                        \
  }
R58_ROWS(r58_affine4_rows_w4, 4)
R58_ROWS(r58_affine4_rows_w8, 8)
