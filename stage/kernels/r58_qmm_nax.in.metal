// R58 prefill track, component 1: MLX's affine quantized matmul y = x . W^T (qmm_t) on the M5
// GPU's matrix units (NAX: Metal Performance Primitives tensor ops), for dense affine8
// projections at prefill row counts. Source: MLX, MIT License, Copyright (c) 2023-2024 Apple Inc.,
// commit 64ea011 as vendored in the pinned Sushi tree (lib/mlx-src). Headers are inlined by
// tools/inline_metal.py into kernels/mlx_qmm_nax.metal. Instantiated here: bf16 activations, BF16
// scales/biases, groups 64 and 128 (both occur in the canonical pack), 8 bits, 64x64x64 tiles with 2x2 simdgroups, BM 64 or 32, N aligned or not.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, bm, bn, bk, wm, wn, aligned, batched) \
  instantiate_kernel(                                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn        \
      "_alN_" #aligned "_batch_" #batched,                                                                     \
      name, type, group_size, bits, aligned, batched, bm, bk, bn, wm, wn)

instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 64, 8, 64, 64, 64, 2, 2, true, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 64, 8, 64, 64, 64, 2, 2, false, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 64, 8, 32, 64, 64, 2, 2, true, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 64, 8, 32, 64, 64, 2, 2, false, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 128, 8, 64, 64, 64, 2, 2, true, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 128, 8, 64, 64, 64, 2, 2, false, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 128, 8, 32, 64, 64, 2, 2, true, 0)
instantiate_quantized_aligned_batched(affine_qmm_t_nax, bfloat16_t, 128, 8, 32, 64, 64, 2, 2, false, 0)
