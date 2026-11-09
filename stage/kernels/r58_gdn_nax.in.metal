// R58 prefill track, component 3: MLX's chunked gated delta rule (WY form, chunks of 16) on the
// M5 GPU's matrix units (NAX), for the 36 GDN layers at prefill row counts. Source: MLX, MIT
// License, Copyright (c) 2023-2024 Apple Inc., commit 64ea011 as vendored in the pinned Sushi tree
// (lib/mlx-src). Inlined by tools/inline_metal.py into kernels/mlx_gdn_nax.metal, with one
// modification (kernels/r58_gdn_nax.patches.json): the decay g and write strength beta are read in
// their own type (FP32 here, as our gates are computed) instead of the activation type, because
// a BF16 decay near 1 moves in ~0.4% steps that compound over long prompts.
// Layouts (MLX's and ours): q, k [T][Hk][Dk]; v, y [T][Hv][Dv]; g, beta [T][Hv]; state [Hv][Dv][Dk].
// Instantiated: bf16 q/k/v/y, FP32 g/beta and state, Dk = Dv = 128, Hk = 16, Hv = 48, C = 16.
#include "mlx/backend/metal/kernels/gated_delta_update_nax.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template [[host_name("gated_delta_fused_nax_bf16_f32gates_128_128_16_48_16")]] [[kernel]]
decltype(gated_delta_fused_nax<bfloat16_t, float, 128, 128, 16, 48, 16>) gated_delta_fused_nax<bfloat16_t, float, 128, 128, 16, 48, 16>;
