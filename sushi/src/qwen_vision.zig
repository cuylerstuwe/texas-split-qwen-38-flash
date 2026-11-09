//! Qwen3.5/3.6 (Qwen3-VL) vision tower + preprocessing math for sushi.
//!
//! Mirrors mlx-vlm `qwen3_vl/vision.py` (the ViT) and `qwen3_vl/processing_qwen3_vl.py`
//! (`_smart_resize_image`). This file currently holds the pure preprocessing math
//! (`smartResizeImage`); the MLX ViT encoder (`QwenVision`) lands alongside it and
//! is dispatched from `vision.VisionEncoder` when `config.qwen_vision` is set.
//!
//! Image tokens per image = `(grid_h/merge) * (grid_w/merge)` where
//! `grid_h = resized_H/patch`, `grid_w = resized_W/patch` — a DYNAMIC count, unlike
//! Gemma 4's fixed 280.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const log = @import("log.zig");
const common = @import("vision_common.zig");

// ─────────────────────────────────────────────────────────────────────────────
/// Transient GPU bytes one image's encode needs on top of the resident tower, at
/// `n_patches` patches. The stream is evaluated per block, so the peak is one
/// block: its f32 score sheet (heads x N^2) and its row activations. Covers the
/// measured peak (`qwen vision ubench`) by >= 25% from 196 to 9216 patches.
pub fn encodeScratchBytes(config: *const ModelConfig, n_patches: u64) u64 {
    const heads: u64 = @max(config.qv_heads, 1);
    const scores = 4 * heads * n_patches * n_patches;
    const per_patch: u64 = 4 * (12 * @as(u64, config.qv_hidden) + 3 * @as(u64, config.qv_intermediate));
    return (32 << 20) + scores * 13 / 10 + n_patches * per_patch;
}

// Qwen3-VL ViT encoder. Mirrors mlx-vlm qwen3_vl/vision.py for a SINGLE still
// image (grid_thw = [[1, grid_h, grid_w]]): full bidirectional attention over all
// patches (cu_seqlens trivial), no windowing, no DeepStack. Dense bf16 — even on
// 4-bit checkpoints the vision tower ships bf16, so no quantized-linear path.
//
// forward(patches [N, C*tps*ps*ps], grid_h, grid_w) → [1, N/merge², out_hidden].
// `patches` is the processor's `pixel_values`, already in merge-block token order
// with feature layout [C, tps, py, px] (see _process_one). The text trunk then
// splices these embeddings at the image-pad token positions.
// ─────────────────────────────────────────────────────────────────────────────

const QBlock = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fc1_w: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
};

pub const QwenVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    /// Evaluate the stream after every block and the output at the end, so one
    /// block's buffers are the encode's peak (`encodeScratchBytes`).
    eval_per_block: bool = true,

    depth: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    merge: u32,
    num_grid_per_side: u32, // = sqrt(num_position_embeddings), e.g. 48
    out_hidden: u32,
    rope_theta: f64 = 10000.0, // VisionRotaryEmbedding default

    patch_w: mlx.mlx_array, // [hidden, C*tps*ps*ps] (transposed from conv layout)
    patch_b: mlx.mlx_array, // [hidden]
    pos_embed: mlx.mlx_array, // [num_position_embeddings, hidden]
    blocks: []QBlock,
    merger_norm_w: mlx.mlx_array,
    merger_norm_b: mlx.mlx_array,
    merger_fc1_w: mlx.mlx_array,
    merger_fc1_b: mlx.mlx_array,
    merger_fc2_w: mlx.mlx_array,
    merger_fc2_b: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !QwenVision {
        const s = mlx.mlx_default_gpu_stream_new();
        var buf: [256]u8 = undefined;
        var kbuf: [256]u8 = undefined;

        const prefix = resolveVisionPrefix(weights) orelse {
            log.warn("MISSING QWEN VISION WEIGHT: {s}patch_embed.proj.weight\n", .{VISION_PREFIXES[0]});
            return error.MissingVisionWeights;
        };
        log.info("[vision] qwen tower prefix: {s}\n", .{prefix});

        const must = struct {
            fn f(w: *const Weights, b: *[256]u8, name: []const u8) !mlx.mlx_array {
                return getWeightLocal(w, b, name) orelse {
                    log.warn("MISSING QWEN VISION WEIGHT: {s}\n", .{name});
                    return error.MissingVisionWeights;
                };
            }
        }.f;

        // patch_embed.proj.weight is a Conv3d whose STORED axis order is the
        // converter's choice. Either way it ends flattened to
        // [out, Cin*kT*ps*ps], matching the processor's [C, tps, py, px]
        // pixel_values feature order — a full-window Conv3d as a plain Linear.
        const conv_w = try must(weights, &buf, fmtKey(&kbuf, prefix, "patch_embed.proj.weight"));
        const cw_shape = mlx.getShape(conv_w);
        std.debug.assert(cw_shape.len == 5);
        const out_c = cw_shape[0];
        const flat_in = cw_shape[1] * cw_shape[2] * cw_shape[3] * cw_shape[4];
        const layout = patchProjLayout(cw_shape, config.qv_temporal_patch, config.qv_patch) orelse {
            log.warn("QWEN VISION: unreadable patch_embed.proj.weight shape (tps={d}, ps={d})\n", .{ config.qv_temporal_patch, config.qv_patch });
            return error.MissingVisionWeights;
        };
        log.info("[vision] qwen patch_embed layout: {s}\n", .{@tagName(layout)});
        var patch_w = mlx.mlx_array_new();
        {
            const flat_shape = [_]c_int{ out_c, flat_in };
            switch (layout) {
                .channels_last => {
                    const perm = [_]c_int{ 0, 4, 1, 2, 3 }; // [out,kT,ps,ps,C] → [out,C,kT,ps,ps]
                    var transposed = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(transposed);
                    try mlx.check(mlx.mlx_transpose_axes(&transposed, conv_w, &perm, 5, s));
                    try mlx.check(mlx.mlx_reshape(&patch_w, transposed, &flat_shape, 2, s));
                },
                // Already [out, C, kT, ps, ps] — flatten as-is.
                .channels_first => try mlx.check(mlx.mlx_reshape(&patch_w, conv_w, &flat_shape, 2, s)),
            }
        }
        const patch_b = try must(weights, &buf, fmtKey(&kbuf, prefix, "patch_embed.proj.bias"));
        const pos_embed = try must(weights, &buf, fmtKey(&kbuf, prefix, "pos_embed.weight"));

        const depth = config.qv_depth;
        var blocks = try allocator.alloc(QBlock, depth);
        errdefer allocator.free(blocks);
        for (0..depth) |i| {
            blocks[i] = .{
                .norm1_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "norm1.weight")),
                .norm1_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "norm1.bias")),
                .norm2_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "norm2.weight")),
                .norm2_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "norm2.bias")),
                .qkv_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "attn.qkv.weight")),
                .qkv_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "attn.qkv.bias")),
                .proj_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "attn.proj.weight")),
                .proj_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "attn.proj.bias")),
                .fc1_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "mlp.linear_fc1.weight")),
                .fc1_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "mlp.linear_fc1.bias")),
                .fc2_w = try must(weights, &buf, fmtLayer(&buf, prefix, i, "mlp.linear_fc2.weight")),
                .fc2_b = try must(weights, &buf, fmtLayer(&buf, prefix, i, "mlp.linear_fc2.bias")),
            };
        }

        return QwenVision{
            .s = s,
            .allocator = allocator,
            .depth = depth,
            .hidden = config.qv_hidden,
            .heads = config.qv_heads,
            .head_dim = config.qv_head_dim,
            .merge = config.qv_merge,
            .num_grid_per_side = @intFromFloat(@sqrt(@as(f64, @floatFromInt(config.qv_num_pos_emb)))),
            .out_hidden = config.qv_out_hidden,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embed = pos_embed,
            .blocks = blocks,
            .merger_norm_w = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.norm.weight")),
            .merger_norm_b = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.norm.bias")),
            .merger_fc1_w = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.linear_fc1.weight")),
            .merger_fc1_b = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.linear_fc1.bias")),
            .merger_fc2_w = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.linear_fc2.weight")),
            .merger_fc2_b = try must(weights, &buf, fmtKey(&kbuf, prefix, "merger.linear_fc2.bias")),
        };
    }

    pub fn deinit(self: *QwenVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        self.allocator.free(self.blocks);
    }

    fn bf16Scalar(self: *QwenVision, v: f32) mlx.mlx_array {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_astype(&out, f, .bfloat16, self.s);
        return out;
    }

    /// y = x · Wᵀ (+ bias). W is [out, in], x is [..., in].
    fn denseLinear(self: *QwenVision, x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array) !mlx.mlx_array {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        if (b) |bv| {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, out, bv, self.s));
            _ = mlx.mlx_array_free(out);
            out = biased;
        }
        return out;
    }

    fn layerNorm6(self: *QwenVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, 1e-6, self.s));
        return out;
    }

    /// nn.GELU(approx="tanh"): 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))).
    fn geluTanh(self: *QwenVision, x: mlx.mlx_array) !mlx.mlx_array {
        const c_coeff = self.bf16Scalar(0.7978845608028654);
        defer _ = mlx.mlx_array_free(c_coeff);
        const c_inner = self.bf16Scalar(0.044715);
        defer _ = mlx.mlx_array_free(c_inner);
        const c_three = self.bf16Scalar(3.0);
        defer _ = mlx.mlx_array_free(c_three);
        const c_one = self.bf16Scalar(1.0);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = self.bf16Scalar(0.5);
        defer _ = mlx.mlx_array_free(c_half);

        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, c_three, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, c_inner, x3, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, c_coeff, sum, self.s));
        var th = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(th);
        try mlx.check(mlx.mlx_tanh(&th, scaled, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, c_one, th, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, c_half, self.s));
        return out;
    }

    /// nn.GELU() exact: 0.5·x·(1+erf(x/√2)). Used by the PatchMerger.
    fn geluExact(self: *QwenVision, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = self.bf16Scalar(0.7071067811865476);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        const c_one = self.bf16Scalar(1.0);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = self.bf16Scalar(0.5);
        defer _ = mlx.mlx_array_free(c_half);
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_multiply(&t, x, inv_sqrt2, self.s));
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_erf(&e, t, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, c_one, e, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, c_half, self.s));
        return out;
    }

    /// rotate_half over the last dim (split at dim/2): concat(-x2, x1).
    fn rotateHalf(self: *QwenVision, x: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        const half = @divExact(hd, 2);
        const strides = [_]c_int{ 1, 1, 1 };
        const start1 = [_]c_int{ 0, 0, half };
        const stop1 = [_]c_int{ n, self.headsC(), hd };
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_slice(&x2, x, &start1, 3, &stop1, 3, &strides, 3, self.s));
        const start0 = [_]c_int{ 0, 0, 0 };
        const stop0 = [_]c_int{ n, self.headsC(), half };
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_slice(&x1, x, &start0, 3, &stop0, 3, &strides, 3, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, x2, self.s));
        const arrs = [_]mlx.mlx_array{ neg, x1 };
        const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
        return out;
    }

    inline fn headsC(self: *QwenVision) c_int {
        return @intCast(self.heads);
    }

    /// Apply vision 2D RoPE: out = x·cos + rotate_half(x)·sin. x is [N, heads, hd],
    /// cos/sin are [N, 1, hd] (broadcast over heads).
    fn applyVisionRope(self: *QwenVision, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        var xcos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xcos);
        try mlx.check(mlx.mlx_multiply(&xcos, x, cos, self.s));
        const rh = try self.rotateHalf(x, n, hd);
        defer _ = mlx.mlx_array_free(rh);
        var rsin = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rsin);
        try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, xcos, rsin, self.s));
        return out;
    }

    /// Build per-token vision-RoPE cos/sin tables [N, 1, head_dim] (bf16) for a
    /// single image, in merge-block token order. VisionRotaryEmbedding(head_dim/2):
    /// 16 freqs; each token gets [h_emb(16) ‖ w_emb(16)], tiled ×2 over head_dim.
    fn buildVisionRope(self: *QwenVision, grid_h: u32, grid_w: u32) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const hd: usize = self.head_dim;
        const half = hd / 2; // 32
        const nfreq = half / 2; // 16
        const merge = self.merge;
        const mh = grid_h / merge;
        const mw = grid_w / merge;
        const n: usize = grid_h * grid_w;

        var inv_freq = try self.allocator.alloc(f64, nfreq);
        defer self.allocator.free(inv_freq);
        for (0..nfreq) |k| {
            const exp = -@as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(half));
            inv_freq[k] = std.math.pow(f64, self.rope_theta, exp);
        }

        var cos_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(cos_buf);
        var sin_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(sin_buf);

        var token: usize = 0;
        var bh: usize = 0;
        while (bh < mh) : (bh += 1) {
            var bw: usize = 0;
            while (bw < mw) : (bw += 1) {
                var ir: usize = 0;
                while (ir < merge) : (ir += 1) {
                    var ic: usize = 0;
                    while (ic < merge) : (ic += 1) {
                        const row: f64 = @floatFromInt(bh * merge + ir);
                        const col: f64 = @floatFromInt(bw * merge + ic);
                        const o = token * hd;
                        for (0..nfreq) |k| {
                            const ah = row * inv_freq[k];
                            const aw = col * inv_freq[k];
                            // emb = [h(16), w(16)] then tile ×2 over head_dim.
                            cos_buf[o + k] = @floatCast(@cos(ah));
                            cos_buf[o + nfreq + k] = @floatCast(@cos(aw));
                            cos_buf[o + half + k] = @floatCast(@cos(ah));
                            cos_buf[o + half + nfreq + k] = @floatCast(@cos(aw));
                            sin_buf[o + k] = @floatCast(@sin(ah));
                            sin_buf[o + nfreq + k] = @floatCast(@sin(aw));
                            sin_buf[o + half + k] = @floatCast(@sin(ah));
                            sin_buf[o + half + nfreq + k] = @floatCast(@sin(aw));
                        }
                        token += 1;
                    }
                }
            }
        }

        const shape = [_]c_int{ @intCast(n), 1, @intCast(hd) };
        const cos_f = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(cos_f);
        const sin_f = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(sin_f);
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cos, cos_f, .bfloat16, self.s));
        var sin = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sin, sin_f, .bfloat16, self.s));
        return .{ .cos = cos, .sin = sin };
    }

    /// Interpolated learned position embeddings, merge-block order → [N, hidden] bf16.
    fn posEmbedInterpolate(self: *QwenVision, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const G = self.num_grid_per_side;
        const merge = self.merge;
        const mh = grid_h / merge;
        const mw = grid_w / merge;
        const n: usize = grid_h * grid_w;

        // Per-axis bilinear endpoints from linspace(0, G-1, grid).
        const Axis = struct { floor: []i32, ceil: []i32, frac: []f64 };
        const mkAxis = struct {
            fn f(a: std.mem.Allocator, len: u32, gside: u32) !Axis {
                const fl = try a.alloc(i32, len);
                const cl = try a.alloc(i32, len);
                const fr = try a.alloc(f64, len);
                const last: f64 = @floatFromInt(gside - 1);
                for (0..len) |i| {
                    const v: f64 = if (len == 1) 0.0 else last * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(len - 1));
                    const flo: i32 = @intFromFloat(v); // truncation toward 0 == floor for v>=0
                    fl[i] = flo;
                    cl[i] = @min(flo + 1, @as(i32, @intCast(gside - 1)));
                    fr[i] = v - @as(f64, @floatFromInt(flo));
                }
                return .{ .floor = fl, .ceil = cl, .frac = fr };
            }
        }.f;

        const ha = try mkAxis(self.allocator, grid_h, G);
        defer {
            self.allocator.free(ha.floor);
            self.allocator.free(ha.ceil);
            self.allocator.free(ha.frac);
        }
        const wa = try mkAxis(self.allocator, grid_w, G);
        defer {
            self.allocator.free(wa.floor);
            self.allocator.free(wa.ceil);
            self.allocator.free(wa.frac);
        }

        // Four corner index arrays + weights, emitted directly in merge-block order.
        var idx: [4][]i32 = undefined;
        var wgt: [4][]f32 = undefined;
        inline for (0..4) |c| {
            idx[c] = try self.allocator.alloc(i32, n);
            wgt[c] = try self.allocator.alloc(f32, n);
        }
        defer inline for (0..4) |c| {
            self.allocator.free(idx[c]);
            self.allocator.free(wgt[c]);
        };

        var token: usize = 0;
        var bh: usize = 0;
        while (bh < mh) : (bh += 1) {
            var bw: usize = 0;
            while (bw < mw) : (bw += 1) {
                var ir: usize = 0;
                while (ir < merge) : (ir += 1) {
                    var ic: usize = 0;
                    while (ic < merge) : (ic += 1) {
                        const row = bh * merge + ir;
                        const col = bw * merge + ic;
                        const hf = ha.floor[row];
                        const hc = ha.ceil[row];
                        const wf = wa.floor[col];
                        const wc = wa.ceil[col];
                        const dh = ha.frac[row];
                        const dw = wa.frac[col];
                        const gi: i32 = @intCast(G);
                        idx[0][token] = hf * gi + wf;
                        idx[1][token] = hf * gi + wc;
                        idx[2][token] = hc * gi + wf;
                        idx[3][token] = hc * gi + wc;
                        wgt[0][token] = @floatCast((1.0 - dh) * (1.0 - dw));
                        wgt[1][token] = @floatCast((1.0 - dh) * dw);
                        wgt[2][token] = @floatCast(dh * (1.0 - dw));
                        wgt[3][token] = @floatCast(dh * dw);
                        token += 1;
                    }
                }
            }
        }

        const idx_shape = [_]c_int{@intCast(n)};
        const wshape = [_]c_int{ @intCast(n), 1 };
        var acc = mlx.mlx_array_new();
        var first = true;
        inline for (0..4) |c| {
            const ix = mlx.mlx_array_new_data(idx[c].ptr, &idx_shape, 1, .int32);
            defer _ = mlx.mlx_array_free(ix);
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            try mlx.check(mlx.mlx_take_axis(&gathered, self.pos_embed, ix, 0, self.s)); // [N, hidden]
            const wf = mlx.mlx_array_new_data(wgt[c].ptr, &wshape, 2, .float32);
            defer _ = mlx.mlx_array_free(wf);
            var wbf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbf);
            try mlx.check(mlx.mlx_astype(&wbf, wf, .bfloat16, self.s));
            var weighted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(weighted);
            try mlx.check(mlx.mlx_multiply(&weighted, gathered, wbf, self.s));
            if (first) {
                // astype-copy so `acc` owns an array independent of `weighted`'s defer.
                try mlx.check(mlx.mlx_astype(&acc, weighted, .bfloat16, self.s));
                first = false;
            } else {
                var sum = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&sum, acc, weighted, self.s));
                _ = mlx.mlx_array_free(acc);
                acc = sum;
            }
        }
        return acc;
    }

    /// Self-attention over all patches (single image → full bidirectional).
    fn attention(self: *QwenVision, normed: mlx.mlx_array, blk: QBlock, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = self.headsC();
        const qkv = try self.denseLinear(normed, blk.qkv_w, blk.qkv_b); // [N, 3*hidden]
        defer _ = mlx.mlx_array_free(qkv);
        // [N, 3, heads, hd]
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        const rshape = [_]c_int{ n, 3, heads, hd };
        try mlx.check(mlx.mlx_reshape(&r, qkv, &rshape, 4, self.s));

        const sl_strides = [_]c_int{ 1, 1, 1, 1 };
        var parts: [3]mlx.mlx_array = undefined;
        inline for (0..3) |j| {
            const start = [_]c_int{ 0, @intCast(j), 0, 0 };
            const stop = [_]c_int{ n, @intCast(j + 1), heads, hd };
            var sliced = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sliced);
            try mlx.check(mlx.mlx_slice(&sliced, r, &start, 4, &stop, 4, &sl_strides, 4, self.s));
            const flat = [_]c_int{ n, heads, hd };
            var part = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&part, sliced, &flat, 3, self.s));
            parts[j] = part; // [N, heads, hd]
        }
        defer for (parts) |p| {
            _ = mlx.mlx_array_free(p);
        };

        const q_rope = try self.applyVisionRope(parts[0], cos, sin, n, hd);
        defer _ = mlx.mlx_array_free(q_rope);
        const k_rope = try self.applyVisionRope(parts[1], cos, sin, n, hd);
        defer _ = mlx.mlx_array_free(k_rope);

        // [N, heads, hd] → [heads, N, hd], in fp32 to mirror the reference's
        // fused SDPA (bf16 in/out, fp32 internal accumulation). Computing the
        // score/softmax/context core in bf16 leaves sparse ~0.9 outliers on
        // high-variance channels; fp32 internals match the reference far closer.
        const perm = [_]c_int{ 1, 0, 2 };
        var qh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qh);
        try mlx.check(mlx.mlx_transpose_axes(&qh, q_rope, &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&qh, qh, .float32, self.s));
        var kh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kh);
        try mlx.check(mlx.mlx_transpose_axes(&kh, k_rope, &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&kh, kh, .float32, self.s));
        var vh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vh);
        try mlx.check(mlx.mlx_transpose_axes(&vh, parts[2], &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&vh, vh, .float32, self.s));

        // scores = q @ k^T * scale
        const kperm = [_]c_int{ 0, 2, 1 };
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, kh, &kperm, 3, self.s));
        var scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores);
        try mlx.check(mlx.mlx_matmul(&scores, qh, kt, self.s));
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, scores, scale, self.s));
        var probs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(probs);
        try mlx.check(mlx.mlx_softmax_axis(&probs, scaled, -1, true, self.s));
        var ctx32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx32);
        try mlx.check(mlx.mlx_matmul(&ctx32, probs, vh, self.s)); // [heads, N, hd] f32
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_astype(&ctx, ctx32, .bfloat16, self.s));

        // [heads, N, hd] → [N, heads*hd]
        const operm = [_]c_int{ 1, 0, 2 };
        var ctt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctt);
        try mlx.check(mlx.mlx_transpose_axes(&ctt, ctx, &operm, 3, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const fshape = [_]c_int{ n, @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&flat, ctt, &fshape, 2, self.s));
        return self.denseLinear(flat, blk.proj_w, blk.proj_b);
    }

    /// Encode one image. `patches` = pixel_values [N, C*tps*ps*ps] (merge order).
    pub fn forward(self: *QwenVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));

        // patch_embed (Linear) + interpolated pos_embed.
        {
            const pe = try self.denseLinear(x, self.patch_w, self.patch_b);
            _ = mlx.mlx_array_free(x);
            x = pe;
        }
        {
            const pos = try self.posEmbedInterpolate(grid_h, grid_w);
            defer _ = mlx.mlx_array_free(pos);
            var added = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&added, x, pos, self.s));
            _ = mlx.mlx_array_free(x);
            x = added;
        }

        const rope = try self.buildVisionRope(grid_h, grid_w);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);

        var dt = mlx.DtypeTrace.begin("qwen3vl-vision", x, if (self.blocks.len > 0) self.blocks[0].qkv_w else null);
        for (self.blocks, 0..) |blk, block_idx| {
            // Attention residual.
            {
                const normed = try self.layerNorm6(x, blk.norm1_w, blk.norm1_b);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, blk, rope.cos, rope.sin, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                _ = mlx.mlx_array_free(x);
                x = h;
            }
            // MLP residual.
            {
                const normed = try self.layerNorm6(x, blk.norm2_w, blk.norm2_b);
                defer _ = mlx.mlx_array_free(normed);
                const fc1 = try self.denseLinear(normed, blk.fc1_w, blk.fc1_b);
                defer _ = mlx.mlx_array_free(fc1);
                const act = try self.geluTanh(fc1);
                defer _ = mlx.mlx_array_free(act);
                const fc2 = try self.denseLinear(act, blk.fc2_w, blk.fc2_b);
                defer _ = mlx.mlx_array_free(fc2);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, fc2, self.s));
                _ = mlx.mlx_array_free(x);
                x = h;
            }
            dt.layer(x, block_idx);
            if (self.eval_per_block) try mlx.check(mlx.mlx_array_eval(x));
        }
        dt.end(x);

        // Merger: LayerNorm(hidden) → reshape [N/merge², hidden·merge²] → fc2(gelu(fc1)).
        const normed = try self.layerNorm6(x, self.merger_norm_w, self.merger_norm_b);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(normed);
        const merge2: c_int = @intCast(self.merge * self.merge);
        const n_merged = @divExact(n, merge2);
        var grouped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grouped);
        const gshape = [_]c_int{ n_merged, @intCast(self.hidden * self.merge * self.merge) };
        try mlx.check(mlx.mlx_reshape(&grouped, normed, &gshape, 2, self.s));
        const m1 = try self.denseLinear(grouped, self.merger_fc1_w, self.merger_fc1_b);
        defer _ = mlx.mlx_array_free(m1);
        const ma = try self.geluExact(m1);
        defer _ = mlx.mlx_array_free(ma);
        const m2 = try self.denseLinear(ma, self.merger_fc2_w, self.merger_fc2_b);
        defer _ = mlx.mlx_array_free(m2);
        // [N_merged, out_hidden] → [1, N_merged, out_hidden].
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        const oshape = [_]c_int{ 1, n_merged, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, m2, &oshape, 3, self.s));
        // Images queued lazily would hold their scratch together.
        if (self.eval_per_block) try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// Encode one VIDEO. `patches` is the concatenation of `grid_t` temporal-
    /// patch groups' pixel_values (see `buildPixelValuesVideo`), each group's
    /// `grid_h*grid_w` rows packed contiguously. mlx-vlm's `cu_seqlens`
    /// boundaries fall exactly at temporal-patch-group edges — the ViT never
    /// attends across frames — so a group is encoded by calling the existing,
    /// UNCHANGED single-group `forward` (pos-embed and vision-rope are spatial
    /// only and already correct per group), concatenating the merged token
    /// outputs along the token axis.
    pub fn forwardVideo(self: *QwenVision, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        std.debug.assert(grid_t > 0);
        if (grid_t == 1) return self.forward(patches, grid_h, grid_w);

        const n_per_group: c_int = @intCast(grid_h * grid_w);
        const pshape = mlx.getShape(patches);
        std.debug.assert(pshape.len == 2);
        const feat = pshape[1];

        var parts = try self.allocator.alloc(mlx.mlx_array, grid_t);
        defer self.allocator.free(parts);
        var built: usize = 0;
        errdefer for (parts[0..built]) |p| { _ = mlx.mlx_array_free(p); };

        var t: u32 = 0;
        while (t < grid_t) : (t += 1) {
            var group = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(group);
            const ti: c_int = @intCast(t);
            const start = [_]c_int{ ti * n_per_group, 0 };
            const stop = [_]c_int{ (ti + 1) * n_per_group, feat };
            const strides = [_]c_int{ 1, 1 };
            try mlx.check(mlx.mlx_slice(&group, patches, &start, 2, &stop, 2, &strides, 2, self.s));
            parts[t] = try self.forward(group, grid_h, grid_w);
            built += 1;
        }
        defer for (parts) |p| { _ = mlx.mlx_array_free(p); };

        const cat_vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, cat_vec, 1, self.s));
        return out;
    }
};

fn getWeightLocal(weights: *const Weights, buf: *[256]u8, name: []const u8) ?mlx.mlx_array {
    _ = buf;
    return weights.get(name);
}

fn fmtLayer(buf: *[256]u8, prefix: []const u8, layer: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}blocks.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
}

fn fmtKey(buf: *[256]u8, prefix: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, suffix }) catch unreachable;
}

/// Tower spellings we serve, most common first. Qwen3-VL checkpoints say
/// `vision_tower.`; avlp12's Qwen3.8 "Alis" packs ship the same 333 tensors
/// under `model.visual.` (pure rename, substructure identical).
pub const VISION_PREFIXES = [_][]const u8{ "vision_tower.", "model.visual." };

/// Stored axis order of `patch_embed.proj.weight`. mlx_lm's Conv3d writes
/// channels LAST (`[out, kT, ps, ps, Cin]`); a straight torch export keeps
/// channels FIRST (`[out, Cin, kT, ps, ps]` — avlp12's Alis packs). Reading
/// one as the other produces a running tower that describes every image as
/// black-and-white stripes, so the layout is DERIVED from the shape.
pub const PatchProjLayout = enum { channels_last, channels_first };

pub fn patchProjLayout(shape: []const c_int, temporal_patch: u32, patch: u32) ?PatchProjLayout {
    if (shape.len != 5 or temporal_patch == 0 or patch == 0) return null;
    const tps: c_int = @intCast(temporal_patch);
    const ps: c_int = @intCast(patch);
    if (shape[1] == tps and shape[2] == ps and shape[3] == ps) return .channels_last;
    if (shape[2] == tps and shape[3] == ps and shape[4] == ps) return .channels_first;
    return null;
}

/// Resolve the tower prefix by PROBING the loaded weights (model.resolveWeightPrefix
/// pattern) — null when no spelling is present, which is what disables vision.
pub fn resolveVisionPrefix(weights: *const Weights) ?[]const u8 {
    var buf: [256]u8 = undefined;
    for (VISION_PREFIXES) |p| {
        if (weights.get(fmtKey(&buf, p, "patch_embed.proj.weight")) != null) return p;
    }
    return null;
}

test "qwen vision tower prefix and patch_embed layout are PROBED, not hardcoded" {
    const allocator = std.testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Qwen3-VL spelling.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "vision_tower.patch_embed.proj.weight");
        try std.testing.expectEqualStrings("vision_tower.", resolveVisionPrefix(&w).?);
    }
    // avlp12 Alis spelling (live: vision silently disabled, 0.92 GB dead weight).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.visual.patch_embed.proj.weight");
        try std.testing.expectEqualStrings("model.visual.", resolveVisionPrefix(&w).?);
    }
    // No tower (text-only pack, or --no-vision dropped it) → vision disabled.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.0.self_attn.q_proj.weight");
        try std.testing.expect(resolveVisionPrefix(&w) == null);
    }
    // patch_embed.proj axis order is derived from the shape, not assumed.
    {
        // mlx-community/Qwen3.5-0.8B-MLX-4bit
        try std.testing.expectEqual(PatchProjLayout.channels_last, patchProjLayout(&.{ 768, 2, 16, 16, 3 }, 2, 16).?);
        // avlp12/Qwen3.8-27B-Alis-MLX-4bit
        try std.testing.expectEqual(PatchProjLayout.channels_first, patchProjLayout(&.{ 1152, 3, 2, 16, 16 }, 2, 16).?);
        try std.testing.expect(patchProjLayout(&.{ 1152, 3, 2, 16 }, 2, 16) == null);
        try std.testing.expect(patchProjLayout(&.{ 1152, 5, 5, 5, 5 }, 2, 16) == null);
    }
    // Layer keys are built from the RESOLVED prefix.
    {
        var buf: [256]u8 = undefined;
        try std.testing.expectEqualStrings(
            "model.visual.blocks.7.attn.qkv.weight",
            fmtLayer(&buf, "model.visual.", 7, "attn.qkv.weight"),
        );
    }
}

fn readBinF32(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]f32 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var rs = file.reader(io, &buf);
    const bytes = try rs.interface.allocRemaining(alloc, .limited(256 * 1024 * 1024));
    // Reinterpret the raw little-endian float32 payload.
    const n = bytes.len / 4;
    const out = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    alloc.free(bytes);
    return out;
}

fn readBinU8(io: std.Io, alloc: std.mem.Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var rs = file.reader(io, &buf);
    return rs.interface.allocRemaining(alloc, .limited(max_bytes));
}

test "qwen preprocessing parity vs reference pixel_values (QWEN_PREPROCESS_FIXTURE)" {
    const fixture_raw = std.c.getenv("QWEN_PREPROCESS_FIXTURE") orelse return error.SkipZigTest;
    const fixture = std.mem.span(fixture_raw);
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    const manifest_path = try std.fmt.allocPrint(allocator, "{s}/manifest.json", .{fixture});
    defer allocator.free(manifest_path);
    const manifest_bytes = try readBinU8(io, allocator, manifest_path, 1024 * 1024);
    defer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const source_h: u32 = @intCast(root.get("source_height").?.integer);
    const source_w: u32 = @intCast(root.get("source_width").?.integer);
    const resized_h: u32 = @intCast(root.get("resized_height").?.integer);
    const resized_w: u32 = @intCast(root.get("resized_width").?.integer);
    const patch: u32 = @intCast(root.get("patch_size").?.integer);
    const tps: u32 = @intCast(root.get("temporal_patch_size").?.integer);
    const merge: u32 = @intCast(root.get("merge_size").?.integer);
    const min_pixels: u32 = @intCast(root.get("min_pixels").?.integer);
    const max_pixels: u32 = @intCast(root.get("max_pixels").?.integer);
    const expected_len: usize = @intCast(root.get("pixel_values_length").?.integer);

    const source_path = try std.fmt.allocPrint(allocator, "{s}/source_rgb.bin", .{fixture});
    defer allocator.free(source_path);
    const source = try readBinU8(io, allocator, source_path, 256 * 1024 * 1024);
    defer allocator.free(source);
    try std.testing.expectEqual(@as(usize, source_h) * source_w * 3, source.len);

    const reference_path = try std.fmt.allocPrint(allocator, "{s}/pixel_values.bin", .{fixture});
    defer allocator.free(reference_path);
    const reference = try readBinF32(io, allocator, reference_path);
    defer allocator.free(reference);
    try std.testing.expectEqual(expected_len, reference.len);

    const factor = patch * merge;
    const resized = common.smartResizeImage(source_h, source_w, factor, min_pixels, max_pixels);
    try std.testing.expectEqual(resized_h, resized.h);
    try std.testing.expectEqual(resized_w, resized.w);

    const plane: usize = @as(usize, resized.h) * resized.w;
    const chw = try allocator.alloc(f32, 3 * plane);
    defer allocator.free(chw);
    try common.resizeRgbBicubicNormalizedChw(allocator, chw, source, source_h, source_w, resized.h, resized.w);

    const actual = try allocator.alloc(f32, reference.len);
    defer allocator.free(actual);
    common.buildPixelValues(actual, chw, 3, resized.h, resized.w, patch, tps, merge);

    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var over_point_one: usize = 0;
    for (actual, reference) |got, want| {
        const diff = @abs(got - want);
        max_abs = @max(max_abs, diff);
        sum_abs += diff;
        if (diff > 0.10) over_point_one += 1;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(actual.len));
    const fraction_over_point_one =
        @as(f64, @floatFromInt(over_point_one)) / @as(f64, @floatFromInt(actual.len));
    std.debug.print(
        "qwen preprocessing parity: max_abs={d:.6} mean_abs={d:.6} over_0.10={d:.4}%\n",
        .{ max_abs, mean_abs, fraction_over_point_one * 100.0 },
    );
    try std.testing.expect(max_abs < 0.30);
    try std.testing.expect(mean_abs < 0.005);
    try std.testing.expect(fraction_over_point_one < 0.002);
}

// Live parity vs the mlx-vlm reference vision tower. Feeds the REFERENCE's own
// pixel_values straight into QwenVision (isolating the ViT math from the
// preprocessing), then compares post-merger embeddings. Build the fixture first:
//   python3 tests/build_qwen_vision_fixture.py --model <dir> --image <img> --out <fix>
// then run via tests/test_qwen_vision_parity.sh (sets the env vars below).
test "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" {
    const model_raw = std.c.getenv("QWEN_VISION_TEST_MODEL") orelse return error.SkipZigTest;
    const fix_raw = std.c.getenv("QWEN_VISION_FIXTURE") orelse return error.SkipZigTest;
    const gh_raw = std.c.getenv("QV_GH") orelse return error.SkipZigTest;
    const gw_raw = std.c.getenv("QV_GW") orelse return error.SkipZigTest;
    const dir = std.mem.sliceTo(model_raw, 0);
    const fix = std.mem.sliceTo(fix_raw, 0);
    if (dir.len == 0 or fix.len == 0) return error.SkipZigTest;
    const grid_h = try std.fmt.parseInt(u32, std.mem.sliceTo(gh_raw, 0), 10);
    const grid_w = try std.fmt.parseInt(u32, std.mem.sliceTo(gw_raw, 0), 10);

    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const config = try model_mod.parseConfig(io, alloc, dir);
    try std.testing.expect(config.qwen_vision);
    var weights = try model_mod.loadWeightsWithVision(io, alloc, dir);
    defer weights.deinit();
    var qv = try QwenVision.init(alloc, config, &weights);
    defer qv.deinit();

    // Reference pixel_values [N, C*tps*ps*ps] and post-merger embeddings.
    const px_path = try std.fmt.allocPrint(alloc, "{s}/pixel_values.bin", .{fix});
    defer alloc.free(px_path);
    const ref_path = try std.fmt.allocPrint(alloc, "{s}/ref_embeds.bin", .{fix});
    defer alloc.free(ref_path);
    const px = try readBinF32(io, alloc, px_path);
    defer alloc.free(px);
    const ref = try readBinF32(io, alloc, ref_path);
    defer alloc.free(ref);

    const n = grid_h * grid_w;
    const feat = px.len / n;
    const px_shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const px_arr = mlx.mlx_array_new_data(px.ptr, &px_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(px_arr);

    const out = try qv.forward(px_arr, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(out);
    var out_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_f32);
    try mlx.check(mlx.mlx_astype(&out_f32, out, .float32, qv.s));
    try mlx.check(mlx.mlx_array_eval(out_f32));
    const od = mlx.mlx_array_data_float32(out_f32) orelse return error.TestUnexpectedNullData;

    try std.testing.expectEqual(ref.len, n / 4 * config.qv_out_hidden);
    const dim = config.qv_out_hidden;
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_abs: f64 = 0;
    var gt_010: usize = 0;
    var gt_030: usize = 0;
    // Per-token max diff to see clustering (a structural bug clusters by token).
    var worst_tok_diff: f64 = 0;
    var worst_tok: usize = 0;
    var tok_sum: f64 = 0;
    var cur_tok: usize = 0;
    for (0..ref.len) |i| {
        const d = @abs(od[i] - ref[i]);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
        if (d > 0.10) gt_010 += 1;
        if (d > 0.30) gt_030 += 1;
        sum_abs += d;
        const tok = i / dim;
        if (tok != cur_tok) {
            if (tok_sum > worst_tok_diff) {
                worst_tok_diff = tok_sum;
                worst_tok = cur_tok;
            }
            tok_sum = 0;
            cur_tok = tok;
        }
        tok_sum += d;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(ref.len));
    std.debug.print("\nqwen vision parity: max_abs={d:.5} mean_abs={d:.6} (N_merged={d}, dim={d})\n", .{ max_abs, mean_abs, n / 4, dim });
    std.debug.print("  max@ token={d} chan={d} ours={d:.4} ref={d:.4}\n", .{ max_idx / dim, max_idx % dim, od[max_idx], ref[max_idx] });
    std.debug.print("  diffs>0.10: {d}/{d}  diffs>0.30: {d}  worst-token meanrow-diff token={d} sum={d:.3}\n", .{ gt_010, ref.len, gt_030, worst_tok, worst_tok_diff / @as(f64, @floatFromInt(dim)) });
    // bf16 ViT through 12 blocks with different reduction orders than the
    // reference: tolerate accumulated rounding + a handful of outliers, catch
    // real bugs (a structural bug blows up mean_abs by 10-100x).
    try std.testing.expect(mean_abs < 0.02);
    try std.testing.expect(gt_030 < ref.len / 1000); // <0.1% of elements may exceed 0.30
}

test "qwen forwardVideo == per-group forward()+concat (self-consistency)" {
    // No hermetic reference exists for ViT-forward CORRECTNESS (that needs a
    // trained checkpoint — see the QWEN_VISION_TEST_MODEL-gated parity test
    // above). What IS hermetically checkable, and what's new in this task, is
    // the forwardVideo WRAPPER: does it slice `grid_t` temporal-patch groups
    // out of the concatenated patches array and route each through the
    // existing, unmodified single-group `forward` correctly? This builds a
    // tiny synthetic tower (weight VALUES are arbitrary) and asserts
    // forwardVideo(3 groups) is bit-identical to manually slicing+forward()ing
    // each group and concatenating — the exact operation forwardVideo performs
    // internally, so any slicing/indexing bug in the wrapper shows up here.
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    const mkArr = struct {
        fn f(shape: []const c_int, val_base: f32) mlx.mlx_array {
            var buf: [64]f32 = undefined;
            var total: usize = 1;
            for (shape) |d| total *= @intCast(d);
            for (0..total) |i| buf[i] = val_base + @as(f32, @floatFromInt(i)) * 0.01;
            return mlx.mlx_array_new_data(&buf, shape.ptr, @intCast(shape.len), .float32);
        }
    }.f;

    // hidden=4, 1 head of head_dim=4, depth=1, merge=1 (no reduction — keeps
    // token counts trivial), grid 2x2 (n=4 patches/group) matching
    // num_grid_per_side=2 exactly (identity pos-embed interpolation).
    const hidden = [_]c_int{4};
    const hidden2 = [_]c_int{ 4, 4 };
    const qkv_w_shape = [_]c_int{ 12, 4 };
    const qkv_b_shape = [_]c_int{12};
    const patch_w_shape = [_]c_int{ 4, 1 };
    const pos_shape = [_]c_int{ 4, 4 };

    var qv = QwenVision{
        .s = s,
        .allocator = a,
        .depth = 1,
        .hidden = 4,
        .heads = 1,
        .head_dim = 4,
        .merge = 1,
        .num_grid_per_side = 2,
        .out_hidden = 4,
        .patch_w = mkArr(&patch_w_shape, 0.1),
        .patch_b = mkArr(&hidden, 0.0),
        .pos_embed = mkArr(&pos_shape, 0.2),
        .blocks = try a.alloc(QBlock, 1),
        .merger_norm_w = mkArr(&hidden, 1.0),
        .merger_norm_b = mkArr(&hidden, 0.0),
        .merger_fc1_w = mkArr(&hidden2, 0.05),
        .merger_fc1_b = mkArr(&hidden, 0.0),
        .merger_fc2_w = mkArr(&hidden2, 0.05),
        .merger_fc2_b = mkArr(&hidden, 0.0),
    };
    qv.blocks[0] = .{
        .norm1_w = mkArr(&hidden, 1.0),
        .norm1_b = mkArr(&hidden, 0.0),
        .norm2_w = mkArr(&hidden, 1.0),
        .norm2_b = mkArr(&hidden, 0.0),
        .qkv_w = mkArr(&qkv_w_shape, 0.03),
        .qkv_b = mkArr(&qkv_b_shape, 0.0),
        .proj_w = mkArr(&hidden2, 0.04),
        .proj_b = mkArr(&hidden, 0.0),
        .fc1_w = mkArr(&hidden2, 0.02),
        .fc1_b = mkArr(&hidden, 0.0),
        .fc2_w = mkArr(&hidden2, 0.02),
        .fc2_b = mkArr(&hidden, 0.0),
    };
    defer {
        _ = mlx.mlx_array_free(qv.patch_w);
        _ = mlx.mlx_array_free(qv.patch_b);
        _ = mlx.mlx_array_free(qv.pos_embed);
        _ = mlx.mlx_array_free(qv.merger_norm_w);
        _ = mlx.mlx_array_free(qv.merger_norm_b);
        _ = mlx.mlx_array_free(qv.merger_fc1_w);
        _ = mlx.mlx_array_free(qv.merger_fc1_b);
        _ = mlx.mlx_array_free(qv.merger_fc2_w);
        _ = mlx.mlx_array_free(qv.merger_fc2_b);
        for (qv.blocks) |b| {
            _ = mlx.mlx_array_free(b.norm1_w);
            _ = mlx.mlx_array_free(b.norm1_b);
            _ = mlx.mlx_array_free(b.norm2_w);
            _ = mlx.mlx_array_free(b.norm2_b);
            _ = mlx.mlx_array_free(b.qkv_w);
            _ = mlx.mlx_array_free(b.qkv_b);
            _ = mlx.mlx_array_free(b.proj_w);
            _ = mlx.mlx_array_free(b.proj_b);
            _ = mlx.mlx_array_free(b.fc1_w);
            _ = mlx.mlx_array_free(b.fc1_b);
            _ = mlx.mlx_array_free(b.fc2_w);
            _ = mlx.mlx_array_free(b.fc2_b);
        }
        a.free(qv.blocks);
    }

    // grid_t=3 groups of a 2x2 grid (n=4 patches/group, feat=1) — 12 rows total.
    const grid_t: u32 = 3;
    const n_per_group: usize = 4;
    var patches_buf: [12]f32 = undefined;
    for (0..12) |i| patches_buf[i] = @as(f32, @floatFromInt(i)) * 0.1;
    const all_shape = [_]c_int{ 12, 1 };
    const patches_all = mlx.mlx_array_new_data(&patches_buf, &all_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(patches_all);

    const out_video = try qv.forwardVideo(patches_all, grid_t, 2, 2);
    defer _ = mlx.mlx_array_free(out_video);

    // Manual reference: slice each group from the SAME source buffer, call
    // forward() directly, concatenate along the token axis.
    var manual_parts: [3]mlx.mlx_array = undefined;
    for (0..grid_t) |g| {
        const group_shape = [_]c_int{ @intCast(n_per_group), 1 };
        const group_arr = mlx.mlx_array_new_data(patches_buf[g * n_per_group ..].ptr, &group_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(group_arr);
        manual_parts[g] = try qv.forward(group_arr, 2, 2);
    }
    defer for (manual_parts) |p| { _ = mlx.mlx_array_free(p); };
    const cat_vec = mlx.mlx_vector_array_new_data(&manual_parts, manual_parts.len);
    defer _ = mlx.mlx_vector_array_free(cat_vec);
    var out_manual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_manual);
    try mlx.check(mlx.mlx_concatenate_axis(&out_manual, cat_vec, 1, s));

    var v_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_f32);
    try mlx.check(mlx.mlx_astype(&v_f32, out_video, .float32, s));
    try mlx.check(mlx.mlx_array_eval(v_f32));
    var m_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_f32);
    try mlx.check(mlx.mlx_astype(&m_f32, out_manual, .float32, s));
    try mlx.check(mlx.mlx_array_eval(m_f32));

    const v_shape = mlx.getShape(v_f32);
    const m_shape = mlx.getShape(m_f32);
    try std.testing.expectEqualSlices(c_int, m_shape, v_shape);
    try std.testing.expectEqual(@as(c_int, 1), v_shape[0]);
    try std.testing.expectEqual(@as(c_int, 12), v_shape[1]); // 3 groups × 4 merged tokens (merge=1)

    const v_data = mlx.mlx_array_data_float32(v_f32) orelse return error.TestUnexpectedNullData;
    const m_data = mlx.mlx_array_data_float32(m_f32) orelse return error.TestUnexpectedNullData;
    const total: usize = 12 * 4; // tokens × out_hidden
    try std.testing.expectEqualSlices(f32, m_data[0..total], v_data[0..total]);
}

test "qwen encodeScratchBytes covers each measured peak by >= 25% without the N^2 formula's over-bill" {
    // Qwen3.8-Flash-Next tower (heads 16, hidden 1152, intermediate 4304).
    var config = ModelConfig{};
    config.qv_heads = 16;
    config.qv_hidden = 1152;
    config.qv_intermediate = 4304;
    // Per-block-eval scratch peaks, bytes (`qwen vision ubench`, Sushi-3bpw tower).
    const measured = [_][2]u64{
        .{ 196, 20_800_000 },
        .{ 3772, 1_225_400_000 },
        .{ 8160, 4_923_400_000 },
        .{ 9216, 6_112_100_000 },
    };
    for (measured) |m| {
        const bill = encodeScratchBytes(&config, m[0]);
        try std.testing.expect(bill * 4 >= m[1] * 5);
        // Where the bill decides admission, it stays within 1.5x of the peak.
        if (m[0] >= 1000) try std.testing.expect(bill * 2 <= m[1] * 3);
    }
}

// Encode time and scratch peak on the real tower, per image, at a small image, a
// 1920x1080 screenshot and the engine's 1536^2 cap (random pixels; numerics are
// the parity test's job), with and without the per-block eval, arms interleaved;
// then per video, its peak from the pixel upload to the evaluated output against
// `server.visionEncodeBill`:
//   QWEN_VISION_TEST_MODEL=<pack> SUSHI_QWEN_VISION_UBENCH=<reps> \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="qwen vision ubench"
test "qwen vision ubench: encode time and scratch peak per image and video" {
    const reps_raw = std.c.getenv("SUSHI_QWEN_VISION_UBENCH") orelse return error.SkipZigTest;
    const model_raw = std.c.getenv("QWEN_VISION_TEST_MODEL") orelse return error.SkipZigTest;
    const reps = std.fmt.parseInt(u32, std.mem.span(reps_raw), 10) catch return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const dir = std.mem.span(model_raw);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    try std.testing.expect(config.qwen_vision);
    var weights = Weights.init(a);
    defer weights.deinit();
    {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/model-vision.safetensors", .{dir}, 0);
        defer a.free(path);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try model_mod.loadSafetensorsFile(a, &weights, path, cpu, true);
    }
    var tower = try QwenVision.init(a, config, &weights);
    defer tower.deinit();

    const feat: c_int = @intCast(3 * config.qv_temporal_patch * config.qv_patch * config.qv_patch);
    const cap_side: u32 = std.math.sqrt(common.ENGINE_MAX_PIXELS) / config.qv_patch;
    // A 1920x1080 screenshot under this pack's own processor bounds.
    const bounds = common.effectivePixelBounds(config.qv_min_pixels, config.qv_max_pixels);
    const shot = common.smartResizeImage(1080, 1920, config.qv_patch * config.qv_merge, bounds.min, bounds.max);
    var under_billed = false;
    for ([_][2]u32{ .{ 14, 14 }, .{ shot.h / config.qv_patch, shot.w / config.qv_patch }, .{ 68, 120 }, .{ cap_side, cap_side } }) |g| {
        const n: c_int = @intCast(g[0] * g[1]);
        var pv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pv);
        const shape = [_]c_int{ n, feat };
        try mlx.check(mlx.mlx_random_normal(&pv, &shape, 2, .float32, 0, 1, .{ .ctx = null }, tower.s));
        try mlx.check(mlx.mlx_array_eval(pv));
        var best = [2]u64{ std.math.maxInt(u64), std.math.maxInt(u64) };
        var peak = [2]usize{ 0, 0 };
        for (0..reps + 1) |r| {
            // Arm 0 = today's lazy graph, arm 1 = eval per block; the order alternates per rep.
            for (0..2) |k| {
                const arm = (k + r) % 2;
                tower.eval_per_block = arm == 1;
                try mlx.check(mlx.mlx_synchronize(tower.s));
                _ = mlx.mlx_clear_cache();
                var before: usize = 0;
                _ = mlx.mlx_get_active_memory(&before);
                _ = mlx.mlx_reset_peak_memory();
                const t0 = std.Io.Timestamp.now(io, .boot);
                const out = try tower.forward(pv, g[0], g[1]);
                try mlx.check(mlx.mlx_array_eval(out));
                try mlx.check(mlx.mlx_synchronize(tower.s));
                const ns: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
                var p: usize = 0;
                _ = mlx.mlx_get_peak_memory(&p);
                _ = mlx.mlx_array_free(out);
                peak[arm] = @max(peak[arm], p -| before);
                if (r > 0) best[arm] = @min(best[arm], ns);
            }
        }
        tower.eval_per_block = true;
        const bill = encodeScratchBytes(&config, @intCast(n));
        const mb = 1e6;
        std.debug.print("[qwen-vit ubench] grid {d}x{d} ({d} patches, {d} tokens): lazy best {d:.1} ms peak {d:.1} MB | per-block eval best {d:.1} ms peak {d:.1} MB | bill {d:.1} MB (reps {d})\n", .{
            g[0],                                  g[1],                                     n,
            @divExact(n, @as(c_int, @intCast(config.qv_merge * config.qv_merge))),
            @as(f64, @floatFromInt(best[0])) / 1e6, @as(f64, @floatFromInt(peak[0])) / mb,
            @as(f64, @floatFromInt(best[1])) / 1e6, @as(f64, @floatFromInt(peak[1])) / mb,
            @as(f64, @floatFromInt(bill)) / mb,     reps,
        });
        if (bill < peak[1]) under_billed = true;
    }
    const server = @import("server.zig");
    const merge2 = config.qv_merge * config.qv_merge;
    for ([_][3]u32{ .{ 1, 46, 82 }, .{ 2, 46, 82 }, .{ 4, 46, 82 }, .{ 8, 46, 82 }, .{ 2, 24, 42 }, .{ 8, 24, 42 }, .{ 2, cap_side, cap_side }, .{ 8, cap_side, cap_side } }) |v| {
        const n: c_int = @intCast(v[0] * v[1] * v[2]);
        var src = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(src);
        const shape = [_]c_int{ n, feat };
        try mlx.check(mlx.mlx_random_normal(&src, &shape, 2, .float32, 0, 1, .{ .ctx = null }, tower.s));
        try mlx.check(mlx.mlx_array_eval(src));
        const host = (mlx.mlx_array_data_float32(src) orelse return error.TestUnexpectedNullData)[0..@intCast(n * feat)];
        var best: u64 = std.math.maxInt(u64);
        var peak: usize = 0;
        for (0..reps + 1) |r| {
            try mlx.check(mlx.mlx_synchronize(tower.s));
            _ = mlx.mlx_clear_cache();
            var before: usize = 0;
            _ = mlx.mlx_get_active_memory(&before);
            _ = mlx.mlx_reset_peak_memory();
            const t0 = std.Io.Timestamp.now(io, .boot);
            // As `scheduler.encodeVideoBlock`: the upload is a copy the encode owns.
            const pixels = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
            const out = tower.forwardVideo(pixels, v[0], v[1], v[2]);
            _ = mlx.mlx_array_free(pixels);
            const emb = try out;
            defer _ = mlx.mlx_array_free(emb);
            try mlx.check(mlx.mlx_array_eval(emb));
            try mlx.check(mlx.mlx_synchronize(tower.s));
            const ns: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
            var p: usize = 0;
            _ = mlx.mlx_get_peak_memory(&p);
            peak = @max(peak, p -| before);
            if (r > 0) best = @min(best, ns);
        }
        const rows: u64 = @as(u64, @intCast(n)) / merge2;
        const video = [_]@import("chat.zig").VideoData{.{ .pixels = std.mem.sliceAsBytes(host), .grid_t = v[0], .grid_h = v[1], .grid_w = v[2] }};
        const bill = server.visionEncodeBill(&config, &.{}, &video, rows).bytes;
        const mb = 1e6;
        const peak_f: f64 = @floatFromInt(peak);
        const bill_f: f64 = @floatFromInt(bill);
        std.debug.print("[qwen-vit ubench] video {d}x{d}x{d} ({d} patches, {d} tokens): best {d:.1} ms peak {d} B ({d:.1} MB) | bill {d} B ({d:.1} MB, {d:.2}x) (reps {d})\n", .{
            v[0], v[1], v[2], n, rows,
            @as(f64, @floatFromInt(best)) / 1e6, peak, peak_f / mb,
            bill, bill_f / mb, bill_f / peak_f, reps,
        });
        if (bill < peak) under_billed = true;
    }
    try std.testing.expect(!under_billed);
}
