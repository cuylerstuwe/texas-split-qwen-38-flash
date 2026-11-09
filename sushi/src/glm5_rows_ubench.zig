//! DIAGNOSTIC (SUSHI_GLM_ROWS_UBENCH=N): at load, N rounds per width B in 1..4 of one plain decode
//! row for each of B requests, two arms interleaved per round: B serial one-row forwards against one
//! grouped verification (`verifyGroups`, one single-row group per request, the DFlash2 capture taps).
//! Requests prefill distinct `_CTX`-token (default 1024) windows of `_TEXT` (else synthetic ids).
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const verifier = @import("glm5_dflash_model.zig");
const adapter = @import("glm5_dflash.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const widths = 4;
const taps = [_]u32{ 5, 14, 24, 33, 42 };

pub fn run(allocator: std.mem.Allocator, target: *forward.Model, latent_bits: u8, source: []const u32, ctx: usize, rounds: usize) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var requests: [widths]forward.Request = undefined;
    var made: usize = 0;
    defer for (requests[0..made]) |*r| r.deinit();
    var next: [widths]u32 = undefined;
    for (&requests, 0..) |*request, b| {
        request.* = try forward.Request.initServing(allocator, target.layers.len);
        made += 1;
        try request.setLatentBits(latent_bits);
        const window = try allocator.alloc(u32, ctx);
        defer allocator.free(window);
        for (window, 0..) |*id, i| id.* = if (source.len > 0) source[(b * ctx + i) % source.len] else @intCast(1 + (i * 7919 + b * 104729) % 150000);
        const ids = mlx.mlx_array_new_data(window.ptr, &.{ 1, @intCast(ctx) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(request, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        var best = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(best);
        try mlx.check(mlx.mlx_argmax_axis(&best, logits, -1, false, target.s));
        try mlx.check(mlx.mlx_array_eval(best));
        var token: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&token, best));
        next[b] = @intCast(token);
    }
    var serial_ns: [widths + 1]u64 = @splat(0);
    var rows_ns: [widths + 1]u64 = @splat(0);
    const binding = verifier.bindDefaultSchedule();
    defer binding.restore();
    // Round 0 compiles every shape and is not counted.
    for (0..rounds + 1) |round| for (1..widths + 1) |width| {
        var sw = io_util.Stopwatch.init(io);
        for (0..width) |b| {
            var clone = try adapter.cloneRequest(&requests[b]);
            defer clone.deinit();
            const ids = mlx.mlx_array_new_data(&next[b], &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const logits = try target.forwardLast(&clone, ids, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        if (round > 0) serial_ns[width] += sw.read();
        var groups: [widths]verifier.Group = undefined;
        for (groups[0..width], 0..) |*g, b| g.* = .{ .request = &requests[b], .tokens = next[b .. b + 1], .parents = &.{-1} };
        var out: [widths]verifier.Verified = undefined;
        sw = io_util.Stopwatch.init(io);
        try verifier.verifyGroups(target, groups[0..width], &taps, .affine_rows_ffn, out[0..width]);
        if (round > 0) rows_ns[width] += sw.read();
        for (out[0..width]) |*v| v.deinit();
    };
    const n: f64 = @floatFromInt(@max(rounds, 1));
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (1..widths + 1) |width| {
        const serial = @as(f64, @floatFromInt(serial_ns[width])) / n / 1e6;
        const rows = @as(f64, @floatFromInt(rows_ns[width])) / n / 1e6;
        const w: f64 = @floatFromInt(width);
        sx += w;
        sy += rows;
        sxx += w * w;
        sxy += w * rows;
        log.info("[glm-rows-ubench] ctx={d} B={d} serial {d:.2} ms ({d:.2} ms/row) grouped {d:.2} ms ({d:.2} ms/row)\n", .{ ctx, width, serial, serial / w, rows, rows / w });
    }
    const k: f64 = widths;
    const per_row = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    log.info("[glm-rows-ubench] grouped fit: fixed {d:.2} ms + {d:.2} ms per row ({d} rounds)\n", .{ (sy - per_row * sx) / k, per_row, rounds });
}
