//! Texas's private native control page and identity-preserving SSD parking.
const std = @import("std");
const mlx = @import("mlx.zig");
pub threadlocal var request_background: bool = false;
var page: ?*anyopaque = null;
pub var nested: bool = false; // Inference owner only.
pub const Field = enum(c_uint) { seq, hold, state, ack, bytes, pauses, tokens, prompt, phase, park_error, restore_wait, freed, available };
extern "c" fn texas_ipc_open([*:0]const u8, c_int) ?*anyopaque;
extern "c" fn texas_ipc_get(?*anyopaque, c_uint) u64;
extern "c" fn texas_ipc_set(?*anyopaque, c_uint, u64) void;
pub extern "c" fn texas_available() u64;
extern "c" fn texas_park([*]const ?*anyopaque, usize, [*]const ?*anyopaque, usize, [*:0]const u8, usize) ?*anyopaque;
extern "c" fn texas_park_bytes(?*anyopaque) usize;
extern "c" fn texas_restore(?*anyopaque) c_int;
extern "c" fn texas_park_free(?*anyopaque) void;
extern "c" fn texas_park_error() [*:0]const u8;
pub extern "c" fn texas_rng_save() ?*anyopaque;
pub extern "c" fn texas_rng_restore(?*anyopaque) void;

pub fn open() void {
    if (std.c.getenv("TEXAS_PRIORITY_IPC")) |path| {
        page = texas_ipc_open(path, 0) orelse @panic("Texas private IPC unavailable");
    }
}
pub fn enabled() bool { return page != null; }
pub fn get(f: Field) u64 { return texas_ipc_get(page, @intFromEnum(f)); }
pub fn set(f: Field, v: u64) void { texas_ipc_set(page, @intFromEnum(f), v); }
pub fn held() bool { return get(.hold) != 0; }
pub fn drained() void { set(.ack, get(.seq)); }
pub fn contextFits(prompt: usize, requested: u32) bool {
    return prompt +| @as(usize, requested) <= 262144;
}
pub fn admissionFits(retained: u64, available: u64) bool {
    return retained <= 24 * 1024 * 1024 * 1024 and available >= retained +| (12 * 1024 * 1024 * 1024);
}
pub fn restoreFits(available: u64, parked: u64) bool {
    return available >= parked +| (12 * 1024 * 1024 * 1024);
}

pub const Roots = struct {
    values: std.ArrayList(?*anyopaque) = .empty,
    allocator: std.mem.Allocator,
    pub fn deinit(self: *Roots) void {
        for (self.values.items) |ctx| _ = mlx.mlx_array_free(.{ .ctx = ctx });
        self.values.deinit(self.allocator);
    }
    // Only value-owned arrays/slices are traversed. Pointer-owned request/model objects
    // are collected explicitly; allocator, lock and opaque C handles are never followed.
    pub fn collect(self: *Roots, value: anytype) !void { try self.walk(value, 12); }
    fn walk(self: *Roots, value: anytype, comptime depth: usize) !void {
        const T = @TypeOf(value);
        if (T == mlx.mlx_array) {
            if (value.ctx != null) {
                var copy = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(copy);
                try mlx.check(mlx.mlx_array_set(&copy, value));
                try self.values.append(self.allocator, copy.ctx);
            }
            return;
        }
        if (depth == 0) return;
        switch (@typeInfo(T)) {
            .@"struct" => |s| inline for (s.field_names, s.field_attrs) |name, attrs| {
                if (!attrs.@"comptime") try self.walk(@field(value, name), depth - 1);
            },
            .optional => if (value) |v| try self.walk(v, depth - 1),
            .array => for (value) |v| try self.walk(v, depth - 1),
            .pointer => |p| if (p.size == .slice and comptime containsArrays(p.child, depth - 1)) {
                for (value) |v| try self.walk(v, depth - 1);
            },
            .@"union" => |u| if (u.tag_type != null) {
                switch (value) { inline else => |v| try self.walk(v, depth - 1) }
            },
            else => {},
        }
    }
};
fn containsArrays(comptime T: type, comptime depth: usize) bool {
    if (T == mlx.mlx_array) return true;
    if (depth == 0) return false;
    return switch (@typeInfo(T)) {
        .@"struct" => |s| blk: {
            inline for (s.field_types) |F| if (containsArrays(F, depth - 1)) break :blk true;
            break :blk false;
        },
        .@"union" => |u| blk: {
            inline for (u.field_types) |F| if (containsArrays(F, depth - 1)) break :blk true;
            break :blk false;
        },
        .optional => |o| containsArrays(o.child, depth - 1),
        .array => |a| containsArrays(a.child, depth - 1),
        .pointer => |p| p.size == .slice and containsArrays(p.child, depth - 1),
        else => false,
    };
}
pub const Parked = struct {
    handle: ?*anyopaque = null,
    pub fn create(roots: *Roots, protected: *Roots, id: u64) Parked {
        const dir = std.c.getenv("TEXAS_PRIORITY_CHECKPOINTS") orelse @panic("Texas checkpoint directory missing");
        var path: [4096]u8 = undefined;
        const p = std.fmt.bufPrint(path[0..path.len-1], "{s}/request-{d}.bin", .{ std.mem.sliceTo(dir, 0), id }) catch @panic("Texas checkpoint path too long");
        path[p.len] = 0;
        const h = texas_park(roots.values.items.ptr, roots.values.items.len,
            protected.values.items.ptr, protected.values.items.len, @ptrCast(p.ptr), 24 * 1024 * 1024 * 1024);
        set(.park_error, if (h == null) 1 else 0);
        if (h == null) @import("log.zig").warn("[texas-priority] parking retained original RAM: {s}\n", .{std.mem.sliceTo(texas_park_error(), 0)});
        set(.bytes, texas_park_bytes(h));
        return .{ .handle = h };
    }
    pub fn restore(self: *Parked) void {
        if (texas_restore(self.handle) != 0) {
            @import("log.zig").err("[texas-priority] restore failed; stopping before tensor reuse: {s}\n", .{std.mem.sliceTo(texas_park_error(), 0)});
            std.process.exit(70);
        }
        texas_park_free(self.handle);
        self.handle = null;
        set(.bytes, 0);
    }
};

test "Texas roots retain private value-owned arrays without following opaque pointers" {
    var r: Roots = .{ .allocator = std.testing.allocator };
    defer r.deinit();
    const a = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(a);
    const dummy = .{ .a = a, .opt = @as(?mlx.mlx_array, a), .slice = @as([]const mlx.mlx_array, &.{a}), .handle = @as(*const mlx.mlx_array, &a) };
    try r.collect(dummy);
    try std.testing.expect(r.values.items.len == 3);
}

test "Texas context rejects overflow before an API can clamp the requested completion" {
    try std.testing.expect(contextFits(262140,4));
    try std.testing.expect(!contextFits(262140,5));
    try std.testing.expect(!contextFits(std.math.maxInt(usize),1));
}
test "Texas admission and restore require their full reserve without a timeout escape" {
    const gib: u64 = 1024 * 1024 * 1024;
    try std.testing.expect(admissionFits(24*gib,36*gib));
    try std.testing.expect(!admissionFits(24*gib+1,128*gib));
    try std.testing.expect(!admissionFits(4*gib,16*gib-1));
    try std.testing.expect(restoreFits(16*gib,4*gib));
    try std.testing.expect(!restoreFits(16*gib-1,4*gib));
    try std.testing.expect(!restoreFits(128*gib,std.math.maxInt(u64)));
}
