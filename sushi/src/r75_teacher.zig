//! texas-split R75 (checklist 0.2): a fidelity panel through the production decode path, debug only.
//!
//! With SUSHI_R75_TEACHER=<jsonl> (one line per panel prompt: {"id":"code-01","prompt_ids":[...],"cont_ids":[...]})
//! and SUSHI_R75_TEACHER_OUT=<dir>, a request whose full prompt equals a panel prompt runs "teacher-forced": every
//! token the sampler would choose is replaced by the teacher's token at that position (the greedy commit path takes
//! the teacher's ids where it would take the verify argmax), so the real draft head, verify forwards, rollback, split
//! prefill, cache restores and stage slots all follow the teacher's history exactly. The logits row that committed
//! each position is written to <dir>/<id>.<n>.f32 (f32, full vocabulary) and its position and token to
//! <dir>/<id>.<n>.ids; n counts the prompt's occurrences, so a repeated prompt (a warm restore) gets its own file.
//! Without the environment variable nothing here runs.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const Entry = struct {
    id: []const u8,
    prompt: []const u32,
    cont: []const u32,
    seen: u32 = 0,
};

var loaded = false;
var entries: []Entry = &.{};

pub fn enabled() bool {
    return std.c.getenv("SUSHI_R75_TEACHER") != null and std.c.getenv("SUSHI_R75_TEACHER_OUT") != null;
}

fn readFileAll(alloc: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TeacherOpenFailed;
    defer _ = std.c.close(fd);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(alloc);
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const got = std.c.read(fd, &chunk, chunk.len);
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.TeacherReadFailed;
        }
        if (got == 0) break;
        try buf.appendSlice(alloc, chunk[0..@intCast(got)]);
    }
    return buf.toOwnedSlice(alloc);
}

fn idsOf(alloc: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const arr = switch (v) {
        .array => |a| a,
        else => return error.TeacherBadIds,
    };
    const out = try alloc.alloc(u32, arr.items.len);
    for (arr.items, 0..) |x, i| out[i] = switch (x) {
        .integer => |n| @intCast(n),
        else => return error.TeacherBadIds,
    };
    return out;
}

fn load() void {
    if (loaded) return;
    loaded = true;
    const alloc = std.heap.page_allocator;
    const raw = std.c.getenv("SUSHI_R75_TEACHER") orelse return;
    const path = std.mem.sliceTo(raw, 0);
    const path_z = std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0) catch return;
    const body = readFileAll(alloc, path_z) catch |e| {
        log.err("[r75-teacher] cannot read the panel: {s}\n", .{@errorName(e)});
        return;
    };
    var list: std.ArrayList(Entry) = .empty;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, line, .{}) catch {
            log.err("[r75-teacher] bad panel line\n", .{});
            continue;
        };
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => continue,
        };
        const id_v = obj.get("id") orelse continue;
        const id = switch (id_v) {
            .string => |t| alloc.dupe(u8, t) catch continue,
            else => continue,
        };
        const prompt = idsOf(alloc, obj.get("prompt_ids") orelse continue) catch continue;
        const cont = idsOf(alloc, obj.get("cont_ids") orelse continue) catch continue;
        list.append(alloc, .{ .id = id, .prompt = prompt, .cont = cont }) catch return;
    }
    entries = list.toOwnedSlice(alloc) catch &.{};
    log.info("[r75-teacher] {d} panel prompts loaded\n", .{entries.len});
}

/// The panel entry whose prompt equals this request's full prompt, if teacher mode is on.
pub fn match(full_prompt: []const u32) ?*Entry {
    if (!enabled()) return null;
    load();
    for (entries) |*e| {
        if (std.mem.eql(u32, e.prompt, full_prompt)) return e;
    }
    return null;
}

pub const Dump = struct {
    entry: *Entry,
    fd: std.c.fd_t,
    ids_fd: std.c.fd_t,
    rows: usize = 0,
    next_pos: usize = 0,
    gaps: usize = 0,

    pub fn open(entry: *Entry) ?Dump {
        const alloc = std.heap.page_allocator;
        const dir = std.mem.sliceTo(std.c.getenv("SUSHI_R75_TEACHER_OUT") orelse return null, 0);
        const n = entry.seen;
        entry.seen += 1;
        const f32_path = std.fmt.allocPrintSentinel(alloc, "{s}/{s}.{d}.f32", .{ dir, entry.id, n }, 0) catch return null;
        defer alloc.free(f32_path);
        const ids_path = std.fmt.allocPrintSentinel(alloc, "{s}/{s}.{d}.ids", .{ dir, entry.id, n }, 0) catch return null;
        defer alloc.free(ids_path);
        const fd = std.c.open(f32_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return null;
        const ids_fd = std.c.open(ids_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (ids_fd < 0) {
            _ = std.c.close(fd);
            return null;
        }
        log.info("[r75-teacher] {s}.{d}: teacher-forced, {d} prompt ids, {d} continuation ids\n", .{ entry.id, n, entry.prompt.len, entry.cont.len });
        return .{ .entry = entry, .fd = fd, .ids_fd = ids_fd };
    }

    /// The teacher's token at generated position `pos`, or null past the continuation.
    pub fn token(self: *const Dump, pos: usize) ?u32 {
        return if (pos < self.entry.cont.len) self.entry.cont[pos] else null;
    }

    fn writeAll(fd: std.c.fd_t, bytes: []const u8) void {
        var done: usize = 0;
        while (done < bytes.len) {
            const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
            if (got < 0) {
                if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
                return;
            }
            if (got == 0) return;
            done += @intCast(got);
        }
    }

    /// Write row `row` of `logits` ([1, S, V], [S, V] or [V]) as generated position `pos`, committed as `tok`.
    pub fn writeRow(self: *Dump, logits: mlx.mlx_array, row: usize, pos: usize, tok: u32, s: mlx.mlx_stream) void {
        self.writeRowInner(logits, row, pos, tok, s) catch |e| log.err("[r75-teacher] row write failed: {s}\n", .{@errorName(e)});
    }

    fn writeRowInner(self: *Dump, logits: mlx.mlx_array, row: usize, pos: usize, tok: u32, s: mlx.mlx_stream) !void {
        if (pos != self.next_pos) self.gaps += 1;
        self.next_pos = pos + 1;
        const shape = mlx.getShape(logits);
        const v: c_int = shape[shape.len - 1];
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const rows_total: c_int = @intCast(@divExact(mlx.mlx_array_size(logits), @as(usize, @intCast(v))));
        try mlx.check(mlx.mlx_reshape(&flat, logits, &[_]c_int{ rows_total, v }, 2, s));
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        const r: c_int = @intCast(row);
        try mlx.check(mlx.mlx_slice(&sliced, flat, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, v }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, sliced, .float32, s));
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
        try mlx.check(mlx.mlx_array_eval(c));
        const data = mlx.mlx_array_data_float32(c) orelse return error.TeacherRowUnreadable;
        writeAll(self.fd, std.mem.sliceAsBytes(data[0..@intCast(v)]));
        var line: [64]u8 = undefined;
        const txt = std.fmt.bufPrint(&line, "{d} {d}\n", .{ pos, tok }) catch return;
        writeAll(self.ids_fd, txt);
        self.rows += 1;
    }

    pub fn close(self: *Dump) void {
        log.info("[r75-teacher] {s}: {d} rows written, {d} position gaps\n", .{ self.entry.id, self.rows, self.gaps });
        _ = std.c.close(self.fd);
        _ = std.c.close(self.ids_fd);
    }
};
