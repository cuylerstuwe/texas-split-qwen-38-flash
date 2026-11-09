//! The `sushi run` prompt's line input. A tty line is read with ICANON off: the kernel caps a canonical line at
//! MAX_CANON (1024 bytes on macOS), so a long paste would never submit.

const std = @import("std");

const max_line = 4 * 1024 * 1024;

/// Builds one input line byte by byte, doing the echo and editing the terminal no longer does:
/// backspace, Ctrl-U, Ctrl-D, and arrow or other escape sequences swallowed.
pub const LineEditor = struct {
    line: std.ArrayList(u8) = .empty,
    esc: enum { none, esc, csi, ss3 } = .none,

    pub const Step = enum { more, line, eof };

    pub fn deinit(e: *LineEditor, allocator: std.mem.Allocator) void {
        e.line.deinit(allocator);
    }

    /// `echo` is null when the terminal still echoes (not a tty).
    pub fn feed(e: *LineEditor, allocator: std.mem.Allocator, byte: u8, echo: ?*std.Io.Writer) !Step {
        // A control byte ends an unfinished escape sequence and is then handled as typed.
        switch (e.esc) {
            .none => {},
            .esc => {
                if (byte >= 0x20 and byte < 0x80) {
                    e.esc = switch (byte) {
                        '[' => .csi,
                        'O' => .ss3,
                        else => .none,
                    };
                    return .more;
                }
                e.esc = .none;
            },
            .csi => {
                if (byte >= 0x20 and byte < 0x80) {
                    if (byte >= 0x40) e.esc = .none;
                    return .more;
                }
                e.esc = .none;
            },
            .ss3 => {
                e.esc = .none;
                if (byte >= 0x20 and byte < 0x80) return .more;
            },
        }
        switch (byte) {
            '\n', '\r' => {
                if (echo) |w| try w.writeByte('\n');
                return .line;
            },
            0x04 => if (e.line.items.len == 0) return .eof,
            0x7f, 0x08 => _ = try e.eraseLast(echo),
            0x15 => while (try e.eraseLast(echo)) {},
            0x1b => e.esc = .esc,
            '\t' => try e.insert(allocator, byte, echo),
            0x20...0x7e, 0x80...0xff => try e.insert(allocator, byte, echo),
            else => {},
        }
        return .more;
    }

    fn insert(e: *LineEditor, allocator: std.mem.Allocator, byte: u8, echo: ?*std.Io.Writer) !void {
        if (e.line.items.len >= max_line) return;
        try e.line.append(allocator, byte);
        if (echo) |w| try w.writeByte(byte);
    }

    /// Drops the last character, a whole UTF-8 sequence; false on an empty line.
    fn eraseLast(e: *LineEditor, echo: ?*std.Io.Writer) !bool {
        while (e.line.pop()) |b| {
            if (b & 0xC0 == 0x80) continue;
            if (echo) |w| try w.writeAll("\x08 \x08");
            return true;
        }
        return false;
    }
};

/// Reads prompt lines from a file descriptor, keeping what a read returned past the line it ended.
pub const Input = struct {
    fd: std.c.fd_t = 0,
    tty: bool,
    editor: LineEditor = .{},
    chunk: [4096]u8 = undefined,
    pos: usize = 0,
    len: usize = 0,

    pub fn deinit(in: *Input, allocator: std.mem.Allocator) void {
        in.editor.deinit(allocator);
    }

    /// The next line without its newline, valid until the next call; null at the end of input.
    pub fn readLine(in: *Input, allocator: std.mem.Allocator, echo: *std.Io.Writer) !?[]const u8 {
        in.editor.line.clearRetainingCapacity();
        const raw = in.tty and rawModeOn(in.fd);
        defer if (raw) rawModeOff();
        const typed_echo: ?*std.Io.Writer = if (raw) echo else null;
        while (true) {
            if (in.pos == in.len) {
                try echo.flush();
                if (!in.fill()) return if (in.editor.line.items.len > 0) in.editor.line.items else null;
            }
            const byte = in.chunk[in.pos];
            in.pos += 1;
            switch (try in.editor.feed(allocator, byte, typed_echo)) {
                .more => {},
                .line => return in.editor.line.items,
                .eof => return null,
            }
        }
    }

    fn fill(in: *Input) bool {
        while (true) {
            const n = std.c.read(in.fd, &in.chunk, in.chunk.len);
            if (n > 0) {
                in.pos = 0;
                in.len = @intCast(n);
                return true;
            }
            if (n == 0 or std.posix.errno(n) != .INTR) return false;
        }
    }

    /// Asks `question` and takes `y` or `yes`; anything else, end of input and a non-tty are a no. Input typed
    /// before the question was shown is dropped, so it cannot answer it.
    pub fn confirm(in: *Input, allocator: std.mem.Allocator, w: *std.Io.Writer, question: []const u8) !bool {
        if (!in.tty) return false;
        in.pos = 0;
        in.len = 0;
        _ = tcflush(in.fd, tciflush);
        try w.print("{s} [y/N] ", .{question});
        try w.flush();
        const answer = (try in.readLine(allocator, w)) orelse return false;
        const t = std.mem.trim(u8, answer, " \t");
        return std.ascii.eqlIgnoreCase(t, "y") or std.ascii.eqlIgnoreCase(t, "yes");
    }
};

extern "c" fn tcflush(fd: std.c.fd_t, queue: c_int) c_int;
extern "c" fn atexit(cb: *const fn () callconv(.c) void) c_int;
const tciflush = 1;

var saved_termios: std.c.termios = undefined;
var saved_fd: std.c.fd_t = 0;
var raw_on = std.atomic.Value(bool).init(false);
var restore_registered = false;

/// Turns off the terminal's line editing and echo, keeping signals (Ctrl-C) and CR to LF. Held only while a
/// line is read; the exit hook puts the terminal back when the process ends in the middle of one.
fn rawModeOn(fd: std.c.fd_t) bool {
    var t = std.posix.tcgetattr(fd) catch return false;
    saved_termios = t;
    saved_fd = fd;
    t.lflag.ICANON = false;
    t.lflag.ECHO = false;
    t.lflag.IEXTEN = false;
    t.cc[@intFromEnum(std.posix.V.MIN)] = 1;
    t.cc[@intFromEnum(std.posix.V.TIME)] = 0;
    std.posix.tcsetattr(fd, .DRAIN, t) catch return false;
    raw_on.store(true, .release);
    if (!restore_registered) restore_registered = atexit(restoreAtExit) == 0;
    return true;
}

fn rawModeOff() void {
    if (raw_on.swap(false, .acq_rel)) std.posix.tcsetattr(saved_fd, .DRAIN, saved_termios) catch {};
}

fn restoreAtExit() callconv(.c) void {
    rawModeOff();
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

fn feedAll(e: *LineEditor, bytes: []const u8, echo: ?*std.Io.Writer) !LineEditor.Step {
    var step: LineEditor.Step = .more;
    for (bytes) |b| {
        step = try e.feed(testing.allocator, b, echo);
        if (step != .more) break;
    }
    return step;
}

test "repl input: the editor assembles a line with backspace over ascii and utf-8, kill and swallowed escapes" {
    const Case = struct { in: []const u8, out: []const u8 };
    for ([_]Case{
        .{ .in = "hello\n", .out = "hello" },
        .{ .in = "hello\r", .out = "hello" },
        .{ .in = "helo\x7f\x7f" ++ "llo\n", .out = "hello" },
        .{ .in = "ab\x08c\n", .out = "ac" },
        .{ .in = "\x7f\x7fx\n", .out = "x" },
        .{ .in = "\xc3\xa9\x7fz\n", .out = "z" },
        .{ .in = "\xe6\x97\xa5\xe6\x9c\xac\x7f\n", .out = "\xe6\x97\xa5" },
        .{ .in = "wrong\x15right\n", .out = "right" },
        .{ .in = "a\x1b[Ab\x1bOAc\x1b[1;5Cd\x1bxe\n", .out = "abcde" },
        .{ .in = "a\x03\x1a\x16b\n", .out = "ab" },
        .{ .in = "tab\there\n", .out = "tab\there" },
    }) |c| {
        var e: LineEditor = .{};
        defer e.deinit(testing.allocator);
        try testing.expectEqual(LineEditor.Step.line, try feedAll(&e, c.in, null));
        try testing.expectEqualStrings(c.out, e.line.items);
    }
}

test "repl input: Ctrl-D ends input only on an empty line" {
    var e: LineEditor = .{};
    defer e.deinit(testing.allocator);
    try testing.expectEqual(LineEditor.Step.eof, try feedAll(&e, "\x04", null));
    try testing.expectEqual(LineEditor.Step.line, try feedAll(&e, "x\x04\n", null));
    try testing.expectEqualStrings("x", e.line.items);
}

test "repl input: a line past the kernel's 1 KiB canonical limit assembles whole" {
    var e: LineEditor = .{};
    defer e.deinit(testing.allocator);
    const paste = try testing.allocator.alloc(u8, 64 * 1024);
    defer testing.allocator.free(paste);
    @memset(paste, 'x');
    try testing.expectEqual(LineEditor.Step.more, try feedAll(&e, paste, null));
    try testing.expectEqual(LineEditor.Step.line, try feedAll(&e, "\n", null));
    try testing.expectEqual(@as(usize, 64 * 1024), e.line.items.len);
}

test "repl input: the editor echoes what the terminal no longer does" {
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var e: LineEditor = .{};
    defer e.deinit(testing.allocator);
    _ = try feedAll(&e, "ab\x7f\xc3\xa9\x7f\x7f\x7f\n", &out.writer);
    try testing.expectEqualStrings("ab\x08 \x08\xc3\xa9\x08 \x08\x08 \x08\n", out.written());
}

fn pipeWith(bytes: []const u8) !std.c.fd_t {
    var fds: [2]std.c.fd_t = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    try testing.expectEqual(@as(isize, @intCast(bytes.len)), std.c.write(fds[1], bytes.ptr, bytes.len));
    _ = std.c.close(fds[1]);
    return fds[0];
}

test "repl input: lines come off a stream whole, in order, with a last line that has no newline" {
    const allocator = testing.allocator;
    var long: [8192]u8 = undefined;
    @memset(&long, 'q');
    const data = try std.mem.concat(allocator, u8, &.{ "one\ntwo\n", &long, "\nlast" });
    defer allocator.free(data);
    var in: Input = .{ .fd = try pipeWith(data), .tty = false };
    defer {
        _ = std.c.close(in.fd);
        in.deinit(allocator);
    }
    var sink: std.Io.Writer.Allocating = .init(allocator);
    defer sink.deinit();
    try testing.expectEqualStrings("one", (try in.readLine(allocator, &sink.writer)).?);
    try testing.expectEqualStrings("two", (try in.readLine(allocator, &sink.writer)).?);
    try testing.expectEqualStrings(&long, (try in.readLine(allocator, &sink.writer)).?);
    try testing.expectEqualStrings("last", (try in.readLine(allocator, &sink.writer)).?);
    try testing.expect(try in.readLine(allocator, &sink.writer) == null);
    try testing.expectEqual(@as(usize, 0), sink.written().len);
}

test "repl input: a confirmation takes y or yes and refuses anything else, end of input and a non-tty" {
    const allocator = testing.allocator;
    const Case = struct { in: []const u8, yes: bool, tty: bool = true };
    for ([_]Case{
        .{ .in = "y\n", .yes = true },
        .{ .in = " YES \n", .yes = true },
        .{ .in = "\n", .yes = false },
        .{ .in = "n\n", .yes = false },
        .{ .in = "yep\n", .yes = false },
        .{ .in = "", .yes = false },
        .{ .in = "y\n", .yes = false, .tty = false },
    }) |c| {
        var in: Input = .{ .fd = try pipeWith(c.in), .tty = c.tty };
        defer {
            _ = std.c.close(in.fd);
            in.deinit(allocator);
        }
        var sink: std.Io.Writer.Allocating = .init(allocator);
        defer sink.deinit();
        try testing.expectEqual(c.yes, try in.confirm(allocator, &sink.writer, "go?"));
    }
}
