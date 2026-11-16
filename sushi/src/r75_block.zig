//! texas-split R75: block verification (Sun et al. 2024, "Block Verification Accelerates Speculative Decoding",
//! arXiv 2403.10444, Algorithm 2) for the host-side acceptance of a sampled draft chain. Lossless: the emitted tokens
//! follow the target's sampling distribution exactly (their Theorem 1), and in expectation it accepts at least as many
//! drafts as token-by-token verification (Theorem 2). Pure functions on small supports; the tests below check the
//! distribution of emitted sequences statistically, including prefix-dependent targets and drafters.
//!
//! Notation, for a round with m drafts x_1..x_m drawn from the drafter:
//!   row r (r = 0..m)      the target's sampling distribution after accepting r drafts (candidate ids and probabilities)
//!   depth r (r = 0..m-1)  the drafter's distribution x_{r+1} was drawn from (q at depth r)
//!   p_0 = 1, p_i = min(p_{i-1} * row_{i-1}(x_i) / depth_{i-1}(x_i), 1)
//!   h_i = S_i / (S_i + 1 - p_i) with S_i = sum_x max(p_i * row_i(x) - depth_i(x), 0)   (i < m),   h_m = p_m
//!   tau = the last i with eta_i <= h_i (0 if none); the correction token comes from max(p_tau * row_tau - depth_tau, 0)
//!   normalised, or from row_m itself when tau = m.
const std = @import("std");

pub const MAX_DEPTH = 16;
pub const MAX_CANDS = 32;

/// One target row: up to MAX_CANDS candidate ids with their sampling probabilities (summing to 1; zeros allowed).
pub const Row = struct {
    ids: []const u32,
    p: []const f64,
};

pub const Decision = struct {
    /// accepted drafts
    tau: usize,
    /// p_tau, the scale of the target row in the correction token's residual (1 when tau = 0 or tau = m)
    scale: f64,
};

/// The accepted count. `acc_p[k]` / `acc_q[k]`: row k's and depth k's probability of draft k; `rows[0..m]` the target
/// rows; `qrow[r][j]` depth r's probability of `rows[r].ids[j]` (r < m); `eta[0..m]` uniform draws in [0, 1).
pub fn decide(m: usize, acc_p: []const f64, acc_q: []const f64, rows: []const Row, qrow: []const []const f64, eta: []const f64) Decision {
    if (m == 0) return .{ .tau = 0, .scale = 1 };
    var pcum: [MAX_DEPTH + 1]f64 = undefined;
    pcum[0] = 1;
    for (0..m) |k| {
        const ratio = if (acc_q[k] > 0) acc_p[k] / acc_q[k] else (if (acc_p[k] > 0) std.math.inf(f64) else 0);
        pcum[k + 1] = @min(pcum[k] * ratio, 1);
    }
    var tau: usize = 0;
    for (1..m + 1) |i| {
        const h: f64 = if (i == m) pcum[m] else blk: {
            var s: f64 = 0;
            for (rows[i].p, 0..) |pv, j| s += @max(pcum[i] * pv - qrow[i][j], 0);
            const den = s + 1 - pcum[i];
            break :blk if (den <= 0) 1 else s / den;
        };
        if (eta[i - 1] <= h) tau = i;
    }
    return .{ .tau = tau, .scale = if (tau == m) 1 else pcum[tau] };
}

/// The correction token: from row `tau` scaled by `scale` minus the drafter's depth `tau` (`q`, aligned with the row's
/// candidates; null at tau = m, where the row itself is sampled). `u` uniform in [0, 1). Falls back to the row itself
/// when no residual mass is left (a numerical corner).
pub fn correction(row: Row, scale: f64, q: ?[]const f64, u: f64) u32 {
    var w: [MAX_CANDS]f64 = undefined;
    var tot: f64 = 0;
    for (row.p, 0..) |pv, j| {
        w[j] = if (q) |qq| @max(scale * pv - qq[j], 0) else pv;
        tot += w[j];
    }
    if (!(tot > 0)) {
        tot = 0;
        for (row.p, 0..) |pv, j| {
            w[j] = pv;
            tot += pv;
        }
    }
    const x = u * tot;
    var acc: f64 = 0;
    var pick: usize = 0;
    for (0..row.p.len) |j| {
        if (w[j] <= 0) continue;
        pick = j;
        acc += w[j];
        if (x < acc) break;
    }
    return row.ids[pick];
}

// ---------------------------------------------------------------------------------------------------------------
// Tests: emitted sequences through draft-then-verify have the target's joint distribution.

const V = 5;

fn sampleFrom(dist: []const f64, rng: std.Random) u32 {
    const u = rng.float(f64);
    var acc: f64 = 0;
    for (dist, 0..) |d, i| {
        acc += d;
        if (u < acc) return @intCast(i);
    }
    return @intCast(dist.len - 1);
}

/// Markov target and drafter over V tokens: P[prev][x], Q[prev][x]. Generates `len` tokens after context token `c0`
/// by rounds of `m` drafts with block verification (or token verification when `block` is false).
fn generate(P: [V][V]f64, Q: [V][V]f64, c0: u32, m: usize, len: usize, block: bool, rng: std.Random, out: []u32) void {
    var n: usize = 0;
    var last = c0;
    var ids_all: [V]u32 = undefined;
    for (0..V) |i| ids_all[i] = @intCast(i);
    while (n < len) {
        var drafts: [MAX_DEPTH]u32 = undefined;
        var prev = last;
        for (0..m) |k| {
            drafts[k] = sampleFrom(&Q[prev], rng);
            prev = drafts[k];
        }
        var rows: [MAX_DEPTH + 1]Row = undefined;
        var qrows: [MAX_DEPTH][]const f64 = undefined;
        var acc_p: [MAX_DEPTH]f64 = undefined;
        var acc_q: [MAX_DEPTH]f64 = undefined;
        var ctx = last;
        for (0..m + 1) |r| {
            rows[r] = .{ .ids = &ids_all, .p = &P[ctx] };
            if (r < m) {
                qrows[r] = &Q[ctx];
                acc_p[r] = P[ctx][drafts[r]];
                acc_q[r] = Q[ctx][drafts[r]];
                ctx = drafts[r];
            }
        }
        var tau: usize = 0;
        var scale: f64 = 1;
        if (block) {
            var eta: [MAX_DEPTH]f64 = undefined;
            for (0..m) |k| eta[k] = rng.float(f64);
            const d = decide(m, acc_p[0..m], acc_q[0..m], rows[0 .. m + 1], qrows[0..m], eta[0..m]);
            tau = d.tau;
            scale = d.scale;
        } else {
            while (tau < m and rng.float(f64) < @min(1.0, acc_p[tau] / acc_q[tau])) tau += 1;
        }
        for (0..tau) |k| {
            if (n < len) out[n] = drafts[k];
            n += 1;
        }
        const y = correction(rows[tau], scale, if (tau < m) qrows[tau] else null, rng.float(f64));
        if (n < len) out[n] = y;
        n += 1;
        last = if (n >= 1 and n <= len) out[n - 1] else y;
    }
}

fn chiSquareOk(P: [V][V]f64, Q: [V][V]f64, m: usize, block: bool, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const N: usize = 300_000;
    var counts: [V * V * V]u64 = @splat(0);
    var seq: [3]u32 = undefined;
    for (0..N) |_| {
        generate(P, Q, 2, m, 3, block, rng, &seq);
        counts[(seq[0] * V + seq[1]) * V + seq[2]] += 1;
    }
    var chi: f64 = 0;
    var cells: usize = 0;
    for (0..V) |a| for (0..V) |b| for (0..V) |c| {
        const pr = P[2][a] * P[a][b] * P[b][c];
        const e = pr * @as(f64, @floatFromInt(N));
        if (e < 5) continue;
        const o: f64 = @floatFromInt(counts[(a * V + b) * V + c]);
        chi += (o - e) * (o - e) / e;
        cells += 1;
    };
    // Chi-square with (cells - 1) degrees of freedom: mean df, sd sqrt(2 df). Allow 5 sd.
    const df: f64 = @floatFromInt(cells - 1);
    try std.testing.expect(chi < df + 5 * @sqrt(2 * df));
}

const P_TEST: [V][V]f64 = .{
    .{ 0.40, 0.25, 0.15, 0.12, 0.08 },
    .{ 0.10, 0.50, 0.20, 0.10, 0.10 },
    .{ 0.30, 0.10, 0.30, 0.20, 0.10 },
    .{ 0.05, 0.15, 0.25, 0.35, 0.20 },
    .{ 0.20, 0.20, 0.20, 0.20, 0.20 },
};
const Q_TEST: [V][V]f64 = .{
    .{ 0.10, 0.40, 0.30, 0.10, 0.10 },
    .{ 0.30, 0.20, 0.20, 0.20, 0.10 },
    .{ 0.25, 0.25, 0.10, 0.30, 0.10 },
    .{ 0.40, 0.10, 0.10, 0.10, 0.30 },
    .{ 0.05, 0.05, 0.60, 0.20, 0.10 },
};

test "block verification emits the target's joint distribution (Markov target and drafter, 3 drafts)" {
    try chiSquareOk(P_TEST, Q_TEST, 3, true, 1);
}

test "block verification emits the target's joint distribution (1 and 5 drafts)" {
    try chiSquareOk(P_TEST, Q_TEST, 1, true, 2);
    try chiSquareOk(P_TEST, Q_TEST, 5, true, 3);
}

test "token verification (the reference) passes the same check" {
    try chiSquareOk(P_TEST, Q_TEST, 3, false, 4);
}

test "a drafter with zeros where the target has mass, and the reverse" {
    var Q2 = Q_TEST;
    Q2[2] = .{ 0.5, 0.5, 0, 0, 0 };
    var P2 = P_TEST;
    P2[0] = .{ 0, 0, 0.5, 0.5, 0 };
    try chiSquareOk(P2, Q2, 3, true, 5);
}

test "block verification accepts at least as many drafts as token verification on average" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    var seq: [64]u32 = undefined;
    // rounds per 64 tokens, block vs token: fewer rounds means more tokens per round
    var rounds_tok: u64 = 0;
    var rounds_blk: u64 = 0;
    for (0..2000) |_| {
        rounds_tok += countRounds(P_TEST, Q_TEST, 3, false, rng, &seq);
        rounds_blk += countRounds(P_TEST, Q_TEST, 3, true, rng, &seq);
    }
    try std.testing.expect(rounds_blk <= rounds_tok);
}

fn countRounds(P: [V][V]f64, Q: [V][V]f64, m: usize, block: bool, rng: std.Random, out: []u32) u64 {
    // the number of rounds `generate` needs for out.len tokens: run it round by round
    var rounds: u64 = 0;
    var produced: usize = 0;
    var tmp: [MAX_DEPTH + 1]u32 = undefined;
    var last: u32 = 2;
    while (produced < out.len) {
        // one round of up to m + 1 tokens from context `last`
        var one: [MAX_DEPTH + 1]u32 = undefined;
        const got = oneRound(P, Q, last, m, block, rng, &one);
        for (0..got) |k| {
            if (produced < out.len) out[produced] = one[k];
            produced += 1;
        }
        last = one[got - 1];
        rounds += 1;
        _ = &tmp;
    }
    return rounds;
}

fn oneRound(P: [V][V]f64, Q: [V][V]f64, c0: u32, m: usize, block: bool, rng: std.Random, out: []u32) usize {
    var ids_all: [V]u32 = undefined;
    for (0..V) |i| ids_all[i] = @intCast(i);
    var drafts: [MAX_DEPTH]u32 = undefined;
    var prev = c0;
    for (0..m) |k| {
        drafts[k] = sampleFrom(&Q[prev], rng);
        prev = drafts[k];
    }
    var rows: [MAX_DEPTH + 1]Row = undefined;
    var qrows: [MAX_DEPTH][]const f64 = undefined;
    var acc_p: [MAX_DEPTH]f64 = undefined;
    var acc_q: [MAX_DEPTH]f64 = undefined;
    var ctx = c0;
    for (0..m + 1) |r| {
        rows[r] = .{ .ids = &ids_all, .p = &P[ctx] };
        if (r < m) {
            qrows[r] = &Q[ctx];
            acc_p[r] = P[ctx][drafts[r]];
            acc_q[r] = Q[ctx][drafts[r]];
            ctx = drafts[r];
        }
    }
    var tau: usize = 0;
    var scale: f64 = 1;
    if (block) {
        var eta: [MAX_DEPTH]f64 = undefined;
        for (0..m) |k| eta[k] = rng.float(f64);
        const d = decide(m, acc_p[0..m], acc_q[0..m], rows[0 .. m + 1], qrows[0..m], eta[0..m]);
        tau = d.tau;
        scale = d.scale;
    } else {
        while (tau < m and rng.float(f64) < @min(1.0, acc_p[tau] / acc_q[tau])) tau += 1;
    }
    for (0..tau) |k| out[k] = drafts[k];
    out[tau] = correction(rows[tau], scale, if (tau < m) qrows[tau] else null, rng.float(f64));
    return tau + 1;
}
