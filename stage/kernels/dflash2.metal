// R58 step 2: DFlash2 drafter on Metal (the target's GPU). The math follows the CPU reference
// (src/dflash_body.rs, dflash_layer.rs, dflash_mlp.rs, dflash_attention.rs, dflash_context.rs)
// with FP32 arithmetic inside each kernel and BF16 at every stage boundary. The linear layers
// run on the target's affine8 dense kernels (weights quantized at load, as Sushi does).
// Geometry: H 2560, 32 query heads and 8 KV heads of 128, MLP 7680, conv group 16 (640 dynamic
// coefficients per row), context features 5 x 2560, at most 16 rows.
#include <metal_stdlib>
using namespace metal;

struct DfParams {
    uint rows;
    uint n;
    uint a;
    uint b;
    uint c;
    uint d;
};

// RMS norm of each row (width n), eps 1e-6, times the BF16 weight. One 256-thread group per row.
kernel void df_rms(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                   device bfloat* y [[buffer(2)]], constant DfParams& p [[buffer(3)]],
                   uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const device bfloat* xr = x + row * p.n;
    float s = 0.f;
    for (uint i = t; i < p.n; i += 256) {
        float v = float(xr[i]);
        s += v * v;
    }
    s = simd_sum(s);
    if (lane == 0) {
        part[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (uint i = 0; i < 8; i++) {
        tot += part[i];
    }
    float inv = 1.0f / sqrt(tot / float(p.n) + 1e-6f);
    for (uint i = t; i < p.n; i += 256) {
        y[row * p.n + i] = bfloat(float(xr[i]) * inv * float(w[i]));
    }
}

// Learned 2-tap causal convolution over the block's rows (zero before row 0). The coefficient of
// (set, tap, channel c) for row r is base[(2 set + tap) H + c] + dyn[r][(2 set + tap) groups + c / group].
// With has_residual (c): y = bf16(res + bf16(conv)), else y = bf16(conv).
// n = H, a = group, b = set, c = has_residual. Grid (H, rows).
kernel void df_conv(const device bfloat* x [[buffer(0)]], const device bfloat* dyn [[buffer(1)]],
                    const device bfloat* base [[buffer(2)]], const device bfloat* res [[buffer(3)]],
                    device bfloat* y [[buffer(4)]], constant DfParams& p [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, r = gid.y;
    if (c >= p.n || r >= p.rows) {
        return;
    }
    uint groups = p.n / p.a;
    float v = 0.f;
    for (uint tap = 0; tap < 2; tap++) {
        if (r < tap) {
            continue;
        }
        float z = float(x[(r - tap) * p.n + c]);
        float k = float(base[(2 * p.b + tap) * p.n + c]) + float(dyn[(r * 4 + p.b * 2 + tap) * groups + c / p.a]);
        v += z * k;
    }
    float o = float(bfloat(v));
    if (p.c != 0) {
        o = float(res[r * p.n + c]) + o;
    }
    y[r * p.n + c] = bfloat(o);
}

// Per-head RMS norm (eps 1e-6, weight [128]) then rotary embedding on pairs (i, i + 64) at
// position a + row, with the reference's FP32 frequencies (buffer 3). x, y: [rows][heads][128].
// n = heads, a = first position. One simdgroup per (head, row).
kernel void df_normrope(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                        device bfloat* y [[buffer(2)]], const device float* freq [[buffer(3)]],
                        constant DfParams& p [[buffer(4)]],
                        uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = g.x, row = g.y;
    const device bfloat* xr = x + (row * p.n + head) * 128;
    float v0 = float(xr[lane]), v1 = float(xr[lane + 32]), v2 = float(xr[lane + 64]), v3 = float(xr[lane + 96]);
    float s = simd_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
    float inv = 1.0f / sqrt(s / 128.f + 1e-6f);
    float n0 = float(bfloat(v0 * inv * float(w[lane])));
    float n1 = float(bfloat(v1 * inv * float(w[lane + 32])));
    float n2 = float(bfloat(v2 * inv * float(w[lane + 64])));
    float n3 = float(bfloat(v3 * inv * float(w[lane + 96])));
    float pos = float(p.a + row);
    device bfloat* yr = y + (row * p.n + head) * 128;
    // pair (lane, lane + 64) and pair (lane + 32, lane + 96)
    float a0 = pos * freq[lane];
    float a1 = pos * freq[lane + 32];
    float s0 = precise::sin(a0), c0 = precise::cos(a0);
    float s1 = precise::sin(a1), c1 = precise::cos(a1);
    yr[lane] = bfloat(n0 * c0 - n2 * s0);
    yr[lane + 64] = bfloat(n0 * s0 + n2 * c0);
    yr[lane + 32] = bfloat(n1 * c1 - n3 * s1);
    yr[lane + 96] = bfloat(n1 * s1 + n3 * c1);
}

// Grouped-query attention of the block's rows (positions anchor + row; queries normed and
// rotated) over the context ring (positions [ctx_lo, anchor), slot = position % ring) and the
// block's own keys (positions anchor + j, all rows: non-causal), masked to |q - k| < window.
// q, out: [rows][32][128]; kb, vb: [rows][8][128]; ck, cv: [ring][8][128].
// n = ctx_lo, a = anchor, b = window, c = ring. One simdgroup per (query head, row).
kernel void df_attn(const device bfloat* q [[buffer(0)]], const device bfloat* kb [[buffer(1)]],
                    const device bfloat* vb [[buffer(2)]], const device bfloat* ck [[buffer(3)]],
                    const device bfloat* cv [[buffer(4)]], device bfloat* out [[buffer(5)]],
                    constant DfParams& p [[buffer(6)]],
                    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = g.x, row = g.y, kvh = head / 4;
    const device bfloat* qr = q + (row * 32 + head) * 128;
    float q0 = float(qr[lane]), q1 = float(qr[lane + 32]), q2 = float(qr[lane + 64]), q3 = float(qr[lane + 96]);
    const float scale = 1.0f / sqrt(128.0f);
    uint qpos = p.a + row;
    float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    uint lo = qpos + 1 > p.b ? qpos + 1 - p.b : 0u;
    lo = max(lo, p.n);
    for (uint pos = lo; pos < p.a; pos++) {
        uint slot = pos % p.c;
        const device bfloat* kr = ck + (slot * 8 + kvh) * 128;
        float s = simd_sum(q0 * float(kr[lane]) + q1 * float(kr[lane + 32]) + q2 * float(kr[lane + 64]) + q3 * float(kr[lane + 96])) * scale;
        float mn = max(m, s);
        float corr = exp(m - mn), e = exp(s - mn);
        const device bfloat* vr = cv + (slot * 8 + kvh) * 128;
        l = l * corr + e;
        o0 = o0 * corr + e * float(vr[lane]);
        o1 = o1 * corr + e * float(vr[lane + 32]);
        o2 = o2 * corr + e * float(vr[lane + 64]);
        o3 = o3 * corr + e * float(vr[lane + 96]);
        m = mn;
    }
    for (uint j = 0; j < p.rows; j++) {
        uint kpos = p.a + j;
        uint dist = kpos > qpos ? kpos - qpos : qpos - kpos;
        if (dist >= p.b) {
            continue;
        }
        const device bfloat* kr = kb + (j * 8 + kvh) * 128;
        float s = simd_sum(q0 * float(kr[lane]) + q1 * float(kr[lane + 32]) + q2 * float(kr[lane + 64]) + q3 * float(kr[lane + 96])) * scale;
        float mn = max(m, s);
        float corr = exp(m - mn), e = exp(s - mn);
        const device bfloat* vr = vb + (j * 8 + kvh) * 128;
        l = l * corr + e;
        o0 = o0 * corr + e * float(vr[lane]);
        o1 = o1 * corr + e * float(vr[lane + 32]);
        o2 = o2 * corr + e * float(vr[lane + 64]);
        o3 = o3 * corr + e * float(vr[lane + 96]);
        m = mn;
    }
    device bfloat* yr = out + (row * 32 + head) * 128;
    float il = 1.0f / l;
    yr[lane] = bfloat(o0 * il);
    yr[lane + 32] = bfloat(o1 * il);
    yr[lane + 64] = bfloat(o2 * il);
    yr[lane + 96] = bfloat(o3 * il);
}

// y = bf16(gate * sigmoid(gate) * up), n = element count.
kernel void df_swiglu(const device bfloat* gt [[buffer(0)]], const device bfloat* up [[buffer(1)]],
                      device bfloat* y [[buffer(2)]], constant DfParams& p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    float gv = float(gt[i]);
    y[i] = bfloat(gv / (1.0f + exp(-gv)) * float(up[i]));
}

// Context rows -> the layer's K/V rings: row r goes to slot (a + r) % c. k, v: [rows][1024].
kernel void df_ring(const device bfloat* k [[buffer(0)]], const device bfloat* v [[buffer(1)]],
                    device bfloat* ck [[buffer(2)]], device bfloat* cv [[buffer(3)]],
                    constant DfParams& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, r = gid.y;
    if (c >= 1024 || r >= p.rows) {
        return;
    }
    uint slot = (p.a + r) % p.c;
    ck[slot * 1024 + c] = k[r * 1024 + c];
    cv[slot * 1024 + c] = v[r * 1024 + c];
}

// DFlash2 target features: one tap layer's 4 HC streams (rows x 4 x 2560) -> their FP32 mean,
// BF16, at column a (tap x 2560) of the 12800-wide feature rows. Grid (2560, rows).
kernel void df_features(const device bfloat* s [[buffer(0)]], device bfloat* f [[buffer(1)]],
                        constant DfParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, r = gid.y;
    if (c >= 2560 || r >= p.rows) {
        return;
    }
    const device bfloat* sr = s + r * 10240;
    float sum = float(sr[c]) + float(sr[2560 + c]) + float(sr[5120 + c]) + float(sr[7680 + c]);
    f[r * 12800 + p.a + c] = bfloat(sum * 0.25f);
}

// Word copy (buffer offsets select the rows): d[i] = s[i] for i < n.
kernel void df_copy(const device uint* s [[buffer(0)]], device uint* d [[buffer(1)]],
                    constant DfParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        d[i] = s[i];
    }
}

// df_attn with the keys split across the threadgroup's simdgroups (each keeps its own online
// softmax over a contiguous slice: context keys first, then the block's keys), merged in
// threadgroup memory. Same parameters and buffers as df_attn; threadgroup = 32 x simdgroups.
kernel void df_attn_split(const device bfloat* q [[buffer(0)]], const device bfloat* kb [[buffer(1)]],
                          const device bfloat* vb [[buffer(2)]], const device bfloat* ck [[buffer(3)]],
                          const device bfloat* cv [[buffer(4)]], device bfloat* out [[buffer(5)]],
                          constant DfParams& p [[buffer(6)]],
                          uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tm[32], tl[32], to[32][128];
    uint head = g.x, row = g.y, kvh = head / 4;
    const device bfloat* qr = q + (row * 32 + head) * 128;
    float q0 = float(qr[lane]), q1 = float(qr[lane + 32]), q2 = float(qr[lane + 64]), q3 = float(qr[lane + 96]);
    const float scale = 1.0f / sqrt(128.0f);
    uint qpos = p.a + row;
    uint lo = qpos + 1 > p.b ? qpos + 1 - p.b : 0u;
    lo = max(lo, p.n);
    uint nctx = p.a > lo ? p.a - lo : 0u;
    uint total = nctx + p.rows;
    uint chunk = (total + nsg - 1) / nsg;
    uint k0 = sg * chunk, k1 = min(total, k0 + chunk);
    float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    for (uint kk = k0; kk < k1; kk++) {
        const device bfloat* kr;
        const device bfloat* vr;
        if (kk < nctx) {
            uint slot = (lo + kk) % p.c;
            kr = ck + (slot * 8 + kvh) * 128;
            vr = cv + (slot * 8 + kvh) * 128;
        } else {
            uint j = kk - nctx;
            uint kpos = p.a + j;
            uint dist = kpos > qpos ? kpos - qpos : qpos - kpos;
            if (dist >= p.b) {
                continue;
            }
            kr = kb + (j * 8 + kvh) * 128;
            vr = vb + (j * 8 + kvh) * 128;
        }
        float s = simd_sum(q0 * float(kr[lane]) + q1 * float(kr[lane + 32]) + q2 * float(kr[lane + 64]) + q3 * float(kr[lane + 96])) * scale;
        float mn = max(m, s);
        float corr = exp(m - mn), e = exp(s - mn);
        l = l * corr + e;
        o0 = o0 * corr + e * float(vr[lane]);
        o1 = o1 * corr + e * float(vr[lane + 32]);
        o2 = o2 * corr + e * float(vr[lane + 64]);
        o3 = o3 * corr + e * float(vr[lane + 96]);
        m = mn;
    }
    if (lane == 0) {
        tm[sg] = m;
        tl[sg] = l;
    }
    to[sg][lane] = o0;
    to[sg][lane + 32] = o1;
    to[sg][lane + 64] = o2;
    to[sg][lane + 96] = o3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float mx = -INFINITY;
    for (uint s = 0; s < nsg; s++) {
        mx = max(mx, tm[s]);
    }
    float lt = 0.f, r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
    for (uint s = 0; s < nsg; s++) {
        float w = tm[s] == -INFINITY ? 0.f : exp(tm[s] - mx);
        lt += tl[s] * w;
        r0 += to[s][lane] * w;
        r1 += to[s][lane + 32] * w;
        r2 += to[s][lane + 64] * w;
        r3 += to[s][lane + 96] * w;
    }
    device bfloat* yr = out + (row * 32 + head) * 128;
    float il = 1.0f / lt;
    yr[lane] = bfloat(r0 * il);
    yr[lane + 32] = bfloat(r1 * il);
    yr[lane + 64] = bfloat(r2 * il);
    yr[lane + 96] = bfloat(r3 * il);
}
