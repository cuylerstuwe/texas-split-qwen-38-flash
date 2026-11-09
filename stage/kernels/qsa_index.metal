// Qwen QSA indexer, text positions. Sushi-derived semantic dtype boundaries
// (MIT, NOTICE.txt); native buffer ABI and ownership are independent.
#include <metal_stdlib>
using namespace metal;
struct QsaParams { uint offset, rows, base, accepted; };

inline bfloat qsa_raw(const device bfloat* qk, const device bfloat* tail,
                     uint old_tail, uint row, uint d) {
    return row < old_tail ? tail[row*128+d] : qk[(row-old_tail)*640+512+d];
}

// One SIMD group per completed block. All four source keys must precede the
// block's end; partial speculative prefixes are never pooled with future keys.
kernel void qsa_pool(const device bfloat* qk [[buffer(0)]],
    const device bfloat* tail [[buffer(1)]],const device bfloat* norm [[buffer(2)]],
    const device float* freq [[buffer(3)]],device bfloat* pooled [[buffer(4)]],
    constant QsaParams& p [[buffer(5)]],uint b [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint old=p.offset/4, rem=p.offset%4; float x[4]; float ss=0.f;
    for(uint j=0;j<4;j++) {
        uint d=lane*4+j;float m=0.f;
        for(uint t=0;t<4;t++) m+=float(qsa_raw(qk,tail,rem,4*b+t,d));
        x[j]=float(bfloat(m*.25f));ss+=x[j]*x[j];
    }
    float inv=precise::rsqrt(simd_sum(ss)/128.f+1e-6f);
    for(uint j=0;j<4;j++) x[j]=float(bfloat(float(bfloat(x[j]*inv))*float(norm[lane*4+j])));
    uint pos=p.base+(old+b)*4;
    for(uint j=0;j<4;j++) {
        uint d=lane*4+j;
        // Every lane participates in the shuffle, even outside rotary dims.
        float other=simd_shuffle(x[j],lane^8u);
        bfloat y=bfloat(x[j]);
        if(d<64) {
            float a=float(pos)*freq[d%32];
            bfloat c=bfloat(precise::cos(a)),s=bfloat(precise::sin(a));
            float signed_other=d<32?-other:other;
            y=bfloat(float(bfloat(x[j]*float(c)))+float(bfloat(signed_other*float(s))));
        }
        pooled[(old+b)*128+d]=y;
    }
}

// One SIMD group per query head. RMS arithmetic rounds before learned scale;
// query RoPE has one final BF16 boundary, unlike pooled-key BF16 trig tables.
kernel void qsa_query(const device bfloat* qk [[buffer(0)]],
    const device bfloat* norm [[buffer(1)]],const device float* freq [[buffer(2)]],
    device bfloat* query [[buffer(3)]],constant QsaParams& p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint h=g.x,r=g.y;float x[4],ss=0.f;
    for(uint j=0;j<4;j++){x[j]=float(qk[r*640+h*128+lane*4+j]);ss+=x[j]*x[j];}
    float inv=precise::rsqrt(simd_sum(ss)/128.f+1e-6f);
    for(uint j=0;j<4;j++) x[j]=float(bfloat(float(bfloat(x[j]*inv))*float(norm[lane*4+j])));
    for(uint j=0;j<4;j++) {
        uint d=lane*4+j;float other=simd_shuffle(x[j],lane^8u),y=x[j];
        if(d<64) {
            float a=float(p.base+p.offset+r)*freq[d%32];
            float c=precise::cos(a),s=precise::sin(a);
            y=x[j]*c+(d<32?-other:other)*s;
        }
        query[(r*4+h)*128+d]=bfloat(y);
    }
}

// Four independent SIMD groups per group, one pooled block each. FP32 scores
// are retained for an independent selector check. Incomplete/future blocks
// are excluded by the causal selector, never by their score magnitude.
kernel void qsa_score(const device bfloat* query [[buffer(0)]],
    const device bfloat* pooled [[buffer(1)]],device float* scores [[buffer(2)]],
    constant QsaParams& p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint b=g.x*4+sg,r=g.y,nb=(p.offset+p.rows)/4;
    if(b>=nb)return;
    float k[4];for(uint j=0;j<4;j++)k[j]=float(pooled[b*128+lane*4+j]);
    float score=0.f;
    for(uint h=0;h<4;h++) {
        float dot=0.f;
        for(uint j=0;j<4;j++)dot+=float(query[(r*4+h)*128+lane*4+j])*k[j];
        score+=max(simd_sum(dot),0.f);
    }
    if(lane==0)scores[r*nb+b]=score;
}

// Device-side raw-tail reconstruction for any accepted prefix, including zero.
// Separate destination avoids racing reads of the previous incomplete block.
// Pooled-prefix commit is metadata-only: appended blocks beyond the committed
// length remain invalid and must be overwritten by the next candidate append.
kernel void qsa_commit_tail(const device bfloat* qk [[buffer(0)]],
    const device bfloat* old_tail [[buffer(1)]],device bfloat* new_tail [[buffer(2)]],
    constant QsaParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint old=p.offset%4, n=(p.offset+p.accepted)%4;
    if(i<n*128) {
        uint row=old+p.accepted-n+i/128;
        new_tail[i]=qsa_raw(qk,old_tail,old,row,i%128);
    }
}
