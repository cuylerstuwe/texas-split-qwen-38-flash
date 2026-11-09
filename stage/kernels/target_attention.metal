// Source-grounded Qwen attention semantics; MLX affine quantizer adaptation
// (Copyright Apple Inc., MIT; NOTICE.txt). Native ABI and SIMD control are new.
#include <metal_stdlib>
using namespace metal;
struct AttnParams {uint offset,rows,base,capacity,sparse,splits,log2_scores;};
kernel void attn_prepare(const device bfloat* qg [[buffer(0)]],const device bfloat* k [[buffer(1)]],
    const device bfloat* qw [[buffer(2)]],const device bfloat* kw [[buffer(3)]],
    const device float* freq [[buffer(4)]],device bfloat* query [[buffer(5)]],
    device bfloat* key [[buffer(6)]],device bfloat* gate [[buffer(7)]],
    constant AttnParams& p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint r=g.y,h=g.x;bool is_q=h<24;uint kh=is_q?h:h-24;
    const device bfloat* src=is_q?qg+(r*24+h)*512:k+(r*2+kh)*256;
    const device bfloat* norm=is_q?qw:kw;float x[8],ss=0.f;
    for(uint j=0;j<8;j++){x[j]=float(src[lane*8+j]);ss+=x[j]*x[j];}
    float inv=precise::rsqrt(simd_sum(ss)/256.f+1e-6f);
    for(uint j=0;j<8;j++)x[j]=float(bfloat(float(bfloat(x[j]*inv))*float(norm[lane*8+j])));
    for(uint j=0;j<8;j++) {
        uint d=lane*8+j;float partner=simd_shuffle(x[j],lane^4u),y=x[j];
        if(d<64){float a=float(p.base+p.offset+r)*freq[d%32];y=x[j]*precise::cos(a)+(d<32?-partner:partner)*precise::sin(a);}
        if(is_q){query[(r*24+h)*256+d]=bfloat(y);gate[(r*24+h)*256+d]=src[256+d];}
        else key[(r*2+kh)*256+d]=bfloat(y);
    }
}

// One32-lane group per64values. Source FP32 edge/code decisions precede
// metadata rounding; each cache row is token-major [token,kvhead,256].
kernel void attn_pack(const device bfloat* key [[buffer(0)]],const device bfloat* value [[buffer(1)]],
    device uchar* kc [[buffer(2)]],device bfloat* ks [[buffer(3)]],device bfloat* kb [[buffer(4)]],
    device uchar* vc [[buffer(5)]],device bfloat* vs [[buffer(6)]],device bfloat* vb [[buffer(7)]],
    constant AttnParams& p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint group=g.x;bool is_k=g.y==0;uint row=group/8,within=group%8;
    const device bfloat* in=is_k?key:value;
    float a=float(in[group*64+2*lane]),b=float(in[group*64+2*lane+1]);
    float lo=simd_min(min(a,b)),hi=simd_max(max(0.f,max(a,b)));
    float scale=max((hi-lo)/255.f,1e-7f);bool side=abs(lo)>abs(hi);scale=side?scale:-scale;
    float edge=side?lo:hi,q0=round(edge/scale);bool at_zero=q0==0.f;
    scale=at_zero?scale:edge/q0;float bias=at_zero?0.f:edge;
    device uchar* codes=is_k?kc:vc;device bfloat* scales=is_k?ks:vs;device bfloat* biases=is_k?kb:vb;
    uint outgroup=(p.offset+row)*8+within;
    if(lane==0){scales[outgroup]=bfloat(scale);biases[outgroup]=bfloat(bias);}
    codes[outgroup*64+2*lane]=uchar(min(round((a-bias)/scale),255.f));
    codes[outgroup*64+2*lane+1]=uchar(min(round((b-bias)/scale),255.f));
}
inline float attn_unpack(const device uchar* c,const device bfloat* s,const device bfloat* b,uint i) {
    return float(bfloat(float(c[i])*float(s[i/64])+float(b[i/64])));
}
inline uint attn_pos(const device int* ids,uint vi,uint selected,uint tail) {
    return vi<selected?uint(ids[vi/4])*4+vi%4:tail+vi-selected;
}
// Independent scalar-SIMD attention control. Every split owns a disjoint
// virtual-key range, including its correct causal ragged tail. Empty splits
// write zero mass, so the merge never evaluates -inf - -inf.
kernel void attn_simd(const device bfloat* query [[buffer(0)]],const device uchar* kc [[buffer(1)]],
    const device uchar* vc [[buffer(2)]],const device bfloat* ks [[buffer(3)]],const device bfloat* kb [[buffer(4)]],
    const device bfloat* vs [[buffer(5)]],const device bfloat* vb [[buffer(6)]],const device int* ids [[buffer(7)]],
    device float* partial [[buffer(8)]],device float* stats [[buffer(9)]],constant AttnParams& p [[buffer(10)]],
    uint3 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint h=g.x,r=g.y,sp=g.z,kh=h/12,pos=p.offset+r,complete=(pos+1)/4;
    uint selected=p.sparse?min(complete,512u)*4:0,tail=p.sparse?complete*4:0,L=selected+pos+1-tail;
    uint per=(L+p.splits-1)/p.splits,lo=sp*per,hi=min(L,lo+per);
    float q[8],acc[8];for(uint j=0;j<8;j++){q[j]=float(query[(r*24+h)*256+lane*8+j]);acc[j]=0.f;}
    float m=-INFINITY,z=0.f;
    for(uint vi=lo;vi<hi;vi++) {
        uint key=attn_pos(ids+r*512,vi,selected,tail),at=(key*2+kh)*256+lane*8;
        float dot=0.f;for(uint j=0;j<8;j++)dot+=q[j]*attn_unpack(kc,ks,kb,at+j);
        dot=simd_sum(dot)*(1.f/16.f);float nm=max(m,dot),factor=precise::exp(m-nm),prob=precise::exp(dot-nm);z=z*factor+prob;m=nm;
        for(uint j=0;j<8;j++)acc[j]=acc[j]*factor+prob*attn_unpack(vc,vs,vb,at+j);
    }
    uint idx=(r*24+h)*p.splits+sp;
    if(lane==0){stats[idx*2]=m;stats[idx*2+1]=z;}
    for(uint j=0;j<8;j++)partial[idx*256+lane*8+j]=acc[j];
}
kernel void attn_merge(const device float* partial [[buffer(0)]],const device float* stats [[buffer(1)]],
    device bfloat* output [[buffer(2)]],constant AttnParams& p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint d [[thread_index_in_threadgroup]]) {
    uint idx=(g.y*24+g.x)*p.splits;float m=-INFINITY;
    for(uint sp=0;sp<p.splits;sp++)if(stats[(idx+sp)*2+1]>0.f)m=max(m,stats[(idx+sp)*2]);
    float z=0.f,y=0.f;
    for(uint sp=0;sp<p.splits;sp++){float mass=stats[(idx+sp)*2+1];if(mass>0.f){float delta=stats[(idx+sp)*2]-m;float w=p.log2_scores?exp2(delta):precise::exp(delta);z+=mass*w;y+=partial[(idx+sp)*256+d]*w;}}
    output[(g.y*24+g.x)*256+d]=bfloat(y/z);
}
kernel void attn_gate(const device bfloat* attention [[buffer(0)]],const device bfloat* gate [[buffer(1)]],
    device bfloat* output [[buffer(2)]],constant AttnParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*6144)return;
    float g=float(gate[i]);bfloat e=bfloat(precise::exp(abs(g)));bfloat den=bfloat(1.f+float(e));bfloat small=bfloat(1.f/float(den));
    bfloat sigmoid=g<0.f?small:bfloat(1.f-float(small));output[i]=bfloat(float(attention[i])*float(sigmoid));
}
