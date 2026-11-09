// Routed branch transforms derived from Sushi TOKEN_PAIR_PREPARE, MID and
// REDUCE_SOURCE (MIT, NOTICE.txt). Native Rust buffers and split sums are new.
// FP32 arithmetic, FP16 prepared vectors, FP32 diagnostic and BF16 final store.
struct MoeParams { uint hidden; uint intermediate; uint topk; uint gate_splits; uint down_splits; uint pad0; uint pad1; uint pad2; };
inline float4 moe_h128(float4 v, uint lane) {
    for(ushort bit=1;bit<=16;bit<<=1) {
        float4 other=simd_shuffle_xor(v,bit);
        v=(lane&bit)==0?v+other:other-v;
    }
    float4 s=float4(v.x+v.y,v.x-v.y,v.z+v.w,v.z-v.w);
    return float4(s.x+s.z,s.y+s.w,s.x-s.z,s.y-s.w)*0.08838834764831845f;
}
inline float4 moe_half4(const device half* x,ulong base,uint lane) {
    return float4(x[base+lane],x[base+lane+32],x[base+lane+64],x[base+lane+96]);
}
inline void moe_store_half4(device half* y,ulong base,uint lane,float4 v) {
    y[base+lane]=half(v.x);y[base+lane+32]=half(v.y);
    y[base+lane+64]=half(v.z);y[base+lane+96]=half(v.w);
}
inline float4 moe_sum4(const device float* x,uint slot,uint dim,uint splits,uint base,uint lane) {
    float4 sum=0;
    for(uint s=0;s<splits;s++) {
        ulong at=(ulong(slot)*splits+s)*dim+base+lane;
        sum+=float4(x[at],x[at+32],x[at+64],x[at+96]);
    }
    return sum;
}
kernel void moe_prepare_pair(const device float* x [[buffer(0)]],
    const device half* suh_g [[buffer(1)]],const device half* suh_u [[buffer(2)]],
    const device uint* ids [[buffer(3)]],const device uint* order [[buffer(4)]],
    device half* gate [[buffer(5)]],device half* up [[buffer(6)]],constant MoeParams& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint slot=group.y,base=group.x*128;
    ulong xb=ulong(order[slot]/p.topk)*p.hidden+base;
    ulong sb=ulong(ids[slot])*p.hidden+base,yb=ulong(slot)*p.hidden+base;
    float4 v=float4(x[xb+lane],x[xb+lane+32],x[xb+lane+64],x[xb+lane+96]);
    moe_store_half4(gate,yb,lane,moe_h128(v*moe_half4(suh_g,sb,lane),lane));
    moe_store_half4(up,yb,lane,moe_h128(v*moe_half4(suh_u,sb,lane),lane));
}
kernel void moe_mid(const device float* gate [[buffer(0)]],const device float* up [[buffer(1)]],
    const device half* svh_g [[buffer(2)]],const device half* svh_u [[buffer(3)]],
    const device half* suh_d [[buffer(4)]],const device uint* ids [[buffer(5)]],
    device half* down_input [[buffer(6)]],constant MoeParams& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint slot=group.y,base=group.x*128;
    ulong sb=ulong(ids[slot])*p.intermediate+base,yb=ulong(slot)*p.intermediate+base;
    float4 g=moe_h128(moe_sum4(gate,slot,p.intermediate,p.gate_splits,base,lane),lane)*moe_half4(svh_g,sb,lane);
    float4 u=moe_h128(moe_sum4(up,slot,p.intermediate,p.gate_splits,base,lane),lane)*moe_half4(svh_u,sb,lane);
    float4 ex=exp(-abs(g)),sig=select(ex/(1.0f+ex),1.0f/(1.0f+ex),g>=0.0f);
    float4 v=g*sig*u*moe_half4(suh_d,sb,lane);
    moe_store_half4(down_input,yb,lane,moe_h128(v,lane));
}
kernel void moe_finish(const device float* down [[buffer(0)]],const device half* svh_d [[buffer(1)]],
    const device uint* ids [[buffer(2)]],const device uint* inverse [[buffer(3)]],
    const device float* scores [[buffer(4)]],device float* out [[buffer(5)]],
    device ushort* out_bf16 [[buffer(6)]],constant MoeParams& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float vals[10*128]; // fixed model top-k, validated by Rust
    uint row=group.y,base=group.x*128,slot=inverse[row*p.topk+sg];
    ulong sb=ulong(ids[slot])*p.hidden+base;
    float4 v=moe_h128(moe_sum4(down,slot,p.hidden,p.down_splits,base,lane),lane)*moe_half4(svh_d,sb,lane);
    vals[sg*128+lane]=v.x;vals[sg*128+lane+32]=v.y;
    vals[sg*128+lane+64]=v.z;vals[sg*128+lane+96]=v.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float4 sum=0;
        for(uint k=0;k<p.topk;k++) {
            ulong b=k*128+lane;
            sum+=float4(vals[b],vals[b+32],vals[b+64],vals[b+96])*scores[row*p.topk+k];
        }
        for(uint j=0;j<4;j++) {
            ulong at=ulong(row)*p.hidden+base+lane+j*32;
            out[at]=sum[j];uint bits=as_type<uint>(sum[j]);
            out_bf16[at]=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
        }
    }
}
