// M5 MPP raw EXL3 projection. Fragment layout and guarded row loads/stores are
// adapted from Sushi's GEMM_NAX_REFERENCE_SOURCE / GEMM_NAX_FRAGS (MIT, NOTICE.txt).
// Fixed K4/MCG/window15, FP16 operands and FP32 accumulation. Packed weights are
// decoded directly into cooperative fragments, never a materialized GPU matrix.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct MppParams { uint input_dim; uint output_dim; uint splits; uint unused1; };
inline half2 mcg_pair(uint2 cw) {
    cw &= uint2(0x7fffu);
    uint2 r=((cw*uint2(0xCBAC1FEDu))&uint2(0x8FFF8FFFu))^uint2(0x3B603B60u);
    half4 h=as_type<half4>(r);
    return half2(h.x+h.y,h.z+h.w);
}
using nfrag=vec<half,8>;
inline nfrag weight_fragment(const device uint* words,uint lane) {
    uint source0=(lane&16u)+((lane&7u)<<1u);
    uint2 current=*(const device uint2*)(words+source0);
    uint previous=words[(source0+31u)&31u];
    uint slot=((lane>>3u)&1u)*2u;
    ulong w0=(ulong(previous)<<32)|ulong(current.x);
    ulong w1=(ulong(current.x)<<32)|ulong(current.y);
    uint s0=28u-slot*4u,s1=28u-(slot+4u)*4u;
    half2 a=mcg_pair(uint2(uint(w0>>s0),uint(w0>>(s0-4u))));
    half2 b=mcg_pair(uint2(uint(w1>>s0),uint(w1>>(s0-4u))));
    half2 c=mcg_pair(uint2(uint(w0>>s1),uint(w0>>(s1-4u))));
    half2 d=mcg_pair(uint2(uint(w1>>s1),uint(w1>>(s1-4u))));
    return nfrag(a.x,a.y,b.x,b.y,c.x,c.y,d.x,d.y);
}
inline short2 fragment_origin(uint lane) {
    short qid=short(lane>>2u);
    return short2(short(((qid&2)|short(lane&1u))*4),short((qid&4)|short((lane>>1u)&3u)));
}
template<typename L> inline void load_left(thread L& left,const device half* x,
    uint row,uint k,uint base,short2 origin,uint count) {
    ulong r0=ulong(row+uint(origin.y))*k+base+uint(origin.x);
    ulong r1=r0+8u*ulong(k);
    half4 a=uint(origin.y)<count ? *(const device half4*)(x+r0) : half4(0);
    half4 b=uint(origin.y)+8u<count ? *(const device half4*)(x+r1) : half4(0);
    for(uint i=0;i<4;i++){left[i]=a[i];left[4+i]=b[i];}
}
template<typename D> inline void store_destination(const thread D& dst,
    device float* y,uint row,uint n,uint base,short2 origin,uint count,uint splits,uint split) {
    ulong o0=(ulong(row+uint(origin.y))*splits+split)*n,o1=o0+8u*ulong(n)*splits;
    for(uint tile=0;tile<2;tile++)for(uint c=0;c<4;c++) {
        uint col=base+tile*16u+uint(origin.x)+c;
        if(col<n) {
            if(uint(origin.y)<count)y[o0+col]=dst[tile*8+c];
            if(uint(origin.y)+8u<count)y[o1+col]=dst[tile*8+4+c];
        }
    }
}
kernel void exl3_mpp(const device half* x [[buffer(0)]],
    const device uint* trellis [[buffer(1)]], device float* y [[buffer(3)]],
    constant MppParams& p [[buffer(4)]], const device uint4* blocks [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint4 b=blocks[group.y]; // contiguous input start, live rows1..32, expert, reserved
    uint start=b.x,count=b.y,eid=b.z,lane=tid&31u,sg=tid>>5;
    if(count==0 || count>32)return;
    uint it=p.input_dim/16,nt=p.output_dim/16,base=group.x*128u+sg*32u;
    uint hi=count>16?count-16:0;
    uint per=(it+p.splits-1)/p.splits,begin=group.z*per,end=min(begin+per,it);
    constexpr auto desc=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroup> op;
    auto left=op.get_left_input_cooperative_tensor<half,half,float>();
    auto right=op.get_right_input_cooperative_tensor<half,half,float>();
    auto dst=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
    auto dst_hi=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
    for(uint s=0;s<dst.get_capacity();s++){dst[s]=0.;if(hi)dst_hi[s]=0.;}
    short2 origin=fragment_origin(lane);
    const device uint* w=trellis+ulong(eid)*it*nt*32;
    #pragma clang loop unroll_count(2)
    for(uint tk=begin;tk<end;tk++) {
        const device uint* words=w+(ulong(tk)*nt+base/16)*32;
        nfrag a=weight_fragment(words,lane),b=weight_fragment(words+32,lane);
        for(uint s=0;s<8;s++){right[s]=a[s];right[8+s]=b[s];}
        load_left(left,x,start,p.input_dim,tk*16,origin,count);
        op.run(left,right,dst);
        if(hi){load_left(left,x,start+16,p.input_dim,tk*16,origin,hi);op.run(left,right,dst_hi);}
    }
    store_destination(dst,y,start,p.output_dim,base,origin,count,p.splits,group.z);
    if(hi)store_destination(dst_hi,y,start+16,p.output_dim,base,origin,hi,p.splits,group.z);
}
