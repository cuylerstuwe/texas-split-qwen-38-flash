// Direct packed K4/MCG/window15 projection. Format/reduction derived from Sushi
// (MIT; NOTICE.txt) and the earlier locally validated CUDA projection.
// Portable SIMD baseline: deliberately no M5 MPP/GEMM specialization yet.
#include <metal_stdlib>
using namespace metal;
struct Params { uint input_dim; uint output_dim; uint splits; uint pad; };
inline float decode_mcg(uint cw) {
    uint p=(((cw & 0x7fffu)*0xCBAC1FEDu)&0x8FFF8FFFu)^0x3B603B60u;
    half sum=as_type<half>(ushort(p))+as_type<half>(ushort(p>>16));
    return float(sum);
}
kernel void exl3_project(const device half* x [[buffer(0)]],
    const device uint* trellis [[buffer(1)]], const device uint* ids [[buffer(2)]],
    device float* output [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[64];
    uint ot=group.x,slot=group.y,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.input_dim/16,nt=p.output_dim/16;
    uint per=(it+p.splits-1)/p.splits,begin=split*per,end=min(begin+per,it);
    uint row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(ids[slot])*it*nt*32;
    float acc[8]={};
    for(uint tk=begin+warp;tk<end;tk+=4) {
        const device uint* words=weights+(ulong(tk)*nt+ot)*32;
        // The funnel is expressed as two uint shifts to avoid slow 64-bit math
        // in the scalar decode loop; shift0 takes only the current word.
        uint previous=words[(lane+31)&31], current=words[lane];
        const device half* xx=x+ulong(slot)*p.input_dim+tk*16;
        float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
        for(uint j=0;j<8;j++) {
            uint shift=28-4*j;
            uint cw=shift ? ((current>>shift)|(previous<<(32-shift))) : current;
            acc[j]=fma(a[j&3],decode_mcg(cw),acc[j]);
        }
    }
    float lo=(acc[0]+acc[1])+(acc[2]+acc[3]);
    float hi=(acc[4]+acc[5])+(acc[6]+acc[7]);
    lo+=simd_shuffle_xor(lo,1);hi+=simd_shuffle_xor(hi,1);
    lo+=simd_shuffle_xor(lo,2);hi+=simd_shuffle_xor(hi,2);
    if((lane&3)==0) {partial[warp*16+col]=lo;partial[warp*16+col+8]=hi;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<16) {
        float sum=0;for(uint g=0;g<4;g++)sum+=partial[g*16+tid];
        output[(ulong(slot)*p.splits+split)*p.output_dim+ot*16+tid]=sum;
    }
}
