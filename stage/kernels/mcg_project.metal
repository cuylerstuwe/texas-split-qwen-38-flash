// K1.75/K4 packed MCG/window15 scalar control, Sushi-derived mapping (NOTICE).
#include <metal_stdlib>
using namespace metal;
struct Params { uint k; uint n; uint splits; uint halfwords; };
inline float mcg(uint cw) {
    uint p=(((cw&0x7fffu)*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    half h=as_type<half>(ushort(p))+as_type<half>(ushort(p>>16));
    return float(h);
}
template<uint H>
inline void project(const device half* x,const device uint* trellis,
    const device uint* ids,device float* y,constant Params& p,
    threadgroup float* partial,uint3 group,uint tid) {
    uint ot=group.x,slot=group.y,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.k/16,nt=p.n/16,per=(it+p.splits-1)/p.splits;
    uint begin=split*per,end=min(begin+per,it),row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(ids[slot])*it*nt*(H/2);
    float acc[8]={};
    for(uint tk=begin+warp;tk<end;tk+=4) {
        const device uint* words=weights+(ulong(tk)*nt+ot)*(H/2);
        const device half* xx=x+ulong(slot)*p.k+tk*16;
        float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
        for(uint pair=0;pair<4;pair++) {
            uint t=lane*8+pair*2,e0=(t+1)*H/16,e1=(t+2)*H/16;
            uint first=(e0+H*16-16)/32,last=(e1+H*16-1)/32;
            uint shift=(last+1)*32-(e1+H*16);
            ulong bits=(ulong(words[first%(H/2)])<<32)|ulong(words[last%(H/2)]);
            uint cw=uint(bits>>shift);
            acc[pair*2]=fma(a[(pair*2)&3],mcg(cw>>(e1-e0)),acc[pair*2]);
            acc[pair*2+1]=fma(a[(pair*2+1)&3],mcg(cw),acc[pair*2+1]);
        }
    }
    float lo=(acc[0]+acc[1])+(acc[2]+acc[3]),hi=(acc[4]+acc[5])+(acc[6]+acc[7]);
    lo+=simd_shuffle_xor(lo,1); hi+=simd_shuffle_xor(hi,1);
    lo+=simd_shuffle_xor(lo,2); hi+=simd_shuffle_xor(hi,2);
    if((lane&3)==0) {partial[warp*16+col]=lo;partial[warp*16+col+8]=hi;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<16) {
        float sum=0;for(uint w=0;w<4;w++)sum+=partial[w*16+tid];
        y[(ulong(slot)*p.splits+split)*p.n+ot*16+tid]=sum;
    }
}
kernel void mcg_project(const device half* x [[buffer(0)]],const device uint* t [[buffer(1)]],
    const device uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant Params& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[64];
    if(p.halfwords==28) project<28>(x,t,ids,y,p,partial,group,tid);
    else if(p.halfwords==64) project<64>(x,t,ids,y,p,partial,group,tid);
}
