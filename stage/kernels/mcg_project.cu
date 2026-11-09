// K1.75 and K4 MCG/window15. Tile mapping and scalar reduction follow the
// licensed Sushi-derived baseline; see NOTICE.txt. No model requantization.
// Self-contained NVRTC source; explicit FP16 codebook rounding / FP32 FMA.
__device__ __forceinline__ float half_float(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
__device__ __forceinline__ unsigned short float_half(float x) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ float mcg(unsigned cw) {
    unsigned p=(((cw&0x7fffu)*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    return half_float(float_half(half_float((unsigned short)p)+half_float((unsigned short)(p>>16))));
}
template<unsigned H>
__device__ void projection(const unsigned short* x,const unsigned* trellis,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits) {
    __shared__ float partial[64];
    const unsigned ot=blockIdx.x,slot=blockIdx.y,split=blockIdx.z;
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned begin=split*per,end=min(begin+per,it),row=(lane&3)*2,col=lane>>2;
    const unsigned* weights=trellis+(unsigned long long)ids[slot]*it*nt*(H/2);
    float acc[8]={};
    for(unsigned tk=begin+warp;tk<end;tk+=4) {
        const unsigned* words=weights+((unsigned long long)tk*nt+ot)*(H/2);
        const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
        const float a[4]={half_float(xx[row]),half_float(xx[row+1]),half_float(xx[row+8]),half_float(xx[row+9])};
        #pragma unroll
        for(unsigned pair=0;pair<4;pair++) {
            const unsigned t=lane*8+pair*2;
            const unsigned e0=(t+1)*H/16,e1=(t+2)*H/16;
            const unsigned first=(e0+H*16-16)/32,last=(e1+H*16-1)/32;
            const unsigned shift=(last+1)*32-(e1+H*16);
            const unsigned long long bits=((unsigned long long)words[first%(H/2)]<<32)|words[last%(H/2)];
            const unsigned cw=(unsigned)(bits>>shift);
            acc[pair*2]=__fmaf_rn(a[(pair*2)&3],mcg(cw>>(e1-e0)),acc[pair*2]);
            acc[pair*2+1]=__fmaf_rn(a[(pair*2+1)&3],mcg(cw),acc[pair*2+1]);
        }
    }
    float lo=(acc[0]+acc[1])+(acc[2]+acc[3]),hi=(acc[4]+acc[5])+(acc[6]+acc[7]);
    lo+=__shfl_xor_sync(0xffffffffu,lo,1); hi+=__shfl_xor_sync(0xffffffffu,hi,1);
    lo+=__shfl_xor_sync(0xffffffffu,lo,2); hi+=__shfl_xor_sync(0xffffffffu,hi,2);
    if((lane&3)==0) { partial[warp*16+col]=lo; partial[warp*16+col+8]=hi; }
    __syncthreads();
    if(tid<16) {
        float sum=0; for(unsigned w=0;w<4;w++)sum+=partial[w*16+tid];
        y[((unsigned long long)slot*splits+split)*n+ot*16+tid]=sum;
    }
}
extern "C" __global__ void mcg_project(const unsigned short* x,const unsigned* t,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits,unsigned halfwords) {
    if(halfwords==28) projection<28>(x,t,ids,y,k,n,splits);
    else if(halfwords==64) projection<64>(x,t,ids,y,k,n,splits);
}
