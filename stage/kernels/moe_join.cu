// Routed expert transforms, adapted from the licensed Sushi-derived Metal
// reference in exl3_moe.metal. See NOTICE.txt. No weight requantization.
// Fixed model H=2560, I=640, topK=10; Rust validates buffers/IDs/launch geometry.
__device__ __forceinline__ float hf(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
__device__ __forceinline__ unsigned short fh(float x) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
struct Four { float v[4]; };
__device__ __forceinline__ Four had(Four x, unsigned lane) {
    #pragma unroll
    for(unsigned bit=1;bit<=16;bit<<=1) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) {
            float other=__shfl_xor_sync(0xffffffffu,x.v[j],bit);
            x.v[j]=(lane&bit)==0?x.v[j]+other:other-x.v[j];
        }
    }
    float a=x.v[0]+x.v[1],b=x.v[0]-x.v[1],c=x.v[2]+x.v[3],d=x.v[2]-x.v[3];
    Four y={{a+c,b+d,a-c,b-d}};
    #pragma unroll
    for(unsigned j=0;j<4;j++)y.v[j]*=0.08838834764831845f;
    return y;
}
// BF16 input is an explicit ABI, not a reinterpretation of the R16 FP32 source.
__device__ __forceinline__ float bf(unsigned short h) { return __uint_as_float((unsigned)h<<16); }
extern "C" __global__ void moe_prepare_bf16(const unsigned short* x,const unsigned short* sg,
    const unsigned short* su,const unsigned* ids,unsigned short* pg,unsigned short* pu) {
    const unsigned lane=threadIdx.x,slot=blockIdx.y,base=blockIdx.x*128;
    const unsigned xb=(slot/10)*2560+base+lane,sb=ids[slot]*2560+base+lane,yb=slot*2560+base+lane;
    Four g,u;
    #pragma unroll
    for(unsigned j=0;j<4;j++) { g.v[j]=bf(x[xb+j*32])*hf(sg[sb+j*32]);u.v[j]=bf(x[xb+j*32])*hf(su[sb+j*32]); }
    g=had(g,lane);u=had(u,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++) { pg[yb+j*32]=fh(g.v[j]);pu[yb+j*32]=fh(u.v[j]); }
}
extern "C" __global__ void moe_add_bf16(const unsigned short* routed,
    const unsigned short* shared, unsigned short* out, unsigned count) {
    unsigned at=blockIdx.x*128+threadIdx.x;
    if(at<count) {
        float v=bf(routed[at])+bf(shared[at]);
        unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(v)); out[at]=h;
    }
}
