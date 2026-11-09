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
__device__ __forceinline__ Four sum_splits(const float* x,unsigned slot,unsigned dim,
    unsigned splits,unsigned base,unsigned lane) {
    Four out={{0,0,0,0}};
    for(unsigned s=0;s<splits;s++) {
        unsigned long long at=((unsigned long long)slot*splits+s)*dim+base+lane;
        #pragma unroll
        for(unsigned j=0;j<4;j++)out.v[j]+=x[at+j*32];
    }
    return out;
}
extern "C" __global__ void mcg_moe_prepare(const float* x,const unsigned short* sg,
    const unsigned short* su,const unsigned* ids,unsigned short* pg,unsigned short* pu) {
    const unsigned lane=threadIdx.x,slot=blockIdx.y,base=blockIdx.x*128;
    const unsigned xb=(slot/10)*2560+base+lane,sb=ids[slot]*2560+base+lane,yb=slot*2560+base+lane;
    Four g,u;
    #pragma unroll
    for(unsigned j=0;j<4;j++) { g.v[j]=x[xb+j*32]*hf(sg[sb+j*32]);u.v[j]=x[xb+j*32]*hf(su[sb+j*32]); }
    g=had(g,lane);u=had(u,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++) { pg[yb+j*32]=fh(g.v[j]);pu[yb+j*32]=fh(u.v[j]); }
}
extern "C" __global__ void mcg_moe_mid(const float* gate,const float* up,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    const unsigned* ids,unsigned short* middle,unsigned splits) {
    const unsigned lane=threadIdx.x,slot=blockIdx.y,base=blockIdx.x*128;
    const unsigned sb=ids[slot]*640+base+lane,yb=slot*640+base+lane;
    Four g=had(sum_splits(gate,slot,640,splits,base,lane),lane);
    Four u=had(sum_splits(up,slot,640,splits,base,lane),lane),v;
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
        float a=g.v[j]*hf(vg[sb+j*32]),b=u.v[j]*hf(vu[sb+j*32]);
        float ex=expf(-fabsf(a)),sig=a<0?ex/(1.0f+ex):1.0f/(1.0f+ex);
        v.v[j]=a*sig*b*hf(sd[sb+j*32]);
    }
    v=had(v,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++)middle[yb+j*32]=fh(v.v[j]);
}
extern "C" __global__ void mcg_moe_finish(const float* down,const unsigned short* sv,
    const unsigned* ids,const float* scores,float* out,unsigned short* bf16,unsigned splits) {
    __shared__ float vals[10*128];
    const unsigned lane=threadIdx.x&31,expert=threadIdx.x>>5,row=blockIdx.y,base=blockIdx.x*128;
    const unsigned slot=row*10+expert,sb=ids[slot]*2560+base+lane;
    Four v=had(sum_splits(down,slot,2560,splits,base,lane),lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++)vals[expert*128+lane+j*32]=v.v[j]*hf(sv[sb+j*32]);
    __syncthreads();
    if(expert==0) {
        Four sum={{0,0,0,0}};
        for(unsigned e=0;e<10;e++) {
            #pragma unroll
            for(unsigned j=0;j<4;j++)sum.v[j]+=vals[e*128+lane+j*32]*scores[row*10+e];
        }
        #pragma unroll
        for(unsigned j=0;j<4;j++) {
            const unsigned at=row*2560+base+lane+j*32;
            out[at]=sum.v[j];unsigned bits=__float_as_uint(sum.v[j]);
            bf16[at]=(unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);
        }
    }
}
