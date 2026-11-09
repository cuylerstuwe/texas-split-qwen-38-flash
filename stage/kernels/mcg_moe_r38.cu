// GENERATED R38 combination: unchanged mcg_moe.cu followed by mcg_moe_r38_tail.cu.
// Regenerate by concatenation; do not edit here. See both sources and NOTICE.txt.
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

// R38 (Opus, direct): load-ahead routed-expert transforms. Appended to
// mcg_moe.cu (uses hf/fh/Four/had). Same arithmetic and order as mcg_moe_mid
// and mcg_moe_finish; split partials and side vectors load before use.
// mcg_moe_finish_add also performs moe_add_bf16 (routed BF16 + shared BF16).
// Requires splits <= 8 (checked by the caller).
extern "C" __global__ void mcg_moe_mid_ahead(const float* gate,const float* up,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    const unsigned* ids,unsigned short* middle,unsigned splits) {
    const unsigned lane=threadIdx.x,slot=blockIdx.y,base=blockIdx.x*128;
    float gr[8][4],ur[8][4];
    #pragma unroll
    for(unsigned s=0;s<8;s++)
        #pragma unroll
        for(unsigned j=0;j<4;j++) {
            const unsigned long long at=((unsigned long long)slot*splits+s)*640+base+lane+j*32;
            gr[s][j]=s<splits?gate[at]:0.f;ur[s][j]=s<splits?up[at]:0.f;
        }
    const unsigned sb=ids[slot]*640+base+lane,yb=slot*640+base+lane;
    unsigned short vgr[4],vur[4],sdr[4];
    #pragma unroll
    for(unsigned j=0;j<4;j++) {vgr[j]=vg[sb+j*32];vur[j]=vu[sb+j*32];sdr[j]=sd[sb+j*32];}
    Four g0={{0,0,0,0}},u0={{0,0,0,0}};
    #pragma unroll
    for(unsigned s=0;s<8;s++) if(s<splits) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) {g0.v[j]+=gr[s][j];u0.v[j]+=ur[s][j];}
    }
    Four g=had(g0,lane),u=had(u0,lane),v;
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
        float a=g.v[j]*hf(vgr[j]),b=u.v[j]*hf(vur[j]);
        float ex=expf(-fabsf(a)),sig=a<0?ex/(1.0f+ex):1.0f/(1.0f+ex);
        v.v[j]=a*sig*b*hf(sdr[j]);
    }
    v=had(v,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++)middle[yb+j*32]=fh(v.v[j]);
}
extern "C" __global__ void mcg_moe_finish_add(const float* down,const unsigned short* sv,
    const unsigned* ids,const float* scores,float* out,unsigned short* bf16,unsigned splits,
    const unsigned short* shared,unsigned short* added) {
    __shared__ float vals[10*128];
    const unsigned lane=threadIdx.x&31,expert=threadIdx.x>>5,row=blockIdx.y,base=blockIdx.x*128;
    const unsigned slot=row*10+expert;
    float dr[8][4];
    #pragma unroll
    for(unsigned s=0;s<8;s++)
        #pragma unroll
        for(unsigned j=0;j<4;j++)
            dr[s][j]=s<splits?down[((unsigned long long)slot*splits+s)*2560+base+lane+j*32]:0.f;
    float sc[10];unsigned short sh[4];
    if(expert==0) {
        #pragma unroll
        for(unsigned e=0;e<10;e++) sc[e]=scores[row*10+e];
        #pragma unroll
        for(unsigned j=0;j<4;j++) sh[j]=shared[row*2560+base+lane+j*32];
    }
    const unsigned sb=ids[slot]*2560+base+lane;
    unsigned short svr[4];
    #pragma unroll
    for(unsigned j=0;j<4;j++) svr[j]=sv[sb+j*32];
    Four acc={{0,0,0,0}};
    #pragma unroll
    for(unsigned s=0;s<8;s++) if(s<splits) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) acc.v[j]+=dr[s][j];
    }
    Four v=had(acc,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++)vals[expert*128+lane+j*32]=v.v[j]*hf(svr[j]);
    __syncthreads();
    if(expert==0) {
        Four sum={{0,0,0,0}};
        for(unsigned e=0;e<10;e++) {
            #pragma unroll
            for(unsigned j=0;j<4;j++)sum.v[j]+=vals[e*128+lane+j*32]*sc[e];
        }
        #pragma unroll
        for(unsigned j=0;j<4;j++) {
            const unsigned at=row*2560+base+lane+j*32;
            out[at]=sum.v[j];unsigned bits=__float_as_uint(sum.v[j]);
            const unsigned short rbits=(unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);
            bf16[at]=rbits;
            const float a=__uint_as_float((unsigned)rbits<<16)+__uint_as_float((unsigned)sh[j]<<16);
            unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(a)); added[at]=h;
        }
    }
}
