// Native CUDA adaptation of affine8_dense.metal's licensed MLX/Sushi QMV
// organization. See NOTICE.txt. Packed 4-bit codes stay resident; no expanded
// whole matrix. BF16 input/metadata, FP32 arithmetic, FP32 and BF16 outputs.
__device__ __forceinline__ float bf(unsigned short x) {
    return __uint_as_float(((unsigned)x)<<16);
}
__device__ __forceinline__ unsigned short fb(float x) {
    unsigned u=__float_as_uint(x);
    return (unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
template<unsigned NV, bool Factored>
__device__ __forceinline__ void affine4_body(const unsigned short* x,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
    float* y,unsigned short* ybf,unsigned k,unsigned n,unsigned group,unsigned rows) {
    unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    unsigned n0=blockIdx.x*4+warp*2,r0=blockIdx.y*NV,groups=k/group;
    float acc[NV][2]={};
    for(unsigned base=0;base<k;base+=256) {
        unsigned k0=base+lane*8;
        float vals[NV][8]={},sums[NV]={};
        #pragma unroll
        for(unsigned r=0;r<NV;r++) {
            #pragma unroll
            for(unsigned i=0;i<8;i++) {
                float a=(r0+r<rows && k0+i<k)?bf(x[(r0+r)*k+k0+i]):0.f;
                vals[r][i]=a;sums[r]+=a;
            }
        }
        if(k0<k) {
            #pragma unroll
            for(unsigned col=0;col<2;col++) if(n0+col<n) {
                unsigned at=(n0+col)*groups+k0/group;
                float s=bf(scales[at]),b=bf(biases[at]);
                unsigned packed=w[(n0+col)*(k/8)+k0/8];
                #pragma unroll
                for(unsigned r=0;r<NV;r++) {
                    if constexpr(Factored) {
                        float dot=0.f;
                        #pragma unroll
                        for(unsigned i=0;i<8;i++)
                            dot=fmaf(vals[r][i],float((packed>>(4*i))&15u),dot);
                        acc[r][col]+=fmaf(s,dot,b*sums[r]);
                    } else {
                        #pragma unroll
                        for(unsigned i=0;i<8;i++) {
                            float decoded=fmaf(s,float((packed>>(4*i))&15u),b);
                            acc[r][col]=fmaf(vals[r][i],decoded,acc[r][col]);
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<NV;r++) {
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
            float v=acc[r][col];
            #pragma unroll
            for(unsigned delta=16;delta;delta>>=1)
                v+=__shfl_down_sync(0xffffffffu,v,delta);
            if(lane==0 && r0+r<rows && n0+col<n) {
                unsigned at=(r0+r)*n+n0+col;y[at]=v;ybf[at]=fb(v);
            }
        }
    }
}
#define ENTRY(NAME,NV,F) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* w, \
 const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf, \
 unsigned k,unsigned n,unsigned group,unsigned rows) { \
 affine4_body<NV,F>(x,w,s,b,y,ybf,k,n,group,rows); \
}
ENTRY(affine4_direct1,1,false)
ENTRY(affine4_factored1,1,true)
ENTRY(affine4_factored2,2,true)
ENTRY(affine4_factored4,4,true)
