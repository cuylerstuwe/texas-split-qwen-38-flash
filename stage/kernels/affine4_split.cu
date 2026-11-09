// R18b: one-GPU split reduction for packed affine4. Same licensed MLX/Sushi
// QMV organization as affine4.cu; see NOTICE.txt. Complete 256-K chunks have
// exclusive owners. No atomic accumulator or host intermediate is required.
__device__ __forceinline__ float bf(unsigned short x) {
    return __uint_as_float(((unsigned)x)<<16);
}
template<unsigned NV>
__device__ __forceinline__ void partial_body(const unsigned short* x,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
    float* partial,unsigned k,unsigned n,unsigned group,unsigned rows,unsigned splits) {
    unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5,part=blockIdx.z;
    unsigned n0=blockIdx.x*4+warp*2,r0=blockIdx.y*NV,groups=k/group;
    unsigned chunks=(k+255)/256,begin=(chunks*part/splits)*256,end=(chunks*(part+1)/splits)*256;
    float acc[NV][2]={};
    for(unsigned base=begin;base<end;base+=256) {
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
                    float dot=0.f;
                    #pragma unroll
                    for(unsigned i=0;i<8;i++)
                        dot=fmaf(vals[r][i],float((packed>>(4*i))&15u),dot);
                    acc[r][col]+=fmaf(s,dot,b*sums[r]);
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
            for(unsigned delta=16;delta;delta>>=1)v+=__shfl_down_sync(0xffffffffu,v,delta);
            if(lane==0 && r0+r<rows && n0+col<n)
                partial[((r0+r)*splits+part)*n+n0+col]=v;
        }
    }
}
#define ENTRY(NAME,NV) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* w, \
 const unsigned short* s,const unsigned short* b,float* partial, \
 unsigned k,unsigned n,unsigned group,unsigned rows,unsigned splits) { \
 partial_body<NV>(x,w,s,b,partial,k,n,group,rows,splits); \
}
ENTRY(affine4_partial1,1)
ENTRY(affine4_partial2,2)
ENTRY(affine4_partial4,4)
extern "C" __global__ void affine4_reduce(const float* partial,float* y,unsigned short* ybf,
    unsigned n,unsigned rows,unsigned splits) {
    unsigned at=blockIdx.x*128+threadIdx.x;
    if(at>=rows*n)return;
    unsigned r=at/n,col=at%n;float value=0.f;
    for(unsigned p=0;p<splits;p++)value+=partial[(r*splits+p)*n+col];
    y[at]=value;unsigned u=__float_as_uint(value);
    ybf[at]=(unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
