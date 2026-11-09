// R38 (Opus, direct): load-ahead packed affine4, v2 (raw loads first). Same licensed MLX/Sushi QMV
// organization and arithmetic as affine4.cu and affine4_split.cu; see NOTICE.txt.
//
// Each warp issues the loads for D consecutive 256-K steps before it computes
// any of them. The lane-to-K mapping, the step order and every FMA are the same
// as affine4_direct1/factored1/partial1, so FP32 and BF16 outputs are intended to
// match those kernels bit for bit. Only the timing of independent loads changes.
// Single-row (NV=1) decode path only. Block: 64 threads (two warps, two output
// columns each), exactly like the originals.
__device__ __forceinline__ float ah_bf(unsigned short x) {
    return __uint_as_float(((unsigned)x)<<16);
}
__device__ __forceinline__ unsigned short ah_fb(float x) {
    unsigned u=__float_as_uint(x);
    return (unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
template<unsigned D,bool Factored>
__device__ __forceinline__ void ahead_acc(const unsigned short* x,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,unsigned k,unsigned n,
    unsigned group,unsigned rows,unsigned begin,unsigned end,unsigned n0,unsigned r0,
    unsigned lane,float acc[2]) {
    const unsigned groups=k/group;
    // 16-byte input loads need 8-element alignment of every row start.
    const bool vec=(k&7u)==0u;
    for(unsigned base=begin;base<end;base+=256u*D) {
        // Load phase: raw bits only. Nothing here consumes a loaded value, so
        // all loads for D steps can be in flight together.
        unsigned xw[D][4],packed[D][2];
        unsigned short sr[D][2],br[D][2];
        #pragma unroll
        for(unsigned d=0;d<D;d++) {
            const unsigned kb=base+d*256u,k0=kb+lane*8u;
            const bool live=kb<end && r0<rows;
            if(vec) {
                const bool ok=live && k0+8u<=k;
                const uint4 q=ok?*reinterpret_cast<const uint4*>(x+(size_t)r0*k+k0):make_uint4(0u,0u,0u,0u);
                xw[d][0]=q.x;xw[d][1]=q.y;xw[d][2]=q.z;xw[d][3]=q.w;
            } else {
                #pragma unroll
                for(unsigned j=0;j<4;j++) {
                    const unsigned i0=2*j,i1=2*j+1;
                    const unsigned lo=(live && k0+i0<k)?x[(size_t)r0*k+k0+i0]:0u;
                    const unsigned hi=(live && k0+i1<k)?x[(size_t)r0*k+k0+i1]:0u;
                    xw[d][j]=lo|(hi<<16);
                }
            }
            #pragma unroll
            for(unsigned col=0;col<2;col++) {
                const bool ok=kb<end && k0<k && n0+col<n;
                const unsigned at=(n0+col)*groups+k0/group;
                sr[d][col]=ok?scales[at]:(unsigned short)0;
                br[d][col]=ok?biases[at]:(unsigned short)0;
                packed[d][col]=ok?w[(size_t)(n0+col)*(k/8u)+k0/8u]:0u;
            }
        }
        // Compute phase: steps in their original order, identical arithmetic.
        #pragma unroll
        for(unsigned d=0;d<D;d++) {
            const unsigned kb=base+d*256u,k0=kb+lane*8u;
            float vals[8],sums=0.f;
            #pragma unroll
            for(unsigned i=0;i<8;i++) {
                const unsigned word=xw[d][i>>1];
                const float a=__uint_as_float((i&1u)?(word&0xffff0000u):(word<<16));
                vals[i]=a;sums+=a;
            }
            if(kb<end && k0<k) {
                #pragma unroll
                for(unsigned col=0;col<2;col++) if(n0+col<n) {
                    const float s=ah_bf(sr[d][col]),b=ah_bf(br[d][col]);
                    const unsigned p=packed[d][col];
                    if constexpr(Factored) {
                        float dot=0.f;
                        #pragma unroll
                        for(unsigned i=0;i<8;i++)
                            dot=fmaf(vals[i],float((p>>(4*i))&15u),dot);
                        acc[col]+=fmaf(s,dot,b*sums);
                    } else {
                        #pragma unroll
                        for(unsigned i=0;i<8;i++) {
                            const float decoded=fmaf(s,float((p>>(4*i))&15u),b);
                            acc[col]=fmaf(vals[i],decoded,acc[col]);
                        }
                    }
                }
            }
        }
    }
}
__device__ __forceinline__ float warp_total(float v) {
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
    return v;
}
template<unsigned D,bool Factored>
__device__ __forceinline__ void ahead_full(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf,
    unsigned k,unsigned n,unsigned group,unsigned rows) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const unsigned n0=blockIdx.x*4+warp*2,r0=blockIdx.y;
    float acc[2]={0.f,0.f};
    ahead_acc<D,Factored>(x,w,s,b,k,n,group,rows,0,k,n0,r0,lane,acc);
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
        const float v=warp_total(acc[col]);
        if(lane==0 && r0<rows && n0+col<n) {
            const unsigned at=r0*n+n0+col;y[at]=v;ybf[at]=ah_fb(v);
        }
    }
}
template<unsigned D>
__device__ __forceinline__ void ahead_partial(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* partial,unsigned k,unsigned n,
    unsigned group,unsigned rows,unsigned splits) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5,part=blockIdx.z;
    const unsigned n0=blockIdx.x*4+warp*2,r0=blockIdx.y;
    const unsigned chunks=(k+255)/256,begin=(chunks*part/splits)*256,end=(chunks*(part+1)/splits)*256;
    float acc[2]={0.f,0.f};
    ahead_acc<D,true>(x,w,s,b,k,n,group,rows,begin,end,n0,r0,lane,acc);
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
        const float v=warp_total(acc[col]);
        if(lane==0 && r0<rows && n0+col<n) partial[(r0*splits+part)*n+n0+col]=v;
    }
}
#define FULL(NAME,D,F) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* w, \
 const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf, \
 unsigned k,unsigned n,unsigned group,unsigned rows) { \
 ahead_full<D,F>(x,w,s,b,y,ybf,k,n,group,rows); \
}
#define PART(NAME,D) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* w, \
 const unsigned short* s,const unsigned short* b,float* partial, \
 unsigned k,unsigned n,unsigned group,unsigned rows,unsigned splits) { \
 ahead_partial<D>(x,w,s,b,partial,k,n,group,rows,splits); \
}
FULL(affine4_ahead_direct1_d2,2,false)
FULL(affine4_ahead_direct1_d4,4,false)
FULL(affine4_ahead_direct1_d8,8,false)
FULL(affine4_ahead_factored1_d4,4,true)
PART(affine4_ahead_partial1_d2,2)
PART(affine4_ahead_partial1_d4,4)
PART(affine4_ahead_partial1_d8,8)
// Four same-input, same-K projections in one launch (GDN qkv/z/a/b, attention
// q/k/v/indexer). Each block computes exactly the block it would compute in its
// own affine4_ahead_direct1_d2 launch; blocks are assigned by prefix ranges.
template<unsigned D,bool Factored>
__device__ __forceinline__ void ahead_block(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf,
    unsigned k,unsigned n,unsigned group,unsigned rows,unsigned bx,unsigned r0) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const unsigned n0=bx*4+warp*2;
    float acc[2]={0.f,0.f};
    ahead_acc<D,Factored>(x,w,s,b,k,n,group,rows,0,k,n0,r0,lane,acc);
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
        const float v=warp_total(acc[col]);
        if(lane==0 && r0<rows && n0+col<n) {
            const unsigned at=r0*n+n0+col;y[at]=v;ybf[at]=ah_fb(v);
        }
    }
}
extern "C" __global__ void affine4_ahead_multi4_d2(const unsigned short* x,
    const unsigned* w0,const unsigned short* s0,const unsigned short* b0,float* y0,unsigned short* yb0,
    const unsigned* w1,const unsigned short* s1,const unsigned short* b1,float* y1,unsigned short* yb1,
    const unsigned* w2,const unsigned short* s2,const unsigned short* b2,float* y2,unsigned short* yb2,
    const unsigned* w3,const unsigned short* s3,const unsigned short* b3,float* y3,unsigned short* yb3,
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned k,unsigned group,unsigned rows) {
    unsigned bx=blockIdx.x;
    const unsigned c0=(n0+3)/4,c1=(n1+3)/4,c2=(n2+3)/4;
    if(bx<c0) {ahead_block<2,false>(x,w0,s0,b0,y0,yb0,k,n0,group,rows,bx,blockIdx.y);return;}
    bx-=c0;
    if(bx<c1) {ahead_block<2,false>(x,w1,s1,b1,y1,yb1,k,n1,group,rows,bx,blockIdx.y);return;}
    bx-=c1;
    if(bx<c2) {ahead_block<2,false>(x,w2,s2,b2,y2,yb2,k,n2,group,rows,bx,blockIdx.y);return;}
    bx-=c2;
    ahead_block<2,false>(x,w3,s3,b3,y3,yb3,k,n3,group,rows,bx,blockIdx.y);
}
