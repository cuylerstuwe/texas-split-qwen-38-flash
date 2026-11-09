// R41 component: original packed affine4 -> FP16 MMA operands, FP32 accumulation.
// Weight layout/affine parameters are unchanged. Only the register operand is
// rounded to FP16; no expanded weight matrix is allocated or transferred.
// Fragment mapping follows NVIDIA PTX mma.m16n8k16 floating-point documentation.
__device__ __forceinline__ float am_bf(unsigned short x) {
  return __uint_as_float((unsigned)x<<16);
}
__device__ __forceinline__ unsigned short am_half(float x) {
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ unsigned short am_out(float x) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ unsigned am_pair(unsigned packed,unsigned shift,float scale,float bias) {
  const float a=fmaf(scale,(float)((packed>>shift)&15u),bias);
  const float b=fmaf(scale,(float)((packed>>(shift+4))&15u),bias);
  return (unsigned)am_half(a)|((unsigned)am_half(b)<<16);
}
__device__ __forceinline__ unsigned am_pair_bf(unsigned packed,unsigned shift,unsigned scale,unsigned bias) {
  // 128+q is exactly representable in BF16. A packed subtract obtains both
  // integer codes without int->float conversions, then a packed BF16 FMA
  // rounds the affine weight once. Metadata is already BF16.
  const unsigned v=packed>>shift;
  unsigned q=(v&15u)|((v&240u)<<12)|0x43004300u;
  asm("add.rn.bf16x2 %0, %0, %1;":"+r"(q):"r"(0xc300c300u));
  unsigned out;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;":"=r"(out):"r"(q),"r"(scale|(scale<<16)),"r"(bias|(bias<<16)));
  return out;
}
template<unsigned W,unsigned D,bool Tiled=false,bool Bf=false>
__device__ void am_run(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* partial,
    float* y,unsigned short* ybf,unsigned k,unsigned n,unsigned group,unsigned splits,float* smem) {
  const unsigned t=threadIdx.x,lane=t&31u,warp=t>>5;
  const unsigned row=blockIdx.x*16+(lane>>2),c=(lane&3u)*2;
  const unsigned it=k/16,per=(it+splits-1)/splits;
  const unsigned begin=blockIdx.z*per,end=min(begin+per,it),ng=k/group;
  float acc[4]={0.f,0.f,0.f,0.f};
  for(unsigned tk0=begin+warp;tk0<end;tk0+=W*D) {
    unsigned p[D][4],sx[D][2],bs[D][2],xx[D][2];
    #pragma unroll
    for(unsigned d=0;d<D;d++) {
      const unsigned tk=tk0+d*W,kk=tk*16;
      const bool live=tk<end;
      unsigned packet=0;
      if constexpr(Tiled) packet=live?w[((size_t)tk*((n+15)/16)+blockIdx.x)*32+lane]:0u;
      #pragma unroll
      for(unsigned j=0;j<2;j++) {
        const unsigned r=row+j*8;
        const bool ok=live && r<n;
        sx[d][j]=ok?s[r*ng+kk/group]:0u;
        bs[d][j]=ok?b[r*ng+kk/group]:0u;
        if constexpr(Tiled) { p[d][j]=(packet>>(8*j))&255u;p[d][j+2]=(packet>>(8*(j+2)))&255u; }
        else {
          p[d][j]=ok?w[(size_t)r*(k/8)+kk/8]:0u;
          p[d][j+2]=ok?w[(size_t)r*(k/8)+kk/8+1]:0u;
        }
      }
      #pragma unroll
      for(unsigned j=0;j<2;j++) {
        const bool ok=live && (lane>>2)==0;
        xx[d][j]=ok?*reinterpret_cast<const unsigned*>(x+kk+c+j*8):0u;
      }
    }
    #pragma unroll
    for(unsigned d=0;d<D;d++) if(tk0+d*W<end) {
      unsigned aa[4],bb[2];
      #pragma unroll
      for(unsigned j=0;j<4;j++) {
        if constexpr(Bf) aa[j]=am_pair_bf(p[d][j],Tiled?0:c*4,sx[d][j&1],bs[d][j&1]);
        else aa[j]=am_pair(p[d][j],Tiled?0:c*4,am_bf(sx[d][j&1]),am_bf(bs[d][j&1]));
      }
      #pragma unroll
      for(unsigned j=0;j<2;j++) {
        if constexpr(Bf) bb[j]=xx[d][j];
        else bb[j]=(unsigned)am_half(am_bf(xx[d][j]&65535u))|((unsigned)am_half(am_bf(xx[d][j]>>16))<<16);
      }
      if constexpr(Bf) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
          : "r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1]));
      } else { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
          : "r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1])); }
    }
  }
  if((lane&3u)==0) { smem[warp*16+(lane>>2)]=acc[0];smem[warp*16+(lane>>2)+8]=acc[2]; }
  __syncthreads();
  if(t<16 && blockIdx.x*16+t<n) {
    float sum=0.f;
    #pragma unroll
    for(unsigned j=0;j<W;j++) sum+=smem[j*16+t];
    const unsigned at=blockIdx.x*16+t;
    if(splits==1) { y[at]=sum;ybf[at]=am_out(sum); }
    else partial[blockIdx.z*n+at]=sum;
  }
}
#define AM_ENTRY(NAME,W,D,T,B) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* w, \
    const unsigned short* s,const unsigned short* b,float* partial,float* y,unsigned short* ybf, \
    unsigned k,unsigned n,unsigned group,unsigned splits) { \
  __shared__ float smem[W*16];am_run<W,D,T,B>(x,w,s,b,partial,y,ybf,k,n,group,splits,smem); \
}
AM_ENTRY(affine4_mma_w4d2,4,2,false,false)
AM_ENTRY(affine4_mma_w8d2,8,2,false,false)
AM_ENTRY(affine4_mma_tile_w4d2,4,2,true,false)
AM_ENTRY(affine4_mma_bf_w4d2,4,2,true,true)
extern "C" __global__ void affine4_mma_reduce(const float* partial,float* y,unsigned short* ybf,
    unsigned n,unsigned splits) {
  const unsigned at=blockIdx.x*128+threadIdx.x;
  if(at<n) {
    float sum=0.f;for(unsigned j=0;j<splits;j++)sum+=partial[j*n+at];
    y[at]=sum;ybf[at]=am_out(sum);
  }
}
