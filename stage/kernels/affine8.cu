// R60: native CUDA port of affine8_dense.metal's licensed MLX/Sushi QMV organization (see
// NOTICE.txt) for the canonical target's 8-bit trunk (pipeline split, layers on the RTX 5090).
// Byte codes stay resident (MLX affine: w = scale * code + bias, BF16 scales/biases, group 64 or
// 128). BF16 input, FP32 arithmetic in MLX's factored order (scale * sum(x*q) + bias * sum(x)),
// FP32 and BF16 outputs. Each warp owns two output columns; a block of two warps owns four. Lanes
// stride K by 8 codes (two 32-bit words), so a lane's codes always share one quantization group.
__device__ __forceinline__ float bf(unsigned short x) {
    return __uint_as_float(((unsigned)x)<<16);
}
__device__ __forceinline__ unsigned short fb(float x) {
    unsigned u=__float_as_uint(x);
    return (unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
template<unsigned NV>
__device__ __forceinline__ void affine8_body_at(unsigned bx,unsigned by,const unsigned short* __restrict__ x,
    const unsigned char* __restrict__ w,const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,float* __restrict__ y,unsigned short* __restrict__ ybf,
    unsigned k,unsigned n,unsigned group,unsigned rows) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const unsigned n0=bx*4+warp*2,r0=by*NV,groups=k/group;
    float acc[NV][2]={};
    for(unsigned base=0;base<k;base+=256) {
        const unsigned k0=base+lane*8;
        float vals[NV][8],sums[NV];
        #pragma unroll
        for(unsigned r=0;r<NV;r++) {
            sums[r]=0.f;
            const bool live=r0+r<rows && k0<k;
            const uint4 packed=live?*reinterpret_cast<const uint4*>(x+(unsigned long long)(r0+r)*k+k0):make_uint4(0,0,0,0);
            const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
            #pragma unroll
            for(unsigned i=0;i<4;i++) {
                const float lo=__uint_as_float(words[i]<<16),hi=__uint_as_float(words[i]&0xffff0000u);
                vals[r][2*i]=lo;vals[r][2*i+1]=hi;
            }
            #pragma unroll
            for(unsigned i=0;i<8;i++) sums[r]+=vals[r][i];
        }
        if(k0<k) {
            #pragma unroll
            for(unsigned col=0;col<2;col++) if(n0+col<n) {
                const unsigned at=(n0+col)*groups+k0/group;
                const float s=bf(scales[at]),b=bf(biases[at]);
                const uint2 q=*reinterpret_cast<const uint2*>(w+(unsigned long long)(n0+col)*k+k0);
                #pragma unroll
                for(unsigned r=0;r<NV;r++) {
                    float dot=0.f;
                    #pragma unroll
                    for(unsigned i=0;i<4;i++) dot=fmaf(vals[r][i],float((q.x>>(8*i))&255u),dot);
                    #pragma unroll
                    for(unsigned i=0;i<4;i++) dot=fmaf(vals[r][4+i],float((q.y>>(8*i))&255u),dot);
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
            for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
            if(lane==0 && r0+r<rows && n0+col<n) {
                const unsigned at=(r0+r)*n+n0+col;y[at]=v;ybf[at]=fb(v);
            }
        }
    }
}
#define ENTRY(NAME,NV) \
extern "C" __global__ void __launch_bounds__(64) NAME(const unsigned short* x,const unsigned char* w, \
 const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf, \
 unsigned k,unsigned n,unsigned group,unsigned rows) { \
 affine8_body_at<NV>(blockIdx.x,blockIdx.y,x,w,s,b,y,ybf,k,n,group,rows); \
}
ENTRY(affine8_f1,1)
ENTRY(affine8_f2,2)
ENTRY(affine8_f4,4)
ENTRY(affine8_f8,8)
// Split-K partials for tall-skinny projections (e.g. the HC down projection, K=10240 -> 320):
// same [row, split, column] FP32 layout and complete-256-K-chunk partition as affine4_split.cu.
template<unsigned NV>
__device__ __forceinline__ void affine8_partial_body(const unsigned short* __restrict__ x,
    const unsigned char* __restrict__ w,const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,float* __restrict__ partial,
    unsigned k,unsigned n,unsigned group,unsigned rows,unsigned splits) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5,part=blockIdx.z;
    const unsigned n0=blockIdx.x*4+warp*2,r0=blockIdx.y*NV,groups=k/group;
    const unsigned chunks=(k+255)/256,begin=(chunks*part/splits)*256,end=(chunks*(part+1)/splits)*256;
    float acc[NV][2]={};
    for(unsigned base=begin;base<end;base+=256) {
        const unsigned k0=base+lane*8;
        float vals[NV][8],sums[NV];
        #pragma unroll
        for(unsigned r=0;r<NV;r++) {
            sums[r]=0.f;
            const bool live=r0+r<rows && k0<k;
            const uint4 packed=live?*reinterpret_cast<const uint4*>(x+(unsigned long long)(r0+r)*k+k0):make_uint4(0,0,0,0);
            const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
            #pragma unroll
            for(unsigned i=0;i<4;i++) {
                vals[r][2*i]=__uint_as_float(words[i]<<16);vals[r][2*i+1]=__uint_as_float(words[i]&0xffff0000u);
            }
            #pragma unroll
            for(unsigned i=0;i<8;i++) sums[r]+=vals[r][i];
        }
        if(k0<k) {
            #pragma unroll
            for(unsigned col=0;col<2;col++) if(n0+col<n) {
                const unsigned at=(n0+col)*groups+k0/group;
                const float s=bf(scales[at]),b=bf(biases[at]);
                const uint2 q=*reinterpret_cast<const uint2*>(w+(unsigned long long)(n0+col)*k+k0);
                #pragma unroll
                for(unsigned r=0;r<NV;r++) {
                    float dot=0.f;
                    #pragma unroll
                    for(unsigned i=0;i<4;i++) dot=fmaf(vals[r][i],float((q.x>>(8*i))&255u),dot);
                    #pragma unroll
                    for(unsigned i=0;i<4;i++) dot=fmaf(vals[r][4+i],float((q.y>>(8*i))&255u),dot);
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
            for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
            if(lane==0 && r0+r<rows && n0+col<n) partial[((r0+r)*splits+part)*n+n0+col]=v;
        }
    }
}
#define PARTIAL(NAME,NV) \
extern "C" __global__ void __launch_bounds__(64) NAME(const unsigned short* x,const unsigned char* w, \
 const unsigned short* s,const unsigned short* b,float* partial, \
 unsigned k,unsigned n,unsigned group,unsigned rows,unsigned splits) { \
 affine8_partial_body<NV>(x,w,s,b,partial,k,n,group,rows,splits); \
}
PARTIAL(affine8_partial1,1)
PARTIAL(affine8_partial2,2)
PARTIAL(affine8_partial4,4)
extern "C" __global__ void affine8_reduce(const float* partial,float* y,unsigned short* ybf,
    unsigned n,unsigned rows,unsigned splits) {
    const unsigned at=blockIdx.x*128+threadIdx.x;
    if(at>=rows*n) return;
    const unsigned r=at/n,col=at%n;float value=0.f;
    for(unsigned p=0;p<splits;p++) value+=partial[(r*splits+p)*n+col];
    y[at]=value;ybf[at]=fb(value);
}
// R60: the canonical HC site in two launches (port of lowbit_hc_r38.cu's v3 hc_r38_norm_down3 and
// hc_r38_front2, NOTICE.txt): byte-code down/up weights in the same factored order as
// affine8_partial*/affine8_f1, so the fused site matches the unfused canonical chain.
typedef unsigned short hr_bf;
__device__ __forceinline__ float hr_load(hr_bf v) { return __uint_as_float((unsigned)v<<16); }
__device__ __forceinline__ hr_bf hr_store(float v) {
  hr_bf out; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(v)); return out;
}
__device__ __forceinline__ float hr_rb(float v) { return hr_load(hr_store(v)); }
__device__ __forceinline__ float hr_sigmoid(float v) { return hr_rb(1.f/(1.f+expf(-v))); }
__device__ __forceinline__ float hr_warp_sum(float v) {
  for(int s=16;s;s>>=1) v+=__shfl_down_sync(0xffffffffU,v,s); return v;
}
#define HR_H 2560u
#define HR_HC 4u
#define HR_F 10240u
#define HR_R 320u
#define HR_SPLITS 4u
#define HR_GEMV_BLOCKS 80u
__device__ __forceinline__ float hr_half_of(unsigned word,unsigned i) {
  return __uint_as_float((i&1u)?(word&0xffff0000u):(word<<16));
}
extern "C" __global__ void __launch_bounds__(64) hc8_norm_down3(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned char* w,const unsigned short* scales,const unsigned short* biases,
    unsigned short* normalized,float* dpart,unsigned short* next,float* partial,
    unsigned group,int pending) {
  const unsigned t=threadIdx.x,lane=t&31u,rw=t>>5;
  const unsigned bx=blockIdx.x,row=blockIdx.y,s=blockIdx.z;
  const bool gemv=bx<HR_GEMV_BLOCKS;
  __shared__ __align__(16) unsigned short sv[HR_H];
  __shared__ float ssum[8];
  const unsigned n0=bx*4+rw*2,groups=HR_F/group;
  const unsigned kbegin=s*HR_H;
  // v3: every step's weights are requested before anything else, so a GEMV
  // block waits for memory once instead of once per step.
  uint2 cp[5][2][2];unsigned cs[5][2][2],cb[5][2][2];
  uint4 cn[5][2];
  if(gemv) {
    #pragma unroll
    for(unsigned g=0;g<5;g++)
      #pragma unroll
      for(unsigned d=0;d<2;d++) {
        const unsigned k0=kbegin+g*512u+d*256u+lane*8u;
        cn[g][d]=*reinterpret_cast<const uint4*>(nw+k0);
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const unsigned at=(n0+col)*groups+k0/group;
          cs[g][d][col]=scales[at];cb[g][d][col]=biases[at];
          cp[g][d][col]=*reinterpret_cast<const uint2*>(w+(size_t)(n0+col)*HR_F+k0);
        }
      }
  }
  uint4 xr[5],br[5];
  #pragma unroll
  for(unsigned q=0;q<5;q++) {
    const unsigned c=t+64u*q;
    xr[q]=*reinterpret_cast<const uint4*>(x+(size_t)row*HR_F+s*HR_H+c*8u);
    br[q]=pending?*reinterpret_cast<const uint4*>(block+(size_t)row*HR_H+c*8u):make_uint4(0u,0u,0u,0u);
  }
  const float inj=pending?hr_load(inject[row*HR_HC+s]):0.f;
  #pragma unroll
  for(unsigned q=0;q<5;q++) {
    const unsigned c=t+64u*q;
    const unsigned xw[4]={xr[q].x,xr[q].y,xr[q].z,xr[q].w};
    const unsigned bw[4]={br[q].x,br[q].y,br[q].z,br[q].w};
    unsigned ow[4];
    #pragma unroll
    for(unsigned e2=0;e2<4;e2++) {
      unsigned lo,hi;
      { float v=hr_half_of(xw[e2],0); if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],0)*inj)); lo=hr_store(v); }
      { float v=hr_half_of(xw[e2],1); if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],1)*inj)); hi=hr_store(v); }
      ow[e2]=lo|(hi<<16);
    }
    *reinterpret_cast<uint4*>(sv+c*8u)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
  }
  __syncthreads();
  float sq[4]={0.f,0.f,0.f,0.f};
  #pragma unroll
  for(unsigned m=0;m<4;m++)
    #pragma unroll
    for(unsigned i=0;i<10;i++) { const float v=hr_load(sv[t+64u*m+256u*i]);sq[m]=fmaf(v,v,sq[m]); }
  #pragma unroll
  for(unsigned m=0;m<4;m++) { const float v=hr_warp_sum(sq[m]); if(lane==0) ssum[rw+2u*m]=v; }
  __syncthreads();
  float total=0;
  #pragma unroll
  for(unsigned q=0;q<8;q++) total+=ssum[q];
  const float rs=rsqrtf(total/float(HR_H)+1e-6f);
  if(!gemv) {
    if(rw!=0) return;
    const unsigned jw=bx-HR_GEMV_BLOCKS;          // virtual warp 0..7
    unsigned short nwr[10],iwr[4][10];
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const unsigned j=32u*jw+lane+256u*i;
      nwr[i]=nw[s*HR_H+j];
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) iwr[cc][i]=iw[(size_t)cc*HR_F+s*HR_H+j];
    }
    float dot[HR_HC]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const unsigned j=32u*jw+lane+256u*i;
      const float v=hr_load(sv[j]);
      const float nv=hr_rb(hr_rb(v*rs)*hr_load(nwr[i]));
      normalized[(size_t)row*HR_F+s*HR_H+j]=hr_store(nv);
      if(pending) next[(size_t)row*HR_F+s*HR_H+j]=sv[j];
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) dot[cc]=fmaf(nv,hr_load(iwr[cc][i]),dot[cc]);
    }
    #pragma unroll
    for(unsigned cc=0;cc<HR_HC;cc++) {
      const float v=hr_warp_sum(dot[cc]);
      if(lane==0) dpart[((row*HR_HC+s)*8u+jw)*HR_HC+cc]=v;
    }
    return;
  }
  float acc[2]={0.f,0.f};
  #pragma unroll
  for(unsigned g=0;g<5;g++) {
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned jj=g*512u+d*256u+lane*8u;
      const uint4 vq=*reinterpret_cast<const uint4*>(sv+jj);
      const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w},ww[4]={cn[g][d].x,cn[g][d].y,cn[g][d].z,cn[g][d].w};
      float vals[8],sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        const float nv=hr_rb(hr_rb(hr_half_of(vw[i>>1],i)*rs)*hr_half_of(ww[i>>1],i));
        vals[i]=nv;sums+=nv;
      }
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)cs[g][d][col]),bi=hr_load((unsigned short)cb[g][d][col]);
        const uint2 pk=cp[g][d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<4;i++) dot=fmaf(vals[i],float((pk.x>>(8*i))&255u),dot);
        #pragma unroll
        for(unsigned i=0;i<4;i++) dot=fmaf(vals[4+i],float((pk.y>>(8*i))&255u),dot);
        acc[col]+=fmaf(sc,dot,bi*sums);
      }
    }
  }
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    float v=acc[col];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
    if(lane==0) partial[((size_t)row*HR_SPLITS+s)*HR_R+n0+col]=v;
  }
}



extern "C" __global__ void __launch_bounds__(128) hc8_front2(
    const float* partial,const float* dpart,float* parts,const unsigned char* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][2];
  // Up-projection weights and the mix's normalized values first: independent.
  uint2 wp[2][2];unsigned ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?*reinterpret_cast<const uint2*>(w+(size_t)(n0+col)*HR_R+k0):make_uint2(0u,0u);
    }
  }
  const unsigned short nm0=normalized[(size_t)row*HR_F+n0],nm1=normalized[(size_t)row*HR_F+n0+1];
  // Replicated split reduction and activation (R37 order and rounding).
  float pv[3][4];
  #pragma unroll
  for(unsigned q=0;q<3;q++) {
    const unsigned j=t+128u*q;
    #pragma unroll
    for(unsigned p=0;p<4;p++) pv[q][p]=j<HR_R?partial[((size_t)row*HR_SPLITS+p)*HR_R+j]:0.f;
  }
  float pt[HR_HC];
  if(blockIdx.x==0 && t<HR_HC) {
    float dq[HR_HC][8];
    #pragma unroll
    for(unsigned hh=0;hh<HR_HC;hh++)
      #pragma unroll
      for(unsigned q=0;q<8;q++) dq[hh][q]=dpart[((row*HR_HC+hh)*8u+q)*HR_HC+t];
    #pragma unroll
    for(unsigned hh=0;hh<HR_HC;hh++) {
      float v=0;
      #pragma unroll
      for(unsigned q=0;q<8;q++) v+=dq[hh][q];
      pt[hh]=v;parts[(row*HR_HC+hh)*HR_HC+t]=v;
    }
  }
  #pragma unroll
  for(unsigned q=0;q<3;q++) {
    const unsigned j=t+128u*q;
    if(j<HR_R) {
      float v=0;
      #pragma unroll
      for(unsigned p=0;p<4;p++) v+=pv[q][p];
      const unsigned short low=hr_store(v);
      const float rounded=hr_load(low);
      const unsigned short a=hr_store(rounded*hr_sigmoid(rounded));
      local_act[j]=a;
      if(blockIdx.x==0) {raw_down[row*HR_R+j]=v;down[row*HR_R+j]=low;activated[row*HR_R+j]=a;}
    }
  }
  if(blockIdx.x==0 && t<HR_HC) {
    float v=0;
    #pragma unroll
    for(unsigned hh=0;hh<HR_HC;hh++) v+=pt[hh];
    inject[row*HR_HC+t]=hr_store(2.f*hr_sigmoid(hr_rb(v)));
  }
  __syncthreads();
  // Up projection (affine4 direct arithmetic) for this warp's two coordinates.
  float acc[2]={0.f,0.f};
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    if(k0<HR_R) {
      float vals[8];
      #pragma unroll
      for(unsigned i=0;i<8;i++) vals[i]=hr_load(local_act[k0+i]);
      float sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) sums+=vals[i];
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const uint2 p=wp[d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<4;i++) dot=fmaf(vals[i],float((p.x>>(8*i))&255u),dot);
        #pragma unroll
        for(unsigned i=0;i<4;i++) dot=fmaf(vals[4+i],float((p.y>>(8*i))&255u),dot);
        acc[col]+=fmaf(sc,dot,bi*sums);
      }
    }
  }
  #pragma unroll
  for(unsigned c=0;c<2;c++) {
    float v=acc[c];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
    if(lane==0) {
      const unsigned at=row*HR_F+n0+c;
      raw_up[at]=v;up[at]=hr_store(v);
      terms[h][c]=hr_rb(hr_sigmoid(hr_rb(v))*hr_load(c?nm1:nm0));
    }
  }
  __syncthreads();
  if(t<2) {
    float sum=0;
    #pragma unroll
    for(unsigned ss=0;ss<HR_HC;ss++) sum+=terms[ss][t];
    mixed[row*HR_H+j0+t]=hr_store(hr_rb(sum)*.25f);
  }
}

// R60: four projections of one input in one launch (GDN qkv/z/a/b, attention q/k/v/index), one
// token row per block row as affine8_f1, so each output equals its own affine8_f1 launch. Blocks
// [0,n0/4) are projection 0, then n1/4 blocks of projection 1, and so on; groups may differ.
extern "C" __global__ void __launch_bounds__(64) affine8_multi4(const unsigned short* x,
    const unsigned char* w0,const unsigned short* s0,const unsigned short* b0,float* y0,unsigned short* yb0,
    const unsigned char* w1,const unsigned short* s1,const unsigned short* b1,float* y1,unsigned short* yb1,
    const unsigned char* w2,const unsigned short* s2,const unsigned short* b2,float* y2,unsigned short* yb2,
    const unsigned char* w3,const unsigned short* s3,const unsigned short* b3,float* y3,unsigned short* yb3,
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned g0,unsigned g1,unsigned g2,unsigned g3,
    unsigned k,unsigned rows) {
    unsigned bx=blockIdx.x;
    const unsigned c0=(n0+3)/4,c1=(n1+3)/4,c2=(n2+3)/4;
    if(bx<c0) { affine8_body_at<1>(bx,blockIdx.y,x,w0,s0,b0,y0,yb0,k,n0,g0,rows); return; }
    bx-=c0;
    if(bx<c1) { affine8_body_at<1>(bx,blockIdx.y,x,w1,s1,b1,y1,yb1,k,n1,g1,rows); return; }
    bx-=c1;
    if(bx<c2) { affine8_body_at<1>(bx,blockIdx.y,x,w2,s2,b2,y2,yb2,k,n2,g2,rows); return; }
    bx-=c2;
    affine8_body_at<1>(bx,blockIdx.y,x,w3,s3,b3,y3,yb3,k,n3,g3,rows);
}
// R60: attn_part (kernels/lowbit_attention.cu) for the canonical layers with the selected keys'
// loads issued four at a time; the per-key arithmetic and the online-softmax order are unchanged
// (compiled with --fmad=false like the original), so results are bit-identical.
typedef unsigned short a8bf;
__device__ __forceinline__ float a8ld(a8bf x){return __uint_as_float((unsigned)x<<16);}
__device__ __forceinline__ a8bf a8st(float x){unsigned u=__float_as_uint(x);if((u&0x7fffffffU)>0x7f800000U)return 0x7fc0;return (a8bf)((u+0x7fffU+((u>>16)&1U))>>16);}
__device__ __forceinline__ float a8sum(float x){for(int m=16;m;m>>=1)x+=__shfl_xor_sync(0xffffffffU,x,m);return x;}
extern "C" __global__ void attn8_part(const a8bf* query,const unsigned char* kc,const unsigned char* vc,
    const a8bf* ks,const a8bf* kb,const a8bf* vs,const a8bf* vb,const unsigned* ids,float* partial,float* stats,
    unsigned offset,unsigned sparse,unsigned splits){
  const unsigned h=blockIdx.x,r=blockIdx.y,sp=blockIdx.z,lane=threadIdx.x,kh=h/12,pos=offset+r,complete=(pos+1)/4;
  const unsigned selected=sparse?min(complete,512U)*4:0,tail=sparse?complete*4:0,L=selected+pos+1-tail;
  const unsigned per=(L+splits-1)/splits,lo=sp*per,hi=min(L,lo+per);
  float q[8],acc[8];for(unsigned j=0;j<8;j++){q[j]=a8ld(query[(r*24+h)*256+lane*8+j]);acc[j]=0;}
  float m=-__int_as_float(0x7f800000),z=0;
  for(unsigned v0=lo;v0<hi;v0+=4){
    uint2 kq[4],vq[4];a8bf kss[4],kbb[4],vss[4],vbb[4];
    #pragma unroll
    for(unsigned u=0;u<4;u++){
      const unsigned vi=v0+u;
      if(vi<hi){
        const unsigned key=vi<selected?ids[r*512+vi/4]*4+vi%4:tail+vi-selected,at=(key*2+kh)*256+lane*8;
        kq[u]=*reinterpret_cast<const uint2*>(kc+at);vq[u]=*reinterpret_cast<const uint2*>(vc+at);
        kss[u]=ks[at/64];kbb[u]=kb[at/64];vss[u]=vs[at/64];vbb[u]=vb[at/64];
      }
    }
    #pragma unroll
    for(unsigned u=0;u<4;u++){
      if(v0+u>=hi) break;
      const float s=a8ld(kss[u]),b=a8ld(kbb[u]);
      float dot=0;
      #pragma unroll
      for(unsigned j=0;j<8;j++){const unsigned code=((j<4?kq[u].x:kq[u].y)>>(8*(j&3)))&255u;dot+=q[j]*a8ld(a8st((float)code*s+b));}
      dot=a8sum(dot)*(1.f/16.f);
      const float nm=fmaxf(m,dot),factor=expf(m-nm),prob=expf(dot-nm);z=z*factor+prob;m=nm;
      const float vsc=a8ld(vss[u]),vbi=a8ld(vbb[u]);
      #pragma unroll
      for(unsigned j=0;j<8;j++){const unsigned code=((j<4?vq[u].x:vq[u].y)>>(8*(j&3)))&255u;acc[j]=acc[j]*factor+prob*a8ld(a8st((float)code*vsc+vbi));}
    }
  }
  const unsigned idx=(r*24+h)*splits+sp;if(lane==0){stats[idx*2]=m;stats[idx*2+1]=z;}
  for(unsigned j=0;j<8;j++)partial[idx*256+lane*8+j]=acc[j];
}
// R60 S4: affine8 QMV v2 (same ABI and launch geometry as affine8_f*: 64 threads = two warps, two
// output columns per warp, NV token rows per block row). Lanes stride K by 16 codes, one 16-byte
// weight load per column per 512-K step, and every weight/scale/bias load of a column pair for up
// to eight steps (4096 K) is issued before any arithmetic: a warp waits on DRAM once per chunk
// instead of once per 256 K, which is what held v1 to ~half of the 5090's bandwidth at K=2560.
// Per 16-code chunk the arithmetic is MLX's factored form (scale * sum(x*q) + bias * sum(x)); the
// K order differs from v1, so results agree to rounding. Needs k % 16 == 0 (a 16-code chunk never
// straddles a 64/128-code group). Each weight byte is converted once (exactly: the float
// 2^23 + b built with PRMT, minus 2^23) and reused for all NV rows; multi-row variants take four
// steps per chunk to bound registers.
template<unsigned NV,unsigned STEPS>
__device__ __forceinline__ void affine8_v2_at(unsigned bx,unsigned by,const unsigned short* __restrict__ x,
    const unsigned char* __restrict__ w,const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,float* __restrict__ y,unsigned short* __restrict__ ybf,
    unsigned k,unsigned n,unsigned group,unsigned rows) {
    const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const unsigned n0=bx*4+warp*2,r0=by*NV,groups=k/group;
    const bool live0=n0<n,live1=n0+1<n;
    const unsigned col0=live0?n0:0u,col1=live1?n0+1:0u;
    const unsigned char* w0=w+(unsigned long long)col0*k;
    const unsigned char* w1=w+(unsigned long long)col1*k;
    const unsigned short* s0=scales+(unsigned long long)col0*groups;
    const unsigned short* s1=scales+(unsigned long long)col1*groups;
    const unsigned short* b0=biases+(unsigned long long)col0*groups;
    const unsigned short* b1=biases+(unsigned long long)col1*groups;
    float acc[NV][2];
    #pragma unroll
    for(unsigned r=0;r<NV;r++) { acc[r][0]=0.f; acc[r][1]=0.f; }
    for(unsigned base=0;base<k;base+=512*STEPS) {
        uint4 q0[STEPS],q1[STEPS];
        float sc0[STEPS],sc1[STEPS],bi0[STEPS],bi1[STEPS];
        #pragma unroll
        for(unsigned j=0;j<STEPS;j++) {
            const unsigned k0=base+j*512+lane*16;
            const bool in=k0<k;
            const unsigned g=in?k0/group:0u;
            q0[j]=in&&live0?__ldg(reinterpret_cast<const uint4*>(w0+k0)):make_uint4(0,0,0,0);
            q1[j]=in&&live1?__ldg(reinterpret_cast<const uint4*>(w1+k0)):make_uint4(0,0,0,0);
            sc0[j]=bf(__ldg(s0+g)); bi0[j]=bf(__ldg(b0+g));
            sc1[j]=bf(__ldg(s1+g)); bi1[j]=bf(__ldg(b1+g));
        }
        #pragma unroll
        for(unsigned j=0;j<STEPS;j++) {
            const unsigned k0=base+j*512+lane*16;
            if(k0<k) {
                const unsigned qa[4]={q0[j].x,q0[j].y,q0[j].z,q0[j].w};
                const unsigned qb[4]={q1[j].x,q1[j].y,q1[j].z,q1[j].w};
                float wa[16],wb[16];
                #pragma unroll
                for(unsigned i=0;i<16;i++) {
                    wa[i]=__uint_as_float(__byte_perm(qa[i>>2],0x4B000000u,0x7540u|(i&3u)))-8388608.f;
                    wb[i]=__uint_as_float(__byte_perm(qb[i>>2],0x4B000000u,0x7540u|(i&3u)))-8388608.f;
                }
                #pragma unroll
                for(unsigned r=0;r<NV;r++) {
                    if(r0+r<rows) {
                        const uint4* xp=reinterpret_cast<const uint4*>(x+(unsigned long long)(r0+r)*k+k0);
                        const uint4 xa=__ldg(xp),xb=__ldg(xp+1);
                        const unsigned xw[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
                        float v[16],sum=0.f;
                        #pragma unroll
                        for(unsigned i=0;i<8;i++) {
                            v[2*i]=__uint_as_float(xw[i]<<16); v[2*i+1]=__uint_as_float(xw[i]&0xffff0000u);
                        }
                        #pragma unroll
                        for(unsigned i=0;i<16;i++) sum+=v[i];
                        float d0=0.f,d1=0.f;
                        #pragma unroll
                        for(unsigned i=0;i<16;i++) {
                            d0=fmaf(v[i],wa[i],d0);
                            d1=fmaf(v[i],wb[i],d1);
                        }
                        acc[r][0]+=fmaf(sc0[j],d0,bi0[j]*sum);
                        acc[r][1]+=fmaf(sc1[j],d1,bi1[j]*sum);
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
            for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
            if(lane==0 && r0+r<rows && n0+col<n) {
                const unsigned at=(r0+r)*n+n0+col;y[at]=v;ybf[at]=fb(v);
            }
        }
    }
}
#define ENTRY_V2(NAME,NV,STEPS) \
extern "C" __global__ void __launch_bounds__(64) NAME(const unsigned short* x,const unsigned char* w, \
 const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf, \
 unsigned k,unsigned n,unsigned group,unsigned rows) { \
 affine8_v2_at<NV,STEPS>(blockIdx.x,blockIdx.y,x,w,s,b,y,ybf,k,n,group,rows); \
}
ENTRY_V2(affine8_v2_1,1,8)
ENTRY_V2(affine8_v2_2,2,8)
ENTRY_V2(affine8_v2_4,4,4)
ENTRY_V2(affine8_v2_8,8,4)
// The four-projection launch (GDN qkv/z/a/b, attention q/k/v/index) on the v2 body, NV rows per
// block row (affine8_multi4 is one row per block row).
#define MULTI_V2(NAME,NV,STEPS) \
extern "C" __global__ void __launch_bounds__(64) NAME(const unsigned short* x, \
    const unsigned char* w0,const unsigned short* s0,const unsigned short* b0,float* y0,unsigned short* yb0, \
    const unsigned char* w1,const unsigned short* s1,const unsigned short* b1,float* y1,unsigned short* yb1, \
    const unsigned char* w2,const unsigned short* s2,const unsigned short* b2,float* y2,unsigned short* yb2, \
    const unsigned char* w3,const unsigned short* s3,const unsigned short* b3,float* y3,unsigned short* yb3, \
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned g0,unsigned g1,unsigned g2,unsigned g3, \
    unsigned k,unsigned rows) { \
    unsigned bx=blockIdx.x; \
    const unsigned c0=(n0+3)/4,c1=(n1+3)/4,c2=(n2+3)/4; \
    if(bx<c0) { affine8_v2_at<NV,STEPS>(bx,blockIdx.y,x,w0,s0,b0,y0,yb0,k,n0,g0,rows); return; } \
    bx-=c0; \
    if(bx<c1) { affine8_v2_at<NV,STEPS>(bx,blockIdx.y,x,w1,s1,b1,y1,yb1,k,n1,g1,rows); return; } \
    bx-=c1; \
    if(bx<c2) { affine8_v2_at<NV,STEPS>(bx,blockIdx.y,x,w2,s2,b2,y2,yb2,k,n2,g2,rows); return; } \
    bx-=c2; \
    affine8_v2_at<NV,STEPS>(bx,blockIdx.y,x,w3,s3,b3,y3,yb3,k,n3,g3,rows); \
}
MULTI_V2(affine8_v2_multi4_1,1,8)
MULTI_V2(affine8_v2_multi4_4,4,4)
MULTI_V2(affine8_v2_multi4_8,8,4)
// R60 S4: affine8 projection on tensor cores for 2-16 token rows (decode verify widths; the
// start of the prefill GEMM). Exact codes, factored arithmetic as the QMV kernels: per 64-code
// sub-group the MMA forms sum(x*q) with q as exact BF16 integers (FP32 accumulate) and a ones-MMA
// forms sum(x); after each quantization group acc += scale * sum(x*q) + bias * sum(x) in FP32.
// Inside each 64-code sub-group K is permuted identically for both operands so that a thread's
// operands over the four k16 steps are contiguous: lane (g = lane/4, m = lane%4) loads codes
// [16m, 16m+16) of weight rows g and g+8 (one 16-byte load each) and x[16m, 16m+16) of token g
// (two 16-byte loads); k16 step s uses codes 16m+4s+{0,1} in fragment slots {2m,2m+1} and
// 16m+4s+{2,3} in slots {2m+8,2m+9}. Each warp: 16 output channels x 8T token rows; W warps split
// the groups and reduce through shared memory. Needs group % 64 == 0 and k % group == 0.
__device__ __forceinline__ unsigned a8m_pair(unsigned word,unsigned lo,unsigned hi) {
    // two code bytes of `word` (byte indices lo, hi) as an exact BF16x2 (low half = byte lo)
    const float f0=__uint_as_float(__byte_perm(word,0x4B000000u,0x7540u|lo))-8388608.f;
    const float f1=__uint_as_float(__byte_perm(word,0x4B000000u,0x7540u|hi))-8388608.f;
    return __byte_perm(__float_as_uint(f0),__float_as_uint(f1),0x7632u);
}
template<unsigned W,unsigned T,unsigned U>
__device__ __forceinline__ void affine8_mma_at(unsigned bx,unsigned by,const unsigned short* __restrict__ x,
    const unsigned char* __restrict__ w,const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,float* __restrict__ y,unsigned short* __restrict__ ybf,
    unsigned k,unsigned n,unsigned group,unsigned rows,float* smem) {
    // Work unit: one 64-code block of K (scale/bias of its group applied per block, which equals
    // per-group application up to FP32 rounding). Each warp takes U consecutive blocks per
    // iteration and issues all of their loads before any MMA (U=4: ~4 KB per warp in flight; the
    // 64-token prefill tile uses U=1 and relies on its many warps instead).
    const unsigned t=threadIdx.x,lane=t&31u,warp=t>>5,g=lane>>2,m=lane&3u;
    const unsigned ch0=bx*16,row0=by*(8*T),groups=k/group,blocks=k/64,per=group/64;
    const unsigned ra=ch0+g,rb=ch0+g+8;
    const bool oka=ra<n,okb=rb<n;
    const unsigned char* wa=w+(unsigned long long)(oka?ra:0u)*k+16*m;
    const unsigned char* wb=w+(unsigned long long)(okb?rb:0u)*k+16*m;
    const unsigned short* sa_=scales+(unsigned long long)(oka?ra:0u)*groups;
    const unsigned short* sb_=scales+(unsigned long long)(okb?rb:0u)*groups;
    const unsigned short* ba_=biases+(unsigned long long)(oka?ra:0u)*groups;
    const unsigned short* bb_=biases+(unsigned long long)(okb?rb:0u)*groups;
    const unsigned short* xr[T];
    bool okx[T];
    #pragma unroll
    for(unsigned h=0;h<T;h++) {
        const unsigned r=row0+8*h+g;okx[h]=r<rows;
        xr[h]=x+(unsigned long long)(okx[h]?r:0u)*k+16*m;
    }
    const unsigned ones=0x3F803F80u;
    float acc[T][4];
    #pragma unroll
    for(unsigned h=0;h<T;h++) { acc[h][0]=0.f;acc[h][1]=0.f;acc[h][2]=0.f;acc[h][3]=0.f; }
    for(unsigned b0i=warp*U;b0i<blocks;b0i+=W*U) {
        uint4 qa[U],qb[U],xa[U][T],xb[U][T];
        float sa[U],sb[U],ba[U],bb[U];
        #pragma unroll
        for(unsigned u=0;u<U;u++) {
            const unsigned bi=b0i+u;
            const bool in=bi<blocks;
            const unsigned kb=(in?bi:0u)*64,gi=(in?bi:0u)/per;
            qa[u]=in&&oka?__ldg(reinterpret_cast<const uint4*>(wa+kb)):make_uint4(0,0,0,0);
            qb[u]=in&&okb?__ldg(reinterpret_cast<const uint4*>(wb+kb)):make_uint4(0,0,0,0);
            sa[u]=in?bf(__ldg(sa_+gi)):0.f; ba[u]=in?bf(__ldg(ba_+gi)):0.f;
            sb[u]=in?bf(__ldg(sb_+gi)):0.f; bb[u]=in?bf(__ldg(bb_+gi)):0.f;
            #pragma unroll
            for(unsigned h=0;h<T;h++) {
                const uint4* p=reinterpret_cast<const uint4*>(xr[h]+kb);
                xa[u][h]=in&&okx[h]?__ldg(p):make_uint4(0,0,0,0);
                xb[u][h]=in&&okx[h]?__ldg(p+1):make_uint4(0,0,0,0);
            }
        }
        #pragma unroll
        for(unsigned u=0;u<U;u++) {
            float cg[T][4],cs[T][4];
            #pragma unroll
            for(unsigned h=0;h<T;h++) { cg[h][0]=cg[h][1]=cg[h][2]=cg[h][3]=0.f; cs[h][0]=cs[h][1]=cs[h][2]=cs[h][3]=0.f; }
            const unsigned qaw[4]={qa[u].x,qa[u].y,qa[u].z,qa[u].w},qbw[4]={qb[u].x,qb[u].y,qb[u].z,qb[u].w};
            #pragma unroll
            for(unsigned s=0;s<4;s++) {
                // codes 4s..4s+3 of this lane's 16 are bytes 0..3 of word s
                const unsigned a0=a8m_pair(qaw[s],0,1),a1=a8m_pair(qbw[s],0,1);
                const unsigned a2=a8m_pair(qaw[s],2,3),a3=a8m_pair(qbw[s],2,3);
                #pragma unroll
                for(unsigned h=0;h<T;h++) {
                    // x values 4s..4s+3 of this lane's 16 (BF16) are words 2s and 2s+1
                    const unsigned xw[8]={xa[u][h].x,xa[u][h].y,xa[u][h].z,xa[u][h].w,xb[u][h].x,xb[u][h].y,xb[u][h].z,xb[u][h].w};
                    const unsigned b0=xw[2*s],b1=xw[2*s+1];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(cg[h][0]),"+f"(cg[h][1]),"+f"(cg[h][2]),"+f"(cg[h][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(cs[h][0]),"+f"(cs[h][1]),"+f"(cs[h][2]),"+f"(cs[h][3])
                        : "r"(ones),"r"(ones),"r"(ones),"r"(ones),"r"(b0),"r"(b1));
                }
            }
            // C element j: channel g + (j>>1)*8, token 2m + (j&1) of the n8 tile
            #pragma unroll
            for(unsigned h=0;h<T;h++) {
                acc[h][0]+=fmaf(sa[u],cg[h][0],ba[u]*cs[h][0]);acc[h][1]+=fmaf(sa[u],cg[h][1],ba[u]*cs[h][1]);
                acc[h][2]+=fmaf(sb[u],cg[h][2],bb[u]*cs[h][2]);acc[h][3]+=fmaf(sb[u],cg[h][3],bb[u]*cs[h][3]);
            }
        }
    }
    #pragma unroll
    for(unsigned h=0;h<T;h++)
        #pragma unroll
        for(unsigned j=0;j<4;j++) smem[(warp*T+h)*128+(2*m+(j&1))*16+g+(j>>1)*8]=acc[h][j];
    __syncthreads();
    for(unsigned i=t;i<T*128;i+=W*32) {
        const unsigned h=i/128,e=i%128,tok=row0+8*h+e/16,ch=ch0+e%16;
        if(tok<rows && ch<n) {
            float v=0.f;
            #pragma unroll
            for(unsigned j=0;j<W;j++) v+=smem[(j*T+h)*128+e];
            const unsigned long long at=(unsigned long long)tok*n+ch;y[at]=v;ybf[at]=fb(v);
        }
    }
}
#define ENTRY_MMA(NAME,W,T,U) \
extern "C" __global__ void __launch_bounds__(W*32) NAME(const unsigned short* x,const unsigned char* w, \
 const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf, \
 unsigned k,unsigned n,unsigned group,unsigned rows) { \
 __shared__ float smem[W*T*128]; \
 affine8_mma_at<W,T,U>(blockIdx.x,blockIdx.y,x,w,s,b,y,ybf,k,n,group,rows,smem); \
}
ENTRY_MMA(affine8_mma_t1,4,1,4)
ENTRY_MMA(affine8_mma_t2,4,2,4)
ENTRY_MMA(affine8_mma_t8,4,8,1)
// Narrow projections (n/16 blocks too few to fill the card with four warps each): 8 or 16 warps
// split K inside a block, same arithmetic per 64-code block, reduced through shared memory.
ENTRY_MMA(affine8_mma_t1_w8,8,1,4)
ENTRY_MMA(affine8_mma_t1_w16,16,1,4)
ENTRY_MMA(affine8_mma_t2_w8,8,2,4)
ENTRY_MMA(affine8_mma_t2_w16,16,2,4)
#define MULTI_MMA(NAME,W,T,U) \
extern "C" __global__ void __launch_bounds__(W*32) NAME(const unsigned short* x, \
    const unsigned char* w0,const unsigned short* s0,const unsigned short* b0,float* y0,unsigned short* yb0, \
    const unsigned char* w1,const unsigned short* s1,const unsigned short* b1,float* y1,unsigned short* yb1, \
    const unsigned char* w2,const unsigned short* s2,const unsigned short* b2,float* y2,unsigned short* yb2, \
    const unsigned char* w3,const unsigned short* s3,const unsigned short* b3,float* y3,unsigned short* yb3, \
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned g0,unsigned g1,unsigned g2,unsigned g3, \
    unsigned k,unsigned rows) { \
    __shared__ float smem[W*T*128]; \
    unsigned bx=blockIdx.x; \
    const unsigned c0=(n0+15)/16,c1=(n1+15)/16,c2=(n2+15)/16; \
    if(bx<c0) { affine8_mma_at<W,T,U>(bx,blockIdx.y,x,w0,s0,b0,y0,yb0,k,n0,g0,rows,smem); return; } \
    bx-=c0; \
    if(bx<c1) { affine8_mma_at<W,T,U>(bx,blockIdx.y,x,w1,s1,b1,y1,yb1,k,n1,g1,rows,smem); return; } \
    bx-=c1; \
    if(bx<c2) { affine8_mma_at<W,T,U>(bx,blockIdx.y,x,w2,s2,b2,y2,yb2,k,n2,g2,rows,smem); return; } \
    bx-=c2; \
    affine8_mma_at<W,T,U>(bx,blockIdx.y,x,w3,s3,b3,y3,yb3,k,n3,g3,rows,smem); \
}
MULTI_MMA(affine8_mma_multi4_t1,4,1,4)
MULTI_MMA(affine8_mma_multi4_t2,4,2,4)
MULTI_MMA(affine8_mma_multi4_t8,4,8,1)
// R60 S4 (prefill v2): attn8_part for prefill rows (splits = 1): one block per (KV head, row), twelve
// warps = the twelve query heads sharing that KV head. Each 32-key tile of the row's selected keys is
// loaded and dequantized ONCE into shared memory as bf16 (exactly attn8_part's a8st(code*s+b)), then
// read by all twelve heads: attn8_part (and v1 of this kernel) dequantized every key once per head,
// and the dequant (I2F + bf16 rounding, ~80% of the instructions) was what bound a 2K prefill. The
// per-key dot order, the key order and the online-softmax order are attn8_part's, as is the
// partial/stats layout, so attn_merge is unchanged and the result is bit-identical.
__device__ __forceinline__ float a8cf(unsigned code){return __uint_as_float(0x4b000000u|code)-8388608.f;}
extern "C" __global__ void __launch_bounds__(384,3) attn8_rows(const a8bf* query,const unsigned char* kc,const unsigned char* vc,
    const a8bf* ks,const a8bf* kb,const a8bf* vs,const a8bf* vb,const unsigned* ids,float* partial,float* stats,
    unsigned offset,unsigned sparse){
  __shared__ __align__(16) a8bf sk[32*256],sv[32*256];
  const unsigned kh=blockIdx.x,r=blockIdx.y,tid=threadIdx.x,warp=tid>>5,lane=tid&31,h=kh*12+warp;
  const unsigned pos=offset+r,complete=(pos+1)/4;
  const unsigned selected=sparse?min(complete,512U)*4:0,tail=sparse?complete*4:0,L=selected+pos+1-tail;
  float q[8],acc[8];for(unsigned j=0;j<8;j++){q[j]=a8ld(query[(r*24+h)*256+lane*8+j]);acc[j]=0;}
  float m=-__int_as_float(0x7f800000),z=0;
  for(unsigned t0=0;t0<L;t0+=32){
    const unsigned tn=min(32U,L-t0);
    __syncthreads();
    // 16 codes per item: (K|V) x key x 16 chunks; scale/bias group = chunk/4 (64 codes)
    for(unsigned i=tid;i<tn*32;i+=384){
      const unsigned which=i/(tn*16),rem=i%(tn*16),u=rem/16,c=rem%16,vi=t0+u;
      const unsigned key=vi<selected?ids[r*512+vi/4]*4+vi%4:tail+vi-selected,row=key*2+kh;
      const uint4 w=*reinterpret_cast<const uint4*>((which?vc:kc)+row*256+c*16);
      const float s=a8ld((which?vs:ks)[row*4+c/4]),b=a8ld((which?vb:kb)[row*4+c/4]);
      const unsigned wd[4]={w.x,w.y,w.z,w.w};
      unsigned o[8];
      #pragma unroll
      for(unsigned j=0;j<8;j++){
        const unsigned lo=(wd[j/2]>>(16*(j&1)))&255u,hi=(wd[j/2]>>(16*(j&1)+8))&255u;
        o[j]=(unsigned)a8st(a8cf(lo)*s+b)|((unsigned)a8st(a8cf(hi)*s+b)<<16);
      }
      a8bf* dst=(which?sv:sk)+u*256+c*16;
      *reinterpret_cast<uint4*>(dst)=make_uint4(o[0],o[1],o[2],o[3]);
      *reinterpret_cast<uint4*>(dst+8)=make_uint4(o[4],o[5],o[6],o[7]);
    }
    __syncthreads();
    #pragma unroll 4
    for(unsigned u=0;u<tn;u++){
      const uint4 kw=*reinterpret_cast<const uint4*>(sk+u*256+lane*8);
      const unsigned kd[4]={kw.x,kw.y,kw.z,kw.w};
      float dot=0;
      #pragma unroll
      for(unsigned j=0;j<8;j++){const float kv=__uint_as_float(j&1?kd[j/2]&0xffff0000u:kd[j/2]<<16);dot+=q[j]*kv;}
      dot=a8sum(dot)*(1.f/16.f);
      const float nm=fmaxf(m,dot),factor=expf(m-nm),prob=expf(dot-nm);z=z*factor+prob;m=nm;
      const uint4 vw=*reinterpret_cast<const uint4*>(sv+u*256+lane*8);
      const unsigned vd[4]={vw.x,vw.y,vw.z,vw.w};
      #pragma unroll
      for(unsigned j=0;j<8;j++){const float vv=__uint_as_float(j&1?vd[j/2]&0xffff0000u:vd[j/2]<<16);acc[j]=acc[j]*factor+prob*vv;}
    }
  }
  const unsigned idx=r*24+h;if(lane==0){stats[idx*2]=m;stats[idx*2+1]=z;}
  for(unsigned j=0;j<8;j++)partial[idx*256+lane*8+j]=acc[j];
}
// R60 S4 (decode): attn8_rows with attn8_part's key splits: one block per (KV head, row, split),
// twelve warps = the twelve query heads of that KV head. The split's keys [lo, hi) are attn8_part's
// (per = ceil(L / splits)); each 32-key tile is dequantized once into shared memory and read by all
// twelve heads (attn8_part dequantizes every key once per head). Per-key arithmetic, key order,
// online-softmax order and the partial/stats layout ((r*24+h)*splits+sp) are attn8_part's:
// bit-identical, attn_merge unchanged.
extern "C" __global__ void __launch_bounds__(384,3) attn8_rowsp(const a8bf* query,const unsigned char* kc,const unsigned char* vc,
    const a8bf* ks,const a8bf* kb,const a8bf* vs,const a8bf* vb,const unsigned* ids,float* partial,float* stats,
    unsigned offset,unsigned sparse,unsigned splits){
  __shared__ __align__(16) a8bf sk[32*256],sv[32*256];
  const unsigned kh=blockIdx.x,r=blockIdx.y,sp=blockIdx.z,tid=threadIdx.x,warp=tid>>5,lane=tid&31,h=kh*12+warp;
  const unsigned pos=offset+r,complete=(pos+1)/4;
  const unsigned selected=sparse?min(complete,512U)*4:0,tail=sparse?complete*4:0,L=selected+pos+1-tail;
  const unsigned per=(L+splits-1)/splits,lo=sp*per,hi=min(L,lo+per);
  float q[8],acc[8];for(unsigned j=0;j<8;j++){q[j]=a8ld(query[(r*24+h)*256+lane*8+j]);acc[j]=0;}
  float m=-__int_as_float(0x7f800000),z=0;
  for(unsigned t0=lo;t0<hi;t0+=32){
    const unsigned tn=min(32U,hi-t0);
    __syncthreads();
    for(unsigned i=tid;i<tn*32;i+=384){
      const unsigned which=i/(tn*16),rem=i%(tn*16),u=rem/16,c=rem%16,vi=t0+u;
      const unsigned key=vi<selected?ids[r*512+vi/4]*4+vi%4:tail+vi-selected,row=key*2+kh;
      const uint4 w=*reinterpret_cast<const uint4*>((which?vc:kc)+row*256+c*16);
      const float s=a8ld((which?vs:ks)[row*4+c/4]),b=a8ld((which?vb:kb)[row*4+c/4]);
      const unsigned wd[4]={w.x,w.y,w.z,w.w};
      unsigned o[8];
      #pragma unroll
      for(unsigned j=0;j<8;j++){
        const unsigned lo8=(wd[j/2]>>(16*(j&1)))&255u,hi8=(wd[j/2]>>(16*(j&1)+8))&255u;
        o[j]=(unsigned)a8st(a8cf(lo8)*s+b)|((unsigned)a8st(a8cf(hi8)*s+b)<<16);
      }
      a8bf* dst=(which?sv:sk)+u*256+c*16;
      *reinterpret_cast<uint4*>(dst)=make_uint4(o[0],o[1],o[2],o[3]);
      *reinterpret_cast<uint4*>(dst+8)=make_uint4(o[4],o[5],o[6],o[7]);
    }
    __syncthreads();
    #pragma unroll 4
    for(unsigned u=0;u<tn;u++){
      const uint4 kw=*reinterpret_cast<const uint4*>(sk+u*256+lane*8);
      const unsigned kd[4]={kw.x,kw.y,kw.z,kw.w};
      float dot=0;
      #pragma unroll
      for(unsigned j=0;j<8;j++){const float kv=__uint_as_float(j&1?kd[j/2]&0xffff0000u:kd[j/2]<<16);dot+=q[j]*kv;}
      dot=a8sum(dot)*(1.f/16.f);
      const float nm=fmaxf(m,dot),factor=expf(m-nm),prob=expf(dot-nm);z=z*factor+prob;m=nm;
      const uint4 vw=*reinterpret_cast<const uint4*>(sv+u*256+lane*8);
      const unsigned vd[4]={vw.x,vw.y,vw.z,vw.w};
      #pragma unroll
      for(unsigned j=0;j<8;j++){const float vv=__uint_as_float(j&1?vd[j/2]&0xffff0000u:vd[j/2]<<16);acc[j]=acc[j]*factor+prob*vv;}
    }
  }
  const unsigned idx=(r*24+h)*splits+sp;if(lane==0){stats[idx*2]=m;stats[idx*2+1]=z;}
  for(unsigned j=0;j<8;j++)partial[idx*256+lane*8+j]=acc[j];
}
// R60 S4 (decode): hc8_norm_down3 for every row of a verify (rows <= 7) in one block per
// (column block, split): the down weights (and the HC norm/mix vectors) are loaded once into
// registers and applied to each row; hc8_norm_down3 ran a block per row and re-read them per row
// (~1.4 us per extra row per launch). Rows go through two shared buffers, the next row's input
// loads issued before the current row's arithmetic. Per-row arithmetic and reduction orders are
// hc8_norm_down3's: bit-identical outputs. Grid (R/4 + 8, 1, HC).
template<bool GEMV>
__device__ __forceinline__ void hc8r_nd_body(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned char* w,const unsigned short* scales,const unsigned short* biases,
    unsigned short* normalized,float* dpart,unsigned short* next,float* partial,
    unsigned group,int pending,unsigned rows,unsigned short (*sv)[HR_H],float (*ssum)[8]) {
  const unsigned t=threadIdx.x,lane=t&31u,rw=t>>5;
  const unsigned bx=blockIdx.x,s=blockIdx.z;
  const unsigned n0=bx*4+rw*2,groups=HR_F/group;
  const unsigned kbegin=s*HR_H;
  const unsigned jw=bx-HR_GEMV_BLOCKS;
  // scale | bias << 16 per (step, column); the norm weights stay in L1 (read per row)
  uint2 cp[GEMV?5:1][2][2];unsigned csb[GEMV?5:1][2][2];
  unsigned short nwr[GEMV?1:10],iwr[4][GEMV?1:10];
  if constexpr(GEMV) {
    #pragma unroll
    for(unsigned g=0;g<5;g++)
      #pragma unroll
      for(unsigned d=0;d<2;d++) {
        const unsigned k0=kbegin+g*512u+d*256u+lane*8u;
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const unsigned at=(n0+col)*groups+k0/group;
          csb[g][d][col]=(unsigned)scales[at]|((unsigned)biases[at]<<16);
          cp[g][d][col]=*reinterpret_cast<const uint2*>(w+(size_t)(n0+col)*HR_F+k0);
        }
      }
  } else {
    if(rw==0) {
      #pragma unroll
      for(unsigned i=0;i<10;i++) {
        const unsigned j=32u*jw+lane+256u*i;
        nwr[i]=nw[s*HR_H+j];
        #pragma unroll
        for(unsigned cc=0;cc<HR_HC;cc++) iwr[cc][i]=iw[(size_t)cc*HR_F+s*HR_H+j];
      }
    }
  }
  uint4 xr[5],br[5];
  float inj=0.f;
  auto load_row=[&](unsigned row){
    #pragma unroll
    for(unsigned q=0;q<5;q++) {
      const unsigned c=t+64u*q;
      xr[q]=*reinterpret_cast<const uint4*>(x+(size_t)row*HR_F+s*HR_H+c*8u);
      br[q]=pending?*reinterpret_cast<const uint4*>(block+(size_t)row*HR_H+c*8u):make_uint4(0u,0u,0u,0u);
    }
    inj=pending?hr_load(inject[row*HR_HC+s]):0.f;
  };
  auto store_row=[&](unsigned buf){
    #pragma unroll
    for(unsigned q=0;q<5;q++) {
      const unsigned c=t+64u*q;
      const unsigned xw[4]={xr[q].x,xr[q].y,xr[q].z,xr[q].w};
      const unsigned bw[4]={br[q].x,br[q].y,br[q].z,br[q].w};
      unsigned ow[4];
      #pragma unroll
      for(unsigned e2=0;e2<4;e2++) {
        unsigned lo,hi;
        { float v=hr_half_of(xw[e2],0); if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],0)*inj)); lo=hr_store(v); }
        { float v=hr_half_of(xw[e2],1); if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],1)*inj)); hi=hr_store(v); }
        ow[e2]=lo|(hi<<16);
      }
      *reinterpret_cast<uint4*>(sv[buf]+c*8u)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
    }
  };
  load_row(0);
  store_row(0);
  __syncthreads();
  for(unsigned row=0;row<rows;row++) {
    const unsigned buf=row&1u;
    if(row+1<rows) load_row(row+1);
    float sq[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(unsigned m=0;m<4;m++)
      #pragma unroll
      for(unsigned i=0;i<10;i++) { const float v=hr_load(sv[buf][t+64u*m+256u*i]);sq[m]=fmaf(v,v,sq[m]); }
    #pragma unroll
    for(unsigned m=0;m<4;m++) { const float v=hr_warp_sum(sq[m]); if(lane==0) ssum[buf][rw+2u*m]=v; }
    __syncthreads();
    float total=0;
    #pragma unroll
    for(unsigned q=0;q<8;q++) total+=ssum[buf][q];
    const float rs=rsqrtf(total/float(HR_H)+1e-6f);
    if constexpr(!GEMV) {
      if(rw==0) {
        float dot[HR_HC]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(unsigned i=0;i<10;i++) {
          const unsigned j=32u*jw+lane+256u*i;
          const float v=hr_load(sv[buf][j]);
          const float nv=hr_rb(hr_rb(v*rs)*hr_load(nwr[i]));
          normalized[(size_t)row*HR_F+s*HR_H+j]=hr_store(nv);
          if(pending) next[(size_t)row*HR_F+s*HR_H+j]=sv[buf][j];
          #pragma unroll
          for(unsigned cc=0;cc<HR_HC;cc++) dot[cc]=fmaf(nv,hr_load(iwr[cc][i]),dot[cc]);
        }
        #pragma unroll
        for(unsigned cc=0;cc<HR_HC;cc++) {
          const float v=hr_warp_sum(dot[cc]);
          if(lane==0) dpart[((row*HR_HC+s)*8u+jw)*HR_HC+cc]=v;
        }
      }
    } else {
      float acc[2]={0.f,0.f};
      #pragma unroll
      for(unsigned g=0;g<5;g++) {
        #pragma unroll
        for(unsigned d=0;d<2;d++) {
          const unsigned jj=g*512u+d*256u+lane*8u;
          const uint4 vq=*reinterpret_cast<const uint4*>(sv[buf]+jj);
          const uint4 cnw=*reinterpret_cast<const uint4*>(nw+kbegin+jj);
          const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w},ww[4]={cnw.x,cnw.y,cnw.z,cnw.w};
          float vals[8],sums=0.f;
          #pragma unroll
          for(unsigned i=0;i<8;i++) {
            const float nv=hr_rb(hr_rb(hr_half_of(vw[i>>1],i)*rs)*hr_half_of(ww[i>>1],i));
            vals[i]=nv;sums+=nv;
          }
          #pragma unroll
          for(unsigned col=0;col<2;col++) {
            const float sc=hr_load((unsigned short)csb[g][d][col]),bi=hr_load((unsigned short)(csb[g][d][col]>>16));
            const uint2 pk=cp[g][d][col];
            float dot=0.f;
            #pragma unroll
            for(unsigned i=0;i<4;i++) dot=fmaf(vals[i],float((pk.x>>(8*i))&255u),dot);
            #pragma unroll
            for(unsigned i=0;i<4;i++) dot=fmaf(vals[4+i],float((pk.y>>(8*i))&255u),dot);
            acc[col]+=fmaf(sc,dot,bi*sums);
          }
        }
      }
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        float v=acc[col];
        #pragma unroll
        for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
        if(lane==0) partial[((size_t)row*HR_SPLITS+s)*HR_R+n0+col]=v;
      }
    }
    if(row+1<rows) store_row(buf^1u);
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(64) hc8r_norm_down3(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned char* w,const unsigned short* scales,const unsigned short* biases,
    unsigned short* normalized,float* dpart,unsigned short* next,float* partial,
    unsigned group,int pending,unsigned rows) {
  __shared__ __align__(16) unsigned short sv[2][HR_H];
  __shared__ float ssum[2][8];
  if(blockIdx.x<HR_GEMV_BLOCKS) hc8r_nd_body<true>(x,nw,iw,block,inject,w,scales,biases,normalized,dpart,next,partial,group,pending,rows,sv,ssum);
  else hc8r_nd_body<false>(x,nw,iw,block,inject,w,scales,biases,normalized,dpart,next,partial,group,pending,rows,sv,ssum);
}
// R60 S4 (decode): hc8_front2 for every row of a verify in one block per column pair (all four
// streams): the up-projection weights are loaded once and applied to each row (hc8_front2 ran a
// block per row and re-read them per row); the next row's split partials and mix inputs are
// loaded before the current row's arithmetic. Per-row arithmetic is hc8_front2's: bit-identical.
// Grid (H/2, 1, 1).
extern "C" __global__ void __launch_bounds__(128) hc8r_front2(
    const float* partial,const float* dpart,float* parts,const unsigned char* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group,unsigned rows) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[2][HR_R];
  __shared__ float terms[2][HR_HC][2];
  uint2 wp[2][2];unsigned ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?*reinterpret_cast<const uint2*>(w+(size_t)(n0+col)*HR_R+k0):make_uint2(0u,0u);
    }
  }
  float pv[3][4];unsigned short nm0,nm1;
  auto load_row=[&](unsigned row){
    nm0=normalized[(size_t)row*HR_F+n0];nm1=normalized[(size_t)row*HR_F+n0+1];
    #pragma unroll
    for(unsigned q=0;q<3;q++) {
      const unsigned j=t+128u*q;
      #pragma unroll
      for(unsigned p=0;p<4;p++) pv[q][p]=j<HR_R?partial[((size_t)row*HR_SPLITS+p)*HR_R+j]:0.f;
    }
  };
  load_row(0);
  for(unsigned row=0;row<rows;row++) {
    const unsigned buf=row&1u;
    const unsigned short m0=nm0,m1=nm1;
    float cur[3][4];
    #pragma unroll
    for(unsigned q=0;q<3;q++)
      #pragma unroll
      for(unsigned p=0;p<4;p++) cur[q][p]=pv[q][p];
    if(row+1<rows) load_row(row+1);
    float pt[HR_HC];
    if(blockIdx.x==0 && t<HR_HC) {
      float dq[HR_HC][8];
      #pragma unroll
      for(unsigned hh=0;hh<HR_HC;hh++)
        #pragma unroll
        for(unsigned q=0;q<8;q++) dq[hh][q]=dpart[((row*HR_HC+hh)*8u+q)*HR_HC+t];
      #pragma unroll
      for(unsigned hh=0;hh<HR_HC;hh++) {
        float v=0;
        #pragma unroll
        for(unsigned q=0;q<8;q++) v+=dq[hh][q];
        pt[hh]=v;parts[(row*HR_HC+hh)*HR_HC+t]=v;
      }
    }
    #pragma unroll
    for(unsigned q=0;q<3;q++) {
      const unsigned j=t+128u*q;
      if(j<HR_R) {
        float v=0;
        #pragma unroll
        for(unsigned p=0;p<4;p++) v+=cur[q][p];
        const unsigned short low=hr_store(v);
        const float rounded=hr_load(low);
        const unsigned short a=hr_store(rounded*hr_sigmoid(rounded));
        local_act[buf][j]=a;
        if(blockIdx.x==0) {raw_down[row*HR_R+j]=v;down[row*HR_R+j]=low;activated[row*HR_R+j]=a;}
      }
    }
    if(blockIdx.x==0 && t<HR_HC) {
      float v=0;
      #pragma unroll
      for(unsigned hh=0;hh<HR_HC;hh++) v+=pt[hh];
      inject[row*HR_HC+t]=hr_store(2.f*hr_sigmoid(hr_rb(v)));
    }
    __syncthreads();
    float acc[2]={0.f,0.f};
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned k0=d*256u+lane*8u;
      if(k0<HR_R) {
        float vals[8];
        #pragma unroll
        for(unsigned i=0;i<8;i++) vals[i]=hr_load(local_act[buf][k0+i]);
        float sums=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) sums+=vals[i];
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
          const uint2 p=wp[d][col];
          float dot=0.f;
          #pragma unroll
          for(unsigned i=0;i<4;i++) dot=fmaf(vals[i],float((p.x>>(8*i))&255u),dot);
          #pragma unroll
          for(unsigned i=0;i<4;i++) dot=fmaf(vals[4+i],float((p.y>>(8*i))&255u),dot);
          acc[col]+=fmaf(sc,dot,bi*sums);
        }
      }
    }
    #pragma unroll
    for(unsigned c=0;c<2;c++) {
      float v=acc[c];
      #pragma unroll
      for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
      if(lane==0) {
        const unsigned at=row*HR_F+n0+c;
        raw_up[at]=v;up[at]=hr_store(v);
        terms[buf][h][c]=hr_rb(hr_sigmoid(hr_rb(v))*hr_load(c?m1:m0));
      }
    }
    __syncthreads();
    if(t<2) {
      float sum=0;
      #pragma unroll
      for(unsigned ss=0;ss<HR_HC;ss++) sum+=terms[buf][ss][t];
      mixed[row*HR_H+j0+t]=hr_store(hr_rb(sum)*.25f);
    }
  }
}
// R60 S4 (decode): qsa_select (kernels/lowbit_attention.cu) with 1024 threads per row. The
// original's one 256-thread block per row leaves the GPU nearly empty (4-5 blocks at a verify)
// while four radix passes and a scan-compaction walk every pooled score (~context/4): ~37 us per
// layer at 32K, growing with context. Same passes, same NaN/zero ordering, same compaction in index
// order (keys above the 512th value, then ties in index order): identical ids.
__device__ __forceinline__ unsigned q1k_ordered(float v){
  if(isnan(v))return 0xffffffffU;if(v==0.f)return 0x80000000U;unsigned u=__float_as_uint(v);return u&0x80000000U?~u:u|0x80000000U;
}
// exclusive block scan over 1024 threads; sh[32] holds the total afterwards
__device__ __forceinline__ unsigned q1k_scan(unsigned v,unsigned* sh){
  unsigned x=v;const unsigned lane=threadIdx.x&31u,warp=threadIdx.x>>5;
  for(unsigned k=1;k<32;k<<=1){unsigned p=__shfl_up_sync(0xffffffffU,x,k);if(lane>=k)x+=p;}
  if(lane==31)sh[warp]=x;
  __syncthreads();
  if(warp==0){
    const unsigned t=sh[lane];unsigned y=t;
    for(unsigned k=1;k<32;k<<=1){unsigned p=__shfl_up_sync(0xffffffffU,y,k);if(lane>=k)y+=p;}
    sh[lane]=y-t;if(lane==31)sh[32]=y;
  }
  __syncthreads();
  return sh[warp]+x-v;
}
extern "C" __global__ void __launch_bounds__(1024) qsa_select1k(const float* scores,unsigned* ids,unsigned offset,unsigned rows){
  const unsigned r=blockIdx.y,tid=threadIdx.x,vb=(offset+r+1)/4,nb=(offset+rows)/4;
  unsigned* out=ids+r*512;const float* sc=scores+r*nb;
  if(vb<=512){for(unsigned i=tid;i<512;i+=1024)out[i]=i<vb?i:0x7fffffffU;return;}
  __shared__ unsigned hist[256],choice[2],sh[33];
  unsigned prefix=0,need=512,fixed=0;
  for(unsigned pass=0;pass<4;pass++){
    if(tid<256)hist[tid]=0;__syncthreads();unsigned last=0xffffffffU,run=0,shift=24-pass*8;
    for(unsigned i=tid;i<vb;i+=1024){unsigned u=q1k_ordered(sc[i]);if(fixed && (u>>(32-fixed))!=prefix)continue;unsigned d=(u>>shift)&255;
      if(d==last)run++;else{if(run)atomicAdd(hist+last,run);last=d;run=1;}}
    if(run)atomicAdd(hist+last,run);__syncthreads();
    if(tid==0){unsigned above=0;for(int d=255;d>=0;d--){unsigned c=hist[d];if(above+c>=need){choice[0]=(unsigned)d;choice[1]=above;break;}above+=c;}}
    __syncthreads();prefix=(prefix<<8)|choice[0];need-=choice[1];fixed+=8;__syncthreads();
  }
  unsigned run_eq=0,run_keep=0;
  for(unsigned start=0;start<vb;start+=1024){const unsigned i=start+tid,u=i<vb?q1k_ordered(sc[i]):0;
    const unsigned eq=(i<vb && u==prefix),before_eq=q1k_scan(eq,sh),total_eq=sh[32];__syncthreads();
    const unsigned keep=(i<vb && (u>prefix || (eq && run_eq+before_eq<need)));
    const unsigned before=q1k_scan(keep,sh),total=sh[32];__syncthreads();
    if(keep && run_keep+before<512)out[run_keep+before]=i;
    run_eq+=total_eq;run_keep+=total;__syncthreads();
  }
}
