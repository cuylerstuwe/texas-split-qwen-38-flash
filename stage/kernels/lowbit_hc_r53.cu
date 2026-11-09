// GENERATED R38 combination: unchanged lowbit_hc.cu followed by hc_r38.cu.
// Regenerate by concatenation; do not edit here. See both sources and NOTICE.txt.
// Native CUDA HC glue. Port of this project's kernels/hc.metal, grounded in
// Sushi qwen4_hc.zig HC_FUSED_N_SOURCE (MIT; see NOTICE.txt). Packed affine4
// projections are separate resident calls. Optional residual write folds into
// the following stream RMSNorm/injection dot. No host state or allocations.
typedef unsigned short bf;
__device__ __forceinline__ float load(bf x) { return __uint_as_float((unsigned)x<<16); }
__device__ __forceinline__ bf store(float x) {
  bf out; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x)); return out;
}
__device__ __forceinline__ float rb(float x) { return load(store(x)); }
__device__ __forceinline__ float sigmoid(float x) { return rb(1.f/(1.f+expf(-x))); }
__device__ __forceinline__ float warp_sum(float x) {
  for(int s=16;s;s>>=1) x+=__shfl_down_sync(0xffffffffU,x,s); return x;
}
enum {H=2560,HC=4,F=10240,R=320};
extern "C" __global__ void hc_norm(const bf* x,const bf* nw,const bf* iw,
 const bf* block,const bf* inject,bf* normalized,float* parts,bf* next,int pending) {
  int h=blockIdx.x,row=blockIdx.y,t=threadIdx.x,sg=t/32,lane=t%32;
  __shared__ float sums[8],dots[8*4];
  float values[10],sum=0;
  for(int i=0;i<10;i++) {
    int j=t+256*i,at=row*F+h*H+j;
    float v=load(x[at]);
    if(pending) { v=rb(v+rb(load(block[row*H+j])*load(inject[row*HC+h]))); next[at]=store(v); }
    values[i]=v; sum=fmaf(v,v,sum);
  }
  sum=warp_sum(sum); if(lane==0) sums[sg]=sum;
  __syncthreads();
  float total=0; for(int s=0;s<8;s++) total+=sums[s];
  float rs=rsqrtf(total/float(H)+1e-6f),dot[4]={};
  for(int i=0;i<10;i++) {
    int j=h*H+t+256*i;
    float v=rb(rb(values[i]*rs)*load(nw[j])); normalized[row*F+j]=store(v);
    for(int c=0;c<HC;c++) dot[c]=fmaf(v,load(iw[c*F+j]),dot[c]);
  }
  for(int c=0;c<HC;c++) { float v=warp_sum(dot[c]); if(lane==0) dots[sg*HC+c]=v; }
  __syncthreads();
  if(t<HC) {float v=0;for(int s=0;s<8;s++)v+=dots[s*HC+t];parts[(row*HC+h)*HC+t]=v;}
}
extern "C" __global__ void hc_activate(const bf* down,const float* parts,
 bf* act,bf* inject,int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*(R+HC))return;
  int row=i/(R+HC),j=i%(R+HC);
  if(j<R) {float v=load(down[row*R+j]);act[row*R+j]=store(v*sigmoid(v));}
  else {int c=j-R;float raw=0;for(int h=0;h<HC;h++)raw+=parts[(row*HC+h)*HC+c];inject[row*HC+c]=store(2.f*sigmoid(rb(raw)));}
}
extern "C" __global__ void hc_mix(const bf* up,const bf* normalized,bf* mixed,int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*H)return;
  int row=i/H,j=i%H;float sum=0;
  for(int h=0;h<HC;h++) {int at=row*F+h*H+j;sum+=rb(sigmoid(load(up[at]))*load(normalized[at]));}
  mixed[i]=store(rb(sum)*0.25f);
}
extern "C" __global__ void hc_write(const bf* x,const bf* block,const bf* inject,
 bf* next,int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*F)return;
  int row=i/F,h=i%F/H,j=i%H;
  next[i]=store(load(x[i])+rb(load(block[row*H+j])*load(inject[row*HC+h])));
}

// R38 (Opus, direct): one hyper-connection site in two launches.
// Arithmetic follows this project's lowbit_hc.cu (hc_write/hc_norm/activate/mix),
// affine4_split.cu (factored split partial), affine4.cu (direct up projection)
// and the R37 hc_replicated_front experiment (hc_fold.cu); see NOTICE.txt.
//
// Launch 1, hc_r38_norm_down: grid (81, rows, 4 streams), 64 threads.
//   Blocks 0..79 each recompute their stream's RMS (reproducing hc_norm's
//   256-thread reduction tree exactly) and compute the split4 down partials for
//   four output columns from the normalized values, which never leave the block.
//   Block 80 of each stream writes normalized/next and the injection dots.
// Launch 2, hc_r38_front: grid (1280, rows), 128 threads. Each block recomputes
//   the 320-element split reduction and activation privately (R37's replicated
//   front), then runs the up projection for its two coordinates in all four
//   streams and the stream mix. Up-projection weights load before the front.
// Intended to match the unfused HC chain bit for bit.
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

extern "C" __global__ void __launch_bounds__(64) hc_r38_norm_down(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
    unsigned short* normalized,float* parts,unsigned short* next,float* partial,
    unsigned group,int pending) {
  const unsigned t=threadIdx.x,lane=t&31u,rw=t>>5;
  const unsigned bx=blockIdx.x,row=blockIdx.y,s=blockIdx.z;
  const bool gemv=bx<HR_GEMV_BLOCKS;
  __shared__ __align__(16) unsigned short sv[HR_H];
  __shared__ __align__(16) unsigned short snw[HR_H];
  __shared__ __align__(16) unsigned short siw[HR_HC*HR_H];
  __shared__ float ssum[8],sdots[8*HR_HC];
  // GEMV geometry: two output columns per warp, this stream's 2560 of K.
  const unsigned n0=bx*4+rw*2,groups=HR_F/group;
  const unsigned kbegin=s*HR_H;
  // Issue the first two K steps' weights before anything else (pipelined).
  unsigned cp[2][2]={},cs[2][2]={},cb[2][2]={};
  if(gemv) {
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned k0=kbegin+d*256u+lane*8u;
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const unsigned at=(n0+col)*groups+k0/group;
        cs[d][col]=scales[at];cb[d][col]=biases[at];
        cp[d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
      }
    }
  }
  // Phase 1: stage this stream (with any pending residual update) and its
  // norm weights; injection weights only for the norm block.
  uint4 xr[5],nr[5],br[5];
  #pragma unroll
  for(unsigned q=0;q<5;q++) {
    const unsigned c=t+64u*q;
    xr[q]=*reinterpret_cast<const uint4*>(x+(size_t)row*HR_F+s*HR_H+c*8u);
    nr[q]=*reinterpret_cast<const uint4*>(nw+s*HR_H+c*8u);
    br[q]=pending?*reinterpret_cast<const uint4*>(block+(size_t)row*HR_H+c*8u):make_uint4(0u,0u,0u,0u);
  }
  uint4 ir[4][5];
  if(!gemv) {
    #pragma unroll
    for(unsigned cc=0;cc<HR_HC;cc++)
      #pragma unroll
      for(unsigned q=0;q<5;q++)
        ir[cc][q]=*reinterpret_cast<const uint4*>(iw+(size_t)cc*HR_F+s*HR_H+(t+64u*q)*8u);
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
      {
        float v=hr_half_of(xw[e2],0);
        if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],0)*inj));
        lo=hr_store(v);
      }
      {
        float v=hr_half_of(xw[e2],1);
        if(pending) v=hr_rb(v+hr_rb(hr_half_of(bw[e2],1)*inj));
        hi=hr_store(v);
      }
      ow[e2]=lo|(hi<<16);
    }
    *reinterpret_cast<uint4*>(sv+c*8u)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
    *reinterpret_cast<uint4*>(snw+c*8u)=nr[q];
    if(!gemv) {
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) *reinterpret_cast<uint4*>(siw+cc*HR_H+c*8u)=ir[cc][q];
    }
  }
  __syncthreads();
  // hc_norm's tree: virtual thread vt=t+64m (0..255) sums j=vt+256i, i=0..9;
  // virtual warp vt/32 lives on real warp rw at lane t%32.
  float sq[4]={0.f,0.f,0.f,0.f};
  #pragma unroll
  for(unsigned m=0;m<4;m++)
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const float v=hr_load(sv[t+64u*m+256u*i]);sq[m]=fmaf(v,v,sq[m]);
    }
  #pragma unroll
  for(unsigned m=0;m<4;m++) {
    const float v=hr_warp_sum(sq[m]);
    if(lane==0) ssum[rw+2u*m]=v;
  }
  __syncthreads();
  float total=0;
  #pragma unroll
  for(unsigned q=0;q<8;q++) total+=ssum[q];
  const float rs=rsqrtf(total/float(HR_H)+1e-6f);
  if(!gemv) {
    // Norm block: unique writer of normalized/next and the injection dots.
    float dot[4][HR_HC];
    #pragma unroll
    for(unsigned m=0;m<4;m++)
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) dot[m][cc]=0.f;
    #pragma unroll
    for(unsigned m=0;m<4;m++)
      #pragma unroll
      for(unsigned i=0;i<10;i++) {
        const unsigned j=t+64u*m+256u*i;
        const float v=hr_load(sv[j]);
        const float nv=hr_rb(hr_rb(v*rs)*hr_load(snw[j]));
        normalized[(size_t)row*HR_F+s*HR_H+j]=hr_store(nv);
        if(pending) next[(size_t)row*HR_F+s*HR_H+j]=sv[j];
        #pragma unroll
        for(unsigned cc=0;cc<HR_HC;cc++) dot[m][cc]=fmaf(nv,hr_load(siw[cc*HR_H+j]),dot[m][cc]);
      }
    #pragma unroll
    for(unsigned m=0;m<4;m++)
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) {
        const float v=hr_warp_sum(dot[m][cc]);
        if(lane==0) sdots[(rw+2u*m)*HR_HC+cc]=v;
      }
    __syncthreads();
    if(t<HR_HC) {
      float v=0;
      #pragma unroll
      for(unsigned q=0;q<8;q++) v+=sdots[q*HR_HC+t];
      parts[(row*HR_HC+s)*HR_HC+t]=v;
    }
    return;
  }
  // GEMV blocks: split partial over this stream, affine4_split factored order.
  float acc[2]={0.f,0.f};
  for(unsigned g=0;g<5;g++) {
    unsigned np[2][2]={},ns[2][2]={},nb[2][2]={};
    if(g+1<5) {
      #pragma unroll
      for(unsigned d=0;d<2;d++) {
        const unsigned k0=kbegin+(g+1)*512u+d*256u+lane*8u;
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const unsigned at=(n0+col)*groups+k0/group;
          ns[d][col]=scales[at];nb[d][col]=biases[at];
          np[d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
        }
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned jj=g*512u+d*256u+lane*8u;
      const uint4 vq=*reinterpret_cast<const uint4*>(sv+jj);
      const uint4 nq=*reinterpret_cast<const uint4*>(snw+jj);
      const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w},ww[4]={nq.x,nq.y,nq.z,nq.w};
      float vals[8],sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        const float nv=hr_rb(hr_rb(hr_half_of(vw[i>>1],i)*rs)*hr_half_of(ww[i>>1],i));
        vals[i]=nv;sums+=nv;
      }
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)cs[d][col]),bi=hr_load((unsigned short)cb[d][col]);
        const unsigned p=cp[d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((p>>(4*i))&15u),dot);
        acc[col]+=fmaf(sc,dot,bi*sums);
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++)
      #pragma unroll
      for(unsigned col=0;col<2;col++) {cp[d][col]=np[d][col];cs[d][col]=ns[d][col];cb[d][col]=nb[d][col];}
  }
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    float v=acc[col];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
    if(lane==0) partial[((size_t)row*HR_SPLITS+s)*HR_R+n0+col]=v;
  }
}

extern "C" __global__ void __launch_bounds__(128) hc_r38_front(
    const float* partial,const float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][2];
  // Up-projection weights and the mix's normalized values first: independent.
  unsigned wp[2][2],ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
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
    #pragma unroll
    for(unsigned hh=0;hh<HR_HC;hh++) pt[hh]=parts[(row*HR_HC+hh)*HR_HC+t];
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
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
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


// v2 (one wave): 8 norm blocks per stream (one per hc_norm virtual warp) and no
// staged norm/injection weights, so a block needs ~5 KB of shared memory and
// all 352 blocks fit at once in the 32 KB carveout. Norm block j writes the
// normalized/next values of virtual warp j and its four injection-dot warp
// sums to dpart[(row*4+s)*8+j][c]; hc_r38_front2 sums them in hc_norm's order.
extern "C" __global__ void __launch_bounds__(64) hc_r38_norm_down2(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
    unsigned short* normalized,float* dpart,unsigned short* next,float* partial,
    unsigned group,int pending) {
  const unsigned t=threadIdx.x,lane=t&31u,rw=t>>5;
  const unsigned bx=blockIdx.x,row=blockIdx.y,s=blockIdx.z;
  const bool gemv=bx<HR_GEMV_BLOCKS;
  __shared__ __align__(16) unsigned short sv[HR_H];
  __shared__ float ssum[8];
  const unsigned n0=bx*4+rw*2,groups=HR_F/group;
  const unsigned kbegin=s*HR_H;
  unsigned cp[2][2]={},cs[2][2]={},cb[2][2]={};
  uint4 cn[2]={};
  if(gemv) {
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned k0=kbegin+d*256u+lane*8u;
      cn[d]=*reinterpret_cast<const uint4*>(nw+k0);
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const unsigned at=(n0+col)*groups+k0/group;
        cs[d][col]=scales[at];cb[d][col]=biases[at];
        cp[d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
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
  for(unsigned g=0;g<5;g++) {
    unsigned np[2][2]={},ns[2][2]={},nb[2][2]={};
    uint4 nn[2]={};
    if(g+1<5) {
      #pragma unroll
      for(unsigned d=0;d<2;d++) {
        const unsigned k0=kbegin+(g+1)*512u+d*256u+lane*8u;
        nn[d]=*reinterpret_cast<const uint4*>(nw+k0);
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const unsigned at=(n0+col)*groups+k0/group;
          ns[d][col]=scales[at];nb[d][col]=biases[at];
          np[d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
        }
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned jj=g*512u+d*256u+lane*8u;
      const uint4 vq=*reinterpret_cast<const uint4*>(sv+jj);
      const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w},ww[4]={cn[d].x,cn[d].y,cn[d].z,cn[d].w};
      float vals[8],sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        const float nv=hr_rb(hr_rb(hr_half_of(vw[i>>1],i)*rs)*hr_half_of(ww[i>>1],i));
        vals[i]=nv;sums+=nv;
      }
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)cs[d][col]),bi=hr_load((unsigned short)cb[d][col]);
        const unsigned pk=cp[d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((pk>>(4*i))&15u),dot);
        acc[col]+=fmaf(sc,dot,bi*sums);
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      cn[d]=nn[d];
      #pragma unroll
      for(unsigned col=0;col<2;col++) {cp[d][col]=np[d][col];cs[d][col]=ns[d][col];cb[d][col]=nb[d][col];}
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

extern "C" __global__ void __launch_bounds__(64) hc_r38_norm_down3(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
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
  unsigned cp[5][2][2],cs[5][2][2],cb[5][2][2];
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
          cp[g][d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
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
        const unsigned pk=cp[g][d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((pk>>(4*i))&15u),dot);
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


extern "C" __global__ void __launch_bounds__(128) hc_r38_front2(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][2];
  // Up-projection weights and the mix's normalized values first: independent.
  unsigned wp[2][2],ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
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
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
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

extern "C" __global__ void __launch_bounds__(64) r39pdl_hc_norm_down3(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
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
  unsigned cp[5][2][2],cs[5][2][2],cb[5][2][2];
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
          cp[g][d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
        }
      }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
        const unsigned pk=cp[g][d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((pk>>(4*i))&15u),dot);
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


extern "C" __global__ void __launch_bounds__(128) r39pdl_hc_front2(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][2];
  // Up-projection weights and the mix's normalized values first: independent.
  unsigned wp[2][2],ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
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

// GENERATED by tools/build_r52_hc_source.py
extern "C" __global__ void __launch_bounds__(64) r39pdl_hc_norm_down4(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
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
  unsigned cp[5][2][2],cs[5][2][2],cb[5][2][2];
  uint4 tn[5];
  if(gemv) {
    #pragma unroll
    for(unsigned g=0;g<5;g++)
      #pragma unroll
      for(unsigned d=0;d<2;d++) {
        const unsigned k0=kbegin+g*512u+d*256u+lane*8u;
        (void)0;
        #pragma unroll
        for(unsigned col=0;col<2;col++) {
          const unsigned at=(n0+col)*groups+k0/group;
          cs[g][d][col]=scales[at];cb[g][d][col]=biases[at];
          cp[g][d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
        }
      }
  }
  if(gemv) {
    #pragma unroll
    for(unsigned q=0;q<5;q++) tn[q]=*reinterpret_cast<const uint4*>(nw+kbegin+q*512u+t*8u);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
  if(gemv) {
    // R52: normalize each element ONCE per block (same rounding), in place.
    #pragma unroll
    for(unsigned q=0;q<5;q++) {
      const unsigned c=t+64u*q;
      const uint4 vq=*reinterpret_cast<const uint4*>(sv+c*8u);
      const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w},ww[4]={tn[q].x,tn[q].y,tn[q].z,tn[q].w};
      unsigned ow[4];
      #pragma unroll
      for(unsigned e2=0;e2<4;e2++) {
        const unsigned lo=hr_store(hr_rb(hr_half_of(vw[e2],0)*rs)*hr_half_of(ww[e2],0));
        const unsigned hi=hr_store(hr_rb(hr_half_of(vw[e2],1)*rs)*hr_half_of(ww[e2],1));
        ow[e2]=lo|(hi<<16);
      }
      *reinterpret_cast<uint4*>(sv+c*8u)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
    }
    __syncthreads();
  }
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
      const unsigned vw[4]={vq.x,vq.y,vq.z,vq.w};
      float vals[8],sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        const float nv=hr_half_of(vw[i>>1],i);
        vals[i]=nv;sums+=nv;
      }
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)cs[g][d][col]),bi=hr_load((unsigned short)cb[g][d][col]);
        const unsigned pk=cp[g][d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((pk>>(4*i))&15u),dot);
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
extern "C" __global__ void __launch_bounds__(128) r39pdl_hc_front2p(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group,const uint4* pump,unsigned pump16) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*2,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][2];
  for(unsigned line=blockIdx.x*128u+threadIdx.x;line*8u<pump16;line+=gridDim.x*128u)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(pump+line*8u));
  // Up-projection weights and the mix's normalized values first: independent.
  unsigned wp[2][2],ws[2][2],wb[2][2];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
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

extern "C" __global__ void __launch_bounds__(128) r39pdl_hc_front8(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*8,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][8];
  unsigned wp[2][8],ws[2][8],wb[2][8];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<8;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint4 nmq=*reinterpret_cast<const uint4*>(normalized+(size_t)row*HR_F+n0);
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
  float acc[8];
  #pragma unroll
  for(unsigned c=0;c<8;c++) acc[c]=0.f;
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    if(k0<HR_R) {
      float vals[8];
      #pragma unroll
      for(unsigned i=0;i<8;i++) vals[i]=hr_load(local_act[k0+i]);
      #pragma unroll
      for(unsigned col=0;col<8;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
      }
    }
  }
  const unsigned nmw[4]={nmq.x,nmq.y,nmq.z,nmq.w};
  #pragma unroll
  for(unsigned c=0;c<8;c++) {
    float v=acc[c];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
    if(lane==0) {
      const unsigned at=row*HR_F+n0+c;
      raw_up[at]=v;up[at]=hr_store(v);
      const unsigned short nm=(unsigned short)((c&1u)?(nmw[c>>1]>>16):(nmw[c>>1]&0xffffu));
      terms[h][c]=hr_rb(hr_sigmoid(hr_rb(v))*hr_load(nm));
    }
  }
  __syncthreads();
  if(t<8) {
    float sum=0;
    #pragma unroll
    for(unsigned ss=0;ss<HR_HC;ss++) sum+=terms[ss][t];
    mixed[row*HR_H+j0+t]=hr_store(hr_rb(sum)*.25f);
  }
}

// GENERATED by tools/build_r53_hc_source.py
extern "C" __global__ void r39pdl_hc_write(const bf* x,const bf* block,const bf* inject,
 bf* next,int rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*F)return;
  int row=i/F,h=i%F/H,j=i%H;
  next[i]=store(load(x[i])+rb(load(block[row*H+j])*load(inject[row*HC+h])));
}

extern "C" __global__ void r39pdl_hc_write_z(const bf* x,const bf* block,const bf* inject,
 bf* next,int rows,unsigned* zctr) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  { const int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z<344 && z!=4) zctr[64+z]=0u;
    if(z==0) zctr[1000]=__ldcg(zctr+1000)+1u; }
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*F)return;
  int row=i/F,h=i%F/H,j=i%H;
  next[i]=store(load(x[i])+rb(load(block[row*H+j])*load(inject[row*HC+h])));
}

// GENERATED by tools/build_r53_hc_source.py (norm_down5)
extern "C" __global__ void __launch_bounds__(64) r39pdl_hc_norm_down5(
    const unsigned short* x,const unsigned short* nw,const unsigned short* iw,
    const unsigned short* block,const unsigned short* inject,
    const unsigned* w,const unsigned short* scales,const unsigned short* biases,
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
  unsigned cp[5][2][2],cs[5][2][2],cb[5][2][2];
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
          cp[g][d][col]=w[(size_t)(n0+col)*(HR_F/8u)+k0/8u];
        }
      }
  }
  unsigned short nwr[10],iwr[4][10];
  if(!gemv && rw==0) {
    const unsigned jw=bx-HR_GEMV_BLOCKS;          // virtual warp 0..7
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const unsigned j=32u*jw+lane+256u*i;
      nwr[i]=nw[s*HR_H+j];
      #pragma unroll
      for(unsigned cc=0;cc<HR_HC;cc++) iwr[cc][i]=iw[(size_t)cc*HR_F+s*HR_H+j];
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
        const unsigned pk=cp[g][d][col];
        float dot=0.f;
        #pragma unroll
        for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((pk>>(4*i))&15u),dot);
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

// GENERATED by tools/build_r53_hc_source.py (front4)
extern "C" __global__ void __launch_bounds__(128) r39pdl_hc_front4(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*4,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][4];
  unsigned wp[2][4],ws[2][4],wb[2][4];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<4;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint2 nmq=*reinterpret_cast<const uint2*>(normalized+(size_t)row*HR_F+n0);
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
  float acc[4];
  #pragma unroll
  for(unsigned c=0;c<4;c++) acc[c]=0.f;
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    if(k0<HR_R) {
      float vals[8];
      #pragma unroll
      for(unsigned i=0;i<8;i++) vals[i]=hr_load(local_act[k0+i]);
      #pragma unroll
      for(unsigned col=0;col<4;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
      }
    }
  }
  const unsigned nmw[2]={nmq.x,nmq.y};
  #pragma unroll
  for(unsigned c=0;c<4;c++) {
    float v=acc[c];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
    if(lane==0) {
      const unsigned at=row*HR_F+n0+c;
      raw_up[at]=v;up[at]=hr_store(v);
      const unsigned short nm=(unsigned short)((c&1u)?(nmw[c>>1]>>16):(nmw[c>>1]&0xffffu));
      terms[h][c]=hr_rb(hr_sigmoid(hr_rb(v))*hr_load(nm));
    }
  }
  __syncthreads();
  if(t<4) {
    float sum=0;
    #pragma unroll
    for(unsigned ss=0;ss<HR_HC;ss++) sum+=terms[ss][t];
    mixed[row*HR_H+j0+t]=hr_store(hr_rb(sum)*.25f);
  }
}

// GENERATED by tools/build_r53_hc_source.py (front4, trigger after the wait)
extern "C" __global__ void __launch_bounds__(128) r39pdl_hc_front4_trig(
    const float* partial,const float* dpart,float* parts,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,const unsigned short* normalized,
    float* raw_down,unsigned short* down,unsigned short* activated,unsigned short* inject,
    float* raw_up,unsigned short* up,unsigned short* mixed,unsigned group) {
  const unsigned t=threadIdx.x,lane=t&31u,h=t>>5;
  const unsigned row=blockIdx.y,j0=blockIdx.x*4,n0=h*HR_H+j0,groups=HR_R/group;
  __shared__ __align__(16) unsigned short local_act[HR_R];
  __shared__ float terms[HR_HC][4];
  unsigned wp[2][4],ws[2][4],wb[2][4];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<4;col++) {
      const bool ok=k0<HR_R;
      const unsigned at=(n0+col)*groups+k0/group;
      ws[d][col]=ok?scales[at]:0u;wb[d][col]=ok?biases[at]:0u;
      wp[d][col]=ok?w[(size_t)(n0+col)*(HR_R/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint2 nmq=*reinterpret_cast<const uint2*>(normalized+(size_t)row*HR_F+n0);
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
  float acc[4];
  #pragma unroll
  for(unsigned c=0;c<4;c++) acc[c]=0.f;
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned k0=d*256u+lane*8u;
    if(k0<HR_R) {
      float vals[8];
      #pragma unroll
      for(unsigned i=0;i<8;i++) vals[i]=hr_load(local_act[k0+i]);
      #pragma unroll
      for(unsigned col=0;col<4;col++) {
        const float sc=hr_load((unsigned short)ws[d][col]),bi=hr_load((unsigned short)wb[d][col]);
        const unsigned p=wp[d][col];
        #pragma unroll
        for(unsigned i=0;i<8;i++) {
          const float decoded=fmaf(sc,float((p>>(4*i))&15u),bi);
          acc[col]=fmaf(vals[i],decoded,acc[col]);
        }
      }
    }
  }
  const unsigned nmw[2]={nmq.x,nmq.y};
  #pragma unroll
  for(unsigned c=0;c<4;c++) {
    float v=acc[c];
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
    if(lane==0) {
      const unsigned at=row*HR_F+n0+c;
      raw_up[at]=v;up[at]=hr_store(v);
      const unsigned short nm=(unsigned short)((c&1u)?(nmw[c>>1]>>16):(nmw[c>>1]&0xffffu));
      terms[h][c]=hr_rb(hr_sigmoid(hr_rb(v))*hr_load(nm));
    }
  }
  __syncthreads();
  if(t<4) {
    float sum=0;
    #pragma unroll
    for(unsigned ss=0;ss<HR_HC;ss++) sum+=terms[ss][t];
    mixed[row*HR_H+j0+t]=hr_store(hr_rb(sum)*.25f);
  }
}
