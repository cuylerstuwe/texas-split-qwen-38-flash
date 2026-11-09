// R37: extends the retained R36 HC-up/mix with small-front fusion.
// Actual producer/consumer fusion: packed HC-up -> BF16 sigmoid -> stream mix.
// Four warps own the four HC streams for the same two output coordinates.
// Source math follows this project's affine4.cu/lowbit_hc.cu (see NOTICE.txt).
// Retain FP32/BF16 projection diagnostics for matched component checks.
typedef unsigned short bf;
__device__ __forceinline__ float load(bf v) {return __uint_as_float((unsigned)v<<16);}
__device__ __forceinline__ bf store(float v) {
  bf out;asm("cvt.rn.bf16.f32 %0, %1;":"=h"(out):"f"(v));return out;
}
__device__ __forceinline__ float rb(float x) {return load(store(x));}
__device__ __forceinline__ float sig(float x) {return rb(1.f/(1.f+expf(-x)));}
enum {H=2560,F=10240,K=320,HC=4};
template<int NV,bool Factored> __device__ void up_mix(const bf* x,const unsigned* w,
 const bf* scales,const bf* biases,const bf* normalized,float* raw,bf* up,bf* mixed,
 unsigned group,unsigned rows) {
  unsigned lane=threadIdx.x&31, h=threadIdx.x>>5;
  unsigned j0=blockIdx.x*2,n0=h*H+j0,r0=blockIdx.y*NV,groups=K/group;
  float acc[NV][2]={};
  __shared__ float terms[NV][HC][2];
  for(unsigned base=0;base<K;base+=256) {
    unsigned k0=base+lane*8;
    float vals[NV][8]={},sums[NV]={};
    #pragma unroll
    for(unsigned r=0;r<NV;r++) {
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        float a=(r0+r<rows && k0+i<K)?load(x[(r0+r)*K+k0+i]):0.f;
        vals[r][i]=a;sums[r]+=a;
      }
    }
    if(k0<K) {
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        unsigned at=(n0+col)*groups+k0/group;
        float s=load(scales[at]),b=load(biases[at]);
        unsigned packed=w[(n0+col)*(K/8)+k0/8];
        #pragma unroll
        for(unsigned r=0;r<NV;r++) {
          if constexpr(Factored) {
            float dot=0.f;
            #pragma unroll
            for(unsigned i=0;i<8;i++) dot=fmaf(vals[r][i],float((packed>>(4*i))&15u),dot);
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
    for(unsigned c=0;c<2;c++) {
      float v=acc[r][c];
      #pragma unroll
      for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffU,v,delta);
      if(lane==0 && r0+r<rows) {
        unsigned at=(r0+r)*F+n0+c;
        raw[at]=v; up[at]=store(v);
        terms[r][h][c]=rb(sig(rb(v))*load(normalized[at]));
      }
    }
  }
  __syncthreads();
  if(threadIdx.x<2) {
    #pragma unroll
    for(unsigned r=0;r<NV;r++) if(r0+r<rows) {
      float sum=0;
      #pragma unroll
      for(unsigned s=0;s<HC;s++) sum+=terms[r][s][threadIdx.x];
      mixed[(r0+r)*H+j0+threadIdx.x]=store(rb(sum)*.25f);
    }
  }
}
#define ENTRY(NAME,NV,FAC) \
extern "C" __global__ void NAME(const bf* x,const unsigned* w,const bf* s,const bf* b, \
 const bf* norm,float* raw,bf* up,bf* mixed,unsigned group,unsigned rows) { \
 up_mix<NV,FAC>(x,w,s,b,norm,raw,up,mixed,group,rows); }
ENTRY(hc_upmix1,1,false)
ENTRY(hc_upmix4,4,true)

extern "C" __global__ void hc_reduce_activate(const float* partial,const float* parts,
 float* raw,bf* lower,bf* act,bf* inject,unsigned rows) {
  unsigned at=blockIdx.x*blockDim.x+threadIdx.x;
  if(at>=rows*(K+HC))return;
  unsigned row=at/(K+HC),j=at%(K+HC);
  if(j<K) {
    float v=0;for(unsigned p=0;p<4;p++)v+=partial[(row*4+p)*K+j];
    raw[row*K+j]=v;lower[row*K+j]=store(v);
    v=rb(v);act[row*K+j]=store(v*sig(v));
  } else {
    float v=0;for(unsigned h=0;h<HC;h++)v+=parts[(row*HC+h)*HC+j-K];
    inject[row*HC+j-K]=store(2.f*sig(rb(v)));
  }
}
extern "C" __global__ void hc_replicated_front(const float* partial,const float* parts,
 const unsigned* w,const bf* scales,const bf* biases,const bf* normalized,
 float* raw_down,bf* down,bf* activated,bf* inject,float* raw_up,bf* up,bf* mixed,
 unsigned group) {
  // One row, unchanged packed weights. The tiny dependency is recomputed
  // privately instead of publishing a grid-wide activation dependency.
  __shared__ bf local_act[K];
  for(unsigned j=threadIdx.x;j<K;j+=blockDim.x) {
    float v=0;for(unsigned p=0;p<4;p++)v+=partial[p*K+j];
    bf low=store(v);float rounded=load(low);bf a=store(rounded*sig(rounded));
    local_act[j]=a;
    if(blockIdx.x==0) {raw_down[j]=v;down[j]=low;activated[j]=a;}
  }
  if(blockIdx.x==0 && threadIdx.x<HC) {
    float v=0;for(unsigned h=0;h<HC;h++)v+=parts[h*HC+threadIdx.x];
    inject[threadIdx.x]=store(2.f*sig(rb(v)));
  }
  __syncthreads();
  up_mix<1,false>(local_act,w,scales,biases,normalized,raw_up,up,mixed,group,1);
}
