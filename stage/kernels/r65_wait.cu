// R65: grid-dependency-wait variants of three small stage kernels, for the shared-expert side branch
// (cuda_stage.rs r65_side_branches, driver tinynv_r65_edge). The bodies are lowbit_shared.cu's shared_activate and
// shared_finish and moe_join.cu's moe_add_bf16, unchanged, built with the same options; the one addition is
// griddepcontrol.wait at entry. With a latch named in the launch's descriptor (WAIT_ON_LATCH) it blocks until that
// latch is acquired, i.e. until the side-branch producer has completed; without one it returns at once.
// R38/R44 (2026-11-05): the descriptor latch alone does not hold a grid - only this wait does.
#define R65_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")

// ---- lowbit_shared.cu (Sushi transformer.zig silu/computeGeglu/moeSharedGateTail, MIT, NOTICE.txt) ----
typedef unsigned short bf;
__device__ __forceinline__ float load(bf x) { return __uint_as_float((unsigned)x<<16); }
__device__ __forceinline__ bf store(float x) {
  unsigned u=__float_as_uint(x);
  return (bf)((u+0x7fffU+((u>>16)&1U))>>16);
}
__device__ __forceinline__ float roundbf(float x) { return load(store(x)); }
__device__ __forceinline__ float sig(float x,int arm) {
  if(arm) return 1.0f/(1.0f+expf(-x));
  float small=roundbf(1.0f/roundbf(1.0f+roundbf(expf(fabsf(x)))));
  return x<0?small:roundbf(1.0f-small);
}
extern "C" __global__ void r65w_shared_activate(const bf* gate,const bf* up,bf* act,
    int rows,int intermediate,int combined,int arm) {
  R65_WAIT();
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j>=rows*intermediate) return;
  int r=j/intermediate,c=j%intermediate;
  float g=load(gate[combined?r*2*intermediate+c:j]);
  float u=load(combined?gate[r*2*intermediate+intermediate+c]:up[j]);
  float a=g*sig(g,arm);
  act[j]=store((arm?a:roundbf(a))*u);
}
extern "C" __global__ void r65w_shared_finish(const bf* x,const bf* weight,const bf* down,
    bf* out,float* raw,bf* logits,bf* sigmoid,int hidden,int arm) {
  R65_WAIT();
  __shared__ float sums[4];
  int r=blockIdx.x,t=threadIdx.x;
  float sum=0;
  for(int k=t;k<hidden;k+=128) sum=fmaf(load(x[r*hidden+k]),load(weight[k]),sum);
  for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
  if((t&31)==0) sums[t>>5]=sum;
  __syncthreads();
  if(t<32) {
    sum=t<4?sums[t]:0;
    for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
    if(t==0) {
      raw[r]=sum;
      bf logit=store(sum);
      float p=sig(load(logit),arm);
      logits[r]=logit; sigmoid[r]=store(p); sums[0]=p;
    }
  }
  __syncthreads();
  for(int k=t;k<hidden;k+=128) out[r*hidden+k]=store(load(down[r*hidden+k])*sums[0]);
}

// ---- moe_join.cu ----
__device__ __forceinline__ float bf_join(unsigned short h) { return __uint_as_float((unsigned)h<<16); }
extern "C" __global__ void r65w_moe_add_bf16(const unsigned short* routed,
    const unsigned short* shared, unsigned short* out, unsigned count) {
    R65_WAIT();
    unsigned at=blockIdx.x*128+threadIdx.x;
    if(at<count) {
        float v=bf_join(routed[at])+bf_join(shared[at]);
        unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(v)); out[at]=h;
    }
}
