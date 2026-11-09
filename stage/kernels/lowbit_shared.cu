// Native lowbit shared expert glue. Source semantics: Sushi transformer.zig
// silu/computeGeglu/moeSharedGateTail, MIT (NOTICE.txt).
// No host runtime or allocations. Staged BF16 is a semantic oracle control,
// not a claim of bit identity with Sushi's Metal-generated sigmoid table.
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
// combined=1 consumes [row,gate640|up640] directly, with no split/copy kernel.
extern "C" __global__ void shared_activate(const bf* gate,const bf* up,bf* act,
    int rows,int intermediate,int combined,int arm) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j>=rows*intermediate) return;
  int r=j/intermediate,c=j%intermediate;
  float g=load(gate[combined?r*2*intermediate+c:j]);
  float u=load(combined?gate[r*2*intermediate+intermediate+c]:up[j]);
  float a=g*sig(g,arm);
  act[j]=store((arm?a:roundbf(a))*u);
}
// One 128-thread block per input row; scalar BF16 gate dot and output gating
// share one launch. Diagnostic raw/logit/sigmoid stores are included in timing.
extern "C" __global__ void shared_finish(const bf* x,const bf* weight,const bf* down,
    bf* out,float* raw,bf* logits,bf* sigmoid,int hidden,int arm) {
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
