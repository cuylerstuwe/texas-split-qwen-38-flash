// GENERATED R38 combination: unchanged lowbit_router.cu followed by router_r38_tail.cu.
// Regenerate by concatenation; do not edit here. See both sources and NOTICE.txt.
// Native512-expert routing. BF16 projection/logit contract and score boundaries
// follow Sushi transformer.zig moeRouterSource(.softmax), MIT (NOTICE.txt).
// New CUDA reduction/selection implementation; no host routing or model runtime.
typedef unsigned short bf;
__device__ __forceinline__ float load(bf x) {return __uint_as_float((unsigned)x<<16);}
__device__ __forceinline__ bf store(float x) {
  unsigned u=__float_as_uint(x);return (bf)((u+0x7fffU+((u>>16)&1U))>>16);
}
__device__ __forceinline__ float rb(float x) {return load(store(x));}
// One warp per output, four outputs per CTA, independent BF16 input rows.
extern "C" __global__ void router_qmv(const bf* x,const bf* w,float* y) {
  int r=blockIdx.y,lane=threadIdx.x&31,e=blockIdx.x*4+(threadIdx.x>>5);
  float acc=0;
  for(int k=lane;k<2560;k+=32) acc=fmaf(load(x[r*2560+k]),load(w[e*2560+k]),acc);
  for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
  if(lane==0)y[r*512+e]=acc;
}
// One warp per row. Each lane retains16 candidate logits. Selection uses the
// lowest original expert ID on ties, before any probability rounding.
// Arm0: full softmax->BF16 probabilities, sequentialBF16 selected sum/divide.
// Arm1: selected-onlyFP32 softmax, an explicitly named arithmetic candidate.
extern "C" __global__ void router_select(const float* raw,bf* logits,unsigned* ids,float* scores,int arm) {
  int r=blockIdx.x,lane=threadIdx.x;
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=raw[r*512+e];bf h=store(v);logits[r*512+e]=h;
    bool bad=!isfinite(v)||!isfinite(load(h));invalid|=bad;
    // A finite sentinel keeps every selected ID in-range and unique even on a
    // failed row. Scores below become NaN so a consumer cannot publish it.
    key[j]=bad?-3.402823466e38f:load(h);mx=fmaxf(mx,key[j]);
  }
  invalid=__any_sync(0xffffffffU,invalid);
  for(int s=16;s;s>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffU,mx,s));
  float norm=0;
  if(arm==0) {
    #pragma unroll
    for(int j=0;j<16;j++)norm+=expf(key[j]-mx);
    for(int s=16;s;s>>=1)norm+=__shfl_xor_sync(0xffffffffU,norm,s);
    norm=1.0f/norm;
  }
  unsigned selected=0;
  float selected_logit=0;
  for(int rank=0;rank<10;rank++) {
    float best=-__int_as_float(0x7f800000);unsigned id=0xffffffffU;
    #pragma unroll
    for(int j=0;j<16;j++) {
      unsigned e=j*32+lane;
      if(key[j]>best || (key[j]==best && e<id)){best=key[j];id=e;}
    }
    for(int s=16;s;s>>=1) {
      float v=__shfl_xor_sync(0xffffffffU,best,s);unsigned e=__shfl_xor_sync(0xffffffffU,id,s);
      if(v>best || (v==best && e<id)){best=v;id=e;}
    }
    if(lane==rank){selected=id;selected_logit=best;}
    #pragma unroll
    for(int j=0;j<16;j++)if((unsigned)(j*32+lane)==id)key[j]=-__int_as_float(0x7f800000);
  }
  float weight=lane<10?expf(selected_logit-mx):0;
  if(arm==0)weight=rb(weight*norm);
  float total=0;
  for(int j=0;j<10;j++) {
    float w=__shfl_sync(0xffffffffU,weight,j);
    total=arm?total+w:rb(total+w);
  }
  if(lane<10) {
    ids[r*10+lane]=selected;
    scores[r*10+lane]=invalid?__int_as_float(0x7fc00000):(arm?weight/total:rb(weight/total));
  }
}

// R38 (Opus, direct): load-ahead router projection. Appended to lowbit_router.cu
// (uses its load/store helpers). Same lane-to-K mapping and FMA order as
// router_qmv: k = lane, lane+32, ... ; loads for 20 steps issue before use.
extern "C" __global__ void router_qmv_ahead(const bf* x,const bf* w,float* y) {
  int r=blockIdx.y,lane=threadIdx.x&31,e=blockIdx.x*4+(threadIdx.x>>5);
  float acc=0;
  #pragma unroll
  for(int b=0;b<4;b++) {
    unsigned short xr[20],wr[20];
    #pragma unroll
    for(int i=0;i<20;i++) {
      const int k=lane+32*(b*20+i);
      xr[i]=x[r*2560+k];wr[i]=w[e*2560+k];
    }
    #pragma unroll
    for(int i=0;i<20;i++) acc=fmaf(load(xr[i]),load(wr[i]),acc);
  }
  for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
  if(lane==0)y[r*512+e]=acc;
}
