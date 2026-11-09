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
