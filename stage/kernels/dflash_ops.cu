// Native DFlash2 elementary operations. BF16 storage, FP32 arithmetic.
// Semantics audited against Sushi's MIT dflash.zig/dflash_conv.zig; NOTICE.txt.
// No CUDA host runtime, device allocation, persistent worker, or global state.
typedef unsigned short bf;
__device__ __forceinline__ float load(bf x) { return __uint_as_float((unsigned)x << 16); }
__device__ __forceinline__ bf store(float x) {
  unsigned u=__float_as_uint(x);
  return (bf)((u + 0x7fffU + ((u>>16)&1U))>>16); // finite RNE
}
__device__ __forceinline__ float roundbf(float x) { return load(store(x)); }

// One 256-thread block per row; shared reduction only within that block.
extern "C" __global__ void dflash_rms(const bf* x,const bf* w,bf* y,int h,float eps) {
  __shared__ float partial[8];
  int row=blockIdx.x,t=threadIdx.x;
  float sum=0;
  for(int c=t;c<h;c+=256) { float v=load(x[row*h+c]); sum+=v*v; }
  for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
  if((t&31)==0) partial[t>>5]=sum;
  __syncthreads();
  if(t<32) {
    sum=t<8?partial[t]:0;
    for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
    if(t==0) partial[0]=rsqrtf(sum/(float)h+eps);
  }
  __syncthreads();
  for(int c=t;c<h;c+=256) y[row*h+c]=store(load(x[row*h+c])*partial[0]*load(w[c]));
}

// base[prepare|finish, tap, channel]; dynamic[row,set,tap,group].
// Dynamic finish coefficients are retained from the normalized INPUT.
// The preceding row at block start is zero; no previous draft's convolution state.
// Arm0 reproduces separate BF16 products/adds. Arm1 retains FP32 until output.
extern "C" __global__ void dflash_conv(const bf* x,const bf* dynamic,const bf* base,
    const bf* residual,bf* y,int rows,int h,int group_size,int set,int arm,int add) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*h) return;
  int r=i/h,c=i%h,g=c/group_size,groups=h/group_size;
  float acc=0;
  for(int tap=0;tap<2;++tap) {
    float v=r>=tap?load(x[(r-tap)*h+c]):0;
    float a=load(base[(set*2+tap)*h+c]);
    float b=load(dynamic[(r*4+set*2+tap)*groups+g]);
    if(arm==0) {
      acc=roundbf(acc+roundbf(a*v));
      acc=roundbf(acc+roundbf(b*v));
    } else { acc=fmaf(a,v,acc); acc=fmaf(b,v,acc); }
  }
  // Even the fused arm retains BF16 sublayer output before residual addition.
  acc=roundbf(acc);
  y[i]=store(add?load(residual[i])+acc:acc);
}

extern "C" __global__ void dflash_swiglu(const bf* gate,const bf* up,bf* y,int count,int arm) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  float g=load(gate[i]),u=load(up[i]);
  float sig=1.0f/(1.0f+expf(-g));
  y[i]=store(arm==0?roundbf(g*roundbf(sig))*u:g*sig*u);
}
