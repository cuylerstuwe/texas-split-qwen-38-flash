// R45 real immutable loads, synthetic independent work, explicit side drain.
// Names implement the previously qualified R44 five-descriptor protocol.
// No cross-grid software spin, speculative pointer, or model-speed claim.
#define SG 256u
__device__ __forceinline__ unsigned long long tick() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
__device__ __forceinline__ uint4 read4(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
  return v;
}
__device__ __forceinline__ unsigned fold(unsigned v) {
  #pragma unroll
  for(unsigned d=16;d;d>>=1) v^=__shfl_down_sync(0xffffffffu,v,d);
  __shared__ unsigned tmp[4];
  if((threadIdx.x&31)==0) tmp[threadIdx.x>>5]=v;
  __syncthreads();
  return tmp[0]^tmp[1]^tmp[2]^tmp[3];
}
__device__ __forceinline__ unsigned scan(const uint4* w,unsigned n4) {
  unsigned v=0;
  for(unsigned i=blockIdx.x*128+threadIdx.x;i<n4;i+=gridDim.x*128) {
    uint4 x=read4(w+i); v^=x.x^x.y^x.z^x.w;
  }
  return fold(v);
}
extern "C" __global__ void __launch_bounds__(128) r45_evict(
    const uint4* w,unsigned n4,unsigned* out) {
  unsigned v=scan(w,n4); if(threadIdx.x==0) out[blockIdx.x]=v;
}
extern "C" __global__ void __launch_bounds__(128) r44_a(
    const unsigned* input,unsigned* out,unsigned long long* stamps) {
  if(threadIdx.x==0) stamps[0]=tick();
  out[threadIdx.x]=input[threadIdx.x]^0x98765432u;
  __syncthreads(); if(threadIdx.x==0) stamps[1]=tick();
}
extern "C" __global__ void __launch_bounds__(128) r44_side(
    const uint4* w,unsigned n4,unsigned enabled,unsigned* out,unsigned long long* stamps) {
  if(threadIdx.x==0) stamps[SG*2+blockIdx.x*2]=tick();
  unsigned v=enabled ? scan(w,n4) : 0;
  if(threadIdx.x==0) out[blockIdx.x]=v;
  __syncthreads(); if(threadIdx.x==0) stamps[SG*2+blockIdx.x*2+1]=tick();
}
extern "C" __global__ void __launch_bounds__(128) r44_b(
    const unsigned* input,unsigned* out,unsigned iterations,unsigned long long* stamps) {
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0) stamps[2*SG*2+blockIdx.x*2]=tick();
  unsigned i=blockIdx.x*128+threadIdx.x;
  unsigned x=input[threadIdx.x]+i;
  #pragma unroll 1
  for(unsigned k=0;k<iterations;k++) x=x*1664525u+1013904223u;
  out[i]=x;
  __syncthreads(); if(threadIdx.x==0) stamps[2*SG*2+blockIdx.x*2+1]=tick();
}
extern "C" __global__ void __launch_bounds__(128) r44_c(
    const uint4* w,unsigned n4,const unsigned* b,unsigned bn,unsigned* out,unsigned long long* stamps) {
  if(threadIdx.x==0) stamps[3*SG*2+blockIdx.x*2]=tick();
  unsigned v=scan(w,n4);
  if(threadIdx.x==0) out[blockIdx.x]=v^b[blockIdx.x%bn];
  __syncthreads(); if(threadIdx.x==0) stamps[3*SG*2+blockIdx.x*2+1]=tick();
}
extern "C" __global__ void __launch_bounds__(128) r44_join(
    const unsigned* c,const unsigned* side,unsigned gs,unsigned* out,unsigned long long* stamps) {
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0) stamps[4*SG*2]=tick();
  for(unsigned i=threadIdx.x;i<SG;i+=128) out[i]=c[i]^side[i%gs];
  __syncthreads(); if(threadIdx.x==0) stamps[4*SG*2+1]=tick();
}
