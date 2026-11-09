// R46: actual packed multi4 consumer, synthetic finite work before it.
// Uses R44's five-node protocol with a final side drain. No software wait.
#define PS 8192u
__device__ __forceinline__ unsigned long long pclock() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
__device__ __forceinline__ unsigned pfold(unsigned v) {
  #pragma unroll
  for(unsigned d=16;d;d>>=1) v^=__shfl_down_sync(0xffffffffu,v,d);
  __shared__ unsigned tmp[4];
  if((threadIdx.x&31)==0) tmp[threadIdx.x>>5]=v;
  __syncthreads(); return tmp[0]^tmp[1]^tmp[2]^tmp[3];
}
__device__ __forceinline__ unsigned pread(const uint4* ptr,unsigned n) {
  unsigned v=0;
  for(unsigned i=blockIdx.x*128+threadIdx.x;i<n;i+=gridDim.x*128) {
    uint4 q;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
      :"=r"(q.x),"=r"(q.y),"=r"(q.z),"=r"(q.w):"l"(ptr+i):"memory");
    v^=q.x^q.y^q.z^q.w;
  }
  return v;
}
extern "C" __global__ void __launch_bounds__(128) p46_evict(const uint4* w,unsigned n,unsigned* out) {
  unsigned v=pfold(pread(w,n));if(threadIdx.x==0)out[blockIdx.x]=v;
}
extern "C" __global__ void __launch_bounds__(128) r44_a(
    const unsigned* in,unsigned* out,unsigned long long* ts) {
  if(threadIdx.x==0)ts[0]=pclock();out[threadIdx.x]=in[threadIdx.x]^0x98765432u;
  __syncthreads();if(threadIdx.x==0)ts[1]=pclock();
}
extern "C" __global__ void __launch_bounds__(128) r44_side(
    const uint4* p0,const uint4* p1,const uint4* p2,const uint4* p3,
    const uint4* p4,const uint4* p5,const uint4* p6,const uint4* p7,
    const uint4* p8,const uint4* p9,const uint4* p10,const uint4* p11,
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned n4,unsigned n5,
    unsigned n6,unsigned n7,unsigned n8,unsigned n9,unsigned n10,unsigned n11,
    unsigned enabled,unsigned* out,unsigned long long* ts) {
  if(threadIdx.x==0)ts[PS*2+blockIdx.x*2]=pclock();
  unsigned v=0;
  if(enabled) {
    #define READ_P(J) v^=pread(p##J,n##J)
    READ_P(0);READ_P(1);READ_P(2);READ_P(3);READ_P(4);READ_P(5);
    READ_P(6);READ_P(7);READ_P(8);READ_P(9);READ_P(10);READ_P(11);
    #undef READ_P
  }
  v=pfold(v);if(threadIdx.x==0)out[blockIdx.x]=v;
  __syncthreads();if(threadIdx.x==0)ts[PS*2+blockIdx.x*2+1]=pclock();
}
extern "C" __global__ void __launch_bounds__(128) r44_b(
    const unsigned* in,unsigned* out,unsigned short* x,unsigned loops,unsigned long long* ts) {
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0)ts[2*PS*2+blockIdx.x*2]=pclock();
  unsigned i=blockIdx.x*128+threadIdx.x,v=in[threadIdx.x]+i;
  #pragma unroll 1
  for(unsigned j=0;j<loops;j++)v=v*1664525u+1013904223u;
  out[i]=v;
  if(i<2560)x[i]=ah_fb(float(int(v&65535u)-32768)*0x1p-16f);
  __syncthreads();if(threadIdx.x==0)ts[2*PS*2+blockIdx.x*2+1]=pclock();
}
extern "C" __global__ void r44_c(const unsigned short* x,
    const unsigned* w0,const unsigned short* s0,const unsigned short* b0,float* y0,unsigned short* yb0,
    const unsigned* w1,const unsigned short* s1,const unsigned short* b1,float* y1,unsigned short* yb1,
    const unsigned* w2,const unsigned short* s2,const unsigned short* b2,float* y2,unsigned short* yb2,
    const unsigned* w3,const unsigned short* s3,const unsigned short* b3,float* y3,unsigned short* yb3,
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,unsigned k,unsigned group,unsigned rows,unsigned long long* ts) {
    if(threadIdx.x==0)ts[3*PS*2+blockIdx.x*2]=pclock();
    unsigned bx=blockIdx.x;
    const unsigned c0=(n0+3)/4,c1=(n1+3)/4,c2=(n2+3)/4;
    if(bx<c0) ahead_block<2,false>(x,w0,s0,b0,y0,yb0,k,n0,group,rows,bx,blockIdx.y);
    else if(bx<c0+c1) ahead_block<2,false>(x,w1,s1,b1,y1,yb1,k,n1,group,rows,bx-c0,blockIdx.y);
    else if(bx<c0+c1+c2) ahead_block<2,false>(x,w2,s2,b2,y2,yb2,k,n2,group,rows,bx-c0-c1,blockIdx.y);
    else ahead_block<2,false>(x,w3,s3,b3,y3,yb3,k,n3,group,rows,bx-c0-c1-c2,blockIdx.y);
    __syncthreads();if(threadIdx.x==0)ts[3*PS*2+blockIdx.x*2+1]=pclock();
}
extern "C" __global__ void __launch_bounds__(128) r44_join(
    const unsigned* y0,const unsigned* y1,const unsigned* y2,const unsigned* y3,
    unsigned n0,unsigned n1,unsigned n2,unsigned n3,const unsigned* side,unsigned gs,
    unsigned* out,unsigned long long* ts) {
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0)ts[4*PS*2]=pclock();
  unsigned v=side[threadIdx.x%gs];
  for(unsigned i=threadIdx.x;i<n0;i+=128)v^=y0[i];
  for(unsigned i=threadIdx.x;i<n1;i+=128)v^=y1[i];
  for(unsigned i=threadIdx.x;i<n2;i+=128)v^=y2[i];
  for(unsigned i=threadIdx.x;i<n3;i+=128)v^=y3[i];
  out[threadIdx.x]=v;
  __syncthreads();if(threadIdx.x==0)ts[4*PS*2+1]=pclock();
}
