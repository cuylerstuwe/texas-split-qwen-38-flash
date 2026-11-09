// R44 synthetic A -> side -> B -> C -> join. Every address is fixed/in-bounds.
// side has finite work; only hardware dependency waits, no software spin waits
// on another grid. Join drains side before graph tail release and buffer reuse.
#define SG 257u
__device__ __forceinline__ unsigned long long timer44() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
__device__ __forceinline__ void delay44(unsigned ns) {
  const unsigned long long start=timer44();
  while(timer44()-start<ns) asm volatile("":::"memory");
}
template<unsigned STAGE,bool WAIT> __device__ __forceinline__ void transform44(
    unsigned* dst,const unsigned* src,unsigned n,unsigned salt,unsigned delay,unsigned long long* stamps) {
  if(WAIT) asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0) stamps[STAGE*SG*2+blockIdx.x*2]=timer44();
  if(blockIdx.x==gridDim.x-1 && threadIdx.x==0) delay44(delay);
  __syncthreads();
  for(unsigned i=blockIdx.x*128+threadIdx.x;i<n;i+=gridDim.x*128) {
    unsigned v=src[(i*17+23)&(n-1)]^salt;
    dst[i]=((v<<7)|(v>>25))+i;
  }
  __syncthreads();
  if(threadIdx.x==0) stamps[STAGE*SG*2+blockIdx.x*2+1]=timer44();
}
#define TR(NAME,STAGE,WAIT) extern "C" __global__ void __launch_bounds__(128) NAME( \
  unsigned* dst,const unsigned* src,unsigned n,unsigned salt,unsigned delay,unsigned long long* stamps) { \
  transform44<STAGE,WAIT>(dst,src,n,salt,delay,stamps); }
TR(r44_a,0,false)
TR(r44_b,2,true)
TR(r44_b_nowait,2,false)
TR(r44_c,3,false)
extern "C" __global__ void __launch_bounds__(128) r44_side(
    const unsigned* weights,unsigned n,unsigned delay,unsigned* out,unsigned long long* stamps) {
  if(threadIdx.x==0) stamps[SG*2+blockIdx.x*2]=timer44();
  if(threadIdx.x==0) delay44(delay);
  __syncthreads();
  unsigned v=0;
  for(unsigned i=blockIdx.x*128+threadIdx.x;i<n;i+=gridDim.x*128) v^=weights[i];
  #pragma unroll
  for(unsigned d=16;d;d>>=1) v^=__shfl_down_sync(0xffffffffu,v,d);
  __shared__ unsigned sums[4];
  if((threadIdx.x&31)==0) sums[threadIdx.x>>5]=v;
  __syncthreads();
  if(threadIdx.x==0) {out[blockIdx.x]=sums[0]^sums[1]^sums[2]^sums[3];stamps[SG*2+blockIdx.x*2+1]=timer44();}
}
extern "C" __global__ void __launch_bounds__(128) r44_join(
    unsigned* dst,const unsigned* src,const unsigned* side,unsigned n,unsigned side_blocks,unsigned long long* stamps) {
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0) stamps[4*SG*2]=timer44();
  for(unsigned i=threadIdx.x;i<n;i+=128) dst[i]=src[i]^side[i%side_blocks];
  __syncthreads();
  if(threadIdx.x==0) stamps[4*SG*2+1]=timer44();
}
