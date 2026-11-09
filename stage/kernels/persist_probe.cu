// R38 (Opus, direct): persistent-kernel stage-barrier probe. No model data.
// Every block runs `stages` rounds of tiny work separated by a grid-wide
// counter barrier (release add / acquire spin with a bounded spin). The last
// block out resets all counters, so the same launch can be replayed.
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
extern "C" __global__ void persist_barrier(unsigned* counters,unsigned stages,unsigned* status,
    unsigned* sink,unsigned spin_limit) {
  const unsigned nb=gridDim.x;
  __shared__ unsigned fail;
  if(threadIdx.x==0) fail=0;
  __syncthreads();
  for(unsigned s=0;s<stages;s++) {
    if(threadIdx.x==0) sink[blockIdx.x]=s;
    __syncthreads();
    if(threadIdx.x==0) {
      __threadfence();
      red_release(&counters[s],1u);
      unsigned spins=0;
      while(ld_acquire(&counters[s])<nb) {
        if(++spins>spin_limit) { fail=1; atomicExch(status,1u); break; }
      }
    }
    __syncthreads();
    if(fail) return;
  }
  if(threadIdx.x==0) {
    __threadfence();
    const unsigned out=atomicAdd(&counters[stages],1u);
    if(out==nb-1) {
      for(unsigned s=0;s<=stages;s++) counters[s]=0;
      __threadfence();
    }
  }
}
