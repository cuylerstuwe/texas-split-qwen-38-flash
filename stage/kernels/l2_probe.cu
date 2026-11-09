// R38 (Opus, direct): L2 prefetch behavior probe. No model data.
// l2p_read streams a buffer (16-byte loads, four in flight per thread) and
// folds it into one word so the loads are live. The prefetch kernels touch a
// buffer with prefetch.global.L2 (one per 128-byte line, optionally
// evict_last) or cp.async.bulk.prefetch.L2 (one per chunk) and exit without
// waiting for anything. l2p_spin holds one block for a number of nanoseconds.
extern "C" __global__ void __launch_bounds__(256) l2p_read(const uint4* w,unsigned long long n16,unsigned* out) {
  const unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned acc=0;
  for(;i+3*stride<n16;i+=4*stride) {
    const uint4 a=w[i],b=w[i+stride],c=w[i+2*stride],d=w[i+3*stride];
    acc^=a.x^a.y^a.z^a.w^b.x^b.y^b.z^b.w^c.x^c.y^c.z^c.w^d.x^d.y^d.z^d.w;
  }
  for(;i<n16;i+=stride) { const uint4 a=w[i]; acc^=a.x^a.y^a.z^a.w; }
  if(acc==0x9e3779b9u) out[0]=acc;
}
extern "C" __global__ void __launch_bounds__(256) l2p_prefetch(const char* w,unsigned long long bytes,unsigned evict_last) {
  const unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x*128ull;
  for(unsigned long long i=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)*128ull;i<bytes;i+=stride) {
    if(evict_last) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(w+i));
    else asm volatile("prefetch.global.L2 [%0];" :: "l"(w+i));
  }
}
extern "C" __global__ void __launch_bounds__(256) l2p_bulk(const char* w,unsigned long long bytes,unsigned chunk) {
  const unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x*chunk;
  for(unsigned long long i=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)*chunk;i<bytes;i+=stride) {
    const unsigned long long left=bytes-i;
    const unsigned sz=left<chunk?(unsigned)left:chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w+i), "r"(sz) : "memory");
  }
}
extern "C" __global__ void l2p_spin(unsigned long long ns) {
  unsigned long long t0,t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while(t-t0<ns);
}
// l2p_bulk with the issuing block kept resident for hold_ns after it issues.
extern "C" __global__ void __launch_bounds__(256) l2p_bulk_hold(const char* w,unsigned long long bytes,unsigned chunk,
    unsigned long long hold_ns) {
  unsigned long long t0,t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  const unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x*chunk;
  for(unsigned long long i=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)*chunk;i<bytes;i+=stride) {
    const unsigned long long left=bytes-i;
    const unsigned sz=left<chunk?(unsigned)left:chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w+i), "r"(sz) : "memory");
  }
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while(t-t0<hold_ns);
}
