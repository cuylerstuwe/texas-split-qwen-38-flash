// R53 hop stress kernels (Opus, direct, 2026-11-06), for the end-of-grid barrier
// experiment (driver TINYNV_R38_PDL bit 256). Bounded: every index is masked into a
// fixed allocation, so an ordering failure produces wrong words, never a stray access.
// Every name starts r39pdl_, so the selective early-launch driver gives every edge a
// latch, exactly the edges bit 256 strips the predecessor's barrier from.

__device__ __forceinline__ void hop_delay(unsigned delay) {
  // The final block stalls before it writes, so its stores are the grid's last and the
  // closest to the grid's completion - the window a missing barrier would expose.
  if (blockIdx.x == gridDim.x - 1 && delay) {
    const long long s = clock64();
    while (clock64() - s < (long long)delay) asm volatile("" ::: "memory");
  }
}

// Chain link: reads its predecessor's output (a, one hop) and the output before that
// (b, two hops) with plain, L1-cacheable loads, so the successor's latch-acquire L1
// invalidate is exercised as well as the visibility of the stores themselves.
extern "C" __global__ void __launch_bounds__(256) r39pdl_r53_hop(
    unsigned* dst, const unsigned* a, const unsigned* b, unsigned n, unsigned salt, unsigned delay) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  hop_delay(delay);
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const unsigned j = (i * 17u + 23u) & (n - 1u), k = (i * 29u + 5u) & (n - 1u);
    const unsigned v = a[j] ^ (b[k] * 3u) ^ salt;
    dst[i] = ((v << 7) | (v >> 25)) + i;
  }
}

// Flag pattern, middle launch: waits for its predecessor, then every block takes a
// ticket on a relaxed counter, as the MoE gate/up launch does after its wait.
extern "C" __global__ void __launch_bounds__(256) r39pdl_r53_hop_g(unsigned* flag, unsigned delay) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  hop_delay(delay);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(flag, 1u);
}

// Flag pattern, reader: launched early behind the middle launch and NOT waiting on the
// grid boundary. It spins (bounded) until the middle launch's first ticket, then reads,
// through L2, the data written two launches back - the MoE down launch reading F0's
// expert ids. A spin that times out sets *status instead of hanging.
extern "C" __global__ void __launch_bounds__(256) r39pdl_r53_hop_h(
    unsigned* dst, const unsigned* d, const unsigned* flag, unsigned* status, unsigned n, unsigned salt) {
  if (threadIdx.x == 0) {
    unsigned spins = 0, v = 0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
      if (v || ++spins >= (1u << 22)) break;
      __nanosleep(64);
    }
    if (!v) atomicExch(status, 1u);
    __threadfence();
  }
  __syncthreads();
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const unsigned j = (i * 13u + 7u) & (n - 1u);
    const unsigned v = __ldcg(d + j) ^ salt;
    dst[i] = ((v << 11) | (v >> 21)) ^ i;
  }
}
