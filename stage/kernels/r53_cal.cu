// R53 calibration kernels (Opus, direct, 2026-11-06). Null launches for an
// empty-step replay with the model's exact launch geometry and dependency edges,
// and a per-SM clock probe. Nothing here reads or writes model data.

// Plain null: the driver serializes it behind its predecessor like any kernel.
extern "C" __global__ void r53_null() {}

// Early-launch successor twin. The r39pdl_ prefix makes the selective-PDL driver
// record the same edge a real r39pdl_ kernel gets; the wait keeps the dependency.
extern "C" __global__ void r39pdl_r53_null() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ unsigned long long r53_gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One thread per block spins for spin_ns of globaltimer and reports SM cycles
// elapsed: out[4*b] = {t0, t1, cycles, smid}. MHz = cycles * 1000 / (t1 - t0).
extern "C" __global__ void r53_clock(unsigned long long* __restrict__ out, unsigned spin_ns) {
  if (threadIdx.x) return;
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const unsigned long long t0 = r53_gt();
  const long long c0 = clock64();
  unsigned long long t1;
  do { t1 = r53_gt(); } while (t1 - t0 < spin_ns);
  const long long c1 = clock64();
  t1 = r53_gt();
  unsigned long long* o = out + 4ull * blockIdx.x;
  o[0] = t0; o[1] = t1; o[2] = (unsigned long long)(c1 - c0); o[3] = smid;
}
