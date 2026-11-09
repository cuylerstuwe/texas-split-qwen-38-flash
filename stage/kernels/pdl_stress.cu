// R39: bounded, adversarial grid-dependency tests; no model pointers or spin
// barriers. Every address remains in a fixed allocation even if ordering fails.
template<int MODE> __device__ __forceinline__ void body(
    unsigned* dst, const unsigned* src, unsigned n, unsigned salt, unsigned delay) {
  if (MODE == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (MODE != 3) asm volatile("griddepcontrol.wait;" ::: "memory");
  // The final producer block can finish later than most of the grid. No block
  // waits for another block, and the delay has a finite cycle bound.
  if (blockIdx.x == gridDim.x - 1 && delay) {
    unsigned long long start = clock64();
    while (clock64() - start < delay) asm volatile("" ::: "memory");
  }
  for (unsigned i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) {
    unsigned j=(i*17u+23u)&(n-1u);
    unsigned v=src[j]^salt;
    dst[i]=((v<<7)|(v>>25))+i;
  }
  if (MODE == 2) {
    __threadfence(); __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
}
#define WRAP(NAME, MODE) extern "C" __global__ void __launch_bounds__(256) NAME( \
    unsigned* dst,const unsigned* src,unsigned n,unsigned salt,unsigned delay) { \
  body<MODE>(dst,src,n,salt,delay); }
WRAP(r39_wait, 0)
WRAP(r39_early, 1)
WRAP(r39_late, 2)
WRAP(r39_plain, 3)
