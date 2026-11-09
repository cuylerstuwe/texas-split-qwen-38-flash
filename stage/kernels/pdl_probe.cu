// R38 (Opus, direct): early dependent launch probe kernels. No model data.
// Dependent 16-byte copies; the variants differ only in the grid-dependency
// instructions: wait = griddepcontrol.wait before the first load; early =
// launch_dependents first, then wait; plain = neither (a race detector when the
// driver launches dependents early).
extern "C" __global__ void __launch_bounds__(256) pdl_copy_wait(uint4* dst,const uint4* src,unsigned n16) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[i];
}
extern "C" __global__ void __launch_bounds__(256) pdl_copy_early(uint4* dst,const uint4* src,unsigned n16) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[i];
}
extern "C" __global__ void __launch_bounds__(256) pdl_copy_plain(uint4* dst,const uint4* src,unsigned n16) {
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[i];
}
// Reversed copies: element i comes from element n16-1-i, so (with more than one
// block) every block reads what a different block of the producer wrote; an even
// number of links restores the input. Separates grid-level ordering from a
// per-block (same block index) dependency.
extern "C" __global__ void __launch_bounds__(256) pdl_rev_wait(uint4* dst,const uint4* src,unsigned n16) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[n16-1-i];
}
extern "C" __global__ void __launch_bounds__(256) pdl_rev_early(uint4* dst,const uint4* src,unsigned n16) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[n16-1-i];
}
extern "C" __global__ void __launch_bounds__(256) pdl_rev_plain(uint4* dst,const uint4* src,unsigned n16) {
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=gridDim.x*blockDim.x) dst[i]=src[n16-1-i];
}
