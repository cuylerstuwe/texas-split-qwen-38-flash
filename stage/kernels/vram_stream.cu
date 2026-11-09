// Bounded read-stream sanity check, not a model or proof of peak DRAM bandwidth.
// Every loaded word contributes to a retained per-warp modulo-2^32 checksum.
// Inline volatile vector loads prevent eliminating repeated resident reads.
__device__ __forceinline__ uint4 read_cg(const uint4* p) {
  uint4 x;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(p) : "memory");
  return x;
}
template<int ILP> __device__ void stream_read(const uint4* input, uint4* output, unsigned n) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = blockDim.x * gridDim.x;
  uint4 sums[ILP];
  #pragma unroll
  for (int j=0;j<ILP;++j) sums[j] = make_uint4(0,0,0,0);
  for (unsigned base=tid;base<n;base+=ILP*stride) {
    #pragma unroll
    for (int j=0;j<ILP;++j) {
      unsigned at=base+j*stride;
      if (at<n) {
        uint4 v=read_cg(input+at);
        sums[j].x+=v.x; sums[j].y+=v.y; sums[j].z+=v.z; sums[j].w+=v.w;
      }
    }
  }
  #pragma unroll
  for (int j=1;j<ILP;++j) {
    sums[0].x+=sums[j].x; sums[0].y+=sums[j].y;
    sums[0].z+=sums[j].z; sums[0].w+=sums[j].w;
  }
  for (int off=16;off;off/=2) {
    sums[0].x+=__shfl_down_sync(0xffffffff,sums[0].x,off);
    sums[0].y+=__shfl_down_sync(0xffffffff,sums[0].y,off);
    sums[0].z+=__shfl_down_sync(0xffffffff,sums[0].z,off);
    sums[0].w+=__shfl_down_sync(0xffffffff,sums[0].w,off);
  }
  if ((threadIdx.x&31)==0) output[tid/32]=sums[0];
}
extern "C" __global__ void read1(const uint4* p,uint4* out,unsigned n) { stream_read<1>(p,out,n); }
extern "C" __global__ void read4(const uint4* p,uint4* out,unsigned n) { stream_read<4>(p,out,n); }
