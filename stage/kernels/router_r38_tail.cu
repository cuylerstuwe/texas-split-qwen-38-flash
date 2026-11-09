// R38 (Opus, direct): load-ahead router projection. Appended to lowbit_router.cu
// (uses its load/store helpers). Same lane-to-K mapping and FMA order as
// router_qmv: k = lane, lane+32, ... ; loads for 20 steps issue before use.
extern "C" __global__ void router_qmv_ahead(const bf* x,const bf* w,float* y) {
  int r=blockIdx.y,lane=threadIdx.x&31,e=blockIdx.x*4+(threadIdx.x>>5);
  float acc=0;
  #pragma unroll
  for(int b=0;b<4;b++) {
    unsigned short xr[20],wr[20];
    #pragma unroll
    for(int i=0;i<20;i++) {
      const int k=lane+32*(b*20+i);
      xr[i]=x[r*2560+k];wr[i]=w[e*2560+k];
    }
    #pragma unroll
    for(int i=0;i<20;i++) acc=fmaf(load(xr[i]),load(wr[i]),acc);
  }
  for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
  if(lane==0)y[r*512+e]=acc;
}
