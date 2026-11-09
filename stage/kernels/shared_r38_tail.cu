// R38 (Opus, direct): load-ahead shared-expert gate tail. Appended to
// lowbit_shared.cu (uses its helpers). Same order as shared_finish; all of a
// thread's input, gate-weight and down-projection loads issue up front.
// Requires hidden <= 2560 (checked by the caller).
extern "C" __global__ void shared_finish_ahead(const bf* x,const bf* weight,const bf* down,
    bf* out,float* raw,bf* logits,bf* sigmoid,int hidden,int arm) {
  __shared__ float sums[4];
  int r=blockIdx.x,t=threadIdx.x;
  unsigned short xr[20],wr[20],dr[20];
  #pragma unroll
  for(int i=0;i<20;i++) {
    const int k=t+128*i;const bool ok=k<hidden;
    xr[i]=ok?x[r*hidden+k]:(unsigned short)0;
    wr[i]=ok?weight[k]:(unsigned short)0;
    dr[i]=ok?down[r*hidden+k]:(unsigned short)0;
  }
  float sum=0;
  #pragma unroll
  for(int i=0;i<20;i++) if(t+128*i<hidden) sum=fmaf(load(xr[i]),load(wr[i]),sum);
  for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
  if((t&31)==0) sums[t>>5]=sum;
  __syncthreads();
  if(t<32) {
    sum=t<4?sums[t]:0;
    for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
    if(t==0) {
      raw[r]=sum;
      bf logit=store(sum);
      float p=sig(load(logit),arm);
      logits[r]=logit; sigmoid[r]=store(p); sums[0]=p;
    }
  }
  __syncthreads();
  const float p=sums[0];
  #pragma unroll
  for(int i=0;i<20;i++) {
    const int k=t+128*i;
    if(k<hidden) out[r*hidden+k]=store(load(dr[i])*p);
  }
}
