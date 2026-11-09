// Native CUDA port of this project's gdn_block.metal. Sushi/MLX-derived K1S
// layout and BF16 boundaries, MIT attribution in NOTICE.txt. FP32 state stays
// resident. All-prefix snapshots are a diagnostic/speculative control; final
// checkpoint mode avoids writing a complete state matrix for every draft row.
typedef unsigned short bf;
__device__ __forceinline__ float load(bf x){return __uint_as_float((unsigned)x<<16);}
__device__ __forceinline__ bf store(float x){bf out;asm("cvt.rn.bf16.f32 %0, %1;":"=h"(out):"f"(x));return out;}
__device__ __forceinline__ float rb(float x){return load(store(x));}
__device__ __forceinline__ float sig(float x){float small=rb(1.f/rb(1.f+rb(expf(fabsf(x)))));return x<0?small:rb(1.f-small);}
__device__ __forceinline__ float sum_all(float x){for(int s=16;s;s>>=1)x+=__shfl_down_sync(0xffffffffU,x,s);return __shfl_sync(0xffffffffU,x,0);}
enum{DK=128,DV=128,HK=16,HV=48,C=10240,V=6144,ST=HV*DV*DK};

__device__ void prepare(const bf* qkv,const bf* a,const bf* b,const bf* conv_in,
 const bf* cw,const bf* alog,const bf* dt,bf* conv_out,int rows,int start,int count,
 int hv,int part,float* qs,float* ks,float* vs,float* gates){
 int tid=threadIdx.x,lane=tid&31,sg=tid>>5,hk=hv/3;
 if(sg<3){
  int cb=sg==0?hk*DK:(sg==1?HK*DK+hk*DK:2*HK*DK+hv*DV);
  for(int t=0;t<count;t++){
   float act[4],sum=0;
   #pragma unroll
   for(int i=0;i<4;i++){
    int ch=cb+lane*4+i;float acc=0;
    #pragma unroll
    for(int tap=0;tap<4;tap++){int w=start+t+tap;float x=load(w<3?conv_in[w*C+ch]:qkv[(w-3)*C+ch]);acc=fmaf(x,load(cw[ch*4+tap]),acc);}
    float cv=rb(acc);act[i]=rb(cv*sig(cv));sum+=act[i]*act[i];
   }
   if(sg<2){
    sum=sum_all(sum);float inv=rsqrtf(sum/float(DK)+1e-6f),scale=sg==0?1.f/128.f:rb(1.f/sqrtf(128.f));
    float* dst=sg==0?qs:ks;
    #pragma unroll
    for(int i=0;i<4;i++)dst[t*DK+lane*4+i]=rb(scale*rb(act[i]*inv));
   }else{
    #pragma unroll
    for(int i=0;i<4;i++)vs[t*DV+lane*4+i]=act[i];
   }
  }
  if(part==0&&start+count==rows&&(sg==2||hv%3==0)){
   #pragma unroll
   for(int i=0;i<4;i++)for(int j=0;j<3;j++){int ch=cb+lane*4+i,w=rows+j;conv_out[j*C+ch]=w<3?conv_in[w*C+ch]:qkv[(w-3)*C+ch];}
  }
 }
 if(sg==3&&lane==31){
  float ea=expf(load(alog[hv]));
  for(int t=0;t<count;t++){
   int row=start+t;float apd=load(a[row*HV+hv])+load(dt[hv]);
   // Stable FP32 softplus, including large positive inputs.
   float sp=fmaxf(apd,0.f)+log1pf(expf(-fabsf(apd)));
   gates[t*2]=expf(-ea*sp);gates[t*2+1]=sig(load(b[row*HV+hv]));
  }
 }
 __syncthreads();
}
#define ARGS const bf* qkv,const bf* z,const bf* a,const bf* b,const bf* conv_in,const float* state_in, const bf* cw,const bf* alog,const bf* dt,const bf* norm,bf* q,bf* k,bf* v,float* gate,bf* y,bf* gated,float* states,bf* conv_out,int rows,int all_states
extern "C" __global__ void gdn_prepare(ARGS){
 int hv=blockIdx.x,t=blockIdx.y,hk=hv/3,lane=threadIdx.x&31,sg=threadIdx.x>>5;
 __shared__ float qs[128],ks[128],vs[128],gb[2];
 prepare(qkv,a,b,conv_in,cw,alog,dt,conv_out,rows,t,1,hv,0,qs,ks,vs,gb);
 if(sg<3)for(int i=0;i<4;i++){int j=lane*4+i;if(sg==0&&hv%3==0)q[(t*HK+hk)*DK+j]=store(qs[j]);if(sg==1&&hv%3==0)k[(t*HK+hk)*DK+j]=store(ks[j]);if(sg==2)v[(t*HV+hv)*DV+j]=store(vs[j]);}
 if(threadIdx.x==127){gate[(t*HV+hv)*2]=gb[0];gate[(t*HV+hv)*2+1]=gb[1];}
}
extern "C" __global__ void gdn_recur(ARGS){
 int hv=blockIdx.x/4,part=blockIdx.x%4,hk=hv/3,lane=threadIdx.x&31,sg=threadIdx.x>>5,row0=part*32+sg*8;
 float st[8][4];
 #pragma unroll
 for(int r=0;r<8;r++)for(int j=0;j<4;j++)st[r][j]=state_in[(hv*DV+row0+r)*DK+lane*4+j];
 for(int t=0;t<rows;t++){
  float qq[4],kk[4];for(int j=0;j<4;j++){qq[j]=load(q[(t*HK+hk)*DK+lane*4+j]);kk[j]=load(k[(t*HK+hk)*DK+lane*4+j]);}
  float g=gate[(t*HV+hv)*2],beta=gate[(t*HV+hv)*2+1];
  #pragma unroll
  for(int r=0;r<8;r++){
   float memory=0;for(int j=0;j<4;j++){st[r][j]*=g;memory+=st[r][j]*kk[j];}memory=sum_all(memory);
   float delta=beta*(load(v[(t*HV+hv)*DV+row0+r])-memory),out=0;
   for(int j=0;j<4;j++){st[r][j]+=delta*kk[j];out+=st[r][j]*qq[j];if(all_states||t+1==rows)states[(all_states?t:0)*ST+(hv*DV+row0+r)*DK+lane*4+j]=st[r][j];}
   out=sum_all(out);if(lane==0)y[(t*HV+hv)*DV+row0+r]=store(out);
  }
 }
}
template<int WIDTH> __device__ void fused(ARGS){
 int hv=blockIdx.x/4,part=blockIdx.x%4,lane=threadIdx.x&31,sg=threadIdx.x>>5,row0=part*32+sg*8;
 __shared__ float qs[WIDTH*128],ks[WIDTH*128],vs[WIDTH*128],gb[WIDTH*2];
 prepare(qkv,a,b,conv_in,cw,alog,dt,conv_out,rows,0,rows,hv,part,qs,ks,vs,gb);
 float st[8][4];
 #pragma unroll
 for(int r=0;r<8;r++)for(int j=0;j<4;j++)st[r][j]=state_in[(hv*DV+row0+r)*DK+lane*4+j];
 for(int t=0;t<rows;t++){
  float qq[4],kk[4];for(int j=0;j<4;j++){qq[j]=qs[t*DK+lane*4+j];kk[j]=ks[t*DK+lane*4+j];}
  float g=gb[t*2],beta=gb[t*2+1];
  #pragma unroll
  for(int r=0;r<8;r++){
   float memory=0;for(int j=0;j<4;j++){st[r][j]*=g;memory+=st[r][j]*kk[j];}memory=sum_all(memory);
   float delta=beta*(vs[t*DV+row0+r]-memory),out=0;
   for(int j=0;j<4;j++){st[r][j]+=delta*kk[j];out+=st[r][j]*qq[j];if(all_states||t+1==rows)states[(all_states?t:0)*ST+(hv*DV+row0+r)*DK+lane*4+j]=st[r][j];}
   out=sum_all(out);if(lane==0)y[(t*HV+hv)*DV+row0+r]=store(out);
  }
 }
}
#define FUSED(NAME,WIDTH) extern "C" __global__ void NAME(ARGS){fused<WIDTH>(qkv,z,a,b,conv_in,state_in,cw,alog,dt,norm,q,k,v,gate,y,gated,states,conv_out,rows,all_states);}
FUSED(gdn_fused1,1)
FUSED(gdn_fused4,4)
FUSED(gdn_fused8,8)
FUSED(gdn_fused16,16)
extern "C" __global__ void gdn_norm_gate(ARGS){
 int t=blockIdx.y,hv=blockIdx.x,lane=threadIdx.x,base=(t*HV+hv)*DV+lane*4;
 float xs[4],sum=0;for(int j=0;j<4;j++){xs[j]=load(y[base+j]);sum+=xs[j]*xs[j];}
 sum=sum_all(sum);float inv=rsqrtf(sum/float(DV)+1e-6f);
 for(int j=0;j<4;j++){float n=rb(load(norm[lane*4+j])*rb(xs[j]*inv));gated[base+j]=store(n*sig(load(z[base+j])));}
}
// Select an exact accepted prefix into disjoint buffers. Separate restore avoids
// an in-place convolution shift racing across thread blocks at accepted1/2.
extern "C" __global__ void gdn_commit(const bf* qkv,const bf* conv_in,const float* state_in,
 const float* states,bf* conv_next,float* state_next,int accepted,int rows,int all_states){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<ST)state_next[i]=accepted?states[(all_states?accepted-1:0)*ST+i]:state_in[i];
 if(i<3*C){int token=i/C,ch=i%C,w=accepted+token;conv_next[i]=w<3?conv_in[w*C+ch]:qkv[(w-3)*C+ch];}
}
extern "C" __global__ void gdn_restore(const bf* conv_next,const float* state_next,bf* conv_in,float* state_in){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<ST)state_in[i]=state_next[i];if(i<3*C)conv_in[i]=conv_next[i];
}
