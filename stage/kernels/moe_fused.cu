// R38 (Opus, direct): staged MoE with "last block done" tails, five launches.
//   F0: router projection -> (last router block) top-10 selection
//       || shared gate/up split4 partials -> (last shared block) reduce + activate
//   F1: moe_prepare_bf16 || shared down projection
//   F2: gate+up MMA (M output tiles per block) -> (last block of each slot) mid
//       || shared gate tail
//   then moe_stage5_m*_r38 (down MMA) and moe_stage6_r38 (finish + shared add).
// Every piece of arithmetic is the island's stage code (moe_island.cu, which
// ports the qualified kernels) executed by a different block; the tails only
// change WHO runs a dependent step, not its operations or their order. Each
// producer fences its writes before counting itself done; the block that sees
// the count complete fences again and reads the other blocks' results through
// L2 (ld.global.cg), then resets its counter for the next replay. See
// moe_island.cu and NOTICE.txt.
#define MF_PARAMS \
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits, \
    unsigned* ids,float* scores,int select_arm, \
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup, \
    float* gpart,float* gu_f,unsigned short* gu_b, \
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db, \
    unsigned dgroup,float* dp_f,unsigned short* dp_b, \
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw, \
    unsigned short* slogits,unsigned short* ssig,int shared_arm, \
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu, \
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar, \
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd, \
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv, \
    float* rout,unsigned short* rout_b,unsigned short* added,unsigned* ctr

// mi_router_select with the raw logits read through L2 (written by other blocks).
__device__ __forceinline__ void mf_router_select(const float* raw,unsigned short* logits,unsigned* ids,
    float* scores,int arm,unsigned lane) {
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=__ldcg(raw+e);unsigned short h=mi_store_rne(v);logits[e]=h;
    bool bad=!isfinite(v)||!isfinite(mi_load(h));invalid|=bad;
    key[j]=bad?-3.402823466e38f:mi_load(h);mx=fmaxf(mx,key[j]);
  }
  invalid=__any_sync(0xffffffffU,invalid);
  for(int s=16;s;s>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffU,mx,s));
  float norm=0;
  if(arm==0) {
    #pragma unroll
    for(int j=0;j<16;j++)norm+=expf(key[j]-mx);
    for(int s=16;s;s>>=1)norm+=__shfl_xor_sync(0xffffffffU,norm,s);
    norm=1.0f/norm;
  }
  unsigned selected=0;
  float selected_logit=0;
  for(int rank=0;rank<10;rank++) {
    float best=-__int_as_float(0x7f800000);unsigned id=0xffffffffU;
    #pragma unroll
    for(int j=0;j<16;j++) {
      unsigned e=j*32+lane;
      if(key[j]>best || (key[j]==best && e<id)){best=key[j];id=e;}
    }
    for(int s=16;s;s>>=1) {
      float v=__shfl_xor_sync(0xffffffffU,best,s);unsigned e=__shfl_xor_sync(0xffffffffU,id,s);
      if(v>best || (v==best && e<id)){best=v;id=e;}
    }
    if((int)lane==rank){selected=id;selected_logit=best;}
    #pragma unroll
    for(int j=0;j<16;j++)if((unsigned)(j*32+lane)==id)key[j]=-__int_as_float(0x7f800000);
  }
  float weight=lane<10?expf(selected_logit-mx):0;
  if(arm==0)weight=mi_rb_rne(weight*norm);
  float total=0;
  for(int j=0;j<10;j++) {
    float wv=__shfl_sync(0xffffffffU,weight,j);
    total=arm?total+wv:mi_rb_rne(total+wv);
  }
  if(lane<10) {
    ids[lane]=selected;
    scores[lane]=invalid?__int_as_float(0x7fc00000):(arm?weight/total:mi_rb_rne(weight/total));
  }
}
// Count this block done for counter c (of n); true in the one block that
// completes the count. Every thread fences its own writes first.
__device__ __forceinline__ bool mf_done(unsigned* ctr,unsigned c,unsigned n,unsigned* flag) {
  __threadfence();
  __syncthreads();
  if(threadIdx.x==0) { const unsigned old=atomicAdd(&ctr[c],1u); *flag=(old==n-1u); }
  __syncthreads();
  const bool last=*flag!=0u;
  if(last) __threadfence();
  return last;
}
// Island stage 4, mid branch, for one (slot, base): gate/up split sums read via L2.
__device__ __forceinline__ void mf_mid(const float* gpar,const float* upar,const unsigned* ids,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,unsigned short* middle,
    unsigned slot,unsigned base,unsigned lane) {
  constexpr unsigned SPLITS=4;
  float gr[SPLITS][4],ur[SPLITS][4];
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      const unsigned long long at=((unsigned long long)slot*SPLITS+s)*640+base+lane+j*32;
      gr[s][j]=__ldcg(gpar+at);ur[s][j]=__ldcg(upar+at);
    }
  const unsigned sb=ids[slot]*640+base+lane,yb=slot*640+base+lane;
  unsigned short vgr[4],vur[4],sdr[4];
  #pragma unroll
  for(unsigned j=0;j<4;j++) {vgr[j]=vg[sb+j*32];vur[j]=vu[sb+j*32];sdr[j]=sd[sb+j*32];}
  MiFour g0={{0,0,0,0}},u0={{0,0,0,0}};
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) {g0.v[j]+=gr[s][j];u0.v[j]+=ur[s][j];}
  MiFour g=mi_had(g0,lane),uu=mi_had(u0,lane),vv;
  #pragma unroll
  for(unsigned j=0;j<4;j++) {
    float a=g.v[j]*mi_hf(vgr[j]),b=uu.v[j]*mi_hf(vur[j]);
    float ex=expf(-fabsf(a)),sig=a<0?ex/(1.0f+ex):1.0f/(1.0f+ex);
    vv.v[j]=a*sig*b*mi_hf(sdr[j]);
  }
  vv=mi_had(vv,lane);
  #pragma unroll
  for(unsigned j=0;j<4;j++)middle[yb+j*32]=mi_fh(vv.v[j]);
}
// Island stage 4, shared gate tail (one 128-thread block, row 0).
__device__ __forceinline__ void mf_shared_finish(const unsigned short* x,const unsigned short* scalar_gate,
    const unsigned short* dp_b,unsigned short* shared_out,float* sraw,unsigned short* slogits,
    unsigned short* ssig,int shared_arm,float* smem) {
  const unsigned t=threadIdx.x;
  unsigned short xr[20],wr[20],dr[20];
  #pragma unroll
  for(int i=0;i<20;i++) {
    const int k=(int)t+128*i;
    xr[i]=x[k];wr[i]=scalar_gate[k];dr[i]=dp_b[k];
  }
  float sum=0;
  #pragma unroll
  for(int i=0;i<20;i++) sum=fmaf(mi_load(xr[i]),mi_load(wr[i]),sum);
  for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
  if((t&31)==0) smem[t>>5]=sum;
  __syncthreads();
  if(t<32) {
    sum=t<4?smem[t]:0;
    for(int s=16;s;s>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,s);
    if(t==0) {
      sraw[0]=sum;
      unsigned short logit=mi_store_rne(sum);
      float p=mi_shared_sig(mi_load(logit),shared_arm);
      slogits[0]=logit; ssig[0]=mi_store_rne(p); smem[8]=p;
    }
  }
  __syncthreads();
  const float p=smem[8];
  #pragma unroll
  for(int i=0;i<20;i++) shared_out[(int)t+128*i]=mi_store_rne(mi_load(dr[i])*p);
  __syncthreads();
}
// F0: 128 router blocks (four experts each) and 640 shared partial vblock pairs.
extern "C" __global__ void __launch_bounds__(128,5) moe_f0_r38(MF_PARAMS) {
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,SPLITS=4;
  __shared__ unsigned flag;
  const bool router=pb<128u;
  if(router) mi_router_qmv(x,rw,rraw,pb*4+warp,lane);
  else {
    const unsigned v=2*(pb-128)+(warp>>1);
    mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
  }
  if(!mf_done(ctr,router?0u:1u,router?128u:640u,&flag)) return;
  if(router) {
    if(warp==0) mf_router_select(rraw,rlogits,ids,scores,select_arm,lane);
  } else {
    // Island stage 1 (reduce, at=t+128i) then stage 2 (activate, j=t+128i):
    // thread t's activation inputs gu_b[j] and gu_b[I+j] are its own values.
    unsigned short bits[10];
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const unsigned at=t+128u*i;
      float value=0.f;
      for(unsigned p=0;p<SPLITS;p++) value+=__ldcg(gpart+p*2*I+at);
      gu_f[at]=value;bits[i]=mi_store_rne(value);gu_b[at]=bits[i];
    }
    #pragma unroll
    for(unsigned i=0;i<5;i++) {
      const unsigned j=t+128u*i;
      const float g=mi_load(bits[i]),uv=mi_load(bits[i+5]);
      const float a=g*mi_shared_sig(g,shared_arm);
      act[j]=mi_store_rne((shared_arm?a:mi_rb_rne(a))*uv);
    }
  }
  if(t==0) ctr[router?0u:1u]=0u;
}
// F1: 50 prepare blocks (200 warps) and 320 shared down vblock pairs.
extern "C" __global__ void __launch_bounds__(128) moe_f1_r38(MF_PARAMS) {
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640;
  (void)t;
  if(pb<50u) {
    const unsigned v=pb*4+warp,slot=v/20,base=(v%20)*128;
    const unsigned xb=base+lane,sb=ids[slot]*2560+base+lane,yb=slot*2560+base+lane;
    MiFour g,uu;
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      g.v[j]=mi_load(x[xb+j*32])*mi_hf(sg[sb+j*32]);uu.v[j]=mi_load(x[xb+j*32])*mi_hf(su[sb+j*32]);
    }
    g=mi_had(g,lane);uu=mi_had(uu,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++) { pg[yb+j*32]=mi_fh(g.v[j]);pu[yb+j*32]=mi_fh(uu.v[j]); }
  } else {
    const unsigned v=2*(pb-50)+(warp>>1);
    mi_direct(act,dw,ds,db,dp_f,dp_b,I,H,dgroup,v,warp&1,lane);
  }
}
// F2: block 0 = shared gate tail; then M-tile gate/up MMA blocks, the last of
// each slot's 2*SPLITS*(40/M) blocks running that slot's mid (five 128-wide
// pieces over the block's four warps).
#define MF_F2(NAME,M) \
extern "C" __global__ void __launch_bounds__(128) NAME(MF_PARAMS) { \
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5; \
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4,NT=40/(M); \
  __shared__ float smem[(M)*64>16?(M)*64:16]; \
  __shared__ unsigned flag; \
  if(pb==0u) { mf_shared_finish(x,scalar_gate,dp_b,shared_out,sraw,slogits,ssig,shared_arm,smem); return; } \
  const unsigned q=pb-1u,g=q%NT,yy=(q/NT)%20,split=q/(NT*20); \
  const bool second=yy>=E; \
  const unsigned slot=second?yy-E:yy; \
  ms_mma_tiles<M>(second?pu:pg,second?tu:tg,second?upar:gpar,ids,slot,H,I,SPLITS,g*(M),split,smem); \
  if(!mf_done(ctr,2u+slot,NT*2u*SPLITS,&flag)) return; \
  for(unsigned c=warp;c<5u;c+=4u) mf_mid(gpar,upar,ids,vg,vu,sd,middle,slot,c*128u,lane); \
  if(t==0) ctr[2u+slot]=0u; \
}
MF_F2(moe_f2_m2_r38,2)
MF_F2(moe_f2_m4_r38,4)

// ---- F0b: F0 with a one-trip router projection and a tree-scan selection ----
// Router warp e accumulates exactly mi_router_qmv's sequence (k = lane+32j,
// j = 0..79 in order, then the same shuffle reduction); its 80 weight words are
// requested before any arithmetic and the block's input vector comes from
// shared memory. The selection's per-round local maximum uses a 4-level tree
// under the same total order (value descending, then lower expert id), which
// has the same unique maximum as the 16-step scan; everything else is
// mf_router_select.
__device__ __forceinline__ void mf_better(float& k1,unsigned& e1,float k2,unsigned e2) {
  if(k2>k1 || (k2==k1 && e2<e1)) { k1=k2;e1=e2; }
}
__device__ __forceinline__ void mf_router_select_tree(const float* raw,unsigned short* logits,unsigned* ids,
    float* scores,int arm,unsigned lane) {
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=__ldcg(raw+e);unsigned short h=mi_store_rne(v);logits[e]=h;
    bool bad=!isfinite(v)||!isfinite(mi_load(h));invalid|=bad;
    key[j]=bad?-3.402823466e38f:mi_load(h);mx=fmaxf(mx,key[j]);
  }
  invalid=__any_sync(0xffffffffU,invalid);
  for(int s=16;s;s>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffU,mx,s));
  float norm=0;
  if(arm==0) {
    #pragma unroll
    for(int j=0;j<16;j++)norm+=expf(key[j]-mx);
    for(int s=16;s;s>>=1)norm+=__shfl_xor_sync(0xffffffffU,norm,s);
    norm=1.0f/norm;
  }
  unsigned selected=0;
  float selected_logit=0;
  for(int rank=0;rank<10;rank++) {
    float k8[8];unsigned e8[8];
    #pragma unroll
    for(int i=0;i<8;i++) {
      k8[i]=key[2*i];e8[i]=(unsigned)(2*i*32)+lane;
      mf_better(k8[i],e8[i],key[2*i+1],(unsigned)((2*i+1)*32)+lane);
    }
    #pragma unroll
    for(int i=0;i<4;i++) mf_better(k8[2*i],e8[2*i],k8[2*i+1],e8[2*i+1]);
    mf_better(k8[0],e8[0],k8[2],e8[2]);mf_better(k8[4],e8[4],k8[6],e8[6]);
    mf_better(k8[0],e8[0],k8[4],e8[4]);
    float best=k8[0];unsigned id=e8[0];
    for(int s=16;s;s>>=1) {
      float v=__shfl_xor_sync(0xffffffffU,best,s);unsigned e=__shfl_xor_sync(0xffffffffU,id,s);
      if(v>best || (v==best && e<id)){best=v;id=e;}
    }
    if((int)lane==rank){selected=id;selected_logit=best;}
    #pragma unroll
    for(int j=0;j<16;j++)if((unsigned)(j*32+lane)==id)key[j]=-__int_as_float(0x7f800000);
  }
  float weight=lane<10?expf(selected_logit-mx):0;
  if(arm==0)weight=mi_rb_rne(weight*norm);
  float total=0;
  for(int j=0;j<10;j++) {
    float wv=__shfl_sync(0xffffffffU,weight,j);
    total=arm?total+wv:mi_rb_rne(total+wv);
  }
  if(lane<10) {
    ids[lane]=selected;
    scores[lane]=invalid?__int_as_float(0x7fc00000):(arm?weight/total:mi_rb_rne(weight/total));
  }
}
extern "C" __global__ void __launch_bounds__(128,5) moe_f0b_r38(MF_PARAMS) {
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,SPLITS=4;
  __shared__ unsigned flag;
  __shared__ __align__(16) unsigned short sxv[H];
  const bool router=pb<128u;
  if(router) {
    const unsigned e=pb*4+warp;
    unsigned short wr[80];
    #pragma unroll
    for(int j=0;j<80;j++) wr[j]=rw[e*2560+lane+32*j];
    for(unsigned i=t;i<H/8;i+=128)
      reinterpret_cast<uint4*>(sxv)[i]=reinterpret_cast<const uint4*>(x)[i];
    __syncthreads();
    float acc=0;
    #pragma unroll
    for(int j=0;j<80;j++) acc=fmaf(mi_load(sxv[lane+32*j]),mi_load(wr[j]),acc);
    for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
    if(lane==0) rraw[e]=acc;
  } else {
    const unsigned v=2*(pb-128)+(warp>>1);
    mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
  }
  if(!mf_done(ctr,router?0u:1u,router?128u:640u,&flag)) return;
  if(router) {
    if(warp==0) mf_router_select_tree(rraw,rlogits,ids,scores,select_arm,lane);
  } else {
    unsigned short bits[10];
    #pragma unroll
    for(unsigned i=0;i<10;i++) {
      const unsigned at=t+128u*i;
      float value=0.f;
      for(unsigned p=0;p<SPLITS;p++) value+=__ldcg(gpart+p*2*I+at);
      gu_f[at]=value;bits[i]=mi_store_rne(value);gu_b[at]=bits[i];
    }
    #pragma unroll
    for(unsigned i=0;i<5;i++) {
      const unsigned j=t+128u*i;
      const float g=mi_load(bits[i]),uv=mi_load(bits[i+5]);
      const float a=g*mi_shared_sig(g,shared_arm);
      act[j]=mi_store_rne((shared_arm?a:mi_rb_rne(a))*uv);
    }
  }
  if(t==0) ctr[router?0u:1u]=0u;
}
