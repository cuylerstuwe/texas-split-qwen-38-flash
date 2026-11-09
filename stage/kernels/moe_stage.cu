// R38 (Opus, direct): the MoE island's seven stages as seven ordinary launches.
// GENERATED from moe_island.cu by splitting its body at the grid barriers; each
// entry point is that stage's loop, unchanged, with P = gridDim.x. Kernel
// boundaries replace the barriers, so each stage can use a full-occupancy grid
// (the island's persistent grid was limited by the driver's 32 KB carveout).
// Stage 2 optionally gives blocks past its 55 work items the island's L2
// prefetch of the ten selected experts. Same per-thread arithmetic and order as
// the island (and the kernels it ports); see moe_island.cu and NOTICE.txt.
// Stage 1 declares one block per SM minimum so the compiler keeps the router
// selection in registers (otherwise it used 8 bytes of local memory, which the
// driver backs with a session-lifetime pool).

extern "C" __global__ void __launch_bounds__(128) moe_stage0_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[1];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 0: router projection || shared gate/up split4 partials (N=1280).
  for(unsigned u=pb;u<128+640;u+=P) {
    if(u<128) mi_router_qmv(x,rw,rraw,u*4+warp,lane);
    else {
      const unsigned v=2*(u-128)+(warp>>1);
      mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128,1) moe_stage1_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[1];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 1: router select || shared gate/up reduce (1280 outputs).
  for(unsigned u=pb;u<1+10;u+=P) {
    if(u==0) { if(warp==0) mi_router_select(rraw,rlogits,ids,scores,select_arm,lane); }
    else {
      const unsigned at=(u-1)*128+t;
      if(at<2*I) {
        float value=0.f;
        for(unsigned p=0;p<SPLITS;p++) value+=gpart[p*2*I+at];
        gu_f[at]=value;gu_b[at]=mi_store_rne(value);
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) moe_stage2_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[1];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Routes are known: prefetch the ten experts' packed weights into L2.
  {
    const unsigned per=2560u*640u/256u*56u;  // bytes per expert per projection
    const unsigned lines=per/128u,total=3u*E*lines;
    if(prefetch && pb>=55u) for(unsigned i=(pb-55u)*128u+t;i<total;i+=(P-55u)*128u) {
      const unsigned proj=i/(E*lines),rem=i%(E*lines),slot=rem/lines,line=rem%lines;
      const unsigned* base=proj==0?tg:(proj==1?tu:td);
      mi_prefetch_l2(reinterpret_cast<const char*>(base)+(size_t)ids[slot]*per+(size_t)line*128u);
    }
  }
  // Stage 2: moe_prepare_bf16 (200 warps) || shared_activate (640).
  for(unsigned u=pb;u<50+5;u+=P) {
    if(u<50) {
      const unsigned v=u*4+warp,slot=v/20,base=(v%20)*128;
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
      const int j=(int)(u-50)*128+(int)t;
      if(j<(int)I) {
        const float g=mi_load(gu_b[j]),uv=mi_load(gu_b[I+j]);
        const float a=g*mi_shared_sig(g,shared_arm);
        act[j]=mi_store_rne((shared_arm?a:mi_rb_rne(a))*uv);
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) moe_stage3_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[512];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 3: gate+up MMA (40 tiles x 20 slot-projections x 4 splits) || shared down.
  for(unsigned u=pb;u<3200+320;u+=P) {
    if(u<3200) {
      const unsigned ot=u%40,y=(u/40)%20,split=u/800;
      const bool second=y>=E;
      mi_mma_tile(second?pu:pg,second?tu:tg,second?upar:gpar,ids,second?y-E:y,H,I,SPLITS,ot,split,smem);
    } else {
      const unsigned v=2*(u-3200)+(warp>>1);
      mi_direct(act,dw,ds,db,dp_f,dp_b,I,H,dgroup,v,warp&1,lane);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) moe_stage4_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[16];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 4: mcg_moe_mid (50 warps) || shared gate tail (one 128-thread block).
  for(unsigned u=pb;u<13+1;u+=P) {
    if(u<13) {
      const unsigned v=u*4+warp;
      if(v<50) {
        const unsigned slot=v/5,base=(v%5)*128;
        float gr[SPLITS][4],ur[SPLITS][4];
        #pragma unroll
        for(unsigned s=0;s<SPLITS;s++)
          #pragma unroll
          for(unsigned j=0;j<4;j++) {
            const unsigned long long at=((unsigned long long)slot*SPLITS+s)*640+base+lane+j*32;
            gr[s][j]=gpar[at];ur[s][j]=upar[at];
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
    } else {
      // shared_finish_ahead, row 0, hidden H.
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
  }
}

extern "C" __global__ void __launch_bounds__(128) moe_stage5_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[512];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 5: routed down MMA (160 tiles x 10 slots x 4 splits).
  for(unsigned u=pb;u<6400;u+=P) {
    const unsigned ot=u%160,slot=(u/160)%10,split=u/1600;
    mi_mma_tile(middle,td,dpar,ids,slot,I,H,SPLITS,ot,split,smem);
  }
}

extern "C" __global__ void __launch_bounds__(128) moe_stage6_r38(
    const unsigned short* x,const unsigned short* rw,float* rraw,unsigned short* rlogits,
    unsigned* ids,float* scores,int select_arm,
    const unsigned* gw,const unsigned short* gs,const unsigned short* gb,unsigned ggroup,
    float* gpart,float* gu_f,unsigned short* gu_b,
    unsigned short* act,const unsigned* dw,const unsigned short* ds,const unsigned short* db,
    unsigned dgroup,float* dp_f,unsigned short* dp_b,
    const unsigned short* scalar_gate,unsigned short* shared_out,float* sraw,
    unsigned short* slogits,unsigned short* ssig,int shared_arm,
    const unsigned short* sg,const unsigned short* su,unsigned short* pg,unsigned short* pu,
    const unsigned* tg,const unsigned* tu,float* gpar,float* upar,
    const unsigned short* vg,const unsigned short* vu,const unsigned short* sd,
    unsigned short* middle,const unsigned* td,float* dpar,const unsigned short* sv,
    float* rout,unsigned short* rout_b,unsigned short* added,
    unsigned prefetch) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[1280];
  (void)P;(void)pb;(void)t;(void)lane;(void)warp;(void)H;(void)I;(void)E;(void)SPLITS;(void)smem;(void)prefetch;
  // Stage 6: finish (four warps cover the ten experts) + shared add.
  for(unsigned u=pb;u<20;u+=P) {
    const unsigned base=u*128;
    for(unsigned e=warp;e<E;e+=4) {
      float dr[SPLITS][4];
      #pragma unroll
      for(unsigned s=0;s<SPLITS;s++)
        #pragma unroll
        for(unsigned j=0;j<4;j++) dr[s][j]=dpar[((unsigned long long)e*SPLITS+s)*2560+base+lane+j*32];
      const unsigned sb=ids[e]*2560+base+lane;
      unsigned short svr[4];
      #pragma unroll
      for(unsigned j=0;j<4;j++) svr[j]=sv[sb+j*32];
      MiFour acc={{0,0,0,0}};
      #pragma unroll
      for(unsigned s=0;s<SPLITS;s++)
        #pragma unroll
        for(unsigned j=0;j<4;j++) acc.v[j]+=dr[s][j];
      MiFour v=mi_had(acc,lane);
      #pragma unroll
      for(unsigned j=0;j<4;j++) smem[e*128+lane+j*32]=v.v[j]*mi_hf(svr[j]);
    }
    __syncthreads();
    if(warp==0) {
      float sc[E];
      #pragma unroll
      for(unsigned e=0;e<E;e++) sc[e]=scores[e];
      MiFour sum={{0,0,0,0}};
      for(unsigned e=0;e<E;e++) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) sum.v[j]+=smem[e*128+lane+j*32]*sc[e];
      }
      #pragma unroll
      for(unsigned j=0;j<4;j++) {
        const unsigned at=base+lane+j*32;
        rout[at]=sum.v[j];
        const unsigned short rbits=mi_store_rne(sum.v[j]);
        rout_b[at]=rbits;
        added[at]=mi_store_cvt(mi_load(rbits)+mi_load(shared_out[at]));
      }
    }
    __syncthreads();
  }
}
