// R38 (Opus, direct): one-launch MoE island for a single decode row.
// Seven stages separated by grid-wide counter barriers replace twelve launches:
// router, shared expert (gate/up split4, reduce, activate, down, gate tail) and
// routed experts (prepare, gate+up MMA, mid, down MMA, finish + shared add).
// Every stage body is a port of the kernel it replaces with the same per-thread
// arithmetic and order (lowbit_router.cu, affine4_split.cu, lowbit_shared.cu,
// moe_join.cu, mcg_tensor.cu via mcg_tensor_ahead.cu, mcg_moe.cu, affine4.cu
// via affine4_ahead.cu); see NOTICE.txt. Outputs are intended to match exactly.
// The grid must be fully resident (the host sizes it); a bounded spin sets
// status and abandons the step instead of hanging. The last block out resets
// the counters so graph replays start clean. After routing, all blocks prefetch
// the ten selected experts' packed weights into L2 while later stages run.

// ---- shared helpers -------------------------------------------------------
__device__ __forceinline__ float mi_load(unsigned short x) { return __uint_as_float((unsigned)x<<16); }
// Manual round-to-nearest-even used by lowbit_router/lowbit_shared/affine4.
__device__ __forceinline__ unsigned short mi_store_rne(float x) {
  unsigned u=__float_as_uint(x);return (unsigned short)((u+0x7fffU+((u>>16)&1U))>>16);
}
__device__ __forceinline__ float mi_rb_rne(float x) { return mi_load(mi_store_rne(x)); }
__device__ __forceinline__ unsigned short mi_store_cvt(float x) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ float mi_hf(unsigned short h) {
  float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
__device__ __forceinline__ unsigned short mi_fh(float x) {
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
struct MiFour { float v[4]; };
__device__ __forceinline__ MiFour mi_had(MiFour x, unsigned lane) {
  #pragma unroll
  for(unsigned bit=1;bit<=16;bit<<=1) {
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      float other=__shfl_xor_sync(0xffffffffu,x.v[j],bit);
      x.v[j]=(lane&bit)==0?x.v[j]+other:other-x.v[j];
    }
  }
  float a=x.v[0]+x.v[1],b=x.v[0]-x.v[1],c=x.v[2]+x.v[3],d=x.v[2]-x.v[3];
  MiFour y={{a+c,b+d,a-c,b-d}};
  #pragma unroll
  for(unsigned j=0;j<4;j++)y.v[j]*=0.08838834764831845f;
  return y;
}
__device__ __forceinline__ float mi_shared_sig(float x,int arm) {
  if(arm) return 1.0f/(1.0f+expf(-x));
  float small=mi_rb_rne(1.0f/mi_rb_rne(1.0f+mi_rb_rne(expf(fabsf(x)))));
  return x<0?small:mi_rb_rne(1.0f-small);
}
__device__ __forceinline__ unsigned mi_ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void mi_red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void mi_prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
// Two-level grid barrier. Region layout: 32 group counters, one top counter,
// one release flag (34 words). Arrivals spread over 32 counters; the last block
// of each group arrives at the top; the last group sets the flag that all
// blocks poll. Returns false on spin timeout (status set, step abandoned).
#define MI_BAR_WORDS 34u
__device__ __forceinline__ void mi_st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ bool mi_barrier(unsigned* bar,unsigned* status,unsigned spin_limit,
    unsigned* fail_flag) {
  __syncthreads();
  if(threadIdx.x==0) {
    const unsigned P=gridDim.x,g=blockIdx.x&31u;
    const unsigned members=(P+31u-g)/32u,groups=P<32u?P:32u;
    __threadfence();
    if(atomicAdd(&bar[g],1u)==members-1u) {
      __threadfence();
      if(atomicAdd(&bar[32],1u)==groups-1u) { __threadfence(); mi_st_release(&bar[33],1u); }
    }
    unsigned spins=0;
    while(mi_ld_acquire(&bar[33])==0u) {
      if(++spins>spin_limit) { *fail_flag=1; atomicExch(status,1u); break; }
    }
    __threadfence();
  }
  __syncthreads();
  return *fail_flag==0;
}

// ---- stage bodies -----------------------------------------------------------
// router_qmv_ahead: one warp per expert e.
__device__ __forceinline__ void mi_router_qmv(const unsigned short* x,const unsigned short* w,float* y,
    unsigned e,unsigned lane) {
  float acc=0;
  #pragma unroll
  for(int b=0;b<4;b++) {
    unsigned short xr[20],wr[20];
    #pragma unroll
    for(int i=0;i<20;i++) { const int k=lane+32*(b*20+i); xr[i]=x[k]; wr[i]=w[e*2560+k]; }
    #pragma unroll
    for(int i=0;i<20;i++) acc=fmaf(mi_load(xr[i]),mi_load(wr[i]),acc);
  }
  for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
  if(lane==0) y[e]=acc;
}
// router_select, one warp, row 0.
__device__ __forceinline__ void mi_router_select(const float* raw,unsigned short* logits,unsigned* ids,
    float* scores,int arm,unsigned lane) {
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=raw[e];unsigned short h=mi_store_rne(v);logits[e]=h;
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
// affine4 load-ahead GEMV body (affine4_ahead.cu ahead_acc, NV=1, D=2).
template<bool Factored>
__device__ __forceinline__ void mi_affine_acc(const unsigned short* x,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,unsigned k,unsigned n,
    unsigned group,unsigned begin,unsigned end,unsigned n0,unsigned lane,float acc[2]) {
  const unsigned groups=k/group;
  for(unsigned base=begin;base<end;base+=512u) {
    unsigned xw[2][4],packed[2][2];
    unsigned short sr[2][2],br[2][2];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned kb=base+d*256u,k0=kb+lane*8u;
      const bool ok=kb<end && k0+8u<=k;
      const uint4 q=ok?*reinterpret_cast<const uint4*>(x+k0):make_uint4(0u,0u,0u,0u);
      xw[d][0]=q.x;xw[d][1]=q.y;xw[d][2]=q.z;xw[d][3]=q.w;
      #pragma unroll
      for(unsigned col=0;col<2;col++) {
        const bool okc=kb<end && k0<k && n0+col<n;
        const unsigned at=(n0+col)*groups+k0/group;
        sr[d][col]=okc?scales[at]:(unsigned short)0;
        br[d][col]=okc?biases[at]:(unsigned short)0;
        packed[d][col]=okc?w[(size_t)(n0+col)*(k/8u)+k0/8u]:0u;
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned kb=base+d*256u,k0=kb+lane*8u;
      float vals[8],sums=0.f;
      #pragma unroll
      for(unsigned i=0;i<8;i++) {
        const unsigned word=xw[d][i>>1];
        const float a=__uint_as_float((i&1u)?(word&0xffff0000u):(word<<16));
        vals[i]=a;sums+=a;
      }
      if(kb<end && k0<k) {
        #pragma unroll
        for(unsigned col=0;col<2;col++) if(n0+col<n) {
          const float s=mi_load(sr[d][col]),b=mi_load(br[d][col]);
          const unsigned p=packed[d][col];
          if constexpr(Factored) {
            float dot=0.f;
            #pragma unroll
            for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((p>>(4*i))&15u),dot);
            acc[col]+=fmaf(s,dot,b*sums);
          } else {
            #pragma unroll
            for(unsigned i=0;i<8;i++) {
              const float decoded=fmaf(s,float((p>>(4*i))&15u),b);
              acc[col]=fmaf(vals[i],decoded,acc[col]);
            }
          }
        }
      }
    }
  }
}
__device__ __forceinline__ float mi_warp_total(float v) {
  #pragma unroll
  for(unsigned delta=16;delta;delta>>=1) v+=__shfl_down_sync(0xffffffffu,v,delta);
  return v;
}
// 64-thread affine4 split partial vblock (bx, part), factored arithmetic.
__device__ __forceinline__ void mi_partial(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* partial,unsigned k,unsigned n,
    unsigned group,unsigned splits,unsigned bx,unsigned part,unsigned warp2,unsigned lane) {
  const unsigned n0=bx*4+warp2*2;
  const unsigned chunks=(k+255)/256,begin=(chunks*part/splits)*256,end=(chunks*(part+1)/splits)*256;
  float acc[2]={0.f,0.f};
  mi_affine_acc<true>(x,w,s,b,k,n,group,begin,end,n0,lane,acc);
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    const float v=mi_warp_total(acc[col]);
    if(lane==0 && n0+col<n) partial[(0*splits+part)*n+n0+col]=v;
  }
}
// 64-thread affine4 direct vblock bx (row 0).
__device__ __forceinline__ void mi_direct(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* y,unsigned short* ybf,unsigned k,
    unsigned n,unsigned group,unsigned bx,unsigned warp2,unsigned lane) {
  const unsigned n0=bx*4+warp2*2;
  float acc[2]={0.f,0.f};
  mi_affine_acc<false>(x,w,s,b,k,n,group,0,k,n0,lane,acc);
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    const float v=mi_warp_total(acc[col]);
    if(lane==0 && n0+col<n) { y[n0+col]=v; ybf[n0+col]=mi_store_rne(v); }
  }
}
// MCG MMA tile (mcg_tensor_ahead.cu mta_project_*): 128 threads, direct mode.
__device__ __forceinline__ unsigned mi_window(unsigned prev_word,unsigned last_word,unsigned lane) {
  const unsigned end=lane*14+14,last=(end-1)>>5,shift=(last+1)*32-end;
  const unsigned long long bits=((unsigned long long)prev_word<<32)|last_word;
  return (unsigned)(bits>>shift);
}
__device__ __forceinline__ unsigned mi_pair_values(unsigned window,unsigned pair) {
  const unsigned offset=pair==0?11:pair==1?7:pair==2?4:0;
  const unsigned cw=window>>offset;
  const unsigned c0=(cw>>2)&0x7fffu,c1=cw&0x7fffu;
  const unsigned p0=((c0*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
  const unsigned p1=((c1*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
  const unsigned lo=(p0&0xffffu)|(p1<<16),hi=(p0>>16)|(p1&0xffff0000u);
  unsigned result; asm("add.rn.f16x2 %0, %1, %2;":"=r"(result):"r"(lo),"r"(hi));
  return result;
}
__device__ __forceinline__ void mi_mma_tile(const unsigned short* x,const unsigned* trellis,float* y,
    const unsigned* desc,unsigned slot,unsigned k,unsigned n,unsigned splits,unsigned ot,
    unsigned split,float* partial) {
  const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
  const unsigned row=(lane&3)*2,col=lane>>2;
  const unsigned word_end=lane*14+14,word_last=(word_end-1)>>5;
  const unsigned word_prev=word_last?word_last-1:13;
  const unsigned begin=split*per,end=min(begin+per,it);
  const unsigned expert=desc[slot];
  const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
  float acc[4]={};
  for(unsigned tk0=begin+warp;tk0<end;tk0+=8) {
    unsigned wp[2],wl[2],b0[2],b1[2];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned tk=tk0+d*4;
      const bool live=tk<end;
      const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
      wp[d]=live?words[word_prev]:0u;
      wl[d]=live?words[word_last]:0u;
      const bool xin=live && col<1;
      const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
      const unsigned x0v=xin?xx[row]:0u,x1v=xin?xx[row+1]:0u;
      const unsigned x8=xin?xx[row+8]:0u,x9=xin?xx[row+9]:0u;
      b0[d]=x0v|(x1v<<16);
      b1[d]=x8|(x9<<16);
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned tk=tk0+d*4;
      if(tk<end) {
        const unsigned window=mi_window(wp[d],wl[d],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=mi_pair_values(window,p);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
      }
    }
  }
  partial[warp*128+row*16+col]=acc[0];
  partial[warp*128+(row+1)*16+col]=acc[1];
  partial[warp*128+row*16+col+8]=acc[2];
  partial[warp*128+(row+1)*16+col+8]=acc[3];
  __syncthreads();
  for(unsigned i=tid;i<16;i+=128) {
    float sum=0;for(unsigned w=0;w<4;w++)sum+=partial[w*128+i];
    y[((unsigned long long)slot*splits+split)*n+ot*16+i%16]=sum;
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(128) moe_island_r38(
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
    unsigned last_stage,unsigned* counters,unsigned* status,unsigned spin_limit) {
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4;
  __shared__ float smem[10*128];
  __shared__ unsigned fail;
  if(t==0) fail=0;
  __syncthreads();
  do {
  // Stage 0: router projection || shared gate/up split4 partials (N=1280).
  for(unsigned u=pb;u<128+640;u+=P) {
    if(u<128) mi_router_qmv(x,rw,rraw,u*4+warp,lane);
    else {
      const unsigned v=2*(u-128)+(warp>>1);
      mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
    }
  }
  if(!mi_barrier(counters+0*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=0u) break;
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
  if(!mi_barrier(counters+1*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=1u) break;
  // Routes are known: prefetch the ten experts' packed weights into L2.
  {
    const unsigned per=2560u*640u/256u*56u;  // bytes per expert per projection
    const unsigned lines=per/128u,total=3u*E*lines;
    for(unsigned i=pb*128u+t;i<total;i+=P*128u) {
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
  if(!mi_barrier(counters+2*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=2u) break;
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
  if(!mi_barrier(counters+3*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=3u) break;
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
  if(!mi_barrier(counters+4*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=4u) break;
  // Stage 5: routed down MMA (160 tiles x 10 slots x 4 splits).
  for(unsigned u=pb;u<6400;u+=P) {
    const unsigned ot=u%160,slot=(u/160)%10,split=u/1600;
    mi_mma_tile(middle,td,dpar,ids,slot,I,H,SPLITS,ot,split,smem);
  }
  if(!mi_barrier(counters+5*MI_BAR_WORDS,status,spin_limit,&fail)) return;
  if(last_stage<=5u) break;
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
  } while(0);
  // Exit and reset for the next replay.
  __syncthreads();
  if(t==0) {
    __threadfence();
    const unsigned out=atomicAdd(&counters[6*MI_BAR_WORDS],1u);
    if(out==P-1) {
      for(unsigned s=0;s<=6*MI_BAR_WORDS;s++) counters[s]=0;
      __threadfence();
    }
  }
}
