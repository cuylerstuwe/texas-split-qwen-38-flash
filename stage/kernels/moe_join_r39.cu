// GENERATED R38 combination: unchanged moe_join.cu followed by moe_island.cu, moe_stage.cu, moe_stage_m.cu and moe_fused.cu.
// Regenerate by concatenation; do not edit here. See the sources and NOTICE.txt.
// Routed expert transforms, adapted from the licensed Sushi-derived Metal
// reference in exl3_moe.metal. See NOTICE.txt. No weight requantization.
// Fixed model H=2560, I=640, topK=10; Rust validates buffers/IDs/launch geometry.
__device__ __forceinline__ float hf(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
__device__ __forceinline__ unsigned short fh(float x) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
struct Four { float v[4]; };
__device__ __forceinline__ Four had(Four x, unsigned lane) {
    #pragma unroll
    for(unsigned bit=1;bit<=16;bit<<=1) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) {
            float other=__shfl_xor_sync(0xffffffffu,x.v[j],bit);
            x.v[j]=(lane&bit)==0?x.v[j]+other:other-x.v[j];
        }
    }
    float a=x.v[0]+x.v[1],b=x.v[0]-x.v[1],c=x.v[2]+x.v[3],d=x.v[2]-x.v[3];
    Four y={{a+c,b+d,a-c,b-d}};
    #pragma unroll
    for(unsigned j=0;j<4;j++)y.v[j]*=0.08838834764831845f;
    return y;
}
// BF16 input is an explicit ABI, not a reinterpretation of the R16 FP32 source.
__device__ __forceinline__ float bf(unsigned short h) { return __uint_as_float((unsigned)h<<16); }
extern "C" __global__ void moe_prepare_bf16(const unsigned short* x,const unsigned short* sg,
    const unsigned short* su,const unsigned* ids,unsigned short* pg,unsigned short* pu) {
    const unsigned lane=threadIdx.x,slot=blockIdx.y,base=blockIdx.x*128;
    const unsigned xb=(slot/10)*2560+base+lane,sb=ids[slot]*2560+base+lane,yb=slot*2560+base+lane;
    Four g,u;
    #pragma unroll
    for(unsigned j=0;j<4;j++) { g.v[j]=bf(x[xb+j*32])*hf(sg[sb+j*32]);u.v[j]=bf(x[xb+j*32])*hf(su[sb+j*32]); }
    g=had(g,lane);u=had(u,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++) { pg[yb+j*32]=fh(g.v[j]);pu[yb+j*32]=fh(u.v[j]); }
}
extern "C" __global__ void moe_add_bf16(const unsigned short* routed,
    const unsigned short* shared, unsigned short* out, unsigned count) {
    unsigned at=blockIdx.x*128+threadIdx.x;
    if(at<count) {
        float v=bf(routed[at])+bf(shared[at]);
        unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;":"=h"(h):"f"(v)); out[at]=h;
    }
}

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

// R38 (Opus, direct): staged MoE MMA stages with M output tiles per block.
// Each block runs mi_mma_tile's work for M consecutive output tiles of one
// (slot, split): every tile keeps its own accumulator, and each tile's MMA
// sequence (same K tiles, same order, same decode and operands) and its
// cross-warp sum (warps 0..3 in order) are exactly mi_mma_tile's, so outputs
// are intended to match bit for bit. Only the scheduling changes: one block
// issues the trellis loads for M tiles per K step, and the grid fits in about
// one wave instead of several waves of very short blocks. Licensed
// Sushi-derived MCG decode as in mcg_tensor.cu; see NOTICE.txt.
template<unsigned M>
__device__ __forceinline__ void ms_mma_tiles(const unsigned short* x,const unsigned* trellis,float* y,
    const unsigned* desc,unsigned slot,unsigned k,unsigned n,unsigned splits,unsigned ot0,
    unsigned split,float* partial) {
  const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
  const unsigned row=(lane&3)*2,col=lane>>2;
  const unsigned word_end=lane*14+14,word_last=(word_end-1)>>5;
  const unsigned word_prev=word_last?word_last-1:13;
  const unsigned begin=split*per,end=min(begin+per,it);
  const unsigned expert=desc[slot];
  const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
  float acc[M][4];
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  for(unsigned tk0=begin+warp;tk0<end;tk0+=8) {
    unsigned wp[2][M],wl[2][M],b0[2],b1[2];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned tk=tk0+d*4;
      const bool live=tk<end;
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned* words=weights+((unsigned long long)tk*nt+ot0+m)*14;
        wp[d][m]=live?words[word_prev]:0u;
        wl[d][m]=live?words[word_last]:0u;
      }
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
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
          unsigned a[4];
          #pragma unroll
          for(unsigned p=0;p<4;p++)a[p]=mi_pair_values(window,p);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
              : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
        }
      }
    }
  }
  // Only output column 0 (the one token) is kept: rows col and col+8 from the
  // lanes whose (lane&3)==0, i.e. mi_mma_tile's partial[w*128+i] for i<16.
  if((lane&3u)==0u) {
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      partial[(m*4+warp)*16+col]=acc[m][0];
      partial[(m*4+warp)*16+col+8]=acc[m][2];
    }
  }
  __syncthreads();
  for(unsigned i=tid;i<M*16;i+=128) {
    const unsigned m=i/16,r=i%16;
    float sum=0;for(unsigned w=0;w<4;w++)sum+=partial[(m*4+w)*16+r];
    y[((unsigned long long)slot*splits+split)*n+(ot0+m)*16+r]=sum;
  }
  __syncthreads();
}
// Stage 3 with M tiles per block; the shared expert's down projection blocks
// come first so they overlap the MMA blocks instead of forming the tail.
#define MS_STAGE3(NAME,M) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
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
    float* rout,unsigned short* rout_b,unsigned short* added,unsigned prefetch) { \
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5; \
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4,NT=40/(M); \
  __shared__ float smem[(M)*64]; \
  for(unsigned u=pb;u<320+NT*20*SPLITS;u+=P) { \
    if(u<320) { \
      const unsigned v=2*u+(warp>>1); \
      mi_direct(act,dw,ds,db,dp_f,dp_b,I,H,dgroup,v,warp&1,lane); \
    } else { \
      const unsigned q=u-320,g=q%NT,yy=(q/NT)%20,split=q/(NT*20); \
      const bool second=yy>=E; \
      ms_mma_tiles<M>(second?pu:pg,second?tu:tg,second?upar:gpar,ids,second?yy-E:yy,H,I,SPLITS,g*(M),split,smem); \
    } \
  } \
  (void)t; \
}
// Stage 5 (routed down MMA) with M tiles per block.
#define MS_STAGE5(NAME,M) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
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
    float* rout,unsigned short* rout_b,unsigned short* added,unsigned prefetch) { \
  const unsigned P=gridDim.x,pb=blockIdx.x; \
  constexpr unsigned H=2560,I=640,SPLITS=4,NT=160/(M); \
  __shared__ float smem[(M)*64]; \
  for(unsigned u=pb;u<NT*10*SPLITS;u+=P) { \
    const unsigned g=u%NT,slot=(u/NT)%10,split=u/(NT*10); \
    ms_mma_tiles<M>(middle,td,dpar,ids,slot,I,H,SPLITS,g*(M),split,smem); \
  } \
}
MS_STAGE3(moe_stage3_m2_r38,2)
MS_STAGE3(moe_stage3_m4_r38,4)
MS_STAGE3(moe_stage3_m8_r38,8)
MS_STAGE5(moe_stage5_m2_r38,2)
MS_STAGE5(moe_stage5_m4_r38,4)
MS_STAGE5(moe_stage5_m8_r38,8)

// Deeper variant: the block first stages its slot's input slice for this split
// in shared memory (u32 pairs, exactly the b0/b1 words the MMA takes), and each
// warp requests the trellis words of D consecutive chain steps for all M tiles
// before decoding any of them (D=10 covers a whole gate/up chain, D=3 a whole
// down chain). Every tile's MMA sequence, operands and cross-warp sum are
// mi_mma_tile's; lanes 4..31 still pass zero B columns.
template<unsigned M,unsigned D>
__device__ __forceinline__ void ms_mma_deep(const unsigned short* x,const unsigned* trellis,float* y,
    const unsigned* desc,unsigned slot,unsigned k,unsigned n,unsigned splits,unsigned ot0,
    unsigned split,float* partial,unsigned* sx) {
  const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
  const unsigned row=(lane&3)*2,col=lane>>2;
  const unsigned word_end=lane*14+14,word_last=(word_end-1)>>5;
  const unsigned word_prev=word_last?word_last-1:13;
  const unsigned begin=split*per,end=min(begin+per,it);
  const unsigned expert=desc[slot];
  const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
  const unsigned* xw=reinterpret_cast<const unsigned*>(x+(unsigned long long)slot*k+begin*16);
  for(unsigned i=tid;i<(end-begin)*8;i+=128) sx[i]=xw[i];
  float acc[M][4];
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  bool synced=false;
  for(unsigned tk0=begin+warp;;tk0+=4*D) {
    // Uniform trip count across the block so the one __syncthreads below is
    // reached by every warp: the chain of warp 0 is the longest.
    if(begin+4*D*((tk0-begin-warp)/(4*D))>=end) break;
    unsigned wp[D][M],wl[D][M];
    #pragma unroll
    for(unsigned d=0;d<D;d++) {
      const unsigned tk=tk0+d*4;
      const bool live=tk<end;
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned* words=weights+((unsigned long long)tk*nt+ot0+m)*14;
        wp[d][m]=live?words[word_prev]:0u;
        wl[d][m]=live?words[word_last]:0u;
      }
    }
    if(!synced) { __syncthreads(); synced=true; }
    #pragma unroll
    for(unsigned d=0;d<D;d++) {
      const unsigned tk=tk0+d*4;
      if(tk<end) {
        const bool xin=col<1;
        const unsigned b0=xin?sx[(tk-begin)*8+row/2]:0u,b1=xin?sx[(tk-begin)*8+row/2+4]:0u;
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
          unsigned a[4];
          #pragma unroll
          for(unsigned p=0;p<4;p++)a[p]=mi_pair_values(window,p);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
              : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
        }
      }
    }
  }
  if(!synced) __syncthreads();
  if((lane&3u)==0u) {
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      partial[(m*4+warp)*16+col]=acc[m][0];
      partial[(m*4+warp)*16+col+8]=acc[m][2];
    }
  }
  __syncthreads();
  for(unsigned i=tid;i<M*16;i+=128) {
    const unsigned m=i/16,r=i%16;
    float sum=0;for(unsigned w=0;w<4;w++)sum+=partial[(m*4+w)*16+r];
    y[((unsigned long long)slot*splits+split)*n+(ot0+m)*16+r]=sum;
  }
  __syncthreads();
}
#define MS_ARGS \
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
    float* rout,unsigned short* rout_b,unsigned short* added,unsigned prefetch
// Stage 3 / stage 5 with ms_mma_deep (same work items as MS_STAGE3/MS_STAGE5).
#define MS_STAGE3D(NAME,M,D) \
extern "C" __global__ void __launch_bounds__(128) NAME(MS_ARGS) { \
  const unsigned P=gridDim.x,pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5; \
  constexpr unsigned H=2560,I=640,E=10,SPLITS=4,NT=40/(M); \
  __shared__ float smem[(M)*64]; \
  __shared__ unsigned sx[40*8]; \
  for(unsigned u=pb;u<320+NT*20*SPLITS;u+=P) { \
    if(u<320) { \
      const unsigned v=2*u+(warp>>1); \
      mi_direct(act,dw,ds,db,dp_f,dp_b,I,H,dgroup,v,warp&1,lane); \
    } else { \
      const unsigned q=u-320,g=q%NT,yy=(q/NT)%20,split=q/(NT*20); \
      const bool second=yy>=E; \
      ms_mma_deep<M,D>(second?pu:pg,second?tu:tg,second?upar:gpar,ids,second?yy-E:yy,H,I,SPLITS,g*(M),split,smem,sx); \
    } \
  } \
  (void)t; \
}
#define MS_STAGE5D(NAME,M,D) \
extern "C" __global__ void __launch_bounds__(128) NAME(MS_ARGS) { \
  const unsigned P=gridDim.x,pb=blockIdx.x; \
  constexpr unsigned H=2560,I=640,SPLITS=4,NT=160/(M); \
  __shared__ float smem[(M)*64]; \
  __shared__ unsigned sx[10*8]; \
  for(unsigned u=pb;u<NT*10*SPLITS;u+=P) { \
    const unsigned g=u%NT,slot=(u/NT)%10,split=u/(NT*10); \
    ms_mma_deep<M,D>(middle,td,dpar,ids,slot,I,H,SPLITS,g*(M),split,smem,sx); \
  } \
}
MS_STAGE3D(moe_stage3_deep2_r38,2,10)
MS_STAGE3D(moe_stage3_deep4_r38,4,5)
MS_STAGE5D(moe_stage5_deep4_r38,4,3)
MS_STAGE5D(moe_stage5_deep8_r38,8,3)

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

// GENERATED by tools/build_r39_moe_source.py; unchanged R38 arithmetic.
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_f0_r38(MF_PARAMS) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
extern "C" __global__ void __launch_bounds__(128) r39pdl_moe_f1_r38(MF_PARAMS) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
#define R39_MF_F2(NAME,M) \
extern "C" __global__ void __launch_bounds__(128) NAME(MF_PARAMS) { \
  asm volatile("griddepcontrol.wait;" ::: "memory"); \
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
R39_MF_F2(r39pdl_moe_f2_m2_r38,2)
#define R39_MS_STAGE5(NAME,M) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
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
    float* rout,unsigned short* rout_b,unsigned short* added,unsigned prefetch) { \
  asm volatile("griddepcontrol.wait;" ::: "memory"); \
  const unsigned P=gridDim.x,pb=blockIdx.x; \
  constexpr unsigned H=2560,I=640,SPLITS=4,NT=160/(M); \
  __shared__ float smem[(M)*64]; \
  for(unsigned u=pb;u<NT*10*SPLITS;u+=P) { \
    const unsigned g=u%NT,slot=(u/NT)%10,split=u/(NT*10); \
    ms_mma_tiles<M>(middle,td,dpar,ids,slot,I,H,SPLITS,g*(M),split,smem); \
  } \
}
R39_MS_STAGE5(r39pdl_moe_stage5_m4_r38,4)
extern "C" __global__ void __launch_bounds__(128) r39pdl_moe_stage6_r38(
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
  asm volatile("griddepcontrol.wait;" ::: "memory");
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
