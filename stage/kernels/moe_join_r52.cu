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

// GENERATED by tools/build_r52_moe_source.py
// ===== R52 (Opus, direct): MoE back half as ONE launch with ticket-ordered dataflow =====
// Replaces F1 (prepare || shared down), F2 (gate/up MMA, per-slot mid, shared gate
// tail), stage5 (down MMA) and stage6 (finish + shared add) of the R38 fused path.
// Each block takes the next ticket from an atomic counter and runs the work item
// with that number. Items are numbered so that every wait is on items with LOWER
// numbers, i.e. on blocks that have already started; a started block is resident
// and finishes, so neither residency nor launch order is assumed:
//   [0,320)          shared down (mi_direct)  -> last block: shared gate tail, flag
//   [320,320+NG)     gate/up MMA, slot-major  -> last block per (slot,128-piece): mid, flag
//   [320+NG,T)       down MMA, slot-major; the block's trellis words are loaded BEFORE
//                    it waits for the (slot,piece) mid flags -> last block per output
//                    128-piece: finish + shared add for that piece.
// The routed input preparation (sign scale + 128-point Hadamard, moe_prepare_bf16)
// is recomputed inside each gate/up block for its own K range with the same
// operations; the first K step's trellis words are requested before it. Each
// value's arithmetic and every reduction order are the R38 fused path's, so the
// outputs are intended to match it bit for bit. Data written by other blocks of
// this launch is read through L2 (ld.global.cg) after an acquire of its flag.
// Counters live at R52_BASE in the engine's existing counter buffer; the last
// block out resets them for the next replay. A bounded spin records a sticky
// status word instead of hanging. Licensed Sushi-derived MCG decode as in
// mcg_tensor.cu; see NOTICE.txt.
#define R52_BASE 64u
#define R52_TICKET (R52_BASE+0u)
#define R52_EXIT (R52_BASE+1u)
#define R52_SD (R52_BASE+2u)
#define R52_SREADY (R52_BASE+3u)
#define R52_STATUS (R52_BASE+4u)
#define R52_IDS (R52_BASE+5u)
#define R52_ACT (R52_BASE+6u)
#define R52_EXIT2 (R52_BASE+7u)
#define R52_GCNT (R52_BASE+8u)
#define R52_MREADY (R52_BASE+64u)
#define R52_FCNT (R52_BASE+120u)
#define R52_DCNT (R52_BASE+144u)
#define R52_END (R52_BASE+344u)

__device__ __forceinline__ void r52_release(unsigned* p) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(1u) : "memory");
}
__device__ __forceinline__ void r52_spin(unsigned* ctr,unsigned idx) {
  unsigned spins=0;
  while(mi_ld_acquire(ctr+idx)==0u) {
    __nanosleep(32);
    if(++spins>(1u<<22)) { atomicExch(ctr+R52_STATUS,1u+idx); break; }
  }
}
// mi_pair_values with explicit instruction selection: the mask-and-flip
// (x & 0x8fff8fff) ^ 0x3b603b60 as ONE lop3 (lut 0x6a = (a & b) ^ c) with the
// flip constant in a register, and the f16 pair packing as two prmt byte
// permutes (0x5410 = low halves, 0x7632 = high halves). Same bits as
// mi_pair_values: lo = (p0 & 0xffff) | (p1 << 16), hi = (p0 >> 16) | (p1 & 0xffff0000).
__device__ __forceinline__ unsigned r52_pair_values(unsigned window,unsigned pair,unsigned flip) {
  const unsigned offset=pair==0?11:pair==1?7:pair==2?4:0;
  const unsigned cw=window>>offset;
  const unsigned c0=(cw>>2)&0x7fffu,c1=cw&0x7fffu;
  unsigned p0=c0*0xcbac1fedu,p1=c1*0xcbac1fedu,lo,hi,result;
  asm("lop3.b32 %0, %0, 0x8fff8fff, %1, 0x6a;" : "+r"(p0) : "r"(flip));
  asm("lop3.b32 %0, %0, 0x8fff8fff, %1, 0x6a;" : "+r"(p1) : "r"(flip));
  asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(p0), "r"(p1));
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi) : "r"(p0), "r"(p1));
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(result) : "r"(lo), "r"(hi));
  return result;
}
// The flip constant kept in a register (opaque to constant folding).
__device__ __forceinline__ unsigned r52_flip() {
  unsigned f; asm volatile("mov.b32 %0, 0x3b603b60;" : "=r"(f)); return f;
}
// Weight pump: L2 prefetches (evict_last), one per 128-byte line, over this
// block's slice of a table of constant regions {u64 address, u32 lines,
// u32 cumulative lines}. Measured to land in L2 (unlike cp.async.cg); no
// registers, shared memory or completion wait needed.
__device__ __forceinline__ void r52_pump(const uint4* table,unsigned entries,unsigned total,
    unsigned share,unsigned shares,uint4* sink) {
  (void)sink;
  if(!table || !total) return;
  const unsigned per=(total+shares-1u)/shares,lo=share*per,hi=min(lo+per,total);
  unsigned e=0;
  uint4 ent=table[0];
  for(unsigned i=lo+threadIdx.x;i<hi;i+=blockDim.x) {
    while(e+1u<entries && i>=ent.w+ent.z) { e++; ent=table[e]; }
    const unsigned long long base=((unsigned long long)ent.y<<32)|ent.x;
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(base+(unsigned long long)(i-ent.w)*128ull));
  }
}
__device__ __forceinline__ void r52_pump_wait() {}
#ifndef R52_FASTDEC
#define R52_FASTDEC 1
#endif
#define R52_PAIR(window,p,flip) (R52_FASTDEC?r52_pair_values((window),(p),(flip)):mi_pair_values((window),(p)))
// Gate or up projection for M output tiles of one (slot, split), input prepared in-block.
template<unsigned M>
__device__ __forceinline__ void r52_gate_up(const unsigned short* x,const unsigned short* s,
    const unsigned* trellis,float* y,unsigned short* prepared,unsigned expert,unsigned slot,
    unsigned split,unsigned ot0,bool write_prepared,float* partial,unsigned short* xs,
    unsigned long long* stamp) {
  constexpr unsigned K=2560,N=640,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS;
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  const unsigned begin=split*PER,end=begin+PER;
  const unsigned* weights=trellis+(unsigned long long)expert*IT*NT*14u;
  unsigned wp[2][M],wl[2][M];
  #pragma unroll
  for(unsigned d=0;d<2;d++) {
    const unsigned tk=begin+warp+d*4u;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=words[word_prev];wl[d][m]=words[word_last];
    }
  }
  // moe_prepare_bf16 for this block's K range: five 128-wide pieces. Warp w
  // takes piece w; warp 0 also takes piece 4 and requests both pieces' inputs
  // before computing either.
  {
    const bool two=warp==0u;
    unsigned short xr[2][4],sr[2][4];
    #pragma unroll
    for(unsigned pc=0;pc<2;pc++) {
      const unsigned base=split*640u+(pc?4u:warp)*128u;
      #pragma unroll
      for(unsigned j=0;j<4;j++) {
        const bool live=pc==0u || two;
        xr[pc][j]=live?__ldcg(x+base+lane+j*32u):(unsigned short)0;
        sr[pc][j]=live?s[expert*2560u+base+lane+j*32u]:(unsigned short)0;
      }
    }
    #pragma unroll
    for(unsigned pc=0;pc<2;pc++) if(pc==0u || two) {
      const unsigned c=pc?4u:warp,base=split*640u+c*128u;
      MiFour g;
      #pragma unroll
      for(unsigned j=0;j<4;j++) g.v[j]=mi_load(xr[pc][j])*mi_hf(sr[pc][j]);
      g=mi_had(g,lane);
      #pragma unroll
      for(unsigned j=0;j<4;j++) {
        const unsigned short h=mi_fh(g.v[j]);
        xs[c*128u+lane+j*32u]=h;
        if(write_prepared) prepared[slot*2560u+base+lane+j*32u]=h;
      }
    }
  }
  __syncthreads();
  if(stamp && tid==0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); *stamp=t; }
  float acc[M][4];
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  for(unsigned tk0=begin+warp;tk0<end;tk0+=8u) {
    unsigned b0[2],b1[2],np[2][M],nl[2][M];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned tk=tk0+d*4u;
      const unsigned short* xx=xs+(tk-begin)*16u;
      const bool xin=col<1u;
      const unsigned x0v=xin?xx[row]:0u,x1v=xin?xx[row+1]:0u;
      const unsigned x8=xin?xx[row+8]:0u,x9=xin?xx[row+9]:0u;
      b0[d]=x0v|(x1v<<16);
      b1[d]=x8|(x9<<16);
      const unsigned tn=tk+8u;
      const bool live=tn<end;
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned* words=weights+((unsigned long long)tn*NT+ot0+m)*14u;
        np[d][m]=live?words[word_prev]:0u;
        nl[d][m]=live?words[word_last]:0u;
      }
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
        wp[d][m]=np[d][m];wl[d][m]=nl[d][m];
      }
    }
  }
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
    y[((unsigned long long)slot*SPLITS+split)*N+(ot0+m)*16+r]=sum;
  }
  __syncthreads();
}
// Down projection for M output tiles of one (slot, split): words first, then wait.
template<unsigned M>
__device__ __forceinline__ void r52_down(const unsigned short* middle,const unsigned* td,float* dpar,
    unsigned* ctr,unsigned expert,unsigned slot,unsigned split,unsigned ot0,float* partial,
    unsigned long long* stamp) {
  constexpr unsigned K=640,N=2560,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS;
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  const unsigned begin=split*PER,end=begin+PER;
  const unsigned* weights=td+(unsigned long long)expert*IT*NT*14u;
  // Warp w's K tiles are begin+w, begin+w+4, begin+w+8 (in this order), as in
  // ms_mma_tiles' two-deep loop.
  unsigned wp[3][M],wl[3][M];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool live=tk<end;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=live?words[word_prev]:0u;
      wl[d][m]=live?words[word_last]:0u;
    }
  }
  if(tid==0) {
    const unsigned c0=(split*160u)/128u,c1=(split*160u+159u)/128u;
    r52_spin(ctr,R52_MREADY+slot*5u+c0);
    if(c1!=c0) r52_spin(ctr,R52_MREADY+slot*5u+c1);
    __threadfence();
    if(stamp) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); *stamp=t; }
  }
  __syncthreads();
  float acc[M][4];
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  unsigned b0[3],b1[3];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool xin=tk<end && col<1u;
    const unsigned short* xx=middle+(unsigned long long)slot*K+tk*16u;
    const unsigned x0v=xin?__ldcg(xx+row):0u,x1v=xin?__ldcg(xx+row+1):0u;
    const unsigned x8=xin?__ldcg(xx+row+8):0u,x9=xin?__ldcg(xx+row+9):0u;
    b0[d]=x0v|(x1v<<16);
    b1[d]=x8|(x9<<16);
  }
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    if(tk<end) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
      }
    }
  }
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
    dpar[((unsigned long long)slot*SPLITS+split)*N+(ot0+m)*16+r]=sum;
  }
  __syncthreads();
}
// Down projection for 8 output tiles of one (slot, split) in two rounds of four:
// round 0's trellis words are requested before the wait, round 1's right after
// it (before round 0 is computed). Each tile's MMA sequence and cross-warp sum
// are ms_mma_tiles' exactly.
__device__ __forceinline__ void r52_down8(const unsigned short* middle,const unsigned* td,float* dpar,
    unsigned* ctr,unsigned expert,unsigned slot,unsigned split,unsigned ot0,float* partial,
    unsigned long long* stamp) {
  constexpr unsigned M=4,K=640,N=2560,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS;
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  const unsigned begin=split*PER,end=begin+PER;
  const unsigned* weights=td+(unsigned long long)expert*IT*NT*14u;
  unsigned wp[3][M],wl[3][M];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool live=tk<end;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=live?words[word_prev]:0u;
      wl[d][m]=live?words[word_last]:0u;
    }
  }
  if(tid==0) {
    const unsigned c0=(split*160u)/128u,c1=(split*160u+159u)/128u;
    r52_spin(ctr,R52_MREADY+slot*5u+c0);
    if(c1!=c0) r52_spin(ctr,R52_MREADY+slot*5u+c1);
    __threadfence();
    if(stamp) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); *stamp=t; }
  }
  __syncthreads();
  unsigned b0[3],b1[3];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool xin=tk<end && col<1u;
    const unsigned short* xx=middle+(unsigned long long)slot*K+tk*16u;
    const unsigned x0v=xin?__ldcg(xx+row):0u,x1v=xin?__ldcg(xx+row+1):0u;
    const unsigned x8=xin?__ldcg(xx+row+8):0u,x9=xin?__ldcg(xx+row+9):0u;
    b0[d]=x0v|(x1v<<16);
    b1[d]=x8|(x9<<16);
  }
  #pragma unroll
  for(unsigned round=0;round<2u;round++) {
    unsigned np[3][M],nl[3][M];
    if(round==0u) {
      #pragma unroll
      for(unsigned d=0;d<3;d++) {
        const unsigned tk=begin+warp+d*4u;
        const bool live=tk<end;
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned* words=weights+((unsigned long long)tk*NT+ot0+4u+m)*14u;
          np[d][m]=live?words[word_prev]:0u;
          nl[d][m]=live?words[word_last]:0u;
        }
      }
    }
    float acc[M][4];
    #pragma unroll
    for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
    #pragma unroll
    for(unsigned d=0;d<3;d++) {
      const unsigned tk=begin+warp+d*4u;
      if(tk<end) {
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
          unsigned a[4];
          #pragma unroll
          for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
              : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
        }
      }
    }
    if((lane&3u)==0u) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        partial[((round*4u+m)*4+warp)*16+col]=acc[m][0];
        partial[((round*4u+m)*4+warp)*16+col+8]=acc[m][2];
      }
    }
    if(round==0u) {
      #pragma unroll
      for(unsigned d=0;d<3;d++)
        #pragma unroll
        for(unsigned m=0;m<M;m++) { wp[d][m]=np[d][m];wl[d][m]=nl[d][m]; }
    }
  }
  __syncthreads();
  for(unsigned i=tid;i<8u*16u;i+=128) {
    const unsigned m=i/16,r=i%16;
    float sum=0;for(unsigned w=0;w<4;w++)sum+=partial[(m*4+w)*16+r];
    dpar[((unsigned long long)slot*SPLITS+split)*N+(ot0+m)*16+r]=sum;
  }
  __syncthreads();
}
// mf_mid with its expert scales already loaded (packed FP16 pairs), same arithmetic.
__device__ __forceinline__ void r52_mid_pre(const float* gpar,const float* upar,unsigned short* middle,
    unsigned slot,unsigned base,unsigned lane,const unsigned pvg[2],const unsigned pvu[2],const unsigned psd[2]) {
  constexpr unsigned SPLITS=4;
  float gr[SPLITS][4],ur[SPLITS][4];
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      const unsigned long long at=((unsigned long long)slot*SPLITS+s)*640+base+lane+j*32;
      gr[s][j]=__ldcg(gpar+at);ur[s][j]=__ldcg(upar+at);
    }
  const unsigned yb=slot*640+base+lane;
  MiFour g0={{0,0,0,0}},u0={{0,0,0,0}};
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) {g0.v[j]+=gr[s][j];u0.v[j]+=ur[s][j];}
  MiFour g=mi_had(g0,lane),uu=mi_had(u0,lane),vv;
  #pragma unroll
  for(unsigned j=0;j<4;j++) {
    const unsigned short vgj=(unsigned short)((j&1u)?(pvg[j>>1]>>16):(pvg[j>>1]&0xffffu));
    const unsigned short vuj=(unsigned short)((j&1u)?(pvu[j>>1]>>16):(pvu[j>>1]&0xffffu));
    const unsigned short sdj=(unsigned short)((j&1u)?(psd[j>>1]>>16):(psd[j>>1]&0xffffu));
    float a=g.v[j]*mi_hf(vgj),b=uu.v[j]*mi_hf(vuj);
    float ex=expf(-fabsf(a)),sig=a<0?ex/(1.0f+ex):1.0f/(1.0f+ex);
    vv.v[j]=a*sig*b*mi_hf(sdj);
  }
  vv=mi_had(vv,lane);
  #pragma unroll
  for(unsigned j=0;j<4;j++)middle[yb+j*32]=mi_fh(vv.v[j]);
}
__device__ __forceinline__ void r52_pack4(const unsigned short* p,unsigned lane,unsigned out[2]) {
  const unsigned a=p[lane],b=p[lane+32],c=p[lane+64],d=p[lane+96];
  out[0]=a|(b<<16);out[1]=c|(d<<16);
}
// mi_direct (affine4 load-ahead GEMV, original arithmetic) with its input read through L2.
__device__ __forceinline__ void r52_direct(const unsigned short* x,const unsigned* w,
    const unsigned short* scales,const unsigned short* biases,float* y,unsigned short* ybf,unsigned k,
    unsigned n,unsigned group,unsigned bx,unsigned warp2,unsigned lane) {
  const unsigned n0=bx*4+warp2*2;
  float acc[2]={0.f,0.f};
  const unsigned groups=k/group,begin=0,end=k;
  for(unsigned base=begin;base<end;base+=512u) {
    unsigned xw[2][4],packed[2][2];
    unsigned short sr[2][2],br[2][2];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned kb=base+d*256u,k0=kb+lane*8u;
      const bool ok=kb<end && k0+8u<=k;
      const uint4 q=ok?__ldcg(reinterpret_cast<const uint4*>(x+k0)):make_uint4(0u,0u,0u,0u);
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
      (void)sums;
      if(kb<end && k0<k) {
        #pragma unroll
        for(unsigned col=0;col<2;col++) if(n0+col<n) {
          const float s=mi_load(sr[d][col]),b=mi_load(br[d][col]);
          const unsigned p=packed[d][col];
          #pragma unroll
          for(unsigned i=0;i<8;i++) {
            const float decoded=fmaf(s,float((p>>(4*i))&15u),b);
            acc[col]=fmaf(vals[i],decoded,acc[col]);
          }
        }
      }
    }
  }
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    const float v=mi_warp_total(acc[col]);
    if(lane==0 && n0+col<n) { y[n0+col]=v; ybf[n0+col]=mi_store_rne(v); }
  }
}
// mf_shared_finish with the shared down result read through L2.
__device__ __forceinline__ void r52_shared_finish(const unsigned short* x,const unsigned short* scalar_gate,
    const unsigned short* dp_b,unsigned short* shared_out,float* sraw,unsigned short* slogits,
    unsigned short* ssig,int shared_arm,float* smem) {
  const unsigned t=threadIdx.x;
  unsigned short xr[20],wr[20],dr[20];
  #pragma unroll
  for(int i=0;i<20;i++) {
    const int k=(int)t+128*i;
    xr[i]=__ldcg(x+k);wr[i]=scalar_gate[k];dr[i]=__ldcg(dp_b+k);
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
// Stage 6 for one 128-wide output piece, experts in rounds of four (smem[512]);
// warp 0 accumulates experts 0..9 in order exactly as stage 6 does.
__device__ __forceinline__ void r52_finish(unsigned piece,const unsigned* ids,const float* scores,
    const float* dpar,const unsigned short* sv,const unsigned short* shared_out,float* rout,
    unsigned short* rout_b,unsigned short* added,float* smem) {
  constexpr unsigned E=10,SPLITS=4;
  const unsigned t=threadIdx.x,lane=t&31u,warp=t>>5;
  const unsigned base=piece*128u;
  MiFour sum={{0,0,0,0}};
  float sc[E];
  if(warp==0) {
    #pragma unroll
    for(unsigned e=0;e<E;e++) sc[e]=scores[e];
  }
  #pragma unroll
  for(unsigned round=0;round<3u;round++) {
    const unsigned e=round*4u+warp;
    if(e<E) {
      float dr[SPLITS][4];
      #pragma unroll
      for(unsigned s=0;s<SPLITS;s++)
        #pragma unroll
        for(unsigned j=0;j<4;j++) dr[s][j]=__ldcg(dpar+((unsigned long long)e*SPLITS+s)*2560+base+lane+j*32);
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
      for(unsigned j=0;j<4;j++) smem[warp*128+lane+j*32]=v.v[j]*mi_hf(svr[j]);
    }
    __syncthreads();
    if(warp==0) {
      #pragma unroll
      for(unsigned w=0;w<4u;w++) {
        const unsigned ee=round*4u+w;
        if(ee<E) {
          #pragma unroll
          for(unsigned j=0;j<4;j++) sum.v[j]+=smem[w*128+lane+j*32]*sc[ee];
        }
      }
    }
    __syncthreads();
  }
  if(warp==0) {
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      const unsigned at=base+lane+j*32;
      rout[at]=sum.v[j];
      const unsigned short rbits=mi_store_rne(sum.v[j]);
      rout_b[at]=rbits;
      added[at]=mi_store_cvt(mi_load(rbits)+mi_load(__ldcg(shared_out+at)));
    }
  }
}
__device__ __forceinline__ unsigned long long r52_now() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
// Stage 6, first half, for one (slot, 128-wide output piece), warp 0: sum the four
// split partials in order, Hadamard, scale by the expert's output scales; stored
// as FP32 so the final sum below is stage 6's arithmetic exactly.
__device__ __forceinline__ void r52_scale_piece(unsigned slot,unsigned piece,const unsigned* ids,
    const float* dpar,const unsigned short* sv,float* fin,unsigned lane) {
  constexpr unsigned SPLITS=4;
  const unsigned base=piece*128u;
  float dr[SPLITS][4];
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) dr[s][j]=__ldcg(dpar+((unsigned long long)slot*SPLITS+s)*2560+base+lane+j*32);
  const unsigned sb=ids[slot]*2560+base+lane;
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
  for(unsigned j=0;j<4;j++) fin[slot*2560u+base+lane+j*32]=v.v[j]*mi_hf(svr[j]);
}
// r52_scale_piece with the expert's output scales already loaded (packed pairs).
__device__ __forceinline__ void r52_scale_piece_pre(unsigned slot,unsigned piece,const float* dpar,
    float* fin,unsigned lane,const unsigned psv[2]) {
  constexpr unsigned SPLITS=4;
  const unsigned base=piece*128u;
  float dr[SPLITS][4];
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) dr[s][j]=__ldcg(dpar+((unsigned long long)slot*SPLITS+s)*2560+base+lane+j*32);
  MiFour acc={{0,0,0,0}};
  #pragma unroll
  for(unsigned s=0;s<SPLITS;s++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) acc.v[j]+=dr[s][j];
  MiFour v=mi_had(acc,lane);
  #pragma unroll
  for(unsigned j=0;j<4;j++) {
    const unsigned short svj=(unsigned short)((j&1u)?(psv[j>>1]>>16):(psv[j>>1]&0xffffu));
    fin[slot*2560u+base+lane+j*32]=v.v[j]*mi_hf(svj);
  }
}
// Stage 6, second half, for one output piece, warp 0: experts 0..9 in order.
__device__ __forceinline__ void r52_sum_piece(unsigned piece,const float* scores,const float* fin,
    const unsigned short* shared_out,float* rout,unsigned short* rout_b,unsigned short* added,unsigned lane) {
  constexpr unsigned E=10;
  const unsigned base=piece*128u;
  float sc[E],fv[E][4];
  #pragma unroll
  for(unsigned e=0;e<E;e++) {
    sc[e]=__ldcg(scores+e);
    #pragma unroll
    for(unsigned j=0;j<4;j++) fv[e][j]=__ldcg(fin+e*2560u+base+lane+j*32);
  }
  MiFour sum={{0,0,0,0}};
  #pragma unroll
  for(unsigned e=0;e<E;e++)
    #pragma unroll
    for(unsigned j=0;j<4;j++) sum.v[j]+=fv[e][j]*sc[e];
  #pragma unroll
  for(unsigned j=0;j<4;j++) {
    const unsigned at=base+lane+j*32;
    rout[at]=sum.v[j];
    const unsigned short rbits=mi_store_rne(sum.v[j]);
    rout_b[at]=rbits;
    added[at]=mi_store_cvt(mi_load(rbits)+mi_load(__ldcg(shared_out+at)));
  }
}
#define MF_ARGS x,rw,rraw,rlogits,ids,scores,select_arm,gw,gs,gb,ggroup,gpart,gu_f,gu_b,act,dw,ds,db, \
    dgroup,dp_f,dp_b,scalar_gate,shared_out,sraw,slogits,ssig,shared_arm,sg,su,pg,pu,tg,tu,gpar,upar, \
    vg,vu,sd,middle,td,dpar,sv,rout,rout_b,added,ctr
// trace (diagnostic entries only): per ticket six words: start, after prepare or
// wait, after the block's main work, end, SM id, kind (0 shared, 1 gate/up, 2 down)
// plus 16 * tail flag. Null in the timed entries, where it folds away.
template<unsigned MG,unsigned MD,bool PRE,bool OVL=false,bool SRED=false,unsigned SR=1,bool LOOP=false,bool NOD=false>
__device__ __forceinline__ void r52_back(MF_PARAMS,float* fin,unsigned long long* trace,
    const uint4* ptab=nullptr,unsigned pent=0,unsigned ptot=0,unsigned pshares=0) {
  // SR: shared-down vblock pairs per block (1 = 320 blocks, 2 = 160 blocks).
  // NOD: no down items here (the separate early-launched down kernel does them
  // and resets the counters).
  constexpr unsigned E=10,NS=320u/(SR),NG=1600u*2u/(MG),ND=NOD?0u:1600u*4u/(MD);
  // Ticket order: gate/up first (all start in the first wave), shared down,
  // then down. Every wait is on a lower-numbered item.
  constexpr unsigned SB=NG,DB=NG+NS,T=NS+NG+ND;
  constexpr unsigned NTG=40u/(MG),NTD=160u/(MD);
  __shared__ __align__(16) float smem[512];
  __shared__ __align__(16) unsigned short xs[640];
  __shared__ unsigned s_ticket,flag,s_next;
  __shared__ __align__(16) uint4 pump_sink[4];
  const unsigned t=threadIdx.x,lane=t&31u,warp=t>>5;
  if(t==0) s_ticket=atomicAdd(ctr+R52_TICKET,1u);
  __syncthreads();
  // LOOP (persistent): after each item the block takes the next ticket, which it
  // requests when the item starts; still every wait is on a lower ticket.
  for(unsigned q=s_ticket;q<T;) {
  if(LOOP && t==0) s_next=atomicAdd(ctr+R52_TICKET,1u);
  unsigned long long* tr=(trace && q<T)?trace+(unsigned long long)q*6u:nullptr;
  if(tr && t==0) tr[0]=r52_now();
  unsigned kind=0,tail=0;
  // Overlap variant: launched early beside F0; every block first acquires the
  // flag of the F0 tail whose output it reads (routes, or shared activation).
  if(OVL) {
    if(t==0) { r52_spin(ctr,(q>=SB && q<DB)?R52_ACT:R52_IDS); __threadfence(); }
    __syncthreads();
  }
  if(q>=SB && q<DB) {
    if(SRED) {
      // F0's shared tail moved here: the first shared item reduces the split
      // partials and activates (same order and rounding), the rest wait for it.
      if(q==SB) {
        unsigned short bits[10];
        #pragma unroll
        for(unsigned i=0;i<10;i++) {
          const unsigned at=t+128u*i;
          float value=0.f;
          for(unsigned p=0;p<4u;p++) value+=__ldcg(gpart+p*1280u+at);
          gu_f[at]=value;bits[i]=mi_store_rne(value);gu_b[at]=bits[i];
        }
        #pragma unroll
        for(unsigned i=0;i<5;i++) {
          const unsigned j=t+128u*i;
          const float g=mi_load(bits[i]),uv=mi_load(bits[i+5]);
          const float a=g*mi_shared_sig(g,shared_arm);
          act[j]=mi_store_rne((shared_arm?a:mi_rb_rne(a))*uv);
        }
        __threadfence(); __syncthreads();
        if(t==0) r52_release(ctr+R52_ACT);
      } else {
        if(t==0) { r52_spin(ctr,R52_ACT); __threadfence(); }
        __syncthreads();
      }
    }
    #pragma unroll
    for(unsigned rr=0;rr<SR;rr++) {
      const unsigned v=2u*SR*(q-SB)+2u*rr+(warp>>1);
      if(OVL || SRED) r52_direct(act,dw,ds,db,dp_f,dp_b,640u,2560u,dgroup,v,warp&1,lane);
      else mi_direct(act,dw,ds,db,dp_f,dp_b,640u,2560u,dgroup,v,warp&1,lane);
    }
    if(tr && t==0) { tr[1]=r52_now(); tr[2]=tr[1]; }
    if(mf_done(ctr,R52_SD,NS,&flag)) {
      tail=1;
      r52_shared_finish(x,scalar_gate,dp_b,shared_out,sraw,slogits,ssig,shared_arm,smem);
      __threadfence(); __syncthreads();
      if(t==0) r52_release(ctr+R52_SREADY);
    }
  } else if(q<SB) {
    kind=1;
    const unsigned r=q,slot=r/(NG/E),w=r%(NG/E);
    const unsigned split=w/(2u*NTG),proj=(w/NTG)%2u,g=w%NTG;
    if(ptab) r52_pump(ptab,pent,ptot,q,pshares,pump_sink);
    const unsigned expert=OVL?__ldcg(ids+slot):ids[slot];
    const unsigned piece=(g*(MG))/8u;
    unsigned pvg[2]={0u,0u},pvu[2]={0u,0u},psd[2]={0u,0u};
    if(PRE && warp==0) {
      const unsigned sb=expert*640u+piece*128u;
      r52_pack4(vg+sb,lane,pvg);r52_pack4(vu+sb,lane,pvu);r52_pack4(sd+sb,lane,psd);
    }
    r52_gate_up<MG>(x,proj?su:sg,proj?tu:tg,proj?upar:gpar,proj?pu:pg,expert,slot,split,g*(MG),g==0u,smem,xs,tr?tr+1:nullptr);
    if(tr && t==0) tr[2]=r52_now();
    if(mf_done(ctr,R52_GCNT+slot*5u+piece,2u*4u*(8u/(MG)),&flag)) {
      tail=1;
      if(warp==0) {
        if(!PRE) {
          const unsigned sb=expert*640u+piece*128u;
          r52_pack4(vg+sb,lane,pvg);r52_pack4(vu+sb,lane,pvu);r52_pack4(sd+sb,lane,psd);
        }
        r52_mid_pre(gpar,upar,middle,slot,piece*128u,lane,pvg,pvu,psd);
      }
      __threadfence(); __syncthreads();
      if(t==0) r52_release(ctr+R52_MREADY+slot*5u+piece);
    }
  } else if(q<T) {
    kind=2;
    const unsigned r=q-DB,slot=r/(ND/E),w=r%(ND/E);
    const unsigned split=w/NTD,g=w%NTD;
    const unsigned expert=OVL?__ldcg(ids+slot):ids[slot];
    const unsigned piece=(g*(MD))/8u;
    unsigned psv[2]={0u,0u};
    if(PRE && warp==0) r52_pack4(sv+expert*2560u+piece*128u,lane,psv);
    if constexpr(MD==8) r52_down8(middle,td,dpar,ctr,expert,slot,split,g*8u,smem,tr?tr+1:nullptr);
    else r52_down<MD>(middle,td,dpar,ctr,expert,slot,split,g*(MD),smem,tr?tr+1:nullptr);
    if(tr && t==0) tr[2]=r52_now();
    if(mf_done(ctr,R52_DCNT+slot*20u+piece,4u*(8u/(MD)),&flag)) {
      tail=1;
      if(warp==0) {
        if(!PRE) r52_pack4(sv+expert*2560u+piece*128u,lane,psv);
        r52_scale_piece_pre(slot,piece,dpar,fin,lane,psv);
      }
      if(mf_done(ctr,R52_FCNT+piece,E,&flag)) {
        tail=2;
        if(t==0) { r52_spin(ctr,R52_SREADY); __threadfence(); }
        __syncthreads();
        if(warp==0) r52_sum_piece(piece,scores,fin,shared_out,rout,rout_b,added,lane);
      }
    }
  }
  __syncthreads();
  if(t==0 && tr) {
    unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
    tr[3]=r52_now(); tr[4]=sm; tr[5]=kind+16u*tail;
  }
  if(!LOOP) break;
  q=s_next;
  __syncthreads();
  }
  if(ptab) r52_pump_wait();
  if(t==0 && !NOD) {
    __threadfence();
    if(atomicAdd(ctr+R52_EXIT,1u)==(LOOP?gridDim.x:T)-1u) {
      for(unsigned i=R52_BASE;i<R52_END;i++) if(i!=R52_STATUS) ctr[i]=0u;
      __threadfence();
    }
  }
}
#define R52_BACK(NAME,TNAME,MG,MD,PRE,MINB) \
extern "C" __global__ void __launch_bounds__(128,MINB) NAME(MF_PARAMS,float* fin) { r52_back<MG,MD,PRE>(MF_ARGS,fin,nullptr); } \
extern "C" __global__ void __launch_bounds__(128,MINB) TNAME(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<MG,MD,PRE>(MF_ARGS,fin,trace); }
R52_BACK(moe_r52_back_g4_d4,moe_r52_trace_g4_d4,4,4,false,6)
R52_BACK(moe_r52_back_g4_d8,moe_r52_trace_g4_d8,4,8,false,5)
R52_BACK(moe_r52_back_g4_d4p,moe_r52_trace_g4_d4p,4,4,true,6)
R52_BACK(moe_r52_back_g4_d8p,moe_r52_trace_g4_d8p,4,8,true,5)

// mf_router_select with optional interior stamps (diagnostic; same arithmetic).
__device__ __forceinline__ void r52_select(const float* raw,unsigned short* logits,unsigned* ids,
    float* scores,int arm,unsigned lane,unsigned long long* st) {
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  float rawv[16];
  #pragma unroll
  for(int j=0;j<16;j++) rawv[j]=__ldcg(raw+j*32+lane);
  if(st && lane==0) { float z=0; for(int j=0;j<16;j++) z+=rawv[j]; st[0]=r52_now()+(z==1.2345f?1:0); }
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=rawv[j];unsigned short h=mi_store_rne(v);logits[e]=h;
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
  if(st && lane==0) st[1]=r52_now()+(norm==1.2345f?1:0);
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
  if(st && lane==0) st[2]=r52_now()+(selected==12345u?1:0);
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
  if(st && lane==0) st[3]=r52_now();
}
// F0 (R39 wait-equipped clone, unchanged arithmetic) whose two tails also release
// flags for the overlapped R52 back half: routes (ids, scores) and shared activation.
// The traced entry also records, per block: start (after the wait), main work
// done, tail done, end, SM id, kind (3 router, 4 shared) + 16 * tail.
__device__ __forceinline__ void r52_f0(MF_PARAMS,unsigned* rc,unsigned long long* trace) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,SPLITS=4;
  __shared__ unsigned flag;
  unsigned long long* tr=trace?trace+16320u+pb*6u:nullptr;
  if(tr && t==0) tr[0]=r52_now();
  const bool router=pb<128u;
  if(router) mi_router_qmv(x,rw,rraw,pb*4+warp,lane);
  else {
    const unsigned v=2*(pb-128)+(warp>>1);
    mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
  }
  if(tr && t==0) { tr[1]=r52_now(); tr[2]=tr[1]; }
  const bool last=mf_done(ctr,router?0u:1u,router?128u:640u,&flag);
  if(last) {
    if(router) {
      if(warp==0) {
        if(trace) r52_select(rraw,rlogits,ids,scores,select_arm,lane,trace+16320u+768u*6u);
        else mf_router_select(rraw,rlogits,ids,scores,select_arm,lane);
      }
      __threadfence(); __syncthreads();
      if(t==0) r52_release(rc+R52_IDS);
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
      __threadfence(); __syncthreads();
      if(t==0) r52_release(rc+R52_ACT);
    }
    if(t==0) ctr[router?0u:1u]=0u;
    if(tr && t==0) tr[2]=r52_now();
  }
  if(tr && t==0) {
    unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
    tr[3]=r52_now(); tr[4]=sm; tr[5]=(router?3u:4u)+(last?16u:0u);
  }
}
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0(MF_PARAMS,unsigned* rc) { r52_f0(MF_ARGS,rc,nullptr); }
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0t(MF_PARAMS,unsigned* rc,unsigned long long* trace) { r52_f0(MF_ARGS,rc,trace); }
// Top-10 selection with the same total order as mf_router_select (key
// descending, then lower expert id): each lane sorts its 16 candidates once
// (Batcher network, verified by the 0-1 principle), then each round takes the
// warp maximum of the lanes' heads with two redux operations. -0 is folded to
// +0 first because the float comparison treats them as equal. Prologue and
// epilogue (rounding, validity, softmax norm, weights) are unchanged.
__device__ __forceinline__ unsigned r52_ord(float v) {
  const unsigned u=__float_as_uint(v==0.f?0.f:v);
  return (u&0x80000000u)?~u:(u|0x80000000u);
}
__device__ __forceinline__ float r52_unord(unsigned o) {
  return __uint_as_float((o&0x80000000u)?(o&0x7fffffffu):~o);
}
#define R52_CX(i,j) do { const unsigned long long a_=k[i],b_=k[j]; k[i]=a_>b_?a_:b_; k[j]=a_>b_?b_:a_; } while(0)
__device__ __forceinline__ void r52_select_fast(const float* raw,unsigned short* logits,unsigned* ids,
    float* scores,int arm,unsigned lane,unsigned long long* st) {
  float key[16],mx=-__int_as_float(0x7f800000);
  bool invalid=false;
  float rawv[16];
  #pragma unroll
  for(int j=0;j<16;j++) rawv[j]=__ldcg(raw+j*32+lane);
  #pragma unroll
  for(int j=0;j<16;j++) {
    int e=j*32+lane;float v=rawv[j];unsigned short h=mi_store_rne(v);logits[e]=h;
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
  if(st && lane==0) st[1]=r52_now()+(norm==1.2345f?1:0);
  unsigned long long k[16];
  #pragma unroll
  for(int j=0;j<16;j++) k[j]=((unsigned long long)r52_ord(key[j])<<32)|(0xffffffffu-(unsigned)(j*32+lane));
  R52_CX(0,1);
  R52_CX(2,3);
  R52_CX(0,2);
  R52_CX(1,3);
  R52_CX(1,2);
  R52_CX(4,5);
  R52_CX(6,7);
  R52_CX(4,6);
  R52_CX(5,7);
  R52_CX(5,6);
  R52_CX(0,4);
  R52_CX(2,6);
  R52_CX(2,4);
  R52_CX(1,5);
  R52_CX(3,7);
  R52_CX(3,5);
  R52_CX(1,2);
  R52_CX(3,4);
  R52_CX(5,6);
  R52_CX(8,9);
  R52_CX(10,11);
  R52_CX(8,10);
  R52_CX(9,11);
  R52_CX(9,10);
  R52_CX(12,13);
  R52_CX(14,15);
  R52_CX(12,14);
  R52_CX(13,15);
  R52_CX(13,14);
  R52_CX(8,12);
  R52_CX(10,14);
  R52_CX(10,12);
  R52_CX(9,13);
  R52_CX(11,15);
  R52_CX(11,13);
  R52_CX(9,10);
  R52_CX(11,12);
  R52_CX(13,14);
  R52_CX(0,8);
  R52_CX(4,12);
  R52_CX(4,8);
  R52_CX(2,10);
  R52_CX(6,14);
  R52_CX(6,10);
  R52_CX(2,4);
  R52_CX(6,8);
  R52_CX(10,12);
  R52_CX(1,9);
  R52_CX(5,13);
  R52_CX(5,9);
  R52_CX(3,11);
  R52_CX(7,15);
  R52_CX(7,11);
  R52_CX(3,5);
  R52_CX(7,9);
  R52_CX(11,13);
  R52_CX(1,2);
  R52_CX(3,4);
  R52_CX(5,6);
  R52_CX(7,8);
  R52_CX(9,10);
  R52_CX(11,12);
  R52_CX(13,14);
  unsigned selected=0;
  float selected_logit=0;
  #pragma unroll
  for(int rank=0;rank<10;rank++) {
    const unsigned hk=(unsigned)(k[0]>>32),hid=0xffffffffu-(unsigned)k[0];
    const unsigned m=__reduce_max_sync(0xffffffffu,hk);
    const unsigned win=__reduce_min_sync(0xffffffffu,hk==m?hid:0xffffffffu);
    if((int)lane==rank){selected=win;selected_logit=r52_unord(m);}
    if(hid==win) {
      #pragma unroll
      for(int i=0;i<15;i++) k[i]=k[i+1];
      k[15]=0ull;
    }
  }
  if(st && lane==0) st[2]=r52_now()+(selected==12345u?1:0);
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
  if(st && lane==0) st[3]=r52_now();
}
// F0x: F0 with a one-trip router projection (the input vector in shared memory,
// all 80 weight values requested first; mi_router_qmv's FMA order and shuffle
// reduction), the fast selection above, and NO shared tail: the shared partials
// are reduced and activated by the back half's first shared-down item instead
// (identical arithmetic, off the critical path).
template<bool STAIL>
__device__ __forceinline__ void r52_f0x(MF_PARAMS,unsigned* rc,unsigned long long* trace) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned pb=blockIdx.x,t=threadIdx.x,lane=t&31u,warp=t>>5;
  constexpr unsigned H=2560,I=640,SPLITS=4;
  __shared__ unsigned flag;
  __shared__ __align__(16) unsigned short sxv[H];
  unsigned long long* tr=trace?trace+16320u+pb*6u:nullptr;
  if(tr && t==0) tr[0]=r52_now();
  const bool router=pb<128u;
  bool last=false;
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
    if(tr && t==0) { tr[1]=r52_now(); tr[2]=tr[1]; }
    last=mf_done(ctr,0u,128u,&flag);
    if(last) {
      if(warp==0) r52_select_fast(rraw,rlogits,ids,scores,select_arm,lane,trace?trace+16320u+768u*6u:nullptr);
      __threadfence(); __syncthreads();
      if(t==0) { r52_release(rc+R52_IDS); ctr[0]=0u; }
      if(tr && t==0) tr[2]=r52_now();
    }
  } else {
    const unsigned v=2*(pb-128)+(warp>>1);
    mi_partial(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
    if(tr && t==0) { tr[1]=r52_now(); tr[2]=tr[1]; }
    if(STAIL) {
      // F0's shared tail, unchanged: reduce the split partials, then activate.
      last=mf_done(ctr,1u,640u,&flag);
      if(last) {
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
        __threadfence(); __syncthreads();
        if(t==0) { r52_release(rc+R52_ACT); ctr[1]=0u; }
        if(tr && t==0) tr[2]=r52_now();
      }
    }
  }
  if(tr && t==0) {
    unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
    tr[3]=r52_now(); tr[4]=sm; tr[5]=(router?3u:4u)+(last?16u:0u);
  }
}
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0x(MF_PARAMS,unsigned* rc) { r52_f0x<false>(MF_ARGS,rc,nullptr); }
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0xt(MF_PARAMS,unsigned* rc,unsigned long long* trace) { r52_f0x<false>(MF_ARGS,rc,trace); }
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0y(MF_PARAMS,unsigned* rc) { r52_f0x<true>(MF_ARGS,rc,nullptr); }
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0yt(MF_PARAMS,unsigned* rc,unsigned long long* trace) { r52_f0x<true>(MF_ARGS,rc,trace); }
// Overlapped back half: early-launch eligible (r39pdl_ prefix) and deliberately
// WITHOUT griddepcontrol.wait; ordering comes only from the F0 flags above.
extern "C" __global__ void __launch_bounds__(128,6) r39pdl_moe_r52o_g4_d4(MF_PARAMS,float* fin) { r52_back<4,4,false,true>(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,6) r39pdl_moe_r52ot_g4_d4(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<4,4,false,true>(MF_ARGS,fin,trace); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_d4s(MF_PARAMS,float* fin) { r52_back<4,4,false,false,true>(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_trace_g4_d4s(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<4,4,false,false,true>(MF_ARGS,fin,trace); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_d4t(MF_PARAMS,float* fin) { r52_back<4,4,false,false,true,2>(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_trace_g4_d4t(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<4,4,false,false,true,2>(MF_ARGS,fin,trace); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_d4u(MF_PARAMS,float* fin) { r52_back<4,4,false,false,true,2,true>(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_trace_g4_d4u(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<4,4,false,false,true,2,true>(MF_ARGS,fin,trace); }

// ---- R52 split down kernel (early-launch eligible, NO griddepcontrol.wait) ----
// One block per (slot, 128-wide output piece): all four K splits of the down
// projection for that piece's eight output tiles. Each split's per-tile MMA
// sequence and cross-warp sum are ms_mma_tiles' (warp w: k tiles begin+w,
// begin+w+4, begin+w+8 in order); the split sums are written to dpar as before
// and also accumulated in split order for stage 6's first half (Hadamard,
// output scales) done in-block; the last of the ten slots of a piece does the
// second half (weighted sum + shared add). The next split's trellis words load
// while the current split computes. It waits only on flags of the gate/up
// launch (mid pieces, shared output) and reads their data through L2.
// The last block out resets every R52 counter of this engine.
__device__ __forceinline__ void r52_down_piece(MF_PARAMS,float* fin,unsigned long long* trace) {
  constexpr unsigned E=10,K=640,N=2560,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS,M=8;
  __shared__ __align__(16) float part[M*4*16];
  __shared__ __align__(16) float sacc[128];
  __shared__ unsigned flag;
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned b=blockIdx.x,slot=b/20u,piece=b%20u,ot0=piece*8u;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  unsigned long long* tr=trace?trace+21120u+b*6u:nullptr;
  if(tr && tid==0) tr[0]=r52_now();
  const unsigned expert=__ldcg(ids+slot);
  const unsigned* weights=td+(unsigned long long)expert*IT*NT*14u;
  unsigned psv[2]={0u,0u};
  if(warp==0) r52_pack4(sv+expert*2560u+piece*128u,lane,psv);
  unsigned wp[3][M],wl[3][M];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=warp+d*4u;
    const bool live=tk<PER;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=live?words[word_prev]:0u;
      wl[d][m]=live?words[word_last]:0u;
    }
  }
  if(tid==0) {
    for(unsigned c=0;c<5u;c++) r52_spin(ctr,R52_MREADY+slot*5u+c);
    __threadfence();
  }
  __syncthreads();
  if(tr && tid==0) tr[1]=r52_now();
  #pragma unroll 1
  for(unsigned split=0;split<SPLITS;split++) {
    const unsigned begin=split*PER,end=begin+PER;
    unsigned np[3][M],nl[3][M];
    if(split+1u<SPLITS) {
      #pragma unroll
      for(unsigned d=0;d<3;d++) {
        const unsigned tk=begin+PER+warp+d*4u;
        const bool live=tk<end+PER;
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
          np[d][m]=live?words[word_prev]:0u;
          nl[d][m]=live?words[word_last]:0u;
        }
      }
    }
    unsigned b0[3],b1[3];
    #pragma unroll
    for(unsigned d=0;d<3;d++) {
      const unsigned tk=begin+warp+d*4u;
      const bool xin=tk<end && col<1u;
      const unsigned short* xx=middle+(unsigned long long)slot*K+tk*16u;
      const unsigned x0v=xin?__ldcg(xx+row):0u,x1v=xin?__ldcg(xx+row+1):0u;
      const unsigned x8=xin?__ldcg(xx+row+8):0u,x9=xin?__ldcg(xx+row+9):0u;
      b0[d]=x0v|(x1v<<16);
      b1[d]=x8|(x9<<16);
    }
    float acc[M][4];
    #pragma unroll
    for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
    #pragma unroll
    for(unsigned d=0;d<3;d++) {
      const unsigned tk=begin+warp+d*4u;
      if(tk<end) {
        #pragma unroll
        for(unsigned m=0;m<M;m++) {
          const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
          unsigned a[4];
          #pragma unroll
          for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
              : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
        }
      }
    }
    if((lane&3u)==0u) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        part[(m*4+warp)*16+col]=acc[m][0];
        part[(m*4+warp)*16+col+8]=acc[m][2];
      }
    }
    __syncthreads();
    {
      const unsigned i=tid,m=i/16,r=i%16;
      float sum=0;for(unsigned w=0;w<4;w++)sum+=part[(m*4+w)*16+r];
      dpar[((unsigned long long)slot*SPLITS+split)*N+(ot0+m)*16+r]=sum;
      sacc[i]=split==0u?0.f+sum:sacc[i]+sum;
    }
    __syncthreads();
    if(split+1u<SPLITS) {
      #pragma unroll
      for(unsigned d=0;d<3;d++)
        #pragma unroll
        for(unsigned m=0;m<M;m++) { wp[d][m]=np[d][m];wl[d][m]=nl[d][m]; }
    }
  }
  if(tr && tid==0) tr[2]=r52_now();
  unsigned tail=0;
  if(warp==0) {
    MiFour a4;
    #pragma unroll
    for(unsigned j=0;j<4;j++) a4.v[j]=sacc[lane+j*32];
    MiFour v=mi_had(a4,lane);
    #pragma unroll
    for(unsigned j=0;j<4;j++) {
      const unsigned short svj=(unsigned short)((j&1u)?(psv[j>>1]>>16):(psv[j>>1]&0xffffu));
      fin[slot*2560u+piece*128u+lane+j*32]=v.v[j]*mi_hf(svj);
    }
  }
  if(mf_done(ctr,R52_FCNT+piece,E,&flag)) {
    tail=1;
    if(tid==0) { r52_spin(ctr,R52_SREADY); __threadfence(); }
    __syncthreads();
    if(warp==0) r52_sum_piece(piece,scores,fin,shared_out,rout,rout_b,added,lane);
  }
  __syncthreads();
  if(tid==0) {
    if(tr) { unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); tr[3]=r52_now(); tr[4]=sm; tr[5]=5u+16u*tail; }
    __threadfence();
    if(atomicAdd(ctr+R52_EXIT2,1u)==gridDim.x-1u) {
      for(unsigned i=R52_BASE;i<R52_END;i++) if(i!=R52_STATUS) ctr[i]=0u;
      __threadfence();
    }
  }
}
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_v(MF_PARAMS,float* fin) { r52_back<4,4,false,false,true,2,false,true>(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_trace_g4_v(MF_PARAMS,float* fin,unsigned long long* trace) { r52_back<4,4,false,false,true,2,false,true>(MF_ARGS,fin,trace); }
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52d(MF_PARAMS,float* fin) { r52_down_piece(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52dt(MF_PARAMS,float* fin,unsigned long long* trace) { r52_down_piece(MF_ARGS,fin,trace); }

// ---- R52 split down kernel, one (slot, piece, split) per block ----
// Same arithmetic as ms_mma_tiles for 8 output tiles of one split, all of the
// block's trellis words requested before it waits on the two mid pieces its K
// range reads. The last of a piece's four splits does stage 6's first half
// (r52_scale_piece_pre, split order), the last of a piece's ten slots the second.
__device__ __forceinline__ void r52_down_split(MF_PARAMS,float* fin,unsigned long long* trace,
    const uint4* ptab=nullptr,unsigned pent=0,unsigned ptot=0,unsigned pshare0=0,unsigned pshares=0) {
  constexpr unsigned E=10,K=640,N=2560,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS,M=8;
  __shared__ __align__(16) float part[M*4*16];
  __shared__ unsigned flag;
  __shared__ __align__(16) uint4 pump_sink[4];
  if(ptab) r52_pump(ptab,pent,ptot,pshare0+blockIdx.x,pshares,pump_sink);
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned b=blockIdx.x,slot=b/80u,piece=(b%80u)/4u,split=b%4u,ot0=piece*8u;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  const unsigned begin=split*PER,end=begin+PER;
  unsigned long long* tr=trace?trace+21120u+b*6u:nullptr;
  if(tr && tid==0) tr[0]=r52_now();
  const unsigned expert=__ldcg(ids+slot);
  const unsigned* weights=td+(unsigned long long)expert*IT*NT*14u;
  unsigned wp[3][M],wl[3][M];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool live=tk<end;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=live?words[word_prev]:0u;
      wl[d][m]=live?words[word_last]:0u;
    }
  }
  unsigned psv[2]={0u,0u};
  if(warp==0) r52_pack4(sv+expert*2560u+piece*128u,lane,psv);
  if(tid==0) {
    const unsigned c0=(split*160u)/128u,c1=(split*160u+159u)/128u;
    r52_spin(ctr,R52_MREADY+slot*5u+c0);
    if(c1!=c0) r52_spin(ctr,R52_MREADY+slot*5u+c1);
    __threadfence();
  }
  __syncthreads();
  if(tr && tid==0) tr[1]=r52_now();
  unsigned b0[3],b1[3];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    const bool xin=tk<end && col<1u;
    const unsigned short* xx=middle+(unsigned long long)slot*K+tk*16u;
    const unsigned x0v=xin?__ldcg(xx+row):0u,x1v=xin?__ldcg(xx+row+1):0u;
    const unsigned x8=xin?__ldcg(xx+row+8):0u,x9=xin?__ldcg(xx+row+9):0u;
    b0[d]=x0v|(x1v<<16);
    b1[d]=x8|(x9<<16);
  }
  float acc[M][4];
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    if(tk<end) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
      }
    }
  }
  if((lane&3u)==0u) {
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      part[(m*4+warp)*16+col]=acc[m][0];
      part[(m*4+warp)*16+col+8]=acc[m][2];
    }
  }
  __syncthreads();
  {
    const unsigned i=tid,m=i/16,r=i%16;
    float sum=0;for(unsigned w=0;w<4;w++)sum+=part[(m*4+w)*16+r];
    dpar[((unsigned long long)slot*SPLITS+split)*N+(ot0+m)*16+r]=sum;
  }
  if(tr && tid==0) tr[2]=r52_now();
  unsigned tail=0;
  if(mf_done(ctr,R52_DCNT+slot*20u+piece,SPLITS,&flag)) {
    tail=1;
    if(warp==0) r52_scale_piece_pre(slot,piece,dpar,fin,lane,psv);
    if(mf_done(ctr,R52_FCNT+piece,E,&flag)) {
      tail=2;
      if(tid==0) { r52_spin(ctr,R52_SREADY); __threadfence(); }
      __syncthreads();
      if(warp==0) r52_sum_piece(piece,scores,fin,shared_out,rout,rout_b,added,lane);
    }
  }
  __syncthreads();
  if(tid==0) {
    if(tr) { unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); tr[3]=r52_now(); tr[4]=sm; tr[5]=5u+16u*tail; }
    __threadfence();
    if(atomicAdd(ctr+R52_EXIT2,1u)==gridDim.x-1u) {
      for(unsigned i=R52_BASE;i<R52_END;i++) if(i!=R52_STATUS) ctr[i]=0u;
      __threadfence();
    }
  }
  if(ptab) r52_pump_wait();
}
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52ds(MF_PARAMS,float* fin) { r52_down_split(MF_ARGS,fin,nullptr); }
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52dst(MF_PARAMS,float* fin,unsigned long long* trace) { r52_down_split(MF_ARGS,fin,trace); }
// Pumping entries (next layer's first projection weights warmed into L2): the 800
// gate/up items of the first launch take shares 0..799, the 800 down blocks 800..1599.
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_vp(MF_PARAMS,float* fin,const uint4* ptab,unsigned pent,unsigned ptot) {
  r52_back<4,4,false,false,true,2,false,true>(MF_ARGS,fin,nullptr,ptab,pent,ptot,1600u);
}
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52dsp(MF_PARAMS,float* fin,const uint4* ptab,unsigned pent,unsigned ptot) {
  r52_down_split(MF_ARGS,fin,nullptr,ptab,pent,ptot,800u,1600u);
}
// Diagnostic: warm a region into L2 before a launch, by mode 0 = cp.async.cg into a
// per-warp shared sink (waited), 1 = ld.global.cg vector loads folded into a value
// stored only if it equals an impossible pattern, 2 = nothing (control).
extern "C" __global__ void __launch_bounds__(128) r52_pump_probe(const uint4* src,unsigned n16,unsigned mode,unsigned* sink) {
  __shared__ __align__(16) uint4 s_sink[4];
  unsigned acc=0;
  for(unsigned i=blockIdx.x*128u+threadIdx.x;i<n16;i+=gridDim.x*128u) {
    if(mode==0u) {
      const unsigned slot=(unsigned)__cvta_generic_to_shared(&s_sink[threadIdx.x>>5]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(slot), "l"(src+i) : "memory");
    } else if(mode==1u) {
      const uint4 v=__ldcg(src+i); acc^=v.x^v.y^v.z^v.w;
    } else if(mode==3u) {
      unsigned long long pol;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      const unsigned slot=(unsigned)__cvta_generic_to_shared(&s_sink[threadIdx.x>>5]);
      asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" :: "r"(slot), "l"(src+i), "l"(pol) : "memory");
    } else if(mode==4u) {
      const unsigned slot=(unsigned)__cvta_generic_to_shared(&s_sink[threadIdx.x>>5]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(slot), "l"(src+i) : "memory");
    } else if(mode==5u) {
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(src+i));
    } else if(mode>=7u && mode<=9u) {
      // one prefetch per 32 (7), 64 (8) or 128 (9) bytes
      const unsigned stride=2u<<(mode-7u);
      if(i%stride==0u) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(src+i));
    } else if(mode==6u) {
      unsigned long long pol;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      uint4 v;
      asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(src+i), "l"(pol));
      acc^=v.x^v.y^v.z^v.w;
    }
  }
  if(mode==0u || mode==3u || mode==4u) { asm volatile("cp.async.commit_group;" ::: "memory"); asm volatile("cp.async.wait_all;" ::: "memory"); }
  if(acc==0x9e3779b9u && sink) sink[0]=acc;
}
// Same down kernel WITHOUT the early-launch prefix: it starts only after the
// gate/up launch completes, so no block holds a slot while waiting for mids.
extern "C" __global__ void __launch_bounds__(128,3) moe_r52ds_late(MF_PARAMS,float* fin) { r52_down_split(MF_ARGS,fin,nullptr); }
// ---- R52 split down kernel, TWO adjacent splits of one (slot, piece) per block ----
// 400 blocks fit the SMs at once (3 per SM at this register budget), so none waits
// for a second wave. Split A's trellis words are loaded before the wait; split
// B's words for k step d are requested into the same registers as soon as split
// A's step d has been decoded. Each split's MMA sequence and cross-warp sum are
// ms_mma_tiles' exactly; the piece's scale tail runs when both blocks of the
// (slot, piece) are done, the final sum when all ten slots of the piece are.
__device__ __forceinline__ void r52_split_mma(const unsigned (&wp)[3][8],const unsigned (&wl)[3][8],
    const unsigned (&b0)[3],const unsigned (&b1)[3],unsigned begin,unsigned end,unsigned warp,unsigned lane,
    unsigned flip,float (&acc)[8][4]) {
  #pragma unroll
  for(unsigned m=0;m<8;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=begin+warp+d*4u;
    if(tk<end) {
      #pragma unroll
      for(unsigned m=0;m<8;m++) {
        const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0[d]),"r"(b1[d]));
      }
    }
  }
}
__device__ __forceinline__ void r52_down_split2(MF_PARAMS,float* fin,unsigned long long* trace=nullptr) {
  constexpr unsigned E=10,K=640,N=2560,SPLITS=4,IT=K/16,NT=N/16,PER=IT/SPLITS,M=8;
  __shared__ __align__(16) float part[M*4*16];
  __shared__ unsigned flag;
  const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
  const unsigned b=blockIdx.x,slot=b/40u,piece=(b%40u)/2u,half=b%2u,ot0=piece*8u;
  const unsigned sA=2u*half,sB=sA+1u;
  const unsigned row=(lane&3u)*2u,col=lane>>2;
  unsigned long long* tr=trace?trace+21120u+b*6u:nullptr;
  if(tr && tid==0) tr[0]=r52_now();
  const unsigned word_end=lane*14u+14u,word_last=(word_end-1u)>>5;
  const unsigned word_prev=word_last?word_last-1u:13u;
  const unsigned flip=r52_flip();
  const unsigned expert=__ldcg(ids+slot);
  const unsigned* weights=td+(unsigned long long)expert*IT*NT*14u;
  unsigned wp[3][M],wl[3][M];
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=sA*PER+warp+d*4u;
    const bool live=tk<sA*PER+PER;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tk*NT+ot0+m)*14u;
      wp[d][m]=live?words[word_prev]:0u;
      wl[d][m]=live?words[word_last]:0u;
    }
  }
  unsigned psv[2]={0u,0u};
  if(warp==0) r52_pack4(sv+expert*2560u+piece*128u,lane,psv);
  if(tid==0) {
    const unsigned c0=(sA*160u)/128u,c1=(sB*160u+159u)/128u;
    for(unsigned c=c0;c<=c1;c++) r52_spin(ctr,R52_MREADY+slot*5u+c);
    __threadfence();
    if(tr) tr[1]=r52_now();
  }
  __syncthreads();
  unsigned bA0[3],bA1[3],bB0[3],bB1[3];
  #pragma unroll
  for(unsigned hh=0;hh<2;hh++) {
    const unsigned sp=hh?sB:sA;
    #pragma unroll
    for(unsigned d=0;d<3;d++) {
      const unsigned tk=sp*PER+warp+d*4u;
      const bool xin=tk<sp*PER+PER && col<1u;
      const unsigned short* xx=middle+(unsigned long long)slot*K+tk*16u;
      const unsigned x0v=xin?__ldcg(xx+row):0u,x1v=xin?__ldcg(xx+row+1):0u;
      const unsigned x8=xin?__ldcg(xx+row+8):0u,x9=xin?__ldcg(xx+row+9):0u;
      if(hh) { bB0[d]=x0v|(x1v<<16); bB1[d]=x8|(x9<<16); } else { bA0[d]=x0v|(x1v<<16); bA1[d]=x8|(x9<<16); }
    }
  }
  float acc[M][4];
  // Split A, requesting split B's words for each k step right after it is decoded.
  #pragma unroll
  for(unsigned m=0;m<M;m++) { acc[m][0]=0.f;acc[m][1]=0.f;acc[m][2]=0.f;acc[m][3]=0.f; }
  #pragma unroll
  for(unsigned d=0;d<3;d++) {
    const unsigned tk=sA*PER+warp+d*4u;
    if(tk<sA*PER+PER) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        const unsigned window=mi_window(wp[d][m],wl[d][m],lane);
        unsigned a[4];
        #pragma unroll
        for(unsigned p=0;p<4;p++)a[p]=R52_PAIR(window,p,flip);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[m][0]),"+f"(acc[m][1]),"+f"(acc[m][2]),"+f"(acc[m][3])
            : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(bA0[d]),"r"(bA1[d]));
      }
    }
    const unsigned tkb=sB*PER+warp+d*4u;
    const bool liveb=tkb<sB*PER+PER;
    #pragma unroll
    for(unsigned m=0;m<M;m++) {
      const unsigned* words=weights+((unsigned long long)tkb*NT+ot0+m)*14u;
      wp[d][m]=liveb?words[word_prev]:0u;
      wl[d][m]=liveb?words[word_last]:0u;
    }
  }
  #pragma unroll
  for(unsigned hh=0;hh<2;hh++) {
    const unsigned sp=hh?sB:sA;
    if(hh) r52_split_mma(wp,wl,bB0,bB1,sB*PER,sB*PER+PER,warp,lane,flip,acc);
    if((lane&3u)==0u) {
      #pragma unroll
      for(unsigned m=0;m<M;m++) {
        part[(m*4+warp)*16+col]=acc[m][0];
        part[(m*4+warp)*16+col+8]=acc[m][2];
      }
    }
    __syncthreads();
    {
      const unsigned i=tid,m=i/16,r=i%16;
      float sum=0;for(unsigned w=0;w<4;w++)sum+=part[(m*4+w)*16+r];
      dpar[((unsigned long long)slot*SPLITS+sp)*N+(ot0+m)*16+r]=sum;
    }
    __syncthreads();
  }
  if(tr && tid==0) tr[2]=r52_now();
  unsigned tail=0;
  if(mf_done(ctr,R52_DCNT+slot*20u+piece,2u,&flag)) {
    tail=1;
    if(warp==0) r52_scale_piece_pre(slot,piece,dpar,fin,lane,psv);
    if(mf_done(ctr,R52_FCNT+piece,E,&flag)) {
      tail=2;
      if(tid==0) { r52_spin(ctr,R52_SREADY); __threadfence(); }
      __syncthreads();
      if(warp==0) r52_sum_piece(piece,scores,fin,shared_out,rout,rout_b,added,lane);
    }
  }
  __syncthreads();
  if(tid==0) {
    if(tr) { unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); tr[3]=r52_now(); tr[4]=sm; tr[5]=5u+16u*tail; }
    __threadfence();
    if(atomicAdd(ctr+R52_EXIT2,1u)==gridDim.x-1u) {
      for(unsigned i=R52_BASE;i<R52_END;i++) if(i!=R52_STATUS) ctr[i]=0u;
      __threadfence();
    }
  }
}
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52ds2(MF_PARAMS,float* fin) { r52_down_split2(MF_ARGS,fin); }
extern "C" __global__ void __launch_bounds__(128,3) r39pdl_moe_r52ds2t(MF_PARAMS,float* fin,unsigned long long* trace) { r52_down_split2(MF_ARGS,fin,trace); }
// B1 (gate/up + shared, no down items) with the mid tail's expert scales preloaded.
extern "C" __global__ void __launch_bounds__(128,6) moe_r52_back_g4_vpre(MF_PARAMS,float* fin) { r52_back<4,4,true,false,true,2,false,true>(MF_ARGS,fin,nullptr); }
// ---- F0z: F0x with every constant weight requested BEFORE griddepcontrol.wait ----
// Router warps request their 80 BF16 weights, shared-partial warps their packed
// codes/scales/biases for the whole K split, then wait for the producer and only
// then read the activation input. Same FMA order and reductions as F0x/mi_partial.
__device__ __forceinline__ void r52_partial_pre(const unsigned short* x,const unsigned* w,
    const unsigned short* s,const unsigned short* b,float* partial,unsigned k,unsigned n,
    unsigned group,unsigned splits,unsigned bx,unsigned part,unsigned warp2,unsigned lane) {
  const unsigned n0=bx*4+warp2*2;
  const unsigned chunks=(k+255)/256,begin=(chunks*part/splits)*256,end=(chunks*(part+1)/splits)*256;
  const unsigned groups=k/group;
  unsigned packed[4][2];
  unsigned short sr[4][2],br[4][2];
  #pragma unroll
  for(unsigned c=0;c<4;c++) {
    const unsigned kb=begin+c*256u,k0=kb+lane*8u;
    #pragma unroll
    for(unsigned col=0;col<2;col++) {
      const bool okc=kb<end && k0<k && n0+col<n;
      const unsigned at=(n0+col)*groups+k0/group;
      sr[c][col]=okc?s[at]:(unsigned short)0;
      br[c][col]=okc?b[at]:(unsigned short)0;
      packed[c][col]=okc?w[(size_t)(n0+col)*(k/8u)+k0/8u]:0u;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float acc[2]={0.f,0.f};
  #pragma unroll
  for(unsigned it=0;it<2;it++) {
    const unsigned base=begin+it*512u;
    if(base>=end) break;
    unsigned xw[2][4];
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned kb=base+d*256u,k0=kb+lane*8u;
      const bool ok=kb<end && k0+8u<=k;
      const uint4 q=ok?*reinterpret_cast<const uint4*>(x+k0):make_uint4(0u,0u,0u,0u);
      xw[d][0]=q.x;xw[d][1]=q.y;xw[d][2]=q.z;xw[d][3]=q.w;
    }
    #pragma unroll
    for(unsigned d=0;d<2;d++) {
      const unsigned c=it*2u+d,kb=base+d*256u,k0=kb+lane*8u;
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
          const float sc=mi_load(sr[c][col]),bi=mi_load(br[c][col]);
          const unsigned p=packed[c][col];
          float dot=0.f;
          #pragma unroll
          for(unsigned i=0;i<8;i++) dot=fmaf(vals[i],float((p>>(4*i))&15u),dot);
          acc[col]+=fmaf(sc,dot,bi*sums);
        }
      }
    }
  }
  #pragma unroll
  for(unsigned col=0;col<2;col++) {
    const float v=mi_warp_total(acc[col]);
    if(lane==0 && n0+col<n) partial[(0*splits+part)*n+n0+col]=v;
  }
}
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0z(MF_PARAMS,unsigned* rc) {
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for(unsigned i=t;i<H/8;i+=128)
      reinterpret_cast<uint4*>(sxv)[i]=reinterpret_cast<const uint4*>(x)[i];
    __syncthreads();
    float acc=0;
    #pragma unroll
    for(int j=0;j<80;j++) acc=fmaf(mi_load(sxv[lane+32*j]),mi_load(wr[j]),acc);
    for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
    if(lane==0) rraw[e]=acc;
    if(mf_done(ctr,0u,128u,&flag)) {
      if(warp==0) r52_select_fast(rraw,rlogits,ids,scores,select_arm,lane,nullptr);
      __threadfence(); __syncthreads();
      if(t==0) { r52_release(rc+R52_IDS); ctr[0]=0u; }
    }
  } else {
    const unsigned v=2*(pb-128)+(warp>>1);
    r52_partial_pre(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
  }
}
// Gate/up launch clone that waits at entry: early-launch eligible, so its blocks
// are resident (and the following down launch can start) when F0 completes.
extern "C" __global__ void __launch_bounds__(128,6) r39pdl_moe_r52_back_g4_v(MF_PARAMS,float* fin) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  r52_back<4,4,false,false,true,2,false,true>(MF_ARGS,fin,nullptr);
}

// ---- small targeted L2 prefetches (QWEN_R52_SPF) ----
// F0z whose router tail, right after selecting the ten experts, prefetches their
// input sign vectors (sg/su, 2 x 5 KB each) and the mid/finish scale vectors
// (vg/vu/sd 1.25 KB each, sv 5 KB) into L2 for the gate/up and down launches.
extern "C" __global__ void __launch_bounds__(128,5) r39pdl_moe_r52f0zp(MF_PARAMS,unsigned* rc) {
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for(unsigned i=t;i<H/8;i+=128)
      reinterpret_cast<uint4*>(sxv)[i]=reinterpret_cast<const uint4*>(x)[i];
    __syncthreads();
    float acc=0;
    #pragma unroll
    for(int j=0;j<80;j++) acc=fmaf(mi_load(sxv[lane+32*j]),mi_load(wr[j]),acc);
    for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffffU,acc,s);
    if(lane==0) rraw[e]=acc;
    if(mf_done(ctr,0u,128u,&flag)) {
      if(warp==0) r52_select_fast(rraw,rlogits,ids,scores,select_arm,lane,nullptr);
      __threadfence(); __syncthreads();
      if(t==0) { r52_release(rc+R52_IDS); ctr[0]=0u; }
      // 10 experts x (sg 40 + su 40 + sv 40 + vg 10 + vu 10 + sd 10 lines of 128 B)
      for(unsigned i=t;i<10u*150u;i+=128u) {
        const unsigned slot=i/150u,r=i%150u,ex=__ldcg(ids+slot);
        const unsigned short* base;unsigned line;
        if(r<40u) { base=sg+ex*2560u; line=r; }
        else if(r<80u) { base=su+ex*2560u; line=r-40u; }
        else if(r<120u) { base=sv+ex*2560u; line=r-80u; }
        else if(r<130u) { base=vg+ex*640u; line=r-120u; }
        else if(r<140u) { base=vu+ex*640u; line=r-130u; }
        else { base=sd+ex*640u; line=r-140u; }
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(base+line*64u));
      }
    }
  } else {
    const unsigned v=2*(pb-128)+(warp>>1);
    r52_partial_pre(x,gw,gs,gb,gpart,H,2*I,ggroup,SPLITS,v%320,v/320,warp&1,lane);
  }
}
