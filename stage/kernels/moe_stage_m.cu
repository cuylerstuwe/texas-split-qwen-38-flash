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
