// R38 (Opus, direct): load-ahead variant of R22's mcg_mma_direct (mcg_tensor.cu).
// Same licensed Sushi-derived MCG bitstream decode and m16n8k16 organization;
// see NOTICE.txt. Each warp issues the trellis-word and input loads for D of its
// K tiles before decoding any of them. Tile order, decode and every MMA are the
// same as mcg_mma_direct, so outputs are intended to match it bit for bit.
// Direct mode (one routed slot per blockIdx.y), Mode 1 (MMA) only.
__device__ __forceinline__ unsigned mta_window(unsigned prev_word,unsigned last_word,unsigned lane) {
    const unsigned end=lane*14+14;
    const unsigned last=(end-1)>>5;
    const unsigned shift=(last+1)*32-end;
    const unsigned long long bits=((unsigned long long)prev_word<<32)|last_word;
    return (unsigned)(bits>>shift);
}
__device__ __forceinline__ unsigned mta_pair_values(unsigned window,unsigned pair) {
    const unsigned offset=pair==0?11:pair==1?7:pair==2?4:0;
    const unsigned cw=window>>offset;
    const unsigned c0=(cw>>2)&0x7fffu,c1=cw&0x7fffu;
    const unsigned p0=((c0*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    const unsigned p1=((c1*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    const unsigned lo=(p0&0xffffu)|(p1<<16),hi=(p0>>16)|(p1&0xffff0000u);
    unsigned result; asm("add.rn.f16x2 %0, %1, %2;":"=r"(result):"r"(lo),"r"(hi));
    return result;
}
template<unsigned W,unsigned D>
__device__ void mta_project_direct(const unsigned short* x,const unsigned* trellis,
    const unsigned* desc,const unsigned* group_count,float* y,
    unsigned k,unsigned n,unsigned splits,float* partial) {
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned word_end=lane*14+14,word_last=(word_end-1)>>5;
    const unsigned word_prev=word_last?word_last-1:13;
    for(unsigned task=blockIdx.x;task<nt;task+=gridDim.x) {
        const unsigned ot=task,split=blockIdx.z;
        const unsigned count=1,begin=split*per,end=min(begin+per,it);
        const unsigned expert=desc[blockIdx.y];
        const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
        const unsigned slot=blockIdx.y;
        float acc[4]={};
        for(unsigned tk0=begin+warp;tk0<end;tk0+=W*D) {
            // Load phase: raw words only; nothing here consumes a loaded value.
            unsigned wp[D],wl[D],b0[D],b1[D];
            #pragma unroll
            for(unsigned d=0;d<D;d++) {
                const unsigned tk=tk0+d*W;
                const bool live=tk<end;
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
                wp[d]=live?words[word_prev]:0u;
                wl[d]=live?words[word_last]:0u;
                const bool xin=live && col<count;
                const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
                const unsigned x0=xin?xx[row]:0u,x1=xin?xx[row+1]:0u;
                const unsigned x8=xin?xx[row+8]:0u,x9=xin?xx[row+9]:0u;
                b0[d]=x0|(x1<<16);
                b1[d]=x8|(x9<<16);
            }
            // Compute phase: tiles in their original order, identical decode/MMA.
            #pragma unroll
            for(unsigned d=0;d<D;d++) {
                const unsigned tk=tk0+d*W;
                if(tk<end) {
                    const unsigned window=mta_window(wp[d],wl[d],lane);
                    unsigned a[4];
                    #pragma unroll
                    for(unsigned p=0;p<4;p++)a[p]=mta_pair_values(window,p);
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
        for(unsigned i=tid;i<count*16;i+=W*32) {
            float sum=0;for(unsigned w=0;w<W;w++)sum+=partial[w*128+i];
            y[((unsigned long long)slot*splits+split)*n+ot*16+i%16]=sum;
        }
        __syncthreads();
    }
}
#define MTA_ENTRY(NAME,W,D) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* t, \
 const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits) { \
 __shared__ float partial[W*128];mta_project_direct<W,D>(x,t,desc,groups,y,k,n,splits,partial); \
}
MTA_ENTRY(mcg_mma_direct_ahead2,4,2)
MTA_ENTRY(mcg_mma_direct_ahead4,4,4)
MTA_ENTRY(mcg_mma_direct_ahead8,4,8)
// Gate and up projections in one launch: blockIdx.y < slots runs (x0,t0,y0),
// otherwise (x1,t1,y1) for slot blockIdx.y-slots. Each block's work is exactly
// mcg_mma_direct_ahead2's for that projection.
template<unsigned W,unsigned D>
__device__ void mta_project_pair(const unsigned short* x0,const unsigned* t0,float* y0,
    const unsigned short* x1,const unsigned* t1,float* y1,const unsigned* desc,
    unsigned slots,unsigned k,unsigned n,unsigned splits,float* partial) {
    const bool second=blockIdx.y>=slots;
    const unsigned slot=second?blockIdx.y-slots:blockIdx.y;
    const unsigned short* x=second?x1:x0;
    const unsigned* trellis=second?t1:t0;
    float* y=second?y1:y0;
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned word_end=lane*14+14,word_last=(word_end-1)>>5;
    const unsigned word_prev=word_last?word_last-1:13;
    for(unsigned task=blockIdx.x;task<nt;task+=gridDim.x) {
        const unsigned ot=task,split=blockIdx.z;
        const unsigned count=1,begin=split*per,end=min(begin+per,it);
        const unsigned expert=desc[slot];
        const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
        float acc[4]={};
        for(unsigned tk0=begin+warp;tk0<end;tk0+=W*D) {
            unsigned wp[D],wl[D],b0[D],b1[D];
            #pragma unroll
            for(unsigned d=0;d<D;d++) {
                const unsigned tk=tk0+d*W;
                const bool live=tk<end;
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
                wp[d]=live?words[word_prev]:0u;
                wl[d]=live?words[word_last]:0u;
                const bool xin=live && col<count;
                const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
                const unsigned x0v=xin?xx[row]:0u,x1v=xin?xx[row+1]:0u;
                const unsigned x8=xin?xx[row+8]:0u,x9=xin?xx[row+9]:0u;
                b0[d]=x0v|(x1v<<16);
                b1[d]=x8|(x9<<16);
            }
            #pragma unroll
            for(unsigned d=0;d<D;d++) {
                const unsigned tk=tk0+d*W;
                if(tk<end) {
                    const unsigned window=mta_window(wp[d],wl[d],lane);
                    unsigned a[4];
                    #pragma unroll
                    for(unsigned p=0;p<4;p++)a[p]=mta_pair_values(window,p);
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
        for(unsigned i=tid;i<count*16;i+=W*32) {
            float sum=0;for(unsigned w=0;w<W;w++)sum+=partial[w*128+i];
            y[((unsigned long long)slot*splits+split)*n+ot*16+i%16]=sum;
        }
        __syncthreads();
    }
}
extern "C" __global__ void mcg_mma_pair_ahead2(const unsigned short* x0,const unsigned* t0,float* y0,
    const unsigned short* x1,const unsigned* t1,float* y1,const unsigned* desc,
    unsigned slots,unsigned k,unsigned n,unsigned splits) {
    __shared__ float partial[4*128];
    mta_project_pair<4,2>(x0,t0,y0,x1,t1,y1,desc,slots,k,n,splits,partial);
}
extern "C" __global__ void mcg_mma_pair_ahead4(const unsigned short* x0,const unsigned* t0,float* y0,
    const unsigned short* x1,const unsigned* t1,float* y1,const unsigned* desc,
    unsigned slots,unsigned k,unsigned n,unsigned splits) {
    __shared__ float partial[4*128];
    mta_project_pair<4,4>(x0,t0,y0,x1,t1,y1,desc,slots,k,n,splits,partial);
}
extern "C" __global__ void mcg_mma_pair_ahead8(const unsigned short* x0,const unsigned* t0,float* y0,
    const unsigned short* x1,const unsigned* t1,float* y1,const unsigned* desc,
    unsigned slots,unsigned k,unsigned n,unsigned splits) {
    __shared__ float partial[4*128];
    mta_project_pair<4,8>(x0,t0,y0,x1,t1,y1,desc,slots,k,n,splits,partial);
}
