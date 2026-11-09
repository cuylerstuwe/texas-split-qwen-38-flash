// R60: kernels/mcg_tensor.cu (R22 grouping512, FP16x2 decode, decode-once MMA) for the canonical
// target's EXL3 K4 trellis (64 halfwords per 16x16 tile). Only the per-lane codeword window
// changes: a lane's eight weights advance 32 bits, so the eight 15-bit codewords lie in a 43-bit
// tail-biting window over the lane's word and the one before it (mcg_project.cu, H=64).
// MCG bitstream/permutation follows the licensed Sushi-derived mcg_project.cu; see NOTICE.txt.
__device__ __forceinline__ float hf(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
// Every lane's eight weights advance 32 bits: the window ends at the lane's last codeword.
__device__ __forceinline__ unsigned long long lane_window(const unsigned* words,unsigned lane) {
    return ((unsigned long long)words[(lane+31u)&31u]<<32)|words[lane];
}
// Weights 2p and 2p+1 of the lane: codewords ending 4*(7-2p) and 4*(6-2p) bits before the end.
__device__ __forceinline__ unsigned pair_values(unsigned long long window,unsigned pair) {
    const unsigned cw=(unsigned)(window>>(24u-8u*pair));
    const unsigned c0=(cw>>4)&0x7fffu,c1=cw&0x7fffu;
    const unsigned p0=((c0*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    const unsigned p1=((c1*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    const unsigned lo=(p0&0xffffu)|(p1<<16),hi=(p0>>16)|(p1&0xffff0000u);
    unsigned result; asm("add.rn.f16x2 %0, %1, %2;":"=r"(result):"r"(lo),"r"(hi));
    return result;
}
// One512-thread block, original IDs0..511; desc=[expert,count,slot0..slot7].
// No host route reads, atomics, global spin or cross-block barrier.
extern "C" __global__ void mcg_group512(const unsigned* ids,unsigned* desc,
    unsigned* group_count,unsigned slots,unsigned experts) {
    __shared__ unsigned counts[512];
    unsigned e=threadIdx.x,count=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)count+=ids[s]==e;
    counts[e]=count; __syncthreads();
    unsigned prefix=0;
    for(unsigned j=0;j<e;j++)prefix+=(counts[j]+7)/8;
    unsigned chunks=(count+7)/8;
    if(e==511)*group_count=prefix+chunks;
    for(unsigned c=0;c<chunks;c++) {
        unsigned* d=desc+(prefix+c)*10;
        d[0]=e;d[1]=min(8u,count-c*8);
        for(unsigned j=0;j<8;j++)d[2+j]=0xffffffffu;
    }
    unsigned at=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)if(ids[s]==e) {
        desc[(prefix+at/8)*10+2+at%8]=s; at++;
    }
}
template<unsigned W,unsigned Mode,bool Direct=false>
__device__ void project(const unsigned short* x,const unsigned* trellis,
    const unsigned* desc,const unsigned* group_count,float* y,
    unsigned k,unsigned n,unsigned splits,float* partial) {
    unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    unsigned row=(lane&3)*2,col=lane>>2;
    constexpr unsigned B=(Mode==2 || Direct)?1:8;
    unsigned tasks=Direct?nt:(*group_count)*nt*splits;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        unsigned ot=Direct?task:task%nt,split=Direct?blockIdx.z:(task/nt)%splits,group=task/(nt*splits);
        const unsigned* d=desc+group*10;
        unsigned count=Direct?1:d[1],begin=split*per,end=min(begin+per,it);
        unsigned expert=Direct?desc[blockIdx.y]:d[0];
        const unsigned* weights=trellis+(unsigned long long)expert*it*nt*32;
        if(Mode==1 || (Mode==2 && count>1)) {
            float acc[4]={};
            for(unsigned tk=begin+warp;tk<end;tk+=W) {
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*32;
                const unsigned long long window=lane_window(words,lane);
                unsigned a[4];
                #pragma unroll
                for(unsigned p=0;p<4;p++)a[p]=pair_values(window,p);
                unsigned b0=0,b1=0;
                if(col<count) {
                    unsigned slot=Direct?blockIdx.y:d[2+col];
                    const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
                    b0=unsigned(xx[row])|(unsigned(xx[row+1])<<16);
                    b1=unsigned(xx[row+8])|(unsigned(xx[row+9])<<16);
                }
                // A = W^T [16out,16in], B = inputs [16in,8rows]. MCG pairs
                // order0,2,1,3 becomes NVIDIA's m16n8k16 A fragment order.
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
                    : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
            }
            partial[warp*128+row*16+col]=acc[0];
            partial[warp*128+(row+1)*16+col]=acc[1];
            partial[warp*128+row*16+col+8]=acc[2];
            partial[warp*128+(row+1)*16+col+8]=acc[3];
        } else {
            float acc[B][8]={};
            for(unsigned tk=begin+warp;tk<end;tk+=W) {
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*32;
                const unsigned long long window=lane_window(words,lane);
                float a[B][4];
                #pragma unroll
                for(unsigned r=0;r<B;r++)if(r<count) {
                    unsigned slot=Direct?blockIdx.y:d[2+r];
                    const unsigned short* xx=x+(unsigned long long)slot*k+tk*16;
                    a[r][0]=hf(xx[row]);a[r][1]=hf(xx[row+1]);
                    a[r][2]=hf(xx[row+8]);a[r][3]=hf(xx[row+9]);
                }
                #pragma unroll
                for(unsigned p=0;p<4;p++) {
                    unsigned packed=pair_values(window,p);
                    float w0=hf((unsigned short)packed),w1=hf((unsigned short)(packed>>16));
                    #pragma unroll
                    for(unsigned r=0;r<B;r++)if(r<count) {
                        acc[r][p*2]=__fmaf_rn(a[r][(p*2)&3],w0,acc[r][p*2]);
                        acc[r][p*2+1]=__fmaf_rn(a[r][(p*2+1)&3],w1,acc[r][p*2+1]);
                    }
                }
            }
            #pragma unroll
            for(unsigned r=0;r<B;r++)if(r<count) {
                float lo=(acc[r][0]+acc[r][1])+(acc[r][2]+acc[r][3]);
                float hi=(acc[r][4]+acc[r][5])+(acc[r][6]+acc[r][7]);
                lo+=__shfl_xor_sync(0xffffffffu,lo,1);hi+=__shfl_xor_sync(0xffffffffu,hi,1);
                lo+=__shfl_xor_sync(0xffffffffu,lo,2);hi+=__shfl_xor_sync(0xffffffffu,hi,2);
                if((lane&3)==0){partial[warp*128+r*16+col]=lo;partial[warp*128+r*16+col+8]=hi;}
            }
        }
        __syncthreads();
        for(unsigned i=tid;i<count*16;i+=W*32) {
            float sum=0;for(unsigned w=0;w<W;w++)sum+=partial[w*128+i];
            unsigned slot=Direct?blockIdx.y:d[2+i/16];
            y[((unsigned long long)slot*splits+split)*n+ot*16+i%16]=sum;
        }
        __syncthreads();
    }
}
#define ENTRY(NAME,W,MODE,DIRECT) \
extern "C" __global__ void NAME(const unsigned short* x,const unsigned* t, \
 const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits) { \
 __shared__ float partial[W*128];project<W,MODE,DIRECT>(x,t,desc,groups,y,k,n,splits,partial); \
}
ENTRY(mcg_scalar8,4,0,false)
ENTRY(mcg_mma8_w4,4,1,false)
ENTRY(mcg_mma8_w1,1,1,false)
ENTRY(mcg_half2_direct,4,0,true)
ENTRY(mcg_mma_direct,4,1,true)
ENTRY(mcg_hybrid8,4,2,false)
// R60 S4 (prefill v2): expert groups of up to 64 slots. mcg_group512_64 is mcg_group512 with
// 64-slot chunks (descriptor = [expert, count, slot0..slot63], 66 words). mcg_mma64 decodes each
// 16x16 tile once per k step (the same fragment as mcg_mma8_w4) and feeds up to eight n8 MMAs, so
// a prefill chunk (~40 tokens per expert) decodes every weight about once instead of once per
// eight slots. x rows are slot / xdiv (xdiv = top-k for gate/up, whose input is per token; 1 for
// down, per slot); y is [slot][split][n] as the decode kernels.
extern "C" __global__ void mcg_group512_64(const unsigned* ids,unsigned* desc,
    unsigned* group_count,unsigned slots,unsigned experts) {
    __shared__ unsigned counts[512];
    unsigned e=threadIdx.x,count=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)count+=ids[s]==e;
    counts[e]=count; __syncthreads();
    unsigned prefix=0;
    for(unsigned j=0;j<e;j++)prefix+=(counts[j]+63)/64;
    unsigned chunks=(count+63)/64;
    if(e==511)*group_count=prefix+chunks;
    for(unsigned c=0;c<chunks;c++) {
        unsigned* d=desc+(prefix+c)*66;
        d[0]=e;d[1]=min(64u,count-c*64);
        for(unsigned j=0;j<64;j++)d[2+j]=0xffffffffu;
    }
    unsigned at=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)if(ids[s]==e) {
        desc[(prefix+at/64)*66+2+at%64]=s; at++;
    }
}
template<unsigned W>
__device__ void project64(const unsigned short* x,const unsigned* trellis,const unsigned* desc,
    const unsigned* group_count,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv,float* partial) {
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned tasks=(*group_count)*nt*splits;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned ot=task%nt,split=(task/nt)%splits,group=task/(nt*splits);
        const unsigned* d=desc+group*66;
        const unsigned count=d[1],tiles=(count+7)/8,begin=split*per,end=min(begin+per,it);
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*32;
        const unsigned short* xs[8];
        #pragma unroll
        for(unsigned t=0;t<8;t++) {
            const unsigned j=t*8+col;
            xs[t]=j<count?x+(unsigned long long)(d[2+j]/xdiv)*k:nullptr;
        }
        float acc[8][4];
        #pragma unroll
        for(unsigned t=0;t<8;t++){acc[t][0]=acc[t][1]=acc[t][2]=acc[t][3]=0.f;}
        for(unsigned tk=begin+warp;tk<end;tk+=W) {
            const unsigned* words=weights+((unsigned long long)tk*nt+ot)*32;
            const unsigned long long window=lane_window(words,lane);
            unsigned a[4];
            #pragma unroll
            for(unsigned p=0;p<4;p++)a[p]=pair_values(window,p);
            #pragma unroll
            for(unsigned t=0;t<8;t++) {
                if(t<tiles) {
                    unsigned b0=0,b1=0;
                    if(xs[t]) {
                        const unsigned short* xx=xs[t]+tk*16;
                        b0=unsigned(xx[row])|(unsigned(xx[row+1])<<16);
                        b1=unsigned(xx[row+8])|(unsigned(xx[row+9])<<16);
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[t][0]),"+f"(acc[t][1]),"+f"(acc[t][2]),"+f"(acc[t][3])
                        : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
                }
            }
        }
        // C element: output row/col as mcg_mma8_w4 (slot = tile*8 + row/row+1, channel col/col+8)
        #pragma unroll
        for(unsigned t=0;t<8;t++) if(t<tiles) {
            float* pp=partial+warp*1024+t*128;
            pp[row*16+col]=acc[t][0];pp[(row+1)*16+col]=acc[t][1];
            pp[row*16+col+8]=acc[t][2];pp[(row+1)*16+col+8]=acc[t][3];
        }
        __syncthreads();
        for(unsigned i=tid;i<count*16;i+=W*32) {
            float sum=0;for(unsigned w=0;w<W;w++)sum+=partial[w*1024+i];
            const unsigned slot=d[2+i/16];
            y[((unsigned long long)slot*splits+split)*n+ot*16+i%16]=sum;
        }
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(128) mcg_mma64(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv) {
    __shared__ float partial[4*1024];
    project64<4>(x,t,desc,groups,y,k,n,splits,xdiv,partial);
}
// R60 S4 (prefill v2): mcg_mma64 re-read each slot's inputs for every 16-column tile, as 2-byte
// loads from eight rows per instruction: ~10x the weight bytes in scattered L2 traffic. mcg_mma64s
// stages each 256-code chunk of the group's inputs in shared memory once per block (rows padded to
// 264 halves: the B-fragment loads are bank-conflict free) and each warp owns two column tiles over
// the whole K (no cross-warp reduction; C goes straight to y). Block = one (group, 8-tile part).
#define M64S_STRIDE 264u
template<unsigned W>
__device__ void project64s(const unsigned short* x,const unsigned* trellis,const unsigned* desc,
    const unsigned* group_count,float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs) {
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,parts=(nt+2*W-1)/(2*W);
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned tasks=(*group_count)*parts;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned group=task/parts,part=task%parts;
        const unsigned* d=desc+group*66;
        const unsigned count=d[1],tiles=(count+7)/8;
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*32;
        const unsigned ot0=part*2*W+2*warp;
        float acc[2][8][4];
        #pragma unroll
        for(unsigned c=0;c<2;c++)
            #pragma unroll
            for(unsigned t=0;t<8;t++){acc[c][t][0]=acc[c][t][1]=acc[c][t][2]=acc[c][t][3]=0.f;}
        for(unsigned kc=0;kc<it;kc+=16) {
            const unsigned steps=min(16u,it-kc);
            __syncthreads();
            // stage inputs [slot j][256 codes of this chunk], 16 bytes per thread per iteration
            for(unsigned e=tid;e<count*32;e+=W*32) {
                const unsigned j=e>>5,q=e&31;
                uint4 v=make_uint4(0,0,0,0);
                if(q*8<steps*16) v=*reinterpret_cast<const uint4*>(x+(unsigned long long)(d[2+j]/xdiv)*k+kc*16+q*8);
                *reinterpret_cast<uint4*>(xs+j*M64S_STRIDE+q*8)=v;
            }
            __syncthreads();
            for(unsigned s0=0;s0<steps;s0+=8) {
              // eight k steps of tile windows for both column tiles in flight before any decode
              unsigned long long win[2][8];
              #pragma unroll
              for(unsigned c=0;c<2;c++)
                #pragma unroll
                for(unsigned u=0;u<8;u++)
                  win[c][u]=(ot0+c<nt && s0+u<steps)?lane_window(weights+((unsigned long long)(kc+s0+u)*nt+ot0+c)*32,lane):0ull;
              #pragma unroll
              for(unsigned u=0;u<8;u++) {
                const unsigned s=s0+u;
                if(s>=steps) break;
                #pragma unroll
                for(unsigned c=0;c<2;c++) {
                    const unsigned ot=ot0+c;
                    if(ot<nt) {
                        const unsigned long long window=win[c][u];
                        unsigned a[4];
                        #pragma unroll
                        for(unsigned p=0;p<4;p++)a[p]=pair_values(window,p);
                        #pragma unroll
                        for(unsigned t=0;t<8;t++) {
                            if(t<tiles) {
                                const unsigned j=t*8+col;
                                unsigned b0=0,b1=0;
                                if(j<count) {
                                    const unsigned short* xr=xs+j*M64S_STRIDE+s*16+row;
                                    b0=*reinterpret_cast<const unsigned*>(xr);
                                    b1=*reinterpret_cast<const unsigned*>(xr+8);
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                    : "+f"(acc[c][t][0]),"+f"(acc[c][t][1]),"+f"(acc[c][t][2]),"+f"(acc[c][t][3])
                                    : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
                            }
                        }
                    }
                }
              }
            }
        }
        // C element: channel col / col+8 of tile ot, slot t*8+row / +1
        #pragma unroll
        for(unsigned c=0;c<2;c++) {
            const unsigned ot=ot0+c;
            if(ot>=nt) continue;
            #pragma unroll
            for(unsigned t=0;t<8;t++) if(t<tiles) {
                const unsigned j0=t*8+row,j1=j0+1;
                if(j0<count){const unsigned long long b=(unsigned long long)d[2+j0]*n+ot*16;y[b+col]=acc[c][t][0];y[b+col+8]=acc[c][t][2];}
                if(j1<count){const unsigned long long b=(unsigned long long)d[2+j1]*n+ot*16;y[b+col]=acc[c][t][1];y[b+col+8]=acc[c][t][3];}
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) mcg_mma64s(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv) {
    __shared__ __align__(16) unsigned short xs[64*M64S_STRIDE];
    project64s<4>(x,t,desc,groups,y,k,n,xdiv,xs);
}
// R60 S4 (prefill v2 pull-back): dst[r][c] = src[row0 + r][col0 + c] for r < rows, c < cols (BF16;
// the last raw QSA keys of a prefill chunk out of the index projection, 640 wide, key at 512).
extern "C" __global__ void rows_cols_copy(const unsigned short* src,unsigned short* dst,unsigned stride,
    unsigned col0,unsigned cols,unsigned row0,unsigned rows) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*cols) return;
    const unsigned r=i/cols,c=i%cols;
    dst[i]=src[(unsigned long long)(row0+r)*stride+col0+c];
}
// R60 S4 (prefill v2): mcg_group512_64 with every thread of one 1024-thread block working the slots
// (shared-memory histogram, group bases, then an atomic scatter; slot order inside a group is free:
// each slot's output row is its own). Same descriptor layout as mcg_group512_64.
extern "C" __global__ void __launch_bounds__(1024) mcg_group512_64f(const unsigned* ids,unsigned* desc,
    unsigned* group_count,unsigned slots,unsigned experts) {
    __shared__ unsigned hist[512],base[512],fill[512];
    const unsigned t=threadIdx.x;
    for(unsigned e=t;e<512;e+=1024){hist[e]=0;fill[e]=0;}
    __syncthreads();
    for(unsigned s=t;s<slots;s+=1024){const unsigned e=ids[s];if(e<experts)atomicAdd(&hist[e],1u);}
    __syncthreads();
    if(t<32) {
        // 32 lanes scan 512 experts' group counts (16 each, then a warp-wide exclusive scan)
        unsigned local[16],sum=0;
        for(unsigned j=0;j<16;j++){local[j]=(hist[t*16+j]+63)/64;sum+=local[j];}
        unsigned incl=sum;
        for(unsigned o=1;o<32;o<<=1){const unsigned v=__shfl_up_sync(0xffffffffu,incl,o);if(t>=o)incl+=v;}
        unsigned run=incl-sum;
        for(unsigned j=0;j<16;j++){base[t*16+j]=run;run+=local[j];}
        if(t==31)*group_count=incl;
    }
    __syncthreads();
    for(unsigned e=t;e<512;e+=1024) {
        const unsigned n=hist[e],chunks=(n+63)/64;
        for(unsigned c=0;c<chunks;c++){unsigned* d=desc+(base[e]+c)*66;d[0]=e;d[1]=min(64u,n-c*64);}
    }
    __syncthreads();
    for(unsigned s=t;s<slots;s+=1024) {
        const unsigned e=ids[s];
        if(e>=experts) continue;
        const unsigned k=atomicAdd(&fill[e],1u);
        desc[(base[e]+k/64)*66+2+k%64]=s;
    }
}
// R60 S4 (prefill v2): project64s with one column tile per warp (half the accumulators and window
// registers of mcg_mma64s, about twice the resident warps); a block is (group, 4-tile part).
template<unsigned W>
__device__ void project64t(const unsigned short* x,const unsigned* trellis,const unsigned* desc,
    const unsigned* group_count,float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs) {
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,parts=(nt+W-1)/W;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned tasks=(*group_count)*parts;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned group=task/parts,part=task%parts;
        const unsigned* d=desc+group*66;
        const unsigned count=d[1],tiles=(count+7)/8;
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*32;
        const unsigned ot=part*W+warp;
        float acc[8][4];
        #pragma unroll
        for(unsigned t=0;t<8;t++){acc[t][0]=acc[t][1]=acc[t][2]=acc[t][3]=0.f;}
        for(unsigned kc=0;kc<it;kc+=16) {
            const unsigned steps=min(16u,it-kc);
            __syncthreads();
            for(unsigned e=tid;e<count*32;e+=W*32) {
                const unsigned j=e>>5,q=e&31;
                uint4 v=make_uint4(0,0,0,0);
                if(q*8<steps*16) v=*reinterpret_cast<const uint4*>(x+(unsigned long long)(d[2+j]/xdiv)*k+kc*16+q*8);
                *reinterpret_cast<uint4*>(xs+j*M64S_STRIDE+q*8)=v;
            }
            __syncthreads();
            if(ot<nt) {
                for(unsigned s0=0;s0<steps;s0+=8) {
                    unsigned long long win[8];
                    #pragma unroll
                    for(unsigned u=0;u<8;u++)
                        win[u]=s0+u<steps?lane_window(weights+((unsigned long long)(kc+s0+u)*nt+ot)*32,lane):0ull;
                    #pragma unroll
                    for(unsigned u=0;u<8;u++) {
                        const unsigned s=s0+u;
                        if(s>=steps) break;
                        unsigned a[4];
                        #pragma unroll
                        for(unsigned p=0;p<4;p++)a[p]=pair_values(win[u],p);
                        #pragma unroll
                        for(unsigned t=0;t<8;t++) {
                            if(t<tiles) {
                                const unsigned j=t*8+col;
                                unsigned b0=0,b1=0;
                                if(j<count) {
                                    const unsigned short* xr=xs+j*M64S_STRIDE+s*16+row;
                                    b0=*reinterpret_cast<const unsigned*>(xr);
                                    b1=*reinterpret_cast<const unsigned*>(xr+8);
                                }
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                    : "+f"(acc[t][0]),"+f"(acc[t][1]),"+f"(acc[t][2]),"+f"(acc[t][3])
                                    : "r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
                            }
                        }
                    }
                }
            }
        }
        if(ot<nt) {
            #pragma unroll
            for(unsigned t=0;t<8;t++) if(t<tiles) {
                const unsigned j0=t*8+row,j1=j0+1;
                if(j0<count){const unsigned long long b=(unsigned long long)d[2+j0]*n+ot*16;y[b+col]=acc[t][0];y[b+col+8]=acc[t][2];}
                if(j1<count){const unsigned long long b=(unsigned long long)d[2+j1]*n+ot*16;y[b+col]=acc[t][1];y[b+col+8]=acc[t][3];}
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) mcg_mma64t(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv) {
    __shared__ __align__(16) unsigned short xs[64*M64S_STRIDE];
    project64t<4>(x,t,desc,groups,y,k,n,xdiv,xs);
}
// R60 S4 (prefill v2): project64s with the slot-tile count fixed per instantiation (TILES =
// ceil(count/8), dispatched once per task). mcg_mma64s guarded every MMA with if(t<tiles) and
// if(j<count): each HMMA sat in its own branch behind a WARPSYNC and two dependent shared loads into
// the same registers, so the shared-memory latency was exposed per MMA (~15% of the tensor peak).
// Here inputs are staged for every row of the instantiation's tiles (zero rows past count), B
// fragments come from ldmatrix.x4 (two slot tiles per instruction, rows 528 B apart: conflict free)
// and every MMA of a k step is unconditional. Requires k % 128 == 0 and (n/16) % (2W) == 0.
__device__ __forceinline__ void m64_ldsm4(unsigned* b,unsigned addr){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]):"r"(addr));
}
__device__ __forceinline__ void m64_mma(float* c,const unsigned* a,unsigned b0,unsigned b1){
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[2]),"r"(a[1]),"r"(a[3]),"r"(b0),"r"(b1));
}
template<unsigned W,unsigned TILES>
__device__ __forceinline__ void project64u_task(const unsigned short* x,const unsigned* weights,const unsigned* d,
    float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs,unsigned part,unsigned count) {
    constexpr unsigned PAIRS=(TILES+1)/2,ROWS=PAIRS*16;
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned ot0=part*2*W+2*warp;
    const unsigned lm=lane>>3,lr=lane&7;
    const unsigned base=(unsigned)__cvta_generic_to_shared(xs)+(((lm>>1)*8+lr)*M64S_STRIDE+(lm&1)*8)*2;
    float acc[2][TILES][4];
    #pragma unroll
    for(unsigned c=0;c<2;c++)
        #pragma unroll
        for(unsigned t=0;t<TILES;t++){acc[c][t][0]=acc[c][t][1]=acc[c][t][2]=acc[c][t][3]=0.f;}
    for(unsigned kc=0;kc<it;kc+=16) {
        const unsigned steps=min(16u,it-kc);
        __syncthreads();
        for(unsigned e=tid;e<ROWS*32;e+=W*32) {
            const unsigned j=e>>5,q=e&31;
            uint4 v=make_uint4(0,0,0,0);
            if(j<count && q*8<steps*16) v=*reinterpret_cast<const uint4*>(x+(unsigned long long)(d[2+j]/xdiv)*k+kc*16+q*8);
            *reinterpret_cast<uint4*>(xs+j*M64S_STRIDE+q*8)=v;
        }
        __syncthreads();
        for(unsigned s0=0;s0<steps;s0+=8) {
            unsigned long long win[2][8];
            #pragma unroll
            for(unsigned c=0;c<2;c++)
                #pragma unroll
                for(unsigned u=0;u<8;u++)
                    win[c][u]=lane_window(weights+((unsigned long long)(kc+s0+u)*nt+ot0+c)*32,lane);
            #pragma unroll
            for(unsigned u=0;u<8;u++) {
                const unsigned s=s0+u;
                unsigned a[2][4];
                #pragma unroll
                for(unsigned c=0;c<2;c++)
                    #pragma unroll
                    for(unsigned p=0;p<4;p++)a[c][p]=pair_values(win[c][u],p);
                #pragma unroll
                for(unsigned tp=0;tp<PAIRS;tp++) {
                    unsigned b[4];
                    m64_ldsm4(b,base+tp*16*M64S_STRIDE*2+s*32);
                    #pragma unroll
                    for(unsigned c=0;c<2;c++) {
                        m64_mma(acc[c][2*tp],a[c],b[0],b[1]);
                        if(2*tp+1<TILES) m64_mma(acc[c][2*tp+1],a[c],b[2],b[3]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for(unsigned c=0;c<2;c++) {
        const unsigned ot=ot0+c;
        #pragma unroll
        for(unsigned t=0;t<TILES;t++) {
            const unsigned j0=t*8+row,j1=j0+1;
            if(j0<count){const unsigned long long b=(unsigned long long)d[2+j0]*n+ot*16;y[b+col]=acc[c][t][0];y[b+col+8]=acc[c][t][2];}
            if(j1<count){const unsigned long long b=(unsigned long long)d[2+j1]*n+ot*16;y[b+col]=acc[c][t][1];y[b+col+8]=acc[c][t][3];}
        }
    }
}
template<unsigned W>
__device__ void project64u(const unsigned short* x,const unsigned* trellis,const unsigned* desc,
    const unsigned* group_count,float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs) {
    const unsigned it=k/16,nt=n/16,parts=nt/(2*W);
    const unsigned tasks=(*group_count)*parts;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned group=task/parts,part=task%parts;
        const unsigned* d=desc+group*66;
        const unsigned count=d[1];
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*32;
        switch((count+7)/8) {
            case 1: project64u_task<W,1>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 2: project64u_task<W,2>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 3: project64u_task<W,3>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 4: project64u_task<W,4>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 5: project64u_task<W,5>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 6: project64u_task<W,6>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 7: project64u_task<W,7>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            default: project64u_task<W,8>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) mcg_mma64u(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv) {
    __shared__ __align__(16) unsigned short xs[64*M64S_STRIDE];
    project64u<4>(x,t,desc,groups,y,k,n,xdiv,xs);
}
// R60 S4 (prefill v2): mcg_mma64u, software-pipelined. Inputs are staged 8 k steps (128 codes) at
// a time into two shared buffers with cp.async (zero-fill past count), the next chunk issued before
// the current one is computed; tile windows run one 4-step group ahead in two register buffers, so
// neither the L2 round trip of the inputs nor the DRAM latency of the weights is exposed per batch.
// Same arithmetic and MMA order as mcg_mma64u (bit-identical). Requires k % 128 == 0, (n/16) % 2W == 0.
#define M64V_STRIDE 136u
__device__ __forceinline__ void m64_cp16(unsigned dst,const void* src,unsigned bytes){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(bytes));
}
template<unsigned TILES>
__device__ __forceinline__ void m64v_steps(float (*acc)[TILES][4],const unsigned long long (*win)[4],unsigned base,unsigned s0){
    constexpr unsigned PAIRS=(TILES+1)/2;
    #pragma unroll
    for(unsigned u=0;u<4;u++) {
        unsigned a[2][4];
        #pragma unroll
        for(unsigned c=0;c<2;c++)
            #pragma unroll
            for(unsigned p=0;p<4;p++)a[c][p]=pair_values(win[c][u],p);
        #pragma unroll
        for(unsigned tp=0;tp<PAIRS;tp++) {
            unsigned b[4];
            m64_ldsm4(b,base+tp*16*M64V_STRIDE*2+(s0+u)*32);
            #pragma unroll
            for(unsigned c=0;c<2;c++) {
                m64_mma(acc[c][2*tp],a[c],b[0],b[1]);
                if(2*tp+1<TILES) m64_mma(acc[c][2*tp+1],a[c],b[2],b[3]);
            }
        }
    }
}
template<unsigned W,unsigned TILES>
__device__ __forceinline__ void project64v_task(const unsigned short* x,const unsigned* weights,const unsigned* d,
    float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs,unsigned part,unsigned count) {
    constexpr unsigned PAIRS=(TILES+1)/2,ROWS=PAIRS*16;
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,chunks=it/8;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned ot0=part*2*W+2*warp;
    const unsigned lm=lane>>3,lr=lane&7;
    const unsigned xs0=(unsigned)__cvta_generic_to_shared(xs);
    const unsigned lane_off=(((lm>>1)*8+lr)*M64V_STRIDE+(lm&1)*8)*2;
    const unsigned buf_bytes=64*M64V_STRIDE*2;
    const unsigned* w0=weights+(unsigned long long)ot0*32;
    float acc[2][TILES][4];
    #pragma unroll
    for(unsigned c=0;c<2;c++)
        #pragma unroll
        for(unsigned t=0;t<TILES;t++){acc[c][t][0]=acc[c][t][1]=acc[c][t][2]=acc[c][t][3]=0.f;}
    auto stage=[&](unsigned buf,unsigned chunk){
        for(unsigned e=tid;e<ROWS*16;e+=W*32) {
            const unsigned j=e>>4,q=e&15;
            const bool ok=j<count;
            const unsigned short* src=ok?x+(unsigned long long)(d[2+j]/xdiv)*k+chunk*128+q*8:x;
            m64_cp16(xs0+buf*buf_bytes+(j*M64V_STRIDE+q*8)*2,src,ok?16u:0u);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto load=[&](unsigned long long (*win)[4],unsigned s){
        #pragma unroll
        for(unsigned c=0;c<2;c++)
            #pragma unroll
            for(unsigned u=0;u<4;u++) win[c][u]=lane_window(w0+((unsigned long long)(s+u)*nt+c)*32,lane);
    };
    unsigned long long win0[2][4],win1[2][4];
    stage(0,0);
    load(win0,0);
    for(unsigned c=0;c<chunks;c++) {
        if(c+1<chunks){stage((c+1)&1,c+1);asm volatile("cp.async.wait_group 1;");}
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const unsigned base=xs0+(c&1)*buf_bytes+lane_off;
        load(win1,c*8+4);
        m64v_steps<TILES>(acc,win0,base,0);
        if(c+1<chunks) load(win0,c*8+8);
        m64v_steps<TILES>(acc,win1,base,4);
        __syncthreads();
    }
    #pragma unroll
    for(unsigned c=0;c<2;c++) {
        const unsigned ot=ot0+c;
        #pragma unroll
        for(unsigned t=0;t<TILES;t++) {
            const unsigned j0=t*8+row,j1=j0+1;
            if(j0<count){const unsigned long long b=(unsigned long long)d[2+j0]*n+ot*16;y[b+col]=acc[c][t][0];y[b+col+8]=acc[c][t][2];}
            if(j1<count){const unsigned long long b=(unsigned long long)d[2+j1]*n+ot*16;y[b+col]=acc[c][t][1];y[b+col+8]=acc[c][t][3];}
        }
    }
}
template<unsigned W>
__device__ void project64v(const unsigned short* x,const unsigned* trellis,const unsigned* desc,
    const unsigned* group_count,float* y,unsigned k,unsigned n,unsigned xdiv,unsigned short* xs) {
    const unsigned it=k/16,nt=n/16,parts=nt/(2*W);
    const unsigned tasks=(*group_count)*parts;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned group=task/parts,part=task%parts;
        const unsigned* d=desc+group*66;
        const unsigned count=d[1];
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*32;
        switch((count+7)/8) {
            case 1: project64v_task<W,1>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 2: project64v_task<W,2>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 3: project64v_task<W,3>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 4: project64v_task<W,4>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 5: project64v_task<W,5>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 6: project64v_task<W,6>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            case 7: project64v_task<W,7>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
            default: project64v_task<W,8>(x,weights,d,y,k,n,xdiv,xs,part,count); break;
        }
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(128) mcg_mma64v(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned xdiv) {
    __shared__ __align__(16) unsigned short xs[2*64*M64V_STRIDE];
    project64v<4>(x,t,desc,groups,y,k,n,xdiv,xs);
}
// R60 S4 (decode): mcg_project (kernels/mcg_project.cu, K4 / halfwords 64) with load-ahead. The
// scalar kernel issued one dependent 128-byte tile load per k step per warp and four 2-byte input
// loads; at the stage's decode geometry (4 k splits: ~10 steps per warp for gate/up, 2-3 for
// down) DRAM latency, not bandwidth, bounded it (~57% of peak). Here a warp issues the tile word
// (one coalesced 4-byte load per lane; the window's previous word comes by shuffle) and two
// 4-byte input loads for up to eight of its k steps before decoding any. Per-lane FMA order, the
// tail-biting window and the reduction are mcg_project's: bit-identical. Same ABI and grid.
__device__ __forceinline__ float m4p_half(unsigned short h){float x;asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h));return x;}
__device__ __forceinline__ unsigned short m4p_tohalf(float x){unsigned short h;asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x));return h;}
__device__ __forceinline__ float m4p_mcg(unsigned cw){
    unsigned p=(((cw&0x7fffu)*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    return m4p_half(m4p_tohalf(m4p_half((unsigned short)p)+m4p_half((unsigned short)(p>>16))));
}
template<unsigned P>
__device__ __forceinline__ void m4p_project(const unsigned short* x,const unsigned* trellis,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits){
    __shared__ float partial[64];
    const unsigned ot=blockIdx.x,slot=blockIdx.y,split=blockIdx.z;
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned begin=split*per,end=min(begin+per,it),row=(lane&3)*2,col=lane>>2;
    const unsigned* wbase=trellis+(unsigned long long)ids[slot]*it*nt*32+ot*32+lane;
    const unsigned short* xb=x+(unsigned long long)slot*k+row;
    float acc[8]={};
    for(unsigned t0=begin+warp;t0<end;t0+=4*P){
        unsigned w[P],x0[P],x1[P];
        #pragma unroll
        for(unsigned i=0;i<P;i++){
            const unsigned tk=t0+4*i;
            if(tk<end){
                w[i]=wbase[(unsigned long long)tk*nt*32];
                x0[i]=*reinterpret_cast<const unsigned*>(xb+tk*16);
                x1[i]=*reinterpret_cast<const unsigned*>(xb+tk*16+8);
            } else {w[i]=0;x0[i]=0;x1[i]=0;}
        }
        #pragma unroll
        for(unsigned i=0;i<P;i++){
            if(t0+4*i>=end) break;
            const unsigned prev=__shfl_sync(0xffffffffu,w[i],(lane+31)&31);
            const unsigned long long window=((unsigned long long)prev<<32)|w[i];
            const float a[4]={m4p_half((unsigned short)x0[i]),m4p_half((unsigned short)(x0[i]>>16)),
                              m4p_half((unsigned short)x1[i]),m4p_half((unsigned short)(x1[i]>>16))};
            #pragma unroll
            for(unsigned pair=0;pair<4;pair++){
                const unsigned cw=(unsigned)(window>>(24u-8u*pair));
                acc[pair*2]=__fmaf_rn(a[(pair*2)&3],m4p_mcg(cw>>4),acc[pair*2]);
                acc[pair*2+1]=__fmaf_rn(a[(pair*2+1)&3],m4p_mcg(cw),acc[pair*2+1]);
            }
        }
    }
    float lo=(acc[0]+acc[1])+(acc[2]+acc[3]),hi=(acc[4]+acc[5])+(acc[6]+acc[7]);
    lo+=__shfl_xor_sync(0xffffffffu,lo,1); hi+=__shfl_xor_sync(0xffffffffu,hi,1);
    lo+=__shfl_xor_sync(0xffffffffu,lo,2); hi+=__shfl_xor_sync(0xffffffffu,hi,2);
    if((lane&3)==0){partial[warp*16+col]=lo;partial[warp*16+col+8]=hi;}
    __syncthreads();
    if(tid<16){
        float sum=0;for(unsigned w2=0;w2<4;w2++)sum+=partial[w2*16+tid];
        y[((unsigned long long)slot*splits+split)*n+ot*16+tid]=sum;
    }
}
// P = k steps per warp loaded ahead: 8 (gate/up at 4 splits: ~10 steps per warp) or 4 with a
// 12-blocks-per-SM register cap (down: 2-3 steps per warp, latency-bound by block waves).
extern "C" __global__ void __launch_bounds__(128) mcg_project4v(const unsigned short* x,const unsigned* trellis,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits,unsigned halfwords){
    if(halfwords==64) m4p_project<8>(x,trellis,ids,y,k,n,splits);
}
extern "C" __global__ void __launch_bounds__(128,12) mcg_project4v4(const unsigned short* x,const unsigned* trellis,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits,unsigned halfwords){
    if(halfwords==64) m4p_project<4>(x,trellis,ids,y,k,n,splits);
}
// P = 10: the gate/up projections at 4 k splits have exactly 10 steps per warp (k 2560): one batch.
extern "C" __global__ void __launch_bounds__(128) mcg_project4v10(const unsigned short* x,const unsigned* trellis,
    const unsigned* ids,float* y,unsigned k,unsigned n,unsigned splits,unsigned halfwords){
    if(halfwords==64) m4p_project<10>(x,trellis,ids,y,k,n,splits);
}
