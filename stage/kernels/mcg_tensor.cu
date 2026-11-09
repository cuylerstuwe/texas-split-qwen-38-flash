// R22: unchanged K1.75 packed values, device grouping and decode-once MMA.
// MCG bitstream/permutation follows the licensed Sushi-derived mcg_project.cu.
// New grouping512, FP16x2 decode and MMA organization; see NOTICE.txt.
__device__ __forceinline__ float hf(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
// Every lane's eight weights advance14bits. One29-bit tail-biting window
// covers all eight15-bit codewords, so two adjacent words suffice for the lane.
__device__ __forceinline__ unsigned lane_window(const unsigned* words,unsigned lane) {
    const unsigned end=lane*14+14,last=(end-1)>>5;
    const unsigned prev=last?last-1:13,shift=(last+1)*32-end;
    const unsigned long long bits=((unsigned long long)words[prev]<<32)|words[last];
    return (unsigned)(bits>>shift);
}
__device__ __forceinline__ unsigned pair_values(unsigned window,unsigned pair) {
    const unsigned offset=pair==0?11:pair==1?7:pair==2?4:0;
    const unsigned cw=window>>offset;
    const unsigned c0=(cw>>2)&0x7fffu,c1=cw&0x7fffu;
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
        const unsigned* weights=trellis+(unsigned long long)expert*it*nt*14;
        if(Mode==1 || (Mode==2 && count>1)) {
            float acc[4]={};
            for(unsigned tk=begin+warp;tk<end;tk+=W) {
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
                const unsigned window=lane_window(words,lane);
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
                const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
                const unsigned window=lane_window(words,lane);
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
