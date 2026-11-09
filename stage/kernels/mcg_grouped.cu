// K1.75 packed projection with device-owned grouping and shared weight decode
// across up to4 rows of one expert. MCG mapping/reduction follows mcg_project.cu
// and its licensed Sushi-derived baseline (NOTICE.txt). No host route readback.
__device__ __forceinline__ float hf(unsigned short h) {
    float x; asm("cvt.f32.f16 %0, %1;":"=f"(x):"h"(h)); return x;
}
__device__ __forceinline__ unsigned short fh(float x) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;":"=h"(h):"f"(x)); return h;
}
__device__ __forceinline__ float mcg(unsigned cw) {
    unsigned p=(((cw&0x7fffu)*0xcbac1fedu)&0x8fff8fffu)^0x3b603b60u;
    return hf(fh(hf((unsigned short)p)+hf((unsigned short)(p>>16))));
}
// One128-thread block. Desc=[expert,count,slot0,slot1,slot2,slot3].
// Every route slot appears exactly once. No atomics and no per-layer CPU work.
extern "C" __global__ void mcg_group_rows(const unsigned* ids,unsigned* desc,
    unsigned* group_count,unsigned slots,unsigned experts,unsigned batch) {
    __shared__ unsigned counts[128];
    const unsigned e=threadIdx.x;
    unsigned count=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)count+=ids[s]==e;
    counts[e]=count;__syncthreads();
    unsigned prefix=0;
    for(unsigned j=0;j<e;j++)prefix+=(counts[j]+batch-1)/batch;
    const unsigned chunks=(count+batch-1)/batch;
    if(e==127)*group_count=prefix+chunks;
    for(unsigned c=0;c<chunks;c++) {
        unsigned* d=desc+(prefix+c)*6;
        d[0]=e;d[1]=min(batch,count-c*batch);
        for(unsigned j=0;j<4;j++)d[2+j]=0xffffffffu;
    }
    unsigned at=0;
    if(e<experts)for(unsigned s=0;s<slots;s++)if(ids[s]==e) {
        desc[(prefix+at/batch)*6+2+at%batch]=s;at++;
    }
}
template<unsigned B>
__device__ void project(const unsigned short* x,const unsigned* trellis,
    const unsigned* desc,const unsigned* group_count,float* y,
    unsigned k,unsigned n,unsigned splits,float* partial) {
    const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned it=k/16,nt=n/16,per=(it+splits-1)/splits;
    const unsigned row=(lane&3)*2,col=lane>>2;
    const unsigned tasks=(*group_count)*nt*splits;
    for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const unsigned ot=task%nt,split=(task/nt)%splits,group=task/(nt*splits);
        const unsigned* d=desc+group*6;
        const unsigned count=d[1],begin=split*per,end=min(begin+per,it);
        const unsigned* weights=trellis+(unsigned long long)d[0]*it*nt*14;
        float acc[B][8]={};
        for(unsigned tk=begin+warp;tk<end;tk+=4) {
            const unsigned* words=weights+((unsigned long long)tk*nt+ot)*14;
            float a[B][4];
            #pragma unroll
            for(unsigned r=0;r<B;r++)if(r<count) {
                const unsigned short* xx=x+(unsigned long long)d[2+r]*k+tk*16;
                a[r][0]=hf(xx[row]);a[r][1]=hf(xx[row+1]);
                a[r][2]=hf(xx[row+8]);a[r][3]=hf(xx[row+9]);
            }
            #pragma unroll
            for(unsigned pair=0;pair<4;pair++) {
                const unsigned t=lane*8+pair*2,e0=(t+1)*28/16,e1=(t+2)*28/16;
                const unsigned first=(e0+448-16)/32,last=(e1+448-1)/32;
                const unsigned shift=(last+1)*32-(e1+448);
                const unsigned long long bits=((unsigned long long)words[first%14]<<32)|words[last%14];
                const unsigned cw=(unsigned)(bits>>shift);
                const float w0=mcg(cw>>(e1-e0)),w1=mcg(cw);
                #pragma unroll
                for(unsigned r=0;r<B;r++)if(r<count) {
                    acc[r][pair*2]=__fmaf_rn(a[r][(pair*2)&3],w0,acc[r][pair*2]);
                    acc[r][pair*2+1]=__fmaf_rn(a[r][(pair*2+1)&3],w1,acc[r][pair*2+1]);
                }
            }
        }
        #pragma unroll
        for(unsigned r=0;r<B;r++)if(r<count) {
            float lo=(acc[r][0]+acc[r][1])+(acc[r][2]+acc[r][3]);
            float hi=(acc[r][4]+acc[r][5])+(acc[r][6]+acc[r][7]);
            lo+=__shfl_xor_sync(0xffffffffu,lo,1);hi+=__shfl_xor_sync(0xffffffffu,hi,1);
            lo+=__shfl_xor_sync(0xffffffffu,lo,2);hi+=__shfl_xor_sync(0xffffffffu,hi,2);
            if((lane&3)==0){partial[r*64+warp*16+col]=lo;partial[r*64+warp*16+col+8]=hi;}
        }
        __syncthreads();
        if(tid<16) {
            #pragma unroll
            for(unsigned r=0;r<B;r++)if(r<count) {
                float sum=0;for(unsigned w=0;w<4;w++)sum+=partial[r*64+w*16+tid];
                y[((unsigned long long)d[2+r]*splits+split)*n+ot*16+tid]=sum;
            }
        }
        __syncthreads(); // no next-task overwrite while a prior reader remains
    }
}
extern "C" __global__ void mcg_project_grouped(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned batch) {
    __shared__ float partial[4*64];
    if(batch==1)project<1>(x,t,desc,groups,y,k,n,splits,partial);
    else if(batch==2)project<2>(x,t,desc,groups,y,k,n,splits,partial);
    else if(batch==4)project<4>(x,t,desc,groups,y,k,n,splits,partial);
}

// Separate entries keep the one-row control's register allocation independent
// from the four-row arm. The host chooses an entry once when recording a plan.
extern "C" __global__ void mcg_project_grouped_b1(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned batch) {
    __shared__ float partial[64];
    if(batch==1)project<1>(x,t,desc,groups,y,k,n,splits,partial);
}
extern "C" __global__ void mcg_project_grouped_b2(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned batch) {
    __shared__ float partial[2*64];
    if(batch==2)project<2>(x,t,desc,groups,y,k,n,splits,partial);
}
extern "C" __global__ void mcg_project_grouped_b4(const unsigned short* x,const unsigned* t,
    const unsigned* desc,const unsigned* groups,float* y,unsigned k,unsigned n,unsigned splits,unsigned batch) {
    __shared__ float partial[4*64];
    if(batch==4)project<4>(x,t,desc,groups,y,k,n,splits,partial);
}
