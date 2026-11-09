// R10c: correctness-first DFlash2 GQA with split-context online softmax.
// BF16 q/k/v, FP32 accumulation; one warp per query-head and context partition.
typedef unsigned short bf;
__device__ __forceinline__ float ld(bf x){return __uint_as_float((unsigned)x<<16);}
__device__ __forceinline__ bf st(float x){unsigned u=__float_as_uint(x);return (bf)((u+0x7fffU+((u>>16)&1U))>>16);}
__device__ __forceinline__ float warp_sum(float x){
  for(int m=16;m;m>>=1)x+=__shfl_xor_sync(0xffffffffU,x,m);return x;
}
// control = {rows, context_rows, context_base, anchor, window}; shape bucket fixed.
// NeoX rotary: pair i with i+64, AFTER per-head weighted RMS rounded to BF16.
extern "C" __global__ void dflash_qk_rope(const bf* q,const bf* k,const bf* qw,
    const bf* kw,const float* freq,bf* qo,bf* ko,const unsigned* ctl){
  const int MAX_ROWS=16,QH=32,KH=8,D=128;
  int g=blockIdx.x,lane=threadIdx.x;
  bool isq=g<MAX_ROWS*QH;
  int a=isq?g:g-MAX_ROWS*QH,heads=isq?QH:KH,row=a/heads;
  if(row>=(int)ctl[0])return;
  const bf* x=(isq?q:k)+a*D;const bf* w=isq?qw:kw;bf* out=(isq?qo:ko)+a*D;
  float sum=0;
  for(int c=lane;c<D;c+=32){float v=ld(x[c]);sum+=v*v;}
  float inv=rsqrtf(warp_sum(sum)/128.0f+1e-6f);
  for(int c=lane;c<64;c+=32){
    float u=ld(st(ld(x[c])*inv*ld(w[c]))),v=ld(st(ld(x[c+64])*inv*ld(w[c+64])));
    float angle=(float)(ctl[3]+row)*freq[c],s,cosa;sincosf(angle,&s,&cosa);
    out[c]=st(u*cosa-v*s);out[c+64]=st(u*s+v*cosa);
  }
}
// Persistent context already has norm/RoPE applied when its K rows were appended.
// Block keys are separate; full future visibility inside the drafted noise block.
extern "C" __global__ void dflash_attn_part(const bf* q,const bf* bk,const bf* bv,
    const bf* ck,const bf* cv,float* part,const unsigned* ctl,int splits){
  int qh=blockIdx.x,sp=blockIdx.y,lane=threadIdx.x;
  int row=qh/32,head=qh%32,kh=head/4,rows=ctl[0],ctx=ctl[1];
  if(row>=rows)return;
  int keys=ctx+rows,chunk=(keys+splits-1)/splits,lo=sp*chunk,hi=min(keys,lo+chunk);
  float qv[4],out[4]={0,0,0,0};
  for(int j=0;j<4;++j)qv[j]=ld(q[qh*128+lane+j*32]);
  float m=-__int_as_float(0x7f800000),z=0;
  for(int key=lo;key<hi;++key){
    int pos=key<ctx?(int)ctl[2]+key:(int)ctl[3]+key-ctx;
    if(abs((int)ctl[3]+row-pos)>=(int)ctl[4])continue;
    const bf* kp=(key<ctx?ck:bk)+((key<ctx?key:key-ctx)*8+kh)*128;
    const bf* vp=(key<ctx?cv:bv)+((key<ctx?key:key-ctx)*8+kh)*128;
    float dot=0;
    for(int j=0;j<4;++j)dot=fmaf(qv[j],ld(kp[lane+j*32]),dot);
    float score=warp_sum(dot)*0.08838834764831845f;
    float next=fmaxf(m,score),a=expf(m-next),b=expf(score-next);
    z=z*a+b;
    for(int j=0;j<4;++j)out[j]=fmaf(b,ld(vp[lane+j*32]),out[j]*a);
    m=next;
  }
  float* p=part+(qh*splits+sp)*130;
  for(int j=0;j<4;++j)p[lane+j*32]=out[j];
  if(lane==0){p[128]=m;p[129]=z;}
}
extern "C" __global__ void dflash_attn_merge(const float* part,bf* y,
    const unsigned* ctl,int splits){
  int qh=blockIdx.x,c=threadIdx.x;
  if(qh/32>=(int)ctl[0])return;
  const float* p=part+qh*splits*130;
  float m=-__int_as_float(0x7f800000);
  for(int s=0;s<splits;++s)if(p[s*130+129]>0)m=fmaxf(m,p[s*130+128]);
  float total=0,z=0;
  for(int s=0;s<splits;++s)if(p[s*130+129]>0){
    float a=expf(p[s*130+128]-m);z+=a*p[s*130+129];total=fmaf(a,p[s*130+c],total);
  }
  y[qh*128+c]=st(z>0?total/z:0);
}
