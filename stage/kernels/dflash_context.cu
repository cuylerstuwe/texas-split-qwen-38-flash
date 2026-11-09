// Committed DFlash2 context update, independent source/destination banks.
// Fixed maximum 16 new rows, 2047 retained rows. Host validates all control.
typedef unsigned short bf;
__device__ __forceinline__ float ld(bf x){return __uint_as_float((unsigned)x<<16);}
__device__ __forceinline__ bf st(float x){unsigned u=__float_as_uint(x);return (bf)((u+0x7fffU+((u>>16)&1U))>>16);}
__device__ __forceinline__ float sumwarp(float x){for(int m=16;m;m>>=1)x+=__shfl_xor_sync(0xffffffffU,x,m);return x;}
// ctl={new_rows,old_len,drop,first_absolute_pos,new_len,new_base}.
extern "C" __global__ void dflash_context_append(const bf* k,const bf* v,
    const bf* kw,const float* freq,const bf* oldk,const bf* oldv,
    bf* nextk,bf* nextv,bf* rotated,const unsigned* ctl){
  int g=blockIdx.x,lane=threadIdx.x,row=g/8,head=g%8;
  if(row>=(int)ctl[4])return;
  int retained=(int)ctl[1]-(int)ctl[2];
  if(row<retained){
    int src=((row+(int)ctl[2])*8+head)*128;
    for(int c=lane;c<128;c+=32){nextk[g*128+c]=oldk[src+c];nextv[g*128+c]=oldv[src+c];}
    return;
  }
  int r=row-retained,a=r*8+head;
  const bf* x=k+a*128;
  float sum=0;
  for(int c=lane;c<128;c+=32){float t=ld(x[c]);sum+=t*t;}
  float inv=rsqrtf(sumwarp(sum)/128.0f+1e-6f);
  for(int c=lane;c<64;c+=32){
    float u=ld(st(ld(x[c])*inv*ld(kw[c]))),z=ld(st(ld(x[c+64])*inv*ld(kw[c+64])));
    float angle=(float)(ctl[3]+r)*freq[c],s,co;sincosf(angle,&s,&co);
    bf p=st(u*co-z*s),q=st(u*s+z*co);
    nextk[g*128+c]=rotated[a*128+c]=p;
    nextk[g*128+c+64]=rotated[a*128+c+64]=q;
  }
  for(int c=lane;c<128;c+=32)nextv[g*128+c]=v[a*128+c];
}
