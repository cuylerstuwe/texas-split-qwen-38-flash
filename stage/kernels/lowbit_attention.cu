// Native CUDA Qwen gated attention + QSA. Semantic boundaries follow the
// qualified Rust/Metal ports of Sushi/MLX (MIT, NOTICE.txt). Resident CUDA ABI,
// bounded radix selector and prefix-tail restore are independent orchestration.
typedef unsigned short bf;
typedef unsigned int uint;
__device__ __forceinline__ float ld(bf x){return __uint_as_float((uint)x<<16);}
__device__ __forceinline__ bf st(float x){uint u=__float_as_uint(x);if((u&0x7fffffffU)>0x7f800000U)return 0x7fc0;return (bf)((u+0x7fffU+((u>>16)&1U))>>16);}
__device__ __forceinline__ float sum(float x){for(int m=16;m;m>>=1)x+=__shfl_xor_sync(0xffffffffU,x,m);return x;}
__device__ __forceinline__ float wmin(float x){for(int m=16;m;m>>=1)x=fminf(x,__shfl_xor_sync(0xffffffffU,x,m));return x;}
__device__ __forceinline__ float wmax(float x){for(int m=16;m;m>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffU,x,m));return x;}

extern "C" __global__ void attn_prepare(const bf* qg,const bf* k,const bf* qw,const bf* kw,
    const float* freq,bf* query,bf* key,bf* gate,uint offset,uint base){
  uint r=blockIdx.y,h=blockIdx.x,lane=threadIdx.x;bool isq=h<24;uint kh=isq?h:h-24;
  const bf* src=isq?qg+(r*24+h)*512:k+(r*2+kh)*256;const bf* norm=isq?qw:kw;
  float x[8],ss=0;
  for(uint j=0;j<8;j++){x[j]=ld(src[lane*8+j]);ss+=x[j]*x[j];}
  float inv=rsqrtf(sum(ss)/256.f+1e-6f);
  for(uint j=0;j<8;j++)x[j]=ld(st(ld(st(x[j]*inv))*ld(norm[lane*8+j])));
  for(uint j=0;j<8;j++){
    uint d=lane*8+j;float other=__shfl_xor_sync(0xffffffffU,x[j],4),y=x[j];
    if(d<64){float s,c;sincosf((float)(base+offset+r)*freq[d%32],&s,&c);y=x[j]*c+(d<32?-other:other)*s;}
    if(isq){query[(r*24+h)*256+d]=st(y);gate[(r*24+h)*256+d]=src[256+d];}
    else key[(r*2+kh)*256+d]=st(y);
  }
}
extern "C" __global__ void attn_pack(const bf* key,const bf* value,unsigned char* kc,bf* ks,bf* kb,
    unsigned char* vc,bf* vs,bf* vb,uint offset){
  uint g=blockIdx.x,lane=threadIdx.x;bool isk=blockIdx.y==0;const bf* in=isk?key:value;
  float a=ld(in[g*64+2*lane]),b=ld(in[g*64+2*lane+1]);
  float lo=wmin(fminf(a,b)),hi=wmax(fmaxf(0.f,fmaxf(a,b)));
  float scale=fmaxf((hi-lo)/255.f,1e-7f);bool side=fabsf(lo)>fabsf(hi);scale=side?scale:-scale;
  float edge=side?lo:hi,q0=roundf(edge/scale);bool zero=q0==0.f;
  scale=zero?scale:edge/q0;float bias=zero?0.f:edge;
  unsigned char* codes=isk?kc:vc;bf* scales=isk?ks:vs;bf* biases=isk?kb:vb;
  uint out=offset*8+g;
  if(lane==0){scales[out]=st(scale);biases[out]=st(bias);}
  codes[out*64+2*lane]=(unsigned char)fminf(roundf((a-bias)/scale),255.f);
  codes[out*64+2*lane+1]=(unsigned char)fminf(roundf((b-bias)/scale),255.f);
}
__device__ __forceinline__ bf raw_key(const bf* qk,const bf* tail,uint rem,uint row,uint d){
  return row<rem?tail[row*128+d]:qk[(row-rem)*640+512+d];
}
extern "C" __global__ void qsa_pool(const bf* qk,const bf* tail,const bf* norm,const float* freq,
    bf* pooled,uint offset,uint base){
  uint b=blockIdx.x,lane=threadIdx.x,old=offset/4,rem=offset%4;float x[4],ss=0;
  for(uint j=0;j<4;j++){uint d=lane*4+j;float m=0;for(uint t=0;t<4;t++)m+=ld(raw_key(qk,tail,rem,4*b+t,d));x[j]=ld(st(m*.25f));ss+=x[j]*x[j];}
  float inv=rsqrtf(sum(ss)/128.f+1e-6f);
  for(uint j=0;j<4;j++)x[j]=ld(st(ld(st(x[j]*inv))*ld(norm[lane*4+j])));
  for(uint j=0;j<4;j++){
    uint d=lane*4+j;float other=__shfl_xor_sync(0xffffffffU,x[j],8),y=x[j];
    if(d<64){float s,c;sincosf((float)(base+(old+b)*4)*freq[d%32],&s,&c);c=ld(st(c));s=ld(st(s));y=ld(st(x[j]*c))+ld(st((d<32?-other:other)*s));}
    pooled[(old+b)*128+d]=st(y);
  }
}
extern "C" __global__ void qsa_query(const bf* qk,const bf* norm,const float* freq,bf* query,uint offset,uint base){
  uint h=blockIdx.x,r=blockIdx.y,lane=threadIdx.x;float x[4],ss=0;
  for(uint j=0;j<4;j++){x[j]=ld(qk[r*640+h*128+lane*4+j]);ss+=x[j]*x[j];}
  float inv=rsqrtf(sum(ss)/128.f+1e-6f);
  for(uint j=0;j<4;j++)x[j]=ld(st(ld(st(x[j]*inv))*ld(norm[lane*4+j])));
  for(uint j=0;j<4;j++){
    uint d=lane*4+j;float other=__shfl_xor_sync(0xffffffffU,x[j],8),y=x[j];
    if(d<64){float s,c;sincosf((float)(base+offset+r)*freq[d%32],&s,&c);y=x[j]*c+(d<32?-other:other)*s;}
    query[(r*4+h)*128+d]=st(y);
  }
}
extern "C" __global__ void qsa_score(const bf* query,const bf* pooled,float* scores,uint nb){
  uint b=blockIdx.x*4+threadIdx.x/32,r=blockIdx.y,lane=threadIdx.x%32;if(b>=nb)return;
  float k[4];for(uint j=0;j<4;j++)k[j]=ld(pooled[b*128+lane*4+j]);float score=0;
  for(uint h=0;h<4;h++){float dot=0;for(uint j=0;j<4;j++)dot+=ld(query[(r*4+h)*128+lane*4+j])*k[j];score+=fmaxf(sum(dot),0.f);}
  if(lane==0)scores[r*nb+b]=score;
}
__device__ __forceinline__ uint ordered(float v){
  if(isnan(v))return 0xffffffffU;if(v==0.f)return 0x80000000U;uint u=__float_as_uint(v);return u&0x80000000U?~u:u|0x80000000U;
}
// Block-wide exclusive scan, 256 participating threads. Caller barriers protect
// shared reuse. No grid synchronization or co-residency assumption is needed.
__device__ uint scan(uint v,uint* shared){
  uint x=v,lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(uint k=1;k<32;k<<=1){uint p=__shfl_up_sync(0xffffffffU,x,k);if(lane>=k)x+=p;}
  if(lane==31)shared[warp]=x;__syncthreads();
  uint off=0,total=0;for(uint w=0;w<8;w++){total+=shared[w];if(w<warp)off+=shared[w];}
  if(threadIdx.x==0)shared[8]=total;__syncthreads();return off+x-v;
}
extern "C" __global__ void qsa_select(const float* scores,uint* ids,uint offset,uint rows){
  uint r=blockIdx.y,tid=threadIdx.x,vb=(offset+r+1)/4,nb=(offset+rows)/4;
  uint* out=ids+r*512;const float* sc=scores+r*nb;
  if(vb<=512){for(uint i=tid;i<512;i+=256)out[i]=i<vb?i:0x7fffffffU;return;}
  __shared__ uint hist[256],choice[2],prefixes[9];
  uint prefix=0,need=512,fixed=0;
  for(uint pass=0;pass<4;pass++){
    hist[tid]=0;__syncthreads();uint last=0xffffffffU,run=0,shift=24-pass*8;
    for(uint i=tid;i<vb;i+=256){uint u=ordered(sc[i]);if(fixed && (u>>(32-fixed))!=prefix)continue;uint d=(u>>shift)&255;
      if(d==last)run++;else{if(run)atomicAdd(hist+last,run);last=d;run=1;}}
    if(run)atomicAdd(hist+last,run);__syncthreads();
    if(tid==0){uint above=0;for(int d=255;d>=0;d--){uint c=hist[d];if(above+c>=need){choice[0]=(uint)d;choice[1]=above;break;}above+=c;}}
    __syncthreads();prefix=(prefix<<8)|choice[0];need-=choice[1];fixed+=8;__syncthreads();
  }
  uint run_eq=0,run_keep=0;
  for(uint start=0;start<vb;start+=256){uint i=start+tid,u=i<vb?ordered(sc[i]):0;
    uint eq=(i<vb && u==prefix),before_eq=scan(eq,prefixes),total_eq=prefixes[8];__syncthreads();
    uint keep=(i<vb && (u>prefix || (eq && run_eq+before_eq<need)));
    uint before=scan(keep,prefixes),total=prefixes[8];__syncthreads();
    if(keep && run_keep+before<512)out[run_keep+before]=i;
    run_eq+=total_eq;run_keep+=total;__syncthreads();
  }
}
__device__ __forceinline__ float unpack(const unsigned char* c,const bf* s,const bf* b,uint i){return ld(st((float)c[i]*ld(s[i/64])+ld(b[i/64])));}
extern "C" __global__ void attn_part(const bf* query,const unsigned char* kc,const unsigned char* vc,
    const bf* ks,const bf* kb,const bf* vs,const bf* vb,const uint* ids,float* partial,float* stats,
    uint offset,uint sparse,uint splits){
  uint h=blockIdx.x,r=blockIdx.y,sp=blockIdx.z,lane=threadIdx.x,kh=h/12,pos=offset+r,complete=(pos+1)/4;
  uint selected=sparse?min(complete,512U)*4:0,tail=sparse?complete*4:0,L=selected+pos+1-tail;
  uint per=(L+splits-1)/splits,lo=sp*per,hi=min(L,lo+per);
  float q[8],acc[8];for(uint j=0;j<8;j++){q[j]=ld(query[(r*24+h)*256+lane*8+j]);acc[j]=0;}
  float m=-__int_as_float(0x7f800000),z=0;
  for(uint vi=lo;vi<hi;vi++){
    uint key=vi<selected?ids[r*512+vi/4]*4+vi%4:tail+vi-selected,at=(key*2+kh)*256+lane*8;
    float dot=0;for(uint j=0;j<8;j++)dot+=q[j]*unpack(kc,ks,kb,at+j);dot=sum(dot)*(1.f/16.f);
    float nm=fmaxf(m,dot),factor=expf(m-nm),prob=expf(dot-nm);z=z*factor+prob;m=nm;
    for(uint j=0;j<8;j++)acc[j]=acc[j]*factor+prob*unpack(vc,vs,vb,at+j);
  }
  uint idx=(r*24+h)*splits+sp;if(lane==0){stats[idx*2]=m;stats[idx*2+1]=z;}
  for(uint j=0;j<8;j++)partial[idx*256+lane*8+j]=acc[j];
}
extern "C" __global__ void attn_merge(const float* partial,const float* stats,bf* output,uint splits){
  uint idx=(blockIdx.y*24+blockIdx.x)*splits,d=threadIdx.x;float m=-__int_as_float(0x7f800000);
  for(uint sp=0;sp<splits;sp++)if(stats[(idx+sp)*2+1]>0)m=fmaxf(m,stats[(idx+sp)*2]);
  float z=0,y=0;for(uint sp=0;sp<splits;sp++){float mass=stats[(idx+sp)*2+1];if(mass>0){float w=expf(stats[(idx+sp)*2]-m);z+=mass*w;y+=partial[(idx+sp)*256+d]*w;}}
  output[(blockIdx.y*24+blockIdx.x)*256+d]=st(y/z);
}
extern "C" __global__ void attn_gate(const bf* attention,const bf* gate,bf* output,uint count){
  uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float g=ld(gate[i]);
  float e=ld(st(expf(fabsf(g)))),den=ld(st(1.f+e)),small=ld(st(1.f/den));
  float sigmoid=g<0.f?small:ld(st(1.f-small));output[i]=st(ld(attention[i])*sigmoid);
}
extern "C" __global__ void qsa_commit_tail(const bf* qk,const bf* tail,bf* next,uint offset,uint accepted){
  uint i=blockIdx.x*blockDim.x+threadIdx.x,n=(offset+accepted)%4,rem=offset%4;
  if(i<n*128){uint row=rem+accepted-n+i/128;next[i]=raw_key(qk,tail,rem,row,i%128);}
}
extern "C" __global__ void qsa_restore_tail(const bf* next,bf* tail){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<384)tail[i]=next[i];}
