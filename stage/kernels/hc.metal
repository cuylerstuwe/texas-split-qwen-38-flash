// Native HC kernels. Normalization/inject partials and optional pending write
// follow Sushi qwen4_hc.zig HC_FUSED_N_SOURCE (MIT; see NOTICE.txt).
// Fixed Qwen geometry, native BF16, independent dense operations and Rust ABI.
#include <metal_stdlib>
using namespace metal;
constant uint H=2560,HC=4,F=10240,R=320;
struct HcParams {uint rows,write;};
kernel void hc_norm(const device bfloat* x [[buffer(0)]],
 const device bfloat* nw [[buffer(1)]],const device bfloat* iw [[buffer(2)]],
 const device bfloat* block [[buffer(3)]],const device bfloat* inject [[buffer(4)]],
 device bfloat* normalized [[buffer(5)]],device float* parts [[buffer(6)]],
 device bfloat* next [[buffer(7)]],constant HcParams& p [[buffer(8)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint h=g.x,row=g.y,sg=tid/32,lane=tid%32;
 threadgroup float sums[8],dots[8*4];
 float values[10],sum=0.;
 for(uint i=0;i<10;i++) {
   uint j=tid+256*i,at=row*F+h*H+j;
   bfloat v=x[at];
   if(p.write) {
     v=bfloat(float(v)+float(bfloat(float(block[row*H+j])*float(inject[row*HC+h]))));
     next[at]=v;
   }
   values[i]=float(v);sum+=values[i]*values[i];
 }
 sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float total=0.;for(uint i=0;i<8;i++)total+=sums[i];
 float rs=rsqrt(total/float(H)+1e-6f),dot[4]={};
 for(uint i=0;i<10;i++) {
   uint j=h*H+tid+256*i;
   bfloat v=bfloat(float(bfloat(values[i]*rs))*float(nw[j]));
   normalized[row*F+j]=v;
   for(uint c=0;c<HC;c++)dot[c]+=float(v)*float(iw[c*F+j]);
 }
 for(uint c=0;c<HC;c++) {float v=simd_sum(dot[c]);if(lane==0)dots[sg*HC+c]=v;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<HC) {float v=0.;for(uint s=0;s<8;s++)v+=dots[s*HC+tid];parts[(row*HC+h)*HC+tid]=v;}
}
kernel void hc_activate(const device bfloat* down [[buffer(0)]],
 const device float* parts [[buffer(1)]],device bfloat* act [[buffer(2)]],
 device bfloat* inject [[buffer(3)]],constant HcParams& p [[buffer(8)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*(R+HC))return;
 uint row=i/(R+HC),j=i%(R+HC);
 if(j<R) {
   bfloat v=down[row*R+j],s=bfloat(1.f/(1.f+exp(-float(v))));
   act[row*R+j]=bfloat(float(v)*float(s));
 } else {
   uint c=j-R;float raw=0.;for(uint h=0;h<HC;h++)raw+=parts[(row*HC+h)*HC+c];
   bfloat v=bfloat(raw),s=bfloat(1.f/(1.f+exp(-float(v))));
   inject[row*HC+c]=bfloat(2.f*float(s));
 }
}
kernel void hc_mix(const device bfloat* up [[buffer(0)]],
 const device bfloat* normalized [[buffer(1)]],device bfloat* mixed [[buffer(2)]],
 constant HcParams& p [[buffer(8)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*H)return;
 uint row=i/H,j=i%H;float sum=0.;
 for(uint h=0;h<HC;h++) {
   uint at=row*F+h*H+j;
   bfloat s=bfloat(1.f/(1.f+exp(-float(up[at]))));
   sum+=float(bfloat(float(s)*float(normalized[at])));
 }
 mixed[i]=bfloat(float(bfloat(sum))*0.25f);
}
kernel void hc_write(const device bfloat* x [[buffer(0)]],
 const device bfloat* block [[buffer(1)]],const device bfloat* inject [[buffer(2)]],
 device bfloat* next [[buffer(3)]],constant HcParams& p [[buffer(8)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*F)return;
 uint row=i/F,h=i%F/H,j=i%H;
 next[i]=bfloat(float(x[i])+float(bfloat(float(block[row*H+j])*float(inject[row*HC+h]))));
}
// R55 fused reductions (bit-identical to frag_reduce + hc_activate / hc_mix): the split-K
// partials [row][split][n] are summed in split order, then consumed in the same kernel.
struct HcFuseParams {uint rows,splits;};
kernel void hc_act_reduce(const device float* partial [[buffer(0)]],
 const device float* parts [[buffer(1)]],device bfloat* act [[buffer(2)]],
 device bfloat* inject [[buffer(3)]],device float* dfull [[buffer(4)]],
 device bfloat* dbf [[buffer(5)]],constant HcFuseParams& p [[buffer(8)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*(R+HC))return;
 uint row=i/(R+HC),j=i%(R+HC);
 if(j<R) {
   float v=0.;for(uint s=0;s<p.splits;s++)v+=partial[(row*p.splits+s)*R+j];
   dfull[row*R+j]=v;bfloat b=bfloat(v);dbf[row*R+j]=b;
   bfloat s=bfloat(1.f/(1.f+exp(-float(b))));
   act[row*R+j]=bfloat(float(b)*float(s));
 } else {
   uint c=j-R;float raw=0.;for(uint h=0;h<HC;h++)raw+=parts[(row*HC+h)*HC+c];
   bfloat v=bfloat(raw),s=bfloat(1.f/(1.f+exp(-float(v))));
   inject[row*HC+c]=bfloat(2.f*float(s));
 }
}
kernel void hc_mix_reduce(const device float* partial [[buffer(0)]],
 const device bfloat* normalized [[buffer(1)]],device bfloat* mixed [[buffer(2)]],
 device float* ufull [[buffer(3)]],device bfloat* ubf [[buffer(4)]],
 constant HcFuseParams& p [[buffer(8)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*H)return;
 uint row=i/H,j=i%H;float sum=0.;
 for(uint h=0;h<HC;h++) {
   uint at=row*F+h*H+j,col=h*H+j;
   float u=0.;for(uint s=0;s<p.splits;s++)u+=partial[(row*p.splits+s)*F+col];
   ufull[at]=u;bfloat ub=bfloat(u);ubf[at]=ub;
   bfloat sg=bfloat(1.f/(1.f+exp(-float(ub))));
   sum+=float(bfloat(float(sg)*float(normalized[at])));
 }
 mixed[i]=bfloat(float(bfloat(sum))*0.25f);
}
// R56 fused HC up projection + mix (<=16 rows; new code). The affine8 up GEMV (k=R, n=F) uses
// the R55 RMV per-lane arithmetic (lane l owns 8 consecutive K of each 256-wide step,
// y += s*dot + b*sum in FP32, one simd_sum per output); each simdgroup computes the 4 stream
// columns h*H+j of one position j and applies hc_mix_reduce's mix to them in the same kernel.
// No split-K partials and no separate mix pass. Values differ from the fragment GEMV path.
struct HcUpMixParams {uint rows,group;};
template<uint RR,uint SG>
inline void hc_up_mix_body(const device bfloat* act,const device uchar* w,
 const device bfloat* scales,const device bfloat* biases,const device bfloat* normalized,
 device bfloat* mixed,constant HcUpMixParams& p,uint3 grid,uint tid) {
 uint lane=tid&31u,sg=tid>>5,j=grid.x*SG+sg,r0=grid.y*RR;
 if(j>=H)return;   // SIMD-uniform
 uint groups=R/p.group;
 float acc[RR][HC];
 for(uint r=0;r<RR;r++)for(uint c=0;c<HC;c++)acc[r][c]=0.;
 for(uint base=0;base<R;base+=256) {
   uint k0=base+lane*8;
   if(k0>=R)continue;
   float vals[RR][8],sums[RR];
   for(uint r=0;r<RR;r++) {
     sums[r]=0.;
     for(uint i=0;i<8;i++){float a=r0+r<p.rows?float(act[(r0+r)*R+k0+i]):0.;vals[r][i]=a;sums[r]+=a;}
   }
   for(uint c=0;c<HC;c++) {
     ulong row=ulong(c*H+j);
     float s=float(scales[row*groups+k0/p.group]),b=float(biases[row*groups+k0/p.group]);
     uint2 raw=*(const device uint2*)(w+row*R+k0);
     uchar4 lo=as_type<uchar4>(raw.x),hi=as_type<uchar4>(raw.y);
     float code[8];
     for(uint i=0;i<4;i++){code[i]=float(lo[i]);code[4+i]=float(hi[i]);}
     for(uint r=0;r<RR;r++){float dot=0.;for(uint i=0;i<8;i++)dot+=vals[r][i]*code[i];acc[r][c]+=s*dot+b*sums[r];}
   }
 }
 for(uint r=0;r<RR;r++) {
   float u[HC];
   for(uint c=0;c<HC;c++)u[c]=simd_sum(acc[r][c]);
   if(lane==0 && r0+r<p.rows) {
     uint row=r0+r;float sum=0.;
     for(uint h=0;h<HC;h++) {
       uint at=row*F+h*H+j;bfloat ub=bfloat(u[h]);
       bfloat sg2=bfloat(1.f/(1.f+exp(-float(ub))));
       sum+=float(bfloat(float(sg2)*float(normalized[at])));
     }
     mixed[row*H+j]=bfloat(float(bfloat(sum))*0.25f);
   }
 }
}
#define HC_UP_MIX(RR) \
kernel void hc_up_mix_r##RR(const device bfloat* act [[buffer(0)]],const device uchar* w [[buffer(1)]], \
 const device bfloat* scales [[buffer(2)]],const device bfloat* biases [[buffer(3)]], \
 const device bfloat* normalized [[buffer(4)]],device bfloat* mixed [[buffer(5)]], \
 constant HcUpMixParams& p [[buffer(8)]],uint3 grid [[threadgroup_position_in_grid]], \
 uint tid [[thread_index_in_threadgroup]]) {hc_up_mix_body<RR,8>(act,w,scales,biases,normalized,mixed,p,grid,tid);}
HC_UP_MIX(1)
HC_UP_MIX(2)
HC_UP_MIX(4)
HC_UP_MIX(8)
// R57 Sushi's fused decode/verify HC down and up kernels (qwen4_hc.zig HC_FUSED_D_SOURCE and
// HC_FUSED_U_SOURCE, MIT; see NOTICE.txt) on this file's buffers: affine8 codes [n][k] (four
// per 32-bit word, little-endian), BF16 scales/biases per group. Down: one 256-thread group per
// output row n (the R down rows, then the HC inject rows), 8 simdgroups each a slice of K, up to
// ROWS rows sharing every weight word; act = v*sigmoid(v) and inject = 2*sigmoid(sum of hc_norm's
// partials), rounded as hc_act_reduce rounds them. Up+mix: one simdgroup per channel j over the HC
// stream rows h*H+j of W_up (K = R), then hc_mix_reduce's mix. FP32 dot products (the split-K
// fragment path differs in summation order and precision).
struct HcRowsParams {uint rows,group;};
template<uint ROWS>
inline void hc_down_rows_body(const device bfloat* xn,const device uint* w,const device bfloat* s,
 const device bfloat* b,const device float* parts,device bfloat* act,device bfloat* inject,
 constant HcRowsParams& p,threadgroup float* part,uint n,uint z,uint tid,uint sg,uint lane) {
 const uint row0=z*ROWS,words=F/4,slice=words/8,iters=slice/32,groups=F/p.group;
 if(n<R) {
   const device uint* wr=w+n*words;
   float a0[ROWS],a1[ROWS],a2[ROWS],a3[ROWS];
   for(uint r=0;r<ROWS;r++){a0[r]=0.;a1[r]=0.;a2[r]=0.;a3[r]=0.;}
   uint p0=sg*slice+lane;
   for(uint i=0;i<iters;i++) {
     uint wi=p0+32*i,k=wi*4;
     uint q=wr[wi];
     float sj=float(s[n*groups+k/p.group]),bj=float(b[n*groups+k/p.group]);
     float w0=float(q&0xffu)*sj+bj,w1=float((q>>8)&0xffu)*sj+bj,w2=float((q>>16)&0xffu)*sj+bj,w3=float(q>>24)*sj+bj;
     for(uint r=0;r<ROWS;r++) {
       const device bfloat* x=xn+min(row0+r,p.rows-1)*F+k;
       a0[r]+=float(x[0])*w0;a1[r]+=float(x[1])*w1;a2[r]+=float(x[2])*w2;a3[r]+=float(x[3])*w3;
     }
   }
   for(uint r=0;r<ROWS;r++) {
     float acc=simd_sum((a0[r]+a1[r])+(a2[r]+a3[r]));
     if(lane==0)part[r*8+sg]=acc;
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(tid<ROWS && row0+tid<p.rows) {
     float t=0.;for(uint g=0;g<8;g++)t+=part[tid*8+g];
     bfloat v=bfloat(t),sv=bfloat(1.f/(1.f+exp(-float(v))));
     act[(row0+tid)*R+n]=bfloat(float(v)*float(sv));
   }
 } else if(tid<ROWS && row0+tid<p.rows) {
   uint row=row0+tid,c=n-R;
   float raw=0.;for(uint h=0;h<HC;h++)raw+=parts[(row*HC+h)*HC+c];
   bfloat v=bfloat(raw),sv=bfloat(1.f/(1.f+exp(-float(v))));
   inject[row*HC+c]=bfloat(2.f*float(sv));
 }
}
#define HC_DOWN_ROWS(ROWS) \
kernel void hc_down_rows_r##ROWS(const device bfloat* xn [[buffer(0)]],const device uint* w [[buffer(1)]], \
 const device bfloat* s [[buffer(2)]],const device bfloat* b [[buffer(3)]],const device float* parts [[buffer(4)]], \
 device bfloat* act [[buffer(5)]],device bfloat* inject [[buffer(6)]],constant HcRowsParams& p [[buffer(8)]], \
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]], \
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
 threadgroup float part[8*ROWS]; \
 hc_down_rows_body<ROWS>(xn,w,s,b,parts,act,inject,p,part,g.y,g.z,tid,sg,lane);}
HC_DOWN_ROWS(1)
HC_DOWN_ROWS(2)
HC_DOWN_ROWS(3)
HC_DOWN_ROWS(4)
HC_DOWN_ROWS(5)
HC_DOWN_ROWS(6)
HC_DOWN_ROWS(7)
HC_DOWN_ROWS(8)
template<uint ROWS>
inline void hc_up_rows_body(const device bfloat* act,const device uint* w,const device bfloat* s,
 const device bfloat* b,const device bfloat* xn,device bfloat* mixed,constant HcRowsParams& p,
 uint j,uint z,uint lane) {
 const uint row0=z*ROWS,words=R/4,rit=(words+31)/32,groups=R/p.group;
 float sum[ROWS];
 for(uint r=0;r<ROWS;r++)sum[r]=0.;
 for(uint h=0;h<HC;h++) {
   uint row=h*H+j;
   const device uint* wr=w+row*words;
   float a0[ROWS],a1[ROWS],a2[ROWS],a3[ROWS];
   for(uint r=0;r<ROWS;r++){a0[r]=0.;a1[r]=0.;a2[r]=0.;a3[r]=0.;}
   for(uint i=0;i<rit;i++) {
     uint wi=lane+32*i;
     if(wi<words) {
       uint q=wr[wi],k=wi*4;
       float sj=float(s[row*groups+k/p.group]),bj=float(b[row*groups+k/p.group]);
       float w0=float(q&0xffu)*sj+bj,w1=float((q>>8)&0xffu)*sj+bj,w2=float((q>>16)&0xffu)*sj+bj,w3=float(q>>24)*sj+bj;
       for(uint r=0;r<ROWS;r++) {
         const device bfloat* a=act+min(row0+r,p.rows-1)*R+k;
         a0[r]+=float(a[0])*w0;a1[r]+=float(a[1])*w1;a2[r]+=float(a[2])*w2;a3[r]+=float(a[3])*w3;
       }
     }
   }
   for(uint r=0;r<ROWS;r++) {
     float acc=simd_sum((a0[r]+a1[r])+(a2[r]+a3[r]));
     bfloat u=bfloat(acc),sv=bfloat(1.f/(1.f+exp(-float(u))));
     sum[r]+=float(bfloat(float(sv)*float(xn[min(row0+r,p.rows-1)*F+row])));
   }
 }
 for(uint r=0;r<ROWS;r++)
   if(lane==0 && row0+r<p.rows)mixed[(row0+r)*H+j]=bfloat(float(bfloat(sum[r]))*0.25f);
}
#define HC_UP_ROWS(ROWS) \
kernel void hc_up_rows_r##ROWS(const device bfloat* act [[buffer(0)]],const device uint* w [[buffer(1)]], \
 const device bfloat* s [[buffer(2)]],const device bfloat* b [[buffer(3)]],const device bfloat* xn [[buffer(4)]], \
 device bfloat* mixed [[buffer(5)]],constant HcRowsParams& p [[buffer(8)]], \
 uint3 t [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) { \
 hc_up_rows_body<ROWS>(act,w,s,b,xn,mixed,p,t.y,t.z,lane);}
HC_UP_ROWS(1)
HC_UP_ROWS(2)
HC_UP_ROWS(3)
HC_UP_ROWS(4)
HC_UP_ROWS(5)
HC_UP_ROWS(6)
HC_UP_ROWS(7)
HC_UP_ROWS(8)
