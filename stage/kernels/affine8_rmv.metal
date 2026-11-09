// R55 multi-row affine8 matrix-vector kernel. Same per-lane arithmetic as affine8_qmv
// (Sushi mtp_qmv organization, MLX-derived; NOTICE.txt): lane l owns 8 consecutive K
// values of every 256-wide step, dot(x, code) and sum(x) per quantization group chunk,
// y += s*dot + b*sum, FP32 throughout, one simd_sum per output at the end. Changes: up to
// 8 rows per pass (weights read once for 8 rows instead of twice), C columns per
// simdgroup (amortizes the x loads), 8-byte code loads, SG simdgroups per threadgroup.
#include <metal_stdlib>
using namespace metal;
struct RmvParams {uint k,n,group,rows,splits,repacked;};

template<uint R,uint C,uint SG>
inline void affine8_rmv_body(const device bfloat* x,const device uchar* w,
 const device bfloat* scales,const device bfloat* biases,device float* y,
 device bfloat* ybf,constant RmvParams& p,uint3 grid,uint tid) {
 uint lane=tid&31u,sg=tid>>5,n0=(grid.x*SG+sg)*C,r0=grid.y*R;
 if(n0>=p.n)return;   // SIMD-uniform; no threadgroup cooperation
 float acc[R][C];
 for(uint r=0;r<R;r++)for(uint c=0;c<C;c++)acc[r][c]=0.;
 uint groups=p.k/p.group;
 for(uint base=0;base<p.k;base+=256) {
   uint k0=base+lane*8;
   float vals[R][8];float sums[R];
   for(uint r=0;r<R;r++) {
     sums[r]=0.;
     for(uint i=0;i<8;i++) {
       float a=(r0+r<p.rows && k0+i<p.k)?float(x[ulong(r0+r)*p.k+k0+i]):0.;
       vals[r][i]=a;sums[r]+=a;
     }
   }
   if(k0<p.k)for(uint c=0;c<C;c++)if(n0+c<p.n) {
     ulong row=ulong(n0+c);
     float s=float(scales[row*groups+k0/p.group]);
     float b=float(biases[row*groups+k0/p.group]);
     float code[8];
     if(k0+8<=p.k) {
       uint2 raw=*(const device uint2*)(w+row*p.k+k0);
       uchar4 lo=as_type<uchar4>(raw.x),hi=as_type<uchar4>(raw.y);
       for(uint i=0;i<4;i++){code[i]=float(lo[i]);code[4+i]=float(hi[i]);}
     } else {
       for(uint i=0;i<8;i++)code[i]=k0+i<p.k?float(w[row*p.k+k0+i]):0.;
     }
     for(uint r=0;r<R;r++) {
       float dot=0.;for(uint i=0;i<8;i++)dot+=vals[r][i]*code[i];
       acc[r][c]+=s*dot+b*sums[r];
     }
   }
 }
 for(uint r=0;r<R;r++)for(uint c=0;c<C;c++) {
   float v=simd_sum(acc[r][c]);
   if(lane==0 && r0+r<p.rows && n0+c<p.n) {
     ulong index=ulong(r0+r)*p.n+n0+c;y[index]=v;ybf[index]=bfloat(v);
   }
 }
}
#define RMV(R,C,SG) \
kernel void affine8_rmv_r##R##_c##C##_s##SG(const device bfloat* x [[buffer(0)]], \
 const device uchar* w [[buffer(1)]],const device bfloat* scales [[buffer(2)]], \
 const device bfloat* biases [[buffer(3)]],device float* y [[buffer(4)]], \
 device bfloat* ybf [[buffer(5)]],constant RmvParams& p [[buffer(6)]], \
 uint3 grid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 affine8_rmv_body<R,C,SG>(x,w,scales,biases,y,ybf,p,grid,tid); \
}
RMV(1,2,2) RMV(1,4,2) RMV(1,8,2) RMV(1,4,4) RMV(1,8,4)
RMV(2,2,2) RMV(2,4,2) RMV(2,8,2) RMV(2,4,4)
RMV(4,2,2) RMV(4,4,2) RMV(4,8,2) RMV(4,4,4)
RMV(8,2,2) RMV(8,4,2) RMV(8,8,2) RMV(8,4,4)
