// Native affine8 screen. The row-reuse QMV organization follows Sushi mtp_qmv
// (MLX-derived); MPP tiled GEMM follows MTPLX/Sushi's VQMM_NAX_SOURCE. Notices
// retained in NOTICE.txt. New native ABI, explicit FP16 precision arms, row/K
// guards and FP32 diagnostic output. No whole-matrix GPU dequantization.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct DenseParams { uint k,n,group,rows; };

template<uint NV>
inline void affine8_qmv_body(const device bfloat* x,
 const device uchar* w,const device bfloat* scales,
 const device bfloat* biases,device float* y,
 device bfloat* ybf,constant DenseParams& p,uint3 grid,uint tid) {
 uint lane=tid&31u,sg=tid>>5,n0=grid.x*4+sg*2,r0=grid.y*NV;
 float accum[NV][2] = {};
 uint groups=p.k/p.group;
 for(uint base=0;base<p.k;base+=256) {
   uint k0=base+lane*8;
   float vals[NV][8]={};float sums[NV]={};
   for(uint r=0;r<NV;r++)for(uint i=0;i<8;i++) {
     float a=(r0+r<p.rows && k0+i<p.k)?float(x[ulong(r0+r)*p.k+k0+i]):0.;
     vals[r][i]=a;sums[r]+=a;
   }
   if(k0<p.k)for(uint n=0;n<2;n++)if(n0+n<p.n) {
     float s=float(scales[ulong(n0+n)*groups+k0/p.group]);
     float b=float(biases[ulong(n0+n)*groups+k0/p.group]);
     uchar codes[8];for(uint i=0;i<8;i++)codes[i]=k0+i<p.k?w[ulong(n0+n)*p.k+k0+i]:0;
     for(uint r=0;r<NV;r++) {
       float dot=0.;for(uint i=0;i<8;i++)dot+=vals[r][i]*float(codes[i]);
       accum[r][n]+=s*dot+b*sums[r];
     }
   }
 }
 for(uint r=0;r<NV;r++)for(uint n=0;n<2;n++) {
   float v=simd_sum(accum[r][n]);
   if(lane==0 && r0+r<p.rows && n0+n<p.n) {
     ulong index=ulong(r0+r)*p.n+n0+n;y[index]=v;ybf[index]=bfloat(v);
   }
 }
}

#define DENSE_QMV(NAME,NV) \
kernel void NAME(const device bfloat* x [[buffer(0)]], \
 const device uchar* w [[buffer(1)]],const device bfloat* scales [[buffer(2)]], \
 const device bfloat* biases [[buffer(3)]],device float* y [[buffer(4)]], \
 device bfloat* ybf [[buffer(5)]],constant DenseParams& p [[buffer(6)]], \
 uint3 grid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 affine8_qmv_body<NV>(x,w,scales,biases,y,ybf,p,grid,tid); \
}
DENSE_QMV(affine8_qmv1,1)
DENSE_QMV(affine8_qmv2,2)
DENSE_QMV(affine8_qmv4,4)

template<typename T,bool Residual>
inline void affine8_mpp_body(const device bfloat* x,const device uchar* w,
 const device bfloat* scales,const device bfloat* biases,device float* y,
 device bfloat* ybf,constant DenseParams& p,uint3 grid,uint tid,
 threadgroup T* aa,threadgroup T* bb,
 threadgroup float* partial) {
 constexpr uint BM=16,BN=32,BK=16,NSG=8;
 uint lane=tid&31u,sg=tid>>5,n0=grid.x*BN,r0=grid.y*BM;
 // Partition whole BK tiles: HC's 320-wide up projection is not divisible
 // by NSG*BK. Scalar K/NSG slices would overlap and overread at that shape.
 uint tiles=p.k/BK,begin=(tiles*sg/NSG)*BK,end=(tiles*(sg+1)/NSG)*BK,groups=p.k/p.group;
 constexpr auto desc=matmul2d_descriptor(16,32,16,false,false,false,
     matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroup> op;
 tensor<threadgroup T,dextents<int,2>,tensor_inline> A(aa+sg*256,
     dextents<int,2>{16,16},array<int,2>{1,16});
 tensor<threadgroup T,dextents<int,2>,tensor_inline> B(bb+sg*512,
     dextents<int,2>{32,16},array<int,2>{1,32});
 tensor<threadgroup float,dextents<int,2>,tensor_inline> C(partial+sg*512,
     dextents<int,2>{32,16},array<int,2>{1,32});
 auto dst=op.template get_destination_cooperative_tensor<
     tensor<threadgroup T,extents<int,16,16>,tensor_inline>,
     tensor<threadgroup T,extents<int,32,16>,tensor_inline>,float>();
 for(uint i=0;i<dst.get_capacity();i++)dst[i]=0.;
 for(uint k0=begin;k0<end;k0+=BK) {
   uint col=n0+lane;
   float s=col<p.n?float(scales[ulong(col)*groups+k0/p.group]):0.;
   float b=col<p.n?float(biases[ulong(col)*groups+k0/p.group]):0.;
   float original[16];
   for(uint i=0;i<16;i++) {
     float v=col<p.n?float(w[ulong(col)*p.k+k0+i])*s+b:0.;
     T hi=T(v);bb[sg*512+i*32+lane]=hi;
     if constexpr(Residual) original[i]=v-float(hi);
   }
   for(uint i=lane;i<256;i+=32) {
     uint row=r0+i/16,kk=k0+i%16;
     aa[sg*256+i]=row<p.rows?T(x[ulong(row)*p.k+kk]):T(0);
   }
   simdgroup_barrier(mem_flags::mem_threadgroup);
   auto a=A.template slice<16,16>(0,0);
   auto bmat=B.template slice<32,16>(0,0);
   op.run(a,bmat,dst);
   if constexpr(Residual) {
     simdgroup_barrier(mem_flags::mem_threadgroup);
     for(uint i=0;i<16;i++)bb[sg*512+i*32+lane]=T(original[i]);
     simdgroup_barrier(mem_flags::mem_threadgroup);
     op.run(a,bmat,dst);
   }
   simdgroup_barrier(mem_flags::mem_threadgroup);
 }
 auto c=C.template slice<32,16>(0,0);dst.store(c);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<512;i+=256) {
   uint row=r0+i/32,col=n0+i%32;
   float a01=partial[i]+partial[512+i],a23=partial[1024+i]+partial[1536+i];
   float a45=partial[2048+i]+partial[2560+i],a67=partial[3072+i]+partial[3584+i];
   float v=(a01+a23)+(a45+a67);
   if(row<p.rows && col<p.n) {ulong at=ulong(row)*p.n+col;y[at]=v;ybf[at]=bfloat(v);}
 }
}

#define DENSE_MPP(NAME,T,RESIDUAL) \
kernel void NAME(const device bfloat* x [[buffer(0)]], \
 const device uchar* w [[buffer(1)]],const device bfloat* scales [[buffer(2)]], \
 const device bfloat* biases [[buffer(3)]],device float* y [[buffer(4)]], \
 device bfloat* ybf [[buffer(5)]],constant DenseParams& p [[buffer(6)]], \
 uint3 grid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup T aa[8*256],bb[8*512]; \
 threadgroup float partial[8*512]; \
 affine8_mpp_body<T,RESIDUAL>(x,w,scales,biases,y,ybf,p,grid,tid,aa,bb,partial); \
}
DENSE_MPP(affine8_mpp_bf16,bfloat,false)
DENSE_MPP(affine8_mpp_f16,half,false)
DENSE_MPP(affine8_mpp_f16x2,half,true)

// Alternative: integer codes are exactly representable in BF16. Accumulate
// each quantization group on matrix hardware, then apply its affine transform:
// s * dot(x, code-128) + (bias+128*s) * sum(x). This avoids rounded per-weight
// dequantization. It changes floating-point reduction order, not stored weights.
kernel void affine8_xsum(const device bfloat* x [[buffer(0)]],
 constant DenseParams& p [[buffer(6)]],device float* sums [[buffer(7)]],
 uint3 grid [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 float v=0.;for(uint i=lane;i<p.group;i+=32)v+=float(x[ulong(grid.y)*p.k+grid.x*p.group+i]);
 v=simd_sum(v);if(lane==0)sums[grid.y*(p.k/p.group)+grid.x]=v;
}
kernel void affine8_mpp_postscale(const device bfloat* x [[buffer(0)]],
 const device uchar* w [[buffer(1)]],const device bfloat* scales [[buffer(2)]],
 const device bfloat* biases [[buffer(3)]],device float* y [[buffer(4)]],
 device bfloat* ybf [[buffer(5)]],constant DenseParams& p [[buffer(6)]],
 const device float* sums [[buffer(7)]],uint3 grid [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
 uint lane=tid&31u,sg=tid>>5,n0=grid.x*32,r0=grid.y*16,groups=p.k/p.group;
 uint per=(groups+7)/8,begin=sg*per,end=min(begin+per,groups),col=n0+lane;
 threadgroup bfloat aa[8*256],bb[8*512];
 threadgroup float partial[8*512];
 tensor<threadgroup bfloat,dextents<int,2>,tensor_inline> A(aa+sg*256,
     dextents<int,2>{16,16},array<int,2>{1,16});
 tensor<threadgroup bfloat,dextents<int,2>,tensor_inline> B(bb+sg*512,
     dextents<int,2>{32,16},array<int,2>{1,32});
 tensor<threadgroup float,dextents<int,2>,tensor_inline> C(partial+sg*512,
     dextents<int,2>{32,16},array<int,2>{1,32});
 constexpr auto desc=matmul2d_descriptor(16,32,16,false,false,false,
     matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroup> op;
 auto dst=op.template get_destination_cooperative_tensor<
     tensor<threadgroup bfloat,extents<int,16,16>,tensor_inline>,
     tensor<threadgroup bfloat,extents<int,32,16>,tensor_inline>,float>();
 float accum[16]={};
 for(uint g=begin;g<end;g++) {
   for(uint i=0;i<dst.get_capacity();i++)dst[i]=0.;
   for(uint k0=g*p.group;k0<(g+1)*p.group;k0+=16) {
     for(uint i=0;i<16;i++)bb[sg*512+i*32+lane]=col<p.n?bfloat(int(w[ulong(col)*p.k+k0+i])-128):bfloat(0);
     for(uint i=lane;i<256;i+=32) {
       uint row=r0+i/16;
       aa[sg*256+i]=row<p.rows?x[ulong(row)*p.k+k0+i%16]:bfloat(0);
     }
     simdgroup_barrier(mem_flags::mem_threadgroup);
     auto a=A.template slice<16,16>(0,0);auto b=B.template slice<32,16>(0,0);
     op.run(a,b,dst);simdgroup_barrier(mem_flags::mem_threadgroup);
   }
   auto c=C.template slice<32,16>(0,0);dst.store(c);
   simdgroup_barrier(mem_flags::mem_threadgroup);
   float scale=col<p.n?float(scales[ulong(col)*groups+g]):0.;
   float bias=col<p.n?float(biases[ulong(col)*groups+g]):0.;
   for(uint r=0;r<16;r++) {
     float sx=r0+r<p.rows?sums[(r0+r)*groups+g]:0.;
     accum[r]+=scale*partial[sg*512+r*32+lane]+(bias+128.*scale)*sx;
   }
   simdgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint r=0;r<16;r++)partial[sg*512+r*32+lane]=accum[r];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<512;i+=256) {
   uint row=r0+i/32,n=n0+i%32;
   float a01=partial[i]+partial[512+i],a23=partial[1024+i]+partial[1536+i];
   float a45=partial[2048+i]+partial[2560+i],a67=partial[3072+i]+partial[3584+i];
   float v=(a01+a23)+(a45+a67);
   if(row<p.rows && n<p.n){ulong at=ulong(row)*p.n+n;y[at]=v;ybf[at]=bfloat(v);}
 }
}
