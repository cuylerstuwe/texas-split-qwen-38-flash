// Direct cooperative-fragment affine8 GEMM. Fragment origins/load/store follow
// Sushi GEMM_NAX_FRAGS (MIT; NOTICE.txt); byte dequantization is new. 32 input
// rows share each weight load. No threadgroup arrays/barriers or unpacked matrix.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct FragParams {uint k,n,group,rows,splits,repacked;};
inline uint2 affine_frag_origin(uint lane) {
 uint qid=lane>>2;
 return uint2(((qid&2u)|(lane&1u))*4u,(qid&4u)|((lane>>1)&3u));
}
template<typename L> inline void affine_frag_left(thread L& left,
 const device bfloat* x,uint row,uint k,uint base,uint2 origin,uint count) {
 using b4=vec<bfloat,4>;
 ulong at=ulong(row+origin.y)*k+base+origin.x;
 b4 a=origin.y<count?*(const device b4*)(x+at):b4(0);
 b4 b=origin.y+8u<count?*(const device b4*)(x+at+8ul*k):b4(0);
 for(uint i=0;i<4;i++){left[i]=half(float(a[i]));left[4+i]=half(float(b[i]));}
}
template<typename D> inline void affine_frag_store(const thread D& dst,
 device float* partial,device float* y,device bfloat* ybf,constant FragParams& p,
 uint row,uint base,uint2 origin,uint count,uint split) {
 for(uint tile=0;tile<2;tile++)for(uint c=0;c<4;c++) {
   uint col=base+tile*16u+origin.x+c;
   if(col>=p.n)continue;
   for(uint high=0;high<2;high++)if(origin.y+8u*high<count) {
     uint r=row+origin.y+8u*high;float v=dst[tile*8u+high*4u+c];
     if(p.splits==1){ulong at=ulong(r)*p.n+col;y[at]=v;ybf[at]=bfloat(v);}
     else partial[(ulong(r)*p.splits+split)*p.n+col]=v;
   }
 }
}
template<bool Residual,bool Packed> inline void affine_frag_body(const device bfloat* x,
 const device uchar* w,const device bfloat* scales,const device bfloat* biases,
 device float* y,device bfloat* ybf,constant FragParams& p,device float* partial,
 uint3 grid,uint tid) {
 uint lane=tid&31u,sg=tid>>5,base=grid.x*128u+sg*32u,row=grid.y*32u;
 // SIMD-uniform return is safe: there is no inter-SIMD-group cooperation.
 if(base>=p.n || row>=p.rows)return;
 uint count=min(32u,p.rows-row),hi=count>16?count-16:0,groups=p.k/p.group;
 uint tiles=p.k/16,per=(tiles+p.splits-1)/p.splits,begin=grid.z*per,end=min(begin+per,tiles);
 uint2 origin=affine_frag_origin(lane);
 constexpr auto desc=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroup> op;
 auto left=op.get_left_input_cooperative_tensor<half,half,float>();
 auto right=op.get_right_input_cooperative_tensor<half,half,float>();
 auto dst=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto dst_hi=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(uint i=0;i<dst.get_capacity();i++){dst[i]=0.;if(hi)dst_hi[i]=0.;}
 uint np=(p.n+31u)&~31u,shift=p.group==128u?7u:6u,cached_group=0xffffffffu;
 float cached_s[4],cached_b[4];
 #pragma clang loop unroll_count(2)
 for(uint tk=begin;tk<end;tk++) {
   uint kk=tk*16u+origin.x;
   if constexpr(Packed) {
     uint g=kk>>shift;
     if(g!=cached_group) {
       for(uint nr=0;nr<4;nr++) {
         uint col=base+origin.y+nr*8u;
         cached_s[nr]=col<p.n?float(scales[ulong(g)*np+col]):0.;
         cached_b[nr]=col<p.n?float(biases[ulong(g)*np+col]):0.;
       }
       cached_group=g;
     }
   }
   float residual[16];
   // Each lane owns four adjacent K values in each of four weight rows.
   // This is the B-transposed cooperative layout, not the A/C output layout.
   for(uint nr=0;nr<4;nr++) {
     uint col=base+origin.y+nr*8u;
     ulong wi=Packed?(ulong(tk)*np+col)*16u+origin.x:ulong(col)*p.k+kk;
     uchar4 code=col<p.n?*(const device uchar4*)(w+wi):uchar4(0);
     float s,b;
     if constexpr(Packed){s=cached_s[nr];b=cached_b[nr];}
     else {ulong si=ulong(col)*groups+kk/p.group;s=col<p.n?float(scales[si]):0.;b=col<p.n?float(biases[si]):0.;}
     for(uint i=0;i<4;i++) {
       float v=float(code[i])*s+b;half vhi=half(v);right[nr*4u+i]=vhi;
       if constexpr(Residual)residual[nr*4u+i]=v-float(vhi);
     }
   }
   affine_frag_left(left,x,row,p.k,tk*16,origin,count);op.run(left,right,dst);
   if(hi){affine_frag_left(left,x,row+16,p.k,tk*16,origin,hi);op.run(left,right,dst_hi);}
   if constexpr(Residual) {
     for(uint i=0;i<16;i++)right[i]=half(residual[i]);
     affine_frag_left(left,x,row,p.k,tk*16,origin,count);op.run(left,right,dst);
     if(hi){affine_frag_left(left,x,row+16,p.k,tk*16,origin,hi);op.run(left,right,dst_hi);}
   }
 }
 affine_frag_store(dst,partial,y,ybf,p,row,base,origin,count,grid.z);
 if(hi)affine_frag_store(dst_hi,partial,y,ybf,p,row+16,base,origin,hi,grid.z);
}
#define FRAG_KERNEL(NAME,RESIDUAL,PACKED) \
kernel void NAME(const device bfloat* x [[buffer(0)]],const device uchar* w [[buffer(1)]], \
 const device bfloat* scales [[buffer(2)]],const device bfloat* biases [[buffer(3)]], \
 device float* y [[buffer(4)]],device bfloat* ybf [[buffer(5)]],constant FragParams& p [[buffer(6)]], \
 device float* partial [[buffer(7)]],uint3 grid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 affine_frag_body<RESIDUAL,PACKED>(x,w,scales,biases,y,ybf,p,partial,grid,tid); \
}
FRAG_KERNEL(affine8_frag_f16,false,false)
FRAG_KERNEL(affine8_frag_f16x2,true,false)
FRAG_KERNEL(affine8_packed_f16,false,true)
FRAG_KERNEL(affine8_packed_f16x2,true,true)
kernel void affine8_frag_reduce(device float* y [[buffer(4)]],device bfloat* ybf [[buffer(5)]],
 constant FragParams& p [[buffer(6)]],const device float* partial [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.rows*p.n)return;
 uint row=i/p.n,col=i%p.n;float v=0.;
 for(uint split=0;split<p.splits;split++)v+=partial[(ulong(row)*p.splits+split)*p.n+col];
 y[i]=v;ybf[i]=bfloat(v);
}
