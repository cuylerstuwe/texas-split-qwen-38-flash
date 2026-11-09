// R55 target layer-1 PLE core on Metal, ported from the CPU reference ple::core_reference
// (source-grounded against Sushi qwen4_exp; MIT, NOTICE.txt). BF16 rounding boundaries
// follow the reference; reductions are FP32 (the reference sums in FP64).
#include <metal_stdlib>
using namespace metal;
constant uint H=2560,HC=4,F=10240,HISTORY=9;
struct PleParams { uint rows; uint pad0; uint pad1; uint pad2; };

// One threadgroup (256 threads) per (stream, row): out = bf16(bf16(x*inv) * w[h*H+j]).
kernel void ple_norm(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant PleParams& p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint h=g.x,row=g.y;
    threadgroup float sums[8];
    float v[10],ss=0.f;
    for(uint i=0;i<10;i++){uint j=tid+256*i;v[i]=float(x[row*F+h*H+j]);ss+=v[i]*v[i];}
    ss=simd_sum(ss);if(lane==0)sums[sg]=ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0.f;for(uint i=0;i<8;i++)total+=sums[i];
    float inv=precise::rsqrt(total/float(H)+1e-6f);
    for(uint i=0;i<10;i++){uint j=tid+256*i;
        out[row*F+h*H+j]=bfloat(float(bfloat(v[i]*inv))*float(w[h*H+j]));}
}
// One threadgroup per (stream, row): dot(key_norm, query_norm) over the stream, scaled by
// bf16(1/sqrt(H)), signed square root, sigmoid gate, gated = gate * value.
kernel void ple_gate(const device bfloat* kn [[buffer(0)]], const device bfloat* qn [[buffer(1)]],
    const device bfloat* value [[buffer(2)]], device bfloat* gated [[buffer(3)]],
    device bfloat* gates [[buffer(4)]], constant PleParams& p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint h=g.x,row=g.y;
    threadgroup float sums[8];
    float d=0.f;
    for(uint i=0;i<10;i++){uint j=row*F+h*H+tid+256*i;d+=float(bfloat(float(kn[j])*float(qn[j])));}
    d=simd_sum(d);if(lane==0)sums[sg]=d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0.f;for(uint i=0;i<8;i++)total+=sums[i];
    float dot=float(bfloat(total));
    float sc=float(bfloat(dot*float(bfloat(precise::rsqrt(float(H))))));
    float mag=float(bfloat(precise::sqrt(max(fabs(sc),float(bfloat(1e-6f))))));
    float root=float(bfloat(sc==0.f?0.f:(sc>0.f?mag:-mag)));
    float gate=float(bfloat(1.f/(1.f+precise::exp(-root))));
    if(tid==0)gates[row*HC+h]=bfloat(gate);
    for(uint i=0;i<10;i++){uint j=tid+256*i;
        gated[row*F+h*H+j]=bfloat(gate*float(value[row*H+j]));}
}
// One thread per (row, channel): causal taps r, r+3, r+6, r+9 of [history; normalized],
// SiLU, output = stream + bf16(gated + act).
kernel void ple_conv_out(const device bfloat* history [[buffer(0)]],
    const device bfloat* normalized [[buffer(1)]], const device bfloat* conv [[buffer(2)]],
    const device bfloat* gated [[buffer(3)]], const device bfloat* stream [[buffer(4)]],
    device bfloat* out [[buffer(5)]], constant PleParams& p [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*F)return;
    uint r=i/F,d=i%F;
    float c=0.f;
    for(uint j=0;j<4;j++){
        uint t=r+3*j;
        float v=t<HISTORY?float(history[t*F+d]):float(normalized[(t-HISTORY)*F+d]);
        c+=v*float(conv[d*4+j]);
    }
    float cb=float(bfloat(c));
    float act=float(bfloat(cb*float(bfloat(1.f/(1.f+precise::exp(-cb))))));
    float y=float(bfloat(float(gated[i])+act));
    out[i]=bfloat(float(stream[i])+y);
}
