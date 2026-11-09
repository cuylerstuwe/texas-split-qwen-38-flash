// R55 expert-grouped EXL3 projection. Same tile decode and per-slot arithmetic as
// exl3_project (Sushi-derived, MIT; NOTICE.txt), but each threadgroup owns one expert
// and up to S of the slots routed to it: a weight tile is fetched and decoded once and
// applied to every one of those slots. Per slot, the FMA order, the lane reductions
// and the warp sum are those of exl3_project, so outputs are unchanged.
// Grouping table (uint): [0] entry count; entry g at 4 + 12*g: expert, slot count,
// slot[8], unused[2]. Entries are in expert order, slots in slot order; an expert with
// more than 8 slots (16-row batches) gets consecutive entries.
constant uint R55_GSLOTS = 8;
struct R55GroupParams { uint slots; uint experts; uint pad0; uint pad1; };
kernel void r55_group_slots(const device uint* ids [[buffer(0)]], device uint* table [[buffer(1)]],
    constant R55GroupParams& p [[buffer(2)]], uint e [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup uint totals[32];
    uint mine[16];
    uint c = 0;
    if (e < p.experts) {
        for (uint s = 0; s < p.slots; s++) {
            if (ids[s] == e && c < 16) { mine[c] = s; c++; }
        }
    }
    uint n = (c + R55_GSLOTS - 1) / R55_GSLOTS;
    uint before = simd_prefix_exclusive_sum(n);
    uint total = simd_sum(n);
    if (lane == 0) totals[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint i = 0; i < sg; i++) base += totals[i];
    uint first = base + before;
    for (uint j = 0; j < n; j++) {
        device uint* t = table + 4 + (first + j) * 12;
        uint lo = j * R55_GSLOTS, cnt = min(R55_GSLOTS, c - lo);
        t[0] = e; t[1] = cnt;
        for (uint i = 0; i < R55_GSLOTS; i++) t[2 + i] = i < cnt ? mine[lo + i] : 0u;
    }
    if (e == nsg * 32 - 1) {
        uint all = 0;
        for (uint i = 0; i < nsg; i++) all += totals[i];
        table[0] = all;
    }
}
struct R55ProjParams { uint input_dim; uint output_dim; uint splits; uint pad; };
inline float r55_decode_mcg(uint cw) {
    uint p=(((cw & 0x7fffu)*0xCBAC1FEDu)&0x8FFF8FFFu)^0x3B603B60u;
    half sum=as_type<half>(ushort(p))+as_type<half>(ushort(p>>16));
    return float(sum);
}
template<uint S>
inline void r55_project_grouped(const device half* x, const device uint* trellis,
    const device uint* table, device float* output, constant R55ProjParams& p,
    uint3 group, uint tid, threadgroup float* partial) {
    uint g=group.y;
    if(g>=table[0])return;   // threadgroup-uniform
    const device uint* t=table+4+g*12;
    uint expert=t[0],cnt=t[1];
    uint slot[S];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)slot[s]=t[2+s];
    uint ot=group.x,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.input_dim/16,nt=p.output_dim/16;
    uint per=(it+p.splits-1)/p.splits,begin=split*per,end=min(begin+per,it);
    uint row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(expert)*it*nt*32;
    float acc[S][8];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)for(uint j=0;j<8;j++)acc[s][j]=0.;
    for(uint tk=begin+warp;tk<end;tk+=4) {
        const device uint* words=weights+(ulong(tk)*nt+ot)*32;
        uint previous=words[(lane+31)&31], current=words[lane];
        float w[8];
        for(uint j=0;j<8;j++) {
            uint shift=28-4*j;
            uint cw=shift ? ((current>>shift)|(previous<<(32-shift))) : current;
            w[j]=r55_decode_mcg(cw);
        }
        _Pragma("clang loop unroll(full)")
        for(uint s=0;s<S;s++) {
            if(s<cnt) {
                const device half* xx=x+ulong(slot[s])*p.input_dim+tk*16;
                float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
                for(uint j=0;j<8;j++)acc[s][j]=fma(a[j&3],w[j],acc[s][j]);
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        if(s<cnt) {
            float lo=(acc[s][0]+acc[s][1])+(acc[s][2]+acc[s][3]);
            float hi=(acc[s][4]+acc[s][5])+(acc[s][6]+acc[s][7]);
            lo+=simd_shuffle_xor(lo,1);hi+=simd_shuffle_xor(hi,1);
            lo+=simd_shuffle_xor(lo,2);hi+=simd_shuffle_xor(hi,2);
            if((lane&3)==0) {partial[s*64+warp*16+col]=lo;partial[s*64+warp*16+col+8]=hi;}
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        if(s<cnt && tid<16) {
            float sum=0;for(uint q=0;q<4;q++)sum+=partial[s*64+q*16+tid];
            output[(ulong(slot[s])*p.splits+split)*p.output_dim+ot*16+tid]=sum;
        }
    }
}
#define R55_GROUPED(S) \
kernel void r55_exl3_project_grouped_s##S(const device half* x [[buffer(0)]], \
    const device uint* trellis [[buffer(1)]], const device uint* table [[buffer(2)]], \
    device float* output [[buffer(3)]], constant R55ProjParams& p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float partial[S*64]; \
    r55_project_grouped<S>(x,trellis,table,output,p,group,tid,partial); \
}
R55_GROUPED(2)
R55_GROUPED(4)
R55_GROUPED(8)
// Slot-outer variant: one expert per threadgroup, its slots one after another. The
// weight tile is fetched from memory once and re-read from cache for the later slots;
// decoding is repeated. Registers and arithmetic per slot are exactly exl3_project's.
kernel void r55_exl3_project_grouped_loop(const device half* x [[buffer(0)]],
    const device uint* trellis [[buffer(1)]], const device uint* table [[buffer(2)]],
    device float* output [[buffer(3)]], constant R55ProjParams& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[64];
    uint g=group.y;
    if(g>=table[0])return;   // threadgroup-uniform
    const device uint* t=table+4+g*12;
    uint expert=t[0],cnt=t[1];
    uint ot=group.x,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.input_dim/16,nt=p.output_dim/16;
    uint per=(it+p.splits-1)/p.splits,begin=split*per,end=min(begin+per,it);
    uint row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(expert)*it*nt*32;
    for(uint s=0;s<cnt;s++) {
        uint slot=t[2+s];
        float acc[8]={};
        for(uint tk=begin+warp;tk<end;tk+=4) {
            const device uint* words=weights+(ulong(tk)*nt+ot)*32;
            uint previous=words[(lane+31)&31], current=words[lane];
            const device half* xx=x+ulong(slot)*p.input_dim+tk*16;
            float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
            for(uint j=0;j<8;j++) {
                uint shift=28-4*j;
                uint cw=shift ? ((current>>shift)|(previous<<(32-shift))) : current;
                acc[j]=fma(a[j&3],r55_decode_mcg(cw),acc[j]);
            }
        }
        float lo=(acc[0]+acc[1])+(acc[2]+acc[3]);
        float hi=(acc[4]+acc[5])+(acc[6]+acc[7]);
        lo+=simd_shuffle_xor(lo,1);hi+=simd_shuffle_xor(hi,1);
        lo+=simd_shuffle_xor(lo,2);hi+=simd_shuffle_xor(hi,2);
        if((lane&3)==0) {partial[warp*16+col]=lo;partial[warp*16+col+8]=hi;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<16) {
            float sum=0;for(uint q=0;q<4;q++)sum+=partial[q*16+tid];
            output[(ulong(slot)*p.splits+split)*p.output_dim+ot*16+tid]=sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// Exact-size buckets. Table (uint): [0..8) entries in bucket S=1..8; [8..32) indirect
// arguments (threadgroups x,y,z) for the gate/up shape per bucket; [32..56) for the down
// shape; entries of bucket S at 64 + (S-1)*bucket_stride*12, each: expert, S, slot[8], pad.
// An expert with c slots becomes pieces of 8 and one remainder, in slot order.
struct R55BucketParams { uint slots; uint experts; uint stride; uint gu_tiles; uint down_tiles; uint splits; uint pad0; uint pad1; };
kernel void r55_bucket_slots(const device uint* ids [[buffer(0)]], device uint* table [[buffer(1)]],
    constant R55BucketParams& p [[buffer(2)]], uint e [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup uint totals[8][32];
    uint mine[16];
    uint c = 0;
    if (e < p.experts) {
        for (uint s = 0; s < p.slots; s++) {
            if (ids[s] == e && c < 16) { mine[c] = s; c++; }
        }
    }
    // pieces: full 8s, then the remainder
    uint full = c / 8, rem = c % 8;
    uint before[8];
    for (uint b = 0; b < 8; b++) {
        uint n = (b == 7 ? full : 0) + (rem == b + 1 ? 1 : 0);
        before[b] = simd_prefix_exclusive_sum(n);
        uint tot = simd_sum(n);
        if (lane == 0) totals[b][sg] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base[8];
    for (uint b = 0; b < 8; b++) {
        uint acc = 0;
        for (uint i = 0; i < sg; i++) acc += totals[b][i];
        base[b] = acc + before[b];
    }
    uint lo = 0;
    for (uint j = 0; j < full; j++) {
        device uint* t = table + 64 + (7 * p.stride + base[7] + j) * 12;
        t[0] = e; t[1] = 8;
        for (uint i = 0; i < 8; i++) t[2 + i] = mine[lo + i];
        lo += 8;
    }
    if (rem) {
        uint b = rem - 1;
        device uint* t = table + 64 + (b * p.stride + base[b]) * 12;
        t[0] = e; t[1] = rem;
        for (uint i = 0; i < 8; i++) t[2 + i] = i < rem ? mine[lo + i] : 0u;
    }
    if (e == nsg * 32 - 1) {
        for (uint b = 0; b < 8; b++) {
            uint all = 0;
            for (uint i = 0; i < nsg; i++) all += totals[b][i];
            table[b] = all;
            table[8 + b * 3] = p.gu_tiles; table[8 + b * 3 + 1] = all; table[8 + b * 3 + 2] = p.splits;
            table[32 + b * 3] = p.down_tiles; table[32 + b * 3 + 1] = all; table[32 + b * 3 + 2] = p.splits;
        }
    }
}
template<uint S>
inline void r55_project_bucket(const device half* x, const device uint* trellis,
    const device uint* table, device float* output, constant R55ProjParams& p,
    uint3 group, uint tid, threadgroup float* partial) {
    const device uint* t=table+64+((S-1)*p.pad+group.y)*12;   // p.pad carries the bucket stride
    uint expert=t[0];
    uint slot[S];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)slot[s]=t[2+s];
    uint ot=group.x,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.input_dim/16,nt=p.output_dim/16;
    uint per=(it+p.splits-1)/p.splits,begin=split*per,end=min(begin+per,it);
    uint row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(expert)*it*nt*32;
    float acc[S][8];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)for(uint j=0;j<8;j++)acc[s][j]=0.;
    for(uint tk=begin+warp;tk<end;tk+=4) {
        const device uint* words=weights+(ulong(tk)*nt+ot)*32;
        uint previous=words[(lane+31)&31], current=words[lane];
        float w[8];
        for(uint j=0;j<8;j++) {
            uint shift=28-4*j;
            uint cw=shift ? ((current>>shift)|(previous<<(32-shift))) : current;
            w[j]=r55_decode_mcg(cw);
        }
        _Pragma("clang loop unroll(full)")
        for(uint s=0;s<S;s++) {
            const device half* xx=x+ulong(slot[s])*p.input_dim+tk*16;
            float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
            for(uint j=0;j<8;j++)acc[s][j]=fma(a[j&3],w[j],acc[s][j]);
        }
    }
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        float lo=(acc[s][0]+acc[s][1])+(acc[s][2]+acc[s][3]);
        float hi=(acc[s][4]+acc[s][5])+(acc[s][6]+acc[s][7]);
        lo+=simd_shuffle_xor(lo,1);hi+=simd_shuffle_xor(hi,1);
        lo+=simd_shuffle_xor(lo,2);hi+=simd_shuffle_xor(hi,2);
        if((lane&3)==0) {partial[s*64+warp*16+col]=lo;partial[s*64+warp*16+col+8]=hi;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        if(tid<16) {
            float sum=0;for(uint q=0;q<4;q++)sum+=partial[s*64+q*16+tid];
            output[(ulong(slot[s])*p.splits+split)*p.output_dim+ot*16+tid]=sum;
        }
    }
}
#define R55_BUCKET(S) \
kernel void r55_exl3_project_bucket_s##S(const device half* x [[buffer(0)]], \
    const device uint* trellis [[buffer(1)]], const device uint* table [[buffer(2)]], \
    device float* output [[buffer(3)]], constant R55ProjParams& p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float partial[S*64]; \
    r55_project_bucket<S>(x,trellis,table,output,p,group,tid,partial); \
}
R55_BUCKET(1) R55_BUCKET(2) R55_BUCKET(3) R55_BUCKET(4)
R55_BUCKET(5) R55_BUCKET(6) R55_BUCKET(7) R55_BUCKET(8)
// Latency variant: each warp issues the weight loads of U K-tiles before decoding them,
// so U lines are in flight per warp instead of one. Per-slot FMA order (tk ascending in
// steps of 4 per warp) is unchanged.
template<uint S,uint U>
inline void r55_project_bucket_u(const device half* x, const device uint* trellis,
    const device uint* table, device float* output, constant R55ProjParams& p,
    uint3 group, uint tid, threadgroup float* partial) {
    const device uint* t=table+64+((S-1)*p.pad+group.y)*12;
    uint expert=t[0];
    uint slot[S];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)slot[s]=t[2+s];
    uint ot=group.x,split=group.z,lane=tid&31,warp=tid>>5;
    uint it=p.input_dim/16,nt=p.output_dim/16;
    uint per=(it+p.splits-1)/p.splits,begin=split*per,end=min(begin+per,it);
    uint row=(lane&3)*2,col=lane>>2;
    const device uint* weights=trellis+ulong(expert)*it*nt*32;
    float acc[S][8];
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++)for(uint j=0;j<8;j++)acc[s][j]=0.;
    for(uint tk0=begin+warp;tk0<end;tk0+=4*U) {
        uint prev[U],cur[U];
        _Pragma("clang loop unroll(full)")
        for(uint u=0;u<U;u++) {
            uint tk=tk0+4*u;
            if(tk<end) {
                const device uint* words=weights+(ulong(tk)*nt+ot)*32;
                prev[u]=words[(lane+31)&31];cur[u]=words[lane];
            }
        }
        _Pragma("clang loop unroll(full)")
        for(uint u=0;u<U;u++) {
            uint tk=tk0+4*u;
            if(tk<end) {
                float w[8];
                for(uint j=0;j<8;j++) {
                    uint shift=28-4*j;
                    uint cw=shift ? ((cur[u]>>shift)|(prev[u]<<(32-shift))) : cur[u];
                    w[j]=r55_decode_mcg(cw);
                }
                _Pragma("clang loop unroll(full)")
                for(uint s=0;s<S;s++) {
                    const device half* xx=x+ulong(slot[s])*p.input_dim+tk*16;
                    float a[4]={float(xx[row]),float(xx[row+1]),float(xx[row+8]),float(xx[row+9])};
                    for(uint j=0;j<8;j++)acc[s][j]=fma(a[j&3],w[j],acc[s][j]);
                }
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        float lo=(acc[s][0]+acc[s][1])+(acc[s][2]+acc[s][3]);
        float hi=(acc[s][4]+acc[s][5])+(acc[s][6]+acc[s][7]);
        lo+=simd_shuffle_xor(lo,1);hi+=simd_shuffle_xor(hi,1);
        lo+=simd_shuffle_xor(lo,2);hi+=simd_shuffle_xor(hi,2);
        if((lane&3)==0) {partial[s*64+warp*16+col]=lo;partial[s*64+warp*16+col+8]=hi;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for(uint s=0;s<S;s++) {
        if(tid<16) {
            float sum=0;for(uint q=0;q<4;q++)sum+=partial[s*64+q*16+tid];
            output[(ulong(slot[s])*p.splits+split)*p.output_dim+ot*16+tid]=sum;
        }
    }
}
#define R55_BUCKET_U(S,U) \
kernel void r55_exl3_project_bucket_u##U##_s##S(const device half* x [[buffer(0)]], \
    const device uint* trellis [[buffer(1)]], const device uint* table [[buffer(2)]], \
    device float* output [[buffer(3)]], constant R55ProjParams& p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float partial[S*64]; \
    r55_project_bucket_u<S,U>(x,trellis,table,output,p,group,tid,partial); \
}
R55_BUCKET_U(1,2) R55_BUCKET_U(2,2) R55_BUCKET_U(3,2) R55_BUCKET_U(4,2)
R55_BUCKET_U(5,2) R55_BUCKET_U(6,2) R55_BUCKET_U(7,2) R55_BUCKET_U(8,2)
R55_BUCKET_U(1,4) R55_BUCKET_U(2,4) R55_BUCKET_U(3,4) R55_BUCKET_U(4,4)
R55_BUCKET_U(5,4) R55_BUCKET_U(6,4) R55_BUCKET_U(7,4) R55_BUCKET_U(8,4)
