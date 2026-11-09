// Native Metal router + BF16 expert input + shared branch. Router semantics
// follow Sushi (MIT, NOTICE.txt) and our qualified native lowbit_router.cu.
// No CPU route planning, floating-point expert expansion, or model runtime.
inline float tm_load(ushort x) { return as_type<float>(uint(x)<<16); }
inline ushort tm_store(float x) {
    uint u=as_type<uint>(x); return ushort((u+0x7fffu+((u>>16)&1u))>>16);
}
inline float tm_rb(float x) { return tm_load(tm_store(x)); }
inline float tm_sigmoid(float x) {
    float small=tm_rb(1.0f/tm_rb(1.0f+tm_rb(exp(abs(x)))));
    return x<0.0f?small:tm_rb(1.0f-small);
}
kernel void tm_router_qmv(const device ushort* x [[buffer(0)]],
    const device ushort* w [[buffer(1)]],device float* y [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.y,e=group.x*4+sg; float acc=0;
    for(uint k=lane;k<2560;k+=32)acc=fma(tm_load(x[row*2560+k]),tm_load(w[e*2560+k]),acc);
    acc=simd_sum(acc);
    if(lane==0)y[row*512+e]=acc;
}
kernel void tm_router_select(const device float* raw [[buffer(0)]],
    device ushort* logits [[buffer(1)]],device uint* ids [[buffer(2)]],
    device float* scores [[buffer(3)]],constant uint& arm [[buffer(4)]],
    constant float& prune [[buffer(5)]],constant uint& renorm [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float key[16],mx=-INFINITY; bool invalid=false;
    for(uint j=0;j<16;j++) {
        uint e=j*32+lane; float v=raw[row*512+e]; ushort h=tm_store(v);
        logits[row*512+e]=h;
        bool bad=!isfinite(v)||!isfinite(tm_load(h)); invalid|=bad;
        key[j]=bad?-3.402823466e38f:tm_load(h); mx=max(mx,key[j]);
    }
    invalid=simd_any(invalid); mx=simd_max(mx);
    float norm=0;
    if(arm==0) {
        for(uint j=0;j<16;j++)norm+=exp(key[j]-mx);
        norm=1.0f/simd_sum(norm);
    }
    uint selected=0; float selected_logit=0;
    for(uint rank=0;rank<10;rank++) {
        float best=-INFINITY; uint id=0xffffffffu;
        for(uint j=0;j<16;j++) {
            uint e=j*32+lane;
            if(key[j]>best||(key[j]==best&&e<id)){best=key[j];id=e;}
        }
        for(ushort s=16;s;s>>=1) {
            float v=simd_shuffle_xor(best,s);uint e=simd_shuffle_xor(id,s);
            if(v>best||(v==best&&e<id)){best=v;id=e;}
        }
        if(lane==rank){selected=id;selected_logit=best;}
        for(uint j=0;j<16;j++)if(j*32+lane==id)key[j]=-INFINITY;
    }
    float weight=lane<10?exp(selected_logit-mx):0;
    if(arm==0)weight=tm_rb(weight*norm);
    float total=0;
    for(ushort j=0;j<10;j++) {
        float w=simd_shuffle(weight,j);total=arm?total+w:tm_rb(total+w);
    }
    float score=arm?weight/total:tm_rb(weight/total);
    // R58 lead 1 (prune < 1): keep the shortest rank prefix of experts whose cumulative weight
    // reaches `prune`; the rest get weight 0 (renorm: the kept weights are rescaled to sum to 1).
    // Lanes are in rank order, so a lane is kept while the weight before it is below the threshold.
    if(prune<1.0f) {
        float before=0,kept=0;
        for(ushort j=0;j<10;j++) {
            float sj=simd_shuffle(score,j);
            if(j<lane)before+=sj;
        }
        bool keep=lane<10&&before<prune;
        for(ushort j=0;j<10;j++)kept+=simd_shuffle(keep?score:0.0f,j);
        score=keep?(renorm?score/kept:score):0.0f;
    }
    if(lane<10) {
        ids[row*10+lane]=selected;
        scores[row*10+lane]=invalid?as_type<float>(0x7fc00000u):score;
    }
}
kernel void tm_prepare_pair(const device ushort* x [[buffer(0)]],
    const device half* suh_g [[buffer(1)]],const device half* suh_u [[buffer(2)]],
    const device uint* ids [[buffer(3)]],const device uint* order [[buffer(4)]],
    device half* gate [[buffer(5)]],device half* up [[buffer(6)]],constant MoeParams& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint slot=group.y,base=group.x*128;
    ulong xb=ulong(order[slot]/p.topk)*p.hidden+base;
    ulong sb=ulong(ids[slot])*p.hidden+base,yb=ulong(slot)*p.hidden+base;
    float4 v=float4(tm_load(x[xb+lane]),tm_load(x[xb+lane+32]),tm_load(x[xb+lane+64]),tm_load(x[xb+lane+96]));
    moe_store_half4(gate,yb,lane,moe_h128(v*moe_half4(suh_g,sb,lane),lane));
    moe_store_half4(up,yb,lane,moe_h128(v*moe_half4(suh_u,sb,lane),lane));
}
kernel void tm_shared_gate(const device ushort* x [[buffer(0)]],
    const device ushort* w [[buffer(1)]],device float* raw [[buffer(2)]],
    device ushort* y [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float acc=0;
    for(uint k=lane;k<2560;k+=32)acc=fma(tm_load(x[row*2560+k]),tm_load(w[k]),acc);
    acc=simd_sum(acc);
    if(lane==0){raw[row]=acc;y[row]=tm_store(acc);}
}
kernel void tm_shared_activate(const device ushort* gate [[buffer(0)]],
    const device ushort* up [[buffer(1)]],device ushort* middle [[buffer(2)]],
    constant uint& n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=n)return;
    float g=tm_load(gate[i]),u=tm_load(up[i]);
    middle[i]=tm_store(tm_rb(g*tm_sigmoid(g))*u);
}
// R58: moe_finish with tm_shared_join folded in (one dispatch and one barrier fewer per layer).
// The routed sum and its RNE BF16 are moe_finish's; the join reads that BF16 back exactly as
// tm_shared_join does, so every output is bit-identical to the two-kernel path.
kernel void moe_finish_join(const device float* down [[buffer(0)]],const device half* svh_d [[buffer(1)]],
    const device uint* ids [[buffer(2)]],const device uint* inverse [[buffer(3)]],
    const device float* scores [[buffer(4)]],device float* out [[buffer(5)]],
    device ushort* out_bf16 [[buffer(6)]],constant MoeParams& p [[buffer(7)]],
    const device ushort* sdown [[buffer(8)]],const device ushort* gate_logits [[buffer(9)]],
    device ushort* gate_sigmoid [[buffer(10)]],device ushort* shared [[buffer(11)]],
    device ushort* output [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float vals[10*128];
    uint row=group.y,base=group.x*128,slot=inverse[row*p.topk+sg];
    ulong sb=ulong(ids[slot])*p.hidden+base;
    float4 v=moe_h128(moe_sum4(down,slot,p.hidden,p.down_splits,base,lane),lane)*moe_half4(svh_d,sb,lane);
    vals[sg*128+lane]=v.x;vals[sg*128+lane+32]=v.y;
    vals[sg*128+lane+64]=v.z;vals[sg*128+lane+96]=v.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float4 sum=0;
        for(uint k=0;k<p.topk;k++) {
            ulong b=k*128+lane;
            sum+=float4(vals[b],vals[b+32],vals[b+64],vals[b+96])*scores[row*p.topk+k];
        }
        float sig=tm_sigmoid(tm_load(gate_logits[row]));
        if(base==0 && lane==0)gate_sigmoid[row]=tm_store(sig);
        for(uint j=0;j<4;j++) {
            ulong at=ulong(row)*p.hidden+base+lane+j*32;
            out[at]=sum[j];uint bits=as_type<uint>(sum[j]);
            ushort rb=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
            out_bf16[at]=rb;
            ushort s=tm_store(tm_load(sdown[at])*sig);shared[at]=s;
            output[at]=tm_store(tm_load(rb)+tm_load(s));
        }
    }
}
kernel void tm_shared_join(const device ushort* down [[buffer(0)]],
    const device ushort* gate_logits [[buffer(1)]],const device ushort* routed [[buffer(2)]],
    device ushort* gate_sigmoid [[buffer(3)]],device ushort* shared [[buffer(4)]],
    device ushort* output [[buffer(5)]],constant uint& n [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=n)return;
    uint row=i/2560;float sig=tm_sigmoid(tm_load(gate_logits[row]));
    if(i%2560==0)gate_sigmoid[row]=tm_store(sig);
    ushort s=tm_store(tm_load(down[i])*sig);shared[i]=s;
    output[i]=tm_store(tm_load(routed[i])+tm_load(s));
}
// R58: tm_router_select folded into tm_prepare_pair, for the identity slot order (slot = row * 10
// + rank; the ungrouped path). Every prepare simdgroup runs its row's selection with
// tm_router_select's exact arithmetic and takes its slot's expert by shuffle. The row's first
// threadgroup (block 0, rank 0) publishes the BF16 logits, ids and scores exactly as
// tm_router_select does. One dispatch and one barrier fewer per layer; bit-identical outputs.
kernel void tm_select_prepare_pair(const device float* raw [[buffer(0)]],
    device ushort* logits [[buffer(1)]],device uint* ids [[buffer(2)]],
    device float* scores [[buffer(3)]],constant uint& arm [[buffer(4)]],
    const device ushort* x [[buffer(5)]],const device half* suh_g [[buffer(6)]],
    const device half* suh_u [[buffer(7)]],const device uint* order [[buffer(8)]],
    device half* gate [[buffer(9)]],device half* up [[buffer(10)]],constant MoeParams& p [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint slot=group.y,base=group.x*128,row=slot/10,rank=slot-row*10;
    bool pub=group.x==0&&rank==0;
    float key[16],mx=-INFINITY; bool invalid=false;
    for(uint j=0;j<16;j++) {
        uint e=j*32+lane; float v=raw[row*512+e]; ushort h=tm_store(v);
        if(pub)logits[row*512+e]=h;
        bool bad=!isfinite(v)||!isfinite(tm_load(h)); invalid|=bad;
        key[j]=bad?-3.402823466e38f:tm_load(h); mx=max(mx,key[j]);
    }
    invalid=simd_any(invalid); mx=simd_max(mx);
    float norm=0;
    if(arm==0) {
        for(uint j=0;j<16;j++)norm+=exp(key[j]-mx);
        norm=1.0f/simd_sum(norm);
    }
    uint selected=0; float selected_logit=0;
    for(uint r=0;r<10;r++) {
        float best=-INFINITY; uint id=0xffffffffu;
        for(uint j=0;j<16;j++) {
            uint e=j*32+lane;
            if(key[j]>best||(key[j]==best&&e<id)){best=key[j];id=e;}
        }
        for(ushort s=16;s;s>>=1) {
            float v=simd_shuffle_xor(best,s);uint e=simd_shuffle_xor(id,s);
            if(v>best||(v==best&&e<id)){best=v;id=e;}
        }
        if(lane==r){selected=id;selected_logit=best;}
        for(uint j=0;j<16;j++)if(j*32+lane==id)key[j]=-INFINITY;
    }
    if(pub) {
        float weight=lane<10?exp(selected_logit-mx):0;
        if(arm==0)weight=tm_rb(weight*norm);
        float total=0;
        for(ushort j=0;j<10;j++) {
            float w=simd_shuffle(weight,j);total=arm?total+w:tm_rb(total+w);
        }
        if(lane<10) {
            ids[row*10+lane]=selected;
            scores[row*10+lane]=invalid?as_type<float>(0x7fc00000u):(arm?weight/total:tm_rb(weight/total));
        }
    }
    uint expert=simd_shuffle(selected,rank);
    ulong xb=ulong(order[slot]/p.topk)*p.hidden+base;
    ulong sb=ulong(expert)*p.hidden+base,yb=ulong(slot)*p.hidden+base;
    float4 v=float4(tm_load(x[xb+lane]),tm_load(x[xb+lane+32]),tm_load(x[xb+lane+64]),tm_load(x[xb+lane+96]));
    moe_store_half4(gate,yb,lane,moe_h128(v*moe_half4(suh_g,sb,lane),lane));
    moe_store_half4(up,yb,lane,moe_h128(v*moe_half4(suh_u,sb,lane),lane));
}
// R58: the row-grouped form of tm_select_prepare_pair: one threadgroup of 10 simdgroups per
// (128-block, row). Simdgroup 0 runs tm_router_select's exact arithmetic once and shares the 10
// ids through threadgroup memory (block 0 also publishes the BF16 logits, ids and scores); then
// simdgroup r prepares slot row * 10 + r. 20 selections per row instead of 200; bit-identical.
kernel void tm_select_prepare_rows(const device float* raw [[buffer(0)]],
    device ushort* logits [[buffer(1)]],device uint* ids [[buffer(2)]],
    device float* scores [[buffer(3)]],constant uint& arm [[buffer(4)]],
    const device ushort* x [[buffer(5)]],const device half* suh_g [[buffer(6)]],
    const device half* suh_u [[buffer(7)]],const device uint* order [[buffer(8)]],
    device half* gate [[buffer(9)]],device half* up [[buffer(10)]],constant MoeParams& p [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint chosen[10];
    uint row=group.y,base=group.x*128;
    if(sg==0) {
        bool pub=group.x==0;
        float key[16],mx=-INFINITY; bool invalid=false;
        for(uint j=0;j<16;j++) {
            uint e=j*32+lane; float v=raw[row*512+e]; ushort h=tm_store(v);
            if(pub)logits[row*512+e]=h;
            bool bad=!isfinite(v)||!isfinite(tm_load(h)); invalid|=bad;
            key[j]=bad?-3.402823466e38f:tm_load(h); mx=max(mx,key[j]);
        }
        invalid=simd_any(invalid); mx=simd_max(mx);
        float norm=0;
        if(arm==0) {
            for(uint j=0;j<16;j++)norm+=exp(key[j]-mx);
            norm=1.0f/simd_sum(norm);
        }
        uint selected=0; float selected_logit=0;
        for(uint r=0;r<10;r++) {
            float best=-INFINITY; uint id=0xffffffffu;
            for(uint j=0;j<16;j++) {
                uint e=j*32+lane;
                if(key[j]>best||(key[j]==best&&e<id)){best=key[j];id=e;}
            }
            for(ushort s=16;s;s>>=1) {
                float v=simd_shuffle_xor(best,s);uint e=simd_shuffle_xor(id,s);
                if(v>best||(v==best&&e<id)){best=v;id=e;}
            }
            if(lane==r){selected=id;selected_logit=best;}
            for(uint j=0;j<16;j++)if(j*32+lane==id)key[j]=-INFINITY;
        }
        if(lane<10)chosen[lane]=selected;
        if(pub) {
            float weight=lane<10?exp(selected_logit-mx):0;
            if(arm==0)weight=tm_rb(weight*norm);
            float total=0;
            for(ushort j=0;j<10;j++) {
                float w=simd_shuffle(weight,j);total=arm?total+w:tm_rb(total+w);
            }
            if(lane<10) {
                ids[row*10+lane]=selected;
                scores[row*10+lane]=invalid?as_type<float>(0x7fc00000u):(arm?weight/total:tm_rb(weight/total));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot=row*10+sg;
    ulong xb=ulong(order[slot]/p.topk)*p.hidden+base;
    ulong sb=ulong(chosen[sg])*p.hidden+base,yb=ulong(slot)*p.hidden+base;
    float4 v=float4(tm_load(x[xb+lane]),tm_load(x[xb+lane+32]),tm_load(x[xb+lane+64]),tm_load(x[xb+lane+96]));
    moe_store_half4(gate,yb,lane,moe_h128(v*moe_half4(suh_g,sb,lane),lane));
    moe_store_half4(up,yb,lane,moe_h128(v*moe_half4(suh_u,sb,lane),lane));
}
