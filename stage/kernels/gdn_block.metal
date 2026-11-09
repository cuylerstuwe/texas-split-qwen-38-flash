// Adapted from Sushi gdn_decode.zig K1S and transformer.zig prework/norm-gate,
// originating in mlx-serve / MLX. See NOTICE.txt. Native buffer ABI, FP32 state,
// staged/fused controls and 1/4/8/16-row bounded kernels added here.
#include <metal_stdlib>
using namespace metal;
constant constexpr uint DK=128, DV=128, HK=16, HV=48, C=10240, V=6144, ST=HV*DV*DK;
struct GdnParams { uint rows; };
inline bfloat bf_sigmoid(bfloat x) {
    // Native MSL bfloat lacks the overloaded math helpers supplied by MLX's
    // generated header. Preserve those BF16 arithmetic boundaries explicitly.
    bfloat e=bfloat(metal::exp(metal::abs(float(x))));
    bfloat denominator=bfloat(1.f+float(e));
    bfloat small=bfloat(1.f/float(denominator));
    return float(x)<0.f?small:bfloat(1.f-float(small));
}
inline float gdn_log1p(float x) {
    float u=1.f+x;
    if(u==metal::numeric_limits<float>::max())return metal::numeric_limits<float>::max();
    if(u==1.f)return x;
    return x*(metal::log(u)/(u-1.f));
}

// Each SIMD group prepares q, k, v, or scalar gates. A value head maps to
// hk=hv/3; only one of those heads writes the shared convolution channels.
inline void prepare(const device bfloat* qkv,const device bfloat* a,
    const device bfloat* b,const device bfloat* conv_in,const device bfloat* cw,
    const device bfloat* alog,const device bfloat* dt,device bfloat* conv_out,
    uint allrows,uint start,uint count,uint hv,uint part,uint tid,
    threadgroup float* qs,threadgroup float* ks,threadgroup float* vs,threadgroup float* gates) {
    uint lane=tid&31,sg=tid>>5,hk=hv/3;
    if(sg<3) {
        uint cb=sg==0?hk*DK:(sg==1?HK*DK+hk*DK:2*HK*DK+hv*DV);
        for(uint t=0;t<count;t++) {
            bfloat act[4]; float sum=0.f;
            for(uint i=0;i<4;i++) {
                uint ch=cb+lane*4+i; float acc=0.f;
                for(uint tap=0;tap<4;tap++) {
                    uint w=start+t+tap;
                    bfloat x=w<3?conv_in[w*C+ch]:qkv[(w-3)*C+ch];
                    acc+=float(x)*float(cw[ch*4+tap]);
                }
                bfloat cv=bfloat(acc);act[i]=cv*bf_sigmoid(cv);
                sum+=float(act[i])*float(act[i]);
            }
            if(sg<2) {
                sum=simd_sum(sum); float inv=metal::precise::rsqrt(sum/float(DK)+1e-6f);
                bfloat scale=sg==0?bfloat(1.f/128.f):bfloat(1.f/metal::sqrt(128.f));
                threadgroup float* dst=sg==0?qs:ks;
                for(uint i=0;i<4;i++)dst[t*DK+lane*4+i]=float(scale*bfloat(float(act[i])*inv));
            } else for(uint i=0;i<4;i++)vs[t*DV+lane*4+i]=float(act[i]);
        }
        if(part==0 && start+count==allrows && (sg==2 || hv%3==0)) {
            for(uint i=0;i<4;i++)for(uint j=0;j<3;j++) {
                uint ch=cb+lane*4+i,w=allrows+j;
                conv_out[j*C+ch]=w<3?conv_in[w*C+ch]:qkv[(w-3)*C+ch];
            }
        }
    }
    if(sg==3 && lane==31) {
        float ea=metal::precise::exp(float(alog[hv]));
        for(uint t=0;t<count;t++) {
            uint row=start+t; float apd=float(a[row*HV+hv])+float(dt[hv]);
            gates[t*2]=metal::precise::exp(-ea*gdn_log1p(metal::precise::exp(apd)));
            gates[t*2+1]=float(bf_sigmoid(b[row*HV+hv]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define GDN_ARGS \
 const device bfloat* qkv [[buffer(0)]],const device bfloat* z [[buffer(1)]], \
 const device bfloat* a [[buffer(2)]],const device bfloat* b [[buffer(3)]], \
 const device bfloat* conv_in [[buffer(4)]],const device float* state_in [[buffer(5)]], \
 const device bfloat* cw [[buffer(6)]],const device bfloat* alog [[buffer(7)]], \
 const device bfloat* dt [[buffer(8)]],const device bfloat* norm [[buffer(9)]], \
 device bfloat* q [[buffer(10)]],device bfloat* k [[buffer(11)]],device bfloat* v [[buffer(12)]], \
 device float* gate [[buffer(13)]],device bfloat* y [[buffer(14)]],device bfloat* gated [[buffer(15)]], \
 device float* states [[buffer(16)]],device bfloat* conv_out [[buffer(17)]], \
 constant GdnParams& p [[buffer(18)]],uint3 grid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]

kernel void gdn_prepare(GDN_ARGS) {
    uint hv=grid.x,t=grid.y,hk=hv/3,lane=tid&31,sg=tid>>5;
    threadgroup float qs[128],ks[128],vs[128],gb[2];
    prepare(qkv,a,b,conv_in,cw,alog,dt,conv_out,p.rows,t,1,hv,0,tid,qs,ks,vs,gb);
    if(sg<3)for(uint i=0;i<4;i++) {
        uint j=lane*4+i;
        if(sg==0 && hv%3==0)q[(t*HK+hk)*DK+j]=bfloat(qs[j]);
        if(sg==1 && hv%3==0)k[(t*HK+hk)*DK+j]=bfloat(ks[j]);
        if(sg==2)v[(t*HV+hv)*DV+j]=bfloat(vs[j]);
    }
    if(tid==127) {gate[(t*HV+hv)*2]=gb[0];gate[(t*HV+hv)*2+1]=gb[1];}
}

kernel void gdn_recur(GDN_ARGS) {
    uint hv=grid.x/4,part=grid.x%4,hk=hv/3,lane=tid&31,sg=tid>>5,row0=part*32+sg*8;
    float st[8][4];
    for(uint r=0;r<8;r++)for(uint j=0;j<4;j++)st[r][j]=state_in[(hv*DV+row0+r)*DK+lane*4+j];
    for(uint t=0;t<p.rows;t++) {
        float qq[4],kk[4];
        for(uint j=0;j<4;j++) {qq[j]=float(q[(t*HK+hk)*DK+lane*4+j]);kk[j]=float(k[(t*HK+hk)*DK+lane*4+j]);}
        float g=gate[(t*HV+hv)*2],beta=gate[(t*HV+hv)*2+1];
        for(uint r=0;r<8;r++) {
            float memory=0.f;
            for(uint j=0;j<4;j++) {st[r][j]*=g;memory+=st[r][j]*kk[j];}
            memory=simd_sum(memory);
            float delta=beta*(float(v[(t*HV+hv)*DV+row0+r])-memory),out=0.f;
            for(uint j=0;j<4;j++) {st[r][j]+=delta*kk[j];out+=st[r][j]*qq[j];states[t*ST+(hv*DV+row0+r)*DK+lane*4+j]=st[r][j];}
            out=simd_sum(out);if(lane==0)y[(t*HV+hv)*DV+row0+r]=bfloat(out);
        }
    }
}

inline void fused(const device float* state_in,device float* states,device bfloat* y,
    uint count,uint hv,uint part,uint tid,threadgroup float* qs,threadgroup float* ks,
    threadgroup float* vs,threadgroup float* gates) {
    uint lane=tid&31,sg=tid>>5,row0=part*32+sg*8;
    float st[8][4];
    for(uint r=0;r<8;r++)for(uint j=0;j<4;j++)st[r][j]=state_in[(hv*DV+row0+r)*DK+lane*4+j];
    for(uint t=0;t<count;t++) {
        float qq[4],kk[4];
        for(uint j=0;j<4;j++){qq[j]=qs[t*DK+lane*4+j];kk[j]=ks[t*DK+lane*4+j];}
        float g=gates[t*2],beta=gates[t*2+1];
        for(uint r=0;r<8;r++) {
            float memory=0.f;
            for(uint j=0;j<4;j++){st[r][j]*=g;memory+=st[r][j]*kk[j];}
            memory=simd_sum(memory);float delta=beta*(vs[t*DV+row0+r]-memory),out=0.f;
            for(uint j=0;j<4;j++){st[r][j]+=delta*kk[j];out+=st[r][j]*qq[j];states[t*ST+(hv*DV+row0+r)*DK+lane*4+j]=st[r][j];}
            out=simd_sum(out);if(lane==0)y[(t*HV+hv)*DV+row0+r]=bfloat(out);
        }
    }
}
#define GDN_FUSED(NAME,WIDTH) \
kernel void NAME(GDN_ARGS) { \
 uint hv=grid.x/4,part=grid.x%4; \
 threadgroup float qs[WIDTH*128],ks[WIDTH*128],vs[WIDTH*128],gb[WIDTH*2]; \
 prepare(qkv,a,b,conv_in,cw,alog,dt,conv_out,p.rows,0,p.rows,hv,part,tid,qs,ks,vs,gb); \
 fused(state_in,states,y,p.rows,hv,part,tid,qs,ks,vs,gb); \
}
GDN_FUSED(gdn_fused1,1)
GDN_FUSED(gdn_fused4,4)
GDN_FUSED(gdn_fused8,8)
GDN_FUSED(gdn_fused16,16)

kernel void gdn_norm_gate(GDN_ARGS) {
    uint t=grid.y,hv=grid.x,lane=tid,base=(t*HV+hv)*DV+lane*4;
    float xs[4],sum=0.f;
    for(uint j=0;j<4;j++){xs[j]=float(y[base+j]);sum+=xs[j]*xs[j];}
    sum=simd_sum(sum);float inv=metal::precise::rsqrt(sum/float(DV)+1e-6f);
    for(uint j=0;j<4;j++) {
        bfloat n=norm[lane*4+j]*bfloat(xs[j]*inv);
        gated[base+j]=n*bf_sigmoid(z[base+j]);
    }
}
