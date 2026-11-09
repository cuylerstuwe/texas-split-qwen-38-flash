// R05c prepared-operand GDN capture/commit experiment.
// SIMD recurrence adapted from this project's gdn_block.metal / Sushi K1S;
// see NOTICE.txt. Host orchestration and compact capture/replay are experimental.
#include <metal_stdlib>
using namespace metal;
constant constexpr uint D=128, HV=48, ST=HV*D*D;
struct ReplayParams { uint rows, accepted, mode; };
#define REPLAY_ARGS \
 const device float* initial [[buffer(0)]],const device float* q [[buffer(1)]], \
 const device float* k [[buffer(2)]],const device float* v [[buffer(3)]], \
 const device float* decay [[buffer(4)]],const device float* beta [[buffer(5)]], \
 device float* y [[buffer(6)]],device float* snapshots [[buffer(7)]], \
 device float* final_state [[buffer(8)]],device float* pseudo [[buffer(9)]], \
 device float* committed [[buffer(10)]],constant ReplayParams& p [[buffer(11)]], \
 uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]

// Identical recurrent arithmetic in all forward arms. Only capture stores differ.
// Input/output vectors are head,token,channel; snapshots are token,head,value,key.
template<bool StoreAll,bool StorePseudo,bool CommitOnly>
inline void recurrent(const device float* initial,const device float* q,
    const device float* k,const device float* v,const device float* decay,
    const device float* beta,device float* y,device float* snapshots,
    device float* final_state,device float* pseudo,device float* committed,
    uint rows,uint accepted,uint group,uint tid) {
    uint h=group/4,part=group%4,lane=tid&31,sg=tid>>5,r0=part*32+sg*8;
    float st[8][4];
    for(uint r=0;r<8;r++)for(uint j=0;j<4;j++)st[r][j]=initial[(h*D+r0+r)*D+lane*4+j];
    uint steps=CommitOnly?accepted:rows;
    for(uint t=0;t<steps;t++) {
        float kk[4],qq[4];uint at=(h*rows+t)*D;
        for(uint j=0;j<4;j++) {kk[j]=k[at+lane*4+j];if(!CommitOnly)qq[j]=q[at+lane*4+j];}
        float a=decay[h*rows+t],b=beta[h*rows+t];
        for(uint r=0;r<8;r++) {
            float memory=0.f;
            for(uint j=0;j<4;j++){st[r][j]*=a;memory+=st[r][j]*kk[j];}
            memory=simd_sum(memory);
            float delta=b*(v[at+r0+r]-memory),out=0.f;
            if(StorePseudo && lane==0)pseudo[at+r0+r]=delta;
            for(uint j=0;j<4;j++) {
                st[r][j]+=delta*kk[j];
                if(!CommitOnly)out+=st[r][j]*qq[j];
                if(StoreAll)snapshots[t*ST+(h*D+r0+r)*D+lane*4+j]=st[r][j];
            }
            if(!CommitOnly){out=simd_sum(out);if(lane==0)y[at+r0+r]=out;}
        }
    }
    if(CommitOnly || !StoreAll)for(uint r=0;r<8;r++)for(uint j=0;j<4;j++) {
        uint index=(h*D+r0+r)*D+lane*4+j;
        if(CommitOnly)committed[index]=st[r][j];else final_state[index]=st[r][j];
    }
}
#define FORWARD(NAME,ALL,PSEUDO) \
kernel void NAME(REPLAY_ARGS) {recurrent<ALL,PSEUDO,false>(initial,q,k,v,decay,beta,y,snapshots,final_state,pseudo,committed,p.rows,p.accepted,group,tid);}
FORWARD(gdn_capture_all,true,false)
FORWARD(gdn_capture_final,false,false)
FORWARD(gdn_capture_compact,false,true)

kernel void gdn_commit(REPLAY_ARGS) {
    uint h=group/4,part=group%4,lane=tid&31,sg=tid>>5,r0=part*32+sg*8;
    // Zero/full acceptance and the snapshot arm need only a state copy.
    if(p.accepted==0 || p.mode==0 || p.accepted==p.rows) {
        const device float* source=p.accepted==0?initial:
            (p.mode==0?snapshots+(p.accepted-1)*ST:final_state);
        for(uint r=0;r<8;r++)for(uint j=0;j<4;j++) {
            uint index=(h*D+r0+r)*D+lane*4+j;committed[index]=source[index];
        }
        return;
    }
    if(p.mode==1) {
        // Same recurrence and prepared inputs, but only the accepted prefix;
        // output/query work is unnecessary when rebuilding a committed state.
        recurrent<false,false,true>(initial,q,k,v,decay,beta,y,snapshots,final_state,pseudo,committed,p.rows,p.accepted,group,tid);
        return;
    }
    // Rank-one reconstruction without division by tiny/zero decay products.
    // Reordered FP32 arithmetic is a named candidate, not a bitwise promise.
    float weights[16],factor=1.f;
    for(uint n=p.accepted;n>0;n--){uint t=n-1;weights[t]=factor;factor*=decay[h*p.rows+t];}
    float st[8][4];
    for(uint r=0;r<8;r++)for(uint j=0;j<4;j++)st[r][j]=initial[(h*D+r0+r)*D+lane*4+j]*factor;
    for(uint t=0;t<p.accepted;t++) {
        uint at=(h*p.rows+t)*D;float kk[4];
        for(uint j=0;j<4;j++)kk[j]=k[at+lane*4+j];
        for(uint r=0;r<8;r++) {
            float u=weights[t]*pseudo[at+r0+r];
            for(uint j=0;j<4;j++)st[r][j]+=u*kk[j];
        }
    }
    for(uint r=0;r<8;r++)for(uint j=0;j<4;j++) {
        uint index=(h*D+r0+r)*D+lane*4+j;committed[index]=st[r][j];
    }
}
