// CPU-only encoding gate; hardware scheduling semantics still need a probe.
#include "libtinynv/src/qmd.h"
#include "libtinynv/src/nv_regs.h"
#include <assert.h>
#include <stdio.h>
#define GET(q,f) tinynv_qmd_get(q,TINYNV_QMD_##f##_LO,TINYNV_QMD_##f##_HI)
int main(void) {
  tinynv_qmd_t a={0},s={0},b={0},c={0},j={0};
  assert(!tinynv_qmd_r44_edge(&a,&s,1,1,1));
  assert(!tinynv_qmd_r44_edge(&s,&b,2,1,1));
  assert(!tinynv_qmd_r44_edge(&c,&j,4,2,0));
  assert(GET(&a,ARRIVE_AT_LATCH_ID)==1 && GET(&a,PRE_EXIT_AT_LAST_CTA_LAUNCH)==1);
  assert(GET(&s,ARRIVE_AT_LATCH_ID)==2 && GET(&s,PRE_EXIT_AT_LAST_CTA_LAUNCH)==1);
  assert(GET(&b,WAIT_ON_LATCH_ID)==1 && GET(&b,WAIT_ON_LATCH_VALID)==1);
  assert(GET(&b,PRE_EXIT_AT_LAST_CTA_LAUNCH)==0 && GET(&c,PRE_EXIT_AT_LAST_CTA_LAUNCH)==0);
  assert(GET(&j,WAIT_ON_LATCH_ID)==2 && GET(&j,WAIT_ON_LATCH_VALID)==1);
  assert(GET(&b,LATCH_ACQUIRE_INVALIDATE_SHADER_DATA_CACHE)==1);
  assert(GET(&j,LATCH_ACQUIRE_INVALIDATE_TEXTURE_DATA_CACHE)==1);
  assert(!a.overlaps && !s.overlaps && !b.overlaps && !c.overlaps && !j.overlaps);
  a.v3=1; assert(tinynv_qmd_r44_edge(&a,&b,1,1,1)!=0);
  puts("R44 five-node QMD encoding and generation rejection passed (CPU only)");
}
