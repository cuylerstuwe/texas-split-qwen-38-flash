// Private cross-process control page. No MLX calls occur here.
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

enum { SEQ, HOLD, STATE, ACK, BYTES, PAUSES, TOKENS, PROMPT, PHASE,
       PARK_ERROR, RESTORE_WAIT, FREED, AVAILABLE, FIELDS };
typedef struct { uint64_t magic; _Atomic uint64_t v[64]; } Page;
#define MAGIC UINT64_C(0x5458505355534831)

void* texas_ipc_open(const char* path, int create) {
  int fd=open(path,O_RDWR|O_NOFOLLOW|(create?(O_CREAT|O_EXCL):0),0600);
  if(fd<0) return NULL;
  struct stat st;
  if(fstat(fd,&st) || st.st_uid!=getuid() || (st.st_mode&0777)!=0600 ||
     (create?ftruncate(fd,4096):(st.st_size!=4096))) {close(fd);return NULL;}
  Page* p=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
  if(p==MAP_FAILED) return NULL;
  if(create) p->magic=MAGIC;
  if(p->magic!=MAGIC) {munmap(p,4096);return NULL;}
  return p;
}
void texas_ipc_close(void* p) {if(p) munmap(p,4096);}
uint64_t texas_ipc_get(void* p,unsigned field) {
  return p && field<FIELDS?atomic_load_explicit(&((Page*)p)->v[field],memory_order_acquire):0;
}
void texas_ipc_set(void* p,unsigned field,uint64_t value) {
  if(p && field<FIELDS) atomic_store_explicit(&((Page*)p)->v[field],value,memory_order_release);
}
uint64_t texas_available(void) {
  // Match Texas's reclaimable-supply probe; do not count speculative pages twice.
  static mach_port_t host=0; if(!host) host=mach_host_self();
  vm_statistics64_data_t s;mach_msg_type_number_t n=HOST_VM_INFO64_COUNT;
  if(host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&s,&n)!=KERN_SUCCESS) return 0;
  uint64_t total=0;size_t z=sizeof(total);if(sysctlbyname("hw.memsize",&total,&z,NULL,0)) return 0;
  uint64_t ext=s.external_page_count>s.speculative_count?s.external_page_count-s.speculative_count:0;
  uint64_t avail=((uint64_t)s.free_count+s.purgeable_count+ext)*vm_kernel_page_size;
  return avail<total?avail:total;
}
