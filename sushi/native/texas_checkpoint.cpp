// Texas lossless parking; MLX 0.32.3 C++ ABI, Sushi owner thread only.
// The caller must drain all streams and exclude every shared/model buffer.
#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/random.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <sys/statvfs.h>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <sys/mman.h>
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
namespace mx=mlx::core;
using Data=mx::array::Data;
// Follow lazy dependencies without evaluating them. Compiled verifier capture
// objects can retain symbolic/retired leaves which are not evaluable requests.
// Their materialized ancestors still need ordinary alias/weight protection.
static std::vector<mx::array> materialized_arrays(const std::vector<mx::array>& roots) {
  std::vector<mx::array> todo=roots, result;
  std::unordered_set<uintptr_t> seen;
  while(!todo.empty()) {
    auto a=std::move(todo.back());todo.pop_back();
    if(!seen.insert(a.id()).second) continue;
    if(seen.size()>1000000) throw std::runtime_error("checkpoint array graph exceeds bounded traversal");
    auto d=a.data_shared_ptr();
    if(a.is_available() && !a.has_primitive() && d && d->buffer.ptr()) {
      result.push_back(std::move(a));
    } else {
      for(const auto& input:a.inputs()) todo.push_back(input);
    }
  }
  return result;
}
static void io_all(int fd, void* ptr, size_t n, off_t off, bool write, unsigned char* digest) {
  CC_SHA256_CTX hash;CC_SHA256_Init(&hash);
  auto p=static_cast<char*>(ptr);
  while(n) {
    auto got=write ? pwrite(fd,p,std::min(n,size_t(16<<20)),off) : pread(fd,p,std::min(n,size_t(16<<20)),off);
    if(got<0 && errno==EINTR) continue;
    if(got<=0) throw std::runtime_error(std::string("checkpoint I/O: ")+strerror(errno));
    CC_SHA256_Update(&hash,p,static_cast<CC_LONG>(got));
    p+=got; off+=got; n-=got;
  }
  CC_SHA256_Final(digest,&hash);
}
struct Group {
  std::vector<std::shared_ptr<Data>> refs;
  size_t bytes=0; off_t offset=0;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
};
struct Parked {
  std::vector<Group> groups;
  int fd=-1;
  size_t bytes=0;
  bool parked=false;
  std::string path;
  Parked(const std::vector<mx::array>& arrays, const std::vector<mx::array>& protect, const std::string& p, size_t max_bytes):path(p) {
    std::unordered_set<const void*> protected_buffers;
    for(const auto& a:protect) if(a.data_shared_ptr()) protected_buffers.insert(a.buffer().ptr());
    std::unordered_set<const void*> seen_data;
    std::unordered_map<const void*,size_t> by_buffer;
    for(const auto& a:arrays) {
      if(!a.is_available() || a.has_primitive()) throw std::runtime_error("checkpoint requires materialized leaf tensors");
      auto d=a.data_shared_ptr();
      if(!d || !d->buffer.ptr() || protected_buffers.count(d->buffer.ptr()) || !seen_data.insert(d.get()).second) continue;
      size_t n=mx::allocator::allocator().size(d->buffer);
      if(n < (1<<20)) continue; // Small logical state stays resident.
      auto it=by_buffer.find(d->buffer.ptr());
      if(it==by_buffer.end()) {
        if(n>max_bytes-bytes) throw std::runtime_error("checkpoint exceeds bounded disk budget");
        by_buffer[d->buffer.ptr()]=groups.size();
        groups.push_back(Group{{d},n,static_cast<off_t>(bytes)});bytes+=n;
      } else groups[it->second].refs.push_back(d);
    }
    struct statvfs space;
    auto parent=path.substr(0,path.find_last_of('/'));
    if(statvfs(parent.c_str(),&space) || (uint64_t)space.f_bavail*space.f_frsize < bytes+(32ull<<30)) throw std::runtime_error("checkpoint SSD reserve unavailable");
    fd=open(path.c_str(),O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0) throw std::runtime_error("cannot create private checkpoint");
    try {
      // F_NOCACHE keeps the checkpoint from becoming another large RAM cache.
      if(fcntl(fd,F_NOCACHE,1)<0) throw std::runtime_error("cannot disable checkpoint file cache");
      for(auto& g:groups) io_all(fd,g.refs[0]->buffer.raw_ptr(),g.bytes,g.offset,true,g.digest);
      if(fsync(fd)<0) throw std::runtime_error("checkpoint sync failed");
    } catch(...) { close(fd);fd=-1;unlink(path.c_str());throw; }
    // No mutation occurs until every byte is safely on disk.
    try {
      for(auto& g:groups) for(auto& d:g.refs) {
        auto b=d->buffer; auto deleter=std::move(d->d);
        d->buffer=mx::allocator::Buffer(nullptr);
        d->d=[](mx::allocator::Buffer){};
        deleter(b);
      }
    } catch(...) { std::terminate(); } // Never expose a partially parked graph.
    parked=true;
  }
  void restore() {
    if(!parked) return;
    // Restore one allocation at a time. A failure leaves the unread groups
    // parked, with the checkpoint open, so the owner can retry safely.
    for(auto& g:groups) {
      if(g.refs[0]->buffer.ptr()) continue;
      auto b=mx::allocator::malloc(g.bytes);
      if(!b.ptr()) throw std::bad_alloc();
      try {
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        io_all(fd,b.raw_ptr(),g.bytes,g.offset,false,digest);
        if(memcmp(digest,g.digest,sizeof(digest))) throw std::runtime_error("checkpoint checksum mismatch");
      }
      catch(...) { mx::allocator::free(b);throw; }
      std::shared_ptr<Data> owner;
      try { owner=std::make_shared<Data>(b); }
      catch(...) { mx::allocator::free(b);throw; }
      std::vector<mx::Deleter> deleters;
      deleters.reserve(g.refs.size());
      for(size_t i=0;i<g.refs.size();++i) deleters.emplace_back([owner](mx::allocator::Buffer){});
      // Allocate all bookkeeping before publication. Moves are noexcept.
      for(size_t i=0;i<g.refs.size();++i) {
        g.refs[i]->buffer=b;
        g.refs[i]->d=std::move(deleters[i]);
      }
    }
    parked=false;
    close(fd);fd=-1;unlink(path.c_str());
  }
  ~Parked() { if(fd>=0) close(fd); if(!path.empty()) unlink(path.c_str()); }
};

static thread_local std::string last_error;
static std::vector<mx::array> roots_from(void** p,size_t n) {
 std::vector<mx::array> out;out.reserve(n);
 for(size_t i=0;i<n;i++) if(p[i]) out.push_back(*static_cast<mx::array*>(p[i]));
 return materialized_arrays(out);
}
extern "C" void* texas_park(void** roots,size_t n,void** protect,size_t pn,const char* path,size_t cap) {
 // Both sets traverse materialized ancestors without evaluating symbolic leaves.
 // Protected roots include the retained weight map and private model buffers.
 try { last_error.clear();return new Parked(roots_from(roots,n),roots_from(protect,pn),path,cap); }
 catch(const std::exception& e){last_error=e.what();return nullptr;}
}
extern "C" size_t texas_park_bytes(void* p){return p?static_cast<Parked*>(p)->bytes:0;}
extern "C" int texas_restore(void* p){
 try {if(p) static_cast<Parked*>(p)->restore();return 0;}
 catch(const std::exception& e){last_error=e.what();return -1;}
}
extern "C" void texas_park_free(void* p){delete static_cast<Parked*>(p);}
extern "C" const char* texas_park_error(){return last_error.c_str();}
// MLX 0.32.3's inline default_() has hidden TLS per dylib. Calling the inline
// here would snapshot a different RNG. Resolve the pinned library's two TLV
// descriptors from its Mach-O symbols; never infer an object layout or offset.
struct Tlv { void* (*address)(Tlv*); uintptr_t key,offset; };
struct RngSymbols { Tlv *key=nullptr,*guard=nullptr; };
static RngSymbols rng_symbols() {
 Dl_info info{};
 if(!dladdr(reinterpret_cast<const void*>(&mx::random::seed),&info)) throw std::runtime_error("MLX RNG image missing");
 int fd=open(info.dli_fname,O_RDONLY|O_NOFOLLOW);
 if(fd<0) throw std::runtime_error("MLX RNG image unreadable");
 struct stat st{};if(fstat(fd,&st)){close(fd);throw std::runtime_error("MLX RNG image stat failed");}
 auto raw=mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
 if(raw==MAP_FAILED) throw std::runtime_error("MLX RNG symbol map failed");
 auto base=static_cast<const char*>(raw);auto mh=reinterpret_cast<const mach_header_64*>(base);
 RngSymbols result; const symtab_command* sym=nullptr;uint64_t text=0;
 if(mh->magic==MH_MAGIC_64 && mh->sizeofcmds+sizeof(*mh)<=static_cast<size_t>(st.st_size)) {
  auto cmd=reinterpret_cast<const load_command*>(mh+1);
  for(uint32_t i=0;i<mh->ncmds;i++) {
   if(cmd->cmd==LC_SYMTAB) sym=reinterpret_cast<const symtab_command*>(cmd);
   if(cmd->cmd==LC_SEGMENT_64) {auto seg=reinterpret_cast<const segment_command_64*>(cmd);if(!strcmp(seg->segname,"__TEXT")) text=seg->vmaddr;}
   cmd=reinterpret_cast<const load_command*>(reinterpret_cast<const char*>(cmd)+cmd->cmdsize);
  }
  if(sym && uint64_t(sym->symoff)+uint64_t(sym->nsyms)*sizeof(nlist_64)<=uint64_t(st.st_size) && uint64_t(sym->stroff)+sym->strsize<=uint64_t(st.st_size)) {
   const auto entries=reinterpret_cast<const nlist_64*>(base+sym->symoff);
   const auto strings=base+sym->stroff;auto slide=reinterpret_cast<uintptr_t>(info.dli_fbase)-text;
   for(uint32_t i=0;i<sym->nsyms;i++) {
    auto idx=entries[i].n_un.n_strx;if(idx>=sym->strsize) continue;
    const auto name=strings+idx;
    if(!strcmp(name,"__ZZN3mlx4core6random11KeySequence8default_EvE2ks")) result.key=reinterpret_cast<Tlv*>(slide+entries[i].n_value);
    if(!strcmp(name,"__ZGVZN3mlx4core6random11KeySequence8default_EvE2ks")) result.guard=reinterpret_cast<Tlv*>(slide+entries[i].n_value);
   }
  }
 }
 munmap(raw,st.st_size);
 if(!result.key || !result.guard) throw std::runtime_error("Pinned MLX RNG TLS ABI unavailable");
 return result;
}
struct SavedRng { mx::random::KeySequence* target; mx::random::KeySequence saved; };
extern "C" void* texas_rng_save(){
 try {
  static auto symbols=rng_symbols();
  if(!(*static_cast<unsigned char*>(symbols.guard->address(symbols.guard))&1)) return nullptr;
  auto target=static_cast<mx::random::KeySequence*>(symbols.key->address(symbols.key));
  return new SavedRng{target,*target};
 } catch(const std::exception& e) {fprintf(stderr,"Texas RNG guard: %s\n",e.what());_exit(78);}
}
extern "C" void texas_rng_restore(void* p){
 if(!p) return;
 auto saved=static_cast<SavedRng*>(p);*saved->target=saved->saved;delete saved;
}
