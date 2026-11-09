#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <mlx/random.h>
#include <mlx/c/random.h>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <signal.h>
#include <fcntl.h>
#include <vector>
extern "C" void* texas_park(void**,size_t,void**,size_t,const char*,size_t);
extern "C" size_t texas_park_bytes(void*);
extern "C" int texas_restore(void*);
extern "C" void texas_park_free(void*);
extern "C" const char* texas_park_error();
extern "C" void* texas_rng_save();
extern "C" void texas_rng_restore(void*);
namespace mx=mlx::core;
int main() {
 mx::random::seed(789);
 auto expected=mx::random::uniform({16});mx::eval(expected);
 mx::random::seed(789);auto saved=texas_rng_save();
 auto interference=mx::random::uniform({16});mx::eval(interference);
 texas_rng_restore(saved);
 auto actual=mx::random::uniform({16});mx::eval(actual);
 assert(memcmp(expected.data<float>(),actual.data<float>(),64)==0);
 auto stream=mlx_default_gpu_stream_new();
 auto draw=[&](){
   auto v=mlx_array_new();int shape[]={16};
   assert(mlx_random_bits(&v,shape,1,4,mlx_array{nullptr},stream)==0 && mlx_array_eval(v)==0);
   std::vector<uint32_t> out(mlx_array_data_uint32(v),mlx_array_data_uint32(v)+16);mlx_array_free(v);return out;
 };
 assert(mlx_random_seed(1234)==0);auto expected_c=draw();
 assert(mlx_random_seed(1234)==0);saved=texas_rng_save();draw();texas_rng_restore(saved);
 assert(draw()==expected_c);mlx_stream_free(stream);
 // Async decode roots may still be lazy. Park their materialized request
 // ancestors while protecting even a model-private lazy dependency.
 {
  auto owned=mx::ones({1048576});auto weight=mx::full({1048576},2.0f);mx::eval(owned,weight);
  auto lazy=mx::add(owned,weight);auto private_root=mx::add(weight,mx::array(1.0f));
  void* roots[]={&lazy};void* protected_roots[]={&private_root};
  const char* path="/tmp/texas-sushi-lazy-fixture.bin";unlink(path);
  auto parked=texas_park(roots,1,protected_roots,1,path,24ull<<30);
  assert(parked && texas_park_bytes(parked)==4194304 && !owned.buffer().ptr() && weight.buffer().ptr());
  assert(texas_restore(parked)==0);texas_park_free(parked);mx::eval(lazy);
  for(size_t i=0;i<lazy.size();i++) assert(lazy.data<float>()[i]==3.0f);
 }
 const char* p="/tmp/texas-sushi-checkpoint-fixture.bin"; unlink(p);
 auto a=mx::arange(0,1048576,1,mx::float32); mx::eval(a);
 auto v=mx::slice(a,{7},{1048570},{3}); mx::eval(v);
 auto w=mx::ones({1048576}); mx::eval(w);
 auto wv=mx::reshape(w,{1024,1024}); mx::eval(wv);
 auto before=std::vector<float>(a.data<float>(),a.data<float>()+a.size());
 auto aid=a.id(); auto data=a.data_shared_ptr().get(); auto vstrides=v.strides();
 void* roots[]={&a,&v,&wv};void* protect[]={&w};
 auto ck=texas_park(roots,3,protect,1,p,24ull<<30);
 assert(ck && texas_park_bytes(ck)==4194304);
 assert(a.data_shared_ptr().get()==data && !a.buffer().ptr() && !v.buffer().ptr());
 assert(w.buffer().ptr() && wv.buffer().ptr());
 struct stat st;assert(stat(p,&st)==0 && (st.st_mode&0777)==0600);
 assert(texas_restore(ck)==0 && a.id()==aid && a.data_shared_ptr().get()==data);
 assert(v.strides()==vstrides && a.buffer().ptr()==v.buffer().ptr());
 assert(memcmp(a.data<float>(),before.data(),4194304)==0);
 for(size_t i=0;i<v.size();i++) assert(v.data<float>()[i*3]==before[7+i*3]);
 texas_park_free(ck);assert(access(p,F_OK)!=0);
 // A failed write leaves every original allocation readable.
 ck=texas_park(roots,2,nullptr,0,"/nonexistent/texas/fixture",24ull<<30);
 assert(!ck && a.buffer().ptr() && memcmp(a.data<float>(),before.data(),4194304)==0);
 ck=texas_park(roots,2,nullptr,0,p,1);
 assert(!ck && a.buffer().ptr());
 struct rlimit limit,small;assert(getrlimit(RLIMIT_FSIZE,&limit)==0);small=limit;small.rlim_cur=3;
 auto old_signal=signal(SIGXFSZ,SIG_IGN);assert(setrlimit(RLIMIT_FSIZE,&small)==0);
 ck=texas_park(roots,2,nullptr,0,p,24ull<<30);
 assert(!ck && a.buffer().ptr() && memcmp(a.data<float>(),before.data(),4194304)==0);
 assert(setrlimit(RLIMIT_FSIZE,&limit)==0);signal(SIGXFSZ,old_signal);
 ck=texas_park(roots,2,nullptr,0,p,24ull<<30);assert(ck);
 assert(truncate(p,3)==0 && texas_restore(ck)!=0);
 assert(!a.buffer().ptr());texas_park_free(ck);
 auto damaged=mx::ones({1048576});mx::eval(damaged);void* dr[]={&damaged};
 ck=texas_park(dr,1,nullptr,0,p,24ull<<30);assert(ck);
 int fd=open(p,O_RDWR);assert(fd>=0);char bad=91;assert(pwrite(fd,&bad,1,100)==1);close(fd);
 assert(texas_restore(ck)!=0 && !damaged.buffer().ptr());texas_park_free(ck);
 return 0;
}
