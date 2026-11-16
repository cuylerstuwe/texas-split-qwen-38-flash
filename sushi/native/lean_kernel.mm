// R74: a lean application path for Sushi's custom Metal kernels (SUSHI_R74_LEAN_KERNELS=1).
//
// Every custom kernel in the engine is applied through MLX's `fast::metal_kernel`. Per application MLX (and mlx-c in
// front of it) copies the kernel's config, rebuilds the kernel's whole source text (header, generated signature,
// body, template instantiation: kilobytes) and its name, and stores both in a new primitive; per evaluation it
// hashes that source again and formats a library name to find the compiled pipeline through two string-keyed maps.
// A verify forward applies ~25 kernels per layer, so this is host time between GPU phases and, with several
// requests, a large part of what the one inference thread has to do per round.
//
// Here the first application of a (kernel, template arguments, argument types) signature goes through MLX as
// before, and what MLX derived for it is kept as a plan. Later applications of the same signature create a
// primitive that points at the plan: nothing is rebuilt, and evaluation binds the buffers and dispatches with the
// pipeline MLX compiled (looked up once). The same pipeline, the same buffers, grid and thread group: the same
// results, bit for bit.
//
// MLX's Metal backend headers need metal-cpp, which is not part of this tree; the few entry points used here are
// declared below as the shipped libmlx exports them. Metal itself is called directly on the encoder MLX hands out.
#import <Metal/Metal.h>
#include <mlx/c/mlx.h>
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/backend/common/metal_kernel.h>
#include <mlx/fast_primitives.h>
#include <mlx/primitives.h>
#include <mlx/c/private/array.h>
#include <mlx/c/private/stream.h>
#include <mlx/c/private/vector.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace MTL {
class Buffer;
class Library;
class ComputePipelineState;
class ComputeCommandEncoder;
class Function;
enum DataType : unsigned long {};
struct Size {
  unsigned long width, height, depth;
};
}  // namespace MTL

namespace mlx::core::metal {
// As exported by libmlx 0.32.3 (mlx/backend/metal/device.h); only what is called here.
class Device {
 public:
  MTL::Library* get_library(const std::string& name, const CompileOptions& compile_options, const std::function<std::string(void)>& builder);
  MTL::ComputePipelineState* get_kernel(const std::string& base_name, MTL::Library* mtl_lib, const std::string& hash_name,
                                        const std::vector<std::tuple<const void*, MTL::DataType, unsigned long>>& func_consts,
                                        const std::vector<MTL::Function*>& linked_functions);
};
class CommandEncoder {
 public:
  void set_input_array(const array& a, int idx, int64_t offset);
  void set_output_array(array& a, int idx, int64_t offset);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  MTL::ComputeCommandEncoder* get_command_encoder();
};
Device& device(mlx::core::Device);
CommandEncoder& get_command_encoder(Stream s);
}  // namespace mlx::core::metal

namespace {
namespace mx = mlx::core;

// mlx-c's kernel config (mlx/c/fast.cpp, private there): the same members in the same order.
struct ConfigCpp {
  std::vector<mx::Shape> output_shapes;
  std::vector<mx::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  std::optional<float> init_value;
  bool verbose;
};

struct Plan {
  std::vector<uint64_t> key;
  bool usable = false;            // false: this signature always takes MLX's path
  std::string name;               // MLX's kernel name for the signature
  std::string source;             // the full source MLX generated (kept for the library's build on a cache miss)
  std::string lib_name;
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  bool ensure_row_contiguous = true;
  int compile_options = 0;
  std::atomic<void*> pso{nullptr};   // set once MLX has compiled the signature's library
  unsigned long max_threads = 0;
  int tries = 0;
};

int g_mode = -1;
std::atomic<int> g_on{1};   // mode 2: switched by the engine per trace window (sushi_lean_kernel_set)
std::mutex g_mu;
std::unordered_multimap<uint64_t, Plan*> g_plans;
std::atomic<uint64_t> g_lean{0}, g_first{0}, g_unusable{0}, g_wrapped{0}, g_pending{0};

int mode() {
  if (g_mode < 0) {
    const char* e = getenv("SUSHI_R74_LEAN_KERNELS");
    g_mode = (e && e[0] >= '1' && e[0] <= '9') ? e[0] - '0' : 0;
  }
  return g_mode;
}

inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h * 0xFF51AFD7ED558CCDull;
}
uint64_t str_hash(const std::string& s) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001B3ull;
  return h;
}

// What decides the source MLX generates for an application of a kernel object: the template arguments (names,
// kinds, values), each input's type and how it is passed (a scalar by reference, fewer than 8 elements in the
// constant address space), and the output types. The stream is part of the primitive, not of the source.
void make_key(std::vector<uint64_t>& k, void* cls, const ConfigCpp& cfg, const std::vector<mx::array>& in) {
  k.clear();
  k.push_back(reinterpret_cast<uint64_t>(cls));
  k.push_back(cfg.template_args.size());
  for (const auto& [name, arg] : cfg.template_args) {
    uint64_t v = 0;
    if (std::holds_alternative<int>(arg)) v = (uint64_t(1) << 60) | uint32_t(std::get<int>(arg));
    else if (std::holds_alternative<bool>(arg)) v = (uint64_t(2) << 60) | uint64_t(std::get<bool>(arg));
    else if (std::holds_alternative<mx::Dtype>(arg)) v = (uint64_t(3) << 60) | uint64_t(static_cast<int>(std::get<mx::Dtype>(arg).val()));
    k.push_back(str_hash(name));
    k.push_back(v);
  }
  k.push_back(in.size());
  for (const auto& a : in)
    k.push_back(uint64_t(static_cast<int>(a.dtype().val())) | (uint64_t(a.ndim() == 0) << 8) | (uint64_t(a.size() < 8) << 9));
  k.push_back(cfg.output_dtypes.size());
  for (const auto& d : cfg.output_dtypes) k.push_back(uint64_t(static_cast<int>(d.val())));
}

Plan* find_plan(const std::vector<uint64_t>& key, uint64_t h) {
  auto range = g_plans.equal_range(h);
  for (auto it = range.first; it != range.second; ++it)
    if (it->second->key == key) return it->second;
  return nullptr;
}

// An input a kernel declared with `ensure_row_contiguous` reads as row-contiguous memory. MLX checks the flag at
// evaluation and copies a strided input first; that copy is not reachable from here, so the lean path only takes an
// application whose inputs are known to be row contiguous when the graph is built: an array that already has its
// buffer says so itself; a lazy one is when its producer writes a fresh buffer, only reshapes one that is, or
// slices one with unit steps so that only the leading axis shrinks (MLX's own rule, `check_contiguity`, on the
// strides the slice will have). Any other application goes through MLX as before.
bool known_contiguous(const mx::array& a, int depth) {
  if (!a.has_primitive()) return a.flags().row_contiguous;  // evaluated or already encoded: its flags are set
  const char* n = a.primitive().name();
  static const char* const fresh[] = {"LeanKernel", "CustomKernel", "Contiguous", "Concatenate", "Matmul", "QuantizedMatmul", "Add",
                                      "Subtract", "Multiply", "Divide", "Maximum", "Minimum", "Sigmoid", "RMSNorm", "Softmax", "Gather",
                                      "GatherMM", "GatherQMM", "ArgSort", "Sort", "Sum", "ScaledDotProductAttention", "Full", "Arange"};
  for (const char* f : fresh)
    if (std::strcmp(n, f) == 0) return true;
  if (depth >= 6 || a.inputs().empty()) return false;
  const mx::array& in = a.inputs()[0];
  static const char* const views[] = {"Reshape", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "StopGradient"};
  for (const char* v : views)
    if (std::strcmp(n, v) == 0) return known_contiguous(in, depth + 1);
  if (std::strcmp(n, "AsType") == 0) return a.dtype() != in.dtype() || known_contiguous(in, depth + 1);
  if (std::strcmp(n, "Slice") == 0) {
    if (!known_contiguous(in, depth + 1)) return false;
    const auto st = static_cast<const mx::Slice&>(a.primitive()).state();
    const mx::Shape& steps = std::get<2>(st);
    const mx::Shape& is = in.shape();
    const mx::Shape& os = a.shape();
    if (steps.size() != is.size() || os.size() != is.size()) return false;
    int64_t in_stride = 1, want = 1;
    for (int ri = static_cast<int>(os.size()) - 1; ri >= 0; ri--) {
      if (steps[ri] < 1) return false;
      if (os[ri] != 1 && in_stride * steps[ri] != want) return false;
      in_stride *= is[ri];
      want *= os[ri];
    }
    return true;
  }
  return false;
}

class LeanKernel : public mx::Primitive {
 public:
  LeanKernel(mx::Stream s, Plan* plan, std::tuple<int, int, int> grid, std::tuple<int, int, int> tg)
      : mx::Primitive(s), plan_(plan), grid_(grid), tg_(tg) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("LeanKernel only runs on the GPU."); }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "LeanKernel"; }

 private:
  Plan* plan_;
  std::tuple<int, int, int> grid_, tg_;
};

// The pipeline MLX compiled for the signature, once it exists: MLX builds the library when an application made
// through its own path is first evaluated, and it cannot be built from here (MLX's kernel preamble is not exported).
// Until then the signature keeps taking MLX's path.
struct NotCompiled {};
bool resolve(Plan& p, const mx::Stream& s) {
  try {
    auto& d = mx::metal::device(s.device);
    mx::CompileOptions opts(p.compile_options);
    MTL::Library* lib = d.get_library(p.lib_name, opts, []() -> std::string { throw NotCompiled(); });
    MTL::ComputePipelineState* pso = d.get_kernel(p.name, lib, std::string(), {}, {});
    p.max_threads = [(__bridge id<MTLComputePipelineState>)(void*)pso maxTotalThreadsPerThreadgroup];
    p.pso.store(pso, std::memory_order_release);
    return true;
  } catch (const NotCompiled&) {
    if (++p.tries > 4096) p.usable = false;  // never evaluated through MLX: leave it there
    return false;
  }
}

void LeanKernel::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  Plan& p = *plan_;
  for (auto& out : outputs) out.set_data(mx::allocator::malloc(out.nbytes()));
  const mx::Stream& s = stream();
  if (p.ensure_row_contiguous)
    for (const auto& in : inputs)
      if (!in.flags().row_contiguous) throw std::runtime_error("[lean-kernel] " + p.name + ": an input is not row contiguous");
  const auto [tx, ty, tz] = tg_;
  if (static_cast<unsigned long>(tx) * ty * tz > p.max_threads) throw std::invalid_argument("[lean-kernel] " + p.name + ": thread group larger than the pipeline allows");
  auto& enc = mx::metal::get_command_encoder(s);
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)(void*)enc.get_command_encoder();
  [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)p.pso.load(std::memory_order_relaxed)];
  int index = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    const mx::array& in = inputs[i];
    enc.set_input_array(in, index++, 0);
    if (in.ndim() > 0) {
      const auto& si = p.shape_infos[i];
      if (std::get<0>(si)) [e setBytes:in.shape().data() length:in.ndim() * sizeof(int) atIndex:index++];
      if (std::get<1>(si)) [e setBytes:in.strides().data() length:in.ndim() * sizeof(int64_t) atIndex:index++];
      if (std::get<2>(si)) {
        const int nd = static_cast<int>(in.ndim());
        [e setBytes:&nd length:sizeof(int) atIndex:index++];
      }
    }
  }
  for (auto& out : outputs) enc.set_output_array(out, index++, 0);
  const auto [gx, gy, gz] = grid_;
  const MTL::Size group{static_cast<unsigned long>(std::min(tx, gx)), static_cast<unsigned long>(std::min(ty, gy)), static_cast<unsigned long>(std::min(tz, gz))};
  const MTL::Size grid{static_cast<unsigned long>(gx), static_cast<unsigned long>(gy), static_cast<unsigned long>(gz)};
  enc.dispatch_threads(grid, group);
}

// The signature's first application went through MLX: keep what it derived.
Plan* learn(std::vector<uint64_t>&& key, uint64_t h, const std::vector<mx::array>& outs, const ConfigCpp& cfg, size_t n_inputs) {
  Plan* p = new Plan();
  p->key = std::move(key);
  if (!outs.empty() && outs[0].has_primitive() && std::strcmp(outs[0].primitive().name(), "CustomKernel") == 0 && !cfg.init_value) {
    const auto st = static_cast<const mx::fast::CustomKernel&>(outs[0].primitive()).state();
    p->name = std::get<0>(st);
    p->source = std::get<1>(st);
    p->shape_infos = std::get<4>(st);
    p->ensure_row_contiguous = std::get<5>(st);
    p->compile_options = std::get<10>(st);
    char buf[64];
    snprintf(buf, sizeof buf, "_%zx_%d", std::hash<std::string>{}(p->source), p->compile_options);
    p->lib_name = p->name + buf;
    p->usable = !std::get<6>(st).has_value() && !std::get<8>(st) && p->shape_infos.size() == n_inputs;
  }
  g_plans.emplace(h, p);
  return p;
}
}  // namespace

extern "C" {
// The engine's kernel applications come through here (src/mlx.zig).
int sushi_lean_kernel_apply(mlx_vector_array* outputs, mlx_fast_metal_kernel cls, const mlx_vector_array inputs, const mlx_fast_metal_kernel_config config,
                            const mlx_stream s) {
  if (mode() == 0 || !g_on.load(std::memory_order_relaxed) || !cls.ctx || !config.ctx || !inputs.ctx) return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
  try {
    const ConfigCpp& cfg = *static_cast<const ConfigCpp*>(config.ctx);
    const std::vector<mx::array>& in = mlx_vector_array_get_(inputs);
    if (cfg.init_value || cfg.verbose) return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
    thread_local std::vector<uint64_t> key;
    make_key(key, cls.ctx, cfg, in);
    uint64_t h = 0;
    for (uint64_t w : key) h = mix(h, w);
    Plan* plan;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      plan = find_plan(key, h);
    }
    if (!plan) {
      const int rc = mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
      if (rc) return rc;
      g_first++;
      std::lock_guard<std::mutex> lock(g_mu);
      if (!find_plan(key, h)) learn(std::vector<uint64_t>(key), h, mlx_vector_array_get_(*outputs), cfg, in.size());
      return 0;
    }
    if (!plan->usable) {
      g_unusable++;
      return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
    }
    const mx::Stream& stream = mlx_stream_get_(s);
    if (stream.device != mx::Device::gpu) return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
    if (plan->pso.load(std::memory_order_acquire) == nullptr) {
      std::lock_guard<std::mutex> lock(g_mu);
      if (plan->pso.load(std::memory_order_acquire) == nullptr && !resolve(*plan, stream)) {
        g_pending++;
        return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
      }
    }
    if (plan->ensure_row_contiguous)
      for (const auto& a : in)
        if (!known_contiguous(a, 0)) {
          g_wrapped++;
          return mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s);
        }
    std::vector<mx::array> outs =
        mx::array::make_arrays(std::vector<mx::Shape>(cfg.output_shapes), cfg.output_dtypes, std::make_shared<LeanKernel>(stream, plan, cfg.grid, cfg.thread_group), in);
    mlx_vector_array_set_(*outputs, outs);
    g_lean++;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

// lean applications, first applications (through MLX), applications of unusable signatures, applications left to
// MLX because an input was not known to be row contiguous, applications that waited for MLX's compile.
void sushi_lean_kernel_stats(uint64_t* v) {
  v[0] = g_lean.load();
  v[1] = g_first.load();
  v[2] = g_unusable.load();
  v[3] = g_wrapped.load();
  v[4] = g_pending.load();
}
int sushi_lean_kernel_mode(void) { return mode(); }
void sushi_lean_kernel_set(int on) { g_on.store(on ? 1 : 0, std::memory_order_relaxed); }
}
