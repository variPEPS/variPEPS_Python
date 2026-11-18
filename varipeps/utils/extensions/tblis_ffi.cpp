// Created with Claude Opus 5.5 (AI-generated).
//
// XLA FFI handler for a pairwise tensor contraction D = A * B in einsum
// notation. The handler only uses the TAPP interface, the row-major buffers of
// XLA are passed as strided tensors without any transposition.
//
// Threading: tblis runs single-threaded (see tapp_tblis.cpp). A large product
// is split along one label into independent parts, which are scheduled on the
// intra-op thread pool of XLA, and the handler returns a future that completes
// with the last part. A label of D gives disjoint parts of D. A contracted
// label (for a small D with a large contraction, e.g. the gradients of the
// PEPS tensors) gives partial results in private buffers, which the last part
// adds up. There is no second threading runtime, no barrier between the parts
// and no thread that spins while it waits.

#include <tapp.h>

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <thread>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace nb = nanobind;
namespace ffi = xla::ffi;

namespace {

// Shortest runtime of a part: a few times the wake-up latency of a pool thread
// (futex wake-up and context switch, 5-50 us on Linux). Not measured, as a
// blocking wait for a pool task could deadlock in the handler.
constexpr double kMinPartSeconds = 20e-6;
// Penalty for splitting a label which is not the outermost axis of every
// tensor: the sliced axis can no longer be folded with its outer neighbour.
constexpr double kFoldPenalty = 1.25;
// Parts of D must not be closer than this (false sharing of cache lines).
constexpr int64_t kMinPartBytes = 128;
// Largest memory for the partial results of a split contracted label.
constexpr double kMaxPartialBytes = 1 << 26;
// Outputs of output splits from this size on are mapped by the parts first, if the parts
// interleave within huge pages (PopulatePart).
constexpr size_t kPrefaultBytes = 4 << 20;

// Releases a buffer from std::aligned_alloc.
struct FreeDeleter {
  void operator()(void* p) const { std::free(p); }
};

template <ffi::DataType dtype>
constexpr TAPP_datatype kTappType = dtype == ffi::DataType::F32   ? TAPP_F32
                                    : dtype == ffi::DataType::F64 ? TAPP_F64
                                    : dtype == ffi::DataType::C64 ? TAPP_C32
                                                                  : TAPP_C64;

// Owns a TAPP object and calls its destroy function on scope exit.
template <TAPP_error (*Destroy)(intptr_t)>
struct Owned {
  intptr_t value = 0;
  ~Owned() {
    if (value) {
      Destroy(value);
    }
  }
};

ffi::Error TappError(TAPP_error err) {
  if (!err) {
    return ffi::Error::Success();
  }
  char message[256];
  TAPP_explain_error(err, sizeof(message), message);
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, std::string("TAPP: ") + message);
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size());
  int64_t stride = 1;
  for (size_t i = dims.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

// Copy of the shapes and labels, the parts may run after the handler returned.
struct Tensor {
  std::vector<int64_t> dims, strides, idx;
  Tensor(ffi::Span<const int64_t> d, ffi::Span<const int64_t> i)
      : dims(d.begin(), d.end()), strides(RowMajorStrides(dims)), idx(i.begin(), i.end()) {}
  Tensor(std::vector<int64_t> d, std::vector<int64_t> i)
      : dims(std::move(d)), strides(RowMajorStrides(dims)), idx(std::move(i)) {}
};

struct Problem {
  Tensor a, b, d;
};

// Split of the axis pos_d of D (and the axes of the same label in A and B, -1
// if absent) into parts of length part_len. pos_d = -1 with more than one part
// is the split of a contracted label.
struct Split {
  int pos_a = -1, pos_b = -1, pos_d = -1;
  int64_t parts = 1, part_len = 0;
};

int Find(const std::vector<int64_t>& idx, int64_t label) {
  auto it = std::find(idx.begin(), idx.end(), label);
  return it == idx.end() ? -1 : static_cast<int>(it - idx.begin());
}

double Size(const std::vector<int64_t>& dims) {
  double size = 1;
  for (int64_t n : dims) {
    size *= n;
  }
  return size;
}

// Machine parameters of the cost model, in multiply-adds of the data type.
struct CostModel {
  double min_part_work;      // work of the shortest part (kMinPartSeconds)
  double move_cost;          // packing an operand element or adding up an element
                             // of a partial result (both memory bound)
  double reduce_cost;        // adding up an element of a partial result
  double max_partial_bytes;  // memory of the partial results of a reduction split
  double even_parts;         // > 0: output splits into part counts dividing the extent (experiment)
  double fold_penalty;       // factor for splitting an inner label (strided parts)
};

// Chooses the label to split along: the one with the smallest estimated time
// per thread, if this is below the time of the unsplit product.
// Length of the parts of a label of `extent` for at most `max_parts` parts, at
// least `min_len`. With `even` (experiment, output splits) the largest part
// count that divides the extent instead, if it is at least half as large.
int64_t PartLen(int64_t extent, int64_t max_parts, int64_t min_len, bool even) {
  int64_t parts = std::min(extent, max_parts);
  for (int64_t n = parts; even && n >= 2 && 2 * n >= parts; --n) {
    if (extent % n == 0) {
      parts = n;
      break;
    }
  }
  return std::max((extent + parts - 1) / parts, min_len);
}

template <typename T>
Split ChooseSplit(const Problem& p, int64_t num_threads, const CostModel& model) {
  std::map<int64_t, int64_t> extents;
  for (const Tensor* t : {&p.a, &p.b}) {
    for (size_t i = 0; i < t->idx.size(); ++i) {
      extents[t->idx[i]] = t->dims[i];
    }
  }
  double work = 1;
  for (const auto& [label, extent] : extents) {
    work *= extent;
  }

  const int64_t max_parts =
      std::min<double>(static_cast<double>(num_threads), work / model.min_part_work);
  Split best;
  double best_time = work;
  for (size_t i = 0; i < p.d.idx.size() && max_parts > 1; ++i) {
    // Parts of at least kMinPartBytes (no false sharing): fewer parts, if needed.
    const int64_t extent = p.d.dims[i];
    const int64_t stride_bytes = p.d.strides[i] * static_cast<int64_t>(sizeof(T));
    Split split{Find(p.a.idx, p.d.idx[i]), Find(p.b.idx, p.d.idx[i]), static_cast<int>(i)};
    split.part_len = PartLen(extent, max_parts, (kMinPartBytes + stride_bytes - 1) / stride_bytes,
                             model.even_parts > 0);
    split.parts = (extent + split.part_len - 1) / split.part_len;
    if (split.parts < 2) {
      continue;
    }

    // A batch label repacks nothing, a label of only A (B) repacks B (A).
    double time = work * split.part_len / extent;
    if (split.pos_a < 0 || split.pos_b < 0) {
      time += model.move_cost * Size(split.pos_a < 0 ? p.a.dims : p.b.dims);
    }
    if (i > 0 || split.pos_a > 0 || split.pos_b > 0) {
      time *= model.fold_penalty;
    }
    if (time < best_time) {
      best = split;
      best_time = time;
    }
  }

  // Contracted labels (VARIPEPS_TBLIS_REDUCTION_SPLIT=0 disables them, for
  // comparisons): no repacking, but partial results of D to add up. The
  // number of parts minimizes work / parts + reduce_cost * parts * |D|, with
  // at most as many parts as their partial results (all parts but one) fit
  // into max_partial_bytes.
  // Labels are unique within a tensor (checked by TAPP_create_tensor_product).
  const double size_d = Size(p.d.dims);
  const int64_t reduce_parts = std::min<double>(
      {static_cast<double>(max_parts),
       std::round(std::sqrt(work / (model.reduce_cost * std::max(size_d, 1.0)))),
       std::floor(model.max_partial_bytes / (std::max(size_d, 1.0) * sizeof(T))) + 1});
  static const bool reduction_split = [] {
    const char* str = std::getenv("VARIPEPS_TBLIS_REDUCTION_SPLIT");
    return !(str && std::string(str) == "0");
  }();
  for (size_t i = 0; i < p.a.idx.size() && reduction_split && size_d > 0 && reduce_parts > 1; ++i) {
    const int pos_b = Find(p.b.idx, p.a.idx[i]);
    if (pos_b < 0 || Find(p.d.idx, p.a.idx[i]) >= 0) {
      continue;
    }
    const int64_t extent = p.a.dims[i];
    Split split{static_cast<int>(i), pos_b, -1};
    split.part_len = PartLen(extent, reduce_parts, 1, false);
    split.parts = (extent + split.part_len - 1) / split.part_len;
    if (split.parts < 2 || (split.parts - 1) * size_d * sizeof(T) > model.max_partial_bytes) {
      continue;
    }

    double time = work * split.part_len / extent + model.reduce_cost * split.parts * size_d;
    if (i > 0 || pos_b > 0) {
      time *= model.fold_penalty;
    }
    if (time < best_time) {
      best = split;
      best_time = time;
    }
  }
  return best;
}

// Contracts part `part` of `split` (the whole product if not split).
template <ffi::DataType dtype>
TAPP_error ContractPartImpl(const Problem& p,
                            const Split& split,
                            int64_t part,
                            const void* a,
                            const void* b,
                            void* d) {
  static TAPP_handle handle = [] {
    TAPP_handle h;
    TAPP_create_handle(&h);
    return h;
  }();
  static TAPP_executor exec = [] {
    TAPP_executor e;
    TAPP_create_executor(&e);
    return e;
  }();
  using T = ffi::NativeType<dtype>;

  // Restricts the split axis of a tensor to the part.
  auto slice = [&](const Tensor& t, int pos, auto* data, auto& info) {
    std::vector<int64_t> dims = t.dims;
    if (pos >= 0) {
      const int64_t start = part * split.part_len;
      dims[pos] = std::min(split.part_len, dims[pos] - start);
      data += start * t.strides[pos];
    }
    TAPP_error err = TAPP_create_tensor_info(&info.value, kTappType<dtype>, dims.size(),
                                             dims.data(), t.strides.data());
    return std::make_pair(err, data);
  };

  Owned<TAPP_destroy_tensor_info> info_a, info_b, info_d;
  Owned<TAPP_destroy_tensor_product> plan;
  auto [err_a, part_a] = slice(p.a, split.pos_a, static_cast<const T*>(a), info_a);
  auto [err_b, part_b] = slice(p.b, split.pos_b, static_cast<const T*>(b), info_b);
  auto [err_d, part_d] = slice(p.d, split.pos_d, static_cast<T*>(d), info_d);
  if (err_a || err_b || err_d) {
    return err_a ? err_a : err_b ? err_b : err_d;
  }

  TAPP_error err = TAPP_create_tensor_product(
      &plan.value, handle, TAPP_IDENTITY, info_a.value, p.a.idx.data(), TAPP_IDENTITY, info_b.value,
      p.b.idx.data(), TAPP_IDENTITY, info_d.value, p.d.idx.data(), TAPP_IDENTITY, info_d.value,
      p.d.idx.data(), TAPP_DEFAULT_PREC);
  if (err) {
    return err;
  }

  const T alpha(1), beta(0);
  TAPP_status status;
  return TAPP_execute_product(plan.value, exec, &status, &alpha, part_a, part_b, &beta, nullptr,
                              part_d);
}

// Exceptions must not leave a pool task or the C ABI of the handler.
template <ffi::DataType dtype>
ffi::Error ContractPart(const Problem& p,
                        const Split& split,
                        int64_t part,
                        const void* a,
                        const void* b,
                        void* d) {
  try {
    return TappError(ContractPartImpl<dtype>(p, split, part, a, b, d));
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("tblis: ") + e.what());
  } catch (...) {
    return ffi::Error(ffi::ErrorCode::kInternal, "tblis: unknown exception");
  }
}

// Number from the environment variable `name`; `fallback` if unset, and if
// not a positive number (with a warning).
double EnvNumber(const char* name, double fallback) {
  const char* str = std::getenv(name);
  if (!str) {
    return fallback;
  }
  char* end = nullptr;
  const double value = std::strtod(str, &end);
  if (end == str || !(value > 0)) {
    std::fprintf(stderr, "varipeps tblis: ignoring %s=%s (not a positive number)\n", name, str);
    return fallback;
  }
  return value;
}

// Rounds to the nearest power of two, so that the noise of the measurement
// rarely changes the split (and with it the summation order of the results).
double RoundToPowerOfTwo(double x) {
  return std::exp2(std::round(std::log2(x)));
}

// Whether the parts of an output split should first map their own range of a new output
// (PopulatePart), decided once per process by a test of this machine: `parts` threads write
// interleaved 64 KiB slices of a fresh buffer (as the parts of an interleaved output split;
// 4 MiB per thread, at least 256 MiB and 2x the L3 cache, at most 512 MiB: a buffer inside the
// cache, e.g. 64 MiB in the 96 MB L3 of a Ryzen 7800X3D, made the mapping look 2x faster), each
// without or with mapping its own range first (best of 3 each, alternating). Whether their first
// writes to the same fresh (huge) pages contend depends on the kernel, not the CPU: kernel 6.12
// with transparent huge pages, i9-10900: contraction ~43 -> ~12 ms with the mapping; kernel 7.2,
// Ryzen 7800X3D: no contention, the mapping costs up to +25-46%. On if it is at least 1.25x
// faster.
// VARIPEPS_TBLIS_PREFAULT=0 or 1 sets it without the test (Linux >= 5.14 only).
struct Prefault {
  bool on = false;
  double plain = 0, mapped = 0;  // seconds of the test
  int64_t threads = 0;
};

#ifdef __linux__
double TimeFirstTouch(int64_t threads, bool mapped) {
  constexpr size_t kHuge = 2 << 20, kSlice = 64 << 10;
  const long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);  // 0 or -1 if unknown
  const size_t bytes = std::min<size_t>(
      size_t(512) << 20, std::max({size_t(256) << 20, static_cast<size_t>(threads) * (4 << 20),
                                   static_cast<size_t>(std::max(l3, 0L)) * 2}));
  void* raw =
      mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) {
    return -1;
  }
  char* data =
      reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + kHuge - 1) & ~(kHuge - 1));
  std::atomic<bool> go{false}, failed{false};
  std::vector<std::thread> workers;
  try {
    for (int64_t t = 0; t < threads; ++t) {
      workers.emplace_back([&, t] {
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        if (failed.load()) {
          return;
        }
        const size_t lo = bytes / threads * t / kHuge * kHuge;
        const size_t hi = t + 1 == threads ? bytes : bytes / threads * (t + 1) / kHuge * kHuge;
        if (mapped && hi > lo && madvise(data + lo, hi - lo, MADV_POPULATE_WRITE) != 0) {
          failed.store(true);
        }
        for (size_t at = t * kSlice; at < bytes; at += threads * kSlice) {
          std::fill(data + at, data + std::min(at + kSlice, bytes), char(1));
        }
      });
    }
  } catch (...) {  // a thread could not be started: release and join the others
    failed.store(true);
  }
  const auto start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto& w : workers) {
    w.join();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  munmap(raw, bytes + kHuge);
  return failed ? -1 : seconds;
}
#endif

const Prefault& GetPrefault() {
  static const Prefault result = [] {
    Prefault r;
    const char* str = std::getenv("VARIPEPS_TBLIS_PREFAULT");
#ifdef __linux__
    if (str && *str) {
      r.on = std::string(str) != "0";
      return r;
    }
    cpu_set_t cpus;
    r.threads = sched_getaffinity(0, sizeof(cpus), &cpus) == 0 ? CPU_COUNT(&cpus) : 1;
    if (const char* n = std::getenv("TBLIS_NUM_THREADS"); n && *n)
      r.threads = std::min<int64_t>(r.threads, std::max<long>(1, std::strtol(n, nullptr, 10)));
    r.threads = std::min<int64_t>(r.threads, 64);
    if (r.threads < 2)
      return r;
    try {
      r.plain = r.mapped = 1e300;
      for (int rep = 0; rep < 3; ++rep) {
        const double plain = TimeFirstTouch(r.threads, false);
        const double mapped = TimeFirstTouch(r.threads, true);
        if (plain < 0 || mapped < 0)
          return Prefault{false, 0, 0, r.threads};  // kernel < 5.14
        r.plain = std::min(r.plain, plain);
        r.mapped = std::min(r.mapped, mapped);
      }
      r.on = r.mapped * 1.25 <= r.plain;
    } catch (...) {  // e.g. threads cannot be started
      r = Prefault{false, 0, 0, r.threads};
    }
    if (std::getenv("VARIPEPS_TBLIS_VERBOSE"))
      std::fprintf(stderr,
                   "varipeps tblis prefault test (%lld threads): plain %.2f ms, mapped %.2f ms "
                   "-> %s\n",
                   static_cast<long long>(r.threads), r.plain * 1e3, r.mapped * 1e3,
                   r.on ? "on" : "off");
#else
    (void)str;
#endif
    return r;
  }();
  return result;
}

// Cost model of this machine for the data type, once per process:
// - min_part_work = kMinPartSeconds / t_fma, with the time of a multiply-add
//   of tblis on a small 64^3 product (including the per-call overhead, like
//   the parts of a split; the in-cache peak rate of large products is ~3x
//   faster and made the threshold ~3x too high, i9-10900), measured single-
//   threaded in the calling thread (a few ms; from Python via cost_model() at
//   trace time, so that it does not compete with an XLA execution) and rounded
//   to a power of two, so that the noise rarely changes the split and the
//   summation order.
// - move_cost = 32, the ratio of the multiply-add rate to the rate of moving
//   (repacking, adding up) elements. Fitted to the measured split choices of
//   one FULL_QR CTMRG step + gradient on an i9-10900 (D=5, chi=64; f64, c128):
//   the chosen splits were on average 1.28x (16) and 1.07x (32) slower than the
//   best measured. Not measured per machine: the memory rate varies by 4x
//   between processes (huge pages, shared bandwidth).
// VARIPEPS_TBLIS_CALIBRATE=0 skips the measurement (min_part_work = 2^17, for
// bitwise reproducible splits across runs), VARIPEPS_TBLIS_MIN_PART_WORK and
// VARIPEPS_TBLIS_MOVE_COST override, VARIPEPS_TBLIS_VERBOSE=1 prints the model.
template <ffi::DataType dtype>
CostModel CalibrateCostModel() {
  using T = ffi::NativeType<dtype>;
  GetPrefault();  // also tested here, from Python at trace time
  CostModel model{1 << 17, 32, 32, kMaxPartialBytes, 0, kFoldPenalty};
  const char* calibrate = std::getenv("VARIPEPS_TBLIS_CALIBRATE");
  const bool measure =
      !std::getenv("VARIPEPS_TBLIS_MIN_PART_WORK") && !(calibrate && std::string(calibrate) == "0");
  double fma = 0;
  if (measure) {
    try {
      constexpr int64_t n = 64, runs = 40;  // 40 products per sample (timer resolution)
      std::vector<T> a(n * n, T(1)), b(n * n, T(1)), d(n * n);
      const Problem p{{{n, n}, {0, 1}}, {{n, n}, {1, 2}}, {{n, n}, {0, 2}}};
      auto run = [&] {
        return ContractPartImpl<dtype>(p, Split{}, 0, a.data(), b.data(), d.data());
      };
      double best = 1e300;
      if (run() == 0) {  // warm-up: first use of tblis, page faults
        for (int rep = 0; rep < 7; ++rep) {
          const auto start = std::chrono::steady_clock::now();
          for (int i = 0; i < runs; ++i) {
            run();
          }
          best = std::min(
              best,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        fma = best / (runs * n * n * n);
      }
    } catch (...) {
      fma = 0;
    }
  }
  if (fma > 0) {
    model.min_part_work = RoundToPowerOfTwo(std::clamp(kMinPartSeconds / fma, 1e4, 1e7));
  }
  // Overrides (benchmarks): scale or value of min_part_work, the costs, the memory limit.
  model.min_part_work *= EnvNumber("VARIPEPS_TBLIS_MIN_PART_SCALE", 1);
  model.min_part_work = EnvNumber("VARIPEPS_TBLIS_MIN_PART_WORK", model.min_part_work);
  model.move_cost = EnvNumber("VARIPEPS_TBLIS_MOVE_COST", model.move_cost);
  model.reduce_cost = EnvNumber("VARIPEPS_TBLIS_REDUCE_COST", model.reduce_cost);
  model.max_partial_bytes = EnvNumber("VARIPEPS_TBLIS_MAX_PARTIAL_BYTES", model.max_partial_bytes);
  model.even_parts = EnvNumber("VARIPEPS_TBLIS_EVEN_PARTS", model.even_parts);
  model.fold_penalty = EnvNumber("VARIPEPS_TBLIS_FOLD_PENALTY", model.fold_penalty);
  if (std::getenv("VARIPEPS_TBLIS_VERBOSE")) {
    std::fprintf(stderr,
                 "varipeps tblis cost model (%zu-byte elements, %s): min_part_work=%.3g "
                 "move_cost=%.3g\n",
                 sizeof(T), fma > 0 ? "measured" : "defaults", model.min_part_work,
                 model.move_cost);
  }
  return model;
}

template <ffi::DataType dtype>
const CostModel& GetCostModel() {
  static const CostModel model = CalibrateCostModel<dtype>();
  return model;
}

// Number of parts: the size of the XLA pool, at most TBLIS_NUM_THREADS if set
// (e.g. the number of physical cores, the pool also counts hyperthreads). On
// Apple Silicon the default is the number of performance cores: a part on an
// efficiency core finishes late and holds up the whole contraction.
int64_t MaxParts(const ffi::ThreadPool& pool) {
  static const int64_t limit = []() -> int64_t {
    if (const char* str = std::getenv("TBLIS_NUM_THREADS"); str && *str) {
      return std::max<int64_t>(1, std::strtol(str, nullptr, 10));
    }
#ifdef __APPLE__
    int cores = 0;
    size_t len = sizeof(cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &len, nullptr, 0) == 0 && cores > 0)
      return cores;
#endif
    return INT64_MAX;
  }();
  return std::min(pool.num_threads(), limit);
}

// Runs `task` on the pool, or in the calling thread if it cannot be scheduled.
template <typename F>
void Schedule(ffi::ThreadPool& pool, F task) {
  try {
    pool.Schedule(F(task));
  } catch (...) {
    task();
  }
}

// Maps the pages of range `part` of `parts` of the output buffer (contiguous; inner bounds
// 2 MiB aligned) without changing its contents (Linux >= 5.14, MADV_POPULATE_WRITE). The
// parts of an output split write interleaved slices of D: their first writes to the same
// fresh (huge) pages of a new buffer contend in the kernel, on 10 cores up to 2-3x the
// contraction time (transparent huge pages, i9-10900: e.g. ~43 -> ~12 ms, kernel time
// ~350 -> 40-55 ms). Each part maps its own range first instead; for a reused buffer this
// is a cheap page-table walk. Only where it pays off (GetPrefault).
void PopulatePart(void* data, size_t bytes, int64_t part, int64_t parts) {
#ifdef __linux__
  if (bytes < kPrefaultBytes || !GetPrefault().on) {
    return;
  }
  constexpr uintptr_t kHuge = 2 << 20;
  static const uintptr_t kPage = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data), end = begin + bytes;
  const auto bound = [&](int64_t k) -> uintptr_t {
    if (k == 0) {
      return begin & ~(kPage - 1);
    }
    if (k == parts) {
      return (end + kPage - 1) & ~(kPage - 1);
    }
    const uintptr_t at = begin + bytes / parts * k;
    return std::min((at + kHuge - 1) & ~(kHuge - 1), (end + kPage - 1) & ~(kPage - 1));
  };
  static std::atomic<bool> supported{true};  // false after EINVAL (kernel < 5.14, sandboxes)
  const uintptr_t lo = bound(part), hi = bound(part + 1);
  if (hi > lo && supported.load(std::memory_order_relaxed) &&
      madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_POPULATE_WRITE) != 0 && errno == EINVAL) {
    supported.store(false, std::memory_order_relaxed);  // other errors: ignored
  }
#else
  (void)data, (void)bytes, (void)part, (void)parts;
#endif
}

template <ffi::DataType dtype>
ffi::Future TblisContractImpl(ffi::ThreadPool pool,
                              ffi::Buffer<dtype> a,
                              ffi::Buffer<dtype> b,
                              ffi::ResultBuffer<dtype> d,
                              ffi::Span<const int64_t> idx_a,
                              ffi::Span<const int64_t> idx_b,
                              ffi::Span<const int64_t> idx_d) {
  ffi::Promise promise;
  ffi::Future future(promise);
  if (idx_a.size() != a.dimensions().size() || idx_b.size() != b.dimensions().size() ||
      idx_d.size() != d->dimensions().size()) {
    promise.SetError(ffi::Error(ffi::ErrorCode::kInvalidArgument,
                                "Number of labels does not match the tensor ranks."));
    return future;
  }

  auto problem = std::make_shared<const Problem>(
      Problem{{a.dimensions(), idx_a}, {b.dimensions(), idx_b}, {d->dimensions(), idx_d}});
  const Split split =
      ChooseSplit<ffi::NativeType<dtype>>(*problem, MaxParts(pool), GetCostModel<dtype>());
  const void* data_a = a.untyped_data();
  const void* data_b = b.untyped_data();
  void* data_d = d->untyped_data();

  if (split.parts < 2) {
    ffi::Error err = ContractPart<dtype>(*problem, split, 0, data_a, data_b, data_d);
    err.success() ? promise.SetAvailable() : promise.SetError(err);
    return future;
  }

  // The buffers stay alive until the future completes.
  if (split.pos_d < 0) {
    using T = ffi::NativeType<dtype>;
    struct Reduction {
      ffi::Promise promise;
      // Parts 1, 2, ...; part 0 writes D. Not initialized: tblis overwrites.
      std::vector<std::unique_ptr<T[], FreeDeleter>> partial;
      std::atomic<int64_t> remaining;
      std::mutex mutex;
      ffi::Error error = ffi::Error::Success();
    };
    std::shared_ptr<Reduction> state;
    const size_t size_d = d->element_count();
    try {
      state = std::make_shared<Reduction>();
      state->partial.reserve(split.parts - 1);
      for (int64_t part = 1; part < split.parts; ++part) {
        // Cache-line aligned like the XLA buffers (malloc: 16 bytes for large sizes).
        T* buffer =
            static_cast<T*>(std::aligned_alloc(64, (size_d * sizeof(T) + 63) & ~size_t(63)));
        if (!buffer) {
          throw std::bad_alloc();
        }
        state->partial.emplace_back(buffer);
      }
    } catch (...) {
      promise.SetError(ffi::Error(ffi::ErrorCode::kResourceExhausted,
                                  "tblis: no memory for the partial results"));
      return future;
    }
    state->promise = std::move(promise);
    state->remaining = split.parts;
    // The last part to finish adds the partial results to D.
    auto finish = [state, data_d, size_d](ffi::Error err) {
      if (!err.success()) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->error = err;
      }
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
      }
      if (!state->error.success()) {
        return state->promise.SetError(state->error);
      }
      // Blocked, so that D is traversed once.
      T* out = static_cast<T*>(data_d);
      constexpr size_t kBlock = 4096;
      for (size_t k0 = 0; k0 < size_d; k0 += kBlock) {
        const size_t k1 = std::min(size_d, k0 + kBlock);
        for (const auto& partial : state->partial) {
          for (size_t k = k0; k < k1; ++k) {
            out[k] += partial[k];
          }
        }
      }
      state->promise.SetAvailable();
    };
    for (int64_t part = 1; part < split.parts; ++part) {
      Schedule(pool, [=]() {
        finish(ContractPart<dtype>(*problem, split, part, data_a, data_b,
                                   state->partial[part - 1].get()));
      });
    }
    finish(ContractPart<dtype>(*problem, split, 0, data_a, data_b, data_d));
    return future;
  }

  ffi::CountDownPromise done(std::move(promise), split.parts);
  // Only if the parts interleave within huge pages (contiguous runs < 2 MiB): parts that
  // write whole huge pages do not contend, and mapping first costs them the cache-hot
  // zeroing (measured: 134 MB output in runs of 16 MiB: +17-21%).
  const size_t item = sizeof(ffi::NativeType<dtype>);
  const bool interleaved = split.part_len * problem->d.strides[split.pos_d] * item < (2 << 20);
  const size_t bytes_d = interleaved ? d->element_count() * item : 0;
  for (int64_t part = 1; part < split.parts; ++part) {
    Schedule(pool, [=]() mutable {
      PopulatePart(data_d, bytes_d, part, split.parts);
      done.CountDown(ContractPart<dtype>(*problem, split, part, data_a, data_b, data_d));
    });
  }
  PopulatePart(data_d, bytes_d, 0, split.parts);
  done.CountDown(ContractPart<dtype>(*problem, split, 0, data_a, data_b, data_d));
  return future;
}

}  // namespace

#define DEFINE_TBLIS_CONTRACT(fname, dtype)                                  \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(fname, TblisContractImpl<dtype>,             \
                                ffi::Ffi::Bind()                             \
                                    .Ctx<ffi::ThreadPool>()                  \
                                    .Arg<ffi::Buffer<dtype>>(/*a*/)          \
                                    .Arg<ffi::Buffer<dtype>>(/*b*/)          \
                                    .Ret<ffi::Buffer<dtype>>(/*d*/)          \
                                    .Attr<ffi::Span<const int64_t>>("idx_a") \
                                    .Attr<ffi::Span<const int64_t>>("idx_b") \
                                    .Attr<ffi::Span<const int64_t>>("idx_d"))

DEFINE_TBLIS_CONTRACT(tblis_contract_f32, ffi::DataType::F32);
DEFINE_TBLIS_CONTRACT(tblis_contract_f64, ffi::DataType::F64);
DEFINE_TBLIS_CONTRACT(tblis_contract_c64, ffi::DataType::C64);
DEFINE_TBLIS_CONTRACT(tblis_contract_c128, ffi::DataType::C128);

template <typename T>
static nb::capsule EncapsulateFfiCall(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

// Split chosen for a contraction (for benchmarks and reports): kind "serial",
// "output" or "reduction", label, number of parts, part length.
template <ffi::DataType dtype>
std::tuple<std::string, int64_t, int64_t, int64_t> SplitPlan(const std::vector<int64_t>& idx_a,
                                                             const std::vector<int64_t>& idx_b,
                                                             const std::vector<int64_t>& idx_d,
                                                             const std::vector<int64_t>& dims_a,
                                                             const std::vector<int64_t>& dims_b,
                                                             const std::vector<int64_t>& dims_d,
                                                             int64_t threads,
                                                             const std::vector<double>& overrides) {
  const Problem p{{dims_a, idx_a}, {dims_b, idx_b}, {dims_d, idx_d}};
  CostModel model = GetCostModel<dtype>();
  double* fields[] = {&model.min_part_work,     &model.move_cost,  &model.reduce_cost,
                      &model.max_partial_bytes, &model.even_parts, &model.fold_penalty};
  for (size_t i = 0; i < std::min<size_t>(overrides.size(), 6); ++i) {
    if (overrides[i] > 0) {
      *fields[i] = overrides[i];
    }
  }
  const Split s = ChooseSplit<ffi::NativeType<dtype>>(p, threads, model);
  if (s.parts < 2) {
    return {"serial", -1, 1, 0};
  }
  if (s.pos_d >= 0) {
    return {"output", p.d.idx[s.pos_d], s.parts, s.part_len};
  }
  return {"reduction", p.a.idx[s.pos_a], s.parts, s.part_len};
}

NB_MODULE(_tblis_contract, m) {
  m.def(
      "split_plan",
      [](const std::vector<int64_t>& idx_a, const std::vector<int64_t>& idx_b,
         const std::vector<int64_t>& idx_d, const std::vector<int64_t>& dims_a,
         const std::vector<int64_t>& dims_b, const std::vector<int64_t>& dims_d,
         const std::string& dtype, int64_t threads, const std::vector<double>& overrides) {
        if (dtype != "f32" && dtype != "f64" && dtype != "c64" && dtype != "c128") {
          throw std::invalid_argument("dtype must be f32, f64, c64 or c128");
        }
        auto plan = dtype == "f32"   ? SplitPlan<ffi::DataType::F32>
                    : dtype == "f64" ? SplitPlan<ffi::DataType::F64>
                    : dtype == "c64" ? SplitPlan<ffi::DataType::C64>
                                     : SplitPlan<ffi::DataType::C128>;
        return plan(idx_a, idx_b, idx_d, dims_a, dims_b, dims_d, threads, overrides);
      },
      nb::arg("idx_a"), nb::arg("idx_b"), nb::arg("idx_d"), nb::arg("dims_a"), nb::arg("dims_b"),
      nb::arg("dims_d"), nb::arg("dtype"), nb::arg("threads"),
      nb::arg("overrides") = std::vector<double>{},
      "Split of the contraction D = A * B chosen for `threads` threads: (kind, label, parts, "
      "part_len); overrides: (min_part_work, move_cost, reduce_cost, max_partial_bytes, "
      "even_parts, "
      "fold_penalty), values "
      "> 0 replace the cost model. For benchmarks and reports.");
  m.def(
      "cost_model",
      [](const std::string& dtype) {
        if (dtype != "f32" && dtype != "f64" && dtype != "c64" && dtype != "c128") {
          throw std::invalid_argument("dtype must be f32, f64, c64 or c128");
        }
        const CostModel& c = dtype == "f32"   ? GetCostModel<ffi::DataType::F32>()
                             : dtype == "f64" ? GetCostModel<ffi::DataType::F64>()
                             : dtype == "c64" ? GetCostModel<ffi::DataType::C64>()
                                              : GetCostModel<ffi::DataType::C128>();
        return std::make_tuple(c.min_part_work, c.move_cost, c.reduce_cost, c.max_partial_bytes);
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Cost model of the thread split for dtype f32/f64/c64/c128: (min_part_work, "
      "move_cost, reduce_cost, max_partial_bytes), measured on the first call.");
  m.def(
      "prefault",
      []() {
        const Prefault& p = GetPrefault();
        return std::make_tuple(p.on, p.plain, p.mapped, p.threads);
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Whether new outputs of output splits are mapped first: (on, seconds without, seconds "
      "with, threads) of the test of this machine (0 if not tested).");
  m.def("tblis_contract_f32", []() { return EncapsulateFfiCall(tblis_contract_f32); });
  m.def("tblis_contract_f64", []() { return EncapsulateFfiCall(tblis_contract_f64); });
  m.def("tblis_contract_c64", []() { return EncapsulateFfiCall(tblis_contract_c64); });
  m.def("tblis_contract_c128", []() { return EncapsulateFfiCall(tblis_contract_c128); });
}
