// Created with Claude Opus 5.5 (AI-generated).
//
// Minimal implementation of the TAPP interface (handles, executors, tensor
// infos and tensor products) on top of tblis. Only the parts needed for
// D = alpha * op(A) * op(B) + beta * op(C) are implemented. Labels that occur
// in only one input tensor and not in D (traces/sums) are rejected, since
// tblis_tensor_mult does not support them; labels repeated within one tensor
// (diagonals) are rejected for simplicity. op_D must be the identity.
// Products are executed single-threaded, see TAPP_create_executor.

#include <tapp.h>
#include <tblis/tblis.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <map>
#include <vector>

namespace {

using tblis::label_type;
using tblis::len_type;
using tblis::stride_type;
using tblis::tblis_tensor;

enum : TAPP_error {
  kSuccess = 0,
  kInvalidDatatype,
  kInvalidNmode,
  kInvalidExtent,
  kRepeatedLabel,
  kSingleOperandLabel,
  kUnknownOutputLabel,
  kExtentMismatch,
  kCMismatchD,
  kTypeMismatch,
  kUnsupportedOp,
  kNullPointer,
  kTooManyLabels,
  kNumErrors
};

const char* const kErrorMessages[kNumErrors] = {
    "Success.",
    "Unsupported datatype.",
    "Negative number of modes.",
    "Negative extent.",
    "Label repeated within one tensor (not supported).",
    "Label only in one of A and B and not in D (not supported).",
    "Label of D neither in A nor in B.",
    "Extents of a label do not match.",
    "C must have the same labels and extents as D.",
    "Datatypes of the tensors differ.",
    "Unsupported element operation on D.",
    "Null pointer argument.",
    "Too many distinct labels.",
};

struct TensorInfo {
  TAPP_datatype type;
  std::vector<len_type> extents;
  std::vector<stride_type> strides;
};

struct Product {
  TAPP_element_op op_a, op_b, op_c;
  TensorInfo a, b, c, d;
  // Labels remapped to 1..k for the char labels of tblis.
  std::vector<label_type> idx_a, idx_b, idx_c, idx_d;
};

TAPP_error validate(const TensorInfo& a,
                    const int64_t* idx_a,
                    const TensorInfo& b,
                    const int64_t* idx_b,
                    const TensorInfo& c,
                    const int64_t* idx_c,
                    const TensorInfo& d,
                    const int64_t* idx_d,
                    std::map<int64_t, label_type>& labels) {
  if (a.type != d.type || b.type != d.type || c.type != d.type) {
    return kTypeMismatch;
  }
  if (c.extents != d.extents || !std::equal(idx_d, idx_d + d.extents.size(), idx_c)) {
    return kCMismatchD;
  }

  // Per label: extent and in which of A, B, D it occurs.
  struct Use {
    int64_t extent;
    int count[3];
  };
  std::map<int64_t, Use> uses;
  const TensorInfo* infos[3] = {&a, &b, &d};
  const int64_t* idxs[3] = {idx_a, idx_b, idx_d};
  for (int t = 0; t < 3; ++t) {
    for (size_t i = 0; i < infos[t]->extents.size(); ++i) {
      auto [it, inserted] = uses.try_emplace(idxs[t][i], Use{infos[t]->extents[i], {0, 0, 0}});
      if (it->second.extent != infos[t]->extents[i]) {
        return kExtentMismatch;
      }
      if (++it->second.count[t] > 1) {
        return kRepeatedLabel;
      }
    }
  }
  for (const auto& [label, use] : uses) {
    if (use.count[2] && !use.count[0] && !use.count[1]) {
      return kUnknownOutputLabel;
    }
    if (!use.count[2] && (!use.count[0] || !use.count[1])) {
      return kSingleOperandLabel;
    }
  }
  if (uses.size() > 126) {
    return kTooManyLabels;
  }

  label_type next = 1;
  for (const auto& [label, use] : uses) {
    labels[label] = next++;
  }
  return kSuccess;
}

template <typename T>
TAPP_error execute(const Product& p,
                   const void* alpha,
                   const void* A,
                   const void* B,
                   const void* beta,
                   const void* C,
                   void* D) {
  if (!alpha || !beta || !A || !B || !D) {
    return kNullPointer;
  }
  const T al = *static_cast<const T*>(alpha);
  const T be = *static_cast<const T*>(beta);

  auto wrap = [](T scalar, bool conj, const void* data, const TensorInfo& info) {
    return tblis_tensor(scalar, conj, static_cast<const T*>(data),
                        static_cast<int>(info.extents.size()), info.extents.data(),
                        info.strides.data());
  };

  if (be != T(0) && (C != D || p.op_c == TAPP_CONJUGATE)) {
    if (!C) {
      return kNullPointer;
    }
    tblis_tensor tc = wrap(T(1), p.op_c == TAPP_CONJUGATE, C, p.c);
    tblis_tensor td = wrap(T(0), false, D, p.d);
    tblis::tblis_tensor_add(tblis_single, nullptr, &tc, p.idx_c.data(), &td, p.idx_d.data());
  }

  tblis_tensor ta = wrap(al, p.op_a == TAPP_CONJUGATE, A, p.a);
  tblis_tensor tb = wrap(T(1), p.op_b == TAPP_CONJUGATE, B, p.b);
  tblis_tensor td = wrap(be, false, D, p.d);
  tblis::tblis_tensor_mult(tblis_single, nullptr, &ta, p.idx_a.data(), &tb, p.idx_b.data(), &td,
                           p.idx_d.data());
  return kSuccess;
}

}  // namespace

TAPP_EXPORT bool TAPP_check_success(TAPP_error error) {
  return error == kSuccess;
}

TAPP_EXPORT size_t TAPP_explain_error(TAPP_error error, size_t maxlen, char* message) {
  const char* text =
      (error >= 0 && error < kNumErrors) ? kErrorMessages[error] : "Unknown error code.";
  const size_t len = std::strlen(text);
  if (maxlen == 0) {
    return len;
  }
  const size_t n = std::min(len, maxlen - 1);
  std::memcpy(message, text, n);
  message[n] = '\0';
  return n;
}

// Handles and executors carry no state. tblis always runs single-threaded
// (tblis_single), as in the TAPP reference; parallelism is up to the caller
// (tblis_ffi.cpp schedules independent parts on the thread pool of XLA).
TAPP_EXPORT TAPP_error TAPP_create_handle(TAPP_handle* handle) {
  if (!handle) {
    return kNullPointer;
  }
  *handle = 1;
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_destroy_handle(TAPP_handle) {
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_create_executor(TAPP_executor* exec) {
  if (!exec) {
    return kNullPointer;
  }
  *exec = 1;
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor) {
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                               TAPP_datatype type,
                                               int nmode,
                                               const int64_t* extents,
                                               const int64_t* strides) {
  if (!info || (nmode > 0 && (!extents || !strides))) {
    return kNullPointer;
  }
  if (type != TAPP_F32 && type != TAPP_F64 && type != TAPP_C32 && type != TAPP_C64) {
    return kInvalidDatatype;
  }
  if (nmode < 0) {
    return kInvalidNmode;
  }
  if (std::any_of(extents, extents + nmode, [](int64_t e) { return e < 0; })) {
    return kInvalidExtent;
  }
  *info = reinterpret_cast<TAPP_tensor_info>(
      new TensorInfo{type, std::vector<len_type>(extents, extents + nmode),
                     std::vector<stride_type>(strides, strides + nmode)});
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info) {
  delete reinterpret_cast<TensorInfo*>(info);
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                                  TAPP_handle,
                                                  TAPP_element_op op_A,
                                                  TAPP_tensor_info A,
                                                  const int64_t* idx_A,
                                                  TAPP_element_op op_B,
                                                  TAPP_tensor_info B,
                                                  const int64_t* idx_B,
                                                  TAPP_element_op op_C,
                                                  TAPP_tensor_info C,
                                                  const int64_t* idx_C,
                                                  TAPP_element_op op_D,
                                                  TAPP_tensor_info D,
                                                  const int64_t* idx_D,
                                                  TAPP_prectype) {
  if (!plan || !A || !B || !C || !D) {
    return kNullPointer;
  }
  if (op_D != TAPP_IDENTITY) {
    return kUnsupportedOp;
  }

  const auto& a = *reinterpret_cast<TensorInfo*>(A);
  const auto& b = *reinterpret_cast<TensorInfo*>(B);
  const auto& c = *reinterpret_cast<TensorInfo*>(C);
  const auto& d = *reinterpret_cast<TensorInfo*>(D);

  std::map<int64_t, label_type> labels;
  if (TAPP_error err = validate(a, idx_A, b, idx_B, c, idx_C, d, idx_D, labels)) {
    return err;
  }

  auto remap = [&](const int64_t* idx, size_t n) {
    std::vector<label_type> out(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
      out[i] = labels.at(idx[i]);
    }
    return out;
  };

  *plan = reinterpret_cast<TAPP_tensor_product>(new Product{
      op_A, op_B, op_C, a, b, c, d, remap(idx_A, a.extents.size()), remap(idx_B, b.extents.size()),
      remap(idx_C, c.extents.size()), remap(idx_D, d.extents.size())});
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan) {
  delete reinterpret_cast<Product*>(plan);
  return kSuccess;
}

TAPP_EXPORT TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                            TAPP_executor,
                                            TAPP_status*,
                                            const void* alpha,
                                            const void* A,
                                            const void* B,
                                            const void* beta,
                                            const void* C,
                                            void* D) {
  if (!plan) {
    return kNullPointer;
  }
  const auto& p = *reinterpret_cast<Product*>(plan);
  switch (p.d.type) {
    case TAPP_F32:
      return execute<float>(p, alpha, A, B, beta, C, D);
    case TAPP_F64:
      return execute<double>(p, alpha, A, B, beta, C, D);
    case TAPP_C32:
      return execute<std::complex<float>>(p, alpha, A, B, beta, C, D);
    case TAPP_C64:
      return execute<std::complex<double>>(p, alpha, A, B, beta, C, D);
    default:
      return kInvalidDatatype;
  }
}
