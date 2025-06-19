#ifndef VARIPEPS_SVD_FFI_H_
#define VARIPEPS_SVD_FFI_H_

#include "xla/ffi/api/ffi.h"

enum class UVtMode : int8_t {
  computeOnlyU = 0,         // Compute only U
  computeOnlyVt = 1,        // Compute only Vt
  computePartialUandVt = 2, // Compute only Vt
};

XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_f32);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_f64);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_c64);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_c128);

XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_qr_f32);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_qr_f64);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_qr_c64);
XLA_FFI_DECLARE_HANDLER_SYMBOL(svd_only_u_vt_qr_c128);

#endif  // VARIPEPS_SVD_FFI_H
