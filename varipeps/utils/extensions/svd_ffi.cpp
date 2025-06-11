#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

using lapack_int = int;

namespace nb = nanobind;
using namespace ::xla;

inline constexpr auto LapackIntDtype = ffi::DataType::S32;
static_assert(std::is_same_v<::xla::ffi::NativeType<LapackIntDtype>, lapack_int>);

template <ffi::DataType dtype>
static ffi::Error SvdOnlyVtImpl(
    ffi::Buffer<dtype> x,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> s,
    ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info) {

  using MachineType = ffi::NativeType<dtype>;
  using RealType = ffi::NativeType<ffi::ToReal(dtype)>;
  using FnSig = std::conditional_t<ffi::IsComplexType<dtype>(),
                                   void(char const* jobz,
                                        lapack_int const* m, lapack_int const* n,
                                        MachineType* A, lapack_int const* lda,
                                        RealType* S,
                                        MachineType* U, lapack_int const* ldu,
                                        MachineType* VT, lapack_int const* ldvt,
                                        MachineType* work, lapack_int const* lwork,
                                        RealType* rwork,
                                        lapack_int* iwork,
                                        lapack_int* info
                                        ),
                                   void(char const* jobz,
                                        lapack_int const* m, lapack_int const* n,
                                        MachineType* A, lapack_int const* lda,
                                        RealType* S,
                                        MachineType* U, lapack_int const* ldu,
                                        MachineType* VT, lapack_int const* ldvt,
                                        MachineType* work, lapack_int const* lwork,
                                        lapack_int* iwork,
                                        lapack_int* info
                                        )>;

  FnSig* fn = nullptr;

  try {
    PyGILState_STATE state = PyGILState_Ensure();

    nb::module_ cython_lapack = nb::module_::import_("scipy.linalg.cython_lapack");

    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");

    auto get_lapack_ptr = [&](const char* name) {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };

    if constexpr (dtype == ffi::DataType::F32) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("sgesdd"));
    }
    if constexpr (dtype == ffi::DataType::F64) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("dgesdd"));
    }
    if constexpr (dtype == ffi::DataType::C64) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("cgesdd"));
    }
    if constexpr (dtype == ffi::DataType::C128) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("zgesdd"));
    }

    PyGILState_Release(state);
  } catch (const nb::python_error &e) {
    std::cerr << e.what() << std::endl;
    throw;
  }

  const auto lapack_int_max = std::numeric_limits<lapack_int>::max();

  ffi::Span<const int64_t> dims = x.dimensions();
  if (dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only 2d arrays supported as input.");
  }
  int64_t x_rows = dims.front();
  int64_t x_cols = dims.back();

  if (x_rows < x_cols) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only matrices with M >= N supported.");
  }

  if (x_rows > lapack_int_max || x_cols > lapack_int_max) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kOutOfRange, "Dimension of input out of range for lapack integer.");
  }
  
  lapack_int x_rows_lapack = static_cast<lapack_int>(x_rows);
  lapack_int x_cols_lapack = static_cast<lapack_int>(x_cols);

  auto* x_out_data = x_out->typed_data();
  auto* s_data = s->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  if (x.typed_data() != x_out_data) {
    std::copy_n(x.typed_data(), x.element_count(), x_out_data);
  }

  ffi::NativeType<dtype> work_size = {};
  lapack_int lwork = -1;
  char jobz = 'O';
  lapack_int ldu = 1;

  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&jobz, &x_rows_lapack, &x_cols_lapack, nullptr,
       &x_rows_lapack, nullptr, nullptr,
       &ldu, nullptr, &x_cols_lapack, &work_size,
       &lwork, nullptr, nullptr, info_data
       );
  } else {
    fn(&jobz, &x_rows_lapack, &x_cols_lapack, nullptr,
       &x_rows_lapack, nullptr, nullptr,
       &ldu, nullptr, &x_cols_lapack,
       &work_size, &lwork, nullptr, info_data
       );
  }

  if (*info_data != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Non zero info returned by lapack.");
  }

  if (std::real(work_size) > lapack_int_max) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kOutOfRange, "Workspace size out of range for lapack integer.");
  }
  lwork = static_cast<lapack_int>(std::real(work_size));

  const auto min_dim = std::min(x_rows, x_cols);
  const auto max_dim = std::max(x_rows, x_cols);
  
  auto work = std::unique_ptr<ffi::NativeType<dtype>[]>(new ffi::NativeType<dtype>[lwork]);

  auto iwork = std::unique_ptr<lapack_int[]>(new lapack_int[8*min_dim]);

  std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]> rwork;
  if constexpr (ffi::IsComplexType<dtype>()) {
    const auto rwork_size = std::max(5 * min_dim * min_dim + 5 * min_dim,
                                     2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim);
    rwork = std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]>(new ffi::NativeType<ffi::ToReal(dtype)>[rwork_size]);
  }

  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&jobz, &x_rows_lapack, &x_cols_lapack, x_out_data,
       &x_rows_lapack, s_data, nullptr,
       &ldu, vt_data, &x_cols_lapack, work.get(),
       &lwork, rwork.get(), iwork.get(), info_data
       );
  } else {
    fn(&jobz, &x_rows_lapack, &x_cols_lapack, x_out_data,
       &x_rows_lapack, s_data, nullptr,
       &ldu, vt_data, &x_cols_lapack,
       work.get(), &lwork, iwork.get(), info_data
       );
  }

  if (*info_data != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Non zero info returned by lapack.");
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
static ffi::Error SvdOnlyVtQRImpl(
    ffi::Buffer<dtype> x,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> s,
    ffi::ResultBuffer<LapackIntDtype> info) {

  using MachineType = ffi::NativeType<dtype>;
  using RealType = ffi::NativeType<ffi::ToReal(dtype)>;
  using FnSig = std::conditional_t<ffi::IsComplexType<dtype>(),
                                   void(char const* jobu, char const* jobvt,
                                        lapack_int const* m, lapack_int const* n,
                                        MachineType* A, lapack_int const* lda,
                                        RealType* S,
                                        MachineType* U, lapack_int const* ldu,
                                        MachineType* VT, lapack_int const* ldvt,
                                        MachineType* work, lapack_int const* lwork,
                                        RealType* rwork,
                                        lapack_int* info
                                        ),
                                   void(char const* jobu, char const* jobvt,
                                        lapack_int const* m, lapack_int const* n,
                                        MachineType* A, lapack_int const* lda,
                                        RealType* S,
                                        MachineType* U, lapack_int const* ldu,
                                        MachineType* VT, lapack_int const* ldvt,
                                        MachineType* work, lapack_int const* lwork,
                                        lapack_int* info
                                        )>;

  FnSig* fn = nullptr;

  try {
    PyGILState_STATE state = PyGILState_Ensure();
    
    nb::module_ cython_lapack = nb::module_::import_("scipy.linalg.cython_lapack");

    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");

    auto get_lapack_ptr = [&](const char* name) {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };

    if constexpr (dtype == ffi::DataType::F32) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("sgesvd"));
    }
    if constexpr (dtype == ffi::DataType::F64) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("dgesvd"));
    }
    if constexpr (dtype == ffi::DataType::C64) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("cgesvd"));
    }
    if constexpr (dtype == ffi::DataType::C128) {
      fn = reinterpret_cast<FnSig*>(get_lapack_ptr("zgesvd"));
    }

    PyGILState_Release(state);
  } catch (const nb::python_error &e) {
    std::cerr << e.what() << std::endl;
    throw;
  }

  const auto lapack_int_max = std::numeric_limits<lapack_int>::max();

  ffi::Span<const int64_t> dims = x.dimensions();
  if (dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only 2d arrays supported as input.");
  }
  int64_t x_rows = dims.front();
  int64_t x_cols = dims.back();

  if (x_rows < x_cols) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only matrices with M >= N supported.");
  }

  if (x_rows > lapack_int_max || x_cols > lapack_int_max) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kOutOfRange, "Dimension of input out of range for lapack integer.");
  }

  lapack_int x_rows_lapack = static_cast<lapack_int>(x_rows);
  lapack_int x_cols_lapack = static_cast<lapack_int>(x_cols);

  auto* x_out_data = x_out->typed_data();
  auto* s_data = s->typed_data();
  // auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  if (x.typed_data() != x_out_data) {
    std::copy_n(x.typed_data(), x.element_count(), x_out_data);
  }

  ffi::NativeType<dtype> work_size = {};
  lapack_int lwork = -1;
  char jobu = 'N';
  char jobvt = 'O';
  lapack_int ldu = 1;

  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&jobu, &jobvt, &x_rows_lapack, &x_cols_lapack, nullptr,
       &x_rows_lapack, nullptr, nullptr,
       &ldu, nullptr, &x_cols_lapack, &work_size,
       &lwork, nullptr, info_data
       );
  } else {
    fn(&jobu, &jobvt, &x_rows_lapack, &x_cols_lapack, nullptr,
       &x_rows_lapack, nullptr, nullptr,
       &ldu, nullptr, &x_cols_lapack,
       &work_size, &lwork, info_data
       );
  }

  if (*info_data != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Non zero info returned by lapack.");
  }

  if (std::real(work_size) > lapack_int_max) [[unlikely]] {
    return ffi::Error(ffi::ErrorCode::kOutOfRange, "Workspace size out of range for lapack integer.");
  }
  lwork = static_cast<lapack_int>(std::real(work_size));

  const auto min_dim = std::min(x_rows, x_cols);
  const auto max_dim = std::max(x_rows, x_cols);

  auto work = std::unique_ptr<ffi::NativeType<dtype>[]>(new ffi::NativeType<dtype>[lwork]);

  std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]> rwork;
  if constexpr (ffi::IsComplexType<dtype>()) {
    const auto rwork_size = 5 * min_dim;
    rwork = std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]>(new ffi::NativeType<ffi::ToReal(dtype)>[rwork_size]);
  }

  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&jobu, &jobvt, &x_rows_lapack, &x_cols_lapack, x_out_data,
       &x_rows_lapack, s_data, nullptr,
       &ldu, nullptr, &x_cols_lapack, work.get(),
       &lwork, rwork.get(), info_data
       );
  } else {
    fn(&jobu, &jobvt, &x_rows_lapack, &x_cols_lapack, x_out_data,
       &x_rows_lapack, s_data, nullptr,
       &ldu, nullptr, &x_cols_lapack,
       work.get(), &lwork, info_data
       );
  }

  if (*info_data != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Non zero info returned by lapack.");
  }

  return ffi::Error::Success();
}

#define DEFINE_REAL_SVD_ONLY_VT(fname, dtype)      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                   \
      fname, SvdOnlyVtImpl<dtype>,                 \
      ffi::Ffi::Bind()                             \
          .Arg<ffi::Buffer<dtype>>(/*x*/)          \
          .Ret<ffi::Buffer<dtype>>(/*x_out*/)      \
          .Ret<ffi::Buffer<dtype>>(/*s*/)          \
          .Ret<ffi::Buffer<dtype>>(/*vt*/)         \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define DEFINE_COMPLEX_SVD_ONLY_VT(fname, dtype)       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                       \
      fname, SvdOnlyVtImpl<dtype>,                     \
      ffi::Ffi::Bind()                                 \
          .Arg<ffi::Buffer<dtype>>(/*x*/)              \
          .Ret<ffi::Buffer<dtype>>(/*x_out*/)          \
          .Ret<ffi::Buffer<ffi::ToReal(dtype)>>(/*s*/) \
          .Ret<ffi::Buffer<dtype>>(/*vt*/)             \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

DEFINE_REAL_SVD_ONLY_VT(svd_only_vt_f32, ffi::DataType::F32);
DEFINE_REAL_SVD_ONLY_VT(svd_only_vt_f64, ffi::DataType::F64);
DEFINE_COMPLEX_SVD_ONLY_VT(svd_only_vt_c64, ffi::DataType::C64);
DEFINE_COMPLEX_SVD_ONLY_VT(svd_only_vt_c128, ffi::DataType::C128);

#define DEFINE_REAL_SVD_ONLY_VT_QR(fname, dtype)   \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                   \
      fname, SvdOnlyVtQRImpl<dtype>,               \
      ffi::Ffi::Bind()                             \
          .Arg<ffi::Buffer<dtype>>(/*x*/)          \
          .Ret<ffi::Buffer<dtype>>(/*x_out*/)      \
          .Ret<ffi::Buffer<dtype>>(/*s*/)          \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define DEFINE_COMPLEX_SVD_ONLY_VT_QR(fname, dtype)    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                       \
      fname, SvdOnlyVtQRImpl<dtype>,                   \
      ffi::Ffi::Bind()                                 \
          .Arg<ffi::Buffer<dtype>>(/*x*/)              \
          .Ret<ffi::Buffer<dtype>>(/*x_out*/)          \
          .Ret<ffi::Buffer<ffi::ToReal(dtype)>>(/*s*/) \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

DEFINE_REAL_SVD_ONLY_VT_QR(svd_only_vt_qr_f32, ffi::DataType::F32);
DEFINE_REAL_SVD_ONLY_VT_QR(svd_only_vt_qr_f64, ffi::DataType::F64);
DEFINE_COMPLEX_SVD_ONLY_VT_QR(svd_only_vt_qr_c64, ffi::DataType::C64);
DEFINE_COMPLEX_SVD_ONLY_VT_QR(svd_only_vt_qr_c128, ffi::DataType::C128);

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_svd_only_vt, m) {
  m.def("svd_only_vt_f32", []() { return EncapsulateFfiCall(svd_only_vt_f32); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_f64", []() { return EncapsulateFfiCall(svd_only_vt_f64); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_c64", []() { return EncapsulateFfiCall(svd_only_vt_c64); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_c128", []() { return EncapsulateFfiCall(svd_only_vt_c128); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_qr_f32", []() { return EncapsulateFfiCall(svd_only_vt_qr_f32); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_qr_f64", []() { return EncapsulateFfiCall(svd_only_vt_qr_f64); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_qr_c64", []() { return EncapsulateFfiCall(svd_only_vt_qr_c64); }, nb::call_guard<nb::gil_scoped_release>());
  m.def("svd_only_vt_qr_c128", []() { return EncapsulateFfiCall(svd_only_vt_qr_c128); }, nb::call_guard<nb::gil_scoped_release>());
}
