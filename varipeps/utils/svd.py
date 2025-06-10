from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import scan
from jax.lax.linalg import svd as lax_svd
from jax import jit, custom_jvp, lax

from jax._src.numpy.util import promote_dtypes_inexact, check_arraylike

from varipeps import varipeps_config

from .extensions import _svd_only_vt as _svd_only_vt_lib

from typing import Tuple


def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


@custom_jvp
def svd_wrapper(a):
    check_arraylike("jnp.linalg.svd", a)
    (a,) = promote_dtypes_inexact(jnp.asarray(a))

    result = lax_svd(a, full_matrices=False, compute_uv=True)

    result = lax.cond(
        jnp.isnan(jnp.sum(result[1])),
        lambda matrix, _: lax_svd(
            matrix,
            full_matrices=False,
            compute_uv=True,
            algorithm=lax.linalg.SvdAlgorithm.QR,
        ),
        lambda _, res: res,
        a,
        result,
    )

    return result


@svd_wrapper.defjvp
def _svd_jvp_rule(primals, tangents):
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = svd_wrapper(A)

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    # s_diffs = jnp.where(s_diffs / (s[0] ** 2) >= 1e-12, s_diffs, 0)
    s_diffs_zeros = jnp.ones((), dtype=A.dtype) * (
        s_diffs == 0.0
    )  # is 1. where s_diffs is 0. and is 0. everywhere else
    s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
    SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = jnp.vectorize(jnp.diag, signature="(k)->(k,k)")(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
    dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        dAV = dA @ V
        dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
    if n > m:
        dAHU = _H(dA) @ U
        dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)

    return (U, s, Vt), (dU, ds, _H(dV))


@partial(jit, inline=True)
def gauge_fixed_svd(
    matrix: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate the gauge-fixed (also called sign-fixed) SVD. To this end, each
    singular vector are rotate in the way that the first element bigger than
    some numerical stability threshold (config parameter eps) is ensured to be
    along the positive real axis.

    Args:
      matrix (:obj:`jnp.ndarray`):
        Matrix to calculate SVD for.
    Returns:
      :obj:`tuple`\\ (:obj:`jnp.ndarray`, :obj:`jnp.ndarray`, :obj:`jnp.ndarray`):
        Tuple with sign-fixed U, S and Vh of the SVD.
    """
    U, S, Vh = svd_wrapper(matrix)

    # Fix the gauge of the SVD
    abs_U = jnp.abs(U)
    max_per_vector = jnp.max(abs_U, axis=0)
    normalized_U = abs_U / max_per_vector[jnp.newaxis, :]

    def phase_f(carry, x):
        U_row, normalized_U_row = x

        already_found, last_step_result = carry

        cond = normalized_U_row >= varipeps_config.svd_sign_fix_eps

        result = jnp.where(
            already_found, last_step_result, jnp.where(cond, U_row, last_step_result)
        )

        return (jnp.logical_or(already_found, cond), result), None

    phases, _ = scan(
        phase_f,
        (jnp.zeros(U.shape[1], dtype=bool), U[0, :]),
        (U, normalized_U),
    )
    phases = phases[1]
    phases /= jnp.abs(phases)

    U = U * phases.conj()[jnp.newaxis, :]
    Vh = Vh * phases[:, jnp.newaxis]

    return U, S, Vh


jax.ffi.register_ffi_target(
    "svd_only_vt_f32", _svd_only_vt_lib.svd_only_vt_f32(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_f64", _svd_only_vt_lib.svd_only_vt_f64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_c64", _svd_only_vt_lib.svd_only_vt_c64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_c128", _svd_only_vt_lib.svd_only_vt_c128(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_qr_f32", _svd_only_vt_lib.svd_only_vt_qr_f32(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_qr_f64", _svd_only_vt_lib.svd_only_vt_qr_f64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_qr_c64", _svd_only_vt_lib.svd_only_vt_qr_c64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_vt_qr_c128", _svd_only_vt_lib.svd_only_vt_qr_c128(), platform="cpu"
)


def svd_only_vt(a, use_qr=False):
    suffix = "_qr" if use_qr else ""

    if a.dtype == jnp.float32:
        fn = f"svd_only_vt{suffix}_f32"
        real_dtype = jnp.float32
    elif a.dtype == jnp.float64:
        fn = f"svd_only_vt{suffix}_f64"
        real_dtype = jnp.float64
    elif a.dtype == jnp.complex64:
        fn = f"svd_only_vt{suffix}_c64"
        real_dtype = jnp.float32
    elif a.dtype == jnp.complex128:
        fn = f"svd_only_vt{suffix}_c128"
        real_dtype = jnp.float64
    else:
        raise ValueError("Unsupported dtype")

    m, n = a.shape

    if m < n:
        raise ValueError("Only arrays with M >= N supported.")

    if use_qr:
        call = jax.ffi.ffi_call(
            fn,
            (
                jax.ShapeDtypeStruct((m, n), a.dtype),
                jax.ShapeDtypeStruct((n,), real_dtype),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            vmap_method="sequential",
            input_layouts=((1, 0),),
            output_layouts=((1, 0), None, None),
            input_output_aliases={0: 0},
        )

        aout, S, info = call(a)

        Vt = aout[:n, :]
    else:
        call = jax.ffi.ffi_call(
            fn,
            (
                jax.ShapeDtypeStruct((m, n), a.dtype),
                jax.ShapeDtypeStruct((n,), real_dtype),
                jax.ShapeDtypeStruct((n, n), a.dtype),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            vmap_method="sequential",
            input_layouts=((1, 0),),
            output_layouts=((1, 0), None, (1, 0), None),
            input_output_aliases={0: 0},
        )

        aout, S, Vt, info = call(a)

    S, Vt = jax.lax.cond(
        info[0] != 0,
        lambda s, vt: (s * jnp.nan, vt * jnp.nan),
        lambda s, vt: (s, vt),
        S,
        Vt,
    )

    return S, Vt
