from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax.linalg import svd as lax_svd
from jax import jit, custom_jvp, lax

from jax._src.numpy.util import promote_dtypes_inexact, check_arraylike

from varipeps import varipeps_config

from .extensions import _svd_only_u_vt as _svd_only_u_vt_lib

from typing import Tuple


def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


def _is_gpu_backend() -> bool:
    return any(d.platform == "gpu" for d in jax.devices())


def _svd_algorithm():
    """
    SVD algorithm selected by :obj:`~varipeps.config.VariPEPS_Config.svd_algorithm`.
    ``'default'`` means the QR algorithm if a GPU is used and JAX's default
    (divide and conquer) otherwise.
    """
    if varipeps_config.svd_algorithm == "default" and _is_gpu_backend():
        return lax.linalg.SvdAlgorithm.QR
    return lax.linalg.SvdAlgorithm[varipeps_config.svd_algorithm.upper()]


def _use_svd_nan_fallback() -> bool:
    """
    Check if the NaN check with the QR-based SVD as fallback should be staged
    out after each SVD. On GPU the :obj:`jax.lax.cond` of the check requires a
    device-to-host synchronization for every SVD and doubles the number of
    compiled SVD kernels, so it is disabled there by default. See
    :obj:`~varipeps.config.VariPEPS_Config.svd_gpu_nan_fallback`.
    """
    return _svd_algorithm() is not lax.linalg.SvdAlgorithm.QR and (
        not _is_gpu_backend() or varipeps_config.svd_gpu_nan_fallback
    )


@partial(custom_jvp, nondiff_argnums=(1,))
def svd_wrapper(a, use_qr=False):
    check_arraylike("jnp.linalg.svd", a)
    (a,) = promote_dtypes_inexact(jnp.asarray(a))

    if use_qr:
        result = lax_svd(
            a,
            full_matrices=False,
            compute_uv=True,
            algorithm=lax.linalg.SvdAlgorithm.QR,
        )
    else:
        result = lax_svd(
            a,
            full_matrices=False,
            compute_uv=True,
            algorithm=_svd_algorithm(),
        )

        if _use_svd_nan_fallback():
            result = _svd_nan_fallback(a, result)

    return result


@jax.custom_batching.custom_vmap
def _svd_nan_fallback(a, result):
    """
    Recalculate the SVD of ``a`` with the QR-based algorithm if ``result``
    contains NaNs. Under :obj:`jax.vmap` the check is done once for the whole
    batch (see :obj:`_svd_nan_fallback_vmap`), so it stays a real
    :obj:`jax.lax.cond` and is not converted into a select which would
    calculate the QR-based SVD always.
    """
    return lax.cond(
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


@_svd_nan_fallback.def_vmap
def _svd_nan_fallback_vmap(axis_size, in_batched, a, result):
    a_batched, result_batched = in_batched

    if not a_batched:
        a = jnp.broadcast_to(a, (axis_size,) + a.shape)
    result = tuple(
        e if e_batched else jnp.broadcast_to(e, (axis_size,) + e.shape)
        for e, e_batched in zip(result, result_batched, strict=True)
    )

    # The unbatched rule works on stacked arrays as well: if any SVD of the
    # batch failed, the whole batch is recalculated with the QR-based SVD.
    return _svd_nan_fallback.fun(a, result), (True, True, True)


def _svd_jvp_rule_impl(primals, tangents, only_u_or_vt=None, use_qr=False):
    (A,) = primals
    (dA,) = tangents

    if use_qr and only_u_or_vt is not None:
        U, s, Vt = _svd_only_u_vt_impl(A, u_or_vt=2, use_qr=True)
    else:
        U, s, Vt = svd_wrapper(A, use_qr=use_qr)

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    s_sums = s_dim + _T(s_dim)
    s_sums = jnp.where(s_sums > 0, s_sums, 1)
    s_diffs = s_dim - _T(s_dim)

    if varipeps_config.svd_ad_use_lorentz_broadening:
        F = s_diffs / (s_diffs**2 + varipeps_config.svd_ad_lorentz_broadening_eps)
    else:
        s_diffs = jnp.where(jnp.abs(s_diffs / s[0]) >= 1e-12, s_diffs, 0)
        s_diffs_zeros = jnp.ones((), dtype=A.dtype) * (
            s_diffs == 0.0
        )  # is 1. where s_diffs is 0. and is 0. everywhere else
        s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
        F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros

    if only_u_or_vt is None or only_u_or_vt == "U":
        dSS = dS * (s_dim / s_sums).astype(A.dtype)  # dS.dot(s_j / (s_i + s_j))
    if only_u_or_vt is None or only_u_or_vt == "Vt":
        SdS = (_T(s_dim) / s_sums).astype(A.dtype) * dS  # (s_i / (s_i + s_j)).dot(dS)

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = jnp.vectorize(jnp.diag, signature="(k)->(k,k)")(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)

    if only_u_or_vt is None:
        # negative sign in dV since sum(dU + dV^\dagger) = dUdV_diag
        dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + 0.5 * dUdV_diag)
        dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)) - 0.5 * dUdV_diag)
    elif only_u_or_vt == "U":
        dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    elif only_u_or_vt == "Vt":
        dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)) - dUdV_diag)

    m, n = A.shape[-2:]
    if m > n and (only_u_or_vt is None or only_u_or_vt == "U"):
        dAV = dA @ V
        dU = dU + (dAV - U @ (Ut @ dAV)) * s_inv.astype(A.dtype)
    if n > m and (only_u_or_vt is None or only_u_or_vt == "Vt"):
        dAHU = _H(dA) @ U
        dV = dV + (dAHU - V @ (Vt @ dAHU)) * s_inv.astype(A.dtype)

    if only_u_or_vt is None:
        return (U, s, Vt), (dU, ds, _H(dV))
    elif only_u_or_vt == "U":
        return (U, s), (dU, ds)
    elif only_u_or_vt == "Vt":
        return (s, Vt), (ds, _H(dV))


@svd_wrapper.defjvp
def _svd_jvp_rule(use_qr, primals, tangents):
    return _svd_jvp_rule_impl(primals, tangents, use_qr=use_qr)


jax.ffi.register_ffi_target(
    "svd_only_u_vt_f32", _svd_only_u_vt_lib.svd_only_u_vt_f32(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_f64", _svd_only_u_vt_lib.svd_only_u_vt_f64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_c64", _svd_only_u_vt_lib.svd_only_u_vt_c64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_c128", _svd_only_u_vt_lib.svd_only_u_vt_c128(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_qr_f32", _svd_only_u_vt_lib.svd_only_u_vt_qr_f32(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_qr_f64", _svd_only_u_vt_lib.svd_only_u_vt_qr_f64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_qr_c64", _svd_only_u_vt_lib.svd_only_u_vt_qr_c64(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "svd_only_u_vt_qr_c128", _svd_only_u_vt_lib.svd_only_u_vt_qr_c128(), platform="cpu"
)


def _svd_only_u_vt_impl(a, u_or_vt, use_qr=True):
    suffix = "_qr" if use_qr else ""

    if a.dtype == jnp.float32:
        fn = f"svd_only_u_vt{suffix}_f32"
        real_dtype = jnp.float32
    elif a.dtype == jnp.float64:
        fn = f"svd_only_u_vt{suffix}_f64"
        real_dtype = jnp.float64
    elif a.dtype == jnp.complex64:
        fn = f"svd_only_u_vt{suffix}_c64"
        real_dtype = jnp.float32
    elif a.dtype == jnp.complex128:
        fn = f"svd_only_u_vt{suffix}_c128"
        real_dtype = jnp.float64
    else:
        raise ValueError("Unsupported dtype")

    m, n = a.shape

    return_only = None
    if m > n and u_or_vt == 0:
        u_or_vt = 2
        return_only = "U"
    elif m < n and u_or_vt == 1:
        u_or_vt = 2
        return_only = "Vt"

    min_dim = min(m, n)

    if use_qr:
        if u_or_vt == 2:
            u_vt_buffer_shape = jax.ShapeDtypeStruct((min_dim, min_dim), a.dtype)
        else:
            u_vt_buffer_shape = jax.ShapeDtypeStruct((0, 0), a.dtype)

        call = jax.ffi.ffi_call(
            fn,
            (
                jax.ShapeDtypeStruct((m, n), a.dtype),
                jax.ShapeDtypeStruct((min_dim,), real_dtype),
                u_vt_buffer_shape,
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            vmap_method="sequential",
            input_layouts=((1, 0),),
            output_layouts=((1, 0), None, (1, 0), None),
            input_output_aliases={0: 0},
        )

        aout, S, u_vt_buffer, info = call(a, mode=np.int8(u_or_vt))

        if u_or_vt == 2:
            if m >= n:
                U = aout
                Vt = u_vt_buffer
            else:
                U = u_vt_buffer
                Vt = aout
        else:
            result = aout[:min_dim, :min_dim]
    else:
        call = jax.ffi.ffi_call(
            fn,
            (
                jax.ShapeDtypeStruct((m, n), a.dtype),
                jax.ShapeDtypeStruct((min_dim,), real_dtype),
                jax.ShapeDtypeStruct((min_dim, min_dim), a.dtype),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            vmap_method="sequential",
            input_layouts=((1, 0),),
            output_layouts=((1, 0), None, (1, 0), None),
            input_output_aliases={0: 0},
        )

        aout, S, u_vt_buffer, info = call(a, mode=np.int8(u_or_vt))

        if u_or_vt == 0:
            if m == n:
                result = aout
            else:
                result = u_vt_buffer
        elif u_or_vt == 1:
            result = u_vt_buffer
        elif u_or_vt == 2:
            if m >= n:
                U = aout
                Vt = u_vt_buffer
            else:
                U = u_vt_buffer
                Vt = aout

    if u_or_vt == 2:
        U, S, Vt = jax.lax.cond(
            info[0] != 0,
            lambda u, s, r: (u * jnp.nan, s * jnp.nan, r * jnp.nan),
            lambda u, s, r: (u, s, r),
            U,
            S,
            Vt,
        )

        if return_only == "U":
            return S, U
        elif return_only == "Vt":
            return S, Vt

        return U, S, Vt

    S, result = jax.lax.cond(
        info[0] != 0,
        lambda s, r: (s * jnp.nan, r * jnp.nan),
        lambda s, r: (s, r),
        S,
        result,
    )

    return S, result


@partial(custom_jvp, nondiff_argnums=(1,))
def svd_only_u(a, use_qr=True):
    S, U = _svd_only_u_vt_impl(a, 0, use_qr)

    if not use_qr:
        S, U = lax.cond(
            jnp.isnan(jnp.sum(S)),
            lambda matrix, s, u: _svd_only_u_vt_impl(matrix, 0, True),
            lambda matrix, s, u: (s, u),
            a,
            S,
            U,
        )

    return U, S


@svd_only_u.defjvp
def _svd_only_u_jvp_rule(use_qr, primals, tangents):
    return _svd_jvp_rule_impl(primals, tangents, only_u_or_vt="U", use_qr=use_qr)


@partial(custom_jvp, nondiff_argnums=(1,))
def svd_only_vt(a, use_qr=True):
    S, Vt = _svd_only_u_vt_impl(a, 1, use_qr)

    if not use_qr:
        S, Vt = lax.cond(
            jnp.isnan(jnp.sum(S)),
            lambda matrix, s, v: _svd_only_u_vt_impl(matrix, 1, True),
            lambda matrix, s, v: (s, v),
            a,
            S,
            Vt,
        )

    return S, Vt


@svd_only_vt.defjvp
def _svd_only_vt_jvp_rule(use_qr, primals, tangents):
    return _svd_jvp_rule_impl(primals, tangents, only_u_or_vt="Vt", use_qr=use_qr)


@partial(jit, inline=True, static_argnums=(1, 2))
def gauge_fixed_svd(
    matrix: jnp.ndarray,
    only_u_or_vh=None,
    use_qr=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate the gauge-fixed (also called sign-fixed) SVD. To this end, each
    singular vector are rotate in the way that the first element bigger than
    some numerical stability threshold (config parameter eps) is ensured to be
    along the positive real axis.

    Args:
      matrix (:obj:`jnp.ndarray`):
        Matrix to calculate SVD for.
    Keyword args:
      only_u_or_vh (:obj:`str`):
        Flag if only U or Uh should be calculated. If `None` (default), calculate
        the full SVD, if `'U'` only calculate U, if `'Vh'` only calculate Vh.
    Returns:
      :obj:`tuple`\\ (:obj:`jnp.ndarray`, :obj:`jnp.ndarray`, :obj:`jnp.ndarray`):
        Tuple with sign-fixed U, S and Vh of the SVD.
    """
    if _is_gpu_backend():
        U, S, Vh = svd_wrapper(matrix, use_qr=use_qr)
        if only_u_or_vh is None:
            gauge_unitary = U
        elif only_u_or_vh == "U":
            gauge_unitary = U
        elif only_u_or_vh == "Vh":
            gauge_unitary = Vh.T.conj()
        else:
            raise ValueError("Invalid value for parameter 'only_u_or_vh'.")
    else:
        if only_u_or_vh is None:
            U, S, Vh = svd_wrapper(matrix, use_qr=use_qr)
            gauge_unitary = U
        elif only_u_or_vh == "U":
            U, S = svd_only_u(matrix, use_qr=use_qr)
            gauge_unitary = U
        elif only_u_or_vh == "Vh":
            S, Vh = svd_only_vt(matrix, use_qr=use_qr)
            gauge_unitary = Vh.T.conj()
        else:
            raise ValueError("Invalid value for parameter 'only_u_or_vh'.")

    # Fix the gauge of the SVD
    abs_gauge_unitary = jnp.abs(gauge_unitary)
    max_per_vector = jnp.max(abs_gauge_unitary, axis=0)
    normalized_gauge_unitary = abs_gauge_unitary / max_per_vector[jnp.newaxis, :]

    # Select for each singular vector the first element above the threshold.
    phases_index = jnp.argmax(
        normalized_gauge_unitary >= varipeps_config.svd_sign_fix_eps, axis=0
    )
    phases = jnp.take_along_axis(gauge_unitary, phases_index[jnp.newaxis, :], axis=0)[0]
    phases /= jnp.abs(phases)

    if only_u_or_vh is None or only_u_or_vh == "U":
        U = U * phases.conj()[jnp.newaxis, :]
    if only_u_or_vh is None or only_u_or_vh == "Vh":
        Vh = Vh * phases[:, jnp.newaxis]

    if only_u_or_vh == "U":
        return U, S

    if only_u_or_vh == "Vh":
        return S, Vh

    return U, S, Vh
