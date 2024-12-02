from functools import partial

import jax.numpy as jnp
from jax.lax import scan
from jax.lax.linalg import svd as lax_svd
from jax import jit, custom_jvp, lax

from jax._src.numpy.util import promote_dtypes_inexact, check_arraylike

from varipeps import varipeps_config

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
