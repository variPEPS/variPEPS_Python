from functools import partial

import jax.numpy as jnp
from jax.lax import scan
from jax import jit, custom_jvp, lax

from peps_ad import peps_ad_config

from typing import Tuple


def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


@custom_jvp
def svd_wrapper(a):
    return jnp.linalg.svd(a, full_matrices=False, compute_uv=True)


@svd_wrapper.defjvp
def _svd_jvp_rule(primals, tangents):
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False, compute_uv=True)

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = jnp.matmul(jnp.matmul(Ut, dA), V)
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = jnp.ones((), dtype=A.dtype) * (
        s_diffs == 0.0
    )  # is 1. where s_diffs is 0. and is 0. everywhere else
    s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim * dS  # dS.dot(jnp.diag(s))
    SdS = _T(s_dim) * dS  # jnp.diag(s).dot(dS)

    s_zeros = jnp.ones((), dtype=A.dtype) * (s == 0.0)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = jnp.vectorize(jnp.diag, signature="(k)->(k,k)")(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat
    dU = jnp.matmul(U, F * (dSS + _H(dSS)) + dUdV_diag)
    dV = jnp.matmul(V, F * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        I = lax.expand_dims(jnp.eye(m, dtype=A.dtype), range(U.ndim - 2))
        dU = dU + jnp.matmul(I - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim
    if n > m:
        I = lax.expand_dims(jnp.eye(n, dtype=A.dtype), range(V.ndim - 2))
        dV = dV + jnp.matmul(I - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim

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
        U_row = x[0, :]
        normalized_U_row = x[1, :]

        already_found = carry[0]
        last_step_result = carry[1]

        cond = normalized_U_row >= peps_ad_config.svd_sign_fix_eps

        result = jnp.where(
            already_found, last_step_result, jnp.where(cond, U_row, last_step_result)
        )

        return (jnp.logical_or(already_found, cond), result), None

    phases, _ = scan(
        phase_f,
        (jnp.zeros(U.shape[0], dtype=bool), U[0, :]),
        jnp.stack((U, normalized_U), axis=1),
    )
    phases = phases[1]
    phases /= jnp.abs(phases)

    U = U * phases.conj()[jnp.newaxis, :]
    Vh = Vh * phases[:, jnp.newaxis]

    return U, S, Vh
