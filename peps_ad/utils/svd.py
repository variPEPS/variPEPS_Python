from functools import partial

import jax.numpy as jnp
from jax.lax import scan
from jax import jit

from peps_ad import peps_ad_config

from typing import Tuple


@partial(jit, inline=True, static_argnums=(1,))
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
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)

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
