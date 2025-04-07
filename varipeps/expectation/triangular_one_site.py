from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from varipeps.contractions import apply_contraction_jitted


@partial(jit, static_argnums=(3,))
def _one_site_workhorse(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix = apply_contraction_jitted(
        "triangular_ctmrg_one_site_expectation",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    norm = jnp.trace(density_matrix)

    if real_result:
        return [
            jnp.real(jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm)
            for g in gates
        ]
    else:
        return [
            jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm for g in gates
        ]


def calc_triangular_one_site(
    peps_tensors,
    peps_tensor_objs,
    gates,
):
    if isinstance(peps_tensors, jnp.ndarray):
        peps_tensors = (peps_tensors,)
        peps_tensor_objs = (peps_tensor_objs,)

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _one_site_workhorse(
        peps_tensors, peps_tensor_objs, tuple(gates), real_result
    )
