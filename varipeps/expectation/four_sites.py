from functools import partial

import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction_jitted

from typing import Sequence, List, Tuple, Literal

Corner_Literal = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


@partial(jit, static_argnums=(5,))
def _four_sites_quadrat_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    real_result: bool = False,
) -> List[jnp.ndarray]:
    density_matrix = jnp.tensordot(top_left, top_right, ((5, 6, 7), (2, 3, 4)))
    density_matrix = jnp.tensordot(density_matrix, bottom_left, ((2, 3, 4), (5, 6, 7)))
    density_matrix = jnp.tensordot(
        density_matrix, bottom_right, ((4, 5, 6, 9, 10, 11), (2, 3, 4, 5, 6, 7))
    )

    density_matrix = density_matrix.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    density_matrix = density_matrix.reshape(
        density_matrix.shape[0]
        * density_matrix.shape[1]
        * density_matrix.shape[2]
        * density_matrix.shape[3],
        density_matrix.shape[4]
        * density_matrix.shape[5]
        * density_matrix.shape[6]
        * density_matrix.shape[7],
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


def calc_four_sites_quadrat_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the four site expectation values for three as quadrat ordered
    PEPS tensor and their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, top-right, bottom-left, bottom-right].

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates which should be applied to the PEPS tensors.
        Gates are expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        List with the calculated expectation values of each gate.
    """
    density_matrix_top_left = apply_contraction_jitted(
        "density_matrix_four_sites_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction_jitted(
        "density_matrix_four_sites_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction_jitted(
        "density_matrix_four_sites_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _four_sites_quadrat_workhorse(
        density_matrix_top_left,
        density_matrix_top_right,
        density_matrix_bottom_left,
        density_matrix_bottom_right,
        tuple(gates),
        real_result,
    )
