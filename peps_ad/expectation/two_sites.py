import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor
from peps_ad.contractions import apply_contraction

from typing import Sequence, List, Tuple


def _two_site_workhorse_body(
    density_matrix_1: jnp.ndarray,
    density_matrix_2: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    real_result: bool = False,
) -> List[jnp.ndarray]:
    density_matrix = jnp.tensordot(
        density_matrix_1, density_matrix_2, ((2, 3, 4, 5), (0, 1, 2, 3))
    )

    density_matrix = density_matrix.transpose((0, 2, 1, 3))

    density_matrix = density_matrix.reshape(
        density_matrix.shape[0] * density_matrix.shape[1],
        density_matrix.shape[2] * density_matrix.shape[3],
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


_two_site_workhorse = jit(_two_site_workhorse_body, static_argnums=(3,))


def calc_two_sites_horizontal_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two horizontal ordered PEPS
    tensor and their environment.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates which should be applied to the PEPS tensors.
        Gates are expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        List with the calculated expectation values of each gate.
    """
    density_matrix_1 = apply_contraction(
        "density_matrix_two_sites_left", [peps_tensors[0]], [peps_tensor_objs[0]], []
    )

    density_matrix_2 = apply_contraction(
        "density_matrix_two_sites_right", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    real_result = all(jnp.allclose(g, jnp.real(g)) for g in gates)

    return _two_site_workhorse(
        density_matrix_1, density_matrix_2, tuple(gates), real_result
    )


def calc_two_sites_horizontal_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for two horizontal ordered PEPS
    tensor and their environment.

    This function just wraps :obj:`~peps_ad.expectation.two_sites.calc_two_sites_horizontal_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_two_sites_horizontal_multiple_gates(
        peps_tensors, peps_tensor_objs, [gate]
    )[0]


def calc_two_sites_vertical_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two vertical ordered PEPS
    tensor and their environment.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates which should be applied to the PEPS tensors.
        Gates are expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        List with the calculated expectation values of each gate.
    """
    density_matrix_1 = apply_contraction(
        "density_matrix_two_sites_top", [peps_tensors[0]], [peps_tensor_objs[0]], []
    )

    density_matrix_2 = apply_contraction(
        "density_matrix_two_sites_buttom", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    real_result = all(jnp.allclose(g, jnp.real(g)) for g in gates)

    return _two_site_workhorse(
        density_matrix_1, density_matrix_2, tuple(gates), real_result
    )


def calc_two_sites_vertical_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for two vertical ordered PEPS
    tensor and their environment.

    This function just wraps :obj:`~peps_ad.expectation.two_sites.calc_two_sites_vertical_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_two_sites_vertical_multiple_gates(
        peps_tensors, peps_tensor_objs, [gate]
    )[0]
