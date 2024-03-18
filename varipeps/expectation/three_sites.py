from functools import partial

import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction

from typing import Sequence, List, Tuple, Literal

Corner_Literal = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


@partial(jit, static_argnums=(5, 6))
def _three_site_triangle_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    traced: Corner_Literal,
    real_result: bool = False,
) -> List[jnp.ndarray]:
    if traced == "top-left":
        one_physical_sites = jnp.tensordot(top_left, top_right, ((3, 4, 5), (2, 3, 4)))
        two_physical_sites = jnp.tensordot(
            bottom_right, bottom_left, ((5, 6, 7), (2, 3, 4))
        )
        transpose_order = (0, 4, 2, 1, 5, 3)
    elif traced == "top-right":
        one_physical_sites = jnp.tensordot(
            top_right, bottom_right, ((3, 4, 5), (2, 3, 4))
        )
        two_physical_sites = jnp.tensordot(
            bottom_left, top_left, ((5, 6, 7), (2, 3, 4))
        )
        transpose_order = (4, 2, 0, 5, 3, 1)
    elif traced == "bottom-left":
        one_physical_sites = jnp.tensordot(
            bottom_left, top_left, ((3, 4, 5), (2, 3, 4))
        )
        two_physical_sites = jnp.tensordot(
            top_right, bottom_right, ((5, 6, 7), (2, 3, 4))
        )
        transpose_order = (0, 2, 4, 1, 3, 5)
    elif traced == "bottom-right":
        one_physical_sites = jnp.tensordot(
            bottom_right, bottom_left, ((3, 4, 5), (2, 3, 4))
        )
        two_physical_sites = jnp.tensordot(top_left, top_right, ((5, 6, 7), (2, 3, 4)))
        transpose_order = (2, 4, 0, 3, 5, 1)
    else:
        raise ValueError('Invalid argument for parameter "traced".')

    density_matrix = jnp.tensordot(
        one_physical_sites, two_physical_sites, ((0, 1, 2, 5, 6, 7), (7, 8, 9, 2, 3, 4))
    )

    density_matrix = density_matrix.transpose(transpose_order)
    density_matrix = density_matrix.reshape(
        density_matrix.shape[0] * density_matrix.shape[1] * density_matrix.shape[2],
        density_matrix.shape[3] * density_matrix.shape[4] * density_matrix.shape[5],
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


def calc_three_sites_triangle_without_top_left_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the three site expectation values for three as triangle ordered
    PEPS tensor and their environment without the top-left tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-right, bottom-left, bottom-right].

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
    traced_density_matrix_top_left = apply_contraction(
        "ctmrg_top_left", [peps_tensors[0]], [peps_tensor_objs[0]], []
    )

    density_matrix_top_right = apply_contraction(
        "density_matrix_four_sites_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction(
        "density_matrix_four_sites_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _three_site_triangle_workhorse(
        traced_density_matrix_top_left,
        density_matrix_top_right,
        density_matrix_bottom_left,
        density_matrix_bottom_right,
        tuple(gates),
        "top-left",
        real_result,
    )


def calc_three_sites_triangle_without_top_left_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for three as triangle ordered
    PEPS tensor and their environment without the top-left tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-right, bottom-left, bottom-right].

    This function just wraps
    :obj:`~varipeps.expectation.three_sites.calc_three_sites_triangle_without_top_left_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_three_sites_triangle_without_top_left_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]


def calc_three_sites_triangle_without_top_right_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the three site expectation values for three as triangle ordered
    PEPS tensor and their environment without the top-right tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, bottom-left, bottom-right].

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
    density_matrix_top_left = apply_contraction(
        "density_matrix_four_sites_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    traced_density_matrix_top_right = apply_contraction(
        "ctmrg_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction(
        "density_matrix_four_sites_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _three_site_triangle_workhorse(
        density_matrix_top_left,
        traced_density_matrix_top_right,
        density_matrix_bottom_left,
        density_matrix_bottom_right,
        tuple(gates),
        "top-right",
        real_result,
    )


def calc_three_sites_triangle_without_top_right_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for three as triangle ordered
    PEPS tensor and their environment without the top-right tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, bottom-left, bottom-right].

    This function just wraps
    :obj:`~varipeps.expectation.three_sites.calc_three_sites_triangle_without_top_right_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_three_sites_triangle_without_top_right_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]


def calc_three_sites_triangle_without_bottom_left_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the three site expectation values for three as triangle ordered
    PEPS tensor and their environment without the bottom-left tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, top-right, bottom-right].

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
    density_matrix_top_left = apply_contraction(
        "density_matrix_four_sites_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction(
        "density_matrix_four_sites_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    traced_density_matrix_bottom_left = apply_contraction(
        "ctmrg_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _three_site_triangle_workhorse(
        density_matrix_top_left,
        density_matrix_top_right,
        traced_density_matrix_bottom_left,
        density_matrix_bottom_right,
        tuple(gates),
        "bottom-left",
        real_result,
    )


def calc_three_sites_triangle_without_bottom_left_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for three as triangle ordered
    PEPS tensor and their environment without the bottom-left tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, top-right, bottom-right].

    This function just wraps
    :obj:`~varipeps.expectation.three_sites.calc_three_sites_triangle_without_bottom_left_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_three_sites_triangle_without_bottom_left_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]


def calc_three_sites_triangle_without_bottom_right_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the three site expectation values for three as triangle ordered
    PEPS tensor and their environment without the bottom-right tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-left].

    The gate is applied in the order [top-left, top-right, bottom-right].

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
    density_matrix_top_left = apply_contraction(
        "density_matrix_four_sites_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction(
        "density_matrix_four_sites_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction(
        "density_matrix_four_sites_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    traced_density_matrix_bottom_right = apply_contraction(
        "ctmrg_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _three_site_triangle_workhorse(
        density_matrix_top_left,
        density_matrix_top_right,
        density_matrix_bottom_left,
        traced_density_matrix_bottom_right,
        tuple(gates),
        "bottom-right",
        real_result,
    )


def calc_three_sites_triangle_without_bottom_right_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for three as triangle ordered
    PEPS tensor and their environment without the bottom-left tensor.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    The gate is applied in the order [top-left, top-right, bottom-left].

    This function just wraps
    :obj:`~varipeps.expectation.three_sites.calc_three_sites_triangle_without_bottom_right_multiple_gates`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays. Have to be the same objects as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        PEPS tensor objects.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensors.
        The gate is expected to be a matrix with first axis corresponding to
        the Hilbert space and the second axis corresponding to the dual room.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_three_sites_triangle_without_bottom_right_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]
