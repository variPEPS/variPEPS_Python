from dataclasses import dataclass
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit

from varipeps import varipeps_config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from .model import Expectation_Model
from .spiral_helpers import apply_unitary

from varipeps.utils.debug_print import debug_print

from typing import Sequence, List, Tuple, Union, Optional


@partial(jit, static_argnums=(3,))
def _two_site_workhorse(
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


@partial(jit, static_argnums=(5,))
def _two_site_diagonal_workhorse(
    density_matrix_1: jnp.ndarray,
    density_matrix_2: jnp.ndarray,
    traced_density_matrix_1: jnp.ndarray,
    traced_density_matrix_2: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    real_result: bool = False,
) -> List[jnp.ndarray]:
    tmp_tensor_1 = jnp.tensordot(
        density_matrix_1, traced_density_matrix_1, ((5, 6, 7), (0, 1, 2))
    )
    tmp_tensor_2 = jnp.tensordot(
        density_matrix_2, traced_density_matrix_2, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        tmp_tensor_1, tmp_tensor_2, ((2, 3, 4, 5, 6, 7), (5, 6, 7, 2, 3, 4))
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


@partial(jit, static_argnums=(2,))
def _two_site_full_density_workhorse(
    density_matrix: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    real_result: bool = False,
) -> List[jnp.ndarray]:
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
    density_matrix_1 = apply_contraction(
        "density_matrix_two_sites_left", [peps_tensors[0]], [peps_tensor_objs[0]], []
    )

    density_matrix_2 = apply_contraction(
        "density_matrix_two_sites_right", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

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

    This function just wraps :obj:`~varipeps.expectation.two_sites.calc_two_sites_horizontal_multiple_gates`.

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
    density_matrix_1 = apply_contraction(
        "density_matrix_two_sites_top", [peps_tensors[0]], [peps_tensor_objs[0]], []
    )

    density_matrix_2 = apply_contraction(
        "density_matrix_two_sites_bottom", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

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

    This function just wraps :obj:`~varipeps.expectation.two_sites.calc_two_sites_vertical_multiple_gates`.

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
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        Calculated expectation value of the gate.
    """
    return calc_two_sites_vertical_multiple_gates(
        peps_tensors, peps_tensor_objs, [gate]
    )[0]


def calc_two_sites_diagonal_top_left_bottom_right_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two from top left to bottom
    right diagonal ordered PEPS tensor and their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

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
        "ctmrg_top_right", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    traced_density_matrix_bottom_left = apply_contraction(
        "ctmrg_bottom_left", [peps_tensors[2]], [peps_tensor_objs[2]], []
    )

    density_matrix_bottom_right = apply_contraction(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _two_site_diagonal_workhorse(
        density_matrix_top_left,
        density_matrix_bottom_right,
        traced_density_matrix_top_right,
        traced_density_matrix_bottom_left,
        tuple(gates),
        real_result,
    )


def calc_two_sites_diagonal_top_left_bottom_right_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for two from top left to bottom
    right diagonal ordered PEPS tensor and their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    This function just wraps
    :obj:`~varipeps.expectation.two_sites.calc_two_sites_diagonal_top_left_bottom_right_multiple_gates`.

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
    return calc_two_sites_diagonal_top_left_bottom_right_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]


def calc_two_sites_diagonal_top_right_bottom_left_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two from top left to bottom
    right diagonal ordered PEPS tensor and their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

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

    traced_density_matrix_bottom_right = apply_contraction(
        "ctmrg_bottom_right", [peps_tensors[3]], [peps_tensor_objs[3]], []
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _two_site_diagonal_workhorse(
        density_matrix_top_right,
        density_matrix_bottom_left,
        traced_density_matrix_bottom_right,
        traced_density_matrix_top_left,
        tuple(gates),
        real_result,
    )


def calc_two_sites_diagonal_top_right_bottom_left_single_gate(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gate: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the two site expectation value for two from top right to bottom
    left diagonal ordered PEPS tensor and their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, bottom-left, bottom-right].

    This function just wraps
    :obj:`~varipeps.expectation.two_sites.calc_two_sites_diagonal_top_right_bottom_left_multiple_gates`.

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
    return calc_two_sites_diagonal_top_right_bottom_left_multiple_gates(
        peps_tensors, peps_tensor_objs, (gate,)
    )[0]


def calc_two_sites_diagonal_horizontal_rectangle_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two from top left to the
    bottom right site of in a 2x3 horizontal rectangle ordered PEPS tensors and
    their environment.

    The order of the PEPS sequence have to be
    [top-left, top-middle, top-right, bottom-left, bottom-middle, bottom-right].

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

    traced_density_matrix_top_right = apply_contraction_jitted(
        "ctmrg_top_right", [peps_tensors[2]], [peps_tensor_objs[2]], []
    )

    traced_density_matrix_bottom_left = apply_contraction_jitted(
        "ctmrg_bottom_left", [peps_tensors[3]], [peps_tensor_objs[3]], []
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[5]],
        [peps_tensor_objs[5]],
        [],
    )

    full_density_matrix = apply_contraction_jitted(
        "density_matrix_two_sites_horizontal_rectangle",
        [peps_tensors[1], peps_tensors[4]],
        [peps_tensor_objs[1], peps_tensor_objs[4]],
        [
            density_matrix_top_left,
            traced_density_matrix_top_right,
            traced_density_matrix_bottom_left,
            density_matrix_bottom_right,
        ],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _two_site_full_density_workhorse(
        full_density_matrix, tuple(gates), real_result
    )


def calc_two_sites_diagonal_vertical_rectangle_multiple_gates(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    gates: Sequence[jnp.ndarray],
) -> List[jnp.ndarray]:
    """
    Calculate the two site expectation values for two from top left to the
    bottom right site of in a 3x2 vertical rectangle ordered PEPS tensors and
    their environment.

    The order of the PEPS sequence have to be
    [top-left, top-right, middle-left, middle-right, bottom-left, bottom-right].

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

    traced_density_matrix_top_right = apply_contraction_jitted(
        "ctmrg_top_right", [peps_tensors[1]], [peps_tensor_objs[1]], []
    )

    traced_density_matrix_bottom_left = apply_contraction_jitted(
        "ctmrg_bottom_left", [peps_tensors[4]], [peps_tensor_objs[4]], []
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "density_matrix_four_sites_bottom_right",
        [peps_tensors[5]],
        [peps_tensor_objs[5]],
        [],
    )

    full_density_matrix = apply_contraction_jitted(
        "density_matrix_two_sites_vertical_rectangle",
        [peps_tensors[2], peps_tensors[3]],
        [peps_tensor_objs[2], peps_tensor_objs[3]],
        [
            density_matrix_top_left,
            traced_density_matrix_top_right,
            traced_density_matrix_bottom_left,
            density_matrix_bottom_right,
        ],
    )

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _two_site_full_density_workhorse(
        full_density_matrix, tuple(gates), real_result
    )


@dataclass
class Two_Sites_Expectation_Value(Expectation_Model):
    horizontal_gates: Sequence[jnp.ndarray]
    vertical_gates: Sequence[jnp.ndarray]

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.horizontal_gates, jnp.ndarray):
            self.horizontal_gates = (self.horizontal_gates,)

        if isinstance(self.vertical_gates, jnp.ndarray):
            self.vertical_gates = (self.vertical_gates,)

        if (
            len(self.horizontal_gates) > 0
            and len(self.horizontal_gates) > 0
            and len(self.horizontal_gates) != len(self.vertical_gates)
        ):
            raise ValueError("Length of horizontal and vertical gates mismatch.")

        if self.is_spiral_peps:
            self._spiral_D, self._spiral_sigma = jnp.linalg.eigh(
                self.spiral_unitary_operator
            )

    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        spiral_vectors: Optional[Union[jnp.ndarray, Sequence[jnp.ndarray]]] = None,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result_type = (
            jnp.float64
            if all(jnp.allclose(g, g.T.conj()) for g in self.horizontal_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.vertical_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(max(len(self.horizontal_gates), len(self.vertical_gates)))
        ]

        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (spiral_vectors,)
            working_h_gates = [
                apply_unitary(
                    h,
                    jnp.array((0, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    int(np.sqrt(h.shape[0])),
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.horizontal_gates
            ]
            working_v_gates = [
                apply_unitary(
                    v,
                    jnp.array((1, 0)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    int(np.sqrt(v.shape[0])),
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for v in self.vertical_gates
            ]
        else:
            working_h_gates = self.horizontal_gates
            working_v_gates = self.vertical_gates

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                if len(self.horizontal_gates) > 0:
                    horizontal_tensors_i = view.get_indices((0, slice(0, 2, None)))
                    horizontal_tensors = [
                        peps_tensors[i] for i in horizontal_tensors_i[0]
                    ]
                    horizontal_tensor_objs = view[0, :2][0]

                    step_result_horizontal = calc_two_sites_horizontal_multiple_gates(
                        horizontal_tensors,
                        horizontal_tensor_objs,
                        working_h_gates,
                    )

                    for sr_i, sr in enumerate(step_result_horizontal):
                        result[sr_i] += sr

                if len(self.vertical_gates) > 0:
                    vertical_tensors_i = view.get_indices((slice(0, 2, None), 0))
                    vertical_tensors = [
                        peps_tensors[vertical_tensors_i[0][0]],
                        peps_tensors[vertical_tensors_i[1][0]],
                    ]
                    vertical_tensor_objs = [view[0, 0][0][0], view[1, 0][0][0]]

                    step_result_vertical = calc_two_sites_vertical_multiple_gates(
                        vertical_tensors, vertical_tensor_objs, working_v_gates
                    )

                    for sr_i, sr in enumerate(step_result_vertical):
                        result[sr_i] += sr

        if normalize_by_size:
            if only_unique:
                size = unitcell.get_len_unique_tensors()
            else:
                size = unitcell.get_size()[0] * unitcell.get_size()[1]
            result = [r / size for r in result]

        if len(result) == 1:
            return result[0]
        else:
            return result
