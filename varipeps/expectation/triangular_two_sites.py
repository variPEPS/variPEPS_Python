from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from varipeps import varipeps_config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction_jitted
from .model import Expectation_Model
from .spiral_helpers import apply_unitary

from varipeps.utils.debug_print import debug_print

from typing import Sequence, List, Tuple, Union, Optional


def calc_triangular_two_sites_workhorse(
    density_matrix_left,
    density_matrix_right,
    gates,
    real_result=False,
):
    density_matrix_left = density_matrix_left.reshape(
        density_matrix_left.shape[0], density_matrix_left.shape[1], -1
    )

    density_matrix_right = density_matrix_right.reshape(
        -1, density_matrix_right.shape[-2], density_matrix_right.shape[-1]
    )

    density_matrix = jnp.tensordot(
        density_matrix_left, density_matrix_right, ((2,), (0,))
    )

    density_matrix = density_matrix.transpose(0, 2, 1, 3)
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


@partial(jit, static_argnums=(3,))
def calc_triangular_two_sites_horizontal(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_left = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_horizontal_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_right = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_horizontal_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    return calc_triangular_two_sites_workhorse(
        density_matrix_left, density_matrix_right, gates, real_result
    )


@partial(jit, static_argnums=(3,))
def calc_triangular_two_sites_vertical(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_vertical_top",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_bottom = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_vertical_bottom",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    return calc_triangular_two_sites_workhorse(
        density_matrix_top, density_matrix_bottom, gates, real_result
    )


@partial(jit, static_argnums=(3,))
def calc_triangular_two_sites_diagonal(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_diagonal_top",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_bottom = apply_contraction_jitted(
        "triangular_ctmrg_two_site_expectation_diagonal_bottom",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    return calc_triangular_two_sites_workhorse(
        density_matrix_top, density_matrix_bottom, gates, real_result
    )


@dataclass
class Triangular_Two_Sites_Expectation_Value(Expectation_Model):
    horizontal_gates: Sequence[jnp.ndarray]
    vertical_gates: Sequence[jnp.ndarray]
    diagonal_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.horizontal_gates, jnp.ndarray):
            self.horizontal_gates = (self.horizontal_gates,)

        if isinstance(self.vertical_gates, jnp.ndarray):
            self.vertical_gates = (self.vertical_gates,)

        if isinstance(self.diagonal_gates, jnp.ndarray):
            self.diagonal_gates = (self.diagonal_gates,)

        if (
            len(self.horizontal_gates) > 0
            and len(self.vertical_gates) > 0
            and len(self.diagonal_gates) > 0
            and len(self.horizontal_gates)
            != len(self.vertical_gates)
            != len(self.diagonal_gates)
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
            and all(jnp.allclose(g, g.T.conj()) for g in self.diagonal_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type) for _ in range(len(self.horizontal_gates))
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
                    self.real_d,
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
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for v in self.vertical_gates
            ]
            working_d_gates = [
                apply_unitary(
                    d,
                    jnp.array((1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for d in self.diagonal_gates
            ]
        else:
            working_h_gates = self.horizontal_gates
            working_v_gates = self.vertical_gates
            working_d_gates = self.diagonal_gates

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                horizontal_tensors_i = view.get_indices((0, slice(0, 2, None)))
                horizontal_tensors = [peps_tensors[i] for i in horizontal_tensors_i[0]]
                horizontal_tensor_objs = view[0, :2][0]

                step_result_horizontal = calc_triangular_two_sites_horizontal(
                    horizontal_tensors,
                    horizontal_tensor_objs,
                    working_h_gates,
                    result_type == jnp.float64,
                )

                vertical_tensors_i = view.get_indices((slice(0, 2, None), 0))
                vertical_tensors = [
                    peps_tensors[vertical_tensors_i[0][0]],
                    peps_tensors[vertical_tensors_i[1][0]],
                ]
                vertical_tensor_objs = [view[0, 0][0][0], view[1, 0][0][0]]

                step_result_vertical = calc_triangular_two_sites_vertical(
                    vertical_tensors,
                    vertical_tensor_objs,
                    working_v_gates,
                    result_type == jnp.float64,
                )

                diagonal_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                diagonal_tensors = [
                    peps_tensors[diagonal_tensors_i[0][0]],
                    peps_tensors[diagonal_tensors_i[1][1]],
                ]
                diagonal_tensor_objs = [view[0, 0][0][0], view[1, 1][0][0]]

                step_result_diagonal = calc_triangular_two_sites_diagonal(
                    diagonal_tensors,
                    diagonal_tensor_objs,
                    working_d_gates,
                    result_type == jnp.float64,
                )

                for sr_i, (sr_h, sr_v, sr_d) in enumerate(
                    jax.util.safe_zip(
                        step_result_horizontal,
                        step_result_vertical,
                        step_result_diagonal,
                    )
                ):
                    result[sr_i] += sr_h + sr_v + sr_d

        if normalize_by_size:
            if only_unique:
                size = unitcell.get_len_unique_tensors()
            else:
                size = unitcell.get_size()[0] * unitcell.get_size()[1]
            size = size * self.normalization_factor
            result = [r / size for r in result]

        if len(result) == 1:
            return result[0]
        else:
            return result
