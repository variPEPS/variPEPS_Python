from dataclasses import dataclass
from functools import partial

import h5py

import jax
import jax.numpy as jnp
from jax import jit

from varipeps import varipeps_config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction_jitted
from .model import Expectation_Model
from .spiral_helpers import apply_unitary
from .triangular_two_sites import (
    calc_triangular_two_sites_horizontal,
    calc_triangular_two_sites_vertical,
    calc_triangular_two_sites_diagonal,
)

from varipeps.utils.debug_print import debug_print

from typing import Sequence, List, Tuple, Union, Optional


@partial(jit, static_argnums=(3,))
def calc_triangular_next_nearest_neg_x_pos_y(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_bottom_left, density_matrix_top_left, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_right,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_top_right,
        ((2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5)),
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
def calc_triangular_next_nearest_pos_x_2_pos_y(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_top_left, density_matrix_top_right, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_left,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_right,
        ((2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5)),
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
def calc_triangular_next_nearest_2_pos_x_pos_y(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_top",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_middle_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_middle_left",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_middle_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_middle_right",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_bottom",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_top, density_matrix_middle_right, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_middle_left,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix, density_matrix_bottom, ((2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5))
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


@dataclass
class Triangular_Next_Nearest_Neighbor_Expectation_Value(Expectation_Model):
    nearest_horizontal_gates: Sequence[jnp.ndarray]
    nearest_vertical_gates: Sequence[jnp.ndarray]
    nearest_diagonal_gates: Sequence[jnp.ndarray]
    next_nearest_neg_x_pos_y_gates: Sequence[jnp.ndarray]
    next_nearest_pos_x_2_pos_y_gates: Sequence[jnp.ndarray]
    next_nearest_2_pos_x_pos_y_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.nearest_horizontal_gates, jnp.ndarray):
            self.nearest_horizontal_gates = (self.nearest_horizontal_gates,)

        if isinstance(self.nearest_vertical_gates, jnp.ndarray):
            self.nearest_vertical_gates = (self.nearest_vertical_gates,)

        if isinstance(self.nearest_diagonal_gates, jnp.ndarray):
            self.nearest_diagonal_gates = (self.nearest_diagonal_gates,)

        if isinstance(self.next_nearest_neg_x_pos_y_gates, jnp.ndarray):
            self.next_nearest_neg_x_pos_y_gates = (self.next_nearest_neg_x_pos_y_gates,)

        if isinstance(self.next_nearest_pos_x_2_pos_y_gates, jnp.ndarray):
            self.next_nearest_pos_x_2_pos_y_gates = (
                self.next_nearest_pos_x_2_pos_y_gates,
            )

        if isinstance(self.next_nearest_2_pos_x_pos_y_gates, jnp.ndarray):
            self.next_nearest_2_pos_x_pos_y_gates = (
                self.next_nearest_2_pos_x_pos_y_gates,
            )

        if (
            len(self.nearest_horizontal_gates) > 0
            and len(self.nearest_vertical_gates) > 0
            and len(self.nearest_diagonal_gates) > 0
            and len(self.next_nearest_neg_x_pos_y_gates) > 0
            and len(self.next_nearest_pos_x_2_pos_y_gates) > 0
            and len(self.next_nearest_2_pos_x_pos_y_gates) > 0
            and len(self.nearest_horizontal_gates)
            != len(self.nearest_vertical_gates)
            != len(self.nearest_diagonal_gates)
            != len(self.next_nearest_neg_x_pos_y_gates)
            != len(self.next_nearest_pos_x_2_pos_y_gates)
            != len(self.next_nearest_2_pos_x_pos_y_gates)
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
            if all(jnp.allclose(g, g.T.conj()) for g in self.nearest_horizontal_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.nearest_vertical_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.nearest_diagonal_gates)
            and all(
                jnp.allclose(g, g.T.conj()) for g in self.next_nearest_neg_x_pos_y_gates
            )
            and all(
                jnp.allclose(g, g.T.conj())
                for g in self.next_nearest_pos_x_2_pos_y_gates
            )
            and all(
                jnp.allclose(g, g.T.conj())
                for g in self.next_nearest_2_pos_x_pos_y_gates
            )
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(len(self.nearest_horizontal_gates))
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
                for h in self.nearest_horizontal_gates
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
                for v in self.nearest_vertical_gates
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
                for d in self.nearest_diagonal_gates
            ]
            working_nn_neg_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((-1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in self.next_nearest_neg_x_pos_y_gates
            ]
            working_nn_pos_2pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 2)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in self.next_nearest_pos_x_2_pos_y_gates
            ]
            working_nn_2pos_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((2, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in self.next_nearest_2_pos_x_pos_y_gates
            ]
        else:
            working_h_gates = self.nearest_horizontal_gates
            working_v_gates = self.nearest_vertical_gates
            working_d_gates = self.nearest_diagonal_gates
            working_nn_neg_pos_gates = self.next_nearest_neg_x_pos_y_gates
            working_nn_pos_2pos_gates = self.next_nearest_pos_x_2_pos_y_gates
            working_nn_2pos_pos_gates = self.next_nearest_2_pos_x_pos_y_gates

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

                nn_neg_pos_tensors_i = view.get_indices(
                    (slice(-1, 1, None), slice(0, 2, None))
                )
                nn_neg_pos_tensors = [
                    peps_tensors[j] for i in nn_neg_pos_tensors_i for j in i
                ]
                nn_neg_pos_tensor_objs = [j for i in view[-1:1, :2] for j in i]

                step_result_nn_neg_pos = calc_triangular_next_nearest_neg_x_pos_y(
                    nn_neg_pos_tensors,
                    nn_neg_pos_tensor_objs,
                    working_nn_neg_pos_gates,
                    result_type == jnp.float64,
                )

                nn_pos_2pos_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 3, None))
                )
                nn_pos_2pos_tensors = [
                    peps_tensors[j] for i in nn_pos_2pos_tensors_i for j in i
                ]
                nn_pos_2pos_tensors = nn_pos_2pos_tensors[:2] + nn_pos_2pos_tensors[4:]
                nn_pos_2pos_tensor_objs = [j for i in view[:2, :3] for j in i]
                nn_pos_2pos_tensor_objs = (
                    nn_pos_2pos_tensor_objs[:2] + nn_pos_2pos_tensor_objs[4:]
                )

                step_result_nn_pos_2pos = calc_triangular_next_nearest_pos_x_2_pos_y(
                    nn_pos_2pos_tensors,
                    nn_pos_2pos_tensor_objs,
                    working_nn_pos_2pos_gates,
                    result_type == jnp.float64,
                )

                nn_2pos_pos_tensors_i = view.get_indices(
                    (slice(0, 3, None), slice(0, 2, None))
                )
                nn_2pos_pos_tensors = [
                    peps_tensors[j] for i in nn_2pos_pos_tensors_i for j in i
                ]
                nn_2pos_pos_tensors = (
                    nn_2pos_pos_tensors[:1]
                    + nn_2pos_pos_tensors[2:4]
                    + nn_2pos_pos_tensors[5:]
                )
                nn_2pos_pos_tensor_objs = [j for i in view[:3, :2] for j in i]
                nn_2pos_pos_tensor_objs = (
                    nn_2pos_pos_tensor_objs[:1]
                    + nn_2pos_pos_tensor_objs[2:4]
                    + nn_2pos_pos_tensor_objs[5:]
                )

                step_result_nn_2pos_pos = calc_triangular_next_nearest_2_pos_x_pos_y(
                    nn_2pos_pos_tensors,
                    nn_2pos_pos_tensor_objs,
                    working_nn_2pos_pos_gates,
                    result_type == jnp.float64,
                )

                for sr_i, (sr_h, sr_v, sr_d, sr_np, sr_p2p, sr_2pp) in enumerate(
                    zip(
                        step_result_horizontal,
                        step_result_vertical,
                        step_result_diagonal,
                        step_result_nn_neg_pos,
                        step_result_nn_pos_2pos,
                        step_result_nn_2pos_pos,
                        strict=True,
                    )
                ):
                    result[sr_i] += sr_h + sr_v + sr_d + sr_np + sr_p2p + sr_2pp

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

    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        grp_gates = grp.create_group("gates", track_order=True)
        grp_gates.attrs["len"] = len(self.nearest_horizontal_gates)
        for i, (
            h_g,
            v_g,
            d_g,
            nn_neg_pos_g,
            nn_pos_2_pos_g,
            nn_2_pos_pos_g,
        ) in enumerate(
            zip(
                self.nearest_horizontal_gates,
                self.nearest_vertical_gates,
                self.nearest_diagonal_gates,
                self.next_nearest_neg_x_pos_y_gates,
                self.next_nearest_pos_x_2_pos_y_gates,
                self.next_nearest_2_pos_x_pos_y_gates,
                strict=True,
            )
        ):
            grp_gates.create_dataset(
                f"nearest_horizontal_gate_{i:d}",
                data=h_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"nearest_vertical_gate_{i:d}",
                data=v_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"nearest_diagonal_gate_{i:d}",
                data=d_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_neg_x_pos_y_gate_{i:d}",
                data=nn_neg_pos_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_pos_x_2_pos_y_gate_{i:d}",
                data=nn_pos_2_pos_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_2_pos_x_pos_y_gate_{i:d}",
                data=nn_2_pos_pos_g,
                compression="gzip",
                compression_opts=6,
            )

        grp.attrs["real_d"] = self.real_d
        grp.attrs["normalization_factor"] = self.normalization_factor
        grp.attrs["is_spiral_peps"] = self.is_spiral_peps

        if self.is_spiral_peps:
            grp.create_dataset(
                "spiral_unitary_operator",
                data=self.spiral_unitary_operator,
                compression="gzip",
                compression_opts=6,
            )

    @classmethod
    def load_from_group(cls, grp: h5py.Group):
        if not grp.attrs["class"] == f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(
                "The HDF5 group suggests that this is not the right class to load data from it."
            )

        horizontal_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_horizontal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        vertical_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_vertical_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        diagonal_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_diagonal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_neg_x_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_neg_x_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_pos_x_2_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_pos_x_2_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_2_pos_x_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_2_pos_x_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )

        is_spiral_peps = grp.attrs["is_spiral_peps"]

        if is_spiral_peps:
            spiral_unitary_operator = jnp.asarray(grp["spiral_unitary_operator"])
        else:
            spiral_unitary_operator = None

        return cls(
            horizontal_gates=horizontal_gates,
            vertical_gates=vertical_gates,
            diagonal_gates=diagonal_gates,
            next_nearest_neg_x_pos_y_gates=next_nearest_neg_x_pos_y_gates,
            next_nearest_pos_x_2_pos_y_gates=next_nearest_pos_x_2_pos_y_gates,
            next_nearest_2_pos_x_pos_y_gates=next_nearest_2_pos_x_pos_y_gates,
            real_d=grp.attrs["real_d"],
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )


def get_nn_neg_x_pos_y_gates(vertical_e, horizontal_e, diagonal_e, nn_neg_x_pos_y_e, d):

    Id_other_site = jnp.eye(d**2)

    vertical_base = jnp.kron(vertical_e, Id_other_site)
    vertical_base = vertical_base.reshape(d, d, d, d, d, d, d, d)

    horizontal_base = jnp.kron(horizontal_e, Id_other_site)
    horizontal_base = horizontal_base.reshape(d, d, d, d, d, d, d, d)

    diagonal_base = jnp.kron(diagonal_e, Id_other_site)
    diagonal_base = diagonal_base.reshape(d, d, d, d, d, d, d, d)

    nn_neg_x_pos_y_base = jnp.kron(nn_neg_x_pos_y_e, Id_other_site)
    nn_neg_x_pos_y_base = nn_neg_x_pos_y_base.reshape(d, d, d, d, d, d, d, d)

    vertical_13 = vertical_base.transpose((0, 2, 1, 3, 4, 6, 5, 7))
    vertical_13 = vertical_13.reshape(d**4, d**4)
    vertical_24 = vertical_base.transpose((2, 0, 3, 1, 6, 4, 7, 5))
    vertical_24 = vertical_24.reshape(d**4, d**4)

    horizontal_12 = horizontal_base.transpose((0, 1, 2, 3, 4, 5, 6, 7))
    horizontal_12 = horizontal_12.reshape(d**4, d**4)
    horizontal_34 = horizontal_base.transpose((2, 3, 0, 1, 6, 7, 4, 5))
    horizontal_34 = horizontal_34.reshape(d**4, d**4)

    diagonal_14 = diagonal_base.transpose((0, 2, 3, 1, 4, 6, 7, 5))
    diagonal_14 = diagonal_14.reshape(d**4, d**4)

    nn_neg_x_pos_y_23 = nn_neg_x_pos_y_base.transpose((2, 0, 1, 3, 6, 4, 5, 7))
    nn_neg_x_pos_y_23 = nn_neg_x_pos_y_23.reshape(d**4, d**4)

    return (
        vertical_13,
        vertical_24,
        horizontal_12,
        horizontal_34,
        diagonal_14,
        nn_neg_x_pos_y_23,
    )


def get_nn_pos_x_2_pos_y_gates(
    vertical_e, horizontal_e, diagonal_e, nn_pos_x_2_pos_y_e, d
):

    Id_other_site = jnp.eye(d**2)

    vertical_base = jnp.kron(vertical_e, Id_other_site)
    vertical_base = vertical_base.reshape(d, d, d, d, d, d, d, d)

    horizontal_base = jnp.kron(horizontal_e, Id_other_site)
    horizontal_base = horizontal_base.reshape(d, d, d, d, d, d, d, d)

    diagonal_base = jnp.kron(diagonal_e, Id_other_site)
    diagonal_base = diagonal_base.reshape(d, d, d, d, d, d, d, d)

    nn_pos_x_2_pos_y_base = jnp.kron(nn_pos_x_2_pos_y_e, Id_other_site)
    nn_pos_x_2_pos_y_base = nn_pos_x_2_pos_y_base.reshape(d, d, d, d, d, d, d, d)

    vertical_23 = vertical_base.transpose((2, 0, 1, 3, 6, 4, 5, 7))
    vertical_23 = vertical_23.reshape(d**4, d**4)

    horizontal_12 = horizontal_base.transpose((0, 1, 2, 3, 4, 5, 6, 7))
    horizontal_12 = horizontal_12.reshape(d**4, d**4)
    horizontal_34 = horizontal_base.transpose((2, 3, 0, 1, 6, 7, 4, 5))
    horizontal_34 = horizontal_34.reshape(d**4, d**4)

    diagonal_13 = diagonal_base.transpose((0, 2, 1, 3, 4, 6, 5, 7))
    diagonal_13 = diagonal_13.reshape(d**4, d**4)
    diagonal_24 = diagonal_base.transpose((2, 0, 3, 1, 6, 4, 7, 5))
    diagonal_24 = diagonal_24.reshape(d**4, d**4)

    nn_pos_x_2_pos_y_14 = nn_pos_x_2_pos_y_base.transpose((0, 2, 3, 1, 4, 6, 7, 5))
    nn_pos_x_2_pos_y_14 = nn_pos_x_2_pos_y_14.reshape(d**4, d**4)

    return (
        vertical_23,
        horizontal_12,
        horizontal_34,
        diagonal_13,
        diagonal_24,
        nn_pos_x_2_pos_y_14,
    )


def get_nn_2_pos_x_pos_y_gates(
    vertical_e, horizontal_e, diagonal_e, nn_2_pos_x_pos_y_e, d
):

    Id_other_site = jnp.eye(d**2)

    vertical_base = jnp.kron(vertical_e, Id_other_site)
    vertical_base = vertical_base.reshape(d, d, d, d, d, d, d, d)

    horizontal_base = jnp.kron(horizontal_e, Id_other_site)
    horizontal_base = horizontal_base.reshape(d, d, d, d, d, d, d, d)

    diagonal_base = jnp.kron(diagonal_e, Id_other_site)
    diagonal_base = diagonal_base.reshape(d, d, d, d, d, d, d, d)

    nn_2_pos_x_pos_y_base = jnp.kron(nn_2_pos_x_pos_y_e, Id_other_site)
    nn_2_pos_x_pos_y_base = nn_2_pos_x_pos_y_base.reshape(d, d, d, d, d, d, d, d)

    vertical_12 = vertical_base.transpose((0, 1, 2, 3, 4, 5, 6, 7))
    vertical_12 = vertical_12.reshape(d**4, d**4)
    vertical_34 = vertical_base.transpose((2, 3, 0, 1, 6, 7, 4, 5))
    vertical_34 = vertical_34.reshape(d**4, d**4)

    horizontal_23 = horizontal_base.transpose((2, 0, 1, 3, 6, 4, 5, 7))
    horizontal_23 = horizontal_23.reshape(d**4, d**4)

    diagonal_13 = diagonal_base.transpose((0, 2, 1, 3, 4, 6, 5, 7))
    diagonal_13 = diagonal_13.reshape(d**4, d**4)
    diagonal_24 = diagonal_base.transpose((2, 0, 3, 1, 6, 4, 7, 5))
    diagonal_24 = diagonal_24.reshape(d**4, d**4)

    nn_2_pos_x_pos_y_14 = nn_2_pos_x_pos_y_base.transpose((0, 2, 3, 1, 4, 6, 7, 5))
    nn_2_pos_x_pos_y_14 = nn_2_pos_x_pos_y_14.reshape(d**4, d**4)

    return (
        vertical_12,
        vertical_34,
        horizontal_23,
        diagonal_13,
        diagonal_24,
        nn_2_pos_x_pos_y_14,
    )


@partial(jit, static_argnums=(4, 5))
def _calc_nn_neg_x_pos_y_gate(
    vertical_gates: Sequence[jnp.ndarray],
    horizontal_gates: Sequence[jnp.ndarray],
    diagonal_gates: Sequence[jnp.ndarray],
    nn_neg_x_pos_y_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    for i, (vertical_e, horizontal_e, diagonal_e, nn_neg_x_pos_y_e) in enumerate(
        zip(
            vertical_gates,
            horizontal_gates,
            diagonal_gates,
            nn_neg_x_pos_y_gates,
            strict=True,
        )
    ):
        (
            vertical_13,
            vertical_24,
            horizontal_12,
            horizontal_34,
            diagonal_14,
            nn_neg_x_pos_y_23,
        ) = get_nn_neg_x_pos_y_gates(
            vertical_e, horizontal_e, diagonal_e, nn_neg_x_pos_y_e, d
        )

        result[i] = (
            1 / 5 * vertical_13
            + 1 / 5 * vertical_24
            + 1 / 5 * horizontal_12
            + 1 / 5 * horizontal_34
            + 1 / 5 * diagonal_14
            + nn_neg_x_pos_y_23
        )

        single_gates[i] = (
            vertical_13,
            vertical_24,
            horizontal_12,
            horizontal_34,
            diagonal_14,
            nn_neg_x_pos_y_23,
        )

    return result, single_gates


@partial(jit, static_argnums=(4, 5))
def _calc_nn_pos_x_2_pos_y_gate(
    vertical_gates: Sequence[jnp.ndarray],
    horizontal_gates: Sequence[jnp.ndarray],
    diagonal_gates: Sequence[jnp.ndarray],
    nn_pos_x_2_pos_y_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    for i, (vertical_e, horizontal_e, diagonal_e, nn_pos_x_2_pos_y_e) in enumerate(
        zip(
            vertical_gates,
            horizontal_gates,
            diagonal_gates,
            nn_pos_x_2_pos_y_gates,
            strict=True,
        )
    ):
        (
            vertical_23,
            horizontal_12,
            horizontal_34,
            diagonal_13,
            diagonal_24,
            nn_pos_x_2_pos_y_14,
        ) = get_nn_pos_x_2_pos_y_gates(
            vertical_e, horizontal_e, diagonal_e, nn_pos_x_2_pos_y_e, d
        )

        result[i] = (
            1 / 5 * vertical_23
            + 1 / 5 * horizontal_12
            + 1 / 5 * horizontal_34
            + 1 / 5 * diagonal_13
            + 1 / 5 * diagonal_24
            + nn_pos_x_2_pos_y_14
        )

        single_gates[i] = (
            vertical_23,
            horizontal_12,
            horizontal_34,
            diagonal_13,
            diagonal_24,
            nn_pos_x_2_pos_y_14,
        )

    return result, single_gates


@partial(jit, static_argnums=(4, 5))
def _calc_nn_2_pos_x_pos_y_gate(
    vertical_gates: Sequence[jnp.ndarray],
    horizontal_gates: Sequence[jnp.ndarray],
    diagonal_gates: Sequence[jnp.ndarray],
    nn_2_pos_x_pos_y_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    for i, (vertical_e, horizontal_e, diagonal_e, nn_2_pos_x_pos_y_e) in enumerate(
        zip(
            vertical_gates,
            horizontal_gates,
            diagonal_gates,
            nn_2_pos_x_pos_y_gates,
            strict=True,
        )
    ):
        (
            vertical_12,
            vertical_34,
            horizontal_23,
            diagonal_13,
            diagonal_24,
            nn_2_pos_x_pos_y_14,
        ) = get_nn_2_pos_x_pos_y_gates(
            vertical_e, horizontal_e, diagonal_e, nn_2_pos_x_pos_y_e, d
        )

        result[i] = (
            1 / 5 * vertical_12
            + 1 / 5 * vertical_34
            + 1 / 5 * horizontal_23
            + 1 / 5 * diagonal_13
            + 1 / 5 * diagonal_24
            + nn_2_pos_x_pos_y_14
        )

        single_gates[i] = (
            vertical_12,
            vertical_34,
            horizontal_23,
            diagonal_13,
            diagonal_24,
            nn_2_pos_x_pos_y_14,
        )

    return result, single_gates


@partial(jit, static_argnums=(3,))
def calc_triangular_next_nearest_neg_x_pos_y_new(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_top_left_open",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_top_right",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_bottom_left",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_neg_x_pos_y_expectation_bottom_right_open",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_bottom_left, density_matrix_top_left, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_right,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_top_right,
        ((2, 3, 4, 7, 8, 9), (0, 1, 2, 3, 4, 5)),
    )

    density_matrix = density_matrix.transpose(2, 6, 0, 4, 3, 7, 1, 5)
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


@partial(jit, static_argnums=(3,))
def calc_triangular_next_nearest_pos_x_2_pos_y_new(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_top_left",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_top_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_top_right_open",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_bottom_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_bottom_left_open",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_pos_x_2_pos_y_expectation_bottom_right",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_top_left, density_matrix_top_right, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_left,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_bottom_right,
        ((2, 3, 4, 7, 8, 9), (0, 1, 2, 3, 4, 5)),
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


@partial(jit, static_argnums=(3,))
def calc_triangular_next_nearest_2_pos_x_pos_y_new(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_top = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_top",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_middle_left = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_middle_left_open",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_middle_right = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_middle_right_open",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix_bottom = apply_contraction_jitted(
        "triangular_ctmrg_next_nearest_2_pos_x_pos_y_expectation_bottom",
        [peps_tensors[3]],
        [peps_tensor_objs[3]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_top, density_matrix_middle_right, ((5, 6, 7), (0, 1, 2))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_middle_left,
        (
            (
                2,
                3,
                4,
                5,
                6,
            ),
            (0, 1, 2, 3, 4),
        ),
    )

    density_matrix = jnp.tensordot(
        density_matrix, density_matrix_bottom, ((2, 3, 4, 7, 8, 9), (0, 1, 2, 3, 4, 5))
    )

    density_matrix = density_matrix.transpose(0, 4, 2, 6, 1, 5, 3, 7)
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


@dataclass
class Triangular_Next_Nearest_Neighbor_Expectation_Value_2(Expectation_Model):
    nearest_horizontal_gates: Sequence[jnp.ndarray]
    nearest_vertical_gates: Sequence[jnp.ndarray]
    nearest_diagonal_gates: Sequence[jnp.ndarray]
    next_nearest_neg_x_pos_y_gates: Sequence[jnp.ndarray]
    next_nearest_pos_x_2_pos_y_gates: Sequence[jnp.ndarray]
    next_nearest_2_pos_x_pos_y_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.nearest_horizontal_gates, jnp.ndarray):
            self.nearest_horizontal_gates = (self.nearest_horizontal_gates,)

        if isinstance(self.nearest_vertical_gates, jnp.ndarray):
            self.nearest_vertical_gates = (self.nearest_vertical_gates,)

        if isinstance(self.nearest_diagonal_gates, jnp.ndarray):
            self.nearest_diagonal_gates = (self.nearest_diagonal_gates,)

        if isinstance(self.next_nearest_neg_x_pos_y_gates, jnp.ndarray):
            self.next_nearest_neg_x_pos_y_gates = (self.next_nearest_neg_x_pos_y_gates,)

        if isinstance(self.next_nearest_pos_x_2_pos_y_gates, jnp.ndarray):
            self.next_nearest_pos_x_2_pos_y_gates = (
                self.next_nearest_pos_x_2_pos_y_gates,
            )

        if isinstance(self.next_nearest_2_pos_x_pos_y_gates, jnp.ndarray):
            self.next_nearest_2_pos_x_pos_y_gates = (
                self.next_nearest_2_pos_x_pos_y_gates,
            )

        if (
            len(self.nearest_horizontal_gates) > 0
            and len(self.nearest_vertical_gates) > 0
            and len(self.nearest_diagonal_gates) > 0
            and len(self.next_nearest_neg_x_pos_y_gates) > 0
            and len(self.next_nearest_pos_x_2_pos_y_gates) > 0
            and len(self.next_nearest_2_pos_x_pos_y_gates) > 0
            and len(self.nearest_horizontal_gates)
            != len(self.nearest_vertical_gates)
            != len(self.nearest_diagonal_gates)
            != len(self.next_nearest_neg_x_pos_y_gates)
            != len(self.next_nearest_pos_x_2_pos_y_gates)
            != len(self.next_nearest_2_pos_x_pos_y_gates)
        ):
            raise ValueError("Length of horizontal and vertical gates mismatch.")

        if self.is_spiral_peps:
            self._spiral_D, self._spiral_sigma = jnp.linalg.eigh(
                self.spiral_unitary_operator
            )

        tmp_result = _calc_nn_neg_x_pos_y_gate(
            self.nearest_vertical_gates,
            self.nearest_horizontal_gates,
            self.nearest_diagonal_gates,
            self.next_nearest_neg_x_pos_y_gates,
            self.real_d,
            len(self.nearest_vertical_gates),
        )
        self._nn_neg_x_pos_y_tuple, self._nn_neg_x_pos_y_single_gates = tuple(
            tmp_result[0]
        ), tuple(tmp_result[1])

        tmp_result = _calc_nn_pos_x_2_pos_y_gate(
            self.nearest_vertical_gates,
            self.nearest_horizontal_gates,
            self.nearest_diagonal_gates,
            self.next_nearest_pos_x_2_pos_y_gates,
            self.real_d,
            len(self.nearest_vertical_gates),
        )
        self._nn_pos_x_2_pos_y_tuple, self._nn_pos_x_2_pos_y_single_gates = tuple(
            tmp_result[0]
        ), tuple(tmp_result[1])

        tmp_result = _calc_nn_2_pos_x_pos_y_gate(
            self.nearest_vertical_gates,
            self.nearest_horizontal_gates,
            self.nearest_diagonal_gates,
            self.next_nearest_2_pos_x_pos_y_gates,
            self.real_d,
            len(self.nearest_vertical_gates),
        )
        self._nn_2_pos_x_pos_y_tuple, self._nn_2_pos_x_pos_y_single_gates = tuple(
            tmp_result[0]
        ), tuple(tmp_result[1])

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
            if all(jnp.allclose(g, g.T.conj()) for g in self.nearest_horizontal_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.nearest_vertical_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.nearest_diagonal_gates)
            and all(
                jnp.allclose(g, g.T.conj()) for g in self.next_nearest_neg_x_pos_y_gates
            )
            and all(
                jnp.allclose(g, g.T.conj())
                for g in self.next_nearest_pos_x_2_pos_y_gates
            )
            and all(
                jnp.allclose(g, g.T.conj())
                for g in self.next_nearest_2_pos_x_pos_y_gates
            )
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(len(self.nearest_horizontal_gates))
        ]

        # set working gates
        working_nn_neg_pos_gates = self._nn_neg_x_pos_y_tuple
        working_nn_pos_2pos_gates = self._nn_pos_x_2_pos_y_tuple
        working_nn_2pos_pos_gates = self._nn_2_pos_x_pos_y_tuple

        # apply unitary transformation if spiral PEPS
        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (spiral_vectors,)

            working_nn_neg_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((0, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_neg_pos_gates
            ]
            working_nn_neg_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 0)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (2,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_neg_pos_gates
            ]
            working_nn_neg_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (3,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_neg_pos_gates
            ]

            working_nn_pos_2pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((0, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_pos_2pos_gates
            ]
            working_nn_pos_2pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (2,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_pos_2pos_gates
            ]
            working_nn_pos_2pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 2)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (3,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_pos_2pos_gates
            ]

            working_nn_2pos_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 0)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_2pos_pos_gates
            ]
            working_nn_2pos_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (2,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_2pos_pos_gates
            ]
            working_nn_2pos_pos_gates = [
                apply_unitary(
                    e,
                    jnp.array((2, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (3,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in working_nn_2pos_pos_gates
            ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:

                nn_neg_pos_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                nn_neg_pos_tensors = [
                    peps_tensors[j] for i in nn_neg_pos_tensors_i for j in i
                ]
                nn_neg_pos_tensor_objs = [j for i in view[:2, :2] for j in i]

                step_result_nn_neg_pos = calc_triangular_next_nearest_neg_x_pos_y_new(
                    nn_neg_pos_tensors,
                    nn_neg_pos_tensor_objs,
                    working_nn_neg_pos_gates,
                    result_type == jnp.float64,
                )

                nn_pos_2pos_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 3, None))
                )
                nn_pos_2pos_tensors = [
                    peps_tensors[j] for i in nn_pos_2pos_tensors_i for j in i
                ]
                nn_pos_2pos_tensors = nn_pos_2pos_tensors[:2] + nn_pos_2pos_tensors[4:]
                nn_pos_2pos_tensor_objs = [j for i in view[:2, :3] for j in i]
                nn_pos_2pos_tensor_objs = (
                    nn_pos_2pos_tensor_objs[:2] + nn_pos_2pos_tensor_objs[4:]
                )

                step_result_nn_pos_2pos = (
                    calc_triangular_next_nearest_pos_x_2_pos_y_new(
                        nn_pos_2pos_tensors,
                        nn_pos_2pos_tensor_objs,
                        working_nn_pos_2pos_gates,
                        result_type == jnp.float64,
                    )
                )

                nn_2pos_pos_tensors_i = view.get_indices(
                    (slice(0, 3, None), slice(0, 2, None))
                )
                nn_2pos_pos_tensors = [
                    peps_tensors[j] for i in nn_2pos_pos_tensors_i for j in i
                ]
                nn_2pos_pos_tensors = (
                    nn_2pos_pos_tensors[:1]
                    + nn_2pos_pos_tensors[2:4]
                    + nn_2pos_pos_tensors[5:]
                )
                nn_2pos_pos_tensor_objs = [j for i in view[:3, :2] for j in i]
                nn_2pos_pos_tensor_objs = (
                    nn_2pos_pos_tensor_objs[:1]
                    + nn_2pos_pos_tensor_objs[2:4]
                    + nn_2pos_pos_tensor_objs[5:]
                )

                step_result_nn_2pos_pos = (
                    calc_triangular_next_nearest_2_pos_x_pos_y_new(
                        nn_2pos_pos_tensors,
                        nn_2pos_pos_tensor_objs,
                        working_nn_2pos_pos_gates,
                        result_type == jnp.float64,
                    )
                )

                for sr_i, (sr_np, sr_p2p, sr_2pp) in enumerate(
                    zip(
                        step_result_nn_neg_pos,
                        step_result_nn_pos_2pos,
                        step_result_nn_2pos_pos,
                        strict=True,
                    )
                ):
                    result[sr_i] += sr_np + sr_p2p + sr_2pp

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

    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        grp_gates = grp.create_group("gates", track_order=True)
        grp_gates.attrs["len"] = len(self.nearest_horizontal_gates)
        for i, (
            h_g,
            v_g,
            d_g,
            nn_neg_pos_g,
            nn_pos_2_pos_g,
            nn_2_pos_pos_g,
        ) in enumerate(
            zip(
                self.nearest_horizontal_gates,
                self.nearest_vertical_gates,
                self.nearest_diagonal_gates,
                self.next_nearest_neg_x_pos_y_gates,
                self.next_nearest_pos_x_2_pos_y_gates,
                self.next_nearest_2_pos_x_pos_y_gates,
                strict=True,
            )
        ):
            grp_gates.create_dataset(
                f"nearest_horizontal_gate_{i:d}",
                data=h_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"nearest_vertical_gate_{i:d}",
                data=v_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"nearest_diagonal_gate_{i:d}",
                data=d_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_neg_x_pos_y_gate_{i:d}",
                data=nn_neg_pos_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_pos_x_2_pos_y_gate_{i:d}",
                data=nn_pos_2_pos_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"next_nearest_2_pos_x_pos_y_gate_{i:d}",
                data=nn_2_pos_pos_g,
                compression="gzip",
                compression_opts=6,
            )

        grp.attrs["real_d"] = self.real_d
        grp.attrs["normalization_factor"] = self.normalization_factor
        grp.attrs["is_spiral_peps"] = self.is_spiral_peps

        if self.is_spiral_peps:
            grp.create_dataset(
                "spiral_unitary_operator",
                data=self.spiral_unitary_operator,
                compression="gzip",
                compression_opts=6,
            )

    @classmethod
    def load_from_group(cls, grp: h5py.Group):
        if not grp.attrs["class"] == f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(
                "The HDF5 group suggests that this is not the right class to load data from it."
            )

        horizontal_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_horizontal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        vertical_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_vertical_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        diagonal_gates = tuple(
            jnp.asarray(grp["gates"][f"nearest_diagonal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_neg_x_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_neg_x_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_pos_x_2_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_pos_x_2_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        next_nearest_2_pos_x_pos_y_gates = tuple(
            jnp.asarray(grp["gates"][f"next_nearest_2_pos_x_pos_y_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )

        is_spiral_peps = grp.attrs["is_spiral_peps"]

        if is_spiral_peps:
            spiral_unitary_operator = jnp.asarray(grp["spiral_unitary_operator"])
        else:
            spiral_unitary_operator = None

        return cls(
            horizontal_gates=horizontal_gates,
            vertical_gates=vertical_gates,
            diagonal_gates=diagonal_gates,
            next_nearest_neg_x_pos_y_gates=next_nearest_neg_x_pos_y_gates,
            next_nearest_pos_x_2_pos_y_gates=next_nearest_pos_x_2_pos_y_gates,
            next_nearest_2_pos_x_pos_y_gates=next_nearest_2_pos_x_pos_y_gates,
            real_d=grp.attrs["real_d"],
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )
