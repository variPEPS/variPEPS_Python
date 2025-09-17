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
                    zip(
                        step_result_horizontal,
                        step_result_vertical,
                        step_result_diagonal,
                        strict=True,
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

    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        grp_gates = grp.create_group("gates", track_order=True)
        grp_gates.attrs["len"] = len(self.horizontal_gates)
        for i, (h_g, v_g, d_g) in enumerate(
            zip(
                self.horizontal_gates,
                self.vertical_gates,
                self.diagonal_gates,
                strict=True,
            )
        ):
            grp_gates.create_dataset(
                f"horizontal_gate_{i:d}",
                data=h_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"vertical_gate_{i:d}", data=v_g, compression="gzip", compression_opts=6
            )
            grp_gates.create_dataset(
                f"diagonal_gate_{i:d}", data=d_g, compression="gzip", compression_opts=6
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
            jnp.asarray(grp["gates"][f"horizontal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        vertical_gates = tuple(
            jnp.asarray(grp["gates"][f"vertical_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        diagonal_gates = tuple(
            jnp.asarray(grp["gates"][f"diagonal_gate_{i:d}"])
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
            real_d=grp.attrs["real_d"],
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )


def get_triangle_gates(vertical_e, horizontal_e, diagonal_e, d):
    Id_other_site = jnp.eye(d**1)

    vertical_12 = jnp.kron(vertical_e, Id_other_site)

    horizontal_23 = jnp.kron(Id_other_site, horizontal_e)

    diagonal_base = jnp.kron(diagonal_e, Id_other_site)
    diagonal_base = diagonal_base.reshape(d, d, d, d, d, d)
    diagonal_13 = diagonal_base.transpose((0, 2, 1, 3, 5, 4))
    diagonal_13 = diagonal_13.reshape(d**3, d**3)

    return (
        vertical_12,
        horizontal_23,
        diagonal_13,
    )


@partial(jit, static_argnums=(3, 4))
def _calc_triangle_gate(
    vertical_gates: Sequence[jnp.ndarray],
    horizontal_gates: Sequence[jnp.ndarray],
    diagonal_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    for i, (vertical_e, horizontal_e, diagonal_e) in enumerate(
        zip(vertical_gates, horizontal_gates, diagonal_gates, strict=True)
    ):
        (
            vertical_12,
            horizontal_23,
            diagonal_13,
        ) = get_triangle_gates(vertical_e, horizontal_e, diagonal_e, d)

        result[i] = vertical_12 + horizontal_23 + diagonal_13

        single_gates[i] = (
            vertical_12,
            horizontal_23,
            diagonal_13,
        )

    return result, single_gates


@partial(jit, static_argnums=(3,))
def calc_triangular_nearest_triangle(
    peps_tensors,
    peps_tensor_objs,
    gates,
    real_result=False,
):
    density_matrix_90 = apply_contraction_jitted(
        "triangular_ctmrg_corner_90_expectation",
        [peps_tensors[0]],
        [peps_tensor_objs[0]],
        [],
    )

    density_matrix_210 = apply_contraction_jitted(
        "triangular_ctmrg_corner_210_expectation",
        [peps_tensors[1]],
        [peps_tensor_objs[1]],
        [],
    )

    density_matrix_330 = apply_contraction_jitted(
        "triangular_ctmrg_corner_330_expectation",
        [peps_tensors[2]],
        [peps_tensor_objs[2]],
        [],
    )

    density_matrix = jnp.tensordot(
        density_matrix_210, density_matrix_90, ((5, 6, 7), (2, 3, 4))
    )

    density_matrix = jnp.tensordot(
        density_matrix,
        density_matrix_330,
        (
            (2, 3, 4, 7, 8, 9),
            (5, 6, 7, 2, 3, 4),
        ),
    )

    density_matrix = density_matrix.transpose(2, 0, 4, 3, 1, 5)
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


@dataclass
class Triangular_Two_Sites_Expectation_Value_2(Expectation_Model):
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
            len(self.vertical_gates) > 0
            and len(self.horizontal_gates) > 0
            and len(self.diagonal_gates) > 0
            and len(self.vertical_gates)
            != len(self.horizontal_gates)
            != len(self.diagonal_gates)
        ):
            raise ValueError("Length of horizontal and vertical gates mismatch.")

        tmp_result = _calc_triangle_gate(
            self.vertical_gates,
            self.horizontal_gates,
            self.diagonal_gates,
            self.real_d,
            len(self.vertical_gates),
        )
        self._full_triangle_tuple, self._triangle_single_gates = tuple(
            tmp_result[0]
        ), tuple(tmp_result[1])

        self._result_type = (
            jnp.float64
            if all(jnp.allclose(g, g.T.conj()) for g in self.vertical_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.horizontal_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.diagonal_gates)
            else jnp.complex128
        )

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

        result = [
            jnp.array(0, dtype=self._result_type)
            for _ in range(len(self.vertical_gates))
        ]

        # if return_single_gate_results:
        #     single_gates_result = [dict()] * len(self.vertical_gates)

        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (spiral_vectors,)

            # apply unitary rotation to index at position (1, 0)
            working_t_gates = [
                apply_unitary(
                    t,
                    jnp.array((1, 0)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for t in self._full_triangle_tuple
            ]

            # apply unitary rotation to index at position (1, 1)
            working_t_gates = [
                apply_unitary(
                    t,
                    jnp.array((1, 1)),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (2,),
                    varipeps_config.spiral_wavevector_type,
                )
                for t in working_t_gates
            ]

        else:
            working_t_gates = self._full_triangle_tuple

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:

                triangle_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                triangle_tensors = [
                    peps_tensors[j] for i in triangle_tensors_i for j in i
                ]
                triangle_tensor_objs = [j for i in view[0:2, :2] for j in i]

                step_result_triangle = calc_triangular_nearest_triangle(
                    (triangle_tensors[0], triangle_tensors[2], triangle_tensors[3]),
                    (
                        triangle_tensor_objs[0],
                        triangle_tensor_objs[2],
                        triangle_tensor_objs[3],
                    ),
                    working_t_gates,
                    self._result_type is jnp.float64,
                )

                for sr_i, (sr_t,) in enumerate(
                    zip(
                        step_result_triangle,
                        strict=True,
                    )
                ):
                    result[sr_i] += sr_t

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
        grp_gates.attrs["len"] = len(self.horizontal_gates)
        for i, (h_g, v_g, d_g) in enumerate(
            zip(
                self.horizontal_gates,
                self.vertical_gates,
                self.diagonal_gates,
                strict=True,
            )
        ):
            grp_gates.create_dataset(
                f"horizontal_gate_{i:d}",
                data=h_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"vertical_gate_{i:d}", data=v_g, compression="gzip", compression_opts=6
            )
            grp_gates.create_dataset(
                f"diagonal_gate_{i:d}", data=d_g, compression="gzip", compression_opts=6
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
            jnp.asarray(grp["gates"][f"horizontal_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        vertical_gates = tuple(
            jnp.asarray(grp["gates"][f"vertical_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        diagonal_gates = tuple(
            jnp.asarray(grp["gates"][f"diagonal_gate_{i:d}"])
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
            real_d=grp.attrs["real_d"],
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )
