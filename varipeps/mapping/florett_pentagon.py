from dataclasses import dataclass
from functools import partial
from os import PathLike

import h5py

import jax.numpy as jnp
from jax import jit

import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, Definitions
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.one_site import calc_one_site_multi_gates
from varipeps.expectation.two_sites import (
    _two_site_workhorse,
    _two_site_diagonal_workhorse,
)
from varipeps.expectation.helpers import (
    partially_traced_four_site_density_matrices,
    partially_traced_horizontal_two_site_density_matrices,
    partially_traced_vertical_two_site_density_matrices,
)
from varipeps.typing import Tensor
from varipeps.mapping import Map_To_PEPS_Model
from varipeps.utils.random import PEPS_Random_Number_Generator

from typing import Sequence, Union, List, Callable, TypeVar, Optional, Tuple, Type


@partial(jit, static_argnums=(3, 4))
def _calc_onsite_gate(
    black_gates: Sequence[jnp.ndarray],
    green_gates: Sequence[jnp.ndarray],
    blue_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    Id_other_sites = jnp.eye(d**7)

    for i, (b_e, g_e, blue_e) in enumerate(
        zip(black_gates, green_gates, blue_gates, strict=True)
    ):
        black_12 = jnp.kron(b_e, Id_other_sites)

        black_13 = black_12.reshape(
            d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d
        )
        black_13 = black_13.transpose(
            0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17
        )
        black_13 = black_13.reshape(d**9, d**9)

        blue_36 = jnp.kron(blue_e, Id_other_sites)
        blue_36 = blue_36.reshape(d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d)
        blue_36 = blue_36.transpose(
            2, 3, 0, 4, 5, 1, 6, 7, 8, 11, 12, 9, 13, 14, 10, 15, 16, 17
        )
        blue_36 = blue_36.reshape(d**9, d**9)

        green_base = jnp.kron(g_e, Id_other_sites)
        green_base = green_base.reshape(
            d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d
        )

        green_24 = green_base.transpose(
            2, 0, 3, 1, 4, 5, 6, 7, 8, 11, 9, 12, 10, 13, 14, 15, 16, 17
        )
        green_24 = green_24.reshape(d**9, d**9)

        green_45 = green_base.transpose(
            2, 3, 4, 0, 1, 5, 6, 7, 8, 11, 12, 13, 9, 10, 14, 15, 16, 17
        )
        green_45 = green_45.reshape(d**9, d**9)

        green_46 = green_base.transpose(
            2, 3, 4, 0, 5, 1, 6, 7, 8, 11, 12, 13, 9, 14, 10, 15, 16, 17
        )
        green_46 = green_46.reshape(d**9, d**9)

        green_37 = green_base.transpose(
            2, 3, 0, 4, 5, 6, 1, 7, 8, 11, 12, 9, 13, 14, 15, 10, 16, 17
        )
        green_37 = green_37.reshape(d**9, d**9)

        green_78 = green_base.transpose(
            2, 3, 4, 5, 6, 7, 0, 1, 8, 11, 12, 13, 14, 15, 16, 9, 10, 17
        )
        green_78 = green_78.reshape(d**9, d**9)

        green_79 = green_base.transpose(
            2, 3, 4, 5, 6, 7, 0, 8, 1, 11, 12, 13, 14, 15, 16, 9, 17, 10
        )
        green_79 = green_79.reshape(d**9, d**9)

        result[i] = (
            black_12
            + black_13
            + blue_36
            + green_24
            + green_45
            + green_46
            + green_37
            + green_78
            + green_79
        )

        single_gates[i] = (
            black_12,
            black_13,
            blue_36,
            green_24,
            green_45,
            green_46,
            green_37,
            green_78,
            green_79,
        )

    return result, single_gates


@partial(jit, static_argnums=(2, 3))
def _calc_right_gate(
    black_gates: Sequence[jnp.ndarray],
    blue_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    Id_other_site = jnp.eye(d)

    for i, (b_e, blue_e) in enumerate(zip(black_gates, blue_gates, strict=True)):
        black_51 = jnp.kron(b_e, Id_other_site)

        blue_59 = jnp.kron(blue_e, Id_other_site)
        blue_59 = blue_59.reshape(d, d, d, d, d, d)
        blue_59 = blue_59.transpose((0, 2, 1, 3, 5, 4))
        blue_59 = blue_59.reshape(d**3, d**3)

        result[i] = black_51 + blue_59

        single_gates[i] = (black_51, blue_59)

    return result, single_gates


@partial(jit, static_argnums=(2, 3))
def _calc_down_gate(
    black_gates: Sequence[jnp.ndarray],
    blue_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    Id_other_site = jnp.eye(d**2)

    for i, (b_e, blue_e) in enumerate(zip(black_gates, blue_gates, strict=True)):
        black_91 = jnp.kron(b_e, Id_other_site)
        black_91 = black_91.reshape(d, d, d, d, d, d, d, d)
        black_91 = black_91.transpose(2, 0, 1, 3, 6, 4, 5, 7)
        black_91 = black_91.reshape(d**4, d**4)

        blue_82 = jnp.kron(blue_e, Id_other_site)
        blue_82 = blue_82.reshape(d, d, d, d, d, d, d, d)
        blue_82 = blue_82.transpose((0, 2, 3, 1, 4, 6, 7, 5))
        blue_82 = blue_82.reshape(d**4, d**4)

        result[i] = black_91 + blue_82

        single_gates[i] = (black_91, blue_82)

    return result, single_gates


@partial(jit, static_argnums=(1, 2))
def _calc_diagonal_gate(
    black_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    Id_other_site = jnp.eye(d)

    for i, b_e in enumerate(black_gates):
        black_81 = jnp.kron(Id_other_site, b_e)

        black_61 = black_81.reshape(d, d, d, d, d, d)
        black_61 = black_61.transpose(1, 0, 2, 4, 3, 5)
        black_61 = black_61.reshape(d**3, d**3)

        result[i] = black_61 + black_81

        single_gates[i] = (black_61, black_81)

    return result, single_gates


@dataclass
class Florett_Pentagon_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Florett Pentagon
    structure.

    .. subfigure:: AB
       :layout-sm: A|B
       :align: center
       :width: 100%
       :subcaptions: below
       :gap: 8px

       .. image:: /images/floret_pentagon_structure.*
          :alt: Structure of the floret-pentagon lattice with the smallest possible unit cell

       .. image:: /images/floret_pentagon_bond_colors.*
          :alt: Structure of the floret-pentagon lattice with the bond types marked in color

    \\

    Args:
      black_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the bonds marked in
        black in the image above.
      green_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the bonds marked in
        green in the image above.
      blue_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the bonds marked in
        blue in the image above.
      real_d (:obj:`int`):
        Physical dimension of a single site before mapping.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        Likely will be 9 for the a single layer structure.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    black_gates: Sequence[jnp.ndarray]
    green_gates: Sequence[jnp.ndarray]
    blue_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 9

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.black_gates, jnp.ndarray):
            self.black_gates = (self.black_gates,)

        if isinstance(self.green_gates, jnp.ndarray):
            self.green_gates = (self.green_gates,)

        if isinstance(self.blue_gates, jnp.ndarray):
            self.blue_gates = (self.blue_gates,)

        if (len(self.green_gates) != len(self.blue_gates)) or (
            len(self.green_gates) != len(self.black_gates)
        ):
            raise ValueError("Lengths of gate lists mismatch.")

        tmp_result = tuple(
            _calc_onsite_gate(
                self.black_gates,
                self.green_gates,
                self.blue_gates,
                self.real_d,
                len(self.black_gates),
            )
        )
        self._full_onsite_tuple, self._onsite_single_gates = tuple(
            tmp_result[0]
        ), tuple(tmp_result[1])

        tmp_result = tuple(
            _calc_right_gate(
                self.black_gates,
                self.blue_gates,
                self.real_d,
                len(self.black_gates),
            )
        )
        self._right_tuple, self._right_single_gates = tuple(tmp_result[0]), tuple(
            tmp_result[1]
        )

        tmp_result = tuple(
            _calc_down_gate(
                self.black_gates,
                self.blue_gates,
                self.real_d,
                len(self.black_gates),
            )
        )
        self._down_tuple, self._down_single_gates = tuple(tmp_result[0]), tuple(
            tmp_result[1]
        )

        tmp_result = tuple(
            _calc_diagonal_gate(
                self.black_gates,
                self.real_d,
                len(self.black_gates),
            )
        )
        self._diagonal_tuple, self._diagonal_single_gates = tuple(tmp_result[0]), tuple(
            tmp_result[1]
        )

        self._result_type = (
            jnp.float64
            if all(jnp.allclose(g, jnp.real(g)) for g in self.green_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.blue_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.black_gates)
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
        return_single_gate_results: bool = False,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result = [
            jnp.array(0, dtype=self._result_type) for _ in range(len(self.black_gates))
        ]

        if return_single_gate_results:
            single_gates_result = [dict()] * len(self.black_gates)

            working_onsite_gates = tuple(
                o for e in self._onsite_single_gates for o in e
            )

        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (
                    spiral_vectors,
                    spiral_vectors,
                    spiral_vectors,
                )
            if len(spiral_vectors) == 1:
                spiral_vectors = (
                    spiral_vectors[0],
                    spiral_vectors[0],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    spiral_vectors[0],
                )
            if len(spiral_vectors) == 4:
                spiral_vectors = (
                    spiral_vectors[0],
                    spiral_vectors[1],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    spiral_vectors[2],
                )
            if len(spiral_vectors) != 9:
                raise ValueError("Length mismatch for spiral vectors!")

            working_h_gates = tuple(
                apply_unitary(
                    h,
                    jnp.array((0, 1)),
                    spiral_vectors[0:9:8],
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self._right_tuple
            )
            working_v_gates = tuple(
                apply_unitary(
                    v,
                    jnp.array((1, 0)),
                    spiral_vectors[:2],
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (2, 3),
                    varipeps_config.spiral_wavevector_type,
                )
                for v in self._down_tuple
            )
            working_d_gates = tuple(
                apply_unitary(
                    d,
                    jnp.array((1, 1)),
                    spiral_vectors[:1],
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (2,),
                    varipeps_config.spiral_wavevector_type,
                )
                for d in self._diagonal_tuple
            )

            if return_single_gate_results:
                working_h_single_gates = tuple(
                    apply_unitary(
                        h,
                        jnp.array((0, 1)),
                        spiral_vectors[0:9:8],
                        self._spiral_D,
                        self._spiral_sigma,
                        self.real_d,
                        3,
                        (1, 2),
                        varipeps_config.spiral_wavevector_type,
                    )
                    for e in self._right_single_gates
                    for h in e
                )
                working_v_single_gates = tuple(
                    apply_unitary(
                        v,
                        jnp.array((1, 0)),
                        spiral_vectors[:2],
                        self._spiral_D,
                        self._spiral_sigma,
                        self.real_d,
                        4,
                        (2, 3),
                        varipeps_config.spiral_wavevector_type,
                    )
                    for e in self._down_single_gates
                    for v in e
                )
                working_d_single_gates = tuple(
                    apply_unitary(
                        d,
                        jnp.array((1, 1)),
                        spiral_vectors[:1],
                        self._spiral_D,
                        self._spiral_sigma,
                        self.real_d,
                        3,
                        (2,),
                        varipeps_config.spiral_wavevector_type,
                    )
                    for e in self._diagonal_single_gates
                    for d in e
                )
        else:
            working_h_gates = self._right_tuple
            working_v_gates = self._down_tuple
            working_d_gates = self._diagonal_tuple

            if return_single_gate_results:
                working_h_single_gates = tuple(
                    h for e in self._right_single_gates for h in e
                )
                working_v_single_gates = tuple(
                    v for e in self._down_single_gates for v in e
                )
                working_d_single_gates = tuple(
                    d for e in self._diagonal_single_gates for d in e
                )

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # On site term
                if len(self.black_gates) > 0:
                    onsite_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
                    onsite_tensor_obj = view[0, 0][0][0]

                    if return_single_gate_results:
                        step_result_onsite = calc_one_site_multi_gates(
                            onsite_tensor,
                            onsite_tensor_obj,
                            self._full_onsite_tuple + working_onsite_gates,
                        )
                    else:
                        step_result_onsite = calc_one_site_multi_gates(
                            onsite_tensor,
                            onsite_tensor_obj,
                            self._full_onsite_tuple,
                        )

                    horizontal_tensors_i = view.get_indices((0, slice(0, 2, None)))
                    horizontal_tensors = [
                        peps_tensors[i] for j in horizontal_tensors_i for i in j
                    ]
                    horizontal_tensor_objs = [t for tl in view[0, :2] for t in tl]
                    (
                        density_matrix_left,
                        density_matrix_right,
                    ) = partially_traced_horizontal_two_site_density_matrices(
                        horizontal_tensors,
                        horizontal_tensor_objs,
                        self.real_d,
                        9,
                        ((5,), (1, 9)),
                    )

                    if return_single_gate_results:
                        step_result_horizontal = _two_site_workhorse(
                            density_matrix_left,
                            density_matrix_right,
                            working_h_gates + working_h_single_gates,
                            self._result_type is jnp.float64,
                        )
                    else:
                        step_result_horizontal = _two_site_workhorse(
                            density_matrix_left,
                            density_matrix_right,
                            working_h_gates,
                            self._result_type is jnp.float64,
                        )

                    vertical_tensors_i = view.get_indices((slice(0, 2, None), 0))
                    vertical_tensors = [
                        peps_tensors[i] for j in vertical_tensors_i for i in j
                    ]
                    vertical_tensor_objs = [t for tl in view[:2, 0] for t in tl]
                    (
                        density_matrix_top,
                        density_matrix_bottom,
                    ) = partially_traced_vertical_two_site_density_matrice(
                        vertical_tensors,
                        vertical_tensor_objs,
                        self.real_d,
                        9,
                        ((8, 9), (1, 2)),
                    )

                    if return_single_gate_results:
                        step_result_vertical = _two_site_workhorse(
                            density_matrix_top,
                            density_matrix_bottom,
                            working_v_gates + working_v_single_gates,
                            self._result_type is jnp.float64,
                        )
                    else:
                        step_result_vertical = _two_site_workhorse(
                            density_matrix_top,
                            density_matrix_bottom,
                            working_v_gates,
                            self._result_type is jnp.float64,
                        )

                    diagonal_tensors_i = view.get_indices(
                        (slice(0, 2, None), slice(0, 2, None))
                    )
                    diagonal_tensors = [
                        peps_tensors[i] for j in diagonal_tensors_i for i in j
                    ]
                    diagonal_tensor_objs = [t for tl in view[:2, :2] for t in tl]
                    (
                        density_matrix_top_left,
                        traced_density_matrix_top_right,
                        traced_density_matrix_bottom_left,
                        density_matrix_bottom_right,
                    ) = partially_traced_four_site_density_matrices(
                        diagonal_tensors,
                        diagonal_tensor_objs,
                        self.real_d,
                        9,
                        ((6, 8), (), (), (1,)),
                    )

                    if return_single_gate_results:
                        step_result_diagonal = _two_site_diagonal_workhorse(
                            density_matrix_top_left,
                            density_matrix_bottom_right,
                            traced_density_matrix_top_right,
                            traced_density_matrix_bottom_left,
                            working_d_gates + working_d_single_gates,
                            self._result_type is jnp.float64,
                        )
                    else:
                        step_result_diagonal = _two_site_diagonal_workhorse(
                            density_matrix_top_left,
                            density_matrix_bottom_right,
                            traced_density_matrix_top_right,
                            traced_density_matrix_bottom_left,
                            working_d_gates,
                            self._result_type is jnp.float64,
                        )

                    for sr_i, (sr_o, sr_h, sr_v, sr_d) in enumerate(
                        zip(
                            step_result_onsite[: len(self.black_gates)],
                            step_result_horizontal[: len(self.black_gates)],
                            step_result_vertical[: len(self.black_gates)],
                            step_result_diagonal[: len(self.black_gates)],
                            strict=True,
                        )
                    ):
                        result[sr_i] += sr_o + sr_h + sr_v + sr_d

                    if return_single_gate_results:
                        for sr_i in range(len(self.black_gates)):
                            index_onsite = (
                                len(self.black_gates)
                                + len(self._onsite_single_gates[0]) * sr_i
                            )
                            index_horizontal = (
                                len(self.black_gates)
                                + len(self._right_single_gates[0]) * sr_i
                            )
                            index_vertical = (
                                len(self.black_gates)
                                + len(self._down_single_gates[0]) * sr_i
                            )
                            index_diagonal = (
                                len(self.black_gates)
                                + len(self._diagonal_single_gates[0]) * sr_i
                            )

                            single_gates_result[sr_i][(x, y)] = dict(
                                zip(
                                    (
                                        "black_12",
                                        "black_13",
                                        "blue_36",
                                        "green_24",
                                        "green_45",
                                        "green_46",
                                        "green_37",
                                        "green_78",
                                        "green_79",
                                        "black_51",
                                        "blue_59",
                                        "black_91",
                                        "blue_82",
                                        "black_61",
                                        "black_81",
                                    ),
                                    (
                                        step_result_onsite[
                                            index_onsite : (
                                                index_onsite
                                                + len(self._onsite_single_gates[0])
                                            )
                                        ]
                                        + step_result_horizontal[
                                            index_horizontal : (
                                                index_horizontal
                                                + len(self._right_single_gates[0])
                                            )
                                        ]
                                        + step_result_vertical[
                                            index_vertical : (
                                                index_vertical
                                                + len(self._down_single_gates[0])
                                            )
                                        ]
                                        + step_result_diagonal[
                                            index_diagonal : (
                                                index_diagonal
                                                + len(self._diagonal_single_gates[0])
                                            )
                                        ]
                                    ),
                                )
                            )

        if normalize_by_size:
            if only_unique:
                size = unitcell.get_len_unique_tensors()
            else:
                size = unitcell.get_size()[0] * unitcell.get_size()[1]
            size = size * self.normalization_factor
            result = [r / size for r in result]

        if len(result) == 1:
            result = result[0]

        if return_single_gate_results:
            return result, single_gates_result
        else:
            return result

    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        grp_gates = grp.create_group("gates", track_order=True)
        grp_gates.attrs["len"] = len(self.green_gates)
        for i, (g_g, b_g, bl_g) in enumerate(
            zip(self.green_gates, self.blue_gates, self.black_gates, strict=True)
        ):
            grp_gates.create_dataset(
                f"green_gate_{i:d}",
                data=g_g,
                compression="gzip",
                compression_opts=6,
            )
            grp_gates.create_dataset(
                f"blue_gate_{i:d}", data=b_g, compression="gzip", compression_opts=6
            )
            grp_gates.create_dataset(
                f"black_gate_{i:d}", data=bl_g, compression="gzip", compression_opts=6
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

        green_gates = tuple(
            jnp.asarray(grp["gates"][f"green_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        blue_gates = tuple(
            jnp.asarray(grp["gates"][f"blue_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )
        black_gates = tuple(
            jnp.asarray(grp["gates"][f"black_gate_{i:d}"])
            for i in range(grp["gates"].attrs["len"])
        )

        is_spiral_peps = grp.attrs["is_spiral_peps"]

        if is_spiral_peps:
            spiral_unitary_operator = jnp.asarray(grp["spiral_unitary_operator"])
        else:
            spiral_unitary_operator = None

        return cls(
            green_gates=green_gates,
            blue_gates=blue_gates,
            black_gates=black_gates,
            real_d=grp.attrs["real_d"],
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
        )
