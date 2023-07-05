from dataclasses import dataclass
from functools import partial
from os import PathLike

import h5py

import jax.numpy as jnp
from jax import jit
import jax.util

import peps_ad.config
from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.contractions import apply_contraction, Definitions
from peps_ad.expectation.model import Expectation_Model
from peps_ad.expectation.one_site import calc_one_site_multi_gates
from peps_ad.expectation.two_sites import (
    _two_site_workhorse,
    _two_site_diagonal_workhorse,
)
from peps_ad.typing import Tensor
from peps_ad.mapping import Map_To_PEPS_Model
from peps_ad.utils.random import PEPS_Random_Number_Generator

from peps_ad.mapping.square_kagome import (
    square_kagome_density_matrix_horizontal,
    square_kagome_density_matrix_vertical,
)

from typing import Sequence, Union, List, Callable, TypeVar, Optional, Tuple, Type

T_Maple_Leaf_Map_PESS_To_PEPS = TypeVar(
    "T_Maple_Leaf_Map_PESS_To_PEPS", bound="Maple_Leaf_Map_PESS_To_PEPS"
)


def maple_leaf_density_matrix_diagonal(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the two parts of the diagonal two sites density matrix of the
    mapped maple leaf lattice. Hereby, one can specify which physical indices
    should be open and which ones be traced before contracting the CTM
    structure.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the PEPS tensors used for the density matrix.
      peps_tensor_objs (:term:`sequence` of :obj:`peps_ad.peps.PEPS_Tensor`):
        Sequence of the corresponding PEPS tensor objects.
      open_physical_indices (:obj:`tuple` of two :obj:`tuple` of :obj:`int`):
        Tuple with two tuples consisting of the physical indices which should
        be kept open for the top left and bottom right site.
    Returns:
      :obj:`tuple` of two:obj:`jax.numpy.ndarray`:
        Top left and bottom right part of the density matrix. Can be used for
        the two sites expectation value functions.
    """
    t_top_left, t_bottom_right = peps_tensors
    t_obj_top_left, t_obj_bottom_right = peps_tensor_objs
    top_left_i, bottom_right_i = open_physical_indices

    new_d = round(t_obj_top_left.d ** (1 / 6))

    t_top_left = t_top_left.reshape(
        t_top_left.shape[0],
        t_top_left.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_top_left.shape[3],
        t_top_left.shape[4],
    )
    t_bottom_right = t_bottom_right.reshape(
        t_bottom_right.shape[0],
        t_bottom_right.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_bottom_right.shape[3],
        t_bottom_right.shape[4],
    )

    phys_contraction_i_top_left = list(range(7, 7 + 6 - len(top_left_i)))
    phys_contraction_i_conj_top_left = list(range(7, 7 + 6 - len(top_left_i)))

    for pos, i in enumerate(top_left_i):
        phys_contraction_i_top_left.insert(i - 1, -(pos + 1))
        phys_contraction_i_conj_top_left.insert(i - 1, -len(top_left_i) - (pos + 1))

    contraction_top_left = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "T4"]],
        "network": [
            [
                [4, -2 * len(top_left_i) - 2]
                + phys_contraction_i_top_left
                + [-2 * len(top_left_i) - 5, 3],  # tensor
                [6, -2 * len(top_left_i) - 3]
                + phys_contraction_i_conj_top_left
                + [-2 * len(top_left_i) - 6, 5],  # tensor_conj
                (2, 1),  # C1
                (1, 3, 5, -2 * len(top_left_i) - 4),  # T1
                (-2 * len(top_left_i) - 1, 6, 4, 2),  # T4
            ]
        ],
    }
    Definitions._process_def(contraction_top_left)

    density_top_left = apply_contraction(
        f"maple_leaf_diagonal_top_left_open_{top_left_i}",
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
        custom_definition=contraction_top_left,
    )

    density_top_left = density_top_left.reshape(
        new_d ** len(top_left_i),
        new_d ** len(top_left_i),
        density_top_left.shape[-6],
        density_top_left.shape[-5],
        density_top_left.shape[-4],
        density_top_left.shape[-3],
        density_top_left.shape[-2],
        density_top_left.shape[-1],
    )

    phys_contraction_i_bottom_right = list(range(7, 7 + 6 - len(bottom_right_i)))
    phys_contraction_i_conj_bottom_right = list(range(7, 7 + 6 - len(bottom_right_i)))

    for pos, i in enumerate(bottom_right_i):
        phys_contraction_i_bottom_right.insert(i - 1, -(pos + 1))
        phys_contraction_i_conj_bottom_right.insert(
            i - 1, -len(bottom_right_i) - (pos + 1)
        )

    contraction_bottom_right = {
        "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
        "network": [
            [
                [-2 * len(bottom_right_i) - 6, 3]
                + phys_contraction_i_bottom_right
                + [4, -2 * len(bottom_right_i) - 3],  # tensor
                [-2 * len(bottom_right_i) - 5, 5]
                + phys_contraction_i_conj_bottom_right
                + [6, -2 * len(bottom_right_i) - 2],  # tensor_conj
                (4, 6, 2, -2 * len(bottom_right_i) - 1),  # T2
                (-2 * len(bottom_right_i) - 4, 1, 5, 3),  # T3
                (1, 2),  # C3
            ]
        ],
    }
    Definitions._process_def(contraction_bottom_right)

    density_bottom_right = apply_contraction(
        f"maple_leap_diagonal_bottom_right_open_{bottom_right_i}",
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
        custom_definition=contraction_bottom_right,
    )

    density_bottom_right = density_bottom_right.reshape(
        new_d ** len(bottom_right_i),
        new_d ** len(bottom_right_i),
        density_bottom_right.shape[-6],
        density_bottom_right.shape[-5],
        density_bottom_right.shape[-4],
        density_bottom_right.shape[-3],
        density_bottom_right.shape[-2],
        density_bottom_right.shape[-1],
    )

    return density_top_left, density_bottom_right


@partial(jit, static_argnums=(3, 4))
def _calc_onsite_gate(
    green_gates: Sequence[jnp.ndarray],
    blue_gates: Sequence[jnp.ndarray],
    red_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    Id_other_sites = jnp.eye(d**4)

    for i, (g_e, b_e, r_e) in enumerate(
        jax.util.safe_zip(green_gates, blue_gates, red_gates)
    ):
        green_12 = jnp.kron(g_e, Id_other_sites)

        green_34 = green_12.reshape(d, d, d, d, d, d, d, d, d, d, d, d)
        green_34 = green_34.transpose((2, 3, 0, 1, 4, 5, 8, 9, 6, 7, 10, 11))
        green_34 = green_34.reshape(d**6, d**6)

        green_56 = jnp.kron(Id_other_sites, g_e)

        blue_base = jnp.kron(b_e, Id_other_sites)
        blue_base = blue_base.reshape(d, d, d, d, d, d, d, d, d, d, d, d)

        blue_15 = blue_base.transpose((0, 2, 3, 4, 1, 5, 6, 8, 9, 10, 7, 11))
        blue_15 = blue_15.reshape(d**6, d**6)

        blue_23 = blue_base.transpose((2, 0, 1, 3, 4, 5, 8, 6, 7, 9, 10, 11))
        blue_23 = blue_23.reshape(d**6, d**6)

        blue_46 = blue_base.transpose((2, 3, 4, 0, 5, 1, 8, 9, 10, 6, 11, 7))
        blue_46 = blue_46.reshape(d**6, d**6)

        red_base = jnp.kron(r_e, Id_other_sites)
        red_base = red_base.reshape(d, d, d, d, d, d, d, d, d, d, d, d)

        red_24 = red_base.transpose((2, 0, 3, 1, 4, 5, 8, 6, 9, 7, 10, 11))
        red_24 = red_24.reshape(d**6, d**6)

        red_25 = red_base.transpose((2, 0, 3, 4, 1, 5, 8, 6, 9, 10, 7, 11))
        red_25 = red_25.reshape(d**6, d**6)

        red_45 = red_base.transpose((2, 3, 4, 0, 1, 5, 8, 9, 10, 6, 7, 11))
        red_45 = red_45.reshape(d**6, d**6)

        result[i] = (
            green_12
            + green_34
            + green_56
            + blue_15
            + blue_23
            + blue_46
            + red_24
            + red_25
            + red_45
        )

    return result


@partial(jit, static_argnums=(2, 3))
def _calc_right_gate(
    blue_gates: Sequence[jnp.ndarray],
    red_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    Id_other_site = jnp.eye(d)

    for i, (b_e, r_e) in enumerate(jax.util.safe_zip(blue_gates, red_gates)):
        red_61 = jnp.kron(r_e, Id_other_site)

        blue_62 = jnp.kron(b_e, Id_other_site)
        blue_62 = blue_62.reshape(d, d, d, d, d, d)
        blue_62 = blue_62.transpose((0, 2, 1, 3, 5, 4))
        blue_62 = blue_62.reshape(d**3, d**3)

        result[i] = red_61 + blue_62

    return result


@partial(jit, static_argnums=(2, 3))
def _calc_down_gate(
    blue_gates: Sequence[jnp.ndarray],
    red_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    Id_other_site = jnp.eye(d)

    for i, (b_e, r_e) in enumerate(jax.util.safe_zip(blue_gates, red_gates)):
        blue_35 = jnp.kron(b_e, Id_other_site)

        red_36 = jnp.kron(r_e, Id_other_site)
        red_36 = red_36.reshape(d, d, d, d, d, d)
        red_36 = red_36.transpose((0, 2, 1, 3, 5, 4))
        red_36 = red_36.reshape(d**3, d**3)

        result[i] = blue_35 + red_36

    return result


@partial(jit, static_argnums=(2, 3))
def _calc_diagonal_gate(
    blue_gates: Sequence[jnp.ndarray],
    red_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    Id_other_site = jnp.eye(d)

    for i, (b_e, r_e) in enumerate(jax.util.safe_zip(blue_gates, red_gates)):
        blue_41 = jnp.kron(Id_other_site, b_e)

        red_31 = jnp.kron(r_e, Id_other_site)
        red_31 = red_31.reshape(d, d, d, d, d, d)
        red_31 = red_31.transpose((0, 2, 1, 3, 5, 4))
        red_31 = red_31.reshape(d**3, d**3)

        result[i] = blue_41 + red_31

    return result


@dataclass
class Maple_Leaf_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Square-Kagome
    structure.

    Args:
      triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the triangles.
      square_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the squares.
      plus_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the plus term.
      cross_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the cross term.
      real_d (:obj:`int`):
        Physical dimension of a single site before mapping.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        Likely will be 6 for the a single layer structure..
    """

    green_gates: Sequence[jnp.ndarray]
    blue_gates: Sequence[jnp.ndarray]
    red_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 6

    def __post_init__(self) -> None:
        if (len(self.green_gates) != len(self.blue_gates)) or (
            len(self.green_gates) != len(self.red_gates)
        ):
            raise ValueError("Lengths of gate lists mismatch.")

        self._full_onsite_tuple = tuple(
            _calc_onsite_gate(
                self.green_gates,
                self.blue_gates,
                self.red_gates,
                self.real_d,
                len(self.green_gates),
            )
        )

        self._right_tuple = tuple(
            _calc_right_gate(
                self.blue_gates,
                self.red_gates,
                self.real_d,
                len(self.blue_gates),
            )
        )

        self._down_tuple = tuple(
            _calc_down_gate(
                self.blue_gates,
                self.red_gates,
                self.real_d,
                len(self.blue_gates),
            )
        )

        self._diagonal_tuple = tuple(
            _calc_diagonal_gate(
                self.blue_gates,
                self.red_gates,
                self.real_d,
                len(self.blue_gates),
            )
        )

        self._result_type = (
            jnp.float64
            if all(jnp.allclose(g, jnp.real(g)) for g in self.green_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.blue_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.red_gates)
            else jnp.complex128
        )

    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result = [
            jnp.array(0, dtype=self._result_type) for _ in range(len(self.green_gates))
        ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # On site term
                if len(self.green_gates) > 0:
                    onsite_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
                    onsite_tensor_obj = view[0, 0][0][0]

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
                    ) = square_kagome_density_matrix_horizontal(
                        horizontal_tensors, horizontal_tensor_objs, ((6,), (1, 2))
                    )

                    step_result_horizontal = _two_site_workhorse(
                        density_matrix_left,
                        density_matrix_right,
                        self._right_tuple,
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
                    ) = square_kagome_density_matrix_vertical(
                        vertical_tensors, vertical_tensor_objs, ((3,), (5, 6))
                    )

                    step_result_vertical = _two_site_workhorse(
                        density_matrix_top,
                        density_matrix_bottom,
                        self._down_tuple,
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
                        density_matrix_bottom_right,
                    ) = maple_leaf_density_matrix_diagonal(
                        (diagonal_tensors[0], diagonal_tensors[3]),
                        (diagonal_tensor_objs[0], diagonal_tensor_objs[3]),
                        ((3, 4), (1,)),
                    )

                    traced_density_matrix_top_right = apply_contraction(
                        "ctmrg_top_right",
                        [diagonal_tensors[1]],
                        [diagonal_tensor_objs[1]],
                        [],
                    )

                    traced_density_matrix_bottom_left = apply_contraction(
                        "ctmrg_bottom_left",
                        [diagonal_tensors[2]],
                        [diagonal_tensor_objs[2]],
                        [],
                    )

                    step_result_diagonal = _two_site_diagonal_workhorse(
                        density_matrix_top_left,
                        density_matrix_bottom_right,
                        traced_density_matrix_top_right,
                        traced_density_matrix_bottom_left,
                        self._diagonal_tuple,
                        self._result_type is jnp.float64,
                    )

                    for sr_i, (sr_o, sr_h, sr_v, sr_d) in enumerate(
                        jax.util.safe_zip(
                            step_result_onsite,
                            step_result_horizontal,
                            step_result_vertical,
                            step_result_diagonal,
                        )
                    ):
                        result[sr_i] += sr_o + sr_h + sr_v + sr_d

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


@dataclass
class Maple_Leaf_Map_PESS_To_PEPS(Map_To_PEPS_Model):
    """
    Map a iPESS structure of the Maple Leaf to a PEPS unitcell.
    The convention for the input tensors is:

    Convention for physical site tensors:
    * t1: [physical, down simplex, up simplex]
    * t2: [physical, down simplex, up simplex]
    * t3: [physical, up simplex, down simplex]

    Convention for simplex tensors:
    * up: [t1, t2, t3]
    * down: [t3, t2, t1]

    Args:
      unitcell_structure (:term:`sequence` of :term:`sequence` of :obj:`int` or 2d array):
        Two dimensional array modeling the structure of the unit cell. For
        details see the description of :obj:`~peps_ad.peps.PEPS_Unit_Cell`.
      chi (:obj:`int`):
        Bond dimension of environment tensors which should be used for the
        unit cell generated.
    """

    unitcell_structure: Sequence[Sequence[int]]
    chi: int

    @staticmethod
    def _map_single_structure(
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        t3: jnp.ndarray,
        up: jnp.ndarray,
        down: jnp.ndarray,
    ):
        tensor = apply_contraction(
            "maple_leaf_pess_mapping",
            [],
            [],
            [t1, t2, t3, up, down],
        )

        return tensor.reshape(
            t1.shape[1],
            t2.shape[1],
            t1.shape[0] * t2.shape[0] * t3.shape[0],
            down.shape[2],
            down.shape[1],
        )

    def __call__(
        self,
        input_tensors: Sequence[jnp.ndarray],
        *,
        generate_unitcell: bool = True,
    ) -> Union[List[jnp.ndarray], Tuple[List[jnp.ndarray], PEPS_Unit_Cell]]:
        num_peps_sites = len(input_tensors) // 5
        if num_peps_sites * 5 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a maple leaf simplex system."
            )

        peps_tensors = [
            self._map_single_structure(*(input_tensors[(i * 5) : (i * 5 + 5)]))
            for i in range(num_peps_sites)
        ]

        if generate_unitcell:
            peps_tensor_objs = [
                PEPS_Tensor.from_tensor(
                    i,
                    i.shape[2],
                    (i.shape[0], i.shape[1], i.shape[3], i.shape[4]),
                    self.chi,
                )
                for i in peps_tensors
            ]
            unitcell = PEPS_Unit_Cell.from_tensor_list(
                peps_tensor_objs, self.unitcell_structure
            )

            return peps_tensors, unitcell

        return peps_tensors

    @classmethod
    def random(
        cls: Type[T_Maple_Leaf_Map_PESS_To_PEPS],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Maple_Leaf_Map_PESS_To_PEPS]:
        structure_arr = jnp.asarray(structure)

        structure_arr, tensors_i = PEPS_Unit_Cell._check_structure(structure_arr)

        # Check the inputs
        if not isinstance(d, int):
            raise ValueError("d has to be a single integer.")

        if not isinstance(D, int):
            raise ValueError("D has to be a single integer.")

        if not isinstance(chi, int):
            raise ValueError("chi has to be a single integer.")

        # Generate the PEPS tensors
        if destroy_random_state:
            PEPS_Random_Number_Generator.destroy_state()

        rng = PEPS_Random_Number_Generator.get_generator(seed, backend="jax")

        result_tensors = []

        for i in tensors_i:
            result_tensors.append(rng.block((d, D, D), dtype=dtype))  # dimer 1
            result_tensors.append(rng.block((d, D, D), dtype=dtype))  # dimer 2
            result_tensors.append(rng.block((d, D, D), dtype=dtype))  # dimer 3
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # up
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # down

        return result_tensors, cls(unitcell_structure=structure, chi=chi)

    @classmethod
    def save_to_file(
        cls: Type[T_Maple_Leaf_Map_PESS_To_PEPS],
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
    ) -> None:
        """
        Save unit cell to a HDF5 file.

        This function creates a single group "maple_leaf_pess" in the file
        and pass this group to the method
        :obj:`~Maple_Leaf_Map_PESS_To_PEPS.save_to_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the new file. Caution: The file will overwritten if existing.
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        with h5py.File(path, "w", libver=("earliest", "v110")) as f:
            grp = f.create_group("maple_leaf_pess")

            cls.save_to_group(grp, tensors, unitcell, store_config=store_config)

    @staticmethod
    def save_to_group(
        grp: h5py.Group,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
    ) -> None:
        """
        Save unit cell to a HDF5 group which is be passed to the method.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to store the data into.
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        num_peps_sites = len(tensors) // 5
        if num_peps_sites * 5 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a maple leaf simplex system."
            )

        grp_pess = grp.create_group("pess_tensors", track_order=True)
        grp_pess.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (t1, t2, t3, up, down) = tensors[(i * 5) : (i * 5 + 5)]

            grp_pess.create_dataset(
                f"site{i}_t1", data=t1, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t2", data=t2, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t3", data=t3, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_up",
                data=up,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_down",
                data=down,
                compression="gzip",
                compression_opts=6,
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Maple_Leaf_Map_PESS_To_PEPS],
        path: PathLike,
        *,
        return_config: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, peps_ad.config.PEPS_AD_Config],
    ]:
        """
        Load unit cell from a HDF5 file.

        This function read the group "maple_leaf_pess" from the file and pass
        this group to the method
        :obj:`~Maple_Leaf_Map_PESS_To_PEPS.load_from_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the HDF5 file.
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`~peps_ad.config.PEPS_AD_Config`):
            The tuple with the list of the PESS tensors and the PEPS unitcell
            is returned. If ``return_config = True``. the config is returned
            as well.
        """
        with h5py.File(path, "r") as f:
            try:
                out = cls.load_from_group(f["maple_leaf_pess"], return_config=return_config)
            except KeyError:
                out = cls.load_from_group(f["maple_lead_pess"], return_config=return_config)

        if return_config:
            return out[0], out[1], out[2]

        return out[0], out[1]

    @staticmethod
    def load_from_group(
        grp: h5py.Group,
        *,
        return_config: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, peps_ad.config.PEPS_AD_Config],
    ]:
        """
        Load the unit cell from a HDF5 group which is be passed to the method.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`~peps_ad.config.PEPS_AD_Config`):
            The tuple with the list of the PESS tensors and the PEPS unitcell
            is returned. If ``return_config = True``. the config is returned
            as well.
        """
        grp_pess = grp["pess_tensors"]
        num_peps_sites = grp_pess.attrs["num_peps_sites"]

        tensors = []

        for i in range(num_peps_sites):
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t1"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t2"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t3"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_up"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_down"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Maple_Leaf_Map_PESS_To_PEPS],
        filename: PathLike,
        tensors: jnp.ndarray,
        unitcell: PEPS_Unit_Cell,
    ) -> None:
        cls.save_to_file(filename, tensors, unitcell)
