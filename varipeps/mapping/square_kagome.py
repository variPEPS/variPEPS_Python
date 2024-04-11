from dataclasses import dataclass
from functools import partial
from os import PathLike

import h5py

import jax.numpy as jnp
from jax import jit
import jax.util

from varipeps import varipeps_config
import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, Definitions
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.one_site import calc_one_site_multi_gates
from varipeps.expectation.two_sites import _two_site_workhorse
from varipeps.expectation.spiral_helpers import apply_unitary
from varipeps.typing import Tensor
from varipeps.mapping import Map_To_PEPS_Model
from varipeps.utils.random import PEPS_Random_Number_Generator

from typing import (
    Sequence,
    Union,
    List,
    Callable,
    TypeVar,
    Optional,
    Tuple,
    Type,
    Dict,
    Any,
)

T_Square_Kagome_Map_PESS_To_PEPS = TypeVar(
    "T_Square_Kagome_Map_PESS_To_PEPS", bound="Square_Kagome_Map_PESS_To_PEPS"
)
T_Square_Kagome_Map_4_1_1_To_PEPS = TypeVar(
    "T_Square_Kagome_Map_4_1_1_To_PEPS", bound="Square_Kagome_Map_4_1_1_To_PEPS"
)


def square_kagome_density_matrix_horizontal(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the two parts of the horizontal two sites density matrix of the
    mapped Square Kagome lattice. Hereby, one can specify which physical indices
    should be open and which ones be traced before contracting the CTM
    structure.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the PEPS tensors used for the density matrix.
      peps_tensor_objs (:term:`sequence` of :obj:`varipeps.peps.PEPS_Tensor`):
        Sequence of the corresponding PEPS tensor objects.
      open_physical_indices (:obj:`tuple` of two :obj:`tuple` of :obj:`int`):
        Tuple with two tuples consisting of the physical indices which should
        be kept open for the left and right site.
    Returns:
      :obj:`tuple` of two:obj:`jax.numpy.ndarray`:
        Left and right part of the density matrix. Can be used for the
        two sites expectation value functions.
    """
    t_left, t_right = peps_tensors
    t_obj_left, t_obj_right = peps_tensor_objs
    left_i, right_i = open_physical_indices

    new_d = round(t_obj_left.d ** (1 / 6))

    t_left = t_left.reshape(
        t_left.shape[0],
        t_left.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_left.shape[3],
        t_left.shape[4],
    )
    t_right = t_right.reshape(
        t_right.shape[0],
        t_right.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_right.shape[3],
        t_right.shape[4],
    )

    if len(left_i) == 0:
        raise NotImplementedError
    elif len(left_i) == 1:
        phys_contraction_i_left = [5, 6, 7, 8, 9]
        phys_contraction_i_left.insert(left_i[0] - 1, -1)
        phys_contraction_i_conj_left = [5, 6, 7, 8, 9]
        phys_contraction_i_conj_left.insert(left_i[0] - 1, -2)

        contraction_left = {
            "tensors": [["tensor", "tensor_conj", "C1", "T1", "T3", "C4", "T4"]],
            "network": [
                [
                    [10, 12] + phys_contraction_i_left + [-4, 14],  # tensor
                    [11, 13] + phys_contraction_i_conj_left + [-5, 15],  # tensor_conj
                    (4, 1),  # C1
                    (1, 14, 15, -3),  # T1
                    (3, -6, 13, 12),  # T3
                    (3, 2),  # C4
                    (2, 11, 10, 4),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_left, f"square_kagome_horizontal_left_open_{left_i[0]}"
        )

        density_left = apply_contraction(
            f"square_kagome_horizontal_left_open_{left_i[0]}",
            [t_left],
            [t_obj_left],
            [],
            disable_identity_check=True,
            custom_definition=contraction_left,
        )
    else:
        phys_contraction_i_left = list(range(11, 11 + 6 - len(left_i)))
        phys_contraction_i_conj_left = list(range(11, 11 + 6 - len(left_i)))

        for pos, i in enumerate(left_i):
            phys_contraction_i_left.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_left.insert(i - 1, -len(left_i) - (pos + 1))

        contraction_left = {
            "tensors": [["tensor", "tensor_conj", "C1", "T1", "T3", "C4", "T4"]],
            "network": [
                [
                    [4, 5]
                    + phys_contraction_i_left
                    + [-2 * len(left_i) - 2, 7],  # tensor
                    [8, 9]
                    + phys_contraction_i_conj_left
                    + [-2 * len(left_i) - 3, 10],  # tensor_conj
                    (6, 1),  # C1
                    (1, 7, 10, -2 * len(left_i) - 1),  # T1
                    (3, -2 * len(left_i) - 4, 9, 5),  # T3
                    (3, 2),  # C4
                    (2, 8, 4, 6),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_left, f"square_kagome_horizontal_left_open_{left_i}"
        )

        density_left = apply_contraction(
            f"square_kagome_horizontal_left_open_{left_i}",
            [t_left],
            [t_obj_left],
            [],
            disable_identity_check=True,
            custom_definition=contraction_left,
        )

        density_left = density_left.reshape(
            new_d ** len(left_i),
            new_d ** len(left_i),
            density_left.shape[-4],
            density_left.shape[-3],
            density_left.shape[-2],
            density_left.shape[-1],
        )

    if len(right_i) == 0:
        raise NotImplementedError
    elif len(right_i) == 1:
        phys_contraction_i_right = [5, 6, 7, 8, 9]
        phys_contraction_i_right.insert(right_i[0] - 1, -5)
        phys_contraction_i_conj_right = [5, 6, 7, 8, 9]
        phys_contraction_i_conj_right.insert(right_i[0] - 1, -6)

        contraction_right = {
            "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2", "T3", "C3"]],
            "network": [
                [
                    [-2, 12] + phys_contraction_i_right + [10, 14],  # tensor
                    [-3, 13] + phys_contraction_i_conj_right + [11, 15],  # tensor_conj
                    (-1, 14, 15, 1),  # T1
                    (1, 4),  # C2
                    (10, 11, 2, 4),  # T2
                    (-4, 3, 13, 12),  # T3
                    (3, 2),  # C3
                ]
            ],
        }
        Definitions._process_def(
            contraction_right, f"square_kagome_horizontal_right_open_{right_i[0]}"
        )

        density_right = apply_contraction(
            f"square_kagome_horizontal_right_open_{right_i[0]}",
            [t_right],
            [t_obj_right],
            [],
            disable_identity_check=True,
            custom_definition=contraction_right,
        )
    else:
        phys_contraction_i_right = list(range(11, 11 + 6 - len(right_i)))
        phys_contraction_i_conj_right = list(range(11, 11 + 6 - len(right_i)))

        for pos, i in enumerate(right_i):
            phys_contraction_i_right.insert(i - 1, -4 - (pos + 1))
            phys_contraction_i_conj_right.insert(i - 1, -4 - len(right_i) - (pos + 1))

        contraction_right = {
            "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2", "T3", "C3"]],
            "network": [
                [
                    [-2, 5] + phys_contraction_i_right + [4, 7],  # tensor
                    [-3, 9] + phys_contraction_i_conj_right + [8, 10],  # tensor_conj
                    (-1, 7, 10, 1),  # T1
                    (1, 6),  # C2
                    (4, 8, 2, 6),  # T2
                    (-4, 3, 9, 5),  # T3
                    (3, 2),  # C3
                ]
            ],
        }
        Definitions._process_def(
            contraction_right, f"square_kagome_horizontal_right_open_{right_i}"
        )

        density_right = apply_contraction(
            f"square_kagome_horizontal_right_open_{right_i}",
            [t_right],
            [t_obj_right],
            [],
            disable_identity_check=True,
            custom_definition=contraction_right,
        )

        density_right = density_right.reshape(
            density_right.shape[0],
            density_right.shape[1],
            density_right.shape[2],
            density_right.shape[3],
            new_d ** len(right_i),
            new_d ** len(right_i),
        )

    return density_left, density_right


def square_kagome_density_matrix_vertical(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the two parts of the vertical two sites density matrix of the
    mapped Square Kagome lattice. Hereby, one can specify which physical indices
    should be open and which ones be traced before contracting the CTM
    structure.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the PEPS tensors used for the density matrix.
      peps_tensor_objs (:term:`sequence` of :obj:`varipeps.peps.PEPS_Tensor`):
        Sequence of the corresponding PEPS tensor objects.
      open_physical_indices (:obj:`tuple` of two :obj:`tuple` of :obj:`int`):
        Tuple with two tuples consisting of the physical indices which should
        be kept open for the top and bottom site.
    Returns:
      :obj:`tuple` of two:obj:`jax.numpy.ndarray`:
        Top and bottom part of the density matrix. Can be used for the
        two sites expectation value functions.
    """
    t_top, t_bottom = peps_tensors
    t_obj_top, t_obj_bottom = peps_tensor_objs
    top_i, bottom_i = open_physical_indices

    new_d = round(peps_tensor_objs[0].d ** (1 / 6))

    t_top = t_top.reshape(
        t_top.shape[0],
        t_top.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_top.shape[3],
        t_top.shape[4],
    )
    t_bottom = t_bottom.reshape(
        t_bottom.shape[0],
        t_bottom.shape[1],
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        new_d,
        t_bottom.shape[3],
        t_bottom.shape[4],
    )

    if len(top_i) == 0:
        raise NotImplementedError
    elif len(top_i) == 1:
        phys_contraction_i_top = [5, 6, 7, 8, 9]
        phys_contraction_i_top.insert(top_i[0] - 1, -1)
        phys_contraction_i_conj_top = [5, 6, 7, 8, 9]
        phys_contraction_i_conj_top.insert(top_i[0] - 1, -2)

        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "T4"]],
            "network": [
                [
                    [12, -4] + phys_contraction_i_top + [14, 10],  # tensor
                    [13, -5] + phys_contraction_i_conj_top + [15, 11],  # tensor_conj
                    (3, 2),  # C1
                    (2, 10, 11, 4),  # T1
                    (4, 1),  # C2
                    (14, 15, -6, 1),  # T2
                    (-3, 13, 12, 3),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_top, f"square_kagome_horizontal_top_open_{top_i[0]}"
        )

        density_top = apply_contraction(
            f"square_kagome_horizontal_top_open_{top_i[0]}",
            [t_top],
            [t_obj_top],
            [],
            disable_identity_check=True,
            custom_definition=contraction_top,
        )
    else:
        phys_contraction_i_top = list(range(11, 11 + 6 - len(top_i)))
        phys_contraction_i_conj_top = list(range(11, 11 + 6 - len(top_i)))

        for pos, i in enumerate(top_i):
            phys_contraction_i_top.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_top.insert(i - 1, -len(top_i) - (pos + 1))

        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "T4"]],
            "network": [
                [
                    [5, -2 * len(top_i) - 2]
                    + phys_contraction_i_top
                    + [7, 4],  # tensor
                    [9, -2 * len(top_i) - 3]
                    + phys_contraction_i_conj_top
                    + [10, 8],  # tensor_conj
                    (3, 2),  # C1
                    (2, 4, 8, 6),  # T1
                    (6, 1),  # C2
                    (7, 10, -2 * len(top_i) - 4, 1),  # T2
                    (-2 * len(top_i) - 1, 9, 5, 3),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_top, f"square_kagome_horizontal_top_open_{top_i}"
        )

        density_top = apply_contraction(
            f"square_kagome_horizontal_top_open_{top_i}",
            [t_top],
            [t_obj_top],
            [],
            disable_identity_check=True,
            custom_definition=contraction_top,
        )

        density_top = density_top.reshape(
            new_d ** len(top_i),
            new_d ** len(top_i),
            density_top.shape[-4],
            density_top.shape[-3],
            density_top.shape[-2],
            density_top.shape[-1],
        )

    if len(bottom_i) == 0:
        raise NotImplementedError
    elif len(bottom_i) == 1:
        phys_contraction_i_bottom = [5, 6, 7, 8, 9]
        phys_contraction_i_bottom.insert(bottom_i[0] - 1, -5)
        phys_contraction_i_conj_bottom = [5, 6, 7, 8, 9]
        phys_contraction_i_conj_bottom.insert(bottom_i[0] - 1, -6)

        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T2", "C3", "T3", "C4", "T4"]],
            "network": [
                [
                    [14, 10] + phys_contraction_i_bottom + [12, -2],  # tensor
                    [15, 11] + phys_contraction_i_conj_bottom + [13, -3],  # tensor_conj
                    (12, 13, 3, -4),  # T2
                    (2, 3),  # C3
                    (4, 2, 11, 10),  # T3
                    (4, 1),  # C4
                    (1, 15, 14, -1),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_bottom, f"square_kagome_horizontal_bottom_open_{bottom_i[0]}"
        )

        density_bottom = apply_contraction(
            f"square_kagome_horizontal_bottom_open_{bottom_i[0]}",
            [t_bottom],
            [t_obj_bottom],
            [],
            disable_identity_check=True,
            custom_definition=contraction_bottom,
        )
    else:
        phys_contraction_i_bottom = list(range(11, 11 + 6 - len(bottom_i)))
        phys_contraction_i_conj_bottom = list(range(11, 11 + 6 - len(bottom_i)))

        for pos, i in enumerate(bottom_i):
            phys_contraction_i_bottom.insert(i - 1, -4 - (pos + 1))
            phys_contraction_i_conj_bottom.insert(i - 1, -4 - len(bottom_i) - (pos + 1))

        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T2", "C3", "T3", "C4", "T4"]],
            "network": [
                [
                    [7, 4] + phys_contraction_i_bottom + [5, -2],  # tensor
                    [10, 8] + phys_contraction_i_conj_bottom + [9, -3],  # tensor_conj
                    (5, 9, 3, -4),  # T2
                    (2, 3),  # C3
                    (6, 2, 8, 4),  # T3
                    (6, 1),  # C4
                    (1, 10, 7, -1),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_bottom, f"square_kagome_horizontal_bottom_open_{bottom_i}"
        )

        density_bottom = apply_contraction(
            f"square_kagome_horizontal_bottom_open_{bottom_i}",
            [t_bottom],
            [t_obj_bottom],
            [],
            disable_identity_check=True,
            custom_definition=contraction_bottom,
        )

        density_bottom = density_bottom.reshape(
            density_bottom.shape[0],
            density_bottom.shape[1],
            density_bottom.shape[2],
            density_bottom.shape[3],
            new_d ** len(bottom_i),
            new_d ** len(bottom_i),
        )

    return density_top, density_bottom


@partial(jit, static_argnums=(2, 3))
def _calc_onsite_gate(
    triangle_gates: Sequence[jnp.ndarray],
    square_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    Id_triangle = jnp.eye(d**3)
    Id_square = jnp.eye(d)

    for i, e in enumerate(triangle_gates):
        gate_triangle_456 = jnp.kron(Id_triangle, e)

        gate_triangle_124 = gate_triangle_456.reshape(
            d, d, d, d, d, d, d, d, d, d, d, d
        )
        gate_triangle_124 = gate_triangle_124.transpose(
            (3, 4, 0, 5, 1, 2, 9, 10, 6, 11, 7, 8)
        )
        gate_triangle_124 = gate_triangle_124.reshape(d**6, d**6)

        result[i] = gate_triangle_124 + gate_triangle_456

    for i, e in enumerate(square_gates):
        gate_square = jnp.kron(jnp.kron(Id_square, e), Id_square)

        if result[i] is None:
            result[i] = gate_square
        else:
            result[i] += gate_square

    return result


@dataclass
class Square_Kagome_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Square-Kagome
    structure.

    .. figure:: /images/square_kagome_structure.*
       :align: center
       :width: 80%
       :alt: Structure of the square Kagome lattice with smallest possible unit
             cell marked by dashed lines.

       Structure of the square Kagome lattice with smallest possible unit cell
       marked by dashed lines.

    \\

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
        Likely will be 6 for the a single layer structure.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    triangle_gates: Sequence[jnp.ndarray]
    square_gates: Sequence[jnp.ndarray]
    plus_gates: Sequence[jnp.ndarray]
    cross_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 6

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.triangle_gates, jnp.ndarray):
            self.triangle_gates = (self.triangle_gates,)

        if isinstance(self.square_gates, jnp.ndarray):
            self.square_gates = (self.square_gates,)

        if isinstance(self.plus_gates, jnp.ndarray):
            self.plus_gates = (self.plus_gates,)

        if isinstance(self.cross_gates, jnp.ndarray):
            self.cross_gates = (self.cross_gates,)

        if len(self.cross_gates) > 0:
            raise NotImplementedError("Cross term calculation is not implemented yet.")

        if (
            (
                len(self.triangle_gates) > 0
                and len(self.square_gates) > 0
                and len(self.triangle_gates) != len(self.square_gates)
            )
            or (
                len(self.triangle_gates) > 0
                and len(self.plus_gates) > 0
                and len(self.triangle_gates) != len(self.plus_gates)
            )
            or (
                len(self.triangle_gates) > 0
                and len(self.cross_gates) > 0
                and len(self.triangle_gates) != len(self.cross_gates)
            )
            or (
                len(self.square_gates) > 0
                and len(self.plus_gates) > 0
                and len(self.square_gates) != len(self.plus_gates)
            )
            or (
                len(self.square_gates) > 0
                and len(self.cross_gates) > 0
                and len(self.square_gates) != len(self.cross_gates)
            )
            or (
                len(self.plus_gates) > 0
                and len(self.cross_gates) > 0
                and len(self.plus_gates) != len(self.cross_gates)
            )
        ):
            raise ValueError("Lengths of gate lists mismatch.")

        self._full_onsite_gates = _calc_onsite_gate(
            self.triangle_gates,
            self.square_gates,
            self.real_d,
            max(len(self.triangle_gates), len(self.square_gates)),
        )

        self._triangle_tuple = tuple(self.triangle_gates)

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
            if all(jnp.allclose(g, jnp.real(g)) for g in self.triangle_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.square_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.plus_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.cross_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(
                max(
                    len(self.triangle_gates),
                    len(self.square_gates),
                    len(self.plus_gates),
                    len(self.cross_gates),
                )
            )
        ]

        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (spiral_vectors, spiral_vectors, spiral_vectors)
            if len(spiral_vectors) == 1:
                spiral_vectors = (
                    None,
                    spiral_vectors[0],
                    spiral_vectors[0],
                    None,
                    spiral_vectors[0],
                    None,
                )
            if len(spiral_vectors) == 3:
                spiral_vectors = (
                    None,
                    spiral_vectors[0],
                    spiral_vectors[1],
                    None,
                    spiral_vectors[2],
                    None,
                )
            if len(spiral_vectors) != 6:
                raise ValueError("Length mismatch for spiral vectors!")

            working_h_gates = tuple(
                apply_unitary(
                    h,
                    jnp.array((0, 1)),
                    spiral_vectors[1:3],
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self._triangle_tuple
            )
            working_v_gates = tuple(
                apply_unitary(
                    v,
                    jnp.array((1, 0)),
                    spiral_vectors[2:3] + spiral_vectors[4:5],
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for v in self._triangle_tuple
            )
        else:
            working_h_gates = self._triangle_tuple
            working_v_gates = self._triangle_tuple

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # On site term
                if len(self._full_onsite_gates) > 0:
                    onsite_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
                    onsite_tensor_obj = view[0, 0][0][0]

                    step_result_onsite = calc_one_site_multi_gates(
                        onsite_tensor,
                        onsite_tensor_obj,
                        self._full_onsite_gates,
                    )

                    for sr_i, sr in enumerate(step_result_onsite):
                        result[sr_i] += sr

                # Horizontal/Vertical terms
                if len(self.triangle_gates) > 0 and len(self.cross_gates) > 0:
                    raise NotImplementedError
                elif len(self.triangle_gates) > 0:
                    horizontal_tensors_i = view.get_indices((0, slice(0, 2, None)))
                    horizontal_tensors = [
                        peps_tensors[i] for j in horizontal_tensors_i for i in j
                    ]
                    horizontal_tensor_objs = [t for tl in view[0, :2] for t in tl]
                    (
                        density_matrix_left,
                        density_matrix_right,
                    ) = square_kagome_density_matrix_horizontal(
                        horizontal_tensors, horizontal_tensor_objs, ((6,), (2, 3))
                    )

                    step_result_horizontal = _two_site_workhorse(
                        density_matrix_left,
                        density_matrix_right,
                        working_h_gates,
                        result_type is jnp.float64,
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
                        vertical_tensors, vertical_tensor_objs, ((1,), (3, 5))
                    )

                    step_result_vertical = _two_site_workhorse(
                        density_matrix_top,
                        density_matrix_bottom,
                        working_v_gates,
                        result_type is jnp.float64,
                    )

                    for sr_i, (sr_h, sr_v) in enumerate(
                        jax.util.safe_zip(step_result_horizontal, step_result_vertical)
                    ):
                        result[sr_i] += sr_h + sr_v
                elif len(self.cross_gates) > 0:
                    raise NotImplementedError

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
class Square_Kagome_Map_PESS_To_PEPS(Map_To_PEPS_Model):
    """
    Map a iPESS structure of the Square-Kagome to a PEPS unitcell.
    The convention for the input tensors is:

    Convention for physical site tensors:
    * t1: [away from square, phys, bottom simplex]
    * t2: [bottom simplex, phys, left simplex]
    * t3: [left simplex, phys, top simplex]
    * t4: [right simplex, phys, bottom simplex]
    * t5: [top simplex, phys, right simplex]
    * t6: [away from square, phys, right simplex]

    Convention for simplex tensors:
    * left: [away from square, t3, t2]
    * top: [away from square, t5, t3]
    * right: [t6, t4, t5]
    * bottom: [t1, t2, t4]

    Args:
      unitcell_structure (:term:`sequence` of :term:`sequence` of :obj:`int` or 2d array):
        Two dimensional array modeling the structure of the unit cell. For
        details see the description of :obj:`~varipeps.peps.PEPS_Unit_Cell`.
      chi (:obj:`int`):
        Bond dimension of environment tensors which should be used for the
        unit cell generated.
      max_chi (:obj:`int`):
        Maximal allowed bond dimension of environment tensors which should be
        used for the unit cell generated.
    """

    unitcell_structure: Sequence[Sequence[int]]
    chi: int
    max_chi: Optional[int] = None

    @staticmethod
    def _map_single_structure(input_tensors: Sequence[jnp.ndarray]):
        (
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            simplex_left,
            simplex_top,
            simplex_right,
            simplex_bottom,
        ) = input_tensors

        peps_tensor = apply_contraction(
            "square_kagome_pess_mapping",
            [],
            [],
            [
                t1,
                t2,
                t3,
                t4,
                t5,
                t6,
                simplex_left,
                simplex_top,
                simplex_right,
                simplex_bottom,
            ],
        )

        return peps_tensor.reshape(
            peps_tensor.shape[0],
            peps_tensor.shape[1],
            -1,
            peps_tensor.shape[8],
            peps_tensor.shape[9],
        )

    def __call__(
        self,
        input_tensors: Sequence[jnp.ndarray],
        *,
        generate_unitcell: bool = True,
    ) -> Union[List[jnp.ndarray], Tuple[List[jnp.ndarray], PEPS_Unit_Cell]]:
        num_peps_sites = len(input_tensors) // 10
        if num_peps_sites * 10 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome simplex system."
            )

        peps_tensors = [
            self._map_single_structure(input_tensors[(i * 10) : (i * 10 + 10)])
            for i in range(num_peps_sites)
        ]

        if generate_unitcell:
            peps_tensor_objs = [
                PEPS_Tensor.from_tensor(
                    i,
                    i.shape[2],
                    (i.shape[0], i.shape[1], i.shape[3], i.shape[4]),
                    self.chi,
                    self.max_chi,
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
        cls: Type[T_Square_Kagome_Map_PESS_To_PEPS],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: int,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Square_Kagome_Map_PESS_To_PEPS]:
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
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t1
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t2
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t3
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t4
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t5
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t6
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_left
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_top
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_right
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_bottom

        return result_tensors, cls(
            unitcell_structure=structure, chi=chi, max_chi=max_chi
        )

    @classmethod
    def save_to_file(
        cls: Type[T_Square_Kagome_Map_PESS_To_PEPS],
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save Square-Kagome PESS tensors and unit cell to a HDF5 file.

        This function creates a single group "square_kagome_pess" in the file
        and pass this group to the method
        :obj:`~Square_Kagome_Map_PESS_To_PEPS.save_to_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the new file. Caution: The file will overwritten if existing.
          tensors (:obj:`list` of :obj:`jax.numpy.ndarray`):
            List with the PEPS tensors which should be stored in the file.
          unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
            Full unit cell object which should be stored in the file.
        Keyword args:
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
          auxiliary_data (:obj:`dict` with :obj:`str` to storable objects, optional):
            Dictionary with string indexed auxiliary HDF5-storable entries which
            should be stored along the other data in the file.
        """
        with h5py.File(path, "w", libver=("earliest", "v110")) as f:
            grp = f.create_group("square_kagome_pess")

            cls.save_to_group(grp, tensors, unitcell, store_config=store_config)

            if auxiliary_data is not None:
                grp_aux = f.create_group("auxiliary_data")

                grp_aux.attrs["keys"] = list(auxiliary_data.keys())

                for key, val in auxiliary_data.items():
                    if key == "keys":
                        raise ValueError(
                            "Name 'keys' forbidden as name for auxiliary data"
                        )

                    if isinstance(
                        val, (jnp.ndarray, np.ndarray, collections.abc.Sequence)
                    ):
                        try:
                            if val.ndim == 0:
                                val = val.reshape(1)
                        except AttributeError:
                            pass

                        grp_aux.create_dataset(
                            key,
                            data=jnp.asarray(val),
                            compression="gzip",
                            compression_opts=6,
                        )
                    else:
                        grp_aux.attrs[key] = val

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
          tensors (:obj:`list` of :obj:`jax.numpy.ndarray`):
            List with the PEPS tensors which should be stored in the file.
          unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
            Full unit cell object which should be stored in the file.
        Keyword args:
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        num_peps_sites = len(tensors) // 10
        if num_peps_sites * 10 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome simplex system."
            )

        grp_pess = grp.create_group("pess_tensors", track_order=True)
        grp_pess.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (
                t1,
                t2,
                t3,
                t4,
                t5,
                t6,
                simplex_left,
                simplex_top,
                simplex_right,
                simplex_bottom,
            ) = tensors[(i * 10) : (i * 10 + 10)]

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
                f"site{i}_t4", data=t4, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t5", data=t5, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t6", data=t6, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_simplex_left",
                data=simplex_left,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_simplex_top",
                data=simplex_top,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_simplex_right",
                data=simplex_right,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_simplex_bottom",
                data=simplex_bottom,
                compression="gzip",
                compression_opts=6,
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Square_Kagome_Map_PESS_To_PEPS],
        path: PathLike,
        *,
        return_config: bool = False,
        return_auxiliary_data: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load Square-Kagome PESS tensors and unit cell from a HDF5 file.

        This function read the group "square_kagome_pess" from the file and pass
        this group to the method
        :obj:`~Square_Kagome_Map_PESS_To_PEPS.load_from_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the HDF5 file.
        Keyword args:
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
          return_auxiliary_data (:obj:`bool`):
            Return dictionary with string indexed auxiliary data which has been
            should be stored along the other data in the file.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`, :obj:`~varipeps.config.VariPEPS_Config`):
            The tuple with the list of the PESS tensors and the PEPS unitcell
            is returned. If ``return_config = True``. the config is returned
            as well. If ``return_auxiliary_data = True``. the auxiliary data is
            returned as well.
        """
        with h5py.File(path, "r") as f:
            out = cls.load_from_group(
                f["square_kagome_pess"], return_config=return_config
            )

            auxiliary_data = {}
            auxiliary_data_grp = f.get("auxiliary_data")
            if auxiliary_data_grp is not None:
                for k in auxiliary_data_grp.attrs["keys"]:
                    aux_d = auxiliary_data_grp.get(k)
                    if aux_d is None:
                        aux_d = auxiliary_data_grp.attrs[k]
                    else:
                        aux_d = jnp.asarray(aux_d)
                    auxiliary_data[k] = aux_d
            else:
                max_trunc_error_list = f.get("max_trunc_error_list")
                if max_trunc_error_list is not None:
                    auxiliary_data["max_trunc_error_list"] = jnp.asarray(
                        max_trunc_error_list
                    )

        if return_config and return_auxiliary_data:
            return out[0], out[1], out[2], auxiliary_data
        elif return_config:
            return out[0], out[1], out[2]
        elif return_auxiliary_data:
            return out[0], out[1], auxiliary_data

        return out[0], out[1]

    @staticmethod
    def load_from_group(
        grp: h5py.Group,
        *,
        return_config: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load the unit cell from a HDF5 group which is be passed to the method.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
        Keyword args:
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`, :obj:`~varipeps.config.VariPEPS_Config`):
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
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t4"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t5"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t6"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_left"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_top"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_right"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_bottom"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Square_Kagome_Map_PESS_To_PEPS],
        filename: PathLike,
        tensors: jnp.ndarray,
        unitcell: PEPS_Unit_Cell,
        counter: Optional[int] = None,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if counter is not None:
            cls.save_to_file(
                f"{str(filename)}.{counter}",
                tensors,
                unitcell,
                auxiliary_data=auxiliary_data,
            )
        else:
            cls.save_to_file(filename, tensors, unitcell, auxiliary_data=auxiliary_data)


@jit
def _map_4_1_1(input_tensors):
    (t1, t6, square_tensor) = input_tensors

    result = jnp.tensordot(t1, square_tensor, ((2,), (1,)))
    result = jnp.tensordot(result, t6, ((7,), (2,)))
    result = result.transpose(2, 0, 1, 3, 4, 5, 6, 9, 8, 7)
    result = result.reshape(
        result.shape[0], result.shape[1], -1, result.shape[8], result.shape[9]
    )
    return result


@dataclass
class Square_Kagome_Map_4_1_1_To_PEPS(Map_To_PEPS_Model):
    """
    Convention for physical site tensors:
    * t1: [away from square, phys, square tensor]
    * t6: [away from square, phys, square tensot]

    Convention for square tensor: [left, bottom, phys, phys, phys, phys, right, top]

    Args:
      unitcell_structure (:term:`sequence` of :term:`sequence` of :obj:`int` or 2d array):
        Two dimensional array modeling the structure of the unit cell. For
        details see the description of :obj:`~varipeps.peps.PEPS_Unit_Cell`.
      chi (:obj:`int`):
        Bond dimension of environment tensors which should be used for the
        unit cell generated.
      max_chi (:obj:`int`):
        Maximal allowed bond dimension of environment tensors which should be
        used for the unit cell generated.
    """

    unitcell_structure: Sequence[Sequence[int]]
    chi: int
    max_chi: Optional[int] = None

    def __call__(
        self,
        input_tensors: Sequence[jnp.ndarray],
        *,
        generate_unitcell: bool = True,
    ) -> Union[List[jnp.ndarray], Tuple[List[jnp.ndarray], PEPS_Unit_Cell]]:
        num_peps_sites = len(input_tensors) // 3
        if num_peps_sites * 3 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome system."
            )

        peps_tensors = [
            _map_4_1_1(input_tensors[(i * 3) : (i * 3 + 3)])
            for i in range(num_peps_sites)
        ]

        if generate_unitcell:
            peps_tensor_objs = [
                PEPS_Tensor.from_tensor(
                    i,
                    i.shape[2],
                    (i.shape[0], i.shape[1], i.shape[3], i.shape[4]),
                    self.chi,
                    self.max_chi,
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
        cls: Type[T_Square_Kagome_Map_4_1_1_To_PEPS],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: int,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Square_Kagome_Map_4_1_1_To_PEPS]:
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
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t1
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # t6
            result_tensors.append(
                rng.block((D, D, d, d, d, d, D, D), dtype=dtype)
            )  # square_kagome

        return result_tensors, cls(
            unitcell_structure=structure, chi=chi, max_chi=max_chi
        )

    @classmethod
    def save_to_file(
        cls: Type[T_Square_Kagome_Map_4_1_1_To_PEPS],
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save Square-Kagome tensors and unit cell to a HDF5 file.

        This function creates a single group "square_kagome_semi_peps" in the file
        and pass this group to the method
        :obj:`~Square_Kagome_Map_4_1_1_To_PEPS.save_to_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the new file. Caution: The file will overwritten if existing.
          tensors (:obj:`list` of :obj:`jax.numpy.ndarray`):
            List with the PEPS tensors which should be stored in the file.
          unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
            Full unit cell object which should be stored in the file.
        Keyword args:
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
          auxiliary_data (:obj:`dict` with :obj:`str` to storable objects, optional):
            Dictionary with string indexed auxiliary HDF5-storable entries which
            should be stored along the other data in the file.
        """
        with h5py.File(path, "w", libver=("earliest", "v110")) as f:
            grp = f.create_group("square_kagome_semi_peps")

            cls.save_to_group(grp, tensors, unitcell, store_config=store_config)

            if auxiliary_data is not None:
                grp_aux = f.create_group("auxiliary_data")

                grp_aux.attrs["keys"] = list(auxiliary_data.keys())

                for key, val in auxiliary_data.items():
                    if key == "keys":
                        raise ValueError(
                            "Name 'keys' forbidden as name for auxiliary data"
                        )

                    if isinstance(
                        val, (jnp.ndarray, np.ndarray, collections.abc.Sequence)
                    ):
                        try:
                            if val.ndim == 0:
                                val = val.reshape(1)
                        except AttributeError:
                            pass

                        grp_aux.create_dataset(
                            key,
                            data=jnp.asarray(val),
                            compression="gzip",
                            compression_opts=6,
                        )
                    else:
                        grp_aux.attrs[key] = val

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
          tensors (:obj:`list` of :obj:`jax.numpy.ndarray`):
            List with the PEPS tensors which should be stored in the file.
          unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
            Full unit cell object which should be stored in the file.
        Keyword args:
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        num_peps_sites = len(tensors) // 3
        if num_peps_sites * 3 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome semi-PEPS system."
            )

        grp_semi_peps = grp.create_group("semi_peps_tensors", track_order=True)
        grp_semi_peps.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (
                t1,
                t6,
                square,
            ) = tensors[(i * 3) : (i * 3 + 3)]

            grp_semi_peps.create_dataset(
                f"site{i}_t1", data=t1, compression="gzip", compression_opts=6
            )
            grp_semi_peps.create_dataset(
                f"site{i}_t6", data=t6, compression="gzip", compression_opts=6
            )
            grp_semi_peps.create_dataset(
                f"site{i}_square",
                data=square,
                compression="gzip",
                compression_opts=6,
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Square_Kagome_Map_4_1_1_To_PEPS],
        path: PathLike,
        *,
        return_config: bool = False,
        return_auxiliary_data: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load Square-Kagome tensors and unit cell from a HDF5 file.

        This function read the group "square_kagome_semi_peps" from the file and pass
        this group to the method
        :obj:`~Square_Kagome_Map_4_1_1_To_PEPS.load_from_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the HDF5 file.
        Keyword args:
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
          return_auxiliary_data (:obj:`bool`):
            Return dictionary with string indexed auxiliary data which has been
            should be stored along the other data in the file.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`, :obj:`~varipeps.config.VariPEPS_Config`):
            The tuple with the list of the PESS tensors and the PEPS unitcell
            is returned. If ``return_config = True``. the config is returned
            as well.
        """
        with h5py.File(path, "r") as f:
            out = cls.load_from_group(
                f["square_kagome_semi_peps"], return_config=return_config
            )

            auxiliary_data = {}
            auxiliary_data_grp = f.get("auxiliary_data")
            if auxiliary_data_grp is not None:
                for k in auxiliary_data_grp.attrs["keys"]:
                    aux_d = auxiliary_data_grp.get(k)
                    if aux_d is None:
                        aux_d = auxiliary_data_grp.attrs[k]
                    else:
                        aux_d = jnp.asarray(aux_d)
                    auxiliary_data[k] = aux_d
            else:
                max_trunc_error_list = f.get("max_trunc_error_list")
                if max_trunc_error_list is not None:
                    auxiliary_data["max_trunc_error_list"] = jnp.asarray(
                        max_trunc_error_list
                    )

        if return_config and return_auxiliary_data:
            return out[0], out[1], out[2], auxiliary_data
        elif return_config:
            return out[0], out[1], out[2]
        elif return_auxiliary_data:
            return out[0], out[1], auxiliary_data

        return out[0], out[1]

    @staticmethod
    def load_from_group(
        grp: h5py.Group,
        *,
        return_config: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load the unit cell from a HDF5 group which is be passed to the method.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
        Keyword args:
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
        Returns:
          :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`) or :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`, :obj:`~varipeps.config.VariPEPS_Config`):
            The tuple with the list of the PESS tensors and the PEPS unitcell
            is returned. If ``return_config = True``. the config is returned
            as well.
        """
        grp_semi_peps = grp["semi_peps_tensors"]
        num_peps_sites = grp_semi_peps.attrs["num_peps_sites"]

        tensors = []

        for i in range(num_peps_sites):
            tensors.append(jnp.asarray(grp_semi_peps[f"site{i}_t1"]))
            tensors.append(jnp.asarray(grp_semi_peps[f"site{i}_t6"]))
            tensors.append(jnp.asarray(grp_semi_peps[f"site{i}_square"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Square_Kagome_Map_4_1_1_To_PEPS],
        filename: PathLike,
        tensors: jnp.ndarray,
        unitcell: PEPS_Unit_Cell,
        counter: Optional[int] = None,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if counter is not None:
            cls.save_to_file(
                f"{str(filename)}.{counter}",
                tensors,
                unitcell,
                auxiliary_data=auxiliary_data,
            )
        else:
            cls.save_to_file(filename, tensors, unitcell, auxiliary_data=auxiliary_data)
