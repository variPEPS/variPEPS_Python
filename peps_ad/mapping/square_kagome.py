from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import jit
import jax.util

from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.contractions import apply_contraction
from peps_ad.expectation.model import Expectation_Model
from peps_ad.expectation.one_site import calc_one_site_multi_gates
from peps_ad.expectation.two_sites import _two_site_workhorse
from peps_ad.typing import Tensor

from typing import Sequence, Union, List, Callable, TypeVar, Optional, Tuple


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
      peps_tensor_objs (:term:`sequence` of :obj:`peps_ad.peps.PEPS_Tensor`):
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
      peps_tensor_objs (:term:`sequence` of :obj:`peps_ad.peps.PEPS_Tensor`):
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

    triangle_gates: Sequence[jnp.ndarray]
    square_gates: Sequence[jnp.ndarray]
    plus_gates: Sequence[jnp.ndarray]
    cross_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 6

    def __post_init__(self) -> None:
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

    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
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
                        self._triangle_tuple,
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
                        self._triangle_tuple,
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
