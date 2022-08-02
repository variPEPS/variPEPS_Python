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

@dataclass
class Square_Kagome_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Square-Kagome
    structure.

    Args:
      triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to x links of the lattice.
      square_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to x links of the lattice.
      plus_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the plus term.
      real_d (:obj:`int`):
        Physical dimension of a single site before mapping.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        Likely will be 6 for the a single layer structure..
    """

    x_gates: Sequence[jnp.ndarray]
    y_gates: Sequence[jnp.ndarray]
    z_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 2

    def __post_init__(self) -> None:
        if ((
                len(self.x_gates) > 0
                and len(self.y_gates) > 0
                and len(self.x_gates) != len(self.y_gates)
            )
            or (
                len(self.x_gates) > 0
                and len(self.z_gates) > 0
                and len(self.x_gates) != len(self.z_gates)
            )
            or (
                len(self.y_gates) > 0
                and len(self.z_gates) > 0
                and len(self.y_gates) != len(self.z_gates)
            )
        ):
            raise ValueError("Lengths of gate lists mismatch.")

        self._y_tuple = tuple(self.y_gates)
        self._z_tuple = tuple(self.z_gates)

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
            if all(jnp.allclose(g, jnp.real(g)) for g in self.x_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.y_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.z_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(
                max(
                    len(self.x_gates),
                    len(self.y_gates),
                    len(self.z_gates),
                )
            )
        ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # On site x term
                if len(self.x_gates) > 0:
                    onsite_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
                    onsite_tensor_obj = view[0, 0][0][0]

                    step_result_x = calc_one_site_multi_gates(
                        onsite_tensor,
                        onsite_tensor_obj,
                        self.x_gates,
                    )

                    for sr_i, sr in enumerate(step_result_x):
                        result[sr_i] += sr

                # y term
                if len(self.y_gates) > 0:
                    horizontal_tensors_i = view.get_indices((0, slice(0, 2, None)))
                    horizontal_tensors = [
                        peps_tensors[i] for j in horizontal_tensors_i for i in j
                    ]
                    horizontal_tensor_objs = [t for tl in view[0, :2] for t in tl]

                    for ti in range(2):
                        t = horizontal_tensors[ti]
                        horizontal_tensors[ti] = t.reshape(
                            t.shape[0],
                            t.shape[1],
                            self.real_d,
                            self.real_d,
                            t.shape[3],
                            t.shape[4],
                        )

                    density_matrix_left = apply_contraction(
                        "honeycomb_density_matrix_left",
                        [horizontal_tensors[0]],
                        [horizontal_tensor_objs[0]],
                        [],
                        disable_identity_check=True,
                    )

                    density_matrix_right = apply_contraction(
                        "honeycomb_density_matrix_right",
                        [horizontal_tensors[1]],
                        [horizontal_tensor_objs[1]],
                        [],
                        disable_identity_check=True,
                    )

                    step_result_y = _two_site_workhorse(
                        density_matrix_left,
                        density_matrix_right,
                        self._y_tuple,
                        result_type is jnp.float64,
                    )

                    for sr_i, sr in enumerate(step_result_y):
                        result[sr_i] += sr

                # z term
                if len(self.z_gates) > 0:
                    vertical_tensors_i = view.get_indices((slice(0, 2, None), 0))
                    vertical_tensors = [
                        peps_tensors[i] for j in vertical_tensors_i for i in j
                    ]
                    vertical_tensor_objs = [t for tl in view[:2, 0] for t in tl]

                    for ti in range(2):
                        t = vertical_tensors[ti]
                        vertical_tensors[ti] = t.reshape(
                            t.shape[0],
                            t.shape[1],
                            self.real_d,
                            self.real_d,
                            t.shape[3],
                            t.shape[4],
                        )

                    density_matrix_top = apply_contraction(
                        "honeycomb_density_matrix_top",
                        [vertical_tensors[0]],
                        [vertical_tensor_objs[0]],
                        [],
                        disable_identity_check=True,
                    )

                    density_matrix_bottom = apply_contraction(
                        "honeycomb_density_matrix_right",
                        [vertical_tensors[1]],
                        [vertical_tensor_objs[1]],
                        [],
                        disable_identity_check=True,
                    )

                    step_result_z = _two_site_workhorse(
                        density_matrix_top,
                        density_matrix_bottom,
                        self._z_tuple,
                        result_type is jnp.float64,
                    )

                    for sr_i, sr in enumerate(step_result_z):
                        result[sr_i] += sr

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
