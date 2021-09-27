"""
PEPS unit cell
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .tensor import PEPS_Tensor
from peps_ad.utils.random import PEPS_Random_Number_Generator

from typing import TypeVar, Type, Union, Optional, Sequence, Tuple, List, Any
from peps_ad.typing import Tensor

T_PEPS_Unit_Cell = TypeVar("T_PEPS_Unit_Cell", bound="PEPS_Unit_Cell")


@dataclass
@register_pytree_node_class
class PEPS_Unit_Cell:
    """
    Class to model a unit cell of a PEPS structure.

    The structure of the unit cell is modeled by a two dimensional array with
    the first index corresponding to the x-axis and the second one to the y-axis.
    The array consists of integer labeling the different unique peps tensors
    in one unit cell (either starting with 0 or 1).
    For example for two rows with a AB/BA structure::

      structure = [[0, 1],
                   [1, 0]]

    Args:
      data (:obj:`Unit_Cell_Data`):
        Instance of unit cell data class
      real_ix (:obj:`int`):
        Which real x index of data.structure correspond to x = 0 of this unit
        cell instance.
      real_iy (:obj:`int`):
        Which real y index of data.structure correspond to y = 0 of this unit
        cell instance.
    """

    data: PEPS_Unit_Cell.Unit_Cell_Data

    real_ix: int = 0
    real_iy: int = 0

    @dataclass
    @register_pytree_node_class
    class Unit_Cell_Data:
        """
        Class to encapsulate the data of the unit cell which can be shared by
        more than one instance.

        Args:
          peps_tensors (:term:`sequence` of :obj:`PEPS_Tensor`):
            Sequence with the unique peps tensors
          structure (2d :obj:`jax.numpy.ndarray`):
            Two dimensional array modeling the structure of the unit cell. For
            details see the description of the parent unit cell class.
        """

        peps_tensors: Sequence[PEPS_Tensor]

        structure: jnp.ndarray

        def tree_flatten(self) -> Tuple[Tuple[...], Tuple[...]]:
            field_names = tuple(self.__dataclass_fields__.keys())
            field_values = tuple(getattr(self, name) for name in field_names)

            return (field_values, field_names)

        @classmethod
        def tree_unflatten(
            cls: Type[PEPS_Unit_Cell._Unit_Cell_Data],
            aux_data: Tuple[...],
            children: Sequence[Any],
        ) -> PEPS_Unit_Cell._Unit_Cell_Data:
            return cls(**dict(jax.util.safe_zip(aux_data, children)))

    def __post_init__(self):
        self._check_structure(self.data.structure)

        unit_cell_len_x = self.data.structure.shape[0]
        unit_cell_len_y = self.data.structure.shape[1]

        if (
            not isinstance(self.real_ix, int)
            or self.real_ix < 0
            or self.real_ix >= unit_cell_len_x
        ):
            raise ValueError("Invalid value for real x index.")

        if (
            not isinstance(self.real_iy, int)
            or self.real_iy < 0
            or self.real_iy >= unit_cell_len_y
        ):
            raise ValueError("Invalid value for real x index.")

    @staticmethod
    def _check_structure(structure: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if structure.ndim != 2 or not jnp.issubdtype(structure.dtype, jnp.integer):
            raise ValueError("Structure is not a 2-d int array.")

        tensors_i = jnp.unique(structure)
        min_i = jnp.min(tensors_i)

        if min_i < 0 or min_i > 2:
            raise ValueError(
                "Found negative index or minimal index > 1 in unit cell structure"
            )

        if min_i == 1:
            tensors_i = tensors_i - 1
            structure = structure - 1  # type: ignore

        max_i = jnp.max(tensors_i)

        if max_i != (tensors_i.size - 1):
            raise ValueError("Indices in structure seem to be non-monotonous.")

        return structure, tensors_i

    @classmethod
    def random(
        cls: Type[T_PEPS_Unit_Cell],
        structure: Union[Sequence[Sequence[int]], Tensor],
        d: Union[int, Sequence[int]],
        D: Union[int, Sequence[Sequence[int]]],
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True
    ) -> T_PEPS_Unit_Cell:
        """
        Randomly initialize the unit cell and its PEPS tensors according to the
        structure given.

        Args:
          structure (:term:`sequence` of :term:`sequence` of :obj:`int` or 2d array):
            Two dimensional array modeling the structure of the unit cell. For
            details see the description of the parent unit cell class.
          d (:obj:`int` or :term:`sequence` of :obj:`int`):
            Physical dimension. If sequence, physical of each unique PEPS
            tensor separately.
          D (:obj:`int` or :term:`sequence` of :term:`sequence` of :obj:`int`):
            Bond dimension of PEPS tensor. If sequence, dimension of each unique
            PEPS tensor separately.
          chi (:obj:`int` or :term:`sequence` of :obj:`int`):
            Bond dimension of environment tensors. If sequence, dimension of
            each unique PEPS tensor separately.
          dtype (:term:`type` of :obj:`jax.numpy.number`):
            Data type of the PEPS tensors.
        Keyword args:
          seed (:obj:`int`, optional):
            Seed for random number generator
          destroy_random_state (:obj:`bool`):
            Destroy state of random number generator and reinitialize it.
            Defaults to True.
        Returns:
          PEPS_Unit_Cell:
            New instance of PEPS unit cell with the initialized tensors.
        """
        structure_arr = jnp.asarray(structure)

        structure_arr, tensors_i = cls._check_structure(structure_arr)

        # Check the inputs
        if isinstance(d, int):
            d = [d] * tensors_i.size

        if not all(isinstance(i, int) for i in d) or len(d) != tensors_i.size:
            raise ValueError(
                "d has to be a single integer or a sequence of ints for every unique tensor of the unit cell."
            )

        if isinstance(D, int):
            D = [(D, D, D, D) for _ in range(tensors_i.size)]

        if (
            not all(isinstance(j, int) for i in D for j in i)
            or len(D) != tensors_i.size
        ):
            raise ValueError(
                "D has to be a single integer or a sequence of a int sequence for every unique tensor of the unit cell."
            )

        if isinstance(chi, int):
            chi = [chi] * tensors_i.size

        if not all(isinstance(i, int) for i in chi) or len(chi) != tensors_i.size:
            raise ValueError(
                "chi has to be a single integer or a sequence of ints for every unique tensor of the unit cell."
            )

        # Generate the PEPS tensors
        if destroy_random_state:
            PEPS_Random_Number_Generator.destroy_state()

        peps_tensors = []

        for i in tensors_i:
            if i > 0:
                seed = None

            peps_tensors.append(
                PEPS_Tensor.random(d=d[i], D=D[i], chi=chi[i], dtype=dtype, seed=seed)
            )

        data = cls.Unit_Cell_Data(peps_tensors=peps_tensors, structure=structure_arr)

        return cls(data=data)

    def __getitem__(
        self, key: Tuple[Union[int, slice], Union[int, slice]]
    ) -> List[List[PEPS_Tensor]]:
        """
        Get PEPS tensors according to unit cell structure.

        Args:
          key (:obj:`tuple` of 2 :obj:`int` or :obj:`slice` elements):
            x and y coordinates to select. Can be either integers or slices.
            Negative numbers as selectors are supported.
        Returns:
          :obj:`list` of :obj:`list` of :obj:`PEPS_Tensor`:
            2d list with the selected PEPS tensor objects.
        """
        if not isinstance(key, tuple) or not len(key) == 2:
            raise TypeError("Expect a tuple with coordinates x and y.")

        x, y = key

        if isinstance(x, int):
            x = slice(x, x + 1, None)
        if isinstance(y, int):
            y = slice(y, y + 1, None)

        if not isinstance(x, slice) or not isinstance(y, slice):
            raise TypeError("Key elements have to be integers or slices.")

        unit_cell_len_x = self.data.structure.shape[0]
        unit_cell_len_y = self.data.structure.shape[1]

        if x.start is not None and x.start < 0:
            shift = (-x.start // unit_cell_len_x + 1) * unit_cell_len_x
            x = slice(x.start + shift, x.stop + shift, x.step)

        if x.start is None and x.stop < 0:
            x = slice(None, x.stop + unit_cell_len_x, x.step)

        if y.start is not None and y.start < 0:
            shift = (-y.start // unit_cell_len_y + 1) * unit_cell_len_y
            y = slice(y.start + shift, y.stop + shift, y.step)

        if y.start is None and y.stop < 0:
            y = slice(None, y.stop + unit_cell_len_y, y.step)

        xarr = jnp.arange(
            x.start or 0, x.stop if x.stop is not None else unit_cell_len_x, x.step
        )
        yarr = jnp.arange(
            y.start or 0, y.stop if y.stop is not None else unit_cell_len_y, y.step
        )

        xarr = (self.real_ix + xarr) % unit_cell_len_x
        yarr = (self.real_iy + yarr) % unit_cell_len_y

        indices = (self.data.structure[xi, yarr] for xi in xarr)

        return [[self.data.peps_tensors[ty] for ty in tx] for tx in indices]

    def move(self: T_PEPS_Unit_Cell, new_xi: int, new_yi: int) -> T_PEPS_Unit_Cell:
        """
        Move origin of the unit cell coordination system.

        This function just creates a new view but do not copy the structure
        object or the PEPS tensors.

        Args:
          new_xi (:obj:`int`):
            New x origin coordinate relative to current origin
          new_yi (:obj:`int`):
            New y origin coordinate relative to current origin
        Returns:
          PEPS_Unit_Cell:
            PEPS unit cell with shifted origin.
        """
        if not isinstance(new_xi, int) or not isinstance(new_yi, int):
            raise ValueError("New indices have to be integers.")

        unit_cell_len_x = self.data.structure.shape[0]
        unit_cell_len_y = self.data.structure.shape[1]

        return type(self)(
            data=self.data,
            real_ix=(self.real_ix + new_xi) % unit_cell_len_x,
            real_iy=(self.real_iy + new_yi) % unit_cell_len_y,
        )

    def tree_flatten(self) -> Tuple[Tuple[...], Tuple[...]]:
        field_names = tuple(self.__dataclass_fields__.keys())
        field_values = tuple(getattr(self, name) for name in field_names)

        return (field_values, field_names)

    @classmethod
    def tree_unflatten(
        cls: T_PEPS_Unit_Cell, aux_data: Tuple[...], children: Sequence[Any]
    ) -> T_PEPS_Unit_Cell:
        return cls(**dict(jax.util.safe_zip(aux_data, children)))
