"""
PEPS unit cell
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from os import PathLike
import subprocess

import h5py
import numpy as np

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .tensor import PEPS_Tensor
import peps_ad
from peps_ad.utils.random import PEPS_Random_Number_Generator
from peps_ad.utils.periodic_indices import calculate_periodic_indices
from peps_ad import peps_ad_config
import peps_ad.config

from typing import (
    TypeVar,
    Type,
    Union,
    Optional,
    Sequence,
    Tuple,
    List,
    Any,
    Iterator,
    Set,
)
from peps_ad.typing import Tensor, is_int

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

    sanity_checks: bool = True

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

        peps_tensors: List[PEPS_Tensor]

        structure: Tuple[Tuple[int, ...], ...]

        def copy(self) -> PEPS_Unit_Cell.Unit_Cell_Data:
            """
            Creates a (flat) copy of the unit cell data.

            Returns:
              ~peps_ad.peps.PEPS_Unit_Cell.Unit_Cell_Data:
                New instance of the unit cell data class.
            """
            return type(self)(
                peps_tensors=[i for i in self.peps_tensors],
                structure=self.structure,
            )

        def replace_peps_tensors(
            self, new_peps_tensors: List[PEPS_Tensor]
        ) -> PEPS_Unit_Cell.Unit_Cell_Data:
            """
            Return new instance with the list of peps tensors replaced.

            Returns:
              ~peps_ad.peps.PEPS_Unit_Cell.Unit_Cell_Data:
                New instance of the unit cell data class with the list replaced.
            """
            return type(self)(
                peps_tensors=new_peps_tensors,
                structure=self.structure,
            )

        def save_to_group(self, grp: h5py.Group) -> None:
            """
            Store the unit cell data into a HDF5 group.

            Args:
              grp (:obj:`h5py.Group`):
                HDF5 group object to save the data into.
            """
            grp.create_dataset("structure", data=jnp.asarray(self.structure))

            grp_tensors = grp.create_group("peps_tensors", track_order=True)
            grp_tensors.attrs["len"] = len(self.peps_tensors)

            for ti, t in enumerate(self.peps_tensors):
                grp_ti = grp_tensors.create_group(f"t_{ti:d}")
                t.save_to_group(grp_ti)

        @classmethod
        def load_from_group(
            cls: Type[PEPS_Unit_Cell.Unit_Cell_Data], grp: h5py.Group
        ) -> PEPS_Unit_Cell.Unit_Cell_Data:
            """
            Load the unit cell data from a HDF5 group.

            Args:
              grp (:obj:`h5py.Group`):
                HDF5 group object to load the data from.
            """
            structure = tuple(tuple(int(j) for j in i) for i in grp["structure"])
            peps_tensors = [
                PEPS_Tensor.load_from_group(grp["peps_tensors"][f"t_{ti:d}"])
                for ti in range(grp["peps_tensors"].attrs["len"])
            ]

            return cls(structure=structure, peps_tensors=peps_tensors)

        def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            return ((self.peps_tensors,), (self.structure,))

        @classmethod
        def tree_unflatten(
            cls: Type[PEPS_Unit_Cell.Unit_Cell_Data],
            aux_data: Tuple[Any, ...],
            children: Tuple[Any, ...],
        ) -> PEPS_Unit_Cell.Unit_Cell_Data:
            (peps_tensors,) = children
            (structure,) = aux_data
            return cls(structure=structure, peps_tensors=peps_tensors)

    def __post_init__(self):
        if not self.sanity_checks or self.data.structure is None:
            return

        self._check_structure(self.data.structure)

        unit_cell_len_x = len(self.data.structure)
        unit_cell_len_y = len(self.data.structure[0])

        if (
            not is_int(self.real_ix)
            or self.real_ix < 0
            or self.real_ix >= unit_cell_len_x
        ):
            raise ValueError("Invalid value for real x index.")

        if (
            not is_int(self.real_iy)
            or self.real_iy < 0
            or self.real_iy >= unit_cell_len_y
        ):
            raise ValueError("Invalid value for real x index.")

        if not all(
            self.data.peps_tensors[0].chi == i.chi for i in self.data.peps_tensors[1:]
        ):
            raise ValueError("CTMRG bond dimension has to be the same for all tensors.")

    @staticmethod
    def _check_structure(
        structure: Tuple[Tuple[int, ...], ...]
    ) -> Tuple[Tuple[Tuple[int, ...], ...], jnp.ndarray]:
        structure = np.array(structure)

        if structure.ndim != 2 or not np.issubdtype(structure.dtype, np.integer):
            raise ValueError("Structure is not a 2-d int array.")

        tensors_i = np.unique(structure)
        min_i = np.min(tensors_i)

        if min_i < 0 or min_i > 2:
            raise ValueError(
                "Found negative index or minimal index > 1 in unit cell structure"
            )

        if min_i == 1:
            tensors_i = tensors_i - 1
            structure = structure - 1  # type: ignore

        max_i = np.max(tensors_i)

        if max_i != (tensors_i.size - 1):
            raise ValueError("Indices in structure seem to be non-monotonous.")

        return tuple(tuple(int(j) for j in i) for i in structure), tensors_i

    @classmethod
    def from_tensor_list(
        cls: Type[T_PEPS_Unit_Cell],
        tensor_list: Sequence[PEPS_Tensor],
        structure: Union[Sequence[Sequence[int]], Tensor],
    ) -> T_PEPS_Unit_Cell:
        structure, tensors_i = cls._check_structure(structure)

        if tensors_i.size != len(tensor_list):
            raise ValueError("Structure and tensor list mismatch.")

        data = cls.Unit_Cell_Data(peps_tensors=list(tensor_list), structure=structure)

        return cls(data=data)

    @classmethod
    def random(
        cls: Type[T_PEPS_Unit_Cell],
        structure: Union[Sequence[Sequence[int]], Tensor],
        d: Union[int, Sequence[int]],
        D: Union[int, Sequence[Sequence[int]]],
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
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
          max_chi (:obj:`int`):
            Maximal bond dimension for the environment tensors
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
        structure, tensors_i = cls._check_structure(structure)

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
                PEPS_Tensor.random(
                    d=d[i], D=D[i], chi=chi[i], dtype=dtype, seed=seed, max_chi=max_chi
                )
            )

        data = cls.Unit_Cell_Data(peps_tensors=peps_tensors, structure=structure)

        return cls(data=data)

    def get_size(self) -> Tuple[int, int]:
        """
        Returns the size of the unit cell as tuple (x_size, y_size).

        Returns:
          :obj:`tuple`\ (:obj:`int`, :obj:`int`):
            Size of the unit cell as tuple (x_size, y_size).
        """
        return (len(self.data.structure), len(self.data.structure[0]))

    def get_len_unique_tensors(self) -> int:
        """
        Return the length of the list of the unique tensors the unit cell
        consists of.

        Returns:
          :obj:`int`:
            Length of list of unique tensors
        """
        return len(self.data.peps_tensors)

    def get_indices(
        self, key: Tuple[Union[int, slice], Union[int, slice]]
    ) -> Sequence[Sequence[int]]:
        """
        Get indices of PEPS tensors according to unit cell structure.

        Args:
          key (:obj:`tuple` of 2 :obj:`int` or :obj:`slice` elements):
            x and y coordinates to select. Can be either integers or slices.
            Negative numbers as selectors are supported.
        Returns:
          :term:`sequence` of :term:`sequence` of :obj:`~PEPS_Tensor`:
            2d sequence with the indices of the selected PEPS tensor objects.
        """
        return calculate_periodic_indices(
            key, self.data.structure, self.real_ix, self.real_iy
        )

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
        indices = self.get_indices(key)

        return [[self.data.peps_tensors[ty] for ty in tx] for tx in indices]

    def __setitem__(self, key: Tuple[int, int], value: PEPS_Tensor) -> None:
        """
        Set PEPS tensors according to unit cell structure.

        Args:
          key (:obj:`tuple` of 2 :obj:`int` elements):
            x and y coordinates to select. Have to be a single integer.
          value (:obj:`~peps_ad.peps.PEPS_Tensor`):
            New PEPS tensor object to be set.
        """
        if not isinstance(value, PEPS_Tensor):
            raise TypeError("Invalid type for value. Expected PEPS_Tensor.")

        x, y = key

        unit_cell_len_x = len(self.data.structure)
        unit_cell_len_y = len(self.data.structure[0])

        x = (self.real_ix + x) % unit_cell_len_x
        y = (self.real_iy + y) % unit_cell_len_y

        self.data.peps_tensors[self.data.structure[x][y]] = value

    def get_unique_tensors(self) -> List[PEPS_Tensor]:
        """
        Get the list of unique tensors the unit cell consists of.

        Returns:
          :obj:`list` of :obj:`~peps_ad.peps.PEPS_Tensor`:
            List of unique tensors.
        """
        return self.data.peps_tensors

    def replace_unique_tensors(
        self: T_PEPS_Unit_Cell, new_unique_tensors: List[PEPS_Tensor]
    ) -> T_PEPS_Unit_Cell:
        """
        Replace the list of unique tensors the unit cell consists of.

        Args:
          new_unique_tensors (:obj:`list` of :obj:`~peps_ad.peps.PEPS_Tensor`):
            New list of unique tensors the unit cell should consists of.
        Returns:
          PEPS_Unit_Cell:
            New instance of PEPS unit cell with the new unique tensor list.
        """
        if len(self.data.peps_tensors) != len(new_unique_tensors):
            raise ValueError("Length of old and new list mismatches.")

        new_data = self.data.replace_peps_tensors(new_unique_tensors)

        return type(self)(
            data=new_data,
            real_ix=self.real_ix,
            real_iy=self.real_iy,
            sanity_checks=False,
        )

    def change_chi(self: T_PEPS_Unit_Cell, new_chi: int) -> T_PEPS_Unit_Cell:
        """
        Change environment bond dimension of all tensors in the unit cell.

        Args:
          new_chi (:obj:`int`):
            New value for the environment bond dimension.
        Returns:
          PEPS_Unit_Cell:
            New instance of PEPS unit cell with the new tensor list.
        """
        new_unique_tensors = type(self.data.peps_tensors)(
            t.change_chi(new_chi) for t in self.data.peps_tensors
        )

        new_data = self.data.replace_peps_tensors(new_unique_tensors)

        return type(self)(
            data=new_data,
            real_ix=self.real_ix,
            real_iy=self.real_iy,
            sanity_checks=False,
        )

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

        unit_cell_len_x = len(self.data.structure)
        unit_cell_len_y = len(self.data.structure[0])

        return type(self)(
            data=self.data,
            real_ix=(self.real_ix + new_xi) % unit_cell_len_x,
            real_iy=(self.real_iy + new_yi) % unit_cell_len_y,
            sanity_checks=False,
        )

    def _iter_one_column_impl(
        self,
        fixed_y: int,
        *,
        only_unique: bool = False,
        unique_memory: Optional[Set[int]] = None,
    ):
        unit_cell_len_x = len(self.data.structure)

        if unique_memory is None:
            unique_memory = set()

        for x in range(unit_cell_len_x):
            view = self.move(x, fixed_y)

            if only_unique:
                ind_up_left = int(view.get_indices((0, 0))[0][0])
                if ind_up_left in unique_memory:
                    continue
                else:
                    unique_memory.add(ind_up_left)

            yield x, view

    def iter_one_column(
        self: T_PEPS_Unit_Cell, fixed_y: int, *, only_unique: bool = False
    ) -> Iterator[Tuple[int, T_PEPS_Unit_Cell]]:
        """
        Get a iterator over a single column with a fixed y value.

        Args:
          fixed_y (int):
            Fixed y value.
        Keyword args:
          only_unique (bool):
            Return only views where each unique PEPS tensor in the unitcell
            is only once at index (0, 0).
        Returns:
          :term:`iterator`\ (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
            Iterator over one column of the unit cell with the current PEPS
            tensor at moved position (0, 0).
        """
        return self._iter_one_column_impl(fixed_y, only_unique=only_unique)

    def iter_all_columns(
        self: T_PEPS_Unit_Cell, *, reverse: bool = False, only_unique: bool = False
    ) -> Iterator[Tuple[int, Iterator[Tuple[int, T_PEPS_Unit_Cell]]]]:
        """
        Get a iterator over all columns.

        This function calls :obj:`~peps_ad.peps.PEPS_Unit_Cell.iter_one_column`
        for all columns and yields the resulting iterator.

        Keyword args:
          reverse (bool):
            Reverse the order of the iteration.
          only_unique (bool):
            Return only views where each unique PEPS tensor in the unitcell
            is only once at index (0, 0).
        Returns:
          :term:`iterator`\ (:term:`iterator`\ (:obj:`~peps_ad.peps.PEPS_Unit_Cell`)):
            Iterator for all columns over the iterator for single column with
            the current PEPS tensor at moved position (0, 0).
        """
        unit_cell_len_y = len(self.data.structure[0])

        if reverse:
            yiter = range(unit_cell_len_y - 1, -1, -1)
        else:
            yiter = range(unit_cell_len_y)

        unique_memory: Set[int] = set()

        for y in yiter:
            yield y, self._iter_one_column_impl(
                y, only_unique=only_unique, unique_memory=unique_memory
            )

    def _iter_one_row_impl(
        self,
        fixed_x: int,
        *,
        only_unique: bool = False,
        unique_memory: Optional[Set[int]] = None,
    ):
        unit_cell_len_y = len(self.data.structure[0])

        if unique_memory is None:
            unique_memory = set()

        for y in range(unit_cell_len_y):
            view = self.move(fixed_x, y)

            if only_unique:
                ind_up_left = int(view.get_indices((0, 0))[0][0])
                if ind_up_left in unique_memory:
                    continue
                else:
                    unique_memory.add(ind_up_left)

            yield y, view

    def iter_one_row(
        self: T_PEPS_Unit_Cell, fixed_x: int, *, only_unique: bool = False
    ) -> Iterator[Tuple[int, T_PEPS_Unit_Cell]]:
        """
        Get a iterator over a single row with a fixed x value.

        Args:
          fixed_x (int):
            Fixed x value.
        Keyword args:
          only_unique (bool):
            Return only views where each unique PEPS tensor in the unitcell
            is only once at index (0, 0).
        Returns:
          :term:`iterator`\ (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
            Iterator over one row of the unit cell with the current PEPS
            tensor at moved position (0, 0).
        """
        return self._iter_one_row_impl(fixed_x, only_unique=only_unique)

    def iter_all_rows(
        self: T_PEPS_Unit_Cell, *, reverse: bool = False, only_unique: bool = False
    ) -> Iterator[Tuple[int, Iterator[Tuple[int, T_PEPS_Unit_Cell]]]]:
        """
        Get a iterator over all rows.

        This function calls :obj:`~peps_ad.peps.PEPS_Unit_Cell.iter_one_row`
        for all rows and yields the resulting iterator.

        Keyword args:
          reverse (bool):
            Reverse the order of the iteration.
          only_unique (bool):
            Return only views where each unique PEPS tensor in the unitcell
            is only once at index (0, 0).
        Returns:
          :term:`iterator`\ (:term:`iterator`\ (:obj:`~peps_ad.peps.PEPS_Unit_Cell`)):
            Iterator for all rows over the iterator for single row with
            the current PEPS tensor at moved position (0, 0).
        """
        unit_cell_len_x = len(self.data.structure)

        if reverse:
            xiter = range(unit_cell_len_x - 1, -1, -1)
        else:
            xiter = range(unit_cell_len_x)

        unique_memory: Set[int] = set()

        for x in xiter:
            yield x, self._iter_one_row_impl(
                x, only_unique=only_unique, unique_memory=unique_memory
            )

    def copy(self: T_PEPS_Unit_Cell) -> T_PEPS_Unit_Cell:
        """
        Performs a (flat) copy of the PEPS unit cell.

        Returns:
          ~peps_ad.peps.PEPS_Unit_Cell:
            Copied instance of the unit cell.
        """
        return type(self)(
            data=self.data.copy(),
            real_ix=self.real_ix,
            real_iy=self.real_iy,
            sanity_checks=False,
        )

    def save_to_file(self, path: PathLike, store_config: bool = True) -> None:
        """
        Save unit cell to a HDF5 file.

        This function creates a single group "unitcell" in the file and pass
        this group to the method :obj:`~PEPS_Unit_Cell.save_to_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the new file. Caution: The file will overwritten if existing.
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        with h5py.File(path, "w", libver=("earliest", "v110")) as f:
            grp = f.create_group("unitcell")

            self.save_to_group(grp, store_config)

    def save_to_group(self, grp: h5py.Group, store_config: bool = True) -> None:
        """
        Save unit cell to a HDF5 group which is be passed to the method.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to store the data into.
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        grp.attrs["real_ix"] = self.real_ix
        grp.attrs["real_iy"] = self.real_iy

        grp_data = grp.create_group("data")

        self.data.save_to_group(grp_data)

        if store_config:
            grp_config = grp.create_group("config")

            for config_attr in peps_ad_config.__dataclass_fields__.keys():
                grp_config.attrs[config_attr] = getattr(peps_ad_config, config_attr)

            grp_version = grp.create_group("version")
            grp_version.attrs["version"] = peps_ad.__version__

            try:
                git_hash = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).parent
                    )
                    .decode("ascii")
                    .strip()
                )
                grp_version.attrs["git_hash"] = git_hash
            except subprocess.CalledProcessError:
                pass

            try:
                git_tag = (
                    subprocess.check_output(
                        ["git", "describe", "--exact-match", "--tags", "HEAD"],
                        cwd=pathlib.Path(__file__).parent,
                    )
                    .decode("ascii")
                    .strip()
                )
                grp_version.attrs["git_tag"] = git_tag
            except subprocess.CalledProcessError:
                pass

    @classmethod
    def load_from_file(
        cls: Type[T_PEPS_Unit_Cell], path: PathLike, return_config: bool = False
    ) -> Union[
        T_PEPS_Unit_Cell, Tuple[T_PEPS_Unit_Cell, peps_ad.config.PEPS_AD_Config]
    ]:
        """
        Load unit cell from a HDF5 file.

        This function read the group "unitcell" from the file and pass
        this group to the method :obj:`~PEPS_Unit_Cell.load_from_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the HDF5 file.
          return_config (:obj:`bool`):
            Return a config object initialized with the values from the HDF5
            files. If no config is stored in the file, just the data is returned.
            Missing config flags in the file uses the default values from the
            config object.
        """
        with h5py.File(path, "r") as f:
            out = cls.load_from_group(f["unitcell"], return_config)

        if return_config:
            return out[0], out[1]

        return out

    @classmethod
    def load_from_group(
        cls: Type[T_PEPS_Unit_Cell], grp: h5py.Group, return_config: bool = False
    ) -> Union[
        T_PEPS_Unit_Cell, Tuple[T_PEPS_Unit_Cell, peps_ad.config.PEPS_AD_Config]
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
        """
        data = cls.Unit_Cell_Data.load_from_group(grp["data"])
        real_ix = int(grp.attrs["real_ix"])
        real_iy = int(grp.attrs["real_iy"])

        if return_config:
            if grp.get("config") is None:
                return cls(data=data, real_ix=real_ix, real_iy=real_iy), None

            config_dict = {
                config_attr: grp["config"].attrs.get(config_attr)
                for config_attr in peps_ad_config.__dataclass_fields__.keys()
                if grp["config"].attrs.get(config_attr) is not None
            }
            if config_dict.get("ctmrg_full_projector_method"):
                config_dict[
                    "ctmrg_full_projector_method"
                ] = peps_ad.config.Projector_Method(
                    config_dict["ctmrg_full_projector_method"]
                )
            if config_dict.get("optimizer_method"):
                config_dict["optimizer_method"] = peps_ad.config.Optimizing_Methods(
                    config_dict["optimizer_method"]
                )
            if config_dict.get("line_search_method"):
                config_dict["line_search_method"] = peps_ad.config.Line_Search_Methods(
                    config_dict["line_search_method"]
                )

            return cls(
                data=data, real_ix=real_ix, real_iy=real_iy
            ), peps_ad.config.PEPS_AD_Config(**config_dict)

        return cls(data=data, real_ix=real_ix, real_iy=real_iy)

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        aux_data = (self.real_ix, self.real_iy)

        return ((self.data,), aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_PEPS_Unit_Cell],
        aux_data: Tuple[Any, ...],
        children: Tuple[Any, ...],
    ) -> T_PEPS_Unit_Cell:
        real_ix, real_iy = aux_data
        (data,) = children

        return cls(
            data=data,
            real_ix=real_ix,
            real_iy=real_iy,
            sanity_checks=False,
        )
