from dataclasses import dataclass
from os import PathLike

import jax.numpy as jnp
from jax import jit
import jax.util
import h5py

import peps_ad.config
from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.contractions import apply_contraction
from peps_ad.expectation.model import Expectation_Model
from peps_ad.expectation.two_sites import (
    calc_two_sites_vertical_multiple_gates,
    calc_two_sites_horizontal_multiple_gates,
    calc_two_sites_diagonal_top_left_bottom_right_multiple_gates,
)
from peps_ad.typing import Tensor
from peps_ad.utils.random import PEPS_Random_Number_Generator
from peps_ad.mapping import Map_To_PEPS_Model

from typing import Sequence, Union, List, Callable, TypeVar, Optional, Tuple, Type

T_Triangular_Map_PESS_To_PEPS = TypeVar(
    "T_Triangular_Map_PESS_To_PEPS", bound="Triangular_Map_PESS_To_PEPS"
)


@dataclass
class Triangular_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped triangular PESS
    structure.

    It is assumed that the simplex tensors are placed in the upper triangular
    and each site tensor is coarse-grained with the simplex tensor sitting
    up right of it.

    Args:
      nearest_neighbor_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each nearest
        neighbor.
      downward_triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the downward
        triangles.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        If for example three sites are mapped into one PEPS site this
        should be 3.
    """

    nearest_neighbor_gates: Sequence[jnp.ndarray]
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

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
        result_type = (
            jnp.float64
            if all(jnp.allclose(g, jnp.real(g)) for g in self.nearest_neighbor_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(len(self.nearest_neighbor_gates))
        ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                x_tensors_i = view.get_indices((slice(0, 2, None), 0))
                x_tensors = [peps_tensors[i] for j in x_tensors_i for i in j]
                x_tensor_objs = [t for tl in view[:2, 0] for t in tl]

                step_result_x = calc_two_sites_vertical_multiple_gates(
                    x_tensors, x_tensor_objs, self.nearest_neighbor_gates
                )

                y_tensors_i = view.get_indices((0, slice(0, 2, None)))
                y_tensors = [peps_tensors[i] for j in y_tensors_i for i in j]
                y_tensor_objs = [t for tl in view[0, :2] for t in tl]

                step_result_y = calc_two_sites_horizontal_multiple_gates(
                    y_tensors, y_tensor_objs, self.nearest_neighbor_gates
                )

                diagonal_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                diagonal_tensors = [
                    peps_tensors[i] for j in diagonal_tensors_i for i in j
                ]
                diagonal_tensor_objs = [t for tl in view[:2, :2] for t in tl]

                step_result_diagonal = (
                    calc_two_sites_diagonal_top_left_bottom_right_multiple_gates(
                        diagonal_tensors,
                        diagonal_tensor_objs,
                        self.nearest_neighbor_gates,
                    )
                )

                for sr_i, (sr_x, sr_y, sr_diagonal) in enumerate(
                    jax.util.safe_zip(
                        step_result_x, step_result_y, step_result_diagonal
                    )
                ):
                    result[sr_i] += sr_x + sr_y + sr_diagonal

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
class Triangular_Map_PESS_To_PEPS(Map_To_PEPS_Model):
    """
    Map a triangular iPESS unit cell to a iPEPS structure.

    The simplex are expected to be in the upper triangle and for the mapping
    to PEPS they are contracted with the site tensor sitting left down of it.

    Convention for physical site tensors: [up left, up right, down, phys]

    Convention for simplex tensors: [up, down left, down right]

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
    max_chi: Optional[int] = None

    @staticmethod
    def _map_single_structure(site: jnp.ndarray, simplex: jnp.ndarray):
        return apply_contraction(
            "triangular_pess_mapping",
            [],
            [],
            [
                site,
                simplex,
            ],
        )

    def __call__(
        self,
        input_tensors: Sequence[jnp.ndarray],
        *,
        generate_unitcell: bool = True,
    ) -> Union[List[jnp.ndarray], Tuple[List[jnp.ndarray], PEPS_Unit_Cell]]:
        num_peps_sites = len(input_tensors) // 2
        if num_peps_sites * 2 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a triangular simplex system."
            )

        peps_tensors = [
            self._map_single_structure(*(input_tensors[(i * 2) : (i * 2 + 2)]))
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
        cls: Type[T_Triangular_Map_PESS_To_PEPS],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: int,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Triangular_Map_PESS_To_PEPS]:
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
            result_tensors.append(rng.block((D, D, D, d), dtype=dtype))  # site
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex

        return result_tensors, cls(
            unitcell_structure=structure, chi=chi, max_chi=max_chi
        )

    @classmethod
    def save_to_file(
        cls: Type[T_Triangular_Map_PESS_To_PEPS],
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
        max_trunc_error_list: Optional[List[float]] = None,
    ) -> None:
        """
        Save unit cell to a HDF5 file.

        This function creates a single group "triangular_pess" in the file
        and pass this group to the method
        :obj:`~Triangular_Map_PESS_To_PEPS.save_to_group` then.

        Args:
          path (:obj:`os.PathLike`):
            Path of the new file. Caution: The file will overwritten if existing.
          store_config (:obj:`bool`):
            Store the current values of the global config object into the HDF5
            file as attrs of an extra group.
        """
        with h5py.File(path, "w", libver=("earliest", "v110")) as f:
            grp = f.create_group("triangular_pess")

            cls.save_to_group(grp, tensors, unitcell, store_config=store_config)

            if max_trunc_error_list is not None:
                f.create_dataset(
                    "max_trunc_error_list",
                    data=jnp.array(max_trunc_error_list),
                    compression="gzip",
                    compression_opts=6,
                )

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
        num_peps_sites = len(tensors) // 2
        if num_peps_sites * 2 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a triangular simplex system."
            )

        grp_pess = grp.create_group("pess_tensors", track_order=True)
        grp_pess.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (
                site,
                simplex,
            ) = tensors[(i * 2) : (i * 2 + 2)]

            grp_pess.create_dataset(
                f"site{i}_site", data=site, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_simplex",
                data=simplex,
                compression="gzip",
                compression_opts=6,
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Triangular_Map_PESS_To_PEPS],
        path: PathLike,
        *,
        return_config: bool = False,
        return_max_trunc_error_list: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, peps_ad.config.PEPS_AD_Config],
    ]:
        """
        Load unit cell from a HDF5 file.

        This function read the group "triangular_pess" from the file and pass
        this group to the method
        :obj:`~Triangular_Map_PESS_To_PEPS.load_from_group` then.

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
            out = cls.load_from_group(f["triangular_pess"], return_config=return_config)
            max_trunc_error_list = f.get("max_trunc_error_list")

        if return_config and return_max_trunc_error_list:
            return out[0], out[1], out[2], max_trunc_error_list
        elif return_config:
            return out[0], out[1], out[2]
        elif return_max_trunc_error_list:
            return out[0], out[1], max_trunc_error_list

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
            tensors.append(jnp.asarray(grp_pess[f"site{i}_site"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Triangular_Map_PESS_To_PEPS],
        filename: PathLike,
        tensors: jnp.ndarray,
        unitcell: PEPS_Unit_Cell,
        counter: Optional[int] = None,
        max_trunc_error_list: Optional[float] = None,
    ) -> None:
        if counter is not None:
            cls.save_to_file(
                f"{str(filename)}.{counter}",
                tensors,
                unitcell,
                max_trunc_error_list=max_trunc_error_list,
            )
        else:
            cls.save_to_file(
                filename, tensors, unitcell, max_trunc_error_list=max_trunc_error_list
            )
