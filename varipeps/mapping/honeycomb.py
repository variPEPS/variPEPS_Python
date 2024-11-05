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

T_Honeycomb_Map_To_Square = TypeVar(
    "T_Honeycomb_Map_To_Square", bound="Honeycomb_Map_To_Square"
)


@dataclass
class Honeycomb_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Honeycomb
    structure.

    .. figure:: /images/honeycomb_structure.*
       :align: center
       :width: 100%
       :alt: Structure of the Honeycomb/Brickwall lattice with links marked.

       Structure of the Honeycomb/Brickwall lattice with links marked.

    \\

    Args:
      x_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to x links of the lattice.
      y_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to y links of the lattice.
      z_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to z links of the lattice.
      real_d (:obj:`int`):
        Physical dimension of a single site before mapping.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        Likely will be 2 for the a single layer structure.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    x_gates: Sequence[jnp.ndarray]
    y_gates: Sequence[jnp.ndarray]
    z_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 2

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.x_gates, jnp.ndarray):
            self.x_gates = (self.x_gates,)

        if isinstance(self.y_gates, jnp.ndarray):
            self.y_gates = (self.y_gates,)

        if isinstance(self.z_gates, jnp.ndarray):
            self.z_gates = (self.z_gates,)

        if (
            (
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

        self._result_type = (
            jnp.float64
            if all(jnp.allclose(g, g.T.conj()) for g in self.x_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.y_gates)
            and all(jnp.allclose(g, g.T.conj()) for g in self.z_gates)
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
            jnp.array(0, dtype=self._result_type)
            for _ in range(
                max(
                    len(self.x_gates),
                    len(self.y_gates),
                    len(self.z_gates),
                )
            )
        ]

        if self.is_spiral_peps:
            if isinstance(spiral_vectors, jnp.ndarray):
                spiral_vectors = (spiral_vectors,)
            if len(spiral_vectors) != 1:
                raise ValueError("Length mismatch for spiral vectors!")

            working_h_gates = tuple(
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
                for h in self._y_tuple
            )
            working_v_gates = tuple(
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
                for v in self._z_tuple
            )
        else:
            working_h_gates = self._y_tuple
            working_v_gates = self._z_tuple

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
                        working_h_gates,
                        self._result_type is jnp.float64,
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
                        "honeycomb_density_matrix_bottom",
                        [vertical_tensors[1]],
                        [vertical_tensor_objs[1]],
                        [],
                        disable_identity_check=True,
                    )

                    step_result_z = _two_site_workhorse(
                        density_matrix_top,
                        density_matrix_bottom,
                        working_v_gates,
                        self._result_type is jnp.float64,
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


@dataclass
class Honeycomb_Map_To_Square(Map_To_PEPS_Model):
    """
    Map the Honeycomb/Brickwall TN structure to a square PEPS unitcell.
    The convention for the input tensors is:

    Convention for physical site tensors:
    * t1: [left, bottom, phys, right]
    * t2: [left, phys, right, top]

    The both tensors will be contracted along the right bond of t1 and the
    left bond of t2.

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
    def _map_single_structure(site1: jnp.ndarray, site2: jnp.ndarray):
        result = jnp.tensordot(site1, site2, ((3,), (0,)))

        return result.reshape(
            result.shape[0],
            result.shape[1],
            -1,
            result.shape[4],
            result.shape[5],
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
                "Input tensors seems not be a list for a Honeycomb TN system."
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
        cls: Type[T_Honeycomb_Map_To_Square],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: int,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Honeycomb_Map_To_Square]:
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
            result_tensors.append(rng.block((D, D, d, D), dtype=dtype))  # t1
            result_tensors.append(rng.block((D, d, D, D), dtype=dtype))  # t2

        return result_tensors, cls(
            unitcell_structure=structure, chi=chi, max_chi=max_chi
        )

    @classmethod
    def save_to_file(
        cls,
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save Honeycomb PEPS tensors and unit cell to a HDF5 file.

        This function creates a single group "honeycomb_peps" in the file
        and pass this group to the method
        :obj:`~Honeycomb_Map_To_Square.save_to_group` then.

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
            grp = f.create_group("honeycomb_peps")

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
        Save Honeycomb PEPS tensors and unit cell to a HDF5 group which is be
        passed to the method.

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
        num_peps_sites = len(tensors) // 2
        if num_peps_sites * 2 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a Honeycomb system."
            )

        grp_peps = grp.create_group("honeycomb_peps_tensors", track_order=True)
        grp_peps.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (
                t1,
                t2,
            ) = tensors[(i * 2) : (i * 2 + 2)]

            grp_peps.create_dataset(
                f"site{i}_t1", data=t1, compression="gzip", compression_opts=6
            )
            grp_peps.create_dataset(
                f"site{i}_t2", data=t2, compression="gzip", compression_opts=6
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Honeycomb_Map_To_Square],
        path: PathLike,
        *,
        return_config: bool = False,
        return_auxiliary_data: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load Honeycomb PEPS tensors and unit cell from a HDF5 file.

        This function read the group "honeycomb_peps" from the file and pass
        this group to the method
        :obj:`~Honeycomb_Map_To_Square.load_from_group` then.

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
            out = cls.load_from_group(f["honeycomb_peps"], return_config=return_config)

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
        grp_peps = grp["honeycomb_peps_tensors"]
        num_peps_sites = grp_peps.attrs["num_peps_sites"]

        tensors = []

        for i in range(num_peps_sites):
            tensors.append(jnp.asarray(grp_peps[f"site{i}_t1"]))
            tensors.append(jnp.asarray(grp_peps[f"site{i}_t2"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Honeycomb_Map_To_Square],
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
