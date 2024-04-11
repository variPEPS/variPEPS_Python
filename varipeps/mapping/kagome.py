from dataclasses import dataclass
from os import PathLike

import jax.numpy as jnp
from jax import jit

import h5py

import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.one_site import calc_one_site_multi_gates
from varipeps.expectation.three_sites import _three_site_triangle_workhorse
from varipeps.typing import Tensor
from varipeps.utils.random import PEPS_Random_Number_Generator
from varipeps.mapping import Map_To_PEPS_Model

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

T_float_complex = TypeVar("T_float_complex", float, complex)
T_Kagome_Map_PESS3_To_Single_PEPS_Site = TypeVar(
    "T_Kagome_Map_PESS3_To_Single_PEPS_Site",
    bound="Kagome_Map_PESS3_To_Single_PEPS_Site",
)


@dataclass
class Kagome_PESS3_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Kagome 3-PESS
    structure.

    .. figure:: /images/kagome_structure.*
       :align: center
       :width: 70%
       :alt: Structure of the Kagome lattice with smallest possible unit cell
             marked by dashed lines.

       Structure of the Kagome lattice with smallest possible unit cell marked
       by dashed lines.

    \\

    Args:
      upward_triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the upward
        triangles.
      downward_triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the downward
        triangles.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        If for example three sites are mapped into one PEPS site this
        should be 3.
      operation_before_sum (:term:`callable` of type :obj:`float`/:obj:`complex` to :obj:`float`/:obj:`complex`):
        Function which should be applied to the expectation values before they
        are summed up.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    upward_triangle_gates: Sequence[jnp.ndarray]
    downward_triangle_gates: Sequence[jnp.ndarray]
    normalization_factor: int = 3
    operation_before_sum: Optional[Callable[[T_float_complex], T_float_complex]] = None

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.upward_triangle_gates, jnp.ndarray):
            self.upward_triangle_gates = (self.upward_triangle_gates,)

        if isinstance(self.downward_triangle_gates, jnp.ndarray):
            self.downward_triangle_gates = (self.downward_triangle_gates,)

        if (
            len(self.upward_triangle_gates) > 0
            and len(self.downward_triangle_gates) > 0
            and len(self.upward_triangle_gates) != len(self.downward_triangle_gates)
        ):
            raise ValueError("Length of upward and downward gates mismatch.")

        if self.is_spiral_peps:
            raise NotImplementedError

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
            if all(jnp.allclose(g, jnp.real(g)) for g in self.upward_triangle_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.downward_triangle_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(
                max(
                    len(self.upward_triangle_gates),
                    len(self.downward_triangle_gates),
                )
            )
        ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                if len(self.upward_triangle_gates) > 0:
                    upward_tensors = peps_tensors[view.get_indices((0, 0))[0][0]]
                    upward_tensor_objs = view[0, 0][0][0]

                    step_result_upward = calc_one_site_multi_gates(
                        upward_tensors,
                        upward_tensor_objs,
                        self.upward_triangle_gates,
                    )

                    for sr_i, sr in enumerate(step_result_upward):
                        if self.operation_before_sum is not None:
                            sr = self.operation_before_sum(sr)
                        result[sr_i] += sr

                if len(self.downward_triangle_gates) > 0:
                    downward_tensors_i = view.get_indices(
                        (slice(0, 2, None), slice(0, 2, None))
                    )
                    downward_tensors = [
                        peps_tensors[i] for j in downward_tensors_i for i in j
                    ]
                    downward_tensor_objs = [t for tl in view[:2, :2] for t in tl]

                    for ti in range(1, 4):
                        t = downward_tensors[ti]
                        new_d = round(t.shape[2] ** (1 / 3))
                        downward_tensors[ti] = t.reshape(
                            t.shape[0],
                            t.shape[1],
                            new_d,
                            new_d,
                            new_d,
                            t.shape[3],
                            t.shape[4],
                        )

                    traced_density_matrix_top_left = apply_contraction(
                        "ctmrg_top_left",
                        [downward_tensors[0]],
                        [downward_tensor_objs[0]],
                        [],
                    )

                    density_matrix_top_right = apply_contraction(
                        "kagome_downward_triangle_top_right",
                        [downward_tensors[1]],
                        [downward_tensor_objs[1]],
                        [],
                        disable_identity_check=True,
                    )

                    density_matrix_bottom_left = apply_contraction(
                        "kagome_downward_triangle_bottom_left",
                        [downward_tensors[2]],
                        [downward_tensor_objs[2]],
                        [],
                        disable_identity_check=True,
                    )

                    density_matrix_bottom_right = apply_contraction(
                        "kagome_downward_triangle_bottom_right",
                        [downward_tensors[3]],
                        [downward_tensor_objs[3]],
                        [],
                        disable_identity_check=True,
                    )

                    step_result_downward = _three_site_triangle_workhorse(
                        traced_density_matrix_top_left,
                        density_matrix_top_right,
                        density_matrix_bottom_left,
                        density_matrix_bottom_right,
                        tuple(self.downward_triangle_gates),
                        "top-left",
                        result_type is jnp.float64,
                    )

                    for sr_i, sr in enumerate(step_result_downward):
                        if self.operation_before_sum is not None:
                            sr = self.operation_before_sum(sr)
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


@jit
def _kagome_mapping_workhorse(
    up: jnp.ndarray,
    down: jnp.ndarray,
    site_1: jnp.ndarray,
    site_2: jnp.ndarray,
    site_3: jnp.ndarray,
):
    result = jnp.tensordot(site_1, up, ((2,), (0,)))
    result = jnp.tensordot(site_2, result, ((2,), (2,)))
    result = jnp.tensordot(site_3, result, ((2,), (4,)))
    result = jnp.tensordot(down, result, ((2,), (4,)))

    result = result.transpose((0, 4, 6, 5, 3, 2, 1))
    result = result.reshape(
        result.shape[0],
        result.shape[1],
        result.shape[2] * result.shape[3] * result.shape[4],
        result.shape[5],
        result.shape[6],
    )

    return result / jnp.linalg.norm(result)


@dataclass
class Kagome_Map_PESS3_To_Single_PEPS_Site(Map_To_PEPS_Model):
    """
    Map a 3-site Kagome iPESS unit cell to a iPEPS structure.

    Create a PEPS unitcell from a Kagome 3-PESS structure. To this end, the
    two simplex tensor and all three sites are mapped into PEPS sites.

    The axes of the simplex tensors are expected to be in the order:
    - Up: PESS site 1, PESS site 2, PESS site 3
    - Down: PESS site 3, PESS site 2, PESS site 1

    The axes of the site tensors are expected to be in the order
    `connection to down simplex, physical bond, connection to up simplex`.

    The PESS structure is contracted in the way that all site tensors are
    connected to the up simplex and the down simplex to site 1.

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
        num_peps_sites = len(input_tensors) // 5
        if num_peps_sites * 5 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome simplex system."
            )

        peps_tensors = [
            _kagome_mapping_workhorse(*(input_tensors[(i * 5) : (i * 5 + 5)]))
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
        cls: Type[T_Kagome_Map_PESS3_To_Single_PEPS_Site],
        structure: Sequence[Sequence[int]],
        d: int,
        D: int,
        chi: Union[int, Sequence[int]],
        dtype: Type[jnp.number],
        max_chi: int,
        *,
        seed: Optional[int] = None,
        destroy_random_state: bool = True,
    ) -> Tuple[List[jnp.ndarray], T_Kagome_Map_PESS3_To_Single_PEPS_Site]:
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
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_up
            result_tensors.append(rng.block((D, D, D), dtype=dtype))  # simplex_down
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # site1
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # site2
            result_tensors.append(rng.block((D, d, D), dtype=dtype))  # site3

        return result_tensors, cls(
            unitcell_structure=structure, chi=chi, max_chi=max_chi
        )

    @classmethod
    def save_to_file(
        cls: Type[T_Kagome_Map_PESS3_To_Single_PEPS_Site],
        path: PathLike,
        tensors: List[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        store_config: bool = True,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save Kagome PESS tensors and unit cell to a HDF5 file.

        This function creates a single group "kagome_pess" in the file
        and pass this group to the method
        :obj:`~Kagome_Map_PESS3_To_Single_PEPS_Site.save_to_group` then.

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
            grp = f.create_group("kagome_pess")

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
        Save Kagome PESS tensors and unit cell to a HDF5 group which is be
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
        num_peps_sites = len(tensors) // 5
        if num_peps_sites * 5 != len(tensors):
            raise ValueError(
                "Input tensors seems not be a list for a Kagome simplex system."
            )

        grp_pess = grp.create_group("pess_tensors", track_order=True)
        grp_pess.attrs["num_peps_sites"] = num_peps_sites

        for i in range(num_peps_sites):
            (
                simplex_up,
                simplex_down,
                t1,
                t2,
                t3,
            ) = tensors[(i * 5) : (i * 5 + 5)]

            grp_pess.create_dataset(
                f"site{i}_simplex_up",
                data=simplex_up,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_simplex_down",
                data=simplex_down,
                compression="gzip",
                compression_opts=6,
            )
            grp_pess.create_dataset(
                f"site{i}_t1", data=t1, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t2", data=t2, compression="gzip", compression_opts=6
            )
            grp_pess.create_dataset(
                f"site{i}_t3", data=t3, compression="gzip", compression_opts=6
            )

        grp_unitcell = grp.create_group("unitcell")
        unitcell.save_to_group(grp_unitcell, store_config=store_config)

    @classmethod
    def load_from_file(
        cls: Type[T_Kagome_Map_PESS3_To_Single_PEPS_Site],
        path: PathLike,
        *,
        return_config: bool = False,
        return_auxiliary_data: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load Kagome PESS tensors and unit cell from a HDF5 file.

        This function read the group "kagome_pess" from the file and pass
        this group to the method
        :obj:`~Kagome_Map_PESS3_To_Single_PEPS_Site.load_from_group` then.

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
            out = cls.load_from_group(f["kagome_pess"], return_config=return_config)

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
        Load the Kagome PESS tensors and  unit cell from a HDF5 group which is
        be passed to the method.

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
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_up"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_simplex_down"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t1"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t2"]))
            tensors.append(jnp.asarray(grp_pess[f"site{i}_t3"]))

        out = PEPS_Unit_Cell.load_from_group(
            grp["unitcell"], return_config=return_config
        )

        if return_config:
            return tensors, out[0], out[1]

        return tensors, out

    @classmethod
    def autosave_wrapper(
        cls: Type[T_Kagome_Map_PESS3_To_Single_PEPS_Site],
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


class iPESS3_9Sites_Three_PEPS_Site:
    """
    Map a 9-sites Kagome iPESS3 unit cell to PEPS structure using a PEPS
    unitcell consisting of three unique sites.
    """

    @staticmethod
    def unitcell_from_pess_tensors(
        up_simplex_1: Tensor,
        up_simplex_2: Tensor,
        up_simplex_3: Tensor,
        down_simplex_1: Tensor,
        down_simplex_2: Tensor,
        down_simplex_3: Tensor,
        site_A1: Tensor,
        site_A2: Tensor,
        site_A3: Tensor,
        site_B1: Tensor,
        site_B2: Tensor,
        site_B3: Tensor,
        site_C1: Tensor,
        site_C2: Tensor,
        site_C3: Tensor,
        d: int,
        D: int,
        chi: int,
    ) -> PEPS_Unit_Cell:
        """
        Create a PEPS unitcell from a Kagome 3-PESS 9-sites structure. To this
        end, the six simplex tensor and all nine sites are mapped into thre
        unique PEPS sites.

        The axes of the simplex tensors are expected to be in the order:
        - Up1: PESS site A1, PESS site B1, PESS site C1
        - Up2: PESS site A2, PESS site B2, PESS site C3
        - Up3: PESS site A3, PESS site B3, PESS site C2
        - Down1: PESS site C2, PESS site B1, PESS site A2
        - Down2: PESS site C1, PESS site B2, PESS site A3
        - Down3: PESS site C3, PESS site B3, PESS site A1

        The axes site tensors are expected to be in the order
        connection to down simplex, physical bond, connection to up simplex.

        The PESS structure is contracted in the way that all site tensors are
        connected to the up simplex and then the down simplex to site A{1,2,3}.

        Args:
          up_simplex_1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the first up simplex.
          up_simplex_2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the second up simplex.
          up_simplex_3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the third up simplex.
          down_simplex_1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the first down simplex.
          down_simplex_2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the second down simplex.
          down_simplex_3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the third down simplex.
          site_A1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site A1.
          site_A2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site A2.
          site_A3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site A3.
          site_B1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site B1.
          site_B2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site B2.
          site_B3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site B3.
          site_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site C1.
          site_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site C2.
          site_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the PESS site C3.
          d (:obj:`int`):
            Physical dimension
          D (:obj:`int`):
            Bond dimension.
          chi (:obj:`int`):
            Environment bond dimension.
        Returns:
          ~varipeps.peps.PEPS_Unit_Cell:
            PEPS unitcell with the mapped PESS structure and initialized
            environment tensors.
        """
        if not isinstance(d, int) or not isinstance(D, int) or not isinstance(chi, int):
            raise ValueError("Dimensions have to be integers.")

        if not all(
            s.shape[1] == d
            for s in (
                site_A1,
                site_A2,
                site_A3,
                site_B1,
                site_B2,
                site_B3,
                site_C1,
                site_C2,
                site_C3,
            )
        ):
            raise ValueError(
                "Dimension of site tensor mismatches physical dimension argument."
            )

        if not all(
            s.shape[0] == s.shape[2] == D
            for s in (
                site_A1,
                site_A2,
                site_A3,
                site_B1,
                site_B2,
                site_B3,
                site_C1,
                site_C2,
                site_C3,
            )
        ) or not all(
            s.shape[0] == s.shape[1] == s.shape[2] == D
            for s in (
                up_simplex_1,
                up_simplex_2,
                up_simplex_3,
                down_simplex_1,
                down_simplex_2,
                down_simplex_3,
            )
        ):
            raise ValueError("Dimension of tensor mismatches bond dimension argument.")

        peps_tensor_1 = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex_1),
            jnp.asarray(down_simplex_3),
            jnp.asarray(site_A1),
            jnp.asarray(site_B1),
            jnp.asarray(site_C1),
        )
        peps_tensor_obj_1 = PEPS_Tensor.from_tensor(peps_tensor_1, d**3, D, chi)

        peps_tensor_2 = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex_3),
            jnp.asarray(down_simplex_2),
            jnp.asarray(site_A3),
            jnp.asarray(site_B3),
            jnp.asarray(site_C2),
        )
        peps_tensor_obj_2 = PEPS_Tensor.from_tensor(peps_tensor_2, d**3, D, chi)

        peps_tensor_3 = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex_2),
            jnp.asarray(down_simplex_1),
            jnp.asarray(site_A2),
            jnp.asarray(site_B2),
            jnp.asarray(site_C3),
        )
        peps_tensor_obj_3 = PEPS_Tensor.from_tensor(peps_tensor_3, d**3, D, chi)

        return PEPS_Unit_Cell.from_tensor_list(
            (peps_tensor_obj_1, peps_tensor_obj_2, peps_tensor_obj_3),
            ((0, 1, 2), (2, 0, 1), (1, 2, 0)),
        )


@dataclass
class Kagome_Upper_Right_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped Kagome 3-PESS
    structure.

    Args:
      upward_triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the upward
        triangles.
      downward_triangle_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to the downward
        triangles.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        If for example three sites are mapped into one PEPS site this
        should be 3.
      operation_before_sum (:term:`callable` of type :obj:`float`/:obj:`complex` to :obj:`float`/:obj:`complex`):
        Function which should be applied to the expectation values before they
        are summed up.
    """

    upward_triangle_gates: Sequence[jnp.ndarray]
    downward_triangle_gates: Sequence[jnp.ndarray]
    normalization_factor: int = 3
    operation_before_sum: Optional[Callable[[T_float_complex], T_float_complex]] = None

    def __post_init__(self) -> None:
        if (
            len(self.upward_triangle_gates) > 0
            and len(self.downward_triangle_gates) > 0
            and len(self.upward_triangle_gates) != len(self.downward_triangle_gates)
        ):
            raise ValueError("Length of upward and downward gates mismatch.")

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
            if all(jnp.allclose(g, jnp.real(g)) for g in self.upward_triangle_gates)
            and all(jnp.allclose(g, jnp.real(g)) for g in self.downward_triangle_gates)
            else jnp.complex128
        )
        result = [
            jnp.array(0, dtype=result_type)
            for _ in range(
                max(
                    len(self.upward_triangle_gates),
                    len(self.downward_triangle_gates),
                )
            )
        ]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                if len(self.upward_triangle_gates) > 0:
                    upward_tensors = peps_tensors[view.get_indices((0, 0))[0][0]]
                    upward_tensor_objs = view[0, 0][0][0]

                    step_result_upward = calc_one_site_multi_gates(
                        upward_tensors,
                        upward_tensor_objs,
                        self.upward_triangle_gates,
                    )

                    for sr_i, sr in enumerate(step_result_upward):
                        if self.operation_before_sum is not None:
                            sr = self.operation_before_sum(sr)
                        result[sr_i] += sr

                if len(self.downward_triangle_gates) > 0:
                    downward_tensors_i = view.get_indices(
                        (slice(0, 2, None), slice(0, 2, None))
                    )
                    downward_tensors = [
                        peps_tensors[i] for j in downward_tensors_i for i in j
                    ]
                    downward_tensor_objs = [t for tl in view[:2, :2] for t in tl]

                    for ti in [0, 1, 3]:
                        t = downward_tensors[ti]
                        new_d = round(t.shape[2] ** (1 / 3))
                        downward_tensors[ti] = t.reshape(
                            t.shape[0],
                            t.shape[1],
                            new_d,
                            new_d,
                            new_d,
                            t.shape[3],
                            t.shape[4],
                        )

                    density_matrix_top_left = apply_contraction(
                        "kagome_downward_triangle_top_left",
                        [downward_tensors[0]],
                        [downward_tensor_objs[0]],
                        [],
                        disable_identity_check=True,
                    )

                    density_matrix_top_right = apply_contraction(
                        "kagome_downward_triangle_top_right",
                        [downward_tensors[1]],
                        [downward_tensor_objs[1]],
                        [],
                        disable_identity_check=True,
                    )

                    traced_density_matrix_bottom_left = apply_contraction(
                        "ctmrg_bottom_left",
                        [downward_tensors[2]],
                        [downward_tensor_objs[2]],
                        [],
                    )

                    density_matrix_bottom_right = apply_contraction(
                        "kagome_downward_triangle_bottom_right",
                        [downward_tensors[3]],
                        [downward_tensor_objs[3]],
                        [],
                        disable_identity_check=True,
                    )

                    step_result_downward = _three_site_triangle_workhorse(
                        density_matrix_top_left,
                        density_matrix_top_right,
                        traced_density_matrix_bottom_left,
                        density_matrix_bottom_right,
                        tuple(self.downward_triangle_gates),
                        "bottom-left",
                        result_type is jnp.float64,
                    )

                    for sr_i, sr in enumerate(step_result_downward):
                        if self.operation_before_sum is not None:
                            sr = self.operation_before_sum(sr)
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


@jit
def _kagome_mapping_workhorse_upper_triangle(
    up: jnp.ndarray,
    down: jnp.ndarray,
    site_1: jnp.ndarray,
    site_2: jnp.ndarray,
    site_3: jnp.ndarray,
):
    peps_tensor = apply_contraction_jitted(
        "kagome_pess_mapping_upper_triangle",
        [],
        [],
        [up, down, site_1, site_2, site_3],
    )

    return peps_tensor.reshape(
        peps_tensor.shape[0],
        peps_tensor.shape[1],
        -1,
        peps_tensor.shape[5],
        peps_tensor.shape[6],
    )


@dataclass
class Kagome_Map_PESS3_To_Single_PEPS_Site_Upper_Triangle(Map_To_PEPS_Model):
    """
    Map a 3-site Kagome iPESS unit cell to a iPEPS structure.

    Create a PEPS unitcell from a Kagome 3-PESS structure. To this end, the
    two simplex tensor and all three sites are mapped into PEPS sites.

    The axes of the simplex tensors are expected to be in the order:
    - Up: PESS site 1, PESS site 2, PESS site 3
    - Down: PESS site 3, PESS site 2, PESS site 1

    The axes of the site tensors are expected to be in the order
    `connection to down simplex, physical bond, connection to up simplex`.

    The PESS structure is contracted in the way that all site tensors are
    connected to the up simplex and the down simplex to site 2.

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
        num_peps_sites = len(input_tensors) // 5
        if num_peps_sites * 5 != len(input_tensors):
            raise ValueError(
                "Input tensors seems not be a list for a square Kagome simplex system."
            )

        peps_tensors = [
            _kagome_mapping_workhorse_upper_triangle(
                *(input_tensors[(i * 5) : (i * 5 + 5)])
            )
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
