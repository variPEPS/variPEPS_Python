import collections.abc
from dataclasses import dataclass
from functools import partial
from os import PathLike

import jax.numpy as jnp
from jax import jit
import jax.util
import h5py

from varipeps import varipeps_config
import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.two_sites import (
    calc_two_sites_vertical_multiple_gates,
    calc_two_sites_horizontal_multiple_gates,
    calc_two_sites_diagonal_top_left_bottom_right_multiple_gates,
    calc_two_sites_diagonal_horizontal_rectangle_multiple_gates,
    calc_two_sites_diagonal_vertical_rectangle_multiple_gates,
)
from varipeps.expectation.four_sites import calc_four_sites_quadrat_multiple_gates
from varipeps.expectation.spiral_helpers import apply_unitary
from varipeps.typing import Tensor
from varipeps.utils.random import PEPS_Random_Number_Generator
from varipeps.mapping import Map_To_PEPS_Model

from varipeps.utils.debug_print import debug_print

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

T_Triangular_Map_PESS_To_PEPS = TypeVar(
    "T_Triangular_Map_PESS_To_PEPS", bound="Triangular_Map_PESS_To_PEPS"
)


@dataclass
class Triangular_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a mapped triangular PESS
    structure.

    .. figure:: /images/triangular_structure.*
       :align: center
       :width: 70%
       :alt: Structure of the triangular lattice with smallest possible unit
             cell marked by dashed lines.

       Structure of the triangular lattice with smallest possible unit cell
       marked by dashed lines.

    \\

    Args:
      horizontal_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each nearest horizontal
        neighbor.
      vertical_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each nearest vertical
        neighbor.
      diagonal_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each nearest diagonal
        neighbor.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        If for example three sites are mapped into one PEPS site this
        should be 1.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    horizontal_gates: Sequence[jnp.ndarray]
    vertical_gates: Sequence[jnp.ndarray]
    diagonal_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.horizontal_gates, jnp.ndarray):
            self.horizontal_gates = (self.horizontal_gates,)
        else:
            self.horizontal_gates = tuple(self.horizontal_gates)

        if isinstance(self.vertical_gates, jnp.ndarray):
            self.vertical_gates = (self.vertical_gates,)
        else:
            self.vertical_gates = tuple(self.vertical_gates)

        if isinstance(self.diagonal_gates, jnp.ndarray):
            self.diagonal_gates = (self.diagonal_gates,)
        else:
            self.diagonal_gates = tuple(self.diagonal_gates)

        self._result_type = (
            jnp.float64
            if all(
                jnp.allclose(g, g.T.conj())
                for g in self.horizontal_gates
                + self.vertical_gates
                + self.diagonal_gates
            )
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
            for _ in range(len(self.horizontal_gates))
        ]

        if self.is_spiral_peps:
            if (
                isinstance(spiral_vectors, collections.abc.Sequence)
                and len(spiral_vectors) == 1
            ):
                spiral_vectors = spiral_vectors[0]

            if not isinstance(spiral_vectors, jnp.ndarray):
                raise ValueError("Expect spiral vector as single jax.numpy array.")

            working_h_gates = tuple(
                apply_unitary(
                    h,
                    jnp.array((0, 1)),
                    (spiral_vectors,),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.horizontal_gates
            )
            working_v_gates = tuple(
                apply_unitary(
                    v,
                    jnp.array((1, 0)),
                    (spiral_vectors,),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for v in self.vertical_gates
            )
            working_d_gates = tuple(
                apply_unitary(
                    d,
                    jnp.array((1, 1)),
                    (spiral_vectors,),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for d in self.diagonal_gates
            )
        else:
            working_h_gates = self.horizontal_gates
            working_v_gates = self.vertical_gates
            working_d_gates = self.diagonal_gates

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                x_tensors_i = view.get_indices((slice(0, 2, None), 0))
                x_tensors = [peps_tensors[i] for j in x_tensors_i for i in j]
                x_tensor_objs = [t for tl in view[:2, 0] for t in tl]

                step_result_x = calc_two_sites_vertical_multiple_gates(
                    x_tensors, x_tensor_objs, working_v_gates
                )

                y_tensors_i = view.get_indices((0, slice(0, 2, None)))
                y_tensors = [peps_tensors[i] for j in y_tensors_i for i in j]
                y_tensor_objs = [t for tl in view[0, :2] for t in tl]

                step_result_y = calc_two_sites_horizontal_multiple_gates(
                    y_tensors, y_tensor_objs, working_h_gates
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
                        working_d_gates,
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
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save Triangular tensors and unit cell to a HDF5 file.

        This function creates a single group "triangular_pess" in the file
        and pass this group to the method
        :obj:`~Triangular_Map_PESS_To_PEPS.save_to_group` then.

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
            grp = f.create_group("triangular_pess")

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
        return_auxiliary_data: bool = False,
    ) -> Union[
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell],
        Tuple[List[jnp.ndarray], PEPS_Unit_Cell, varipeps.config.VariPEPS_Config],
    ]:
        """
        Load Triangular tensors and unit cell from a HDF5 file.

        This function read the group "triangular_pess" from the file and pass
        this group to the method
        :obj:`~Triangular_Map_PESS_To_PEPS.load_from_group` then.

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
            out = cls.load_from_group(f["triangular_pess"], return_config=return_config)

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


@partial(jit, static_argnums=(2, 3))
def _calc_quadrat_gate_next_nearest(
    nearest_gates: Sequence[jnp.ndarray],
    next_nearest_gates: Sequence[jnp.ndarray],
    d: int,
    result_length: int,
):
    result = [None] * result_length

    single_gates = [None] * result_length

    Id_other_sites = jnp.eye(d**2)

    for i, (n_e, n_n_e) in enumerate(
        jax.util.safe_zip(nearest_gates, next_nearest_gates)
    ):
        nearest_34 = jnp.kron(Id_other_sites, n_e)

        nearest_13 = nearest_34.reshape(d, d, d, d, d, d, d, d)
        nearest_14 = nearest_34.reshape(d, d, d, d, d, d, d, d)

        nearest_13 = nearest_13.transpose(2, 0, 3, 1, 6, 4, 7, 5)
        nearest_13 = nearest_13.reshape(d**4, d**4)

        nearest_14 = nearest_14.transpose(2, 0, 1, 3, 6, 4, 5, 7)
        nearest_14 = nearest_14.reshape(d**4, d**4)

        next_nearest_23 = jnp.kron(n_n_e, Id_other_sites)
        next_nearest_23 = next_nearest_23.reshape(d, d, d, d, d, d, d, d)
        next_nearest_23 = next_nearest_23.transpose(2, 0, 1, 3, 6, 4, 5, 7)
        next_nearest_23 = next_nearest_23.reshape(d**4, d**4)

        result[i] = nearest_13 + nearest_14 + nearest_34 + next_nearest_23

        single_gates[i] = (nearest_13, nearest_14, nearest_34, next_nearest_23)

    return result, single_gates


@dataclass
class Triangular_Next_Nearest_Expectation_Value(Expectation_Model):
    """
    Class to calculate expectation values for a triangular structure with
    next-nearest interactions.

    Args:
      nearest_neighbor_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each nearest
        neighbor.
      next_nearest_neighbor_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates that should be applied to each next-nearest
        neighbor.
      real_d (:obj:`int`):
        Physical dimension of a single site before mapping.
      normalization_factor (:obj:`int`):
        Factor which should be used to normalize the calculated values.
        For a single layer triangular structure this should be normally 1.
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
      spiral_unitary_operator (:obj:`jax.numpy.ndarray`):
        Operator used to generate unitary for spiral iPEPS ansatz. Required
        if spiral iPEPS ansatz is used.
    """

    nearest_neighbor_gates: Sequence[jnp.ndarray]
    next_nearest_neighbor_gates: Sequence[jnp.ndarray]
    real_d: int
    normalization_factor: int = 1

    is_spiral_peps: bool = False
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.nearest_neighbor_gates, jnp.ndarray):
            self.nearest_neighbor_gates = (self.nearest_neighbor_gates,)
        if isinstance(self.next_nearest_neighbor_gates, jnp.ndarray):
            self.next_nearest_neighbor_gates = (self.next_nearest_neighbor_gates,)
        if len(self.nearest_neighbor_gates) != len(self.next_nearest_neighbor_gates):
            raise ValueError("Length mismatch for sequence of gates.")

        tmp_result = _calc_quadrat_gate_next_nearest(
            self.nearest_neighbor_gates,
            self.next_nearest_neighbor_gates,
            self.real_d,
            len(self.nearest_neighbor_gates),
        )
        self._quadrat_tuple, self._quadrat_single_gates = tuple(tmp_result[0]), tuple(
            tmp_result[1]
        )

        self._result_type = (
            jnp.float64
            if all(
                jnp.allclose(g, g.T.conj())
                for g in self.nearest_neighbor_gates + self.next_nearest_neighbor_gates
            )
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
            for _ in range(len(self.nearest_neighbor_gates))
        ]

        if return_single_gate_results:
            single_gates_result = [dict()] * len(self.nearest_neighbor_gates)

        if self.is_spiral_peps:
            if (
                isinstance(spiral_vectors, collections.abc.Sequence)
                and len(spiral_vectors) == 1
            ):
                spiral_vectors = spiral_vectors[0]

            if not isinstance(spiral_vectors, jnp.ndarray):
                raise ValueError("Expect spiral vector as single jax.numpy array.")

            working_q_gates = tuple(
                apply_unitary(
                    q,
                    (jnp.array((0, 1)), jnp.array((1, 0)), jnp.array((1, 1))),
                    (spiral_vectors, spiral_vectors, spiral_vectors),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    4,
                    (1, 2, 3),
                    varipeps_config.spiral_wavevector_type,
                )
                for q in self._quadrat_tuple
            )

            working_next_horizontal_gates = tuple(
                apply_unitary(
                    e,
                    jnp.array((1, 2)),
                    (spiral_vectors,),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in self.next_nearest_neighbor_gates
            )

            working_next_vertical_gates = tuple(
                apply_unitary(
                    e,
                    jnp.array((2, 1)),
                    (spiral_vectors,),
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    2,
                    (1,),
                    varipeps_config.spiral_wavevector_type,
                )
                for e in self.next_nearest_neighbor_gates
            )

            if return_single_gate_results:
                working_q_single_gates = tuple(
                    e
                    for q in self._quadrat_single_gates
                    for e in (
                        apply_unitary(
                            q[0],
                            jnp.array((1, 0)),
                            (spiral_vectors,),
                            self._spiral_D,
                            self._spiral_sigma,
                            self.real_d,
                            4,
                            (2,),
                            varipeps_config.spiral_wavevector_type,
                        ),
                        apply_unitary(
                            q[1],
                            jnp.array((1, 1)),
                            (spiral_vectors,),
                            self._spiral_D,
                            self._spiral_sigma,
                            self.real_d,
                            4,
                            (3,),
                            varipeps_config.spiral_wavevector_type,
                        ),
                        apply_unitary(
                            q[2],
                            (jnp.array((1, 0)), jnp.array((1, 1))),
                            (spiral_vectors, spiral_vectors),
                            self._spiral_D,
                            self._spiral_sigma,
                            self.real_d,
                            4,
                            (2, 3),
                            varipeps_config.spiral_wavevector_type,
                        ),
                        apply_unitary(
                            q[3],
                            (jnp.array((0, 1)), jnp.array((1, 0))),
                            (spiral_vectors, spiral_vectors),
                            self._spiral_D,
                            self._spiral_sigma,
                            self.real_d,
                            4,
                            (1, 2),
                            varipeps_config.spiral_wavevector_type,
                        ),
                    )
                )
        else:
            working_q_gates = self._quadrat_tuple
            working_next_horizontal_gates = self.next_nearest_neighbor_gates
            working_next_vertical_gates = self.next_nearest_neighbor_gates

            if return_single_gate_results:
                working_q_single_gates = tuple(
                    q for e in self._quadrat_single_gates for q in e
                )

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                quadrat_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                quadrat_tensors = [
                    peps_tensors[i] for j in quadrat_tensors_i for i in j
                ]
                quadrat_tensor_objs = [t for tl in view[:2, :2] for t in tl]

                if return_single_gate_results:
                    step_result_quadrat = calc_four_sites_quadrat_multiple_gates(
                        quadrat_tensors,
                        quadrat_tensor_objs,
                        working_q_gates + working_q_single_gates,
                    )
                else:
                    step_result_quadrat = calc_four_sites_quadrat_multiple_gates(
                        quadrat_tensors,
                        quadrat_tensor_objs,
                        working_q_gates,
                    )

                horizontal_rect_tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 3, None))
                )
                horizontal_rect_tensors = [
                    peps_tensors[i] for j in horizontal_rect_tensors_i for i in j
                ]
                horizontal_rect_tensor_objs = [t for tl in view[:2, :3] for t in tl]

                step_result_horizontal_rect = (
                    calc_two_sites_diagonal_horizontal_rectangle_multiple_gates(
                        horizontal_rect_tensors,
                        horizontal_rect_tensor_objs,
                        working_next_horizontal_gates,
                    )
                )

                vertical_rect_tensors_i = view.get_indices(
                    (slice(0, 3, None), slice(0, 2, None))
                )
                vertical_rect_tensors = [
                    peps_tensors[i] for j in vertical_rect_tensors_i for i in j
                ]
                vertical_rect_tensor_objs = [t for tl in view[:3, :2] for t in tl]

                step_result_vertical_rect = (
                    calc_two_sites_diagonal_vertical_rectangle_multiple_gates(
                        vertical_rect_tensors,
                        vertical_rect_tensor_objs,
                        working_next_vertical_gates,
                    )
                )

                for sr_i, (sr_q, sr_h, sr_v) in enumerate(
                    jax.util.safe_zip(
                        step_result_quadrat[: len(self.nearest_neighbor_gates)],
                        step_result_horizontal_rect[: len(self.nearest_neighbor_gates)],
                        step_result_vertical_rect[: len(self.nearest_neighbor_gates)],
                    )
                ):
                    result[sr_i] += sr_q + sr_h + sr_v

                if return_single_gate_results:
                    for sr_i in range(len(self.nearest_neighbor_gates)):
                        index_quadrat = (
                            len(self.nearest_neighbor_gates)
                            + len(self._quadrat_single_gates[0]) * sr_i
                        )

                        single_gates_result[sr_i][(x, y)] = dict(
                            zip(
                                (
                                    "nearest_13",
                                    "nearest_14",
                                    "nearest_34",
                                    "next_nearest_23",
                                    "next_nearest_horizontal_rect",
                                    "next_nearest_vertical_rect",
                                ),
                                (
                                    step_result_quadrat[
                                        index_quadrat : (
                                            index_quadrat
                                            + len(self._quadrat_single_gates[0])
                                        )
                                    ]
                                    + step_result_horizontal_rect[sr_i : sr_i + 1]
                                    + step_result_vertical_rect[sr_i : sr_i + 1]
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
