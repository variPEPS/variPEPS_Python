from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.contractions import apply_contraction
from peps_ad.expectation.model import Expectation_Model
from peps_ad.expectation.one_site import calc_one_site_multi_gates
from peps_ad.expectation.three_sites import _three_site_triangle_workhorse
from peps_ad.typing import Tensor

from typing import Sequence, Union, List


@dataclass
class Kagome_PESS3_Expectation_Value(Expectation_Model):
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
    """

    upward_triangle_gates: Sequence[jnp.ndarray]
    downward_triangle_gates: Sequence[jnp.ndarray]
    normalization_factor: int = 3

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


class iPESS3_Single_PEPS_Site:
    """
    Map a 3-site Kagome PESS unit cell to PEPS structure using a PEPS unitcell
    consisting of one site.
    """

    @staticmethod
    def unitcell_from_pess_tensors(
        up_simplex: Tensor,
        down_simplex: Tensor,
        site_1: Tensor,
        site_2: Tensor,
        site_3: Tensor,
        d: int,
        D: int,
        chi: int,
    ) -> PEPS_Unit_Cell:
        """
        Create a PEPS unitcell from a Kagome 3-PESS 3-sites structure. To this
        end, the two simplex tensor and all three sites are mapped into a single
        PEPS site.

        The axes of the simplex tensors are expected to be in the order:
        - Up: PESS site 1, PESS site 2, PESS site 3
        - Down: PESS site 3, PESS site 2, PESS site 1

        The axes site tensors are expected to be in the order
        connection to down simplex, physical bond, connection to up simplex.

        The PESS structure is contracted in the way that all site tensors are
        connected to the up simplex and the down simplex to site 1.

        Args:
          up_simplex (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the up simplex.
          down_simplex (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the down simplex.
          site_1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the first PESS site.
          site_2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the second PESS site.
          site_3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            The tensor of the third PESS site.
          d (:obj:`int`):
            Physical dimension
          D (:obj:`int`):
            Bond dimension.
          chi (:obj:`int`):
            Environment bond dimension.
        Returns:
          ~peps_ad.peps.PEPS_Unit_Cell:
            PEPS unitcell with the mapped PESS structure and initialized
            environment tensors.
        """
        if not isinstance(d, int) or not isinstance(D, int) or not isinstance(chi, int):
            raise ValueError("Dimensions have to be integers.")

        if not (site_1.shape[1] == site_2.shape[1] == site_3.shape[1] == d):
            raise ValueError(
                "Dimension of site tensor mismatches physical dimension argument."
            )

        if not (
            site_1.shape[0]
            == site_1.shape[2]
            == site_2.shape[0]
            == site_2.shape[2]
            == site_3.shape[0]
            == site_3.shape[2]
            == D
        ) or not (
            up_simplex.shape[0]
            == up_simplex.shape[1]
            == up_simplex.shape[2]
            == down_simplex.shape[0]
            == down_simplex.shape[1]
            == down_simplex.shape[2]
            == D
        ):
            raise ValueError("Dimension of tensor mismatches bond dimension argument.")

        peps_tensor = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex),
            jnp.asarray(down_simplex),
            jnp.asarray(site_1),
            jnp.asarray(site_2),
            jnp.asarray(site_3),
        )

        peps_tensor_obj = PEPS_Tensor.from_tensor(peps_tensor, d ** 3, D, chi)

        return PEPS_Unit_Cell.from_tensor_list((peps_tensor_obj,), ((0,),))


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
          ~peps_ad.peps.PEPS_Unit_Cell:
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
        peps_tensor_obj_1 = PEPS_Tensor.from_tensor(peps_tensor_1, d ** 3, D, chi)

        peps_tensor_2 = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex_3),
            jnp.asarray(down_simplex_2),
            jnp.asarray(site_A3),
            jnp.asarray(site_B3),
            jnp.asarray(site_C2),
        )
        peps_tensor_obj_2 = PEPS_Tensor.from_tensor(peps_tensor_2, d ** 3, D, chi)

        peps_tensor_3 = _kagome_mapping_workhorse(
            jnp.asarray(up_simplex_2),
            jnp.asarray(down_simplex_1),
            jnp.asarray(site_A2),
            jnp.asarray(site_B2),
            jnp.asarray(site_C3),
        )
        peps_tensor_obj_3 = PEPS_Tensor.from_tensor(peps_tensor_3, d ** 3, D, chi)

        return PEPS_Unit_Cell.from_tensor_list(
            (peps_tensor_obj_1, peps_tensor_obj_2, peps_tensor_obj_3),
            ((0, 1, 2), (2, 0, 1), (1, 2, 0)),
        )
