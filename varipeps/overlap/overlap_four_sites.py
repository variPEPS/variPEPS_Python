import numpy as np
import jax.numpy as jnp

from jax import jit

from varipeps.contractions import apply_contraction_jitted


class Overlap_Four_Sites_Square:
    @staticmethod
    @jit
    def calc_overlap(unitcell):
        top_left = apply_contraction_jitted(
            "ctmrg_top_left", [unitcell[0, 0][0][0].tensor], [unitcell[0, 0][0][0]], []
        )
        top_left = top_left.reshape(
            np.prod(top_left.shape[:3]), np.prod(top_left.shape[3:])
        )

        top_right = apply_contraction_jitted(
            "ctmrg_top_right", [unitcell[0, 1][0][0].tensor], [unitcell[0, 1][0][0]], []
        )
        top_right = top_right.reshape(
            np.prod(top_right.shape[:3]), np.prod(top_right.shape[3:])
        )

        bottom_left = apply_contraction_jitted(
            "ctmrg_bottom_left",
            [unitcell[1, 0][0][0].tensor],
            [unitcell[1, 0][0][0]],
            [],
        )
        bottom_left = bottom_left.reshape(
            np.prod(bottom_left.shape[:3]), np.prod(bottom_left.shape[3:])
        )

        bottom_right = apply_contraction_jitted(
            "ctmrg_bottom_right",
            [unitcell[1, 1][0][0].tensor],
            [unitcell[1, 1][0][0]],
            [],
        )
        bottom_right = bottom_right.reshape(
            np.prod(bottom_right.shape[:3]), np.prod(bottom_right.shape[3:])
        )

        norm_with_sites = jnp.trace(top_left @ top_right @ bottom_left @ bottom_right)

        norm_corners = apply_contraction_jitted(
            "overlap_four_sites_square_only_corners",
            [
                unitcell[0, 0][0][0].tensor,
                unitcell[0, 1][0][0].tensor,
                unitcell[1, 1][0][0].tensor,
                unitcell[1, 0][0][0].tensor,
            ],
            [
                unitcell[0, 0][0][0],
                unitcell[0, 1][0][0],
                unitcell[1, 1][0][0],
                unitcell[1, 0][0][0],
            ],
            [],
        )

        norm_horizontal = apply_contraction_jitted(
            "overlap_four_sites_square_transfer_horizontal",
            [
                unitcell[0, 0][0][0].tensor,
                unitcell[0, 1][0][0].tensor,
                unitcell[1, 1][0][0].tensor,
                unitcell[1, 0][0][0].tensor,
            ],
            [
                unitcell[0, 0][0][0],
                unitcell[0, 1][0][0],
                unitcell[1, 1][0][0],
                unitcell[1, 0][0][0],
            ],
            [],
        )

        norm_vertical = apply_contraction_jitted(
            "overlap_four_sites_square_transfer_vertical",
            [
                unitcell[0, 0][0][0].tensor,
                unitcell[0, 1][0][0].tensor,
                unitcell[1, 1][0][0].tensor,
                unitcell[1, 0][0][0].tensor,
            ],
            [
                unitcell[0, 0][0][0],
                unitcell[0, 1][0][0],
                unitcell[1, 1][0][0],
                unitcell[1, 0][0][0],
            ],
            [],
        )

        return (norm_with_sites * norm_corners) / (norm_horizontal * norm_vertical)
