import jax.numpy as jnp

from jax import jit

from varipeps.contractions import apply_contraction_jitted


class Overlap_Single_Site:
    @staticmethod
    @jit
    def calc_overlap(unitcell):
        density_matrix = apply_contraction_jitted(
            "density_matrix_one_site",
            [unitcell[0, 0][0][0].tensor],
            [unitcell[0, 0][0][0]],
            [],
        )

        norm_with_site = jnp.trace(density_matrix)

        norm_corners = apply_contraction_jitted(
            "overlap_one_site_only_corners",
            [unitcell[0, 0][0][0].tensor],
            [unitcell[0, 0][0][0]],
            [],
        )

        norm_horizontal = apply_contraction_jitted(
            "overlap_one_site_only_transfer_horizontal",
            [unitcell[0, 0][0][0].tensor],
            [unitcell[0, 0][0][0]],
            [],
        )

        norm_vertical = apply_contraction_jitted(
            "overlap_one_site_only_transfer_vertical",
            [unitcell[0, 0][0][0].tensor],
            [unitcell[0, 0][0][0]],
            [],
        )

        return (norm_with_site * norm_corners) / (norm_horizontal * norm_vertical)
