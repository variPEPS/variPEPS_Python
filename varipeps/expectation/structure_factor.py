from functools import partial

import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction_jitted

from typing import Optional, Sequence


@partial(jit, static_argnums=(4, 5))
def calc_structure_factor_expectation(
    peps_tensor_obj: PEPS_Tensor,
    alpha_gate: jnp.ndarray,
    beta_gate: jnp.ndarray,
    structure_factor_inner_factors: Optional[Sequence[complex]],
    real_d: int,
    num_sites: int,
):
    """
    Contract a structure factor with alpha_gate at the reference site and
    beta_gate in the phase environments. The latter must have been built with
    beta_gate embedded on each basis site and the same inner factors.

    For a basis of num_sites spins, use inner factors exp(1j * q @ b_m).
    This returns the sum over all reference and target basis sites, including
    coincident spins. Divide by num_sites for normalization per physical spin.
    For one spin per site, None or an empty sequence means a unit inner factor.
    """
    if num_sites < 1 or peps_tensor_obj.d != real_d**num_sites:
        raise ValueError("Physical dimension does not match the basis sites.")
    if (
        structure_factor_inner_factors is None
        or len(structure_factor_inner_factors) == 0
    ):
        structure_factor_inner_factors = (1,) * num_sites
    if len(structure_factor_inner_factors) != num_sites:
        raise ValueError("Length mismatch for inner factor list.")

    full_alpha_gate = []
    full_beta_gate = []
    for i, factor in enumerate(structure_factor_inner_factors):
        left_id = jnp.eye(real_d**i)
        right_id = jnp.eye(real_d ** (num_sites - i - 1))
        full_alpha_gate.append(
            jnp.kron(left_id, jnp.kron(alpha_gate, right_id)) * jnp.conj(factor)
        )
        full_beta_gate.append(jnp.kron(left_id, jnp.kron(beta_gate, right_id)) * factor)

    # The product includes all num_sites**2 local correlations, also i == j.
    alpha_beta_gate = sum(full_alpha_gate) @ sum(full_beta_gate)

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site", [peps_tensor_obj.tensor], [peps_tensor_obj], []
    )

    norm = jnp.trace(density_matrix)

    result = jnp.tensordot(density_matrix, alpha_beta_gate, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C1_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C2_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C3_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C4_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T1_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T2_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T3_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T4_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (1, 0))) / norm

    return result
