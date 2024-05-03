from functools import partial

import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction_jitted

from typing import Sequence


@partial(jit, static_argnums=(4, 5))
def calc_structure_factor_expectation(
    peps_tensor_obj: PEPS_Tensor,
    alpha_gate: jnp.ndarray,
    beta_gate: jnp.array,
    structure_factor_inner_factors: Sequence[float],
    real_d: int,
    num_sites: int,
):
    if num_sites > 1:
        Id = jnp.eye(real_d ** (num_sites - 1))

        full_alpha_gate = [jnp.kron(alpha_gate, Id)]

        alpha_gate_tmp = full_alpha_gate[0].reshape((real_d,) * 2 * num_sites)
        for i in range(1, num_sites):
            trans_order = list(range(1, num_sites)) + list(
                range(num_sites + 1, 2 * num_sites)
            )
            trans_order.insert(i, 0)
            trans_order.insert(num_sites + i, num_sites)

            tmp_gate = alpha_gate_tmp.transpose(trans_order).reshape(
                real_d**num_sites, real_d**num_sites
            )

            full_alpha_gate.append(tmp_gate * structure_factor_inner_factors[i].conj())

        alpha_beta_gate = 0
        # alpha_beta_gate_tmp_1 = alpha_beta_gate.reshape((real_d,) * 2 * num_sites)
        alpha_beta_gate_tmp_2 = jnp.kron(
            jnp.kron(alpha_gate, beta_gate), jnp.eye(real_d ** (num_sites - 2))
        ).reshape((real_d,) * 2 * num_sites)

        for i in range(num_sites):
            for j in range(num_sites):
                if i == j:
                    continue

                # if i == j:
                #     trans_order = list(range(1, num_sites)) + list(
                #         range(num_sites + 1, 2 * num_sites)
                #     )
                #     trans_order.insert(i, 0)
                #     trans_order.insert(num_sites + i, num_sites)
                #
                #     tmp_gate = alpha_beta_gate_tmp_1.transpose(trans_order)
                #     tmp_gate = tmp_gate.reshape(real_d**num_sites, real_d**num_sites)
                #
                #     alpha_beta_gate += tmp_gate
                # else:
                trans_order = list(range(2, num_sites)) + list(
                    range(num_sites + 2, 2 * num_sites)
                )
                if i <= j:
                    trans_order.insert(i, 0)
                    trans_order.insert(j, 1)
                    trans_order.insert(num_sites + i, num_sites)
                    trans_order.insert(num_sites + j, num_sites + 1)
                else:
                    trans_order.insert(j, 1)
                    trans_order.insert(i, 0)
                    trans_order.insert(num_sites + j, num_sites + 1)
                    trans_order.insert(num_sites + i, num_sites)

                tmp_gate = alpha_beta_gate_tmp_2.transpose(trans_order)
                tmp_gate = tmp_gate.reshape(real_d**num_sites, real_d**num_sites)

                alpha_beta_gate += (
                    tmp_gate
                    * structure_factor_inner_factors[i].conj()
                    * structure_factor_inner_factors[j]
                )
    else:
        full_alpha_gate = [alpha_gate]
        alpha_beta_gate = alpha_gate @ beta_gate

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site", [peps_tensor_obj.tensor], [peps_tensor_obj], []
    )

    norm = jnp.trace(density_matrix)

    result = jnp.tensordot(density_matrix, alpha_beta_gate, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C1_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C2_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C3_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_C4_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T1_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T2_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T3_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    density_matrix = apply_contraction_jitted(
        "density_matrix_one_site_T4_phase",
        [peps_tensor_obj.tensor],
        [peps_tensor_obj],
        [],
    )

    for g in full_alpha_gate:
        result += jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm

    return result
