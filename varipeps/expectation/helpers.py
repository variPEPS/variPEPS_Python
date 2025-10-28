import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction, Definitions

from typing import Sequence, Tuple


def partially_traced_four_site_density_matrices(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int], Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t_top_left, t_top_right, t_bottom_left, t_bottom_right = peps_tensors
    t_obj_top_left, t_obj_top_right, t_obj_bottom_left, t_obj_bottom_right = (
        peps_tensor_objs
    )
    top_left_i, top_right_i, bottom_left_i, bottom_right_i = open_physical_indices

    for t in ("t_top_left", "t_top_right", "t_bottom_left", "t_bottom_right"):
        locals()[t] = locals()[t].reshape(
            t_left.shape[0],
            t_left.shape[1],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
            t_left.shape[3],
            t_left.shape[4],
        )

    if (
        not hasattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_top_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}",
        )
        or not hasattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_top_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}",
        )
        or not hasattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_bottom_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}",
        )
        or not hasattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_bottom_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}",
        )
    ):
        phys_contraction_i_top_left = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(top_left_i))
        )
        phys_contraction_i_conj_top_left = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(top_left_i))
        )
        for pos, i in enumerate(top_left_i):
            phys_contraction_i_top_left.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_top_left.insert(i - 1, -len(top_left_i) - (pos + 1))
        phys_contraction_i_top_left = tuple(phys_contraction_i_top_left)
        phys_contraction_i_conj_top_left = tuple(phys_contraction_i_conj_top_left)

        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T4", "C1", "T1"]],
            "network": [
                [
                    (3, -2 * len(top_left_i) - 2)
                    + phys_contraction_i_top_left
                    + (-2 * len(top_left_i) - 5, 4),  # tensor
                    (5, -2 * len(top_left_i) - 3)
                    + phys_contraction_i_conj_top_left
                    + (-2 * len(top_left_i) - 6, 6),  # tensor_conj
                    (-2 * len(top_left_i) - 1, 5, 3, 1),  # T4
                    (1, 2),  # C1
                    (2, 4, 6, -2 * len(top_left_i) - 4),  # T1
                ]
            ],
        }
        Definitions._process_def(
            contraction_top_left,
            f"partially_traced_four_site_density_matrices_top_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}",
        )
        setattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_top_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}",
            contraction_top_left,
        )

        phys_contraction_i_top_right = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(top_right_i))
        )
        phys_contraction_i_conj_top_right = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(top_right_i))
        )
        for pos, i in enumerate(top_right_i):
            phys_contraction_i_top_right.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_top_right.insert(
                i - 1, -len(top_right_i) - (pos + 1)
            )
        phys_contraction_i_top_right = tuple(phys_contraction_i_top_right)
        phys_contraction_i_conj_top_right = tuple(phys_contraction_i_conj_top_right)

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2"]],
            "network": [
                [
                    (-2 * len(top_right_i) - 2, -2 * len(top_right_i) - 6)
                    + phys_contraction_i_top_right
                    + (4, 3),  # tensor
                    (-2 * len(top_right_i) - 3, -2 * len(top_right_i) - 5)
                    + phys_contraction_i_conj_top_right
                    + (6, 5),  # tensor_conj
                    (-2 * len(top_right_i) - 1, 3, 5, 1),  # T1
                    (1, 2),  # C2
                    (4, 6, -2 * len(top_right_i) - 4, 2),  # T2
                ]
            ],
        }
        Definitions._process_def(
            contraction_top_right,
            f"partially_traced_four_site_density_matrices_top_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}",
        )
        setattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_top_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}",
            contraction_top_right,
        )

        phys_contraction_i_bottom_left = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(bottom_left_i))
        )
        phys_contraction_i_conj_bottom_left = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(bottom_left_i))
        )
        for pos, i in enumerate(bottom_left_i):
            phys_contraction_i_bottom_left.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_bottom_left.insert(
                i - 1, -len(bottom_left_i) - (pos + 1)
            )
        phys_contraction_i_bottom_left = tuple(phys_contraction_i_bottom_left)
        phys_contraction_i_conj_bottom_left = tuple(phys_contraction_i_conj_bottom_left)

        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T3", "C4", "T4"]],
            "network": [
                [
                    (3, 4)
                    + phys_contraction_i_bottom_left
                    + (
                        -2 * len(bottom_left_i) - 3,
                        -2 * len(bottom_left_i) - 5,
                    ),  # tensor
                    (5, 6)
                    + phys_contraction_i_conj_bottom_left
                    + (
                        -2 * len(bottom_left_i) - 2,
                        -2 * len(bottom_left_i) - 6,
                    ),  # tensor_conj
                    (2, -2 * len(bottom_left_i) - 1, 6, 4),  # T3
                    (2, 1),  # C4
                    (1, 5, 3, -2 * len(bottom_left_i) - 4),  # T4
                ]
            ],
        }
        Definitions._process_def(
            contraction_bottom_left,
            f"partially_traced_four_site_density_matrices_bottom_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}",
        )
        setattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_bottom_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}",
            contraction_bottom_left,
        )

        phys_contraction_i_bottom_right = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(bottom_right_i))
        )
        phys_contraction_i_conj_bottom_right = list(
            range(7, 7 + num_coarse_grained_physical_indices - len(bottom_right_i))
        )
        for pos, i in enumerate(bottom_right_i):
            phys_contraction_i_bottom_right.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_bottom_right.insert(
                i - 1, -len(bottom_right_i) - (pos + 1)
            )
        phys_contraction_i_bottom_right = tuple(phys_contraction_i_bottom_right)
        phys_contraction_i_conj_bottom_right = tuple(
            phys_contraction_i_conj_bottom_right
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
            "network": [
                [
                    (-2 * len(bottom_right_i) - 6, 3)
                    + phys_contraction_i_bottom_right
                    + (4, -2 * len(bottom_right_i) - 3),  # tensor
                    (-2 * len(bottom_right_i) - 5, 5)
                    + phys_contraction_i_conj_bottom_right
                    + (6, -2 * len(bottom_right_i) - 2),  # tensor_conj
                    (4, 6, 2, -2 * len(bottom_right_i) - 1),  # T2
                    (-2 * len(bottom_right_i) - 4, 1, 5, 3),  # T3
                    (1, 2),  # C3
                ]
            ],
        }
        Definitions._process_def(
            contraction_bottom_right,
            f"partially_traced_four_site_density_matrices_bottom_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}",
        )
        setattr(
            Definitions,
            f"partially_traced_four_site_density_matrices_bottom_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}",
            contraction_bottom_right,
        )

    density_top_left = apply_contraction(
        f"partially_traced_four_site_density_matrices_top_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}",
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    if len(top_left_i) > 0:
        density_top_left = density_top_left.reshape(
            real_physical_dimension ** len(top_left_i),
            real_physical_dimension ** len(top_left_i),
            density_top_left.shape[-6],
            density_top_left.shape[-5],
            density_top_left.shape[-4],
            density_top_left.shape[-3],
            density_top_left.shape[-2],
            density_top_left.shape[-1],
        )

    density_top_right = apply_contraction(
        f"partially_traced_four_site_density_matrices_top_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}",
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    if len(top_right_i) > 0:
        density_top_right = density_top_right.reshape(
            real_physical_dimension ** len(top_right_i),
            real_physical_dimension ** len(top_right_i),
            density_top_right.shape[-6],
            density_top_right.shape[-5],
            density_top_right.shape[-4],
            density_top_right.shape[-3],
            density_top_right.shape[-2],
            density_top_right.shape[-1],
        )

    density_bottom_left = apply_contraction(
        f"partially_traced_four_site_density_matrices_bottom_left_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}",
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    if len(bottom_left_i) > 0:
        density_bottom_left = density_bottom_left.reshape(
            real_physical_dimension ** len(bottom_left_i),
            real_physical_dimension ** len(bottom_left_i),
            density_bottom_left.shape[-6],
            density_bottom_left.shape[-5],
            density_bottom_left.shape[-4],
            density_bottom_left.shape[-3],
            density_bottom_left.shape[-2],
            density_bottom_left.shape[-1],
        )

    density_bottom_right = apply_contraction(
        f"partially_traced_four_site_density_matrices_bottom_right_{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}",
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    if len(bottom_right_i) > 0:
        density_bottom_right = density_bottom_right.reshape(
            real_physical_dimension ** len(bottom_right_i),
            real_physical_dimension ** len(bottom_right_i),
            density_bottom_right.shape[-6],
            density_bottom_right.shape[-5],
            density_bottom_right.shape[-4],
            density_bottom_right.shape[-3],
            density_bottom_right.shape[-2],
            density_bottom_right.shape[-1],
        )

    return (
        density_top_left,
        density_top_right,
        density_bottom_left,
        density_bottom_right,
    )
