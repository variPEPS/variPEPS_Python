import jax.numpy as jnp

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction, Definitions

from .physical_indices import build_physical_contraction_indices

from typing import Sequence, Tuple


# ========================================
# N two site triangular gates
# ========================================
def partially_traced_vertical_two_site_density_matrices_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top, t_bottom = peps_tensors
    t_obj_top, t_obj_bottom = peps_tensor_objs
    top_i, bottom_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_vertical_two_site_density_matrices_triangular_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["top", "bottom"], strict=True
        )
    ):

        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=0,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "C2", "C3", "T3b"]],
            "network": [
                [
                    (4, 5, 7, 13, -2 * len(top_i) - 2, 3)
                    + phys_contraction_i_top,  # tensor
                    (9, 10, 11, 14, -2 * len(top_i) - 3, 8)
                    + phys_contraction_i_conj_top,  # tensor_conj
                    (-2 * len(top_i) - 1, 3, 8, 1),  # T6a
                    (1, 4, 9, 2),  # C1
                    (2, 5, 10, 6),  # C2
                    (6, 7, 11, 12),  # C3
                    (12, 13, 14, -2 * len(top_i) - 4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_vertical_two_site_density_matrices_triangular_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "C5", "C6", "T6b"]],
            "network": [
                [
                    (13, -2, 3, 4, 5, 7) + phys_contraction_i_bottom,  # tensor
                    (14, -3, 8, 9, 10, 11)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-4, 3, 8, 1),  # T3a
                    (1, 4, 9, 2),  # C4
                    (2, 5, 10, 6),  # C5
                    (6, 7, 11, 12),  # C6
                    (12, 13, 14, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_vertical_two_site_density_matrices_triangular_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    density_top = apply_contraction(
        (
            f"partially_traced_vertical_two_site_density_matrices_triangular_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    density_top = density_top.reshape(
        real_physical_dimension ** len(top_i),
        real_physical_dimension ** len(top_i),
        density_top.shape[-4],
        density_top.shape[-3],
        density_top.shape[-2],
        density_top.shape[-1],
    )

    density_bottom = apply_contraction(
        (
            f"partially_traced_vertical_two_site_density_matrices_triangular_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    density_bottom = density_bottom.reshape(
        density_bottom.shape[0],
        density_bottom.shape[1],
        density_bottom.shape[2],
        density_bottom.shape[3],
        real_physical_dimension ** len(bottom_i),
        real_physical_dimension ** len(bottom_i),
    )

    return (
        density_top,
        density_bottom,
    )


def partially_traced_horizontal_two_site_density_matrices_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_left, t_right = peps_tensors
    t_obj_left, t_obj_right = peps_tensor_objs
    left_i, right_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_horizontal_two_site_density_matrices_triangular_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["left", "right"], strict=True
        )
    ):

        phys_contraction_i_left, phys_contraction_i_conj_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=0,
            )
        )
        contraction_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "C1", "T1b"]],
            "network": [
                [
                    (7, 13, -2 * len(left_i) - 2, 3, 4, 5)
                    + phys_contraction_i_left,  # tensor
                    (11, 14, -2 * len(left_i) - 3, 8, 9, 10)
                    + phys_contraction_i_conj_left,  # tensor_conj
                    (-2 * len(left_i) - 4, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, 12),  # C1
                    (12, 13, 14, -2 * len(left_i) - 1),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_horizontal_two_site_density_matrices_triangular_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{left_i}"
            ),
            contraction_left,
        )

        phys_contraction_i_right, phys_contraction_i_conj_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "C4", "T4b"]],
            "network": [
                [
                    (3, 4, 5, 7, 13, -2) + phys_contraction_i_right,  # tensor
                    (8, 9, 10, 11, 14, -3)
                    + phys_contraction_i_conj_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T1a
                    (1, 4, 9, 2),  # C2
                    (2, 5, 10, 6),  # C3
                    (6, 7, 11, 12),  # C4
                    (12, 13, 14, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_horizontal_two_site_density_matrices_triangular_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{right_i}"
            ),
            contraction_right,
        )

    density_left = apply_contraction(
        (
            f"partially_traced_horizontal_two_site_density_matrices_triangular_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{left_i}"
        ),
        [t_left],
        [t_obj_left],
        [],
        disable_identity_check=True,
    )
    density_left = density_left.reshape(
        real_physical_dimension ** len(left_i),
        real_physical_dimension ** len(left_i),
        density_left.shape[-4],
        density_left.shape[-3],
        density_left.shape[-2],
        density_left.shape[-1],
    )

    density_right = apply_contraction(
        (
            f"partially_traced_horizontal_two_site_density_matrices_triangular_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{right_i}"
        ),
        [t_right],
        [t_obj_right],
        [],
        disable_identity_check=True,
    )
    density_right = density_right.reshape(
        density_right.shape[0],
        density_right.shape[1],
        density_right.shape[2],
        density_right.shape[3],
        real_physical_dimension ** len(right_i),
        real_physical_dimension ** len(right_i),
    )

    return (
        density_left,
        density_right,
    )


def partially_traced_diagonal_two_site_density_matrices_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top, t_bottom = peps_tensors
    t_obj_top, t_obj_bottom = peps_tensor_objs
    top_i, bottom_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_diagonal_two_site_density_matrices_triangular_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["top", "bottom"], strict=True
        )
    ):

        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=0,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "C2", "T2b"]],
            "network": [
                [
                    (5, 7, 13, -2 * len(top_i) - 2, 3, 4)
                    + phys_contraction_i_top,  # tensor
                    (10, 11, 14, -2 * len(top_i) - 3, 8, 9)
                    + phys_contraction_i_conj_top,  # tensor_conj
                    (-2 * len(top_i) - 1, 3, 8, 1),  # T5a
                    (1, 4, 9, 2),  # C6
                    (2, 5, 10, 6),  # C1
                    (6, 7, 11, 12),  # C2
                    (12, 13, 14, -2 * len(top_i) - 4),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_diagonal_two_site_density_matrices_triangular_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "C5", "T5b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, 13) + phys_contraction_i_bottom,  # tensor
                    (-3, 8, 9, 10, 11, 14)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-4, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, 12),  # C5
                    (12, 13, 14, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_diagonal_two_site_density_matrices_triangular_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    density_top = apply_contraction(
        (
            f"partially_traced_diagonal_two_site_density_matrices_triangular_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    density_top = density_top.reshape(
        real_physical_dimension ** len(top_i),
        real_physical_dimension ** len(top_i),
        density_top.shape[-4],
        density_top.shape[-3],
        density_top.shape[-2],
        density_top.shape[-1],
    )

    density_bottom = apply_contraction(
        (
            f"partially_traced_diagonal_two_site_density_matrices_triangular_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    density_bottom = density_bottom.reshape(
        density_bottom.shape[0],
        density_bottom.shape[1],
        density_bottom.shape[2],
        density_bottom.shape[3],
        real_physical_dimension ** len(bottom_i),
        real_physical_dimension ** len(bottom_i),
    )

    return (
        density_top,
        density_bottom,
    )


# ========================================
# N three site triangular gates
# ========================================
def partially_traced_upward_three_site_density_matrix_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    # top = corner_90; bottom_left = corner_210; bottom_right = corner_330
    t_corner_90, t_corner_210, t_corner_330 = peps_tensors
    t_obj_corner_90, t_obj_corner_210, t_obj_corner_330 = peps_tensor_objs
    corner_90_i, corner_210_i, corner_330_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_upward_three_site_density_matrix_triangular_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["top", "bottom_left", "bottom_right"], strict=True
        )
    ):

        # ======================================================================
        # top = 90º corner
        phys_contraction_i_corner_90, phys_contraction_i_conj_corner_90 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_90_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_90 = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "C2", "T2b"]],
            "network": [
                [
                    (3, 4, 10, -5, -2, 5) + phys_contraction_i_corner_90,  # tensor
                    (6, 7, 11, -6, -3, 8)
                    + phys_contraction_i_conj_corner_90,  # tensor_conj
                    (-1, 5, 8, 1),  # T6a
                    (1, 3, 6, 2),  # C1
                    (2, 4, 7, 9),  # C2
                    (9, 10, 11, -4),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_upward_three_site_density_matrix_triangular_corner_90_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_90_i}"
            ),
            contraction_corner_90,
        )

        # ======================================================================
        # bottom_left = 210º corner
        phys_contraction_i_corner_210, phys_contraction_i_conj_corner_210 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_210_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_210 = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "T6b"]],
            "network": [
                [
                    (10, -5, -2, 4, 5, 3) + phys_contraction_i_corner_210,  # tensor
                    (11, -6, -3, 7, 8, 6)
                    + phys_contraction_i_conj_corner_210,  # tensor_conj
                    (-1, 4, 7, 1),  # T4a
                    (1, 5, 8, 2),  # C5
                    (2, 3, 6, 9),  # C6
                    (9, 10, 11, -4),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_upward_three_site_density_matrix_triangular_corner_210_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_210_i}"
            ),
            contraction_corner_210,
        )

        # ======================================================================
        # bottom_right = 330º corner
        phys_contraction_i_corner_330, phys_contraction_i_conj_corner_330 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_330_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_330 = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "T4b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 10, -5) + phys_contraction_i_corner_330,  # tensor
                    (-3, 6, 7, 8, 11, -6)
                    + phys_contraction_i_conj_corner_330,  # tensor_conj
                    (-1, 3, 6, 1),  # T2a
                    (1, 4, 7, 2),  # C3
                    (2, 5, 8, 9),  # C4
                    (9, 10, 11, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_upward_three_site_density_matrix_triangular_corner_330_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_330_i}"
            ),
            contraction_corner_330,
        )

    # ======================================================================
    # Apply contractions
    rho_corner_90 = apply_contraction(
        (
            f"partially_traced_upward_three_site_density_matrix_triangular_corner_90_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_90_i}"
        ),
        [t_corner_90],
        [t_obj_corner_90],
        [],
        disable_identity_check=True,
    )
    # len(corner_90_i) > 0 is always true by default
    rho_corner_90 = rho_corner_90.reshape(
        rho_corner_90.shape[0],  # T6a↙
        rho_corner_90.shape[1],  # ket↙
        rho_corner_90.shape[2],  # bra↙
        rho_corner_90.shape[3],  # T2b↘
        rho_corner_90.shape[4],  # ket↘
        rho_corner_90.shape[5],  # bra↘
        real_physical_dimension ** len(corner_90_i),  # physical ket
        real_physical_dimension ** len(corner_90_i),  # physical bra
    )

    rho_corner_210 = apply_contraction(
        (
            f"partially_traced_upward_three_site_density_matrix_triangular_corner_210_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_210_i}"
        ),
        [t_corner_210],
        [t_obj_corner_210],
        [],
        disable_identity_check=True,
    )
    # len(corner_210_i) > 0 is always true by default
    rho_corner_210 = rho_corner_210.reshape(
        rho_corner_210.shape[0],  # T4a→
        rho_corner_210.shape[1],  # ket→
        rho_corner_210.shape[2],  # bra→
        rho_corner_210.shape[3],  # T6b↗
        rho_corner_210.shape[4],  # ket↗
        rho_corner_210.shape[5],  # bra↗
        real_physical_dimension ** len(corner_210_i),  # physical ket
        real_physical_dimension ** len(corner_210_i),  # physical bra
    )

    rho_corner_330 = apply_contraction(
        (
            f"partially_traced_upward_three_site_density_matrix_triangular_corner_330_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_330_i}"
        ),
        [t_corner_330],
        [t_obj_corner_330],
        [],
        disable_identity_check=True,
    )
    # len(corner_330_i) > 0 is always true by default
    rho_corner_330 = rho_corner_330.reshape(
        rho_corner_330.shape[0],  # T2a↖
        rho_corner_330.shape[1],  # ket↖
        rho_corner_330.shape[2],  # bra↖
        rho_corner_330.shape[3],  # T4b←
        rho_corner_330.shape[4],  # ket←
        rho_corner_330.shape[5],  # bra←
        real_physical_dimension ** len(corner_330_i),  # physical ket
        real_physical_dimension ** len(corner_330_i),  # physical bra
    )

    return (
        rho_corner_90,
        rho_corner_210,
        rho_corner_330,
    )


def partially_traced_downward_three_site_density_matrix_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    # top_left = 150º; top_right = 30º; bottom = 270º
    t_corner_150, t_corner_30, t_corner_270 = peps_tensors
    t_obj_corner_150, t_obj_corner_30, t_obj_corner_270 = peps_tensor_objs
    corner_150_i, corner_30_i, corner_270_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_downward_three_site_density_matrix_triangular_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["corner_150", "corner_30", "corner_270"],
            strict=True,
        )
    ):

        # ======================================================================
        # top_left = 150º corner
        phys_contraction_i_corner_150, phys_contraction_i_conj_corner_150 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_150_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_150 = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "T1b"]],
            "network": [
                [
                    (3, 10, -5, -2, 4, 5) + phys_contraction_i_corner_150,  # tensor
                    (6, 11, -6, -3, 7, 8)
                    + phys_contraction_i_conj_corner_150,  # tensor_conj
                    (-1, 4, 7, 1),  # T5a
                    (1, 5, 8, 2),  # C6
                    (2, 3, 6, 9),  # C1
                    (9, 10, 11, -4),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_downward_three_site_density_matrix_triangular_corner_150_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_150_i}"
            ),
            contraction_corner_150,
        )

        # ======================================================================
        # top_right = 30º corner
        phys_contraction_i_corner_30, phys_contraction_i_conj_corner_30 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_30_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_30 = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "T3b"]],
            "network": [
                [
                    (3, 4, 5, 10, -5, -2) + phys_contraction_i_corner_30,  # tensor
                    (6, 7, 8, 11, -6, -3)
                    + phys_contraction_i_conj_corner_30,  # tensor_conj
                    (-1, 3, 6, 1),  # T1a
                    (1, 4, 7, 2),  # C2
                    (2, 5, 8, 9),  # C3
                    (9, 10, 11, -4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_downward_three_site_density_matrix_triangular_corner_30_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_30_i}"
            ),
            contraction_corner_30,
        )

        # ======================================================================
        # bottom = 270º corner
        phys_contraction_i_corner_270, phys_contraction_i_conj_corner_270 = (
            build_physical_contraction_indices(
                site_open_physical_indices=corner_270_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_corner_270 = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "C5", "T5b"]],
            "network": [
                [
                    (-5, -2, 3, 4, 5, 10) + phys_contraction_i_corner_270,  # tensor
                    (-6, -3, 6, 7, 8, 11)
                    + phys_contraction_i_conj_corner_270,  # tensor_conj
                    (-1, 3, 6, 1),  # T3a
                    (1, 4, 7, 2),  # C4
                    (2, 5, 8, 9),  # C5
                    (9, 10, 11, -4),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_downward_three_site_density_matrix_triangular_corner_270_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_270_i}"
            ),
            contraction_corner_270,
        )

    # ======================================================================
    # Apply contractions
    rho_corner_150 = apply_contraction(
        (
            f"partially_traced_downward_three_site_density_matrix_triangular_corner_150_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_150_i}"
        ),
        [t_corner_150],
        [t_obj_corner_150],
        [],
        disable_identity_check=True,
    )
    # len(corner_150_i) > 0 is always true by default
    rho_corner_150 = rho_corner_150.reshape(
        rho_corner_150.shape[0],  # T5a↘
        rho_corner_150.shape[1],  # ket↘
        rho_corner_150.shape[2],  # bra↘
        rho_corner_150.shape[3],  # T1b→
        rho_corner_150.shape[4],  # ket→
        rho_corner_150.shape[5],  # bra→
        real_physical_dimension ** len(corner_150_i),  # physical ket
        real_physical_dimension ** len(corner_150_i),  # physical bra
    )

    rho_corner_30 = apply_contraction(
        (
            f"partially_traced_downward_three_site_density_matrix_triangular_corner_30_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_30_i}"
        ),
        [t_corner_30],
        [t_obj_corner_30],
        [],
        disable_identity_check=True,
    )
    # len(corner_30_i) > 0 is always true by default
    rho_corner_30 = rho_corner_30.reshape(
        rho_corner_30.shape[0],  # T1a←
        rho_corner_30.shape[1],  # ket←
        rho_corner_30.shape[2],  # bra←
        rho_corner_30.shape[3],  # T3b↙
        rho_corner_30.shape[4],  # ket↙
        rho_corner_30.shape[5],  # bra↙
        real_physical_dimension ** len(corner_30_i),  # physical ket
        real_physical_dimension ** len(corner_30_i),  # physical bra
    )

    rho_corner_270 = apply_contraction(
        (
            f"partially_traced_downward_three_site_density_matrix_triangular_corner_270_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{corner_270_i}"
        ),
        [t_corner_270],
        [t_obj_corner_270],
        [],
        disable_identity_check=True,
    )
    # len(corner_270_i) > 0 is always true by default
    rho_corner_270 = rho_corner_270.reshape(
        rho_corner_270.shape[0],  # T3a↗
        rho_corner_270.shape[1],  # ket↗
        rho_corner_270.shape[2],  # bra↗
        rho_corner_270.shape[3],  # T5b↖
        rho_corner_270.shape[4],  # ket↖
        rho_corner_270.shape[5],  # bra↖
        real_physical_dimension ** len(corner_270_i),  # physical ket
        real_physical_dimension ** len(corner_270_i),  # physical bra
    )

    return (
        rho_corner_150,
        rho_corner_30,
        rho_corner_270,
    )


# ========================================
# NN four site triangular gates
# ========================================
def partially_traced_four_site_density_matrices_triangular_nx_py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top_left, t_top_right, t_bottom_left, t_bottom_right = peps_tensors
    t_obj_top_left, t_obj_top_right, t_obj_bottom_left, t_obj_bottom_right = (
        peps_tensor_objs
    )
    top_left_i, top_right_i, bottom_left_i, bottom_right_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_four_site_density_matrices_triangular_nx_py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["top_left", "top_right", "bottom_left", "bottom_right"],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "T1b"]],
            "network": [
                [
                    (4, 5, -7, -4, -2, 3) + phys_contraction_i_top_left,  # tensor
                    (7, 8, -8, -5, -3, 6)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 6, 1),  # T6a
                    (1, 4, 7, 2),  # C1
                    (2, 5, 8, -6),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_nx_py_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TR corner (has all open indices by default)
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "T3b"]],
            "network": [
                [
                    (3, 4, 5, 7, -5, -2) + phys_contraction_i_top_right,  # tensor
                    (8, 9, 10, 11, -6, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T1a
                    (1, 4, 9, 2),  # C2
                    (2, 5, 10, 6),  # C3
                    (6, 7, 11, -4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_nx_py_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner (has all open indices by default)
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "T6b"]],
            "network": [
                [
                    (
                        7,
                        -5,
                        -2,
                        3,
                        4,
                        5,
                    )
                    + phys_contraction_i_bottom_left,  # tensor
                    (11, -6, -3, 8, 9, 10)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, -4),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_nx_py_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # BR corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "T4b"]],
            "network": [
                [
                    (-4, -7, 3, 4, 5, -2) + phys_contraction_i_bottom_right,  # tensor
                    (-5, -8, 6, 7, 8, -3)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-6, 3, 6, 1),  # T3a
                    (1, 4, 7, 2),  # C4
                    (2, 5, 8, -1),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_nx_py_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_nx_py_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T6a↙
            rho_top_left.shape[1],  # ket↙
            rho_top_left.shape[2],  # bra↙
            rho_top_left.shape[3],  # ket↘
            rho_top_left.shape[4],  # bra↘
            rho_top_left.shape[5],  # T1b→
            rho_top_left.shape[6],  # ket→
            rho_top_left.shape[7],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_nx_py_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    # len(top_right_i) > 0 is always true by default
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # T3b↙
            rho_top_right.shape[4],  # ket↙
            rho_top_right.shape[5],  # bra↙
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_nx_py_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    # len(bottom_left_i) > 0 is always true by default
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T4a→
            rho_bottom_left.shape[1],  # ket→
            rho_bottom_left.shape[2],  # bra→
            rho_bottom_left.shape[3],  # T6b↗
            rho_bottom_left.shape[4],  # ket↗
            rho_bottom_left.shape[5],  # bra↗
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_nx_py_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T4b←
            rho_bottom_right.shape[1],  # ket←
            rho_bottom_right.shape[2],  # bra←
            rho_bottom_right.shape[3],  # ket↖
            rho_bottom_right.shape[4],  # bra↖
            rho_bottom_right.shape[5],  # T3a↗
            rho_bottom_right.shape[6],  # ket↗
            rho_bottom_right.shape[7],  # bra↗
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_right,
    )


def partially_traced_four_site_density_matrices_triangular_px_2py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top_left, t_top_right, t_bottom_left, t_bottom_right = peps_tensors
    t_obj_top_left, t_obj_top_right, t_obj_bottom_left, t_obj_bottom_right = (
        peps_tensor_objs
    )
    top_left_i, top_right_i, bottom_left_i, bottom_right_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_four_site_density_matrices_triangular_px_2py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["top_left", "top_right", "bottom_left", "bottom_right"],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has all open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "T1b"]],
            "network": [
                [
                    (5, 7, -5, -2, 3, 4) + phys_contraction_i_top_left,  # tensor
                    (10, 11, -6, -3, 8, 9)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T5a
                    (1, 4, 9, 2),  # C6
                    (2, 5, 10, 6),  # C1
                    (6, 7, 11, -4),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_px_2py_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TR corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "T2b"]],
            "network": [
                [
                    (3, 4, 5, -7, -4, -2) + phys_contraction_i_top_right,  # tensor
                    (6, 7, 8, -8, -5, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T1a
                    (1, 4, 7, 2),  # C2
                    (2, 5, 8, -6),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_px_2py_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "T5b"]],
            "network": [
                [
                    (-2, -4, -7, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (-3, -5, -8, 6, 7, 8)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-6, 3, 6, 1),  # T4a
                    (1, 4, 7, 2),  # C5
                    (2, 5, 8, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_px_2py_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # BR corner (has all open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "T4b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, -5) + phys_contraction_i_bottom_right,  # tensor
                    (-3, 8, 9, 10, 11, -6)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_px_2py_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_px_2py_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    # len(top_left_i) > 0 is always true by default
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T5a↘
            rho_top_left.shape[1],  # ket↘
            rho_top_left.shape[2],  # bra↘
            rho_top_left.shape[3],  # T1b→
            rho_top_left.shape[4],  # ket→
            rho_top_left.shape[5],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_px_2py_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # ket↙
            rho_top_right.shape[4],  # bra↙
            rho_top_right.shape[5],  # T2b↘
            rho_top_right.shape[6],  # ket↘
            rho_top_right.shape[7],  # bra↘
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_px_2py_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T5b↖
            rho_bottom_left.shape[1],  # ket↖
            rho_bottom_left.shape[2],  # bra↖
            rho_bottom_left.shape[3],  # ket↗
            rho_bottom_left.shape[4],  # bra↗
            rho_bottom_left.shape[5],  # T4a→
            rho_bottom_left.shape[6],  # ket→
            rho_bottom_left.shape[7],  # bra→
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_px_2py_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    # len(bottom_right_i) > 0 is always true by default
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T2a↖
            rho_bottom_right.shape[1],  # ket↖
            rho_bottom_right.shape[2],  # bra↖
            rho_bottom_right.shape[3],  # T4b←
            rho_bottom_right.shape[4],  # ket←
            rho_bottom_right.shape[5],  # bra←
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_right,
    )


def partially_traced_four_site_density_matrices_triangular_2px_py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top, t_middle_left, t_middle_right, t_bottom = peps_tensors
    t_obj_top, t_obj_middle_left, t_obj_middle_right, t_obj_bottom = peps_tensor_objs
    top_i, middle_left_i, middle_right_i, bottom_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_four_site_density_matrices_triangular_2px_py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["top", "middle_left", "middle_right", "bottom"],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has all open indices by default)
        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "C2", "T2b"]],
            "network": [
                [
                    (4, 5, 7, -5, -2, 3) + phys_contraction_i_top,  # tensor
                    (9, 10, 11, -6, -3, 8) + phys_contraction_i_conj_top,  # tensor_conj
                    (-1, 3, 8, 1),  # T6a
                    (1, 4, 9, 2),  # C1
                    (2, 5, 10, 6),  # C2
                    (6, 7, 11, -4),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_2px_py_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        # ======================================================================
        # ML corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_middle_left, phys_contraction_i_conj_middle_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=middle_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_middle_left = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "T6b"]],
            "network": [
                [
                    (5, -2, -4, -7, 3, 4) + phys_contraction_i_middle_left,  # tensor
                    (8, -3, -5, -8, 6, 7)
                    + phys_contraction_i_conj_middle_left,  # tensor_conj
                    (-6, 3, 6, 1),  # T5a
                    (1, 4, 7, 2),  # C6
                    (2, 5, 8, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_2px_py_middle_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_left_i}"
            ),
            contraction_middle_left,
        )

        # ======================================================================
        # MR corner (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_middle_right, phys_contraction_i_conj_middle_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=middle_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_middle_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "T3b"]],
            "network": [
                [
                    (-2, 3, 4, 5, -7, -4) + phys_contraction_i_middle_right,  # tensor
                    (-3, 6, 7, 8, -8, -5)
                    + phys_contraction_i_conj_middle_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T2a
                    (1, 4, 7, 2),  # C3
                    (2, 5, 8, -6),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_2px_py_middle_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_right_i}"
            ),
            contraction_middle_right,
        )

        # ======================================================================
        # BR corner (has all open indices by default)
        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "C5", "T5b"]],
            "network": [
                [
                    (-5, -2, 3, 4, 5, 7) + phys_contraction_i_bottom,  # tensor
                    (-6, -3, 8, 9, 10, 11)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-1, 3, 8, 1),  # T3a
                    (1, 4, 9, 2),  # C4
                    (2, 5, 10, 6),  # C5
                    (6, 7, 11, -4),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_four_site_density_matrices_triangular_2px_py_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    # ======================================================================
    # Apply contractions
    rho_top = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_2px_py_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    # len(top_i) > 0 is always true by default
    if top_i or not squeeze_closed_physical_legs:
        rho_top = rho_top.reshape(
            rho_top.shape[0],  # T6a↙
            rho_top.shape[1],  # ket↙
            rho_top.shape[2],  # bra↙
            rho_top.shape[3],  # T2b↘
            rho_top.shape[4],  # ket↘
            rho_top.shape[5],  # bra↘
            real_physical_dimension ** len(top_i),  # physical ket
            real_physical_dimension ** len(top_i),  # physical bra
        )

    rho_middle_left = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_2px_py_middle_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_left_i}"
        ),
        [t_middle_left],
        [t_obj_middle_left],
        [],
        disable_identity_check=True,
    )
    if middle_left_i or not squeeze_closed_physical_legs:
        rho_middle_left = rho_middle_left.reshape(
            rho_middle_left.shape[0],  # T6b↗
            rho_middle_left.shape[1],  # ket↗
            rho_middle_left.shape[2],  # bra↗
            rho_middle_left.shape[3],  # ket→
            rho_middle_left.shape[4],  # bra→
            rho_middle_left.shape[5],  # T5a↘
            rho_middle_left.shape[6],  # ket↘
            rho_middle_left.shape[7],  # bra↘
            real_physical_dimension ** len(middle_left_i),  # physical ket
            real_physical_dimension ** len(middle_left_i),  # physical bra
        )

    rho_middle_right = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_2px_py_middle_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_right_i}"
        ),
        [t_middle_right],
        [t_obj_middle_right],
        [],
        disable_identity_check=True,
    )
    if middle_right_i or not squeeze_closed_physical_legs:
        rho_middle_right = rho_middle_right.reshape(
            rho_middle_right.shape[0],  # T2a↖
            rho_middle_right.shape[1],  # ket↖
            rho_middle_right.shape[2],  # bra↖
            rho_middle_right.shape[3],  # ket←
            rho_middle_right.shape[4],  # bra←
            rho_middle_right.shape[5],  # T3b↙
            rho_middle_right.shape[6],  # ket↙
            rho_middle_right.shape[7],  # bra↙
            real_physical_dimension ** len(middle_right_i),  # physical ket
            real_physical_dimension ** len(middle_right_i),  # physical bra
        )

    rho_bottom = apply_contraction(
        (
            f"partially_traced_four_site_density_matrices_triangular_2px_py_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    # len(bottom_i) > 0 is always true by default
    if bottom_i or not squeeze_closed_physical_legs:
        rho_bottom = rho_bottom.reshape(
            rho_bottom.shape[0],  # T3a↗
            rho_bottom.shape[1],  # ket↗
            rho_bottom.shape[2],  # bra↗
            rho_bottom.shape[3],  # T5b↖
            rho_bottom.shape[4],  # ket↖
            rho_bottom.shape[5],  # bra↖
            real_physical_dimension ** len(bottom_i),  # physical ket
            real_physical_dimension ** len(bottom_i),  # physical bra
        )

    return (
        rho_top,
        rho_middle_left,
        rho_middle_right,
        rho_bottom,
    )


# ========================================
# NN three site triangular gates
# ========================================
def partially_traced_three_site_density_matrix_triangular_horizontal_2(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_left, t_middle, t_right = peps_tensors
    t_obj_left, t_obj_middle, t_obj_right = peps_tensor_objs
    left_i, middle_i, right_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_three_site_density_matrix_triangular_horizontal_2_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["left", "middle", "right"], strict=True
        )
    ):

        # ======================================================================
        # Left (has all open indices by default)
        phys_contraction_i_left, phys_contraction_i_conj_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "C1", "T1b"]],
            "network": [
                [
                    (7, 13, -2, 3, 4, 5) + phys_contraction_i_left,  # tensor
                    (11, 14, -3, 8, 9, 10)
                    + phys_contraction_i_conj_left,  # tensor_conj
                    (-4, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, 12),  # C1
                    (12, 13, 14, -1),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_horizontal_2_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{left_i}"
            ),
            contraction_left,
        )

        # ======================================================================
        # Middle (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_middle, phys_contraction_i_conj_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=11,
                negative_offset=-8,
            )
        )
        contraction_middle = {
            "tensors": [["tensor", "tensor_conj", "T1a", "T1b", "T4a", "T4b"]],
            "network": [
                [
                    (3, 4, -6, 7, 8, -2) + phys_contraction_i_middle,  # tensor
                    (5, 6, -7, 9, 10, -3)
                    + phys_contraction_i_conj_middle,  # tensor_conj
                    (-1, 3, 5, 1),  # T1a
                    (1, 4, 6, -5),  # T1b
                    (-8, 7, 9, 2),  # T4a
                    (2, 8, 10, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_horizontal_2_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
            ),
            contraction_middle,
        )

        # ======================================================================
        # Right (has all open indices by default)
        phys_contraction_i_right, phys_contraction_i_conj_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "C4", "T4b"]],
            "network": [
                [
                    (3, 4, 5, 7, 13, -2) + phys_contraction_i_right,  # tensor
                    (8, 9, 10, 11, 14, -3)
                    + phys_contraction_i_conj_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T1a
                    (1, 4, 9, 2),  # C2
                    (2, 5, 10, 6),  # C3
                    (6, 7, 11, 12),  # C4
                    (12, 13, 14, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_horizontal_2_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{right_i}"
            ),
            contraction_right,
        )

    # ======================================================================
    # Apply contractions
    rho_left = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_horizontal_2_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{left_i}"
        ),
        [t_left],
        [t_obj_left],
        [],
        disable_identity_check=True,
    )
    # len(left_i) > 0 is always true by default
    if left_i or not squeeze_closed_physical_legs:
        rho_left = rho_left.reshape(
            rho_left.shape[0],  # T1b→
            rho_left.shape[1],  # ket→
            rho_left.shape[2],  # bra→
            rho_left.shape[3],  # T4a→
            real_physical_dimension ** len(left_i),  # physical ket
            real_physical_dimension ** len(left_i),  # physical bra
        )

    rho_middle = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_horizontal_2_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
        ),
        [t_middle],
        [t_obj_middle],
        [],
        disable_identity_check=True,
    )
    if middle_i or not squeeze_closed_physical_legs:
        rho_middle = rho_middle.reshape(
            rho_middle.shape[0],  # T1a←
            rho_middle.shape[1],  # ket←
            rho_middle.shape[2],  # bra←
            rho_middle.shape[3],  # T4b←
            rho_middle.shape[4],  # T1b→
            rho_middle.shape[5],  # ket→
            rho_middle.shape[6],  # bra→
            rho_middle.shape[7],  # T4a→
            real_physical_dimension ** len(middle_i),  # physical ket
            real_physical_dimension ** len(middle_i),  # physical bra
        )

    rho_right = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_horizontal_2_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{right_i}"
        ),
        [t_right],
        [t_obj_right],
        [],
        disable_identity_check=True,
    )
    # len(right_i) > 0 is always true by default
    if right_i or not squeeze_closed_physical_legs:
        rho_right = rho_right.reshape(
            rho_right.shape[0],  # T1a←
            rho_right.shape[1],  # ket←
            rho_right.shape[2],  # bra←
            rho_right.shape[3],  # T4b←
            real_physical_dimension ** len(right_i),  # physical ket
            real_physical_dimension ** len(right_i),  # physical bra
        )

    return (
        rho_left,
        rho_middle,
        rho_right,
    )


def partially_traced_three_site_density_matrix_triangular_vertical_2(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top, t_middle, t_bottom = peps_tensors
    t_obj_top, t_obj_middle, t_obj_bottom = peps_tensor_objs
    top_i, middle_i, bottom_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_three_site_density_matrix_triangular_vertical_2_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["top", "middle", "bottom"], strict=True
        )
    ):

        # ======================================================================
        # top (has all open indices by default)
        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "C2", "C3", "T3b"]],
            "network": [
                [
                    (4, 5, 7, 13, -2, 3) + phys_contraction_i_top,  # tensor
                    (9, 10, 11, 14, -3, 8) + phys_contraction_i_conj_top,  # tensor_conj
                    (-1, 3, 8, 1),  # T6a
                    (1, 4, 9, 2),  # C1
                    (2, 5, 10, 6),  # C2
                    (6, 7, 11, 12),  # C3
                    (12, 13, 14, -4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_vertical_2_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        # ======================================================================
        # Middle (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_middle, phys_contraction_i_conj_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=11,
                negative_offset=-8,
            )
        )
        contraction_middle = {
            "tensors": [["tensor", "tensor_conj", "T3a", "T3b", "T6a", "T6b"]],
            "network": [
                [
                    (3, -2, 4, 5, -6, 6) + phys_contraction_i_middle,  # tensor
                    (7, -3, 8, 9, -7, 10)
                    + phys_contraction_i_conj_middle,  # tensor_conj
                    (-4, 4, 8, 1),  # T3a
                    (1, 5, 9, -8),  # T3b
                    (-5, 6, 10, 2),  # T6a
                    (2, 3, 7, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_vertical_2_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
            ),
            contraction_middle,
        )

        # ======================================================================
        # bottom (has all open indices by default)
        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "C5", "C6", "T6b"]],
            "network": [
                [
                    (13, -2, 3, 4, 5, 7) + phys_contraction_i_bottom,  # tensor
                    (14, -3, 8, 9, 10, 11)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-4, 3, 8, 1),  # T3a
                    (1, 4, 9, 2),  # C4
                    (2, 5, 10, 6),  # C5
                    (6, 7, 11, 12),  # C6
                    (12, 13, 14, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_vertical_2_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    # ======================================================================
    # Apply contractions
    rho_top = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_vertical_2_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    # len(top_i) > 0 is always true by default
    if top_i or not squeeze_closed_physical_legs:
        rho_top = rho_top.reshape(
            rho_top.shape[0],  # T6a↙
            rho_top.shape[1],  # ket↙
            rho_top.shape[2],  # bra↙
            rho_top.shape[3],  # T3b↙
            real_physical_dimension ** len(top_i),  # physical ket
            real_physical_dimension ** len(top_i),  # physical bra
        )

    rho_middle = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_vertical_2_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
        ),
        [t_middle],
        [t_obj_middle],
        [],
        disable_identity_check=True,
    )
    if middle_i or not squeeze_closed_physical_legs:
        rho_middle = rho_middle.reshape(
            rho_middle.shape[0],  # T6b↗
            rho_middle.shape[1],  # ket↗
            rho_middle.shape[2],  # bra↗
            rho_middle.shape[3],  # T3a↗
            rho_middle.shape[4],  # T6a↙
            rho_middle.shape[5],  # ket↙
            rho_middle.shape[6],  # bra↙
            rho_middle.shape[7],  # T3b↙
            real_physical_dimension ** len(middle_i),  # physical ket
            real_physical_dimension ** len(middle_i),  # physical bra
        )

    rho_bottom = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_vertical_2_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    # len(bottom_i) > 0 is always true by default
    if bottom_i or not squeeze_closed_physical_legs:
        rho_bottom = rho_bottom.reshape(
            rho_bottom.shape[0],  # T6b↗
            rho_bottom.shape[1],  # ket↗
            rho_bottom.shape[2],  # bra↗
            rho_bottom.shape[3],  # T3a↗
            real_physical_dimension ** len(bottom_i),  # physical ket
            real_physical_dimension ** len(bottom_i),  # physical bra
        )

    return (
        rho_top,
        rho_middle,
        rho_bottom,
    )


def partially_traced_three_site_density_matrix_triangular_diagonal_2(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    t_top, t_middle, t_bottom = peps_tensors
    t_obj_top, t_obj_middle, t_obj_bottom = peps_tensor_objs
    top_i, middle_i, bottom_i = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_three_site_density_matrix_triangular_diagonal_2_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices, ["top", "middle", "bottom"], strict=True
        )
    ):

        # ======================================================================
        # top (has all open indices by default)
        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "C2", "T2b"]],
            "network": [
                [
                    (5, 7, 13, -2, 3, 4) + phys_contraction_i_top,  # tensor
                    (10, 11, 14, -3, 8, 9) + phys_contraction_i_conj_top,  # tensor_conj
                    (-1, 3, 8, 1),  # T5a
                    (1, 4, 9, 2),  # C6
                    (2, 5, 10, 6),  # C1
                    (6, 7, 11, 12),  # C2
                    (12, 13, 14, -4),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_diagonal_2_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        # ======================================================================
        # Middle (does not have open indices by default, only in 3-/4- site case)
        phys_contraction_i_middle, phys_contraction_i_conj_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=11,
                negative_offset=-8,
            )
        )
        contraction_middle = {
            "tensors": [["tensor", "tensor_conj", "T2a", "T2b", "T5a", "T5b"]],
            "network": [
                [
                    (-2, 3, 4, -6, 5, 6) + phys_contraction_i_middle,  # tensor
                    (-3, 7, 8, -7, 9, 10)
                    + phys_contraction_i_conj_middle,  # tensor_conj
                    (-4, 3, 7, 1),  # T2a
                    (1, 4, 8, -8),  # T2b
                    (-5, 5, 9, 2),  # T5a
                    (2, 6, 10, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_diagonal_2_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
            ),
            contraction_middle,
        )

        # ======================================================================
        # bottom (has all open indices by default)
        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=15,
                negative_offset=-4,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "C5", "T5b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, 13) + phys_contraction_i_bottom,  # tensor
                    (-3, 8, 9, 10, 11, 14)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-4, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, 12),  # C5
                    (12, 13, 14, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_three_site_density_matrix_triangular_diagonal_2_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    # ======================================================================
    # Apply contractions
    rho_top = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_diagonal_2_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    # len(top_i) > 0 is always true by default
    if top_i or not squeeze_closed_physical_legs:
        rho_top = rho_top.reshape(
            rho_top.shape[0],  # T5a↘
            rho_top.shape[1],  # ket↘
            rho_top.shape[2],  # bra↘
            rho_top.shape[3],  # T3b↘
            real_physical_dimension ** len(top_i),  # physical ket
            real_physical_dimension ** len(top_i),  # physical bra
        )

    rho_middle = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_diagonal_2_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{middle_i}"
        ),
        [t_middle],
        [t_obj_middle],
        [],
        disable_identity_check=True,
    )
    if middle_i or not squeeze_closed_physical_legs:
        rho_middle = rho_middle.reshape(
            rho_middle.shape[0],  # T5b↖
            rho_middle.shape[1],  # ket↖
            rho_middle.shape[2],  # bra↖
            rho_middle.shape[3],  # T2a↖
            rho_middle.shape[4],  # T5a↘
            rho_middle.shape[5],  # ket↘
            rho_middle.shape[6],  # bra↘
            rho_middle.shape[7],  # T2b↘
            real_physical_dimension ** len(middle_i),  # physical ket
            real_physical_dimension ** len(middle_i),  # physical bra
        )

    rho_bottom = apply_contraction(
        (
            f"partially_traced_three_site_density_matrix_triangular_diagonal_2_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    # len(bottom_i) > 0 is always true by default
    if bottom_i or not squeeze_closed_physical_legs:
        rho_bottom = rho_bottom.reshape(
            rho_bottom.shape[0],  # T5b↖
            rho_bottom.shape[1],  # ket↖
            rho_bottom.shape[2],  # bra↖
            rho_bottom.shape[3],  # T2a↖
            real_physical_dimension ** len(bottom_i),  # physical ket
            real_physical_dimension ** len(bottom_i),  # physical bra
        )

    return (
        rho_top,
        rho_middle,
        rho_bottom,
    )


# ========================================
# NN five site triangular gates
# ========================================
def partially_traced_five_site_density_matrix_triangular_trapezoid_upward(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top_left,
        t_top_right,
        t_bottom_left,
        t_bottom_middle,
        t_bottom_right,
    ) = peps_tensors
    (
        t_obj_top_left,
        t_obj_top_right,
        t_obj_bottom_left,
        t_obj_bottom_middle,
        t_obj_bottom_right,
    ) = peps_tensor_objs
    top_left_i, top_right_i, bottom_left_i, bottom_middle_i, bottom_right_i = (
        open_physical_indices
    )

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["top_left", "top_right", "bottom_left", "bottom_middle", "bottom_right"],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "T1b"]],
            "network": [
                [
                    (4, 5, -7, -4, -2, 3) + phys_contraction_i_top_left,  # tensor
                    (7, 8, -8, -5, -3, 6)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 6, 1),  # T6a
                    (1, 4, 7, 2),  # C1
                    (2, 5, 8, -6),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TR corner (has open indices by default)
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "T2b"]],
            "network": [
                [
                    (3, 4, 5, -7, -4, -2) + phys_contraction_i_top_right,  # tensor
                    (6, 7, 8, -8, -5, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T1a
                    (1, 4, 7, 2),  # C2
                    (2, 5, 8, -6),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner (has open indices by default)
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "T6b"]],
            "network": [
                [
                    (7, -5, -2, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (11, -6, -3, 8, 9, 10)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, -4),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_bottom_middle, phys_contraction_i_conj_bottom_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_bottom_middle = {
            "tensors": [["tensor", "tensor_conj", "T4a", "T4b"]],
            "network": [
                [
                    (-6, -4, -2, 2, 3, -9) + phys_contraction_i_bottom_middle,  # tensor
                    (-7, -5, -3, 4, 5, -10)
                    + phys_contraction_i_conj_bottom_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T4a
                    (1, 3, 5, -8),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
            ),
            contraction_bottom_middle,
        )

        # ======================================================================
        # BR corner (has open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "T4b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, -5) + phys_contraction_i_bottom_right,  # tensor
                    (-3, 8, 9, 10, 11, -6)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    # len(top_left_i) > 0 is always true by default
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T6a↙
            rho_top_left.shape[1],  # ket↙
            rho_top_left.shape[2],  # bra↙
            rho_top_left.shape[3],  # ket↘
            rho_top_left.shape[4],  # bra↘
            rho_top_left.shape[5],  # T1b→
            rho_top_left.shape[6],  # ket→
            rho_top_left.shape[7],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    # len(top_right_i) > 0 is always true by default
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # ket↙
            rho_top_right.shape[4],  # bra↙
            rho_top_right.shape[5],  # T2b↘
            rho_top_right.shape[6],  # ket↘
            rho_top_right.shape[7],  # bra↘
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    # len(bottom_left_i) > 0 is always true by default
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T4a→
            rho_bottom_left.shape[1],  # ket→
            rho_bottom_left.shape[2],  # bra→
            rho_bottom_left.shape[3],  # T6b↗
            rho_bottom_left.shape[4],  # ket↗
            rho_bottom_left.shape[5],  # bra↗
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_middle = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
        ),
        [t_bottom_middle],
        [t_obj_bottom_middle],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_i or not squeeze_closed_physical_legs:
        rho_bottom_middle = rho_bottom_middle.reshape(
            rho_bottom_middle.shape[0],  # T4a→
            rho_bottom_middle.shape[1],  # ket→
            rho_bottom_middle.shape[2],  # bra→
            rho_bottom_middle.shape[3],  # ket↗
            rho_bottom_middle.shape[4],  # bra↗
            rho_bottom_middle.shape[5],  # ket↖
            rho_bottom_middle.shape[6],  # bra↖
            rho_bottom_middle.shape[7],  # T1b←
            rho_bottom_middle.shape[8],  # ket←
            rho_bottom_middle.shape[9],  # bra←
            real_physical_dimension ** len(bottom_middle_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_upward_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    # len(bottom_right_i) > 0 is always true by default
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T2a↖
            rho_bottom_right.shape[1],  # ket↖
            rho_bottom_right.shape[2],  # bra↖
            rho_bottom_right.shape[3],  # T4b←
            rho_bottom_right.shape[4],  # ket←
            rho_bottom_right.shape[5],  # bra←
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_middle,
        rho_bottom_right,
    )


def partially_traced_five_site_density_matrix_triangular_trapezoid_downward(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top_left,
        t_top_middle,
        t_top_right,
        t_bottom_left,
        t_bottom_right,
    ) = peps_tensors
    (
        t_obj_top_left,
        t_obj_top_middle,
        t_obj_top_right,
        t_obj_bottom_left,
        t_obj_bottom_right,
    ) = peps_tensor_objs
    top_left_i, top_middle_i, top_right_i, bottom_left_i, bottom_right_i = (
        open_physical_indices
    )

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            ["top_left", "top_middle", "top_right", "bottom_left", "bottom_right"],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "T1b"]],
            "network": [
                [
                    (5, 7, -5, -2, 3, 4) + phys_contraction_i_top_left,  # tensor
                    (10, 11, -6, -3, 8, 9)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T5a
                    (1, 4, 9, 2),  # C6
                    (2, 5, 10, 6),  # C1
                    (6, 7, 11, -4),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_top_middle, phys_contraction_i_conj_top_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_top_middle = {
            "tensors": [["tensor", "tensor_conj", "T1a", "T1b"]],
            "network": [
                [
                    (2, 3, -9, -6, -4, -2) + phys_contraction_i_top_middle,  # tensor
                    (4, 5, -10, -7, -5, -3)
                    + phys_contraction_i_conj_top_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T1a
                    (1, 3, 5, -8),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
            ),
            contraction_top_middle,
        )

        # ======================================================================
        # TR corner (has open indices by default)
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "T3b"]],
            "network": [
                [
                    (3, 4, 5, 7, -5, -2) + phys_contraction_i_top_right,  # tensor
                    (8, 9, 10, 11, -6, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T1a
                    (1, 4, 9, 2),  # C2
                    (2, 5, 10, 6),  # C3
                    (6, 7, 11, -4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner (has open indices by default)
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "T5b"]],
            "network": [
                [
                    (-2, -4, -7, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (-3, -5, -8, 6, 7, 8)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-6, 3, 6, 1),  # T4a
                    (1, 4, 7, 2),  # C5
                    (2, 5, 8, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # BR corner (has open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "T4b"]],
            "network": [
                [
                    (-4, -7, 3, 4, 5, -2) + phys_contraction_i_bottom_right,  # tensor
                    (-5, -8, 6, 7, 8, -3)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-6, 3, 6, 1),  # T3a
                    (1, 4, 7, 2),  # C4
                    (2, 5, 8, -1),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    # len(top_left_i) > 0 is always true by default
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T5a↘
            rho_top_left.shape[1],  # ket↘
            rho_top_left.shape[2],  # bra↘
            rho_top_left.shape[3],  # T1b→
            rho_top_left.shape[4],  # ket→
            rho_top_left.shape[5],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_middle = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
        ),
        [t_top_middle],
        [t_obj_top_middle],
        [],
        disable_identity_check=True,
    )
    if top_middle_i or not squeeze_closed_physical_legs:
        rho_top_middle = rho_top_middle.reshape(
            rho_top_middle.shape[0],  # T1a←
            rho_top_middle.shape[1],  # ket←
            rho_top_middle.shape[2],  # bra←
            rho_top_middle.shape[3],  # ket↙
            rho_top_middle.shape[4],  # bra↙
            rho_top_middle.shape[5],  # ket↘
            rho_top_middle.shape[6],  # bra↘
            rho_top_middle.shape[7],  # T1b→
            rho_top_middle.shape[8],  # ket→
            rho_top_middle.shape[9],  # bra→
            real_physical_dimension ** len(top_middle_i),  # physical ket
            real_physical_dimension ** len(top_middle_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    # len(top_right_i) > 0 is always true by default
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # T2b↘
            rho_top_right.shape[4],  # ket↘
            rho_top_right.shape[5],  # bra↘
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    # len(bottom_left_i) > 0 is always true by default
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T5b↖
            rho_bottom_left.shape[1],  # ket↖
            rho_bottom_left.shape[2],  # bra↖
            rho_bottom_left.shape[3],  # ket↗
            rho_bottom_left.shape[4],  # bra↗
            rho_bottom_left.shape[5],  # T4a→
            rho_bottom_left.shape[6],  # ket→
            rho_bottom_left.shape[7],  # bra→
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_five_site_density_matrix_triangular_trapezoid_downward_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    # len(bottom_right_i) > 0 is always true by default
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T4b←
            rho_bottom_right.shape[1],  # ket←
            rho_bottom_right.shape[2],  # bra←
            rho_bottom_right.shape[3],  # ket↖
            rho_bottom_right.shape[4],  # bra↖
            rho_bottom_right.shape[5],  # T3a↗
            rho_bottom_right.shape[6],  # ket↗
            rho_bottom_right.shape[7],  # bra↗
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_middle,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_right,
    )


# ========================================
# NN six site triangular gates
# ========================================
def partially_traced_six_site_density_matrix_triangular_px_3py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top_left,
        t_top_middle,
        t_top_right,
        t_bottom_left,
        t_bottom_middle,
        t_bottom_right,
    ) = peps_tensors
    (
        t_obj_top_left,
        t_obj_top_middle,
        t_obj_top_right,
        t_obj_bottom_left,
        t_obj_bottom_middle,
        t_obj_bottom_right,
    ) = peps_tensor_objs
    (
        top_left_i,
        top_middle_i,
        top_right_i,
        bottom_left_i,
        bottom_middle_i,
        bottom_right_i,
    ) = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            [
                "top_left",
                "top_middle",
                "top_right",
                "bottom_left",
                "bottom_middle",
                "bottom_right",
            ],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "C1", "T1b"]],
            "network": [
                [
                    (5, 7, -5, -2, 3, 4) + phys_contraction_i_top_left,  # tensor
                    (10, 11, -6, -3, 8, 9)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T5a
                    (1, 4, 9, 2),  # C6
                    (2, 5, 10, 6),  # C1
                    (6, 7, 11, -4),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_top_middle, phys_contraction_i_conj_top_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_top_middle = {
            "tensors": [["tensor", "tensor_conj", "T1a", "T1b"]],
            "network": [
                [
                    (2, 3, -9, -6, -4, -2) + phys_contraction_i_top_middle,  # tensor
                    (4, 5, -10, -7, -5, -3)
                    + phys_contraction_i_conj_top_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T1a
                    (1, 3, 5, -8),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_top_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
            ),
            contraction_top_middle,
        )

        # ======================================================================
        # TR corner
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "T2b"]],
            "network": [
                [
                    (3, 4, 5, -7, -4, -2) + phys_contraction_i_top_right,  # tensor
                    (6, 7, 8, -8, -5, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T1a
                    (1, 4, 7, 2),  # C2
                    (2, 5, 8, -6),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "T5b"]],
            "network": [
                [
                    (-2, -4, -7, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (-3, -5, -8, 6, 7, 8)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-6, 3, 6, 1),  # T4a
                    (1, 4, 7, 2),  # C5
                    (2, 5, 8, -1),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_bottom_middle, phys_contraction_i_conj_bottom_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_bottom_middle = {
            "tensors": [["tensor", "tensor_conj", "T4a", "T4b"]],
            "network": [
                [
                    (-6, -4, -2, 2, 3, -9) + phys_contraction_i_bottom_middle,  # tensor
                    (-7, -5, -3, 4, 5, -10)
                    + phys_contraction_i_conj_bottom_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T4a
                    (1, 3, 5, -8),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
            ),
            contraction_bottom_middle,
        )

        # ======================================================================
        # BR corner (has open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "T4b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, -5) + phys_contraction_i_bottom_right,  # tensor
                    (-3, 8, 9, 10, 11, -6)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    # len(top_left_i) > 0 is always true by default
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T5a↘
            rho_top_left.shape[1],  # ket↘
            rho_top_left.shape[2],  # bra↘
            rho_top_left.shape[3],  # T1b→
            rho_top_left.shape[4],  # ket→
            rho_top_left.shape[5],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_middle = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_top_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
        ),
        [t_top_middle],
        [t_obj_top_middle],
        [],
        disable_identity_check=True,
    )
    if top_middle_i or not squeeze_closed_physical_legs:
        rho_top_middle = rho_top_middle.reshape(
            rho_top_middle.shape[0],  # T1a←
            rho_top_middle.shape[1],  # ket←
            rho_top_middle.shape[2],  # bra←
            rho_top_middle.shape[3],  # ket↙
            rho_top_middle.shape[4],  # bra↙
            rho_top_middle.shape[5],  # ket↘
            rho_top_middle.shape[6],  # bra↘
            rho_top_middle.shape[7],  # T1b→
            rho_top_middle.shape[8],  # ket→
            rho_top_middle.shape[9],  # bra→
            real_physical_dimension ** len(top_middle_i),  # physical ket
            real_physical_dimension ** len(top_middle_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # ket↙
            rho_top_right.shape[4],  # bra↙
            rho_top_right.shape[5],  # T2b↘
            rho_top_right.shape[6],  # ket↘
            rho_top_right.shape[7],  # bra↘
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T5b↖
            rho_bottom_left.shape[1],  # ket↖
            rho_bottom_left.shape[2],  # bra↖
            rho_bottom_left.shape[3],  # ket↗
            rho_bottom_left.shape[4],  # bra↗
            rho_bottom_left.shape[5],  # T4a→
            rho_bottom_left.shape[6],  # ket→
            rho_bottom_left.shape[7],  # bra→
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_middle = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
        ),
        [t_bottom_middle],
        [t_obj_bottom_middle],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_i or not squeeze_closed_physical_legs:
        rho_bottom_middle = rho_bottom_middle.reshape(
            rho_bottom_middle.shape[0],  # T4a→
            rho_bottom_middle.shape[1],  # ket→
            rho_bottom_middle.shape[2],  # bra→
            rho_bottom_middle.shape[3],  # ket↗
            rho_bottom_middle.shape[4],  # bra↗
            rho_bottom_middle.shape[5],  # ket↖
            rho_bottom_middle.shape[6],  # bra↖
            rho_bottom_middle.shape[7],  # T1b←
            rho_bottom_middle.shape[8],  # ket←
            rho_bottom_middle.shape[9],  # bra←
            real_physical_dimension ** len(bottom_middle_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_px_3py_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    # len(bottom_right_i) > 0 is always true by default
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T2a↖
            rho_bottom_right.shape[1],  # ket↖
            rho_bottom_right.shape[2],  # bra↖
            rho_bottom_right.shape[3],  # T4b←
            rho_bottom_right.shape[4],  # ket←
            rho_bottom_right.shape[5],  # bra←
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_middle,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_middle,
        rho_bottom_right,
    )


def partially_traced_six_site_density_matrix_triangular_nx_2py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top_left,
        t_top_middle,
        t_top_right,
        t_bottom_left,
        t_bottom_middle,
        t_bottom_right,
    ) = peps_tensors
    (
        t_obj_top_left,
        t_obj_top_middle,
        t_obj_top_right,
        t_obj_bottom_left,
        t_obj_bottom_middle,
        t_obj_bottom_right,
    ) = peps_tensor_objs
    (
        top_left_i,
        top_middle_i,
        top_right_i,
        bottom_left_i,
        bottom_middle_i,
        bottom_right_i,
    ) = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            [
                "top_left",
                "top_middle",
                "top_right",
                "bottom_left",
                "bottom_middle",
                "bottom_right",
            ],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "T1b"]],
            "network": [
                [
                    (4, 5, -7, -4, -2, 3) + phys_contraction_i_top_left,  # tensor
                    (7, 8, -8, -5, -3, 6)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 6, 1),  # T6a
                    (1, 4, 7, 2),  # C1
                    (2, 5, 8, -6),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_top_middle, phys_contraction_i_conj_top_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_top_middle = {
            "tensors": [["tensor", "tensor_conj", "T1a", "T1b"]],
            "network": [
                [
                    (2, 3, -9, -6, -4, -2) + phys_contraction_i_top_middle,  # tensor
                    (4, 5, -10, -7, -5, -3)
                    + phys_contraction_i_conj_top_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T1a
                    (1, 3, 5, -8),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
            ),
            contraction_top_middle,
        )

        # ======================================================================
        # TR corner
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "C3", "T3b"]],
            "network": [
                [
                    (3, 4, 5, 7, -5, -2) + phys_contraction_i_top_right,  # tensor
                    (8, 9, 10, 11, -6, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T1a
                    (1, 4, 9, 2),  # C2
                    (2, 5, 10, 6),  # C3
                    (6, 7, 11, -4),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "T6b"]],
            "network": [
                [
                    (7, -5, -2, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (11, -6, -3, 8, 9, 10)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, -4),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_bottom_middle, phys_contraction_i_conj_bottom_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_bottom_middle = {
            "tensors": [["tensor", "tensor_conj", "T4a", "T4b"]],
            "network": [
                [
                    (-6, -4, -2, 2, 3, -9) + phys_contraction_i_bottom_middle,  # tensor
                    (-7, -5, -3, 4, 5, -10)
                    + phys_contraction_i_conj_bottom_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T4a
                    (1, 3, 5, -8),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
            ),
            contraction_bottom_middle,
        )

        # ======================================================================
        # BR corner (has open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "T4b"]],
            "network": [
                [
                    (-4, -7, 3, 4, 5, -2) + phys_contraction_i_bottom_right,  # tensor
                    (-5, -8, 6, 7, 8, -3)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-6, 3, 6, 1),  # T3a
                    (1, 4, 7, 2),  # C4
                    (2, 5, 8, -1),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T6a↙
            rho_top_left.shape[1],  # ket↙
            rho_top_left.shape[2],  # bra↙
            rho_top_left.shape[3],  # ket↘
            rho_top_left.shape[4],  # bra↘
            rho_top_left.shape[5],  # T1b→
            rho_top_left.shape[6],  # ket→
            rho_top_left.shape[7],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_middle = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
        ),
        [t_top_middle],
        [t_obj_top_middle],
        [],
        disable_identity_check=True,
    )
    if top_middle_i or not squeeze_closed_physical_legs:
        rho_top_middle = rho_top_middle.reshape(
            rho_top_middle.shape[0],  # T1a←
            rho_top_middle.shape[1],  # ket←
            rho_top_middle.shape[2],  # bra←
            rho_top_middle.shape[3],  # ket↙
            rho_top_middle.shape[4],  # bra↙
            rho_top_middle.shape[5],  # ket↘
            rho_top_middle.shape[6],  # bra↘
            rho_top_middle.shape[7],  # T1b→
            rho_top_middle.shape[8],  # ket→
            rho_top_middle.shape[9],  # bra→
            real_physical_dimension ** len(top_middle_i),  # physical ket
            real_physical_dimension ** len(top_middle_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    # len(top_right_i) > 0 is always true by default
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # T3b↙
            rho_top_right.shape[4],  # ket↙
            rho_top_right.shape[5],  # bra↙
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    # len(bottom_left_i) > 0 is always true by default
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T4a→
            rho_bottom_left.shape[1],  # ket→
            rho_bottom_left.shape[2],  # bra→
            rho_bottom_left.shape[3],  # T6b↗
            rho_bottom_left.shape[4],  # ket↗
            rho_bottom_left.shape[5],  # bra↗
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_middle = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_i}"
        ),
        [t_bottom_middle],
        [t_obj_bottom_middle],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_i or not squeeze_closed_physical_legs:
        rho_bottom_middle = rho_bottom_middle.reshape(
            rho_bottom_middle.shape[0],  # T4a→
            rho_bottom_middle.shape[1],  # ket→
            rho_bottom_middle.shape[2],  # bra→
            rho_bottom_middle.shape[3],  # ket↗
            rho_bottom_middle.shape[4],  # bra↗
            rho_bottom_middle.shape[5],  # ket↖
            rho_bottom_middle.shape[6],  # bra↖
            rho_bottom_middle.shape[7],  # T1b←
            rho_bottom_middle.shape[8],  # ket←
            rho_bottom_middle.shape[9],  # bra←
            real_physical_dimension ** len(bottom_middle_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_nx_2py_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T4b←
            rho_bottom_right.shape[1],  # ket←
            rho_bottom_right.shape[2],  # bra←
            rho_bottom_right.shape[3],  # ket↖
            rho_bottom_right.shape[4],  # bra↖
            rho_bottom_right.shape[5],  # T3a↗
            rho_bottom_right.shape[6],  # ket↗
            rho_bottom_right.shape[7],  # bra↗
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_middle,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_middle,
        rho_bottom_right,
    )


def partially_traced_six_site_density_matrix_triangular_3px_py(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top,
        t_top_middle_left,
        t_top_middle_right,
        t_bottom_middle_left,
        t_bottom_middle_right,
        t_bottom,
    ) = peps_tensors
    (
        t_obj_top,
        t_obj_top_middle_left,
        t_obj_top_middle_right,
        t_obj_bottom_middle_left,
        t_obj_bottom_middle_right,
        t_obj_bottom,
    ) = peps_tensor_objs
    (
        top_i,
        top_middle_left_i,
        top_middle_right_i,
        bottom_middle_left_i,
        bottom_middle_right_i,
        bottom_i,
    ) = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            [
                "top",
                "top_middle_left",
                "top_middle_right",
                "bottom_middle_left",
                "bottom_middle_right",
                "bottom",
            ],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has all open indices by default)
        phys_contraction_i_top, phys_contraction_i_conj_top = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_top = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "C2", "T2b"]],
            "network": [
                [
                    (4, 5, 7, -5, -2, 3) + phys_contraction_i_top,  # tensor
                    (9, 10, 11, -6, -3, 8) + phys_contraction_i_conj_top,  # tensor_conj
                    (-1, 3, 8, 1),  # T6a
                    (1, 4, 9, 2),  # C1
                    (2, 5, 10, 6),  # C2
                    (6, 7, 11, -4),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_top_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
            ),
            contraction_top,
        )

        # ======================================================================
        # TML corner
        phys_contraction_i_top_middle_left, phys_contraction_i_conj_top_middle_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_middle_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_top_middle_left = {
            "tensors": [["tensor", "tensor_conj", "T6a", "T6b"]],
            "network": [
                [
                    (2, -2, -4, -6, -9, 3)
                    + phys_contraction_i_top_middle_left,  # tensor
                    (4, -3, -5, -7, -10, 5)
                    + phys_contraction_i_conj_top_middle_left,  # tensor_conj
                    (-8, 3, 5, 1),  # T6a
                    (1, 2, 4, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_top_middle_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_left_i}"
            ),
            contraction_top_middle_left,
        )

        # ======================================================================
        # TMR corner
        (
            phys_contraction_i_top_middle_right,
            phys_contraction_i_conj_top_middle_right,
        ) = build_physical_contraction_indices(
            site_open_physical_indices=top_middle_right_i,
            num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
            start_contraction_at=9,
            negative_offset=-8,
        )
        contraction_top_middle_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "T3b"]],
            "network": [
                [
                    (-2, 3, 4, 5, -7, -4)
                    + phys_contraction_i_top_middle_right,  # tensor
                    (-3, 6, 7, 8, -8, -5)
                    + phys_contraction_i_conj_top_middle_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T2a
                    (1, 4, 7, 2),  # C3
                    (2, 5, 8, -6),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_top_middle_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_right_i}"
            ),
            contraction_top_middle_right,
        )

        # ======================================================================
        # BML corner
        (
            phys_contraction_i_bottom_middle_left,
            phys_contraction_i_conj_bottom_middle_left,
        ) = build_physical_contraction_indices(
            site_open_physical_indices=bottom_middle_left_i,
            num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
            start_contraction_at=9,
            negative_offset=-8,
        )
        contraction_bottom_middle_left = {
            "tensors": [["tensor", "tensor_conj", "T5a", "C6", "T6b"]],
            "network": [
                [
                    (5, -2, -4, -7, 3, 4)
                    + phys_contraction_i_bottom_middle_left,  # tensor
                    (8, -3, -5, -8, 6, 7)
                    + phys_contraction_i_conj_bottom_middle_left,  # tensor_conj
                    (-6, 3, 6, 1),  # T5a
                    (1, 4, 7, 2),  # C6
                    (2, 5, 8, -1),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_middle_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_left_i}"
            ),
            contraction_bottom_middle_left,
        )

        # ======================================================================
        # BMR corner
        (
            phys_contraction_i_bottom_middle_right,
            phys_contraction_i_conj_bottom_middle_right,
        ) = build_physical_contraction_indices(
            site_open_physical_indices=bottom_middle_right_i,
            num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
            start_contraction_at=6,
            negative_offset=-10,
        )
        contraction_bottom_middle_right = {
            "tensors": [["tensor", "tensor_conj", "T3a", "T3b"]],
            "network": [
                [
                    (-6, -9, 2, 3, -2, -4)
                    + phys_contraction_i_bottom_middle_right,  # tensor
                    (-7, -10, 4, 5, -3, -5)
                    + phys_contraction_i_conj_bottom_middle_right,  # tensor_conj
                    (-8, 2, 4, 1),  # T3a
                    (1, 3, 5, -1),  # T3b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_middle_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_right_i}"
            ),
            contraction_bottom_middle_right,
        )

        # ======================================================================
        # B corner (has all open indices by default)
        phys_contraction_i_bottom, phys_contraction_i_conj_bottom = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom = {
            "tensors": [["tensor", "tensor_conj", "T3a", "C4", "C5", "T5b"]],
            "network": [
                [
                    (-5, -2, 3, 4, 5, 7) + phys_contraction_i_bottom,  # tensor
                    (-6, -3, 8, 9, 10, 11)
                    + phys_contraction_i_conj_bottom,  # tensor_conj
                    (-1, 3, 8, 1),  # T3a
                    (1, 4, 9, 2),  # C4
                    (2, 5, 10, 6),  # C5
                    (6, 7, 11, -4),  # T5b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
            ),
            contraction_bottom,
        )

    # ======================================================================
    # Apply contractions
    rho_top = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_top_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_i}"
        ),
        [t_top],
        [t_obj_top],
        [],
        disable_identity_check=True,
    )
    # len(top_i) > 0 is always true by default
    if top_i or not squeeze_closed_physical_legs:
        rho_top = rho_top.reshape(
            rho_top.shape[0],  # T6a↙
            rho_top.shape[1],  # ket↙
            rho_top.shape[2],  # bra↙
            rho_top.shape[3],  # T2b↘
            rho_top.shape[4],  # ket↘
            rho_top.shape[5],  # bra↘
            real_physical_dimension ** len(top_i),  # physical ket
            real_physical_dimension ** len(top_i),  # physical bra
        )

    rho_top_middle_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_top_middle_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_left_i}"
        ),
        [t_top_middle_left],
        [t_obj_top_middle_left],
        [],
        disable_identity_check=True,
    )
    if top_middle_left_i or not squeeze_closed_physical_legs:
        rho_top_middle_left = rho_top_middle_left.reshape(
            rho_top_middle_left.shape[0],  # T6b↗
            rho_top_middle_left.shape[1],  # ket↗
            rho_top_middle_left.shape[2],  # bra↗
            rho_top_middle_left.shape[3],  # ket→
            rho_top_middle_left.shape[4],  # bra→
            rho_top_middle_left.shape[5],  # ket↘
            rho_top_middle_left.shape[6],  # bra↘
            rho_top_middle_left.shape[7],  # T6a↙
            rho_top_middle_left.shape[8],  # ket↙
            rho_top_middle_left.shape[9],  # bra↙
            real_physical_dimension ** len(top_middle_left_i),  # physical ket
            real_physical_dimension ** len(top_middle_left_i),  # physical bra
        )

    rho_top_middle_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_top_middle_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_right_i}"
        ),
        [t_top_middle_right],
        [t_obj_top_middle_right],
        [],
        disable_identity_check=True,
    )
    if top_middle_right_i or not squeeze_closed_physical_legs:
        rho_top_middle_right = rho_top_middle_right.reshape(
            rho_top_middle_right.shape[0],  # T2a↖
            rho_top_middle_right.shape[1],  # ket↖
            rho_top_middle_right.shape[2],  # bra↖
            rho_top_middle_right.shape[3],  # ket←
            rho_top_middle_right.shape[4],  # bra←
            rho_top_middle_right.shape[5],  # T3b↙
            rho_top_middle_right.shape[6],  # ket↙
            rho_top_middle_right.shape[7],  # bra↙
            real_physical_dimension ** len(top_middle_right_i),  # physical ket
            real_physical_dimension ** len(top_middle_right_i),  # physical bra
        )

    rho_bottom_middle_left = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_middle_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_left_i}"
        ),
        [t_bottom_middle_left],
        [t_obj_bottom_middle_left],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_left_i or not squeeze_closed_physical_legs:
        rho_bottom_middle_left = rho_bottom_middle_left.reshape(
            rho_bottom_middle_left.shape[0],  # T6b↗
            rho_bottom_middle_left.shape[1],  # ket↗
            rho_bottom_middle_left.shape[2],  # bra↗
            rho_bottom_middle_left.shape[3],  # ket→
            rho_bottom_middle_left.shape[4],  # bra→
            rho_bottom_middle_left.shape[5],  # T5a↘
            rho_bottom_middle_left.shape[6],  # ket↘
            rho_bottom_middle_left.shape[7],  # bra↘
            real_physical_dimension ** len(bottom_middle_left_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_left_i),  # physical bra
        )

    rho_bottom_middle_right = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_middle_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_right_i}"
        ),
        [t_bottom_middle_right],
        [t_obj_bottom_middle_right],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_right_i or not squeeze_closed_physical_legs:
        rho_bottom_middle_right = rho_bottom_middle_right.reshape(
            rho_bottom_middle_right.shape[0],  # T3b↙
            rho_bottom_middle_right.shape[1],  # ket↙
            rho_bottom_middle_right.shape[2],  # bra↙
            rho_bottom_middle_right.shape[3],  # ket←
            rho_bottom_middle_right.shape[4],  # bra←
            rho_bottom_middle_right.shape[5],  # ket↖
            rho_bottom_middle_right.shape[6],  # bra↖
            rho_bottom_middle_right.shape[7],  # T3a↗
            rho_bottom_middle_right.shape[8],  # ket↗
            rho_bottom_middle_right.shape[9],  # bra↗
            real_physical_dimension ** len(bottom_middle_right_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_right_i),  # physical bra
        )

    rho_bottom = apply_contraction(
        (
            f"partially_traced_six_site_density_matrix_triangular_3px_py_bottom_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_i}"
        ),
        [t_bottom],
        [t_obj_bottom],
        [],
        disable_identity_check=True,
    )
    # len(bottom_i) > 0 is always true by default
    if bottom_i or not squeeze_closed_physical_legs:
        rho_bottom = rho_bottom.reshape(
            rho_bottom.shape[0],  # T3a↗
            rho_bottom.shape[1],  # ket↗
            rho_bottom.shape[2],  # bra↗
            rho_bottom.shape[3],  # T5b↖
            rho_bottom.shape[4],  # ket↖
            rho_bottom.shape[5],  # bra↖
            real_physical_dimension ** len(bottom_i),  # physical ket
            real_physical_dimension ** len(bottom_i),  # physical bra
        )

    return (
        rho_top,
        rho_top_middle_left,
        rho_top_middle_right,
        rho_bottom_middle_left,
        rho_bottom_middle_right,
        rho_bottom,
    )


# ========================================
# NN seven site triangular gates
# ========================================
def partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
    squeeze_closed_physical_legs: bool = True,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup)
        for tup in open_physical_indices
    ):
        raise TypeError(
            "All elements of each tuple must be integers (or the tuple may be empty)."
        )
    if not all(
        len(tup) <= num_coarse_grained_physical_indices for tup in open_physical_indices
    ):
        raise ValueError(
            f"At least one tuple in `open_physical_indices` {open_physical_indices} has length greater"
            f"than the number of coarse-grained physical sites {num_coarse_grained_physical_indices}"
        )
    peps_tensors = [
        t.reshape(
            t.shape[0],
            t.shape[1],
            t.shape[2],
            t.shape[3],
            t.shape[4],
            t.shape[5],
            *((real_physical_dimension,) * num_coarse_grained_physical_indices),
        )
        for t in peps_tensors
    ]
    (
        t_top_left,
        t_top_middle,
        t_top_right,
        t_bottom_left,
        t_bottom_middle_left,
        t_bottom_middle_right,
        t_bottom_right,
    ) = peps_tensors
    (
        t_obj_top_left,
        t_obj_top_middle,
        t_obj_top_right,
        t_obj_bottom_left,
        t_obj_bottom_middle_left,
        t_obj_bottom_middle_right,
        t_obj_bottom_right,
    ) = peps_tensor_objs
    (
        top_left_i,
        top_middle_i,
        top_right_i,
        bottom_left_i,
        bottom_middle_left_i,
        bottom_middle_right_i,
        bottom_right_i,
    ) = open_physical_indices

    if any(
        not hasattr(
            Definitions,
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_{pos_name}_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{pos_idx}"
            ),
        )
        for pos_idx, pos_name in zip(
            open_physical_indices,
            [
                "top_left",
                "top_middle",
                "top_right",
                "bottom_left",
                "bottom_middle_left",
                "bottom_middle_right",
                "bottom_right",
            ],
            strict=True,
        )
    ):

        # ======================================================================
        # TL corner (has open indices by default)
        phys_contraction_i_top_left, phys_contraction_i_conj_top_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )
        contraction_top_left = {
            "tensors": [["tensor", "tensor_conj", "T6a", "C1", "T1b"]],
            "network": [
                [
                    (4, 5, -7, -4, -2, 3) + phys_contraction_i_top_left,  # tensor
                    (7, 8, -8, -5, -3, 6)
                    + phys_contraction_i_conj_top_left,  # tensor_conj
                    (-1, 3, 6, 1),  # T6a
                    (1, 4, 7, 2),  # C1
                    (2, 5, 8, -6),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
            ),
            contraction_top_left,
        )

        # ======================================================================
        # TM
        phys_contraction_i_top_middle, phys_contraction_i_conj_top_middle = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_middle_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=6,
                negative_offset=-10,
            )
        )
        contraction_top_middle = {
            "tensors": [["tensor", "tensor_conj", "T1a", "T1b"]],
            "network": [
                [
                    (2, 3, -9, -6, -4, -2) + phys_contraction_i_top_middle,  # tensor
                    (4, 5, -10, -7, -5, -3)
                    + phys_contraction_i_conj_top_middle,  # tensor_conj
                    (-1, 2, 4, 1),  # T1a
                    (1, 3, 5, -8),  # T1b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_middle_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
            ),
            contraction_top_middle,
        )

        # ======================================================================
        # TR corner (has open indices by default)
        phys_contraction_i_top_right, phys_contraction_i_conj_top_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=top_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=9,
                negative_offset=-8,
            )
        )

        contraction_top_right = {
            "tensors": [["tensor", "tensor_conj", "T1a", "C2", "T2b"]],
            "network": [
                [
                    (3, 4, 5, -7, -4, -2) + phys_contraction_i_top_right,  # tensor
                    (6, 7, 8, -8, -5, -3)
                    + phys_contraction_i_conj_top_right,  # tensor_conj
                    (-1, 3, 6, 1),  # T1a
                    (1, 4, 7, 2),  # C2
                    (2, 5, 8, -6),  # T2b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
            ),
            contraction_top_right,
        )

        # ======================================================================
        # BL corner (has open indices by default)
        phys_contraction_i_bottom_left, phys_contraction_i_conj_bottom_left = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_left_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )
        contraction_bottom_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "C5", "C6", "T6b"]],
            "network": [
                [
                    (7, -5, -2, 3, 4, 5) + phys_contraction_i_bottom_left,  # tensor
                    (11, -6, -3, 8, 9, 10)
                    + phys_contraction_i_conj_bottom_left,  # tensor_conj
                    (-1, 3, 8, 1),  # T4a
                    (1, 4, 9, 2),  # C5
                    (2, 5, 10, 6),  # C6
                    (6, 7, 11, -4),  # T6b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
            ),
            contraction_bottom_left,
        )

        # ======================================================================
        # BML
        (
            phys_contraction_i_bottom_middle_left,
            phys_contraction_i_conj_bottom_middle_left,
        ) = build_physical_contraction_indices(
            site_open_physical_indices=bottom_middle_left_i,
            num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
            start_contraction_at=6,
            negative_offset=-10,
        )
        contraction_bottom_middle_left = {
            "tensors": [["tensor", "tensor_conj", "T4a", "T4b"]],
            "network": [
                [
                    (-6, -4, -2, 2, 3, -9)
                    + phys_contraction_i_bottom_middle_left,  # tensor
                    (-7, -5, -3, 4, 5, -10)
                    + phys_contraction_i_conj_bottom_middle_left,  # tensor_conj
                    (-1, 2, 4, 1),  # T4a
                    (1, 3, 5, -8),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_middle_left_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_left_i}"
            ),
            contraction_bottom_middle_left,
        )

        # ======================================================================
        # BMR
        (
            phys_contraction_i_bottom_middle_right,
            phys_contraction_i_conj_bottom_middle_right,
        ) = build_physical_contraction_indices(
            site_open_physical_indices=bottom_middle_right_i,
            num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
            start_contraction_at=6,
            negative_offset=-10,
        )
        contraction_bottom_middle_right = {
            "tensors": [["tensor", "tensor_conj", "T4a", "T4b"]],
            "network": [
                [
                    (-6, -4, -2, 2, 3, -9)
                    + phys_contraction_i_bottom_middle_right,  # tensor
                    (-7, -5, -3, 4, 5, -10)
                    + phys_contraction_i_conj_bottom_middle_right,  # tensor_conj
                    (-1, 2, 4, 1),  # T4a
                    (1, 3, 5, -8),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_middle_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_right_i}"
            ),
            contraction_bottom_middle_right,
        )

        # ======================================================================
        # BR corner (has open indices by default)
        phys_contraction_i_bottom_right, phys_contraction_i_conj_bottom_right = (
            build_physical_contraction_indices(
                site_open_physical_indices=bottom_right_i,
                num_coarse_grained_physical_indices=num_coarse_grained_physical_indices,
                start_contraction_at=12,
                negative_offset=-6,
            )
        )

        contraction_bottom_right = {
            "tensors": [["tensor", "tensor_conj", "T2a", "C3", "C4", "T4b"]],
            "network": [
                [
                    (-2, 3, 4, 5, 7, -5) + phys_contraction_i_bottom_right,  # tensor
                    (-3, 8, 9, 10, 11, -6)
                    + phys_contraction_i_conj_bottom_right,  # tensor_conj
                    (-1, 3, 8, 1),  # T2a
                    (1, 4, 9, 2),  # C3
                    (2, 5, 10, 6),  # C4
                    (6, 7, 11, -4),  # T4b
                ],
            ],
        }
        Definitions.add_def(
            (
                f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_right_"
                f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
            ),
            contraction_bottom_right,
        )

    # ======================================================================
    # Apply contractions
    rho_top_left = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_left_i}"
        ),
        [t_top_left],
        [t_obj_top_left],
        [],
        disable_identity_check=True,
    )
    # len(top_left_i) > 0 is always true by default
    if top_left_i or not squeeze_closed_physical_legs:
        rho_top_left = rho_top_left.reshape(
            rho_top_left.shape[0],  # T6a↙
            rho_top_left.shape[1],  # ket↙
            rho_top_left.shape[2],  # bra↙
            rho_top_left.shape[3],  # ket↘
            rho_top_left.shape[4],  # bra↘
            rho_top_left.shape[5],  # T1b→
            rho_top_left.shape[6],  # ket→
            rho_top_left.shape[7],  # bra→
            real_physical_dimension ** len(top_left_i),  # physical ket
            real_physical_dimension ** len(top_left_i),  # physical bra
        )

    rho_top_middle = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_middle_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_middle_i}"
        ),
        [t_top_middle],
        [t_obj_top_middle],
        [],
        disable_identity_check=True,
    )
    if top_middle_i or not squeeze_closed_physical_legs:
        rho_top_middle = rho_top_middle.reshape(
            rho_top_middle.shape[0],  # T1a←
            rho_top_middle.shape[1],  # ket←
            rho_top_middle.shape[2],  # bra←
            rho_top_middle.shape[3],  # ket↙
            rho_top_middle.shape[4],  # bra↙
            rho_top_middle.shape[5],  # ket↘
            rho_top_middle.shape[6],  # bra↘
            rho_top_middle.shape[7],  # T1b→
            rho_top_middle.shape[8],  # ket→
            rho_top_middle.shape[9],  # bra→
            real_physical_dimension ** len(top_middle_i),  # physical ket
            real_physical_dimension ** len(top_middle_i),  # physical bra
        )

    rho_top_right = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_top_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{top_right_i}"
        ),
        [t_top_right],
        [t_obj_top_right],
        [],
        disable_identity_check=True,
    )
    # len(top_right_i) > 0 is always true by default
    if top_right_i or not squeeze_closed_physical_legs:
        rho_top_right = rho_top_right.reshape(
            rho_top_right.shape[0],  # T1a←
            rho_top_right.shape[1],  # ket←
            rho_top_right.shape[2],  # bra←
            rho_top_right.shape[3],  # ket↙
            rho_top_right.shape[4],  # bra↙
            rho_top_right.shape[5],  # T2b↘
            rho_top_right.shape[6],  # ket↘
            rho_top_right.shape[7],  # bra↘
            real_physical_dimension ** len(top_right_i),  # physical ket
            real_physical_dimension ** len(top_right_i),  # physical bra
        )

    rho_bottom_left = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_left_i}"
        ),
        [t_bottom_left],
        [t_obj_bottom_left],
        [],
        disable_identity_check=True,
    )
    # len(bottom_left_i) > 0 is always true by default
    if bottom_left_i or not squeeze_closed_physical_legs:
        rho_bottom_left = rho_bottom_left.reshape(
            rho_bottom_left.shape[0],  # T4a→
            rho_bottom_left.shape[1],  # ket→
            rho_bottom_left.shape[2],  # bra→
            rho_bottom_left.shape[3],  # T6b↗
            rho_bottom_left.shape[4],  # ket↗
            rho_bottom_left.shape[5],  # bra↗
            real_physical_dimension ** len(bottom_left_i),  # physical ket
            real_physical_dimension ** len(bottom_left_i),  # physical bra
        )

    rho_bottom_middle_left = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_middle_left_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_left_i}"
        ),
        [t_bottom_middle_left],
        [t_obj_bottom_middle_left],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_left_i or not squeeze_closed_physical_legs:
        rho_bottom_middle_left = rho_bottom_middle_left.reshape(
            rho_bottom_middle_left.shape[0],  # T4a→
            rho_bottom_middle_left.shape[1],  # ket→
            rho_bottom_middle_left.shape[2],  # bra→
            rho_bottom_middle_left.shape[3],  # ket↗
            rho_bottom_middle_left.shape[4],  # bra↗
            rho_bottom_middle_left.shape[5],  # ket↖
            rho_bottom_middle_left.shape[6],  # bra↖
            rho_bottom_middle_left.shape[7],  # T1b←
            rho_bottom_middle_left.shape[8],  # ket←
            rho_bottom_middle_left.shape[9],  # bra←
            real_physical_dimension ** len(bottom_middle_left_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_left_i),  # physical bra
        )

    rho_bottom_middle_right = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_middle_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_middle_right_i}"
        ),
        [t_bottom_middle_right],
        [t_obj_bottom_middle_right],
        [],
        disable_identity_check=True,
    )
    if bottom_middle_right_i or not squeeze_closed_physical_legs:
        rho_bottom_middle_right = rho_bottom_middle_right.reshape(
            rho_bottom_middle_right.shape[0],  # T4a→
            rho_bottom_middle_right.shape[1],  # ket→
            rho_bottom_middle_right.shape[2],  # bra→
            rho_bottom_middle_right.shape[3],  # ket↗
            rho_bottom_middle_right.shape[4],  # bra↗
            rho_bottom_middle_right.shape[5],  # ket↖
            rho_bottom_middle_right.shape[6],  # bra↖
            rho_bottom_middle_right.shape[7],  # T1b←
            rho_bottom_middle_right.shape[8],  # ket←
            rho_bottom_middle_right.shape[9],  # bra←
            real_physical_dimension ** len(bottom_middle_right_i),  # physical ket
            real_physical_dimension ** len(bottom_middle_right_i),  # physical bra
        )

    rho_bottom_right = apply_contraction(
        (
            f"partially_traced_seven_site_density_matrix_triangular_double_trapezoid_upward_bottom_right_"
            f"{real_physical_dimension}_{num_coarse_grained_physical_indices}_{bottom_right_i}"
        ),
        [t_bottom_right],
        [t_obj_bottom_right],
        [],
        disable_identity_check=True,
    )
    # len(bottom_right_i) > 0 is always true by default
    if bottom_right_i or not squeeze_closed_physical_legs:
        rho_bottom_right = rho_bottom_right.reshape(
            rho_bottom_right.shape[0],  # T2a↖
            rho_bottom_right.shape[1],  # ket↖
            rho_bottom_right.shape[2],  # bra↖
            rho_bottom_right.shape[3],  # T4b←
            rho_bottom_right.shape[4],  # ket←
            rho_bottom_right.shape[5],  # bra←
            real_physical_dimension ** len(bottom_right_i),  # physical ket
            real_physical_dimension ** len(bottom_right_i),  # physical bra
        )

    return (
        rho_top_left,
        rho_top_middle,
        rho_top_right,
        rho_bottom_left,
        rho_bottom_middle_left,
        rho_bottom_middle_right,
        rho_bottom_right,
    )
