import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction, Definitions

from typing import Sequence, Tuple


def partially_traced_vertical_two_site_density_matrices_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    real_physical_dimension: int,
    num_coarse_grained_physical_indices: int,
    open_physical_indices: Tuple[Tuple[int], Tuple[int]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if not all(
        all(isinstance(open_idx, int) for open_idx in tup) or len(tup) == 0
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

        phys_contraction_i_top = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(top_i))
        )
        phys_contraction_i_conj_top = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(top_i))
        )
        for pos, i in enumerate(top_i):
            phys_contraction_i_top.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_top.insert(i - 1, -len(top_i) - (pos + 1))
        phys_contraction_i_top = tuple(phys_contraction_i_top)
        phys_contraction_i_conj_top = tuple(phys_contraction_i_conj_top)

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

        phys_contraction_i_bottom = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(bottom_i))
        )
        phys_contraction_i_conj_bottom = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(bottom_i))
        )
        for pos, i in enumerate(bottom_i):
            phys_contraction_i_bottom.insert(i - 1, -4 - (pos + 1))
            phys_contraction_i_conj_bottom.insert(i - 1, -4 - len(bottom_i) - (pos + 1))
        phys_contraction_i_bottom = tuple(phys_contraction_i_bottom)
        phys_contraction_i_conj_bottom = tuple(phys_contraction_i_conj_bottom)

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
    if len(top_i) > 0:
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
    if len(bottom_i) > 0:
        # Physical indices are at the end (ket, bra), for consistency with _two_site_workhorse
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
        all(isinstance(open_idx, int) for open_idx in tup) or len(tup) == 0
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

        phys_contraction_i_left = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(left_i))
        )
        phys_contraction_i_conj_left = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(left_i))
        )
        for pos, i in enumerate(left_i):
            phys_contraction_i_left.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_left.insert(i - 1, -len(left_i) - (pos + 1))
        phys_contraction_i_left = tuple(phys_contraction_i_left)
        phys_contraction_i_conj_left = tuple(phys_contraction_i_conj_left)

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

        phys_contraction_i_right = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(right_i))
        )
        phys_contraction_i_conj_right = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(right_i))
        )
        for pos, i in enumerate(right_i):
            phys_contraction_i_right.insert(i - 1, -4 - (pos + 1))
            phys_contraction_i_conj_right.insert(i - 1, -4 - len(right_i) - (pos + 1))
        phys_contraction_i_right = tuple(phys_contraction_i_right)
        phys_contraction_i_conj_right = tuple(phys_contraction_i_conj_right)

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
    if len(left_i) > 0:
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
    if len(right_i) > 0:
        # Physical indices are at the end (ket, bra), for consistency with _two_site_workhorse
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
        all(isinstance(open_idx, int) for open_idx in tup) or len(tup) == 0
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

        phys_contraction_i_top = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(top_i))
        )
        phys_contraction_i_conj_top = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(top_i))
        )
        for pos, i in enumerate(top_i):
            phys_contraction_i_top.insert(i - 1, -(pos + 1))
            phys_contraction_i_conj_top.insert(i - 1, -len(top_i) - (pos + 1))
        phys_contraction_i_top = tuple(phys_contraction_i_top)
        phys_contraction_i_conj_top = tuple(phys_contraction_i_conj_top)

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

        phys_contraction_i_bottom = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(bottom_i))
        )
        phys_contraction_i_conj_bottom = list(
            range(15, 15 + num_coarse_grained_physical_indices - len(bottom_i))
        )
        for pos, i in enumerate(bottom_i):
            phys_contraction_i_bottom.insert(i - 1, -4 - (pos + 1))
            phys_contraction_i_conj_bottom.insert(i - 1, -4 - len(bottom_i) - (pos + 1))
        phys_contraction_i_bottom = tuple(phys_contraction_i_bottom)
        phys_contraction_i_conj_bottom = tuple(phys_contraction_i_conj_bottom)

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
    if len(top_i) > 0:
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
    if len(bottom_i) > 0:
        # Physical indices are at the end (ket, bra), for consistency with _two_site_workhorse
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
