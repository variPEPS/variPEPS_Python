from functools import partial

import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from .absorption import do_absorption_step

from typing import Sequence, Tuple, List


@partial(jit, static_argnums=(1,))
def _calc_corner_svds(
    peps_tensors: List[PEPS_Tensor], tensor_shape: Tuple[int, int, int]
) -> jnp.ndarray:
    step_corner_svd = jnp.zeros(tensor_shape, dtype=jnp.float64)

    for ti, t in enumerate(peps_tensors):
        C1_svd = jnp.linalg.svd(t.C1, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 0, : C1_svd.shape[0]].set(
            C1_svd, indices_are_sorted=True, unique_indices=True
        )

        C2_svd = jnp.linalg.svd(t.C2, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 1, : C2_svd.shape[0]].set(
            C2_svd, indices_are_sorted=True, unique_indices=True
        )

        C3_svd = jnp.linalg.svd(t.C3, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 2, : C3_svd.shape[0]].set(
            C3_svd, indices_are_sorted=True, unique_indices=True
        )

        C4_svd = jnp.linalg.svd(t.C4, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 3, : C4_svd.shape[0]].set(
            C4_svd, indices_are_sorted=True, unique_indices=True
        )

    return step_corner_svd


def calc_ctmrg_env(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    *,
    eps: float = 1e-8,
    max_steps: int = 200,
) -> Tuple[PEPS_Unit_Cell, List[jnp.ndarray]]:
    shape_corner_svd = (unitcell.get_len_unique_tensors(), 4, unitcell[0, 0][0][0].chi)
    corner_singular_values = [
        _calc_corner_svds(unitcell.get_unique_tensors(), shape_corner_svd)
    ]

    working_unitcell = unitcell
    count = 0
    conv = 1000

    while conv > eps and count < max_steps:
        working_unitcell = do_absorption_step(peps_tensors, working_unitcell)

        corner_singular_values.append(
            _calc_corner_svds(working_unitcell.get_unique_tensors(), shape_corner_svd)
        )

        conv = jnp.linalg.norm(corner_singular_values[-1] - corner_singular_values[-2])

        count += 1

        print(f"{count}: {conv}")

    return working_unitcell, corner_singular_values