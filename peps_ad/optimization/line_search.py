import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import calc_ctmrg_expectation

from typing import Sequence, Tuple, List


def simple_line_search(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    descent_direction: Sequence[jnp.ndarray],
    current_value: float,
    *,
    ctmrg_eps: float = 1e-5,
    ctmrg_max_steps: int = 50,
    initial_step_size: float = 1.0,
    reduction_factor: float = 0.5,
    max_steps: int = 20
) -> Tuple[List[jnp.ndarray], PEPS_Unit_Cell, float]:
    unitcell_tensors = unitcell.get_unique_tensors()
    if len(peps_tensors) != len(unitcell_tensors) or not all(
        peps_tensors[i] is unitcell_tensors[i].tensor for i in range(len(peps_tensors))
    ):
        raise ValueError("PEPS tensor sequence mismatches the unitcell.")

    alpha = initial_step_size

    new_tensors_func = jit(
        lambda p_t, des_dir, alpha: [
            p_t[i] + alpha * des_dir[i] for i in range(len(p_t))
        ]
    )

    count = 0
    while count < max_steps:
        new_tensors = new_tensors_func(peps_tensors, descent_direction, alpha)

        unitcell_tensors = unitcell.get_unique_tensors()
        new_unitcell = unitcell.replace_unique_tensors(
            [
                unitcell_tensors[i].replace_tensor(
                    new_tensors[i], reinitialize_env_as_identities=True
                )
                for i in range(len(new_tensors))
            ]
        )

        new_value, (new_unitcell, _) = calc_ctmrg_expectation(
            new_tensors,
            new_unitcell,
            expectation_func,
            eps=ctmrg_eps,
            max_steps=ctmrg_max_steps,
        )

        if new_value < current_value:
            break

        alpha = reduction_factor * alpha
        count += 1

    if count == 100:
        raise RuntimeError("Line search does not find a suitable step size.")

    return new_tensors, new_unitcell, new_value
