import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import calc_ctmrg_expectation

from typing import Sequence, Tuple, List, Literal

Line_Search_Methods = Literal["simple", "armijo", "wolfe"]


@jit
def _line_search_new_tensors(peps_tensors, descent_dir, alpha):
    return [peps_tensors[i] + alpha * descent_dir[i] for i in range(len(peps_tensors))]


@jit
def _armijo_value(current_val, descent_dir, gradient, alpha, const_factor):
    return current_val + const_factor * alpha * jnp.real(jnp.sum(descent_dir.conj() * gradient))


def line_search(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    gradient: Sequence[jnp.ndarray],
    descent_direction: Sequence[jnp.ndarray],
    current_value: float,
    *,
    method: Line_Search_Methods = "simple",
    ctmrg_eps: float = 1e-5,
    ctmrg_max_steps: int = 20,
    initial_step_size: float = 1.0,
    reduction_factor: float = 0.25,
    max_steps: int = 20,
    armijo_constant_factor: float = 1e-4,
    wolfe_curvature_factor: float = 0.1,
) -> Tuple[List[jnp.ndarray], PEPS_Unit_Cell, float]:
    unitcell_tensors = unitcell.get_unique_tensors()
    if len(peps_tensors) != len(unitcell_tensors) or not all(
        peps_tensors[i] is unitcell_tensors[i].tensor for i in range(len(peps_tensors))
    ):
        raise ValueError("PEPS tensor sequence mismatches the unitcell.")

    alpha = initial_step_size

    count = 0
    while count < max_steps:
        new_tensors = _line_search_new_tensors(peps_tensors, descent_direction, alpha)

        unitcell_tensors = unitcell.get_unique_tensors()
        new_unitcell = unitcell.replace_unique_tensors(
            [
                unitcell_tensors[i].replace_tensor(
                    new_tensors[i], reinitialize_env_as_identities=True
                )
                for i in range(len(new_tensors))
            ]
        )

        if method == "simple" or method == "armijo":
            new_value, (new_unitcell, _) = calc_ctmrg_expectation(
                new_tensors,
                new_unitcell,
                expectation_func,
                eps=ctmrg_eps,
                max_steps=ctmrg_max_steps,
            )
        elif method == "wolfe":
            raise NotImplementedError("Wolfe condition not implemented yet.")
        else:
            raise ValueError("Unknown line search method.")

        if method == "simple":
            if new_value < current_value:
                break
        elif method == "armijo":
            cmp_value = _armijo_value(
                current_value,
                descent_direction,
                gradient,
                alpha,
                armijo_constant_factor,
            )
            if new_value < cmp_value:
                break
        elif method == "wolfe":
            pass

        alpha = reduction_factor * alpha
        count += 1

    if count == max_steps:
        raise RuntimeError("Line search does not find a suitable step size.")

    return new_tensors, new_unitcell, new_value
