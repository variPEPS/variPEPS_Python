from jax import jit
import jax.numpy as jnp

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
)
from .line_search import line_search, Line_Search_Methods

from typing import Literal

Optimizing_Methods = Literal["steepest", "cg"]


@jit
def _cg_workhorse(new_gradient, old_gradient, old_descent_dir):
    dx = -new_gradient
    dx_old = -old_gradient
    # PRP
    # beta = jnp.sum(dx.conj() * (dx - dx_old)) / jnp.sum(dx_old.conj() * dx_old)
    # LS parameter
    beta = jnp.sum(dx.conj() * (dx_old - dx)) / jnp.sum(old_descent_dir.conj() * dx_old)
    beta = jnp.fmax(0, beta)
    return dx + beta * old_descent_dir


def optimize_peps_network(
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    *,
    method: Optimizing_Methods = "steepest",
    line_search_method: Line_Search_Methods = "simple",
    max_steps: int = 100,
    eps: float = 1e-5,
):
    working_tensors = [i.tensor for i in unitcell.get_unique_tensors()]
    working_unitcell = unitcell

    gradient_list = []
    descent_dirs = []

    count = 0

    while count < max_steps:
        (
            working_value,
            (working_unitcell, _, _),
        ), working_gradient = calc_preconverged_ctmrg_value_and_grad(
            working_tensors, working_unitcell, expectation_func
        )

        print(f"{count} before: Value {working_value}")

        working_gradient = jnp.asarray([elem.conj() for elem in working_gradient])
        gradient_list.append(working_gradient)

        if method == "steepest":
            descent_dirs.append(-working_gradient)
        elif method == "cg":
            if len(descent_dirs) == 0:
                descent_dirs.append(-working_gradient)
            else:
                descent_dirs.append(
                    _cg_workhorse(working_gradient, gradient_list[-2], descent_dirs[-1])
                )
        else:
            raise ValueError("Unknown optimization method.")

        working_tensors, working_unitcell, working_value = line_search(
            working_tensors,
            working_unitcell,
            expectation_func,
            working_gradient,
            descent_dirs[-1],
            working_value,
            method=line_search_method,
        )

        if jnp.linalg.norm(working_gradient) < eps:
            working_value, (working_unitcell, _) = calc_ctmrg_expectation(
                working_tensors, working_unitcell, expectation_func
            )
            break

        count += 1

        print(f"{count} after: Value {working_value}")

    return working_unitcell, working_value
