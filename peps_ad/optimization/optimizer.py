import jax.numpy as jnp

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
)
from .line_search import simple_line_search

from typing import Literal

Optimizing_Methods = Literal["steepest"]


def optimize_peps_network(
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    *,
    method: Optimizing_Methods = "steepest",
    max_steps: int = 100,
    eps: float = 1e-6
):
    working_tensors = [i.tensor for i in unitcell.get_unique_tensors()]
    working_unitcell = unitcell

    descent_dirs = []

    count = 0
    converged = False

    while count < max_steps:
        (
            working_value,
            (working_unitcell, _, _),
        ), working_gradient = calc_preconverged_ctmrg_value_and_grad(
            working_tensors, working_unitcell, expectation_func
        )

        print(f"{count} before: Value {working_value}")

        working_gradient = [elem.conj() for elem in working_gradient]

        if method == "steepest":
            if len(descent_dirs) > 0 and all(
                jnp.linalg.norm(descent_dirs[-1][i] - working_gradient[i]) < eps
                for i in range(len(working_gradient))
            ):
                converged = True

            descent_dirs.append(working_gradient)

            working_tensors, working_unitcell, working_value = simple_line_search(
                working_tensors,
                working_unitcell,
                expectation_func,
                working_gradient,
                working_value,
            )
        else:
            raise ValueError("Unknown optimization method.")

        if converged:
            working_value, (working_unitcell, _) = calc_ctmrg_expectation(
                working_tensors, working_unitcell, expectation_func
            )
            break

        count += 1

        print(f"{count} after: Value {working_value}")

#        return working_tensors, working_unitcell

    return working_unitcell, working_value
