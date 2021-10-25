import jax.numpy as jnp
from jax import value_and_grad

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model
from peps_ad.ctmrg import calc_ctmrg_env

from typing import Sequence


def calc_ctmrg_expectation(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    *,
    eps: float = 1e-8,
    max_steps: int = 200,
) -> jnp.ndarray:
    new_unitcell, corner_svds = calc_ctmrg_env(
        peps_tensors, unitcell, eps=eps, max_steps=max_steps
    )

    return expectation_func(peps_tensors, new_unitcell), (new_unitcell, corner_svds)


calc_ctmrg_expectation_value_and_grad = value_and_grad(calc_ctmrg_expectation, has_aux=True)
