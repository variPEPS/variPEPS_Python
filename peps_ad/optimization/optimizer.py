from functools import partial

from jax import jit
import jax.numpy as jnp

from peps_ad import peps_ad_config
from peps_ad.config import Optimizing_Methods
from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)
from .line_search import line_search

from typing import List, Union, Tuple, cast


@jit
def _cg_workhorse(new_gradient, old_gradient, old_descent_dir):
    dx = -new_gradient
    dx_old = -old_gradient
    dx_real = jnp.concatenate((jnp.real(dx), jnp.imag(dx)))
    dx_old_real = jnp.concatenate((jnp.real(dx_old), jnp.imag(dx_old)))
    old_des_dir_real = jnp.concatenate(
        (jnp.real(old_descent_dir), jnp.imag(old_descent_dir))
    )
    # PRP
    # beta = jnp.sum(dx_real * (dx_real - dx_old_real)) / jnp.sum(dx_old_real * dx_old_real)
    # LS parameter
    beta = jnp.sum(dx_real * (dx_old_real - dx_real)) / jnp.sum(
        old_des_dir_real * dx_old_real
    )
    beta = jnp.fmax(0, beta)
    return dx + beta * old_descent_dir, beta


@partial(jit, static_argnums=(5,))
def _bfgs_workhorse(
    new_gradient, old_gradient, old_descent_dir, old_alpha, B_inv, calc_new_B_inv
):
    new_grad_vec = jnp.concatenate((jnp.real(new_gradient), jnp.imag(new_gradient)))
    new_grad_vec = jnp.ravel(new_grad_vec)

    if calc_new_B_inv:
        old_grad_vec = jnp.concatenate((jnp.real(old_gradient), jnp.imag(old_gradient)))
        old_grad_vec = jnp.ravel(old_grad_vec)
        old_descent_dir_vec = jnp.concatenate(
            (jnp.real(old_descent_dir), jnp.imag(old_descent_dir))
        )
        old_descent_dir_vec = jnp.ravel(old_descent_dir_vec)

        sk = old_alpha * old_descent_dir_vec
        yk = new_grad_vec - old_grad_vec

        skyk_scalar = jnp.dot(sk, yk)
        B_inv_yk = jnp.dot(B_inv, yk)

        new_B_inv = (
            B_inv
            + ((skyk_scalar + jnp.dot(yk, B_inv_yk)) / (skyk_scalar**2))
            * jnp.outer(sk, sk)
            - (jnp.outer(B_inv_yk, sk) + jnp.outer(sk, B_inv_yk)) / skyk_scalar
        )
    else:
        new_B_inv = B_inv

    result_vec = -jnp.dot(new_B_inv, new_grad_vec)

    result = result_vec[: new_gradient.size] + 1j * result_vec[new_gradient.size :]

    return result.reshape(new_gradient.shape), new_B_inv


def optimize_peps_network(
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
) -> Tuple[PEPS_Unit_Cell, Union[float, jnp.ndarray]]:
    """
    Optimize a PEPS unitcell using a variational method.

    As convergence criterion the norm of the gradient is used.

    Args:
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate one expectation value which is used as loss
        loss function of the model. Likely the function to calculate the energy.
    Returns:
      :obj:`tuple`\ (:obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`float`):
        Tuple with the optimized network and the final expectation value.
    """
    working_tensors = cast(
        List[jnp.ndarray], [i.tensor for i in unitcell.get_unique_tensors()]
    )
    working_unitcell = unitcell

    old_gradient = None
    old_descent_dir = None
    descent_dir = None

    if peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
        bfgs_B_inv = jnp.eye(2 * sum([t.size for t in working_tensors]))

    count = 0
    linesearch_step: Union[
        float, jnp.ndarray
    ] = peps_ad_config.line_search_initial_step_size
    working_value: Union[float, jnp.ndarray]

    while count < peps_ad_config.optimizer_max_steps:
        if peps_ad_config.ad_use_custom_vjp:
            (
                working_value,
                working_unitcell,
            ), working_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                working_tensors,
                working_unitcell,
                expectation_func,
            )
        else:
            (
                working_value,
                working_unitcell,
            ), working_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                working_tensors,
                working_unitcell,
                expectation_func,
                calc_preconverged=(count == 0),
            )

        print(f"{count} before: Value {working_value}")

        working_gradient = jnp.asarray([elem.conj() for elem in working_gradient_seq])

        if peps_ad_config.optimizer_method is Optimizing_Methods.STEEPEST:
            descent_dir = -working_gradient
        elif peps_ad_config.optimizer_method is Optimizing_Methods.CG:
            if count == 0:
                descent_dir = -working_gradient
            else:
                descent_dir, beta = _cg_workhorse(
                    working_gradient, old_gradient, old_descent_dir
                )
                print(beta)
        elif peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
            if count == 0:
                descent_dir, _ = _bfgs_workhorse(
                    working_gradient, None, None, None, bfgs_B_inv, False
                )
            else:
                descent_dir, bfgs_B_inv = _bfgs_workhorse(
                    working_gradient,
                    old_gradient,
                    old_descent_dir,
                    linesearch_step,
                    bfgs_B_inv,
                    True,
                )
        else:
            raise ValueError("Unknown optimization method.")

        working_tensors, working_unitcell, working_value, linesearch_step = line_search(
            working_tensors,
            working_unitcell,
            expectation_func,
            working_gradient,
            descent_dir,
            working_value,
            linesearch_step,
        )

        conv = jnp.linalg.norm(working_gradient)
        if conv < peps_ad_config.optimizer_convergence_eps:
            working_value, (working_unitcell, _) = calc_ctmrg_expectation(
                working_tensors,
                working_unitcell,
                expectation_func,
                enforce_elementwise_convergence=peps_ad_config.ad_use_custom_vjp,
            )
            break

        old_descent_dir = descent_dir
        old_gradient = working_gradient

        count += 1

        print(
            f"{count} after: Value {working_value}, Conv: {conv}, Alpha: {linesearch_step}"
        )

    return working_unitcell, working_value
