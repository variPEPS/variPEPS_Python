from collections import deque
from functools import partial

from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from jax.flatten_util import ravel_pytree

from peps_ad import peps_ad_config, peps_ad_global_state
from peps_ad.config import Optimizing_Methods
from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model
from peps_ad.config import Projector_Method
from peps_ad.mapping import Map_To_PEPS_Model

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)
from .line_search import line_search, NoSuitableStepSizeError, _scalar_descent_grad

from typing import List, Union, Tuple, cast, Sequence, Callable, Optional


@jit
def _cg_workhorse(new_gradient, old_gradient, old_descent_dir):
    new_grad_vec, new_grad_unravel = ravel_pytree(new_gradient)
    old_grad_vec, old_grad_unravel = ravel_pytree(old_gradient)
    old_des_dir_vec, old_des_dir_unravel = ravel_pytree(old_descent_dir)

    new_grad_len = new_grad_vec.size
    iscomplex = jnp.iscomplexobj(new_grad_vec)

    if iscomplex:
        new_grad_vec = jnp.concatenate((jnp.real(new_grad_vec), jnp.imag(new_grad_vec)))
        old_grad_vec = jnp.concatenate((jnp.real(old_grad_vec), jnp.imag(old_grad_vec)))
        old_des_dir_vec = jnp.concatenate(
            (jnp.real(old_des_dir_vec), jnp.imag(old_des_dir_vec))
        )

    grad_diff = new_grad_vec - old_grad_vec

    # dx = -new_grad_vec
    # dx_old = -old_grad_vec
    # dx_real = jnp.concatenate((jnp.real(dx), jnp.imag(dx)))
    # dx_old_real = jnp.concatenate((jnp.real(dx_old), jnp.imag(dx_old)))
    # old_des_dir_real = jnp.concatenate(
    #     (jnp.real(old_descent_dir), jnp.imag(old_descent_dir))
    # )
    # PRP
    # beta = jnp.sum(dx_real * (dx_real - dx_old_real)) / jnp.sum(dx_old_real * dx_old_real)
    # LS parameter
    # beta = jnp.sum(dx_real * (dx_old_real - dx_real)) / jnp.sum(
    #     old_des_dir_real * dx_old_real
    # )

    # Hager-Zhang
    eta = 0.4
    eta_k = -1 / (
        jnp.linalg.norm(old_des_dir_vec) * jnp.fmin(eta, jnp.linalg.norm(old_grad_vec))
    )
    old_des_grad_diff = jnp.dot(old_des_dir_vec, grad_diff)
    beta = (
        grad_diff - 2 * jnp.linalg.norm(grad_diff) * old_des_dir_vec / old_des_grad_diff
    )
    beta = jnp.dot(beta, new_grad_vec) / old_des_grad_diff
    beta = jnp.fmax(eta_k, beta)

    beta = jnp.fmax(0, beta)

    result = -new_grad_vec + beta * old_des_dir_vec

    if iscomplex:
        result = result[:new_grad_len] + 1j * result[new_grad_len:]

    return new_grad_unravel(result), beta


@partial(jit, static_argnums=(5,))
def _bfgs_workhorse(
    new_gradient, old_gradient, old_descent_dir, old_alpha, B_inv, calc_new_B_inv
):
    new_grad_vec, new_grad_unravel = ravel_pytree(new_gradient)
    new_grad_len = new_grad_vec.size

    iscomplex = jnp.iscomplexobj(new_grad_vec)
    if iscomplex:
        new_grad_vec = jnp.concatenate((jnp.real(new_grad_vec), jnp.imag(new_grad_vec)))

    if calc_new_B_inv:
        old_grad_vec, old_grad_unravel = ravel_pytree(old_gradient)
        old_descent_dir_vec, old_descent_dir_unravel = ravel_pytree(old_descent_dir)
        if iscomplex:
            old_grad_vec = jnp.concatenate(
                (jnp.real(old_grad_vec), jnp.imag(old_grad_vec))
            )
            old_descent_dir_vec = jnp.concatenate(
                (jnp.real(old_descent_dir_vec), jnp.imag(old_descent_dir_vec))
            )

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

    result = -jnp.dot(new_B_inv, new_grad_vec)

    if iscomplex:
        result = result[:new_grad_len] + 1j * result[new_grad_len:]

    return new_grad_unravel(result), new_B_inv


@jit
def _l_bfgs_workhorse(value_tuple, gradient_tuple):
    gradient_elem_0, gradient_unravel = ravel_pytree(gradient_tuple[0])
    gradient_len = gradient_elem_0.size

    iscomplex = jnp.iscomplexobj(gradient_elem_0)

    def _make_1d(x):
        x_1d, _ = ravel_pytree(x)
        if iscomplex:
            return jnp.concatenate((jnp.real(x_1d), jnp.imag(x_1d)))
        return x_1d

    value_arr = jnp.asarray([_make_1d(e) for e in value_tuple])
    gradient_arr = jnp.asarray([_make_1d(e) for e in gradient_tuple])

    s_arr = -jnp.diff(value_arr, axis=0)
    y_arr = -jnp.diff(gradient_arr, axis=0)
    pho_arr = 1 / jnp.sum(y_arr * s_arr, axis=1)

    def first_loop(q, x):
        pho_s, y = x
        alpha_i = jnp.sum(pho_s * q)
        return q - alpha_i * y, alpha_i

    q, alpha_arr = scan(
        first_loop,
        gradient_arr[0],
        (pho_arr[:, jnp.newaxis] * s_arr, y_arr),
    )

    gamma = jnp.sum(s_arr[0] * y_arr[0]) / jnp.sum(y_arr[0] * y_arr[0])

    z_result = gamma * q

    def second_loop(z, x):
        pho_y, s, alpha_i = x
        beta_i = jnp.sum(pho_y * z)
        return z + s * (alpha_i - beta_i), None

    z_result, _ = scan(
        second_loop,
        z_result,
        (pho_arr[:, jnp.newaxis] * y_arr, s_arr, alpha_arr),
    )

    z_result = -z_result
    if iscomplex:
        z_result = z_result[:gradient_len] + 1j * z_result[gradient_len:]
    return gradient_unravel(z_result)


def optimize_peps_network(
    input_tensors: Union[PEPS_Unit_Cell, Sequence[jnp.ndarray]],
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None,
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell, Union[float, jnp.ndarray]]:
    """
    Optimize a PEPS unitcell using a variational method.

    As convergence criterion the norm of the gradient is used.

    Args:
      input_tensors (:obj:`~peps_ad.peps.PEPS_Unit_Cell` or :term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS unitcell to work on or the tensors which should be mapped by
        `convert_to_unitcell_func` to a PEPS unitcell.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate one expectation value which is used as loss
        loss function of the model. Likely the function to calculate the energy.
      convert_to_unitcell_func (:obj:`~peps_ad.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the first input parameter.
    Returns:
      :obj:`tuple`\ (:term:`sequence`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`float`):
        Tuple with the optimized tensors, network and the final expectation value.
    """
    if isinstance(input_tensors, PEPS_Unit_Cell):
        working_tensors = cast(
            List[jnp.ndarray], [i.tensor for i in input_tensors.get_unique_tensors()]
        )
        working_unitcell = input_tensors
    else:
        working_tensors = input_tensors
        working_unitcell = None

    old_gradient = None
    old_descent_dir = None
    descent_dir = None

    if peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
        bfgs_B_inv = jnp.eye(2 * sum([t.size for t in working_tensors]))
    elif peps_ad_config.optimizer_method is Optimizing_Methods.L_BFGS:
        l_bfgs_x_cache = deque(maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1)
        l_bfgs_grad_cache = deque(maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1)

    count = 0
    linesearch_step: Union[
        float, jnp.ndarray
    ] = peps_ad_config.line_search_initial_step_size
    working_value: Union[float, jnp.ndarray]

    if peps_ad_config.optimizer_preconverge_with_half_projectors:
        peps_ad_global_state.ctmrg_projector_method = Projector_Method.HALF
    else:
        peps_ad_global_state.ctmrg_projector_method = None

    while count < peps_ad_config.optimizer_max_steps:
        if peps_ad_config.ad_use_custom_vjp:
            (
                working_value,
                working_unitcell,
            ), working_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                working_tensors,
                working_unitcell,
                expectation_func,
                convert_to_unitcell_func,
            )
        else:
            (
                working_value,
                working_unitcell,
            ), working_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                working_tensors,
                working_unitcell,
                expectation_func,
                convert_to_unitcell_func,
                calc_preconverged=(count == 0),
            )

        print(f"{count} before: Value {working_value}")

        working_gradient = [elem.conj() for elem in working_gradient_seq]

        if peps_ad_config.optimizer_method is Optimizing_Methods.STEEPEST:
            descent_dir = [-elem for elem in working_gradient]
        elif peps_ad_config.optimizer_method is Optimizing_Methods.CG:
            if count == 0:
                descent_dir = [-elem for elem in working_gradient]
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
        elif peps_ad_config.optimizer_method is Optimizing_Methods.L_BFGS:
            l_bfgs_x_cache.appendleft(tuple(working_tensors))
            l_bfgs_grad_cache.appendleft(tuple(working_gradient))

            if count == 0:
                descent_dir = [-elem for elem in working_gradient]
            else:
                descent_dir = _l_bfgs_workhorse(
                    tuple(l_bfgs_x_cache), tuple(l_bfgs_grad_cache)
                )
        else:
            raise ValueError("Unknown optimization method.")

        if _scalar_descent_grad(descent_dir, working_gradient) > 0:
            print("Found bad descent dir. Reset to negative gradient!")
            descent_dir = [-elem for elem in working_gradient]

        conv = jnp.linalg.norm(ravel_pytree(working_gradient)[0])

        try:
            (
                working_tensors,
                working_unitcell,
                working_value,
                linesearch_step,
            ) = line_search(
                working_tensors,
                working_unitcell,
                expectation_func,
                working_gradient,
                descent_dir,
                working_value,
                linesearch_step,
                convert_to_unitcell_func,
            )
        except NoSuitableStepSizeError:
            if peps_ad_config.optimizer_fail_if_no_step_size_found:
                raise
            else:
                conv = 0

        if conv < peps_ad_config.optimizer_convergence_eps:
            working_value, working_unitcell = calc_ctmrg_expectation(
                working_tensors,
                working_unitcell,
                expectation_func,
                convert_to_unitcell_func,
                enforce_elementwise_convergence=peps_ad_config.ad_use_custom_vjp,
            )
            peps_ad_global_state.ctmrg_projector_method = None
            break

        if (
            peps_ad_config.optimizer_preconverge_with_half_projectors
            and conv < peps_ad_config.optimizer_preconverge_with_half_projectors_eps
        ):
            peps_ad_global_state.ctmrg_projector_method = (
                peps_ad_config.ctmrg_full_projector_method
            )

        old_descent_dir = descent_dir
        old_gradient = working_gradient

        count += 1

        print(
            f"{count} after: Value {working_value}, Conv: {conv}, Alpha: {linesearch_step}"
        )

    return working_tensors, working_unitcell, working_value
