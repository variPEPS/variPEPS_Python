from tqdm_loggable.auto import tqdm

import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree

from peps_ad import peps_ad_config
from peps_ad.config import Line_Search_Methods
from peps_ad.ctmrg import CTMRGNotConvergedError, CTMRGGradientNotConvergedError
from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model
from peps_ad.mapping import Map_To_PEPS_Model

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)

from typing import Sequence, Tuple, List, Union, Optional


@jit
def _scalar_descent_grad(descent_dir, gradient):
    descent_dir_real, _ = ravel_pytree(descent_dir)
    gradient_real, _ = ravel_pytree(gradient)
    if jnp.iscomplexobj(descent_dir_real):
        descent_dir_real = jnp.concatenate(
            (jnp.real(descent_dir_real), jnp.imag(descent_dir_real))
        )
        gradient_real = jnp.concatenate(
            (jnp.real(gradient_real), jnp.imag(gradient_real))
        )
    return jnp.sum(descent_dir_real * gradient_real)


@jit
def _line_search_new_tensors(peps_tensors, descent_dir, alpha):
    return [peps_tensors[i] + alpha * descent_dir[i] for i in range(len(peps_tensors))]


@jit
def _armijo_value(current_val, descent_dir, gradient, alpha, const_factor):
    descent_dir_real, _ = ravel_pytree(descent_dir)
    gradient_real, _ = ravel_pytree(gradient)
    if jnp.iscomplexobj(descent_dir_real):
        descent_dir_real = jnp.concatenate(
            (jnp.real(descent_dir_real), jnp.imag(descent_dir_real))
        )
        gradient_real = jnp.concatenate(
            (jnp.real(gradient_real), jnp.imag(gradient_real))
        )
    return jnp.fmin(
        current_val,
        current_val + const_factor * alpha * jnp.sum(descent_dir_real * gradient_real),
    )


@jit
def _wolfe_value(
    current_val,
    descent_dir,
    gradient,
    new_gradient,
    alpha,
    armijo_const_factor,
    wolfe_const_factor,
):
    descent_dir_real, _ = ravel_pytree(descent_dir)
    gradient_real, _ = ravel_pytree(gradient)
    new_gradient_real, _ = ravel_pytree(new_gradient)

    if jnp.iscomplexobj(descent_dir_real):
        descent_dir_real = jnp.concatenate(
            (jnp.real(descent_dir_real), jnp.imag(descent_dir_real))
        )
        gradient_real = jnp.concatenate(
            (jnp.real(gradient_real), jnp.imag(gradient_real))
        )
        new_gradient_real = jnp.concatenate(
            (jnp.real(new_gradient_real), jnp.imag(new_gradient_real))
        )

    scalar_descent_grad = jnp.sum(descent_dir_real * gradient_real)

    cmp_value = current_val + armijo_const_factor * alpha * scalar_descent_grad

    scalar_descent_new_grad = jnp.sum(descent_dir_real * new_gradient_real)
    strong_wolfe_left_side = -scalar_descent_new_grad
    strong_wolfe_right_side = -wolfe_const_factor * scalar_descent_grad

    return (
        cmp_value,
        strong_wolfe_left_side,
        strong_wolfe_right_side,
        scalar_descent_new_grad,
    )


@jit
def _wolfe_new_alpha(
    alpha,
    last_alpha,
    value,
    last_value,
    descent_grad,
    descent_last_grad,
    lower_bound,
    upper_bound,
):
    d1 = (
        descent_last_grad
        + descent_grad
        - 3 * (last_value - value) / (last_alpha - alpha)
    )
    d2 = jnp.sign(alpha - last_alpha) * jnp.sqrt(
        d1**2 - descent_last_grad * descent_grad
    )
    new_alpha = alpha - (alpha - last_alpha) * (descent_grad + d2 - d1) / (
        descent_grad - descent_last_grad + 2 * d2
    )
    return jnp.where(
        jnp.isinf(value)
        | jnp.isinf(last_value)
        | (new_alpha <= lower_bound)
        | (new_alpha >= upper_bound)
        | jnp.isnan(new_alpha),
        lower_bound + (upper_bound - lower_bound) / 2,
        new_alpha,
    )


class NoSuitableStepSizeError(Exception):
    pass


def line_search(
    input_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    gradient: jnp.ndarray,
    descent_direction: jnp.ndarray,
    current_value: Union[float, jnp.ndarray],
    last_step_size: Union[float, jnp.ndarray],
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None,
) -> Tuple[
    List[jnp.ndarray],
    PEPS_Unit_Cell,
    Union[float, jnp.ndarray],
    Union[float, jnp.ndarray],
]:
    """
    Run two-way backtracing line search method for the CTMRG routine.

    Args:
      input_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the current tensors which should be optimized.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate one expectation value which is used as loss
        loss function of the model. Likely the function to calculate the energy.
      gradient (:obj:`jax.numpy.ndarray`):
        The gradient of the CTMRG method and expectation function for the
        current step.
      descent_direction (:obj:`jax.numpy.ndarray`):
        The descent direction which should be used for the line search.
      current_value (:obj:`float` or :obj:`jax.numpy.ndarray`):
        The current value of the evaluation of the expectation function.
      last_step_size (:obj:`float` or :obj:`jax.numpy.ndarray`):
        The step size found in the last line search.
      convert_to_unitcell_func (:obj:`~peps_ad.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the input.
    Returns:
      :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`float`, :obj:`float`):
        Tuple with the optimized tensors, the new unitcell, the reduced
        expectation value and the step size found in the line search.
    Raises:
      :obj:`ValueError`: The parameters mismatch the expected inputs.
      :obj:`RuntimeError`: The line search does not converge.
    """

    alpha = (
        last_step_size
        if last_step_size is not None and peps_ad_config.line_search_use_last_step_size
        else peps_ad_config.line_search_initial_step_size
    )
    has_been_increased = False
    incrementation_not_helped = False
    enforce_elementwise_convergence = (
        peps_ad_config.ctmrg_enforce_elementwise_convergence
        or peps_ad_config.ad_use_custom_vjp
    )

    wolfe_upper_bound = None
    wolfe_lower_bound = None
    wolfe_alpha_last_step = 0
    wolfe_descent_new_grad = _scalar_descent_grad(descent_direction, gradient)

    new_value = current_value

    count = 0
    while count < peps_ad_config.line_search_max_steps:
        new_tensors = _line_search_new_tensors(input_tensors, descent_direction, alpha)

        if convert_to_unitcell_func is None:
            unitcell_tensors = unitcell.get_unique_tensors()
            new_unitcell = unitcell.replace_unique_tensors(
                [
                    unitcell_tensors[i].replace_tensor(
                        new_tensors[i], reinitialize_env_as_identities=True
                    )
                    for i in range(len(new_tensors))
                ]
            )
        else:
            new_unitcell = None

        if (
            peps_ad_config.line_search_method is Line_Search_Methods.SIMPLE
            or peps_ad_config.line_search_method is Line_Search_Methods.ARMIJO
        ):
            try:
                new_value, new_unitcell = calc_ctmrg_expectation(
                    new_tensors,
                    new_unitcell,
                    expectation_func,
                    convert_to_unitcell_func,
                    enforce_elementwise_convergence=enforce_elementwise_convergence,
                )
            except CTMRGNotConvergedError:
                new_value = jnp.inf
        elif peps_ad_config.line_search_method is Line_Search_Methods.WOLFE:
            wolfe_value_last_step = new_value

            try:
                if peps_ad_config.ad_use_custom_vjp:
                    (
                        new_value,
                        new_unitcell,
                    ), new_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                        new_tensors,
                        new_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                    )
                else:
                    (
                        new_value,
                        new_unitcell,
                    ), new_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                        new_tensors,
                        new_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        calc_preconverged=True,
                    )
                new_gradient = [elem.conj() for elem in new_gradient_seq]
            except (CTMRGNotConvergedError, CTMRGGradientNotConvergedError):
                new_value = jnp.inf
                new_gradient = gradient
        else:
            raise ValueError("Unknown line search method.")

        if peps_ad_config.line_search_method is Line_Search_Methods.SIMPLE:
            smaller_value_found = (
                new_value <= current_value or (new_value - current_value) <= 1e-13
            )
        elif peps_ad_config.line_search_method is Line_Search_Methods.ARMIJO:
            cmp_value = _armijo_value(
                current_value,
                descent_direction,
                gradient,
                alpha,
                peps_ad_config.line_search_armijo_const,
            )
            smaller_value_found = (
                new_value <= cmp_value or (new_value - cmp_value) <= 1e-13
            )
        elif peps_ad_config.line_search_method is Line_Search_Methods.WOLFE:
            wolfe_descent_last_grad = wolfe_descent_new_grad
            (
                cmp_value,
                wolfe_left_side,
                wolfe_right_side,
                wolfe_descent_new_grad,
            ) = _wolfe_value(
                current_value,
                descent_direction,
                gradient,
                new_gradient,
                alpha,
                peps_ad_config.line_search_armijo_const,
                peps_ad_config.line_search_wolfe_const,
            )
            wolfe_cond_1 = new_value <= cmp_value or (new_value - cmp_value) <= 1e-13
            wolfe_cond_2 = (
                wolfe_left_side <= wolfe_right_side
                or (wolfe_left_side - wolfe_right_side) <= 1e-13
            )

        if (
            peps_ad_config.line_search_method is Line_Search_Methods.SIMPLE
            or peps_ad_config.line_search_method is Line_Search_Methods.ARMIJO
        ):
            if smaller_value_found:
                if (
                    alpha >= peps_ad_config.line_search_initial_step_size
                    or incrementation_not_helped
                ):
                    break

                has_been_increased = True
                alpha /= peps_ad_config.line_search_reduction_factor
            else:
                if has_been_increased:
                    incrementation_not_helped = True

                alpha = peps_ad_config.line_search_reduction_factor * alpha
        elif peps_ad_config.line_search_method is Line_Search_Methods.WOLFE:
            if wolfe_upper_bound is None and wolfe_lower_bound is None:
                if jnp.isinf(new_value):
                    alpha /= peps_ad_config.line_search_reduction_factor
                elif not wolfe_cond_1 or (
                    count > 0 and new_value >= wolfe_value_last_step
                ):
                    wolfe_lower_bound = wolfe_alpha_last_step
                    wolfe_lower_bound_value = wolfe_value_last_step
                    wolfe_upper_bound = alpha
                    wolfe_upper_bound_value = new_value
                    tmp_alpha = alpha
                    alpha = _wolfe_new_alpha(
                        alpha,
                        wolfe_alpha_last_step,
                        new_value,
                        wolfe_value_last_step,
                        wolfe_descent_new_grad,
                        wolfe_descent_last_grad,
                        jnp.fmin(wolfe_lower_bound, wolfe_upper_bound),
                        jnp.fmax(wolfe_lower_bound, wolfe_upper_bound),
                    )
                    wolfe_alpha_last_step = tmp_alpha
                elif wolfe_cond_2:
                    break
                elif wolfe_descent_new_grad >= 0:
                    wolfe_lower_bound = alpha
                    wolfe_lower_bound_value = new_value
                    wolfe_upper_bound = wolfe_alpha_last_step
                    wolfe_upper_bound_value = wolfe_value_last_step
                    tmp_alpha = alpha
                    alpha = _wolfe_new_alpha(
                        alpha,
                        wolfe_alpha_last_step,
                        new_value,
                        wolfe_value_last_step,
                        wolfe_descent_new_grad,
                        wolfe_descent_last_grad,
                        jnp.fmin(wolfe_lower_bound, wolfe_upper_bound),
                        jnp.fmax(wolfe_lower_bound, wolfe_upper_bound),
                    )
                    wolfe_alpha_last_step = tmp_alpha
                else:
                    wolfe_alpha_last_step = alpha
                    alpha /= peps_ad_config.line_search_reduction_factor
            elif jnp.isinf(new_value):
                alpha = alpha + (wolfe_upper_bound - alpha) / 2
            else:
                if new_value > cmp_value or new_value >= wolfe_lower_bound_value:
                    wolfe_upper_bound = alpha
                    wolfe_upper_bound_value = new_value
                else:
                    if wolfe_cond_2:
                        break

                    if (
                        wolfe_descent_new_grad * (wolfe_upper_bound - wolfe_lower_bound)
                        >= 0
                    ):
                        wolfe_upper_bound = wolfe_lower_bound
                        wolfe_upper_bound_value = wolfe_lower_bound_value

                    wolfe_lower_bound = alpha
                    wolfe_lower_bound_value = new_value

                tmp_alpha = alpha
                alpha = _wolfe_new_alpha(
                    alpha,
                    wolfe_alpha_last_step,
                    new_value,
                    wolfe_value_last_step,
                    wolfe_descent_new_grad,
                    wolfe_descent_last_grad,
                    jnp.fmin(wolfe_lower_bound, wolfe_upper_bound),
                    jnp.fmax(wolfe_lower_bound, wolfe_upper_bound),
                )
                wolfe_alpha_last_step = tmp_alpha

        count += 1

    if count == peps_ad_config.line_search_max_steps:
        raise NoSuitableStepSizeError(f"Count {count}, Last alpha {alpha}")

    return new_tensors, new_unitcell, new_value, alpha
