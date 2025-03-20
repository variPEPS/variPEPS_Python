import enum

from tqdm_loggable.auto import tqdm

import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree

from varipeps import varipeps_config
from varipeps.config import Line_Search_Methods, Wavevector_Type
from varipeps.ctmrg import CTMRGNotConvergedError, CTMRGGradientNotConvergedError
from varipeps.peps import PEPS_Unit_Cell
from varipeps.expectation import Expectation_Model
from varipeps.mapping import Map_To_PEPS_Model
from varipeps.utils.debug_print import debug_print

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)

from typing import Sequence, Tuple, List, Union, Optional, Dict


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


def _get_new_unitcell(
    new_tensors,
    unitcell,
    spiral_indices,
    convert_to_unitcell_func,
    generate_unitcell,
    reinitialize_env_as_identities,
):
    if spiral_indices is not None:
        for i in spiral_indices:
            if (
                varipeps_config.spiral_wavevector_type
                is Wavevector_Type.TWO_PI_POSITIVE_ONLY
            ):
                new_tensors[i] = new_tensors[i] % 2
            elif (
                varipeps_config.spiral_wavevector_type
                is Wavevector_Type.TWO_PI_SYMMETRIC
            ):
                new_tensors[i] = new_tensors[i] % 4 - 2
            else:
                raise ValueError("Unknown wavevector type!")

    if convert_to_unitcell_func is None or generate_unitcell:
        unitcell_tensors = unitcell.get_unique_tensors()
        new_unitcell = unitcell.replace_unique_tensors(
            [
                unitcell_tensors[i].replace_tensor(
                    new_tensors[i],
                    reinitialize_env_as_identities=reinitialize_env_as_identities,
                )
                for i in range(unitcell.get_len_unique_tensors())
            ]
        )
    else:
        new_unitcell = None

    return new_tensors, new_unitcell


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


@jit
def _hager_zhang_initial_zero(input_tensors, gradient, config):
    input_tensors_real, _ = ravel_pytree(input_tensors)
    gradient_real, _ = ravel_pytree(gradient)

    if jnp.iscomplexobj(input_tensors_real):
        input_tensors_real = jnp.concatenate(
            (jnp.real(input_tensors_real), jnp.imag(input_tensors_real))
        )
        gradient_real = jnp.concatenate(
            (jnp.real(gradient_real), jnp.imag(gradient_real))
        )

    result = config.line_search_hager_zhang_psi_0
    result *= jnp.linalg.norm(input_tensors_real, ord=jnp.inf)
    result /= jnp.linalg.norm(gradient_real, ord=jnp.inf)

    result = jnp.where(result == 0, config.line_search_initial_step_size, result)

    return result


class _Hager_Zhang_Initial_State(enum.Enum):
    NOT_FOUND = enum.auto()
    FOUND = enum.auto()
    SCALAR_LOWER_VALUE_GREATER = enum.auto()


class _Hager_Zhang_State(enum.Enum):
    NONE = enum.auto()
    UPDATE = enum.auto()
    UPDATE_INNER = enum.auto()


@jit
def _hager_zhang_initial_quad_step_inner(
    old_value, new_value, gradient, descent_direction, alpha, fallback_alpha
):
    g_d_term = _scalar_descent_grad(descent_direction, gradient)

    sum_term = old_value + alpha * g_d_term
    sum_term -= new_value

    alpha = jnp.where(
        sum_term < 0, alpha**2 * g_d_term / (2 * sum_term), fallback_alpha
    )

    return alpha


def _hager_zhang_initial_quad_step(
    input_tensors,
    unitcell,
    gradient,
    descent_direction,
    old_alpha,
    old_value,
    spiral_indices,
    convert_to_unitcell_func,
    generate_unitcell,
    expectation_func,
    additional_input,
    reinitialize_env_as_identities,
    enforce_elementwise_convergence,
):
    alpha = varipeps_config.line_search_hager_zhang_psi_1 * old_alpha

    new_tensors = _line_search_new_tensors(input_tensors, descent_direction, alpha)
    new_tensors, new_unitcell = _get_new_unitcell(
        new_tensors,
        unitcell,
        spiral_indices,
        convert_to_unitcell_func,
        generate_unitcell,
        reinitialize_env_as_identities,
    )

    new_value, (new_unitcell, _) = calc_ctmrg_expectation(
        new_tensors,
        new_unitcell,
        expectation_func,
        convert_to_unitcell_func,
        additional_input,
        enforce_elementwise_convergence=enforce_elementwise_convergence,
    )

    fallback_alpha = varipeps_config.line_search_hager_zhang_psi_2 * old_alpha

    return jnp.where(
        new_value <= old_value,
        _hager_zhang_initial_quad_step_inner(
            old_value, new_value, gradient, descent_direction, alpha, fallback_alpha
        ),
        fallback_alpha,
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
    last_step_size: Optional[Union[float, jnp.ndarray]] = None,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None,
    generate_unitcell: bool = False,
    spiral_indices: Optional[Sequence[int]] = None,
    additional_input: Dict[str, jnp.ndarray] = {},
    reinitialize_env_as_identities: bool = True,
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
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~varipeps.expectation.Expectation_Model`):
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
      convert_to_unitcell_func (:obj:`~varipeps.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the input.
      generate_unitcell (:obj:`bool`):
        Force generation of unitcell from new tensors
      spiral_indices (:term:`sequence` of :obj:`int`):
        If spiral iPEPS ansatz is used, this argument contains the indices
        of the wave vectors in the input tensor list.
      additional_input (:obj:`dict` of :obj:`str` to :obj:`jax.numpy.ndarray` mapping):
        Dict with additional inputs which should be considered in the
        calculation of the expectation value.
      reinitialize_env_as_identities (:obj:`bool`):
        Flag if the env tensors should be reinitialized with identities.
    Returns:
      :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~varipeps.peps.PEPS_Unit_Cell`, :obj:`float`, :obj:`float`):
        Tuple with the optimized tensors, the new unitcell, the reduced
        expectation value and the step size found in the line search.
    Raises:
      :obj:`ValueError`: The parameters mismatch the expected inputs.
      :obj:`RuntimeError`: The line search does not converge.
    """

    has_been_increased = False
    incrementation_not_helped = False
    enforce_elementwise_convergence = (
        varipeps_config.ctmrg_enforce_elementwise_convergence
        or varipeps_config.ad_use_custom_vjp
    )

    if varipeps_config.line_search_method is Line_Search_Methods.HAGERZHANG:
        if last_step_size is None:
            alpha = _hager_zhang_initial_zero(input_tensors, gradient, varipeps_config)
        elif varipeps_config.line_search_hager_zhang_quad_step:
            try:
                alpha = _hager_zhang_initial_quad_step(
                    input_tensors,
                    unitcell,
                    gradient,
                    descent_direction,
                    last_step_size,
                    current_value,
                    spiral_indices,
                    convert_to_unitcell_func,
                    generate_unitcell,
                    expectation_func,
                    additional_input,
                    reinitialize_env_as_identities,
                    enforce_elementwise_convergence,
                )
            except CTMRGNotConvergedError:
                alpha = varipeps_config.line_search_hager_zhang_psi_2 * last_step_size
        else:
            alpha = varipeps_config.line_search_hager_zhang_psi_2 * last_step_size
    else:
        alpha = (
            last_step_size
            if last_step_size is not None
            and varipeps_config.line_search_use_last_step_size
            else varipeps_config.line_search_initial_step_size
        )

    wolfe_upper_bound = None
    wolfe_lower_bound = None
    wolfe_alpha_last_step = 0
    wolfe_descent_new_grad = _scalar_descent_grad(descent_direction, gradient)

    hager_zhang_lower_bound = 0
    hager_zhang_lower_bound_value = current_value
    hager_zhang_lower_bound_grad = gradient
    hager_zhang_lower_bound_des_grad = wolfe_descent_new_grad
    hager_zhang_upper_bound = alpha
    hager_zhang_upper_bound_value = None
    hager_zhang_upper_bound_grad = None
    hager_zhang_upper_bound_des_grad = None
    hager_zhang_alpha_last_step = 0
    hager_zhang_initial_found = _Hager_Zhang_Initial_State.NOT_FOUND
    hager_zhang_descent_grad = wolfe_descent_new_grad
    hager_zhang_state = _Hager_Zhang_State.NONE

    new_value = current_value

    tmp_value = None
    tmp_unitcell = None
    tmp_gradient = None
    tmp_descent_direction = None

    signal_reset_descent_dir = False

    cache_original_unitcell = {
        unitcell[0, 0][0][0].chi: (unitcell, gradient, descent_direction, current_value)
    }

    max_trunc_error = jnp.nan

    count = 0
    while count < varipeps_config.line_search_max_steps:
        new_tensors = _line_search_new_tensors(input_tensors, descent_direction, alpha)

        new_tensors, new_unitcell = _get_new_unitcell(
            new_tensors,
            unitcell,
            spiral_indices,
            convert_to_unitcell_func,
            generate_unitcell,
            reinitialize_env_as_identities,
        )

        if (
            varipeps_config.line_search_method is Line_Search_Methods.SIMPLE
            or varipeps_config.line_search_method is Line_Search_Methods.ARMIJO
        ):
            try:
                new_value, (new_unitcell, max_trunc_error) = calc_ctmrg_expectation(
                    new_tensors,
                    new_unitcell,
                    expectation_func,
                    convert_to_unitcell_func,
                    additional_input,
                    enforce_elementwise_convergence=enforce_elementwise_convergence,
                )

                if new_unitcell[0, 0][0][0].chi > unitcell[0, 0][0][0].chi:
                    tmp_value = current_value
                    tmp_unitcell = unitcell
                    tmp_gradient = gradient
                    tmp_descent_direction = descent_direction

                    if (
                        cache_original_unitcell.get(new_unitcell[0, 0][0][0].chi)
                        is not None
                    ):
                        (
                            unitcell,
                            gradient,
                            descent_direction,
                            current_value,
                        ) = cache_original_unitcell[new_unitcell[0, 0][0][0].chi]
                    else:
                        unitcell = unitcell.change_chi(new_unitcell[0, 0][0][0].chi)

                        debug_print(
                            "Line search: Recalculate original unitcell with higher chi {}.",
                            new_unitcell[0, 0][0][0].chi,
                        )

                        if varipeps_config.ad_use_custom_vjp:
                            (
                                current_value,
                                (unitcell, max_trunc_error),
                            ), tmp_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                                input_tensors,
                                unitcell,
                                expectation_func,
                                convert_to_unitcell_func,
                                additional_input,
                            )
                        else:
                            (
                                current_value,
                                (unitcell, max_trunc_error),
                            ), tmp_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                                input_tensors,
                                unitcell,
                                expectation_func,
                                convert_to_unitcell_func,
                                additional_input,
                                calc_preconverged=True,
                            )
                        gradient = [elem.conj() for elem in tmp_gradient_seq]
                        descent_direction = [-elem for elem in tmp_gradient]

                        cache_original_unitcell[new_unitcell[0, 0][0][0].chi] = (
                            unitcell,
                            gradient,
                            descent_direction,
                            current_value,
                        )

                    signal_reset_descent_dir = True
            except CTMRGNotConvergedError:
                new_value = jnp.inf
        elif (
            varipeps_config.line_search_method is Line_Search_Methods.WOLFE
            or varipeps_config.line_search_method is Line_Search_Methods.HAGERZHANG
        ):
            wolfe_value_last_step = new_value

            try:
                if varipeps_config.ad_use_custom_vjp:
                    (
                        new_value,
                        (new_unitcell, max_trunc_error),
                    ), new_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                        new_tensors,
                        new_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        additional_input,
                    )
                else:
                    (
                        new_value,
                        (new_unitcell, max_trunc_error),
                    ), new_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                        new_tensors,
                        new_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        additional_input,
                        calc_preconverged=True,
                    )
                new_gradient = [elem.conj() for elem in new_gradient_seq]

                if new_unitcell[0, 0][0][0].chi > unitcell[0, 0][0][0].chi:
                    tmp_value = current_value
                    tmp_unitcell = unitcell
                    tmp_gradient = gradient
                    tmp_descent_direction = descent_direction

                    if (
                        cache_original_unitcell.get(new_unitcell[0, 0][0][0].chi)
                        is not None
                    ):
                        (
                            unitcell,
                            gradient,
                            descent_direction,
                            current_value,
                        ) = cache_original_unitcell[new_unitcell[0, 0][0][0].chi]
                    else:
                        unitcell = unitcell.change_chi(new_unitcell[0, 0][0][0].chi)

                        debug_print(
                            "Line search: Recalculate original unitcell with higher chi {}.",
                            new_unitcell[0, 0][0][0].chi,
                        )

                        if varipeps_config.ad_use_custom_vjp:
                            (
                                current_value,
                                (unitcell, max_trunc_error),
                            ), tmp_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                                input_tensors,
                                unitcell,
                                expectation_func,
                                convert_to_unitcell_func,
                                additional_input,
                            )
                        else:
                            (
                                current_value,
                                (unitcell, max_trunc_error),
                            ), tmp_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                                input_tensors,
                                unitcell,
                                expectation_func,
                                convert_to_unitcell_func,
                                additional_input,
                                calc_preconverged=True,
                            )
                        gradient = [elem.conj() for elem in tmp_gradient_seq]
                        descent_direction = [-elem for elem in tmp_gradient]

                        cache_original_unitcell[new_unitcell[0, 0][0][0].chi] = (
                            unitcell,
                            gradient,
                            descent_direction,
                            current_value,
                        )

                    signal_reset_descent_dir = True
            except (CTMRGNotConvergedError, CTMRGGradientNotConvergedError):
                new_value = jnp.inf
                new_gradient = gradient
        else:
            raise ValueError("Unknown line search method.")

        if varipeps_config.line_search_method is Line_Search_Methods.HAGERZHANG:
            descent_new_grad = _scalar_descent_grad(descent_direction, new_gradient)

            hz_wolfe_1_left = (
                varipeps_config.line_search_hager_zhang_delta * hager_zhang_descent_grad
            )
            hz_wolfe_1_right = (new_value - current_value) / alpha

            hz_wolfe_2_right = (
                varipeps_config.line_search_hager_zhang_sigma * hager_zhang_descent_grad
            )

            if descent_new_grad >= hz_wolfe_2_right:
                if hz_wolfe_1_left >= hz_wolfe_1_right and new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    break

                hz_approx_wolfe_left = (
                    2 * varipeps_config.line_search_hager_zhang_delta - 1
                ) * hager_zhang_descent_grad

                if hz_approx_wolfe_left >= hager_zhang_descent_grad and new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    break

        if (
            varipeps_config.line_search_method is Line_Search_Methods.HAGERZHANG
            and hager_zhang_initial_found is not _Hager_Zhang_Initial_State.FOUND
        ):
            if (
                hager_zhang_initial_found
                is _Hager_Zhang_Initial_State.SCALAR_LOWER_VALUE_GREATER
            ):
                if descent_new_grad >= 0:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = descent_new_grad
                    hager_zhang_initial_found = _Hager_Zhang_Initial_State.FOUND
                elif new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    hager_zhang_lower_bound = alpha
                    hager_zhang_lower_bound_value = new_value
                    hager_zhang_lower_bound_grad = new_gradient
                    hager_zhang_lower_bound_des_grad = descent_new_grad
                    alpha = (
                        (1 - varipeps_config.line_search_hager_zhang_theta)
                        * hager_zhang_lower_bound
                        + varipeps_config.line_search_hager_zhang_theta
                        * hager_zhang_upper_bound
                    )

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False

                    count += 1
                    continue
                else:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = descent_new_grad
                    alpha = (
                        (1 - varipeps_config.line_search_hager_zhang_theta)
                        * hager_zhang_lower_bound
                        + varipeps_config.line_search_hager_zhang_theta
                        * hager_zhang_upper_bound
                    )

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue
            elif descent_new_grad >= 0:
                hager_zhang_upper_bound = alpha
                hager_zhang_upper_bound_value = new_value
                hager_zhang_upper_bound_grad = new_gradient
                hager_zhang_upper_bound_des_grad = descent_new_grad
                hager_zhang_initial_found = _Hager_Zhang_Initial_State.FOUND
            elif descent_new_grad < 0 and new_value > (
                current_value + varipeps_config.line_search_hager_zhang_eps
            ):
                alpha = varipeps_config.line_search_hager_zhang_theta * alpha
                hager_zhang_initial_found = (
                    _Hager_Zhang_Initial_State.SCALAR_LOWER_VALUE_GREATER
                )

                if tmp_value is not None:
                    current_value, unitcell, gradient, descent_direction = (
                        tmp_value,
                        tmp_unitcell,
                        tmp_gradient,
                        tmp_descent_direction,
                    )
                    tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                        None,
                        None,
                        None,
                        None,
                    )
                    signal_reset_descent_dir = False
                count += 1
                continue
            else:
                if new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    hager_zhang_lower_bound = alpha
                    hager_zhang_lower_bound_value = new_value
                    hager_zhang_lower_bound_grad = new_gradient
                    hager_zhang_lower_bound_des_grad = descent_new_grad

                alpha *= varipeps_config.line_search_hager_zhang_rho

                if tmp_value is not None:
                    current_value, unitcell, gradient, descent_direction = (
                        tmp_value,
                        tmp_unitcell,
                        tmp_gradient,
                        tmp_descent_direction,
                    )
                    tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                        None,
                        None,
                        None,
                        None,
                    )
                    signal_reset_descent_dir = False
                count += 1
                continue

        if varipeps_config.line_search_method is Line_Search_Methods.SIMPLE:
            smaller_value_found = (
                new_value <= current_value or (new_value - current_value) <= 1e-13
            )
        elif varipeps_config.line_search_method is Line_Search_Methods.ARMIJO:
            cmp_value = _armijo_value(
                current_value,
                descent_direction,
                gradient,
                alpha,
                varipeps_config.line_search_armijo_const,
            )
            smaller_value_found = (
                new_value <= cmp_value or (new_value - cmp_value) <= 1e-13
            )
        elif varipeps_config.line_search_method is Line_Search_Methods.WOLFE:
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
                varipeps_config.line_search_armijo_const,
                varipeps_config.line_search_wolfe_const,
            )
            wolfe_cond_1 = new_value <= cmp_value or (new_value - cmp_value) <= 1e-13
            wolfe_cond_2 = (
                wolfe_left_side <= wolfe_right_side
                or (wolfe_left_side - wolfe_right_side) <= 1e-13
            )

        if (
            varipeps_config.line_search_method is Line_Search_Methods.SIMPLE
            or varipeps_config.line_search_method is Line_Search_Methods.ARMIJO
        ):
            if smaller_value_found:
                if (
                    alpha >= varipeps_config.line_search_initial_step_size
                    or incrementation_not_helped
                ):
                    break

                has_been_increased = True
                alpha /= varipeps_config.line_search_reduction_factor
            else:
                if has_been_increased:
                    incrementation_not_helped = True

                alpha = varipeps_config.line_search_reduction_factor * alpha
        elif varipeps_config.line_search_method is Line_Search_Methods.WOLFE:
            if wolfe_upper_bound is None and wolfe_lower_bound is None:
                if jnp.isinf(new_value):
                    alpha /= varipeps_config.line_search_reduction_factor
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
                    alpha /= varipeps_config.line_search_reduction_factor
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
        elif varipeps_config.line_search_method is Line_Search_Methods.HAGERZHANG:
            if hager_zhang_state is _Hager_Zhang_State.UPDATE:
                if descent_new_grad >= 0:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = descent_new_grad
                    hager_zhang_state = _Hager_Zhang_State.NONE
                elif new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    hager_zhang_lower_bound = alpha
                    hager_zhang_lower_bound_value = new_value
                    hager_zhang_lower_bound_grad = new_gradient
                    hager_zhang_lower_bound_des_grad = descent_new_grad
                    hager_zhang_state = _Hager_Zhang_State.NONE
                else:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = _scalar_descent_grad(
                        descent_direction, new_gradient
                    )
                    alpha = (
                        (1 - varipeps_config.line_search_hager_zhang_theta)
                        * hager_zhang_lower_bound
                        + varipeps_config.line_search_hager_zhang_theta
                        * hager_zhang_upper_bound
                    )
                    hager_zhang_state = _Hager_Zhang_State.UPDATE_INNER

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue
            elif hager_zhang_state is _Hager_Zhang_State.UPDATE_INNER:
                if descent_new_grad >= 0:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = descent_new_grad
                    hager_zhang_state = _Hager_Zhang_State.NONE
                elif new_value <= (
                    current_value + varipeps_config.line_search_hager_zhang_eps
                ):
                    hager_zhang_lower_bound = alpha
                    hager_zhang_lower_bound_value = new_value
                    hager_zhang_lower_bound_grad = new_gradient
                    hager_zhang_lower_bound_des_grad = descent_new_grad
                    alpha = (
                        (1 - varipeps_config.line_search_hager_zhang_theta)
                        * hager_zhang_lower_bound
                        + varipeps_config.line_search_hager_zhang_theta
                        * hager_zhang_upper_bound
                    )

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue
                else:
                    hager_zhang_upper_bound = alpha
                    hager_zhang_upper_bound_value = new_value
                    hager_zhang_upper_bound_grad = new_gradient
                    hager_zhang_upper_bound_des_grad = _scalar_descent_grad(
                        descent_direction, new_gradient
                    )
                    alpha = (
                        (1 - varipeps_config.line_search_hager_zhang_theta)
                        * hager_zhang_lower_bound
                        + varipeps_config.line_search_hager_zhang_theta
                        * hager_zhang_upper_bound
                    )

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue
            else:
                alpha = hager_zhang_lower_bound * hager_zhang_upper_bound_des_grad
                alpha -= hager_zhang_upper_bound * hager_zhang_lower_bound_des_grad
                alpha /= (
                    hager_zhang_upper_bound_des_grad - hager_zhang_lower_bound_des_grad
                )

                hz_secant_alpha = alpha

                hz_secant_lower = hager_zhang_lower_bound
                hz_secant_lower_value = hager_zhang_lower_bound_value
                hz_secant_lower_grad = hager_zhang_lower_bound_grad
                hz_secant_lower_des_grad = hager_zhang_lower_bound_des_grad

                hz_secant_upper = hager_zhang_upper_bound
                hz_secant_upper_value = hager_zhang_upper_bound_value
                hz_secant_upper_grad = hager_zhang_upper_bound_grad
                hz_secant_upper_des_grad = hager_zhang_upper_bound_des_grad

                if hager_zhang_lower_bound < alpha < hager_zhang_upper_bound:
                    hager_zhang_state = _Hager_Zhang_State.UPDATE

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue

            if hz_secant_alpha is not None and (
                hz_secant_alpha == hager_zhang_lower_bound
                or hz_secant_alpha == hager_zhang_upper_bound
            ):
                if hz_secant_alpha == hager_zhang_lower_bound:
                    alpha = hz_secant_lower * hager_zhang_lower_bound_des_grad
                    alpha -= hager_zhang_lower_bound * hz_secant_lower_des_grad
                    alpha /= hager_zhang_lower_bound_des_grad - hz_secant_lower_des_grad
                else:
                    alpha = hz_secant_upper * hager_zhang_upper_bound_des_grad
                    alpha -= hager_zhang_upper_bound * hz_secant_upper_des_grad
                    alpha /= hager_zhang_upper_bound_des_grad - hz_secant_upper_des_grad

                hz_secant_alpha = None

                if hager_zhang_lower_bound < alpha < hager_zhang_upper_bound:
                    hager_zhang_state = _Hager_Zhang_State.UPDATE

                    if tmp_value is not None:
                        current_value, unitcell, gradient, descent_direction = (
                            tmp_value,
                            tmp_unitcell,
                            tmp_gradient,
                            tmp_descent_direction,
                        )
                        tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                            None,
                            None,
                            None,
                            None,
                        )
                        signal_reset_descent_dir = False
                    count += 1
                    continue
            hz_secant_alpha = None

            if hz_secant_lower is not None and (
                (hager_zhang_upper_bound - hager_zhang_lower_bound)
                > varipeps_config.line_search_hager_zhang_gamma
                * (hz_secant_upper - hz_secant_lower)
            ):
                alpha = (hager_zhang_lower_bound + hager_zhang_upper_bound) / 2
                hz_secant_lower = None
                hager_zhang_state = _Hager_Zhang_State.UPDATE

                if tmp_value is not None:
                    current_value, unitcell, gradient, descent_direction = (
                        tmp_value,
                        tmp_unitcell,
                        tmp_gradient,
                        tmp_descent_direction,
                    )
                    tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                        None,
                        None,
                        None,
                        None,
                    )
                    signal_reset_descent_dir = False
                count += 1
                continue
            hz_secant_lower = None

        if tmp_value is not None:
            current_value, unitcell, gradient, descent_direction = (
                tmp_value,
                tmp_unitcell,
                tmp_gradient,
                tmp_descent_direction,
            )
            tmp_value, tmp_unitcell, tmp_gradient, tmp_descent_direction = (
                None,
                None,
                None,
                None,
            )
            signal_reset_descent_dir = False

        count += 1

    if count == varipeps_config.line_search_max_steps:
        raise NoSuitableStepSizeError(f"Count {count}, Last alpha {alpha}")

    return (
        new_tensors,
        new_unitcell,
        new_value,
        alpha,
        signal_reset_descent_dir,
        max_trunc_error,
    )
