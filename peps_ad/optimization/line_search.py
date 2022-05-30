import jax.numpy as jnp
from jax import jit

from peps_ad import peps_ad_config
from peps_ad.config import Line_Search_Methods
from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model

from .inner_function import calc_ctmrg_expectation

from typing import Sequence, Tuple, List, Union


@jit
def _line_search_new_tensors(peps_tensors, descent_dir, alpha):
    return [peps_tensors[i] + alpha * descent_dir[i] for i in range(len(peps_tensors))]


@jit
def _armijo_value(current_val, descent_dir, gradient, alpha, const_factor):
    descent_dir_real = jnp.concatenate((jnp.real(descent_dir), jnp.imag(descent_dir)))
    gradient_real = jnp.concatenate((jnp.real(gradient), jnp.imag(gradient)))
    return jnp.fmin(
        current_val,
        current_val + const_factor * alpha * jnp.sum(descent_dir_real * gradient_real),
    )


class NoSuitableStepSizeError(Exception):
    pass


def line_search(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    gradient: jnp.ndarray,
    descent_direction: jnp.ndarray,
    current_value: Union[float, jnp.ndarray],
    last_step_size: Union[float, jnp.ndarray],
) -> Tuple[
    List[jnp.ndarray],
    PEPS_Unit_Cell,
    Union[float, jnp.ndarray],
    Union[float, jnp.ndarray],
]:
    """
    Run two-way backtracing line search method for the CTMRG routine.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the current peps tensors which should be optimized.
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
    Returns:
      :obj:`tuple`\ (:obj:`list`\ (:obj:`jax.numpy.ndarray`), :obj:`~peps_ad.peps.PEPS_Unit_Cell`, :obj:`float`, :obj:`float`):
        Tuple with the optimized tensors, the new unitcell, the reduced
        expectation value and the step size found in the line search.
    Raises:
      :obj:`ValueError`: The parameters mismatch the expected inputs.
      :obj:`RuntimeError`: The line search does not converge.
    """

    unitcell_tensors = unitcell.get_unique_tensors()
    if len(peps_tensors) != len(unitcell_tensors) or not all(
        peps_tensors[i] is unitcell_tensors[i].tensor for i in range(len(peps_tensors))
    ):
        raise ValueError("PEPS tensor sequence mismatches the unitcell.")

    alpha = (
        last_step_size
        if last_step_size is not None
        else peps_ad_config.line_search_initial_step_size
    )
    has_been_increased = False
    incrementation_not_helped = False
    enforce_elementwise_convergence = (
        peps_ad_config.ctmrg_enforce_elementwise_convergence
        or peps_ad_config.ad_use_custom_vjp
    )

    count = 0
    while count < peps_ad_config.line_search_max_steps:
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

        if (
            peps_ad_config.line_search_method is Line_Search_Methods.SIMPLE
            or peps_ad_config.line_search_method is Line_Search_Methods.ARMIJO
        ):
            new_value, new_unitcell = calc_ctmrg_expectation(
                new_tensors,
                new_unitcell,
                expectation_func,
                enforce_elementwise_convergence=enforce_elementwise_convergence,
            )
        elif peps_ad_config.line_search_method is Line_Search_Methods.WOLFE:
            raise NotImplementedError("Wolfe condition not implemented yet.")
        else:
            raise ValueError("Unknown line search method.")

        if peps_ad_config.line_search_method is Line_Search_Methods.SIMPLE:
            smaller_value_found = new_value < current_value
        elif peps_ad_config.line_search_method is Line_Search_Methods.ARMIJO:
            cmp_value = _armijo_value(
                current_value,
                descent_direction,
                gradient,
                alpha,
                peps_ad_config.line_search_armijo_const,
            )
            smaller_value_found = new_value < cmp_value
        elif peps_ad_config.line_search_method is Line_Search_Methods.WOLFE:
            pass

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

        count += 1

    if count == peps_ad_config.line_search_max_steps:
        raise NoSuitableStepSizeError

    return new_tensors, new_unitcell, new_value, alpha
