import jax.numpy as jnp
from jax import value_and_grad

from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model
from peps_ad.ctmrg import calc_ctmrg_env, calc_ctmrg_env_custom_rule

from typing import Sequence, Tuple, cast


def calc_ctmrg_expectation(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    *,
    eps: float = 1e-8,
    max_steps: int = 200,
    enforce_elementwise_convergence: bool = True,
) -> Tuple[jnp.ndarray, PEPS_Unit_Cell]:
    """
    Calculate the CTMRG environment and the (energy) expectation value for a
    iPEPS unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate the expectation value.
    Keyword args:
      eps (:obj:`float`):
        The convergence criterion for the CTMRG routine.
      max_steps (:obj:`int`):
        Maximal number of CTMRG steps.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        Tuple consisting of the calculated expectation value and the new unitcell.
    """
    new_unitcell = calc_ctmrg_env(
        peps_tensors,
        unitcell,
        eps=eps,
        max_steps=max_steps,
        enforce_elementwise_convergence=enforce_elementwise_convergence,
    )

    return cast(jnp.ndarray, expectation_func(peps_tensors, new_unitcell)), new_unitcell


calc_ctmrg_expectation_value_and_grad = value_and_grad(
    calc_ctmrg_expectation, has_aux=True
)


def calc_preconverged_ctmrg_value_and_grad(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    *,
    calc_preconverged: bool = True,
    preconverged_eps: float = 1e-5,
    final_eps: float = 1e-8,
    preconverged_max_steps: int = 100,
    final_max_steps: int = 10,
) -> Tuple[Tuple[jnp.ndarray, PEPS_Unit_Cell], Sequence[jnp.ndarray]]:
    """
    Calculate the CTMRG environment and the (energy) expectation value as well
    as the gradient of this steps for a iPEPS unitcell.

    To reduce the memory footprint of the automatic differentiation this
    function first calculates only the CTMRG env without the gradient for a
    less strict convergence and then calculates the gradient for the remaining
    CTMRG steps.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate the expectation value.
    Keyword args:
      calc_preconverged (:obj:`bool`):
        Flag if the above described procedure to calculate a pre-converged
        environment should be used.
      preconverged_eps (:obj:`float`):
        The convergence criterion for the CTMRG routine used in the pre-converged
        procedure.
      final_eps (:obj:`float`):
        The convergence criterion for the CTMRG routine used in the calculation
        of gradient.
      preconverged_max_steps (:obj:`int`):
        Maximal number of CTMRG steps used in the pre-converged
        procedure.
      final_max_steps (:obj:`int`):
        Maximal number of CTMRG steps used in the calculation
        of gradient.
    Returns:
      :obj:`tuple`\ (:obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`~peps_ad.peps.PEPS_Unit_Cell`), :obj:`tuple`\ (:obj:`jax.numpy.ndarray`)):
        Tuple with two element:
        1. Tuple consisting of the calculated expectation value and the new
        unitcell.
        2. The calculated gradient.
    """
    if calc_preconverged:
        preconverged_unitcell = calc_ctmrg_env(
            peps_tensors,
            unitcell,
            eps=preconverged_eps,
            max_steps=preconverged_max_steps,
        )
    else:
        preconverged_unitcell = unitcell

    (
        expectation_value,
        final_unitcell,
    ), gradient = calc_ctmrg_expectation_value_and_grad(
        peps_tensors,
        preconverged_unitcell,
        expectation_func,
        eps=final_eps,
        max_steps=final_max_steps,
    )

    return (expectation_value, final_unitcell), gradient


def calc_ctmrg_expectation_custom(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
) -> Tuple[jnp.ndarray, PEPS_Unit_Cell]:
    new_unitcell = calc_ctmrg_env_custom_rule(peps_tensors, unitcell)

    return cast(jnp.ndarray, expectation_func(peps_tensors, new_unitcell)), new_unitcell


calc_ctmrg_expectation_custom_value_and_grad = value_and_grad(
    calc_ctmrg_expectation_custom, has_aux=True
)
