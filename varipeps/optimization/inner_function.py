import jax.numpy as jnp
from jax import value_and_grad
from jax.util import safe_zip

from varipeps import varipeps_config
from varipeps.peps import PEPS_Unit_Cell
from varipeps.expectation import Expectation_Model
from varipeps.ctmrg import calc_ctmrg_env, calc_ctmrg_env_custom_rule
from varipeps.mapping import Map_To_PEPS_Model

from typing import Sequence, Tuple, cast, Optional, Callable, Dict


def _map_tensors(
    input_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model],
    is_spiral_peps: bool = False,
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell]:
    if convert_to_unitcell_func is not None:
        if unitcell is None:
            if is_spiral_peps:
                peps_tensors, unitcell, spiral_vectors = convert_to_unitcell_func(
                    input_tensors, generate_unitcell=True
                )
            else:
                peps_tensors, unitcell = convert_to_unitcell_func(
                    input_tensors, generate_unitcell=True
                )
        else:
            if is_spiral_peps:
                peps_tensors, spiral_vectors = convert_to_unitcell_func(
                    input_tensors, generate_unitcell=False
                )
            else:
                peps_tensors = convert_to_unitcell_func(
                    input_tensors, generate_unitcell=False
                )
            old_tensors = unitcell.get_unique_tensors()
            if not all(
                jnp.allclose(ti, tj_obj.tensor)
                for ti, tj_obj in safe_zip(peps_tensors, old_tensors)
            ):
                raise ValueError(
                    "Input tensors and provided unitcell are not the same state."
                )
            unitcell = unitcell.replace_unique_tensors(
                [
                    old_tensors[i].replace_tensor(
                        peps_tensors[i], reinitialize_env_as_identities=False
                    )
                    for i in range(len(peps_tensors))
                ]
            )
    else:
        peps_tensors = input_tensors

    if is_spiral_peps:
        return peps_tensors, unitcell, spiral_vectors
    return peps_tensors, unitcell


def calc_ctmrg_expectation(
    input_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model],
    additional_input: Dict[str, jnp.ndarray] = dict(),
    *,
    enforce_elementwise_convergence: Optional[bool] = None,
) -> Tuple[jnp.ndarray, PEPS_Unit_Cell]:
    """
    Calculate the CTMRG environment and the (energy) expectation value for a
    iPEPS unitcell.

    Args:
      input_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~varipeps.expectation.Expectation_Model`):
        Callable to calculate the expectation value.
      convert_to_unitcell_func (:obj:`~varipeps.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the input.
      additional_input (:obj:`dict` of :obj:`str` to :obj:`jax.numpy.ndarray` mapping):
        Optional dict with additional inputs which should be considered in the
        calculation of the expectation value.
    Keyword args:
      enforce_elementwise_convergence (obj:`bool`):
        Enforce elementwise convergence of the CTM tensors instead of only
        convergence of the singular values of the corners.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`~varipeps.peps.PEPS_Unit_Cell`):
        Tuple consisting of the calculated expectation value and the new unitcell.
    """
    state_split_transfer = unitcell.is_split_transfer()

    spiral_vectors = additional_input.get("spiral_vectors")
    if expectation_func.is_spiral_peps and spiral_vectors is None:
        peps_tensors, unitcell, spiral_vectors = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, True
        )

        if any(i.size == 1 for i in spiral_vectors):
            spiral_vectors_x = additional_input.get("spiral_vectors_x")
            spiral_vectors_y = additional_input.get("spiral_vectors_y")
            if spiral_vectors_x is not None:
                if isinstance(spiral_vectors_x, jnp.ndarray):
                    spiral_vectors_x = (spiral_vectors_x,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors_x, spiral_vectors)
                )
            elif spiral_vectors_y is not None:
                if isinstance(spiral_vectors_y, jnp.ndarray):
                    spiral_vectors_y = (spiral_vectors_y,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors, spiral_vectors_y)
                )
    else:
        peps_tensors, unitcell = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, False
        )

    if state_split_transfer != unitcell.is_split_transfer():
        raise ValueError("Map function is not split transfer aware. Please fix that!")

    new_unitcell, max_trunc_error = calc_ctmrg_env(
        peps_tensors,
        unitcell,
        enforce_elementwise_convergence=enforce_elementwise_convergence,
    )

    exp_unitcell = new_unitcell.convert_to_full_transfer()

    if expectation_func.is_spiral_peps:
        return cast(
            jnp.ndarray, expectation_func(peps_tensors, exp_unitcell, spiral_vectors)
        ), (
            new_unitcell,
            max_trunc_error,
        )
    return cast(jnp.ndarray, expectation_func(peps_tensors, exp_unitcell)), (
        new_unitcell,
        max_trunc_error,
    )


calc_ctmrg_expectation_value_and_grad = value_and_grad(
    calc_ctmrg_expectation, has_aux=True
)


def calc_preconverged_ctmrg_value_and_grad(
    input_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model],
    additional_input: Dict[str, jnp.ndarray] = dict(),
    *,
    calc_preconverged: bool = True,
) -> Tuple[Tuple[jnp.ndarray, PEPS_Unit_Cell], Sequence[jnp.ndarray]]:
    """
    Calculate the CTMRG environment and the (energy) expectation value as well
    as the gradient of this steps for a iPEPS unitcell.

    To reduce the memory footprint of the automatic differentiation this
    function first calculates only the CTMRG env without the gradient for a
    less strict convergence and then calculates the gradient for the remaining
    CTMRG steps.

    Args:
      input_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~varipeps.expectation.Expectation_Model`):
        Callable to calculate the expectation value.
      convert_to_unitcell_func (:obj:`~varipeps.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the input.
      additional_input (:obj:`dict` of :obj:`str` to :obj:`jax.numpy.ndarray` mapping):
        Optional dict with additional inputs which should be considered in the
        calculation of the expectation value.
    Keyword args:
      calc_preconverged (:obj:`bool`):
        Flag if the above described procedure to calculate a pre-converged
        environment should be used.
    Returns:
      :obj:`tuple`\ (:obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`~varipeps.peps.PEPS_Unit_Cell`), :obj:`tuple`\ (:obj:`jax.numpy.ndarray`)):
        Tuple with two element:
        1. Tuple consisting of the calculated expectation value and the new
        unitcell.
        2. The calculated gradient.
    """
    state_split_transfer = unitcell.is_split_transfer()

    spiral_vectors = additional_input.get("spiral_vectors")
    if expectation_func.is_spiral_peps and spiral_vectors is None:
        peps_tensors, unitcell, spiral_vectors = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, True
        )

        if any(i.size == 1 for i in spiral_vectors):
            spiral_vectors_x = additional_input.get("spiral_vectors_x")
            spiral_vectors_y = additional_input.get("spiral_vectors_y")
            if spiral_vectors_x is not None:
                if isinstance(spiral_vectors_x, jnp.ndarray):
                    spiral_vectors_x = (spiral_vectors_x,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors_x, spiral_vectors)
                )
            elif spiral_vectors_y is not None:
                if isinstance(spiral_vectors_y, jnp.ndarray):
                    spiral_vectors_y = (spiral_vectors_y,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors, spiral_vectors_y)
                )
    else:
        peps_tensors, unitcell = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, False
        )

    if state_split_transfer != unitcell.is_split_transfer():
        raise ValueError("Map function is not split transfer aware. Please fix that!")

    if calc_preconverged:
        preconverged_unitcell, _ = calc_ctmrg_env(
            peps_tensors,
            unitcell,
            eps=varipeps_config.optimizer_ctmrg_preconverged_eps,
        )
    else:
        preconverged_unitcell = unitcell

    (
        expectation_value,
        (final_unitcell, max_trunc_error),
    ), gradient = calc_ctmrg_expectation_value_and_grad(
        peps_tensors,
        preconverged_unitcell,
        expectation_func,
    )

    return (expectation_value, final_unitcell, max_trunc_error), gradient


def calc_ctmrg_expectation_custom(
    input_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model],
    additional_input: Dict[str, jnp.ndarray] = dict(),
) -> Tuple[jnp.ndarray, PEPS_Unit_Cell]:
    """
    Calculate the CTMRG environment and the (energy) expectation value for a
    iPEPS unitcell using the custom VJP rule implementation.

    Args:
      input_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The PEPS unitcell to work on.
      expectation_func (:obj:`~varipeps.expectation.Expectation_Model`):
        Callable to calculate the expectation value.
      convert_to_unitcell_func (:obj:`~varipeps.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the input.
      additional_input (:obj:`dict` of :obj:`str` to :obj:`jax.numpy.ndarray` mapping):
        Dict with additional inputs which should be considered in the
        calculation of the expectation value.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`~varipeps.peps.PEPS_Unit_Cell`):
        Tuple consisting of the calculated expectation value and the new unitcell.
    """
    state_split_transfer = unitcell.is_split_transfer()

    spiral_vectors = additional_input.get("spiral_vectors")
    if expectation_func.is_spiral_peps and spiral_vectors is None:
        peps_tensors, unitcell, spiral_vectors = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, True
        )

        if any(i.size == 1 for i in spiral_vectors):
            spiral_vectors_x = additional_input.get("spiral_vectors_x")
            spiral_vectors_y = additional_input.get("spiral_vectors_y")
            if spiral_vectors_x is not None:
                if isinstance(spiral_vectors_x, jnp.ndarray):
                    spiral_vectors_x = (spiral_vectors_x,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors_x, spiral_vectors)
                )
            elif spiral_vectors_y is not None:
                if isinstance(spiral_vectors_y, jnp.ndarray):
                    spiral_vectors_y = (spiral_vectors_y,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors, spiral_vectors_y)
                )
    else:
        peps_tensors, unitcell = _map_tensors(
            input_tensors, unitcell, convert_to_unitcell_func, False
        )

    if state_split_transfer != unitcell.is_split_transfer():
        raise ValueError("Map function is not split transfer aware. Please fix that!")

    new_unitcell, max_trunc_error = calc_ctmrg_env_custom_rule(peps_tensors, unitcell)

    exp_unitcell = new_unitcell.convert_to_full_transfer()

    if expectation_func.is_spiral_peps:
        return cast(
            jnp.ndarray, expectation_func(peps_tensors, exp_unitcell, spiral_vectors)
        ), (
            new_unitcell,
            max_trunc_error,
        )
    return cast(jnp.ndarray, expectation_func(peps_tensors, exp_unitcell)), (
        new_unitcell,
        max_trunc_error,
    )


calc_ctmrg_expectation_custom_value_and_grad = value_and_grad(
    calc_ctmrg_expectation_custom, has_aux=True
)
