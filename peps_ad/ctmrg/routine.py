from functools import partial
import enum

import jax.numpy as jnp
from jax import jit, custom_vjp, vjp, tree_util
import jax.util
from jax.lax import cond, while_loop
import jax.debug as jdebug

from peps_ad import peps_ad_config, peps_ad_global_state
from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.utils.debug_print import debug_print
from .absorption import do_absorption_step

from typing import Sequence, Tuple, List, Optional


@enum.unique
class CTM_Enum(enum.IntEnum):
    C1 = enum.auto()
    C2 = enum.auto()
    C3 = enum.auto()
    C4 = enum.auto()
    T1 = enum.auto()
    T2 = enum.auto()
    T3 = enum.auto()
    T4 = enum.auto()


class CTMRGNotConvergedError(Exception):
    """
    Exception if the CTM routine does not converge.
    """

    pass


class CTMRGGradientNotConvergedError(Exception):
    """
    Exception if the custom rule for the gradient of the the CTM routine does
    not converge.
    """

    pass


@partial(jit, static_argnums=(2,), inline=True)
def _calc_corner_svds(
    peps_tensors: List[PEPS_Tensor],
    old_corner_svd: jnp.ndarray,
    tensor_shape: Optional[Tuple[int, int, int]],
) -> jnp.ndarray:
    if tensor_shape is None:
        step_corner_svd = jnp.zeros_like(old_corner_svd)
    else:
        step_corner_svd = jnp.zeros(tensor_shape, dtype=jnp.float64)

    for ti, t in enumerate(peps_tensors):
        C1_svd = jnp.linalg.svd(t.C1, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 0, : C1_svd.shape[0]].set(
            C1_svd, indices_are_sorted=True, unique_indices=True
        )

        C2_svd = jnp.linalg.svd(t.C2, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 1, : C2_svd.shape[0]].set(
            C2_svd, indices_are_sorted=True, unique_indices=True
        )

        C3_svd = jnp.linalg.svd(t.C3, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 2, : C3_svd.shape[0]].set(
            C3_svd, indices_are_sorted=True, unique_indices=True
        )

        C4_svd = jnp.linalg.svd(t.C4, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 3, : C4_svd.shape[0]].set(
            C4_svd, indices_are_sorted=True, unique_indices=True
        )

    return step_corner_svd


@partial(jit, static_argnums=(3,), inline=True)
def _is_element_wise_converged(
    old_peps_tensors: List[PEPS_Tensor],
    new_peps_tensors: List[PEPS_Tensor],
    eps: float,
    verbose: bool = False,
) -> Tuple[bool, float, Optional[List[Tuple[int, CTM_Enum, float]]]]:
    result = 0
    measure = jnp.zeros((len(old_peps_tensors), 8), dtype=jnp.float64)

    verbose_data = [] if verbose else None

    for ti in range(len(old_peps_tensors)):
        old_shape = old_peps_tensors[ti].C1.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C1[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C1
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 0].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C1, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C2.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C2[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C2
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 1].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C2, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C3.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C3[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C3
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 2].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C3, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C4.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C4[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C4
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 3].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C4, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].T1.shape
        diff = jnp.abs(
            new_peps_tensors[ti].T1[
                : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
            ]
            - old_peps_tensors[ti].T1
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 4].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.T1, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].T2.shape
        diff = jnp.abs(
            new_peps_tensors[ti].T2[
                : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
            ]
            - old_peps_tensors[ti].T2
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 5].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.T2, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].T3.shape
        diff = jnp.abs(
            new_peps_tensors[ti].T3[
                : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
            ]
            - old_peps_tensors[ti].T3
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 6].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.T3, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].T4.shape
        diff = jnp.abs(
            new_peps_tensors[ti].T4[
                : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
            ]
            - old_peps_tensors[ti].T4
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 7].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.T4, jnp.amax(diff)))

    return result == 0, jnp.linalg.norm(measure), verbose_data


@jit
def _ctmrg_body_func(carry):
    (
        w_tensors,
        w_unitcell_last_step,
        converged,
        last_corner_svd,
        eps,
        count,
        elementwise_conv,
        state,
        config,
    ) = carry

    w_unitcell = do_absorption_step(w_tensors, w_unitcell_last_step, config, state)

    def elementwise_func(old, new, old_corner, conv_eps, config):
        converged, measure, verbose_data = _is_element_wise_converged(
            old,
            new,
            conv_eps,
            verbose=config.ctmrg_verbose_output,
        )
        return converged, measure, verbose_data, old_corner

    def corner_svd_func(old, new, old_corner, conv_eps, config):
        if old_corner is None:
            return (
                False,
                jnp.nan,
                [] if config.ctmrg_verbose_output else None,
                old_corner,
            )
        corner_svd = _calc_corner_svds(new, old_corner, None)
        measure = jnp.linalg.norm(corner_svd - old_corner)
        converged = measure < conv_eps
        return (
            converged,
            measure,
            [] if config.ctmrg_verbose_output else None,
            corner_svd,
        )

    converged, measure, verbose_data, corner_svd = cond(
        elementwise_conv,
        elementwise_func,
        corner_svd_func,
        w_unitcell_last_step.get_unique_tensors(),
        w_unitcell.get_unique_tensors(),
        last_corner_svd,
        eps,
        config,
    )

    if config.ctmrg_print_steps:
        debug_print("CTMRG: {}: {}", count, measure)
        if config.ctmrg_verbose_output:
            for ti, ctm_enum_i, diff in verbose_data:
                debug_print(
                    "CTMRG: Verbose: ti {}, CTM tensor {}, Diff {}",
                    ti,
                    CTM_Enum(ctm_enum_i).name,
                    diff,
                )

    count += 1

    return (
        w_tensors,
        w_unitcell,
        converged,
        corner_svd,
        eps,
        count,
        elementwise_conv,
        state,
        config,
    )


@jit
def _ctmrg_while_wrapper(start_carry):
    def cond_func(carry):
        _, _, converged, _, _, count, _, _, config = carry
        return jnp.logical_not(converged) & (count < config.ctmrg_max_steps)

    _, working_unitcell, converged, _, _, end_count, _, _, _ = while_loop(
        cond_func, _ctmrg_body_func, start_carry
    )

    return working_unitcell, converged, end_count


def calc_ctmrg_env(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    *,
    eps: Optional[float] = None,
    enforce_elementwise_convergence: Optional[bool] = None,
    _return_truncation_eps: bool = False,
) -> PEPS_Unit_Cell:
    """
    Calculate the new converged CTMRG tensors for the unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Keyword args:
      eps (:obj:`float`):
        The convergence criterion.
      enforce_elementwise_convergence (obj:`bool`):
        Enforce elementwise convergence of the CTM tensors instead of only
        convergence of the singular values of the corners.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with all updated converged CTMRG tensors of
        all elements of the unitcell.
    """
    eps = eps if eps is not None else peps_ad_config.ctmrg_convergence_eps
    enforce_elementwise_convergence = (
        enforce_elementwise_convergence
        if enforce_elementwise_convergence is not None
        else peps_ad_config.ctmrg_enforce_elementwise_convergence
    )
    init_corner_singular_vals = None

    if enforce_elementwise_convergence:
        last_step_tensors = unitcell.get_unique_tensors()
    else:
        shape_corner_svd = (
            unitcell.get_len_unique_tensors(),
            4,
            unitcell[0, 0][0][0].chi,
        )
        init_corner_singular_vals = _calc_corner_svds(
            unitcell.get_unique_tensors(), None, shape_corner_svd
        )

    working_unitcell = unitcell
    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    while True:
        tmp_count = 0
        corner_singular_vals = None

        while any(
            i.C1.shape[0] != i.chi for i in working_unitcell.get_unique_tensors()
        ):
            (
                _,
                working_unitcell,
                _,
                corner_singular_vals,
                _,
                tmp_count,
                _,
                _,
                _,
            ) = _ctmrg_body_func(
                (
                    peps_tensors,
                    working_unitcell,
                    False,
                    init_corner_singular_vals,
                    eps,
                    0,
                    enforce_elementwise_convergence,
                    peps_ad_global_state,
                    peps_ad_config,
                )
            )

        working_unitcell, converged, end_count = _ctmrg_while_wrapper(
            (
                peps_tensors,
                working_unitcell,
                False,
                corner_singular_vals
                if corner_singular_vals is not None
                else init_corner_singular_vals,
                eps,
                tmp_count,
                enforce_elementwise_convergence,
                peps_ad_global_state,
                peps_ad_config,
            )
        )

        if (
            peps_ad_config.ctmrg_increase_truncation_eps
            and end_count == peps_ad_config.ctmrg_max_steps
            and not converged
        ):
            old_truncation_eps = (
                peps_ad_config.ctmrg_truncation_eps
                if peps_ad_global_state.ctmrg_effective_truncation_eps is None
                else peps_ad_global_state.ctmrg_effective_truncation_eps
            )
            new_truncation_eps = (
                old_truncation_eps * peps_ad_config.ctmrg_increase_truncation_eps_factor
            )
            if (
                new_truncation_eps
                <= peps_ad_config.ctmrg_increase_truncation_eps_max_value
            ):
                debug_print(
                    "CTMRG: Increasing SVD truncation eps to {}.",
                    new_truncation_eps,
                )
                peps_ad_global_state.ctmrg_effective_truncation_eps = new_truncation_eps
                working_unitcell = unitcell
                continue

        break

    if _return_truncation_eps:
        last_truncation_eps = peps_ad_global_state.ctmrg_effective_truncation_eps
    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    if (
        peps_ad_config.ctmrg_fail_if_not_converged
        and end_count == peps_ad_config.ctmrg_max_steps
        and not converged
    ):
        raise CTMRGNotConvergedError

    if _return_truncation_eps:
        return working_unitcell, last_truncation_eps

    return working_unitcell


@custom_vjp
def calc_ctmrg_env_custom_rule(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    _return_truncation_eps: bool = False,
) -> PEPS_Unit_Cell:
    """
    Wrapper function of :obj:`~peps_ad.ctmrg.routine.calc_ctmrg_env` which
    enables the use of the custom VJP for the calculation of the gradient.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with all updated converged CTMRG tensors of
        all elements of the unitcell.
    """
    return calc_ctmrg_env(
        peps_tensors,
        unitcell,
        enforce_elementwise_convergence=True,
        _return_truncation_eps=_return_truncation_eps,
    )


def calc_ctmrg_env_fwd(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    _return_truncation_eps: bool = False,
) -> Tuple[PEPS_Unit_Cell, Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell]]:
    """
    Internal helper function of custom VJP to calculate the values in
    the forward sweep.
    """
    new_unitcell, last_truncation_eps = calc_ctmrg_env_custom_rule(
        peps_tensors, unitcell, _return_truncation_eps=True
    )
    return new_unitcell, (peps_tensors, new_unitcell, last_truncation_eps)


def _ctmrg_rev_while_body(carry):
    (
        vjp_env,
        initial_bar,
        bar_fixed_point_last_step,
        converged,
        count,
        config,
        state,
    ) = carry

    new_env_bar = vjp_env(bar_fixed_point_last_step)[0]

    bar_fixed_point = bar_fixed_point_last_step.replace_unique_tensors(
        [
            t_old.__add__(t_new, checks=False)
            for t_old, t_new in jax.util.safe_zip(
                initial_bar.get_unique_tensors(), new_env_bar.get_unique_tensors()
            )
        ]
    )

    converged, measure, verbose_data = _is_element_wise_converged(
        bar_fixed_point_last_step.get_unique_tensors(),
        bar_fixed_point.get_unique_tensors(),
        config.ad_custom_convergence_eps,
        verbose=config.ad_custom_verbose_output,
    )

    count += 1

    if config.ad_custom_print_steps:
        debug_print("Custom VJP: {}: {}", count, measure)
        if config.ad_custom_verbose_output:
            for ti, ctm_enum_i, diff in verbose_data:
                debug_print(
                    "Custom VJP: Verbose: ti {}, CTM tensor {}, Diff {}",
                    ti,
                    CTM_Enum(ctm_enum_i).name,
                    diff,
                )

    return vjp_env, initial_bar, bar_fixed_point, converged, count, config, state


@jit
def _ctmrg_rev_workhorse(peps_tensors, new_unitcell, new_unitcell_bar, config, state):
    _, vjp_peps_tensors = vjp(
        lambda t: do_absorption_step(t, new_unitcell, config, state), peps_tensors
    )

    vjp_env = tree_util.Partial(
        vjp(lambda u: do_absorption_step(peps_tensors, u, config, state), new_unitcell)[
            1
        ]
    )

    def cond_func(carry):
        _, _, _, converged, count, config, state = carry

        return jnp.logical_not(converged) & (count < config.ad_custom_max_steps)

    _, _, env_fixed_point, converged, end_count, _, _ = while_loop(
        cond_func,
        _ctmrg_rev_while_body,
        (vjp_env, new_unitcell_bar, new_unitcell_bar, False, 0, config, state),
    )

    (t_bar,) = vjp_peps_tensors(env_fixed_point)

    return t_bar, converged, end_count


def calc_ctmrg_env_rev(
    res: Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell], unitcell_bar: PEPS_Unit_Cell
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell]:
    """
    Internal helper function of custom VJP to calculate the gradient in
    the backward sweep.
    """
    peps_tensors, new_unitcell, last_truncation_eps = res

    peps_ad_global_state.ctmrg_effective_truncation_eps = last_truncation_eps

    t_bar, converged, end_count = _ctmrg_rev_workhorse(
        peps_tensors, new_unitcell, unitcell_bar, peps_ad_config, peps_ad_global_state
    )

    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    if end_count == peps_ad_config.ad_custom_max_steps and not converged:
        raise CTMRGGradientNotConvergedError

    empty_t = [t.zeros_like_self() for t in new_unitcell.get_unique_tensors()]

    return (
        t_bar,
        new_unitcell.replace_unique_tensors(empty_t),
        jnp.zeros((), dtype=bool),
    )


calc_ctmrg_env_custom_rule.defvjp(calc_ctmrg_env_fwd, calc_ctmrg_env_rev)
