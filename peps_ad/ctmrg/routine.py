from functools import partial
import enum

import jax.numpy as jnp
from jax import jit, custom_vjp, vjp
import jax.util

from peps_ad import peps_ad_config, peps_ad_global_state
from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
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


@partial(jit, static_argnums=(1,))
def _calc_corner_svds(
    peps_tensors: List[PEPS_Tensor], tensor_shape: Tuple[int, int, int]
) -> jnp.ndarray:
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


@partial(jit, static_argnums=(3,))
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

    if enforce_elementwise_convergence:
        last_step_tensors = unitcell.get_unique_tensors()
    else:
        shape_corner_svd = (
            unitcell.get_len_unique_tensors(),
            4,
            unitcell[0, 0][0][0].chi,
        )
        corner_singular_values = [
            _calc_corner_svds(unitcell.get_unique_tensors(), shape_corner_svd)
        ]

    working_unitcell = unitcell
    count = 0
    converged = False
    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    while not converged and count < peps_ad_config.ctmrg_max_steps:
        working_unitcell = do_absorption_step(peps_tensors, working_unitcell)

        if enforce_elementwise_convergence:
            converged, measure, verbose_data = _is_element_wise_converged(
                last_step_tensors,
                working_unitcell.get_unique_tensors(),
                eps,
                verbose=peps_ad_config.ctmrg_verbose_output,
            )
            last_step_tensors = working_unitcell.get_unique_tensors()
        else:
            corner_singular_values.append(
                _calc_corner_svds(
                    working_unitcell.get_unique_tensors(), shape_corner_svd
                )
            )

            measure = jnp.linalg.norm(
                corner_singular_values[-1] - corner_singular_values[-2]
            )
            converged = measure < eps

        count += 1

        if peps_ad_config.ctmrg_print_steps:
            print(f"CTMRG: {count}: {measure}")
            if peps_ad_config.ctmrg_verbose_output:
                verbose_data = [
                    (int(ti), CTM_Enum(ctm_enum_i).name, float(diff))
                    for ti, ctm_enum_i, diff in verbose_data
                ]
                print(verbose_data)

        if (
            peps_ad_config.ctmrg_increase_truncation_eps
            and count == peps_ad_config.ctmrg_max_steps
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
                print(f"CTMRG: Increasing SVD truncation eps to {new_truncation_eps}.")
                peps_ad_global_state.ctmrg_effective_truncation_eps = new_truncation_eps
                working_unitcell = unitcell
                count = 0

    if _return_truncation_eps:
        last_truncation_eps = peps_ad_global_state.ctmrg_effective_truncation_eps
    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    if (
        peps_ad_config.ctmrg_fail_if_not_converged
        and count == peps_ad_config.ctmrg_max_steps
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


def calc_ctmrg_env_rev(
    res: Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell], unitcell_bar: PEPS_Unit_Cell
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell]:
    """
    Internal helper function of custom VJP to calculate the gradient in
    the backward sweep.
    """
    peps_tensors, new_unitcell, last_truncation_eps = res

    peps_ad_global_state.ctmrg_effective_truncation_eps = last_truncation_eps

    _, vjp_peps_tensors = vjp(
        lambda t: do_absorption_step(t, new_unitcell), peps_tensors
    )

    _, vjp_env = vjp(lambda u: do_absorption_step(peps_tensors, u), new_unitcell)

    converged = False
    count = 0
    env_fixed_point = unitcell_bar
    env_bar_tensors = unitcell_bar.get_unique_tensors()

    while not converged and count < peps_ad_config.ad_custom_max_steps:
        env_fixed_point_last_step = env_fixed_point

        new_env_bar = vjp_env(env_fixed_point)[0]
        new_tensors = new_env_bar.get_unique_tensors()

        env_fixed_point = env_fixed_point.replace_unique_tensors(
            [
                t_old + t_new
                for t_old, t_new in jax.util.safe_zip(env_bar_tensors, new_tensors)
            ]
        )

        converged, measure, verbose_data = _is_element_wise_converged(
            env_fixed_point_last_step.get_unique_tensors(),
            env_fixed_point.get_unique_tensors(),
            peps_ad_config.ad_custom_convergence_eps,
            verbose=peps_ad_config.ad_custom_verbose_output,
        )

        count += 1

        if peps_ad_config.ad_custom_print_steps:
            print(f"Custom VJP: {count}: {measure}")
            if peps_ad_config.ad_custom_verbose_output:
                verbose_data = [
                    (int(ti), CTM_Enum(ctm_enum_i).name, float(diff))
                    for ti, ctm_enum_i, diff in verbose_data
                ]
                print(verbose_data)

    if count == peps_ad_config.ad_custom_max_steps and not converged:
        peps_ad_global_state.ctmrg_effective_truncation_eps = None
        raise CTMRGGradientNotConvergedError

    (t_bar,) = vjp_peps_tensors(env_fixed_point)

    empty_t = [t.zeros_like_self() for t in new_unitcell.get_unique_tensors()]

    peps_ad_global_state.ctmrg_effective_truncation_eps = None

    return t_bar, new_unitcell.replace_unique_tensors(empty_t), jnp.zeros((), dtype=bool)


calc_ctmrg_env_custom_rule.defvjp(calc_ctmrg_env_fwd, calc_ctmrg_env_rev)
