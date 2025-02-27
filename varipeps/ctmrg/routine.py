from functools import partial
import enum

import jax.numpy as jnp
from jax import jit, custom_vjp, vjp, tree_util
import jax.util
from jax.lax import cond, while_loop
import jax.debug as jdebug

from varipeps import varipeps_config, varipeps_global_state
from varipeps.peps import PEPS_Tensor, PEPS_Tensor_Split_Transfer, PEPS_Unit_Cell
from varipeps.utils.debug_print import debug_print
from .absorption import do_absorption_step, do_absorption_step_split_transfer

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
    T1_ket = enum.auto()
    T1_bra = enum.auto()
    T2_ket = enum.auto()
    T2_bra = enum.auto()
    T3_ket = enum.auto()
    T3_bra = enum.auto()
    T4_ket = enum.auto()
    T4_bra = enum.auto()


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


@partial(jit, static_argnums=(3, 4), inline=True)
def _is_element_wise_converged(
    old_peps_tensors: List[PEPS_Tensor],
    new_peps_tensors: List[PEPS_Tensor],
    eps: float,
    verbose: bool = False,
    split_transfer: bool = False,
) -> Tuple[bool, float, Optional[List[Tuple[int, CTM_Enum, float]]]]:
    result = 0

    if split_transfer:
        measure = jnp.zeros((len(old_peps_tensors), 12), dtype=jnp.float64)
    else:
        measure = jnp.zeros((len(old_peps_tensors), 8), dtype=jnp.float64)

    verbose_data = [] if verbose else None

    for ti in range(len(old_peps_tensors)):
        old_shape = old_peps_tensors[ti].C1.shape
        new_shape = new_peps_tensors[ti].C1.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C1[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C1[: new_shape[0], : new_shape[1]]
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 0].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C1, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C2.shape
        new_shape = new_peps_tensors[ti].C2.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C2[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C2[: new_shape[0], : new_shape[1]]
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 1].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C2, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C3.shape
        new_shape = new_peps_tensors[ti].C4.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C3[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C3[: new_shape[0], : new_shape[1]]
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 2].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C3, jnp.amax(diff)))

        old_shape = old_peps_tensors[ti].C4.shape
        new_shape = new_peps_tensors[ti].C4.shape
        diff = jnp.abs(
            new_peps_tensors[ti].C4[: old_shape[0], : old_shape[1]]
            - old_peps_tensors[ti].C4[: new_shape[0], : new_shape[1]]
        )
        result += jnp.sum(diff > eps)
        measure = measure.at[ti, 3].set(
            jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
        )
        if verbose:
            verbose_data.append((ti, CTM_Enum.C4, jnp.amax(diff)))

        if split_transfer:
            old_shape = old_peps_tensors[ti].T1_ket.shape
            new_shape = new_peps_tensors[ti].T1_ket.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T1_ket[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T1_ket[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 4].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T1_ket, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T1_bra.shape
            new_shape = new_peps_tensors[ti].T1_bra.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T1_bra[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T1_bra[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 5].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T1_bra, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T2_ket.shape
            new_shape = new_peps_tensors[ti].T2_ket.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T2_ket[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T2_ket[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 6].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T2_ket, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T2_bra.shape
            new_shape = new_peps_tensors[ti].T2_bra.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T2_bra[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T2_bra[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 7].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T2_bra, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T3_ket.shape
            new_shape = new_peps_tensors[ti].T3_ket.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T3_ket[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T3_ket[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 8].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T3_ket, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T3_bra.shape
            new_shape = new_peps_tensors[ti].T3_bra.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T3_bra[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T3_bra[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 9].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T3_bra, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T4_ket.shape
            new_shape = new_peps_tensors[ti].T4_ket.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T4_ket[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T4_ket[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 10].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T4_ket, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T4_bra.shape
            new_shape = new_peps_tensors[ti].T4_bra.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T4_bra[
                    : old_shape[0], : old_shape[1], : old_shape[2]
                ]
                - old_peps_tensors[ti].T4_bra[
                    : new_shape[0], : new_shape[1], : new_shape[2]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 11].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T4_bra, jnp.amax(diff)))
        else:
            old_shape = old_peps_tensors[ti].T1.shape
            new_shape = new_peps_tensors[ti].T1.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T1[
                    : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
                ]
                - old_peps_tensors[ti].T1[
                    : new_shape[0], : new_shape[1], : new_shape[2], : new_shape[3]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 4].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T1, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T2.shape
            new_shape = new_peps_tensors[ti].T2.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T2[
                    : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
                ]
                - old_peps_tensors[ti].T2[
                    : new_shape[0], : new_shape[1], : new_shape[2], : new_shape[3]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 5].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T2, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T3.shape
            new_shape = new_peps_tensors[ti].T3.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T3[
                    : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
                ]
                - old_peps_tensors[ti].T3[
                    : new_shape[0], : new_shape[1], : new_shape[2], : new_shape[3]
                ]
            )
            result += jnp.sum(diff > eps)
            measure = measure.at[ti, 6].set(
                jnp.linalg.norm(diff), indices_are_sorted=True, unique_indices=True
            )
            if verbose:
                verbose_data.append((ti, CTM_Enum.T3, jnp.amax(diff)))

            old_shape = old_peps_tensors[ti].T4.shape
            new_shape = new_peps_tensors[ti].T4.shape
            diff = jnp.abs(
                new_peps_tensors[ti].T4[
                    : old_shape[0], : old_shape[1], : old_shape[2], : old_shape[3]
                ]
                - old_peps_tensors[ti].T4[
                    : new_shape[0], : new_shape[1], : new_shape[2], : new_shape[3]
                ]
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
        norm_smallest_S,
        state,
        config,
    ) = carry

    if w_unitcell_last_step.is_split_transfer():
        w_unitcell, norm_smallest_S = do_absorption_step_split_transfer(
            w_tensors, w_unitcell_last_step, config, state
        )
    else:
        w_unitcell, norm_smallest_S = do_absorption_step(
            w_tensors, w_unitcell_last_step, config, state
        )

    def elementwise_func(old, new, old_corner, conv_eps, config):
        converged, measure, verbose_data = _is_element_wise_converged(
            old,
            new,
            conv_eps,
            verbose=config.ctmrg_verbose_output,
            split_transfer=w_unitcell.is_split_transfer(),
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
        norm_smallest_S,
        state,
        config,
    )


@jit
def _ctmrg_while_wrapper(start_carry):
    def cond_func(carry):
        _, _, converged, _, _, count, _, _, _, config = carry
        return jnp.logical_not(converged) & (count < config.ctmrg_max_steps)

    (
        _,
        working_unitcell,
        converged,
        _,
        _,
        end_count,
        _,
        norm_smallest_S,
        _,
        _,
    ) = while_loop(cond_func, _ctmrg_body_func, start_carry)

    return working_unitcell, converged, end_count, norm_smallest_S


def calc_ctmrg_env(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    *,
    eps: Optional[float] = None,
    enforce_elementwise_convergence: Optional[bool] = None,
    _return_truncation_eps: bool = False,
) -> PEPS_Unit_Cell:
    """
    Calculate the new converged CTMRG tensors for the unit cell. The function
    updates the environment all iPEPS tensors in the unit cell according to the
    periodic structure.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Keyword args:
      eps (:obj:`float`):
        The convergence criterion.
      enforce_elementwise_convergence (obj:`bool`):
        Enforce elementwise convergence of the CTM tensors instead of only
        convergence of the singular values of the corners.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with all updated converged CTMRG tensors of
        all elements of the unitcell.
    """
    eps = eps if eps is not None else varipeps_config.ctmrg_convergence_eps
    enforce_elementwise_convergence = (
        enforce_elementwise_convergence
        if enforce_elementwise_convergence is not None
        else varipeps_config.ctmrg_enforce_elementwise_convergence
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

    initial_unitcell = unitcell
    working_unitcell = unitcell
    varipeps_global_state.ctmrg_effective_truncation_eps = None

    norm_smallest_S = jnp.nan
    already_tried_chi = {working_unitcell[0, 0][0][0].chi}

    while True:
        tmp_count = 0
        corner_singular_vals = None

        while tmp_count < varipeps_config.ctmrg_max_steps and (
            any(
                getattr(i, j).shape[0] != i.chi or getattr(i, j).shape[1] != i.chi
                for i in working_unitcell.get_unique_tensors()
                for j in ("C1", "C2", "C3", "C4")
            )
            or (
                hasattr(working_unitcell.get_unique_tensors()[0], "T4_ket")
                and any(
                    getattr(i, j).shape[0] != i.interlayer_chi
                    for i in working_unitcell.get_unique_tensors()
                    for j in ("T1_bra", "T2_ket", "T3_bra", "T4_ket")
                )
            )
        ):
            (
                _,
                working_unitcell,
                _,
                corner_singular_vals,
                _,
                tmp_count,
                _,
                norm_smallest_S,
                _,
                _,
            ) = _ctmrg_body_func(
                (
                    peps_tensors,
                    working_unitcell,
                    False,
                    init_corner_singular_vals,
                    eps,
                    tmp_count,
                    enforce_elementwise_convergence,
                    jnp.inf,
                    varipeps_global_state,
                    varipeps_config,
                )
            )

        if tmp_count < varipeps_config.ctmrg_max_steps:
            working_unitcell, converged, end_count, norm_smallest_S = (
                _ctmrg_while_wrapper(
                    (
                        peps_tensors,
                        working_unitcell,
                        False,
                        (
                            corner_singular_vals
                            if corner_singular_vals is not None
                            else init_corner_singular_vals
                        ),
                        eps,
                        tmp_count,
                        enforce_elementwise_convergence,
                        jnp.inf,
                        varipeps_global_state,
                        varipeps_config,
                    )
                )
            )
        else:
            converged = False
            end_count = tmp_count

        current_truncation_eps = (
            varipeps_config.ctmrg_truncation_eps
            if varipeps_global_state.ctmrg_effective_truncation_eps is None
            else varipeps_global_state.ctmrg_effective_truncation_eps
        )

        if (
            varipeps_config.ctmrg_heuristic_increase_chi
            and norm_smallest_S > varipeps_config.ctmrg_heuristic_increase_chi_threshold
            and working_unitcell[0, 0][0][0].chi < working_unitcell[0, 0][0][0].max_chi
        ):
            new_chi = (
                working_unitcell[0, 0][0][0].chi
                + varipeps_config.ctmrg_heuristic_increase_chi_step_size
            )
            if new_chi > working_unitcell[0, 0][0][0].max_chi:
                new_chi = working_unitcell[0, 0][0][0].max_chi

            if not new_chi in already_tried_chi:
                working_unitcell = working_unitcell.change_chi(new_chi)
                initial_unitcell = initial_unitcell.change_chi(new_chi)

                if varipeps_config.ctmrg_print_steps:
                    debug_print(
                        "CTMRG: Increasing chi to {} since smallest SVD Norm was {}.",
                        new_chi,
                        norm_smallest_S,
                    )

                already_tried_chi.add(new_chi)

                continue
        elif (
            varipeps_config.ctmrg_heuristic_decrease_chi
            and norm_smallest_S < current_truncation_eps
            and working_unitcell[0, 0][0][0].chi > 2
        ):
            new_chi = (
                working_unitcell[0, 0][0][0].chi
                - varipeps_config.ctmrg_heuristic_decrease_chi_step_size
            )
            if new_chi < 2:
                new_chi = 2

            if not new_chi in already_tried_chi:
                working_unitcell = working_unitcell.change_chi(new_chi)

                if varipeps_config.ctmrg_print_steps:
                    debug_print(
                        "CTMRG: Decreasing chi to {} since smallest SVD Norm was {}.",
                        new_chi,
                        norm_smallest_S,
                    )

                already_tried_chi.add(new_chi)

                continue

        if (
            varipeps_config.ctmrg_increase_truncation_eps
            and end_count == varipeps_config.ctmrg_max_steps
            and not converged
        ):
            new_truncation_eps = (
                current_truncation_eps
                * varipeps_config.ctmrg_increase_truncation_eps_factor
            )
            if (
                new_truncation_eps
                <= varipeps_config.ctmrg_increase_truncation_eps_max_value
            ):
                if varipeps_config.ctmrg_print_steps:
                    debug_print(
                        "CTMRG: Increasing SVD truncation eps to {}.",
                        new_truncation_eps,
                    )
                varipeps_global_state.ctmrg_effective_truncation_eps = (
                    new_truncation_eps
                )
                working_unitcell = initial_unitcell
                already_tried_chi = {working_unitcell[0, 0][0][0].chi}
                continue

        break

    if _return_truncation_eps:
        last_truncation_eps = varipeps_global_state.ctmrg_effective_truncation_eps
    varipeps_global_state.ctmrg_effective_truncation_eps = None

    if (
        varipeps_config.ctmrg_fail_if_not_converged
        and end_count == varipeps_config.ctmrg_max_steps
        and not converged
    ):
        raise CTMRGNotConvergedError

    if _return_truncation_eps:
        return working_unitcell, last_truncation_eps, norm_smallest_S

    return working_unitcell, norm_smallest_S


@custom_vjp
def calc_ctmrg_env_custom_rule(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    _return_truncation_eps: bool = False,
) -> PEPS_Unit_Cell:
    """
    Wrapper function of :obj:`~varipeps.ctmrg.routine.calc_ctmrg_env` which
    enables the use of the custom VJP for the calculation of the gradient.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
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
    new_unitcell, last_truncation_eps, norm_smallest_S = calc_ctmrg_env_custom_rule(
        peps_tensors, unitcell, _return_truncation_eps=True
    )
    return (new_unitcell, norm_smallest_S), (
        peps_tensors,
        new_unitcell,
        unitcell,
        last_truncation_eps,
    )


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

    new_env_bar = vjp_env((bar_fixed_point_last_step, jnp.array(0, dtype=jnp.float64)))[
        0
    ]

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
        split_transfer=bar_fixed_point.is_split_transfer(),
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
    if new_unitcell.is_split_transfer():
        _, vjp_peps_tensors = vjp(
            lambda t: do_absorption_step_split_transfer(t, new_unitcell, config, state),
            peps_tensors,
        )

        vjp_env = tree_util.Partial(
            vjp(
                lambda u: do_absorption_step_split_transfer(
                    peps_tensors, u, config, state
                ),
                new_unitcell,
            )[1]
        )
    else:
        _, vjp_peps_tensors = vjp(
            lambda t: do_absorption_step(t, new_unitcell, config, state), peps_tensors
        )

        vjp_env = tree_util.Partial(
            vjp(
                lambda u: do_absorption_step(peps_tensors, u, config, state),
                new_unitcell,
            )[1]
        )

    def cond_func(carry):
        _, _, _, converged, count, config, state = carry

        return jnp.logical_not(converged) & (count < config.ad_custom_max_steps)

    _, _, env_fixed_point, converged, end_count, _, _ = while_loop(
        cond_func,
        _ctmrg_rev_while_body,
        (vjp_env, new_unitcell_bar, new_unitcell_bar, False, 0, config, state),
    )

    (t_bar,) = vjp_peps_tensors((env_fixed_point, jnp.array(0, dtype=jnp.float64)))

    return t_bar, converged, end_count


def calc_ctmrg_env_rev(
    res: Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell],
    input_bar: Tuple[PEPS_Unit_Cell, float],
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell]:
    """
    Internal helper function of custom VJP to calculate the gradient in
    the backward sweep.
    """
    unitcell_bar, _ = input_bar
    peps_tensors, new_unitcell, input_unitcell, last_truncation_eps = res

    varipeps_global_state.ctmrg_effective_truncation_eps = last_truncation_eps

    t_bar, converged, end_count = _ctmrg_rev_workhorse(
        peps_tensors, new_unitcell, unitcell_bar, varipeps_config, varipeps_global_state
    )

    varipeps_global_state.ctmrg_effective_truncation_eps = None

    if end_count == varipeps_config.ad_custom_max_steps and not converged:
        raise CTMRGGradientNotConvergedError

    empty_t = [t.zeros_like_self() for t in input_unitcell.get_unique_tensors()]

    return (
        t_bar,
        input_unitcell.replace_unique_tensors(empty_t),
        jnp.zeros((), dtype=bool),
    )


calc_ctmrg_env_custom_rule.defvjp(calc_ctmrg_env_fwd, calc_ctmrg_env_rev)
