from functools import partial
import enum

import jax.numpy as jnp
from jax import jit, custom_vjp, vjp, tree_util
import jax.util
from jax.lax import cond, while_loop
import jax.debug as jdebug

from varipeps import varipeps_config, varipeps_global_state
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.utils.debug_print import debug_print
from .structure_factor_absorption import do_absorption_step_structure_factor
from .routine import CTMRGNotConvergedError

from typing import Sequence, Tuple, List, Optional


@enum.unique
class CTM_Enum_Structure_Factor(enum.IntEnum):
    C1 = enum.auto()
    C2 = enum.auto()
    C3 = enum.auto()
    C4 = enum.auto()
    C1_Phase = enum.auto()
    C2_Phase = enum.auto()
    C3_Phase = enum.auto()
    C4_Phase = enum.auto()
    T1 = enum.auto()
    T2 = enum.auto()
    T3 = enum.auto()
    T4 = enum.auto()


@partial(jit, static_argnums=(2,), inline=True)
def _calc_corner_svds_structure_factor(
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

        C1_phase_svd = jnp.linalg.svd(t.C1_phase, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 4, : C1_phase_svd.shape[0]].set(
            C1_phase_svd, indices_are_sorted=True, unique_indices=True
        )

        C2_phase_svd = jnp.linalg.svd(t.C2_phase, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 5, : C2_phase_svd.shape[0]].set(
            C2_phase_svd, indices_are_sorted=True, unique_indices=True
        )

        C3_phase_svd = jnp.linalg.svd(t.C3_phase, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 6, : C3_phase_svd.shape[0]].set(
            C3_phase_svd, indices_are_sorted=True, unique_indices=True
        )

        C4_phase_svd = jnp.linalg.svd(t.C4_phase, full_matrices=False, compute_uv=False)
        step_corner_svd = step_corner_svd.at[ti, 7, : C4_phase_svd.shape[0]].set(
            C4_phase_svd, indices_are_sorted=True, unique_indices=True
        )

    return step_corner_svd


@jit
def _ctmrg_body_func_structure_factor(carry):
    (
        w_tensors,
        w_unitcell_last_step,
        converged,
        last_corner_svd,
        eps,
        count,
        norm_smallest_S,
        structure_factor_gates,
        structure_factor_outer_factors,
        structure_factor_inner_factors,
        state,
        config,
    ) = carry

    w_unitcell, norm_smallest_S = do_absorption_step_structure_factor(
        w_tensors,
        w_unitcell_last_step,
        structure_factor_gates,
        structure_factor_outer_factors,
        structure_factor_inner_factors,
        config,
        state,
    )

    verbose_data = [] if config.ctmrg_verbose_output else None
    if last_corner_svd is None:
        corner_svd = None
        converged = False
        measure = jnp.nan
    else:
        corner_svd = _calc_corner_svds_structure_factor(
            w_unitcell.get_unique_tensors(), last_corner_svd, None
        )
        measure = jnp.linalg.norm(corner_svd - last_corner_svd)
        converged = measure < eps

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
        norm_smallest_S,
        structure_factor_gates,
        structure_factor_outer_factors,
        structure_factor_inner_factors,
        state,
        config,
    )


@jit
def _ctmrg_while_wrapper_structure_factor(start_carry):
    def cond_func(carry):
        _, _, converged, _, _, count, _, _, _, _, _, config = carry
        return jnp.logical_not(converged) & (count < config.ctmrg_max_steps)

    (
        _,
        working_unitcell,
        converged,
        _,
        _,
        end_count,
        norm_smallest_S,
        _,
        _,
        _,
        _,
        _,
    ) = while_loop(cond_func, _ctmrg_body_func_structure_factor, start_carry)

    return working_unitcell, converged, end_count, norm_smallest_S


def calc_ctmrg_env_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factors: Sequence[float],
    structure_factor_inner_factors: Sequence[float],
    *,
    eps: Optional[float] = None,
    _return_truncation_eps: bool = False,
) -> PEPS_Unit_Cell:
    """
    Calculate the new converged CTMRG tensors for the unit cell. The function
    updates the environment all iPEPS tensors in the unit cell according to the
    periodic structure. This routine also calculates the tensors including
    the phase factor for a structure factor calculation.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      structure_factor_gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence with the observables which is absorbed into the CTM tensors
        containing the phase for the structure factor calculation. Expected to
        be a sequence where the gate is already multiplied with identities to
        match the physical dimension of the coarse-grained tensor
      structure_factor_outer_factors (:obj:`float`):
        The sequence with factors used to calculate the new tensors by shifting
        one site in the square lattice. Likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``. If length two, the first
        argument will be used for bottom absorption and its complex conjugate for
        top and the second one for right and its complex conjugate for left
        absorption. If length four, it will be used in the order
        (top, bottom, left, right).
      structure_factor_inner_factors (:term:`sequence` of :obj:`float`):
        For coarse-grained systems the sequence with the factors used to
        calculate the phase by shifting one site inside one coarse-grained
        square site. Set it to None, [] or [1] if system has
        no coarsed-grained structure. If used likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
    Keyword args:
      eps (:obj:`float`):
        The convergence criterion.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with all updated converged CTMRG tensors of
        all elements of the unitcell.
    """
    eps = eps if eps is not None else varipeps_config.ctmrg_convergence_eps

    shape_corner_svd = (
        unitcell.get_len_unique_tensors(),
        8,
        unitcell[0, 0][0][0].chi,
    )
    init_corner_singular_vals = _calc_corner_svds_structure_factor(
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
                _,
                _,
                _,
            ) = _ctmrg_body_func_structure_factor(
                (
                    peps_tensors,
                    working_unitcell,
                    False,
                    init_corner_singular_vals,
                    eps,
                    tmp_count,
                    jnp.inf,
                    structure_factor_gates,
                    structure_factor_outer_factors,
                    structure_factor_inner_factors,
                    varipeps_global_state,
                    varipeps_config,
                )
            )

        working_unitcell, converged, end_count, norm_smallest_S = (
            _ctmrg_while_wrapper_structure_factor(
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
                    jnp.inf,
                    structure_factor_gates,
                    structure_factor_outer_factors,
                    structure_factor_inner_factors,
                    varipeps_global_state,
                    varipeps_config,
                )
            )
        )

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
