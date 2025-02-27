import collections.abc
from dataclasses import dataclass, field
from operator import itemgetter

import jax.numpy as jnp
from jax import jit
from jax.lax import scan, cond

from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps.utils.svd import gauge_fixed_svd
from varipeps.utils.periodic_indices import calculate_periodic_indices
from varipeps.utils.projector_dict import Projector_Dict
from .projectors import (
    calc_left_projectors,
    calc_right_projectors,
    calc_top_projectors,
    calc_bottom_projectors,
    calc_left_projectors_split_transfer,
    calc_right_projectors_split_transfer,
    calc_top_projectors_split_transfer,
    calc_bottom_projectors_split_transfer,
)
from varipeps.expectation.one_site import calc_one_site_single_gate_obj
from varipeps.config import VariPEPS_Config
from varipeps.global_state import VariPEPS_Global_State

from typing import Sequence, Tuple, List, Dict, Literal

CTMRG_Orientation = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


def _tensor_list_from_indices(
    peps_tensors: Sequence[jnp.ndarray], indices: Sequence[Sequence[int]]
) -> List[List[jnp.ndarray]]:
    return [[peps_tensors[ty] for ty in tx] for tx in indices]


def _get_ctmrg_2x2_structure(
    peps_tensors: Sequence[jnp.ndarray],
    view: PEPS_Unit_Cell,
    orientation: CTMRG_Orientation,
) -> Tuple[List[List[jnp.ndarray]], List[List[PEPS_Tensor]]]:
    if orientation == "top-left":
        x_slice = slice(0, 2, None)
        y_slice = slice(0, 2, None)
    elif orientation == "top-right":
        x_slice = slice(0, 2, None)
        y_slice = slice(-1, 1, None)
    elif orientation == "bottom-left":
        x_slice = slice(-1, 1, None)
        y_slice = slice(0, 2, None)
    elif orientation == "bottom-right":
        x_slice = slice(-1, 1, None)
        y_slice = slice(-1, 1, None)
    else:
        raise ValueError("Invalid orientation.")

    indices = view.get_indices((x_slice, y_slice))
    view_tensors = _tensor_list_from_indices(peps_tensors, indices)

    view_tensor_objs = view[x_slice, y_slice]

    return view_tensors, view_tensor_objs


def _get_ctmrg_2x1_structure(
    peps_tensors: Sequence[jnp.ndarray],
    view: PEPS_Unit_Cell,
) -> Tuple[List[List[jnp.ndarray]], List[List[PEPS_Tensor]]]:
    x_slice = slice(0, 2, None)
    y_slice = slice(0, 1, None)

    indices = view.get_indices((x_slice, y_slice))
    view_tensors = _tensor_list_from_indices(peps_tensors, indices)

    view_tensor_objs = view[x_slice, y_slice]

    return view_tensors, view_tensor_objs


def _get_ctmrg_1x2_structure(
    peps_tensors: Sequence[jnp.ndarray],
    view: PEPS_Unit_Cell,
) -> Tuple[List[List[jnp.ndarray]], List[List[PEPS_Tensor]]]:
    x_slice = slice(0, 1, None)
    y_slice = slice(0, 2, None)

    indices = view.get_indices((x_slice, y_slice))
    view_tensors = _tensor_list_from_indices(peps_tensors, indices)

    view_tensor_objs = view[x_slice, y_slice]

    return view_tensors, view_tensor_objs


def _post_process_CTM_tensors(a: jnp.ndarray, config: VariPEPS_Config) -> jnp.ndarray:
    a = a / jnp.linalg.norm(a)
    a_abs = jnp.abs(a)
    a_abs_max = jnp.max(a_abs)

    def scan_max_element(carry, x):
        x_a, x_a_abs = x
        found, phase = carry

        def new_phase(ph, curr_x, curr_x_abs):
            return cond(
                curr_x_abs >= (config.svd_sign_fix_eps * a_abs_max),
                lambda p, c_x, c_x_a: c_x / c_x_a,
                lambda p, c_x, c_x_a: p,
                ph,
                curr_x,
                curr_x_abs,
            )

        phase = cond(
            found, lambda ph, curr_x, curr_x_abs: ph, new_phase, phase, x_a, x_a_abs
        )

        return (jnp.logical_not(jnp.isnan(phase)), phase), None

    (_, phase), _ = scan(
        scan_max_element,
        (jnp.array(False), jnp.array(jnp.nan, dtype=a.dtype)),
        (a.flatten(), a_abs.flatten()),
    )

    return a * phase.conj()


def do_left_absorption(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the left CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated left CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    left_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for y, iter_columns in working_unitcell.iter_all_columns(only_unique=True):
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            left_proj, smallest_S = calc_left_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left"), config, state
            )
            left_projectors[(x, y)] = left_proj
            smallest_S_list.append(smallest_S)

        new_C1 = []
        new_T4 = []
        new_C4 = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_projector = left_projectors.get_projector(x, y, -1, 0).bottom
            new_C1_tmp = apply_contraction_jitted(
                "ctmrg_absorption_left_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1.append(_post_process_CTM_tensors(new_C1_tmp, config))

            T4_projector_top = left_projectors.get_projector(x, y, -1, 0).top
            T4_projector_bottom = left_projectors.get_projector(x, y, 0, 0).bottom

            if (
                working_tensor_obj.d > working_tensor_obj.chi
                or working_tensor_obj.d > working_tensor_obj.D[0] ** 2
            ):
                new_T4_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_left_T4_large_d",
                    [working_tensor],
                    [working_tensor_obj],
                    [T4_projector_top, T4_projector_bottom],
                )
            else:
                new_T4_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_left_T4",
                    [working_tensor],
                    [working_tensor_obj],
                    [T4_projector_top, T4_projector_bottom],
                )
            new_T4.append(_post_process_CTM_tensors(new_T4_tmp, config))

            C4_projector = left_projectors.get_projector(x, y, 0, 0).top
            new_C4_tmp = apply_contraction_jitted(
                "ctmrg_absorption_left_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4.append(_post_process_CTM_tensors(new_C4_tmp, config))

        for x, view in column_views:
            view[0, 1] = view[0, 1][0][0].replace_left_env_tensors(
                new_C1[x], new_T4[x], new_C4[x]
            )

    return working_unitcell, smallest_S_list


def do_right_absorption(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the right CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated right CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    right_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for y, iter_columns in working_unitcell.iter_all_columns(
        reverse=True, only_unique=True
    ):
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            right_proj, smallest_S = calc_right_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-right"),
                config,
                state,
            )
            right_projectors[(x, y)] = right_proj
            smallest_S_list.append(smallest_S)

        new_C2 = []
        new_T2 = []
        new_C3 = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C2_projector = right_projectors.get_projector(x, y, -1, 0).bottom
            new_C2_tmp = apply_contraction_jitted(
                "ctmrg_absorption_right_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2.append(_post_process_CTM_tensors(new_C2_tmp, config))

            T2_projector_top = right_projectors.get_projector(x, y, -1, 0).top
            T2_projector_bottom = right_projectors.get_projector(x, y, 0, 0).bottom
            if (
                working_tensor_obj.d > working_tensor_obj.chi
                or working_tensor_obj.d > working_tensor_obj.D[2] ** 2
            ):
                new_T2_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_right_T2_large_d",
                    [working_tensor],
                    [working_tensor_obj],
                    [T2_projector_top, T2_projector_bottom],
                )
            else:
                new_T2_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_right_T2",
                    [working_tensor],
                    [working_tensor_obj],
                    [T2_projector_top, T2_projector_bottom],
                )
            new_T2.append(_post_process_CTM_tensors(new_T2_tmp, config))

            C3_projector = right_projectors.get_projector(x, y, 0, 0).top
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_absorption_right_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3.append(_post_process_CTM_tensors(new_C3_tmp, config))

        for x, view in column_views:
            view[0, -1] = view[0, -1][0][0].replace_right_env_tensors(
                new_C2[x], new_T2[x], new_C3[x]
            )

    return working_unitcell, smallest_S_list


def do_top_absorption(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the top CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated top CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    top_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            top_proj, smallest_S = calc_top_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left"), config, state
            )
            top_projectors[(x, y)] = top_proj
            smallest_S_list.append(smallest_S)

        new_C1 = []
        new_T1 = []
        new_C2 = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_projector = top_projectors.get_projector(x, y, 0, -1).right  # type: ignore
            new_C1_tmp = apply_contraction_jitted(
                "ctmrg_absorption_top_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1.append(_post_process_CTM_tensors(new_C1_tmp, config))

            T1_projector_left = top_projectors.get_projector(x, y, 0, -1).left  # type: ignore
            T1_projector_right = top_projectors.get_projector(x, y, 0, 0).right  # type: ignore
            if (
                working_tensor_obj.d > working_tensor_obj.chi
                or working_tensor_obj.d > working_tensor_obj.D[3] ** 2
            ):
                new_T1_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_top_T1_large_d",
                    [working_tensor],
                    [working_tensor_obj],
                    [T1_projector_left, T1_projector_right],
                )
            else:
                new_T1_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_top_T1",
                    [working_tensor],
                    [working_tensor_obj],
                    [T1_projector_left, T1_projector_right],
                )
            new_T1.append(_post_process_CTM_tensors(new_T1_tmp, config))

            C2_projector = top_projectors.get_projector(x, y, 0, 0).left  # type: ignore
            new_C2_tmp = apply_contraction_jitted(
                "ctmrg_absorption_top_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2.append(_post_process_CTM_tensors(new_C2_tmp, config))

        for y, view in row_views:
            view[1, 0] = view[1, 0][0][0].replace_top_env_tensors(
                new_C1[y], new_T1[y], new_C2[y]
            )

    return working_unitcell, smallest_S_list


def do_bottom_absorption(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the bottom CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated bottom CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    bottom_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for x, iter_rows in working_unitcell.iter_all_rows(reverse=True, only_unique=True):
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            bottom_proj, smallest_S = calc_bottom_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "bottom-left"),
                config,
                state,
            )
            bottom_projectors[(x, y)] = bottom_proj
            smallest_S_list.append(smallest_S)

        new_C4 = []
        new_T3 = []
        new_C3 = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C4_projector = bottom_projectors.get_projector(x, y, 0, -1).right  # type: ignore
            new_C4_tmp = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4.append(_post_process_CTM_tensors(new_C4_tmp, config))

            T3_projector_left = bottom_projectors.get_projector(x, y, 0, -1).left  # type: ignore
            T3_projector_right = bottom_projectors.get_projector(x, y, 0, 0).right  # type: ignore
            if (
                working_tensor_obj.d > working_tensor_obj.chi
                or working_tensor_obj.d > working_tensor_obj.D[1] ** 2
            ):
                new_T3_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_bottom_T3_large_d",
                    [working_tensor],
                    [working_tensor_obj],
                    [T3_projector_left, T3_projector_right],
                )
            else:
                new_T3_tmp = apply_contraction_jitted(
                    "ctmrg_absorption_bottom_T3",
                    [working_tensor],
                    [working_tensor_obj],
                    [T3_projector_left, T3_projector_right],
                )
            new_T3.append(_post_process_CTM_tensors(new_T3_tmp, config))

            C3_projector = bottom_projectors.get_projector(x, y, 0, 0).left  # type: ignore
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3.append(_post_process_CTM_tensors(new_C3_tmp, config))

        for y, view in row_views:
            view[-1, 0] = view[-1, 0][0][0].replace_bottom_env_tensors(
                new_C4[y], new_T3[y], new_C3[y]
            )

    return working_unitcell, smallest_S_list


def gauge_fix_ctmrg_tensors(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    left_unitaries = []
    right_unitaries = []
    top_unitaries = []
    bottom_unitaries = []

    working_unitcell = unitcell.copy()

    for ti, working_tensor_obj in enumerate(working_unitcell.get_unique_tensors()):
        C1_U, C1_S, C1_Vh = gauge_fixed_svd(working_tensor_obj.C1)
        C3_U, C3_S, C3_Vh = gauge_fixed_svd(working_tensor_obj.C3)

        left_unitaries.append(C1_U)
        top_unitaries.append(C1_Vh)
        bottom_unitaries.append(C3_U)
        right_unitaries.append(C3_Vh)

        working_unitcell.data.peps_tensors[ti] = working_tensor_obj.replace_C1_C3(
            jnp.diag(C1_S), jnp.diag(C3_S)
        )

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            i_0_0 = calculate_periodic_indices((0, 0), view.data.structure, x, y)[0][0]
            i_0_1 = calculate_periodic_indices((0, 1), view.data.structure, x, y)[0][0]
            i_0_neg1 = calculate_periodic_indices((0, -1), view.data.structure, x, y)[
                0
            ][0]
            i_1_0 = calculate_periodic_indices((1, 0), view.data.structure, x, y)[0][0]
            i_neg1_0 = calculate_periodic_indices((-1, 0), view.data.structure, x, y)[
                0
            ][0]

            T1_left_unitary = top_unitaries[i_0_0]
            T1_right_unitary = top_unitaries[i_0_1].conj()
            new_T1 = apply_contraction_jitted(
                "ctmrg_gauge_fix_T1",
                [working_tensor],
                [working_tensor_obj],
                [T1_left_unitary, T1_right_unitary],
            )

            C2_left_unitary = top_unitaries[i_0_1]
            C2_bottom_unitary = right_unitaries[i_neg1_0]
            new_C2 = apply_contraction_jitted(
                "ctmrg_gauge_fix_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_left_unitary, C2_bottom_unitary],
            )

            T2_top_unitary = right_unitaries[i_neg1_0].conj()
            T2_bottom_unitary = right_unitaries[i_0_0]
            new_T2 = apply_contraction_jitted(
                "ctmrg_gauge_fix_T2",
                [working_tensor],
                [working_tensor_obj],
                [T2_top_unitary, T2_bottom_unitary],
            )

            T3_left_unitary = bottom_unitaries[i_0_neg1].conj()
            T3_right_unitary = bottom_unitaries[i_0_0]
            new_T3 = apply_contraction_jitted(
                "ctmrg_gauge_fix_T3",
                [working_tensor],
                [working_tensor_obj],
                [T3_left_unitary, T3_right_unitary],
            )

            C4_top_unitary = left_unitaries[i_1_0]
            C4_right_unitary = bottom_unitaries[i_0_neg1]
            new_C4 = apply_contraction_jitted(
                "ctmrg_gauge_fix_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_top_unitary, C4_right_unitary],
            )

            T4_top_unitary = left_unitaries[i_0_0]
            T4_bottom_unitary = left_unitaries[i_1_0].conj()
            new_T4 = apply_contraction_jitted(
                "ctmrg_gauge_fix_T4",
                [working_tensor],
                [working_tensor_obj],
                [T4_top_unitary, T4_bottom_unitary],
            )

            view[0, 0] = working_tensor_obj.replace_T1_C2_T2_T3_C4_T4(
                new_T1, new_C2, new_T2, new_T3, new_C4, new_T4
            )

    return working_unitcell


def do_absorption_step(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the all CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the all updated CTMRG tensors of
        all elements of the unitcell.
    """
    result, left_smallest_S = do_left_absorption(peps_tensors, unitcell, config, state)
    result, top_smallest_S = do_top_absorption(peps_tensors, result, config, state)
    result, right_smallest_S = do_right_absorption(peps_tensors, result, config, state)
    result, bottom_smallest_S = do_bottom_absorption(
        peps_tensors, result, config, state
    )
    # result = gauge_fix_ctmrg_tensors(peps_tensors, result)
    norm_smallest_S = jnp.linalg.norm(
        jnp.asarray(
            left_smallest_S + top_smallest_S + right_smallest_S + bottom_smallest_S
        ),
        ord=jnp.inf,
    )
    return result, norm_smallest_S


def do_left_absorption_split_transfer(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the left CTMRG tensors after one absorption step and returns
    the updated unitcell. This functions uses the CTMRG method with split
    transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated left CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    left_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for y, iter_columns in working_unitcell.iter_all_columns(only_unique=True):
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            (
                left_proj,
                smallest_S_ket,
                smallest_S_bra,
                smallest_S_phys_ket,
                smallest_S_phys_bra,
            ) = calc_left_projectors_split_transfer(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left"), config, state
            )
            left_projectors[(x, y)] = left_proj
            smallest_S_list.append(smallest_S_ket)
            smallest_S_list.append(smallest_S_bra)
            smallest_S_list.append(smallest_S_phys_ket)
            smallest_S_list.append(smallest_S_phys_bra)

        new_C1_list = []
        new_T4_ket_list = []
        new_T4_bra_list = []
        new_C4_list = []
        new_T4_tmp_list = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_ket_projector = left_projectors.get_projector(x, y, -1, 0).bottom_ket
            C1_bra_projector = left_projectors.get_projector(x, y, -1, 0).bottom_bra
            new_C1_tmp = apply_contraction(
                "ctmrg_split_transfer_absorption_left_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_ket_projector, C1_bra_projector],
            )
            new_C1_list.append(_post_process_CTM_tensors(new_C1_tmp, config))

            T4_ket_projector_top = left_projectors.get_projector(x, y, -1, 0).top_ket
            T4_bra_projector_top = left_projectors.get_projector(x, y, -1, 0).top_bra
            T4_phys_ket_projector_top = left_projectors.get_projector(
                x, y, 0, 0
            ).top_phys_ket
            T4_phys_bra_projector_top = left_projectors.get_projector(
                x, y, 0, 0
            ).top_phys_bra
            T4_ket_projector_bottom = left_projectors.get_projector(
                x, y, 0, 0
            ).bottom_ket
            T4_bra_projector_bottom = left_projectors.get_projector(
                x, y, 0, 0
            ).bottom_bra
            T4_phys_ket_projector_bottom = left_projectors.get_projector(
                x, y, 0, 0
            ).bottom_phys_ket
            T4_phys_bra_projector_bottom = left_projectors.get_projector(
                x, y, 0, 0
            ).bottom_phys_bra

            new_T4_ket = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_left_T4_ket",
                [working_tensor],
                [working_tensor_obj],
                [
                    T4_ket_projector_top,
                    T4_bra_projector_top,
                    T4_phys_ket_projector_bottom,
                    T4_phys_bra_projector_bottom,
                ],
            )

            new_T4_bra = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_left_T4_bra",
                [working_tensor],
                [working_tensor_obj],
                [
                    T4_phys_ket_projector_top,
                    T4_phys_bra_projector_top,
                    T4_ket_projector_bottom,
                    T4_bra_projector_bottom,
                ],
            )

            new_T4_ket_list.append(_post_process_CTM_tensors(new_T4_ket, config))
            new_T4_bra_list.append(_post_process_CTM_tensors(new_T4_bra, config))

            C4_ket_projector = left_projectors.get_projector(x, y, 0, 0).top_ket
            C4_bra_projector = left_projectors.get_projector(x, y, 0, 0).top_bra
            new_C4_tmp = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_left_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_ket_projector, C4_bra_projector],
            )
            new_C4_list.append(_post_process_CTM_tensors(new_C4_tmp, config))

        for x, view in column_views:
            view[0, 1] = view[0, 1][0][0].replace_left_env_tensors(
                new_C1_list[x], new_T4_ket_list[x], new_T4_bra_list[x], new_C4_list[x]
            )

    return working_unitcell, smallest_S_list  # , new_T4_tmp_list


def do_right_absorption_split_transfer(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the right CTMRG tensors after one absorption step and returns
    the updated unitcell. This functions uses the CTMRG method with split
    transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated right CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    right_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for y, iter_columns in working_unitcell.iter_all_columns(
        reverse=True, only_unique=True
    ):
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            (
                right_proj,
                smallest_S_ket,
                smallest_S_bra,
                smallest_S_phys_ket,
                smallest_S_phys_bra,
            ) = calc_right_projectors_split_transfer(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-right"),
                config,
                state,
            )
            right_projectors[(x, y)] = right_proj
            smallest_S_list.append(smallest_S_ket)
            smallest_S_list.append(smallest_S_bra)
            smallest_S_list.append(smallest_S_phys_ket)
            smallest_S_list.append(smallest_S_phys_bra)

        new_C2_list = []
        new_T2_ket_list = []
        new_T2_bra_list = []
        new_C3_list = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C2_ket_projector = right_projectors.get_projector(x, y, -1, 0).bottom_ket
            C2_bra_projector = right_projectors.get_projector(x, y, -1, 0).bottom_bra
            new_C2_tmp = apply_contraction(
                "ctmrg_split_transfer_absorption_right_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_ket_projector, C2_bra_projector],
            )
            new_C2_list.append(_post_process_CTM_tensors(new_C2_tmp, config))

            T2_ket_projector_top = right_projectors.get_projector(x, y, -1, 0).top_ket
            T2_bra_projector_top = right_projectors.get_projector(x, y, -1, 0).top_bra
            T2_phys_ket_projector_top = right_projectors.get_projector(
                x, y, 0, 0
            ).top_phys_ket
            T2_phys_bra_projector_top = right_projectors.get_projector(
                x, y, 0, 0
            ).top_phys_bra
            T2_ket_projector_bottom = right_projectors.get_projector(
                x, y, 0, 0
            ).bottom_ket
            T2_bra_projector_bottom = right_projectors.get_projector(
                x, y, 0, 0
            ).bottom_bra
            T2_phys_ket_projector_bottom = right_projectors.get_projector(
                x, y, 0, 0
            ).bottom_phys_ket
            T2_phys_bra_projector_bottom = right_projectors.get_projector(
                x, y, 0, 0
            ).bottom_phys_bra

            new_T2_ket = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_right_T2_ket",
                [working_tensor],
                [working_tensor_obj],
                [
                    T2_ket_projector_top,
                    T2_bra_projector_top,
                    T2_phys_ket_projector_bottom,
                    T2_phys_bra_projector_bottom,
                ],
            )

            new_T2_bra = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_right_T2_bra",
                [working_tensor],
                [working_tensor_obj],
                [
                    T2_phys_ket_projector_top,
                    T2_phys_bra_projector_top,
                    T2_ket_projector_bottom,
                    T2_bra_projector_bottom,
                ],
            )

            new_T2_ket_list.append(_post_process_CTM_tensors(new_T2_ket, config))
            new_T2_bra_list.append(_post_process_CTM_tensors(new_T2_bra, config))

            C3_ket_projector = right_projectors.get_projector(x, y, 0, 0).top_ket
            C3_bra_projector = right_projectors.get_projector(x, y, 0, 0).top_bra
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_right_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_ket_projector, C3_bra_projector],
            )
            new_C3_list.append(_post_process_CTM_tensors(new_C3_tmp, config))

        for x, view in column_views:
            view[0, -1] = view[0, -1][0][0].replace_right_env_tensors(
                new_C2_list[x], new_T2_ket_list[x], new_T2_bra_list[x], new_C3_list[x]
            )

    return working_unitcell, smallest_S_list


def do_top_absorption_split_transfer(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the top CTMRG tensors after one absorption step and returns
    the updated unitcell. This functions uses the CTMRG method with split
    transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated top CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    top_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            (
                top_proj,
                smallest_S_ket,
                smallest_S_bra,
                smallest_S_phys_ket,
                smallest_S_phys_bra,
            ) = calc_top_projectors_split_transfer(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left"), config, state
            )
            top_projectors[(x, y)] = top_proj
            smallest_S_list.append(smallest_S_ket)
            smallest_S_list.append(smallest_S_bra)
            smallest_S_list.append(smallest_S_phys_ket)
            smallest_S_list.append(smallest_S_phys_bra)

        new_C1_list = []
        new_T1_ket_list = []
        new_T1_bra_list = []
        new_C2_list = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_ket_projector = top_projectors.get_projector(x, y, 0, -1).right_ket
            C1_bra_projector = top_projectors.get_projector(x, y, 0, -1).right_bra
            new_C1_tmp = apply_contraction(
                "ctmrg_split_transfer_absorption_top_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_ket_projector, C1_bra_projector],
            )
            new_C1_list.append(_post_process_CTM_tensors(new_C1_tmp, config))

            T1_ket_projector_left = top_projectors.get_projector(x, y, 0, -1).left_ket
            T1_bra_projector_left = top_projectors.get_projector(x, y, 0, -1).left_bra
            T1_phys_ket_projector_left = top_projectors.get_projector(
                x, y, 0, 0
            ).left_phys_ket
            T1_phys_bra_projector_left = top_projectors.get_projector(
                x, y, 0, 0
            ).left_phys_bra
            T1_ket_projector_right = top_projectors.get_projector(x, y, 0, 0).right_ket
            T1_bra_projector_right = top_projectors.get_projector(x, y, 0, 0).right_bra
            T1_phys_ket_projector_right = top_projectors.get_projector(
                x, y, 0, 0
            ).right_phys_ket
            T1_phys_bra_projector_right = top_projectors.get_projector(
                x, y, 0, 0
            ).right_phys_bra

            new_T1_ket = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_top_T1_ket",
                [working_tensor],
                [working_tensor_obj],
                [
                    T1_ket_projector_left,
                    T1_bra_projector_left,
                    T1_phys_ket_projector_right,
                    T1_phys_bra_projector_right,
                ],
            )

            new_T1_bra = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_top_T1_bra",
                [working_tensor],
                [working_tensor_obj],
                [
                    T1_phys_ket_projector_left,
                    T1_phys_bra_projector_left,
                    T1_ket_projector_right,
                    T1_bra_projector_right,
                ],
            )

            new_T1_ket_list.append(_post_process_CTM_tensors(new_T1_ket, config))
            new_T1_bra_list.append(_post_process_CTM_tensors(new_T1_bra, config))

            C2_ket_projector = top_projectors.get_projector(x, y, 0, 0).left_ket
            C2_bra_projector = top_projectors.get_projector(x, y, 0, 0).left_bra
            new_C2_tmp = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_top_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_ket_projector, C2_bra_projector],
            )
            new_C2_list.append(_post_process_CTM_tensors(new_C2_tmp, config))

        for y, view in row_views:
            view[1, 0] = view[1, 0][0][0].replace_top_env_tensors(
                new_C1_list[y], new_T1_ket_list[y], new_T1_bra_list[y], new_C2_list[y]
            )

    return working_unitcell, smallest_S_list


def do_bottom_absorption_split_transfer(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the left CTMRG tensors after one absorption step and returns
    the updated unitcell. This functions uses the CTMRG method with split
    transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the updated left CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    bottom_projectors = Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    smallest_S_list = []

    for x, iter_rows in working_unitcell.iter_all_rows(reverse=True, only_unique=True):
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            (
                bottom_proj,
                smallest_S_ket,
                smallest_S_bra,
                smallest_S_phys_ket,
                smallest_S_phys_bra,
            ) = calc_bottom_projectors_split_transfer(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "bottom-left"),
                config,
                state,
            )
            bottom_projectors[(x, y)] = bottom_proj
            smallest_S_list.append(smallest_S_ket)
            smallest_S_list.append(smallest_S_bra)
            smallest_S_list.append(smallest_S_phys_ket)
            smallest_S_list.append(smallest_S_phys_bra)

        new_C4_list = []
        new_T3_ket_list = []
        new_T3_bra_list = []
        new_C3_list = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C4_ket_projector = bottom_projectors.get_projector(x, y, 0, -1).right_ket
            C4_bra_projector = bottom_projectors.get_projector(x, y, 0, -1).right_bra
            new_C4_tmp = apply_contraction(
                "ctmrg_split_transfer_absorption_bottom_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_ket_projector, C4_bra_projector],
            )
            new_C4_list.append(_post_process_CTM_tensors(new_C4_tmp, config))

            T3_ket_projector_left = bottom_projectors.get_projector(
                x, y, 0, -1
            ).left_ket
            T3_bra_projector_left = bottom_projectors.get_projector(
                x, y, 0, -1
            ).left_bra
            T3_phys_ket_projector_left = bottom_projectors.get_projector(
                x, y, 0, 0
            ).left_phys_ket
            T3_phys_bra_projector_left = bottom_projectors.get_projector(
                x, y, 0, 0
            ).left_phys_bra
            T3_ket_projector_right = bottom_projectors.get_projector(
                x, y, 0, 0
            ).right_ket
            T3_bra_projector_right = bottom_projectors.get_projector(
                x, y, 0, 0
            ).right_bra
            T3_phys_ket_projector_right = bottom_projectors.get_projector(
                x, y, 0, 0
            ).right_phys_ket
            T3_phys_bra_projector_right = bottom_projectors.get_projector(
                x, y, 0, 0
            ).right_phys_bra

            new_T3_ket = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_bottom_T3_ket",
                [working_tensor],
                [working_tensor_obj],
                [
                    T3_ket_projector_left,
                    T3_bra_projector_left,
                    T3_phys_ket_projector_right,
                    T3_phys_bra_projector_right,
                ],
            )

            new_T3_bra = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_bottom_T3_bra",
                [working_tensor],
                [working_tensor_obj],
                [
                    T3_phys_ket_projector_left,
                    T3_phys_bra_projector_left,
                    T3_ket_projector_right,
                    T3_bra_projector_right,
                ],
            )

            new_T3_ket_list.append(_post_process_CTM_tensors(new_T3_ket, config))
            new_T3_bra_list.append(_post_process_CTM_tensors(new_T3_bra, config))

            C3_ket_projector = bottom_projectors.get_projector(x, y, 0, 0).left_ket
            C3_bra_projector = bottom_projectors.get_projector(x, y, 0, 0).left_bra
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_split_transfer_absorption_bottom_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_ket_projector, C3_bra_projector],
            )
            new_C3_list.append(_post_process_CTM_tensors(new_C3_tmp, config))

        for y, view in row_views:
            view[-1, 0] = view[-1, 0][0][0].replace_bottom_env_tensors(
                new_C4_list[y], new_T3_ket_list[y], new_T3_bra_list[y], new_C3_list[y]
            )

    return working_unitcell, smallest_S_list


def do_absorption_step_split_transfer(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the all CTMRG tensors after one absorption step and returns
    the updated unitcell. This functions uses the CTMRG method with split
    transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
        The unitcell to work on.
    Returns:
      :obj:`~varipeps.peps.PEPS_Unit_Cell`:
        New instance of the unitcell with the all updated CTMRG tensors of
        all elements of the unitcell.
    """
    result, left_smallest_S = do_left_absorption_split_transfer(
        peps_tensors, unitcell, config, state
    )
    result, top_smallest_S = do_top_absorption_split_transfer(
        peps_tensors, result, config, state
    )
    result, right_smallest_S = do_right_absorption_split_transfer(
        peps_tensors, result, config, state
    )
    result, bottom_smallest_S = do_bottom_absorption_split_transfer(
        peps_tensors, result, config, state
    )
    norm_smallest_S = jnp.linalg.norm(
        jnp.asarray(
            left_smallest_S + top_smallest_S + right_smallest_S + bottom_smallest_S
        ),
        ord=jnp.inf,
    )

    return result, norm_smallest_S
