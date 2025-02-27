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


def do_left_absorption_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factor: float,
    structure_factor_inner_factors: Sequence[float],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the left CTMRG tensors after one absorption step and returns
    the updated unitcell. This routine also calculates the tensors including
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
      structure_factor_outer_factor (:obj:`float`):
        The factor used to calculate the new tensors by shifting one site in
        the square lattice to the left. Likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
      structure_factor_inner_factors (:term:`sequence` of :obj:`float`):
        For coarse-grained systems the sequence with the factors used to
        calculate the phase by shifting one site inside one coarse-grained
        square site. Set it to None, [] or [1] if system has
        no coarsed-grained structure. If used likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
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

    if (
        structure_factor_inner_factors is None
        or len(structure_factor_inner_factors) == 0
    ):
        structure_factor_inner_factors = (1,)

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
        new_C1_phase = []
        new_T4_phase = []
        new_C4_phase = []

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
            new_C1_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_left_C1_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_left_C1_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1_phase_tmp = (
                new_C1_phase_tmp_1 + new_C1_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C1_tmp_norm = jnp.linalg.norm(new_C1_tmp)
            new_C1.append(new_C1_tmp / new_C1_tmp_norm)
            new_C1_phase.append(new_C1_phase_tmp / new_C1_tmp_norm / 2)

            T4_projector_top = left_projectors.get_projector(x, y, -1, 0).top
            T4_projector_bottom = left_projectors.get_projector(x, y, 0, 0).bottom
            new_T4_tmp = apply_contraction_jitted(
                "ctmrg_absorption_left_T4_large_d",
                [working_tensor],
                [working_tensor_obj],
                [T4_projector_top, T4_projector_bottom],
            )
            new_T4_phase_tmp = apply_contraction_jitted(
                "ctmrg_absorption_left_T4_large_d_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [T4_projector_top, T4_projector_bottom],
            )

            for i, factor in enumerate(structure_factor_inner_factors):
                new_T4_phase_tmp += (
                    apply_contraction_jitted(
                        "ctmrg_absorption_left_T4_large_d_phase_2",
                        [working_tensor],
                        [working_tensor_obj],
                        [
                            T4_projector_top,
                            T4_projector_bottom,
                            structure_factor_gates[i],
                        ],
                    )
                    * factor
                )

            new_T4_phase_tmp = new_T4_phase_tmp * structure_factor_outer_factor
            new_T4_tmp_norm = jnp.linalg.norm(new_T4_tmp)
            new_T4.append(new_T4_tmp / new_T4_tmp_norm)
            new_T4_phase.append(
                new_T4_phase_tmp
                / new_T4_tmp_norm
                / (1 + len(structure_factor_inner_factors))
            )

            C4_projector = left_projectors.get_projector(x, y, 0, 0).top
            new_C4_tmp = apply_contraction_jitted(
                "ctmrg_absorption_left_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_left_C4_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_left_C4_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4_phase_tmp = (
                new_C4_phase_tmp_1 + new_C4_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C4_tmp_norm = jnp.linalg.norm(new_C4_tmp)
            new_C4.append(new_C4_tmp / new_C4_tmp_norm)
            new_C4_phase.append(new_C4_phase_tmp / new_C4_tmp_norm / 2)

        for x, view in column_views:
            view[0, 1] = view[0, 1][0][0].replace_left_env_tensors(
                new_C1[x],
                new_T4[x],
                new_C4[x],
                new_C1_phase[x],
                new_T4_phase[x],
                new_C4_phase[x],
            )

    return working_unitcell, smallest_S_list


def do_right_absorption_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factor: float,
    structure_factor_inner_factors: Sequence[float],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the right CTMRG tensors after one absorption step and returns
    the updated unitcell. This routine also calculates the tensors including
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
      structure_factor_outer_factor (:obj:`float`):
        The factor used to calculate the new tensors by shifting one site in
        the square lattice to the right. Likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
      structure_factor_inner_factors (:term:`sequence` of :obj:`float`):
        For coarse-grained systems the sequence with the factors used to
        calculate the phase by shifting one site inside one coarse-grained
        square site. Set it to None, [] or [1] if system has
        no coarsed-grained structure. If used likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
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

    if (
        structure_factor_inner_factors is None
        or len(structure_factor_inner_factors) == 0
    ):
        structure_factor_inner_factors = (1,)

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
        new_C2_phase = []
        new_T2_phase = []
        new_C3_phase = []

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
            new_C2_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_right_C2_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_right_C2_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2_phase_tmp = (
                new_C2_phase_tmp_1 + new_C2_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C2_tmp_norm = jnp.linalg.norm(new_C2_tmp)
            new_C2.append(new_C2_tmp / new_C2_tmp_norm)
            new_C2_phase.append(new_C2_phase_tmp / new_C2_tmp_norm / 2)

            T2_projector_top = right_projectors.get_projector(x, y, -1, 0).top
            T2_projector_bottom = right_projectors.get_projector(x, y, 0, 0).bottom
            new_T2_tmp = apply_contraction_jitted(
                "ctmrg_absorption_right_T2_large_d",
                [working_tensor],
                [working_tensor_obj],
                [T2_projector_top, T2_projector_bottom],
            )
            new_T2_phase_tmp = apply_contraction_jitted(
                "ctmrg_absorption_right_T2_large_d_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [T2_projector_top, T2_projector_bottom],
            )

            for i, factor in enumerate(structure_factor_inner_factors):
                new_T2_phase_tmp += (
                    apply_contraction_jitted(
                        "ctmrg_absorption_right_T2_large_d_phase_2",
                        [working_tensor],
                        [working_tensor_obj],
                        [
                            T2_projector_top,
                            T2_projector_bottom,
                            structure_factor_gates[i],
                        ],
                    )
                    * factor
                )

            new_T2_phase_tmp = new_T2_phase_tmp * structure_factor_outer_factor
            new_T2_tmp_norm = jnp.linalg.norm(new_T2_tmp)
            new_T2.append(new_T2_tmp / new_T2_tmp_norm)
            new_T2_phase.append(
                new_T2_phase_tmp
                / new_T2_tmp_norm
                / (1 + len(structure_factor_inner_factors))
            )

            C3_projector = right_projectors.get_projector(x, y, 0, 0).top
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_absorption_right_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_right_C3_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_right_C3_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp = (
                new_C3_phase_tmp_1 + new_C3_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C3_tmp_norm = jnp.linalg.norm(new_C3_tmp)
            new_C3.append(new_C3_tmp / new_C3_tmp_norm)
            new_C3_phase.append(new_C3_phase_tmp / new_C3_tmp_norm / 2)

        for x, view in column_views:
            view[0, -1] = view[0, -1][0][0].replace_right_env_tensors(
                new_C2[x],
                new_T2[x],
                new_C3[x],
                new_C2_phase[x],
                new_T2_phase[x],
                new_C3_phase[x],
            )

    return working_unitcell, smallest_S_list


def do_top_absorption_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factor: float,
    structure_factor_inner_factors: Sequence[float],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the top CTMRG tensors after one absorption step and returns
    the updated unitcell. This routine also calculates the tensors including
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
      structure_factor_outer_factor (:obj:`float`):
        The factor used to calculate the new tensors by shifting one site in
        the square lattice to the top. Likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
      structure_factor_inner_factors (:term:`sequence` of :obj:`float`):
        For coarse-grained systems the sequence with the factors used to
        calculate the phase by shifting one site inside one coarse-grained
        square site. Set it to None, [] or [1] if system has
        no coarsed-grained structure. If used likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
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

    if (
        structure_factor_inner_factors is None
        or len(structure_factor_inner_factors) == 0
    ):
        structure_factor_inner_factors = (1,)

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
        new_C1_phase = []
        new_T1_phase = []
        new_C2_phase = []

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
            new_C1_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_top_C1_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_top_C1_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1_phase_tmp = (
                new_C1_phase_tmp_1 + new_C1_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C1_tmp_norm = jnp.linalg.norm(new_C1_tmp)
            new_C1.append(new_C1_tmp / new_C1_tmp_norm)
            new_C1_phase.append(new_C1_phase_tmp / new_C1_tmp_norm / 2)

            T1_projector_left = top_projectors.get_projector(x, y, 0, -1).left  # type: ignore
            T1_projector_right = top_projectors.get_projector(x, y, 0, 0).right  # type: ignore
            new_T1_tmp = apply_contraction_jitted(
                "ctmrg_absorption_top_T1_large_d",
                [working_tensor],
                [working_tensor_obj],
                [T1_projector_left, T1_projector_right],
            )
            new_T1_phase_tmp = apply_contraction_jitted(
                "ctmrg_absorption_top_T1_large_d_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [T1_projector_left, T1_projector_right],
            )

            for i, factor in enumerate(structure_factor_inner_factors):
                new_T1_phase_tmp += (
                    apply_contraction_jitted(
                        "ctmrg_absorption_top_T1_large_d_phase_2",
                        [working_tensor],
                        [working_tensor_obj],
                        [
                            T1_projector_left,
                            T1_projector_right,
                            structure_factor_gates[i],
                        ],
                    )
                    * factor
                )

            new_T1_phase_tmp = new_T1_phase_tmp * structure_factor_outer_factor
            new_T1_tmp_norm = jnp.linalg.norm(new_T1_tmp)
            new_T1.append(new_T1_tmp / new_T1_tmp_norm)
            new_T1_phase.append(
                new_T1_phase_tmp
                / new_T1_tmp_norm
                / (1 + len(structure_factor_inner_factors))
            )

            C2_projector = top_projectors.get_projector(x, y, 0, 0).left  # type: ignore
            new_C2_tmp = apply_contraction_jitted(
                "ctmrg_absorption_top_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_top_C2_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_top_C2_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2_phase_tmp = (
                new_C2_phase_tmp_1 + new_C2_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C2_tmp_norm = jnp.linalg.norm(new_C2_tmp)
            new_C2.append(new_C2_tmp / new_C2_tmp_norm)
            new_C2_phase.append(new_C2_phase_tmp / new_C2_tmp_norm / 2)

        for y, view in row_views:
            view[1, 0] = view[1, 0][0][0].replace_top_env_tensors(
                new_C1[y],
                new_T1[y],
                new_C2[y],
                new_C1_phase[y],
                new_T1_phase[y],
                new_C2_phase[y],
            )

    return working_unitcell, smallest_S_list


def do_bottom_absorption_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factor: float,
    structure_factor_inner_factors: Sequence[float],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the bottom CTMRG tensors after one absorption step and returns
    the updated unitcell. This routine also calculates the tensors including
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
      structure_factor_outer_factor (:obj:`float`):
        The factor used to calculate the new tensors by shifting one site in
        the square lattice to the bottom. Likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
      structure_factor_inner_factors (:term:`sequence` of :obj:`float`):
        For coarse-grained systems the sequence with the factors used to
        calculate the phase by shifting one site inside one coarse-grained
        square site. Set it to None, [] or [1] if system has
        no coarsed-grained structure. If used likely something like
        ``jnp.exp(- 1j * q_vector @ r_vector)``.
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

    if (
        structure_factor_inner_factors is None
        or len(structure_factor_inner_factors) == 0
    ):
        structure_factor_inner_factors = (1,)

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
        new_C4_phase = []
        new_T3_phase = []
        new_C3_phase = []

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
            new_C4_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C4_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C4_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4_phase_tmp = (
                new_C4_phase_tmp_1 + new_C4_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C4_tmp_norm = jnp.linalg.norm(new_C4_tmp)
            new_C4.append(new_C4_tmp / new_C4_tmp_norm)
            new_C4_phase.append(new_C4_phase_tmp / new_C4_tmp_norm / 2)

            T3_projector_left = bottom_projectors.get_projector(x, y, 0, -1).left  # type: ignore
            T3_projector_right = bottom_projectors.get_projector(x, y, 0, 0).right  # type: ignore
            new_T3_tmp = apply_contraction_jitted(
                "ctmrg_absorption_bottom_T3_large_d",
                [working_tensor],
                [working_tensor_obj],
                [T3_projector_left, T3_projector_right],
            )
            new_T3_phase_tmp = apply_contraction_jitted(
                "ctmrg_absorption_bottom_T3_large_d_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [T3_projector_left, T3_projector_right],
            )

            for i, factor in enumerate(structure_factor_inner_factors):
                new_T3_phase_tmp += (
                    apply_contraction_jitted(
                        "ctmrg_absorption_bottom_T3_large_d_phase_2",
                        [working_tensor],
                        [working_tensor_obj],
                        [
                            T3_projector_left,
                            T3_projector_right,
                            structure_factor_gates[i],
                        ],
                    )
                    * factor
                )

            new_T3_phase_tmp = new_T3_phase_tmp * structure_factor_outer_factor
            new_T3_tmp_norm = jnp.linalg.norm(new_T3_tmp)
            new_T3.append(new_T3_tmp / new_T3_tmp_norm)
            new_T3_phase.append(
                new_T3_phase_tmp
                / new_T3_tmp_norm
                / (1 + len(structure_factor_inner_factors))
            )

            C3_projector = bottom_projectors.get_projector(x, y, 0, 0).left  # type: ignore
            new_C3_tmp = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp_1 = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C3_phase_1",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp_2 = apply_contraction_jitted(
                "ctmrg_absorption_bottom_C3_phase_2",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3_phase_tmp = (
                new_C3_phase_tmp_1 + new_C3_phase_tmp_2
            ) * structure_factor_outer_factor
            new_C3_tmp_norm = jnp.linalg.norm(new_C3_tmp)
            new_C3.append(new_C3_tmp / new_C3_tmp_norm)
            new_C3_phase.append(new_C3_phase_tmp / new_C3_tmp_norm / 2)

        for y, view in row_views:
            view[-1, 0] = view[-1, 0][0][0].replace_bottom_env_tensors(
                new_C4[y],
                new_T3[y],
                new_C3[y],
                new_C4_phase[y],
                new_T3_phase[y],
                new_C3_phase[y],
            )

    return working_unitcell, smallest_S_list


def do_absorption_step_structure_factor(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    structure_factor_gates: Sequence[jnp.ndarray],
    structure_factor_outer_factors: Sequence[float],
    structure_factor_inner_factors: Sequence[float],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    """
    Calculate the all CTMRG tensors after one absorption step and returns
    the updated unitcell. This routine also calculates the tensors including
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

    if len(structure_factor_outer_factors) == 2:
        structure_factor_outer_factors = (
            structure_factor_outer_factors[0].conj(),
            structure_factor_outer_factors[0],
            structure_factor_outer_factors[1].conj(),
            structure_factor_outer_factors[1],
        )
    if len(structure_factor_outer_factors) != 4:
        raise ValueError("Length mismatch for outer factor list.")

    result, left_smallest_S = do_left_absorption_structure_factor(
        peps_tensors,
        unitcell,
        structure_factor_gates,
        structure_factor_outer_factors[2],
        structure_factor_inner_factors,
        config,
        state,
    )
    result, top_smallest_S = do_top_absorption_structure_factor(
        peps_tensors,
        result,
        structure_factor_gates,
        structure_factor_outer_factors[0],
        structure_factor_inner_factors,
        config,
        state,
    )
    result, right_smallest_S = do_right_absorption_structure_factor(
        peps_tensors,
        result,
        structure_factor_gates,
        structure_factor_outer_factors[3],
        structure_factor_inner_factors,
        config,
        state,
    )
    result, bottom_smallest_S = do_bottom_absorption_structure_factor(
        peps_tensors,
        result,
        structure_factor_gates,
        structure_factor_outer_factors[1],
        structure_factor_inner_factors,
        config,
        state,
    )
    norm_smallest_S = jnp.linalg.norm(
        jnp.asarray(
            left_smallest_S + top_smallest_S + right_smallest_S + bottom_smallest_S
        ),
        ord=jnp.inf,
    )
    return result, norm_smallest_S
