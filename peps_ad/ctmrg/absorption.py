import collections.abc
from dataclasses import dataclass, field
from operator import itemgetter

import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor, PEPS_Unit_Cell
from peps_ad.contractions import apply_contraction
from .projectors import (
    calc_left_projectors,
    calc_right_projectors,
    calc_top_projectors,
    calc_bottom_projectors,
    T_Projector,
)

from typing import Sequence, Tuple, List, Dict, Literal

CTMRG_Orientation = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


@dataclass
class _Projector_Dict(collections.abc.MutableMapping):
    max_x: int
    max_y: int
    projector_dict: Dict[Tuple[int, int], T_Projector] = field(default_factory=dict)  # type: ignore

    def __getitem__(self, key: Tuple[int, int]) -> T_Projector:
        return self.projector_dict[key]

    def __setitem__(self, key: Tuple[int, int], value: T_Projector) -> None:
        self.projector_dict[key] = value

    def __delitem__(self, key: Tuple[int, int]) -> None:
        self.projector_dict.__delitem__(key)

    def __iter__(self):
        return self.projector_dict.__iter__()

    def __len__(self):
        return self.projector_dict.__len__()

    def get_projector(
        self,
        current_x: int,
        current_y: int,
        relative_x: int,
        relative_y: int,
    ) -> T_Projector:
        select_x = (current_x + relative_x) % self.max_x
        select_y = (current_y + relative_y) % self.max_y

        return self.projector_dict[(select_x, select_y)]


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


def do_left_absorption(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    """
    Calculate the left CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_cell`:
        New instance of the unitcell with the updated left CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    left_projectors = _Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    for y, iter_columns in working_unitcell.iter_all_columns():
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            left_projectors[(x, y)] = calc_left_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left")
            )

        new_C1 = []
        new_T4 = []
        new_C4 = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_projector = left_projectors.get_projector(x, y, -1, 0).bottom
            new_C1_tmp = apply_contraction(
                "ctmrg_absorption_left_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1.append(new_C1_tmp / jnp.linalg.norm(new_C1_tmp))

            T4_projector_top = left_projectors.get_projector(x, y, -1, 0).top
            T4_projector_bottom = left_projectors.get_projector(x, y, 0, 0).bottom
            new_T4_tmp = apply_contraction(
                "ctmrg_absorption_left_T4",
                [working_tensor],
                [working_tensor_obj],
                [T4_projector_top, T4_projector_bottom],
            )
            new_T4.append(new_T4_tmp / jnp.linalg.norm(new_T4_tmp))

            C4_projector = left_projectors.get_projector(x, y, 0, 0).top
            new_C4_tmp = apply_contraction(
                "ctmrg_absorption_left_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4.append(new_C4_tmp / jnp.linalg.norm(new_C4_tmp))

        for x, view in column_views:
            view[0, 1] = view[0, 1][0][0].replace_left_env_tensors(
                new_C1[x], new_T4[x], new_C4[x]
            )

    return working_unitcell


def do_right_absorption(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    """
    Calculate the right CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_cell`:
        New instance of the unitcell with the updated right CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    right_projectors = _Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    for y, iter_columns in working_unitcell.iter_all_columns(reverse=True):
        column_views = [view for view in iter_columns]

        for x, view in column_views:
            right_projectors[(x, y)] = calc_right_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-right")
            )

        new_C2 = []
        new_T2 = []
        new_C3 = []

        for x, view in column_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C2_projector = right_projectors.get_projector(x, y, -1, 0).bottom
            new_C2_tmp = apply_contraction(
                "ctmrg_absorption_right_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2.append(new_C2_tmp / jnp.linalg.norm(new_C2_tmp))

            T2_projector_top = right_projectors.get_projector(x, y, -1, 0).top
            T2_projector_bottom = right_projectors.get_projector(x, y, 0, 0).bottom
            new_T2_tmp = apply_contraction(
                "ctmrg_absorption_right_T2",
                [working_tensor],
                [working_tensor_obj],
                [T2_projector_top, T2_projector_bottom],
            )
            new_T2.append(new_T2_tmp / jnp.linalg.norm(new_T2_tmp))

            C3_projector = right_projectors.get_projector(x, y, 0, 0).top
            new_C3_tmp = apply_contraction(
                "ctmrg_absorption_right_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3.append(new_C3_tmp / jnp.linalg.norm(new_C3_tmp))

        for x, view in column_views:
            view[0, -1] = view[0, -1][0][0].replace_right_env_tensors(
                new_C2[x], new_T2[x], new_C3[x]
            )

    return working_unitcell


def do_top_absorption(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    """
    Calculate the top CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_cell`:
        New instance of the unitcell with the updated top CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    top_projectors = _Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    for x, iter_rows in working_unitcell.iter_all_rows():
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            top_projectors[(x, y)] = calc_top_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "top-left")
            )

        new_C1 = []
        new_T1 = []
        new_C2 = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_projector = top_projectors.get_projector(x, y, 0, -1).right
            new_C1_tmp = apply_contraction(
                "ctmrg_absorption_top_C1",
                [working_tensor],
                [working_tensor_obj],
                [C1_projector],
            )
            new_C1.append(new_C1_tmp / jnp.linalg.norm(new_C1_tmp))

            T1_projector_left = top_projectors.get_projector(x, y, 0, -1).left
            T1_projector_right = top_projectors.get_projector(x, y, 0, 0).right
            new_T1_tmp = apply_contraction(
                "ctmrg_absorption_top_T1",
                [working_tensor],
                [working_tensor_obj],
                [T1_projector_left, T1_projector_right],
            )
            new_T1.append(new_T1_tmp / jnp.linalg.norm(new_T1_tmp))

            C2_projector = top_projectors.get_projector(x, y, 0, 0).left
            new_C2_tmp = apply_contraction(
                "ctmrg_absorption_top_C2",
                [working_tensor],
                [working_tensor_obj],
                [C2_projector],
            )
            new_C2.append(new_C2_tmp / jnp.linalg.norm(new_C2_tmp))

        for y, view in row_views:
            view[1, 0] = view[1, 0][0][0].replace_top_env_tensors(
                new_C1[y], new_T1[y], new_C2[y]
            )

    return working_unitcell


def do_bottom_absorption(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    """
    Calculate the bottom CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_cell`:
        New instance of the unitcell with the updated bottom CTMRG tensors of
        all elements of the unitcell.
    """
    max_x, max_y = unitcell.get_size()
    bottom_projectors = _Projector_Dict(max_x=max_x, max_y=max_y)

    working_unitcell = unitcell.copy()

    for x, iter_rows in working_unitcell.iter_all_rows(reverse=True):
        row_views = [view for view in iter_rows]

        for y, view in row_views:
            bottom_projectors[(x, y)] = calc_bottom_projectors(
                *_get_ctmrg_2x2_structure(peps_tensors, view, "bottom-left")
            )

        new_C4 = []
        new_T3 = []
        new_C3 = []

        for y, view in row_views:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C4_projector = bottom_projectors.get_projector(x, y, 0, -1).right
            new_C4_tmp = apply_contraction(
                "ctmrg_absorption_bottom_C4",
                [working_tensor],
                [working_tensor_obj],
                [C4_projector],
            )
            new_C4.append(new_C4_tmp / jnp.linalg.norm(new_C4_tmp))

            T3_projector_left = bottom_projectors.get_projector(x, y, 0, -1).left
            T3_projector_right = bottom_projectors.get_projector(x, y, 0, 0).right
            new_T3_tmp = apply_contraction(
                "ctmrg_absorption_bottom_T3",
                [working_tensor],
                [working_tensor_obj],
                [T3_projector_left, T3_projector_right],
            )
            new_T3.append(new_T3_tmp / jnp.linalg.norm(new_T3_tmp))

            C3_projector = bottom_projectors.get_projector(x, y, 0, 0).left
            new_C3_tmp = apply_contraction(
                "ctmrg_absorption_bottom_C3",
                [working_tensor],
                [working_tensor_obj],
                [C3_projector],
            )
            new_C3.append(new_C3_tmp / jnp.linalg.norm(new_C3_tmp))

        for y, view in row_views:
            view[-1, 0] = view[-1, 0][0][0].replace_bottom_env_tensors(
                new_C4[y], new_T3[y], new_C3[y]
            )

    return working_unitcell


def do_absorption_step(
    peps_tensors: Sequence[jnp.ndarray], unitcell: PEPS_Unit_Cell
) -> PEPS_Unit_Cell:
    """
    Calculate the all CTMRG tensors after one absorption step and returns
    the updated unitcell.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The sequence of unique PEPS tensors the unitcell consists of.
      unitcell (:obj:`~peps_ad.peps.PEPS_Unit_cell`):
        The unitcell to work on.
    Returns:
      :obj:`~peps_ad.peps.PEPS_Unit_cell`:
        New instance of the unitcell with the all updated CTMRG tensors of
        all elements of the unitcell.
    """
    result = do_left_absorption(peps_tensors, unitcell)
    result = do_top_absorption(peps_tensors, result)
    result = do_right_absorption(peps_tensors, result)
    result = do_bottom_absorption(peps_tensors, result)
    return result
