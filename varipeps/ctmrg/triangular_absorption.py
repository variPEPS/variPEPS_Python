import jax
import jax.numpy as jnp

from varipeps.peps import PEPS_Tensor_Triangular, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction_jitted
from varipeps.utils.svd import gauge_fixed_svd
from varipeps.utils.projector_dict import Projector_Dict_Triangular
from varipeps.config import VariPEPS_Config
from varipeps.global_state import VariPEPS_Global_State

from .absorption import _tensor_list_from_indices, _post_process_CTM_tensors
from .triangular_projectors import (
    calc_corner_projectors,
    calc_T_30_150_270_projectors,
    calc_T_90_210_330_projectors,
)

from typing import Sequence, Tuple, List


def _get_triangular_ctmrg_2x2_structure(
    peps_tensors: Sequence[jnp.ndarray],
    view: PEPS_Unit_Cell,
) -> Tuple[List[List[jnp.ndarray]], List[List[PEPS_Tensor_Triangular]]]:
    x_slice = slice(0, 2, None)
    y_slice = slice(0, 2, None)

    indices = view.get_indices((x_slice, y_slice))
    view_tensors = _tensor_list_from_indices(peps_tensors, indices)

    view_tensor_objs = view[x_slice, y_slice]

    return view_tensors, view_tensor_objs


def do_absorption_step_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    max_x, max_y = unitcell.get_size()

    truncation_eps = (
        config.ctmrg_truncation_eps
        if state.ctmrg_effective_truncation_eps is None
        else state.ctmrg_effective_truncation_eps
    )

    corner_30_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_90_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_150_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_210_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_270_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_330_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )

    T_30_projectors = Projector_Dict_Triangular(view=unitcell, max_x=max_x, max_y=max_y)
    T_90_projectors = Projector_Dict_Triangular(view=unitcell, max_x=max_x, max_y=max_y)
    T_150_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    T_210_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    T_270_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    T_330_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )

    smallest_S_list = []

    working_unitcell = unitcell.copy()

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            (
                proj_30,
                proj_90,
                proj_150,
                proj_210,
                proj_270,
                proj_330,
                smallest_S_corner,
            ) = calc_corner_projectors(
                *_get_triangular_ctmrg_2x2_structure(peps_tensors, view), config, state
            )

            corner_30_projectors[(x, y)] = proj_30
            corner_90_projectors[(x, y)] = proj_90
            corner_150_projectors[(x, y)] = proj_150
            corner_210_projectors[(x, y)] = proj_210
            corner_270_projectors[(x + 1, y)] = proj_270
            corner_330_projectors[(x, y + 1)] = proj_330

            smallest_S_list.append(smallest_S_corner)

    new_C1_list = []
    new_C2_list = []
    new_C3_list = []
    new_C4_list = []
    new_C5_list = []
    new_C6_list = []
    new_T1a_list = []
    new_T1b_list = []
    new_T2a_list = []
    new_T2b_list = []
    new_T3a_list = []
    new_T3b_list = []
    new_T4a_list = []
    new_T4b_list = []
    new_T5a_list = []
    new_T5b_list = []
    new_T6a_list = []
    new_T6b_list = []
    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_proj_left = corner_150_projectors.get_projector(x, y, 0, 0)[0]
            C1_proj_right = corner_90_projectors.get_projector(x, y, 0, 0)[1]
            new_C1 = apply_contraction_jitted(
                "triangular_ctmrg_C1_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C1_proj_left, C1_proj_right],
            )
            new_C1_list.append(_post_process_CTM_tensors(new_C1, config))

            C2_proj_left = corner_90_projectors.get_projector(x, y, 0, -1)[0]
            C2_proj_right = corner_30_projectors.get_projector(x, y, 0, 0)[1]
            new_C2 = apply_contraction_jitted(
                "triangular_ctmrg_C2_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C2_proj_left, C2_proj_right],
            )
            new_C2_list.append(_post_process_CTM_tensors(new_C2, config))

            C3_proj_left = corner_30_projectors.get_projector(x, y, -1, -1)[0]
            C3_proj_right = corner_330_projectors.get_projector(x, y, 0, 0)[1]
            new_C3 = apply_contraction_jitted(
                "triangular_ctmrg_C3_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C3_proj_left, C3_proj_right],
            )
            new_C3_list.append(_post_process_CTM_tensors(new_C3, config))

            C4_proj_left = corner_330_projectors.get_projector(x, y, -1, 0)[0]
            C4_proj_right = corner_270_projectors.get_projector(x, y, 0, -1)[1]
            new_C4 = apply_contraction_jitted(
                "triangular_ctmrg_C4_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C4_proj_left, C4_proj_right],
            )
            new_C4_list.append(_post_process_CTM_tensors(new_C4, config))

            C5_proj_left = corner_270_projectors.get_projector(x, y, 0, 0)[0]
            C5_proj_right = corner_210_projectors.get_projector(x, y, -1, -1)[1]
            new_C5 = apply_contraction_jitted(
                "triangular_ctmrg_C5_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C5_proj_left, C5_proj_right],
            )
            new_C5_list.append(_post_process_CTM_tensors(new_C5, config))

            C6_proj_left = corner_210_projectors.get_projector(x, y, 0, 0)[0]
            C6_proj_right = corner_150_projectors.get_projector(x, y, -1, 0)[1]
            new_C6 = apply_contraction_jitted(
                "triangular_ctmrg_C6_absorption",
                [working_tensor],
                [working_tensor_obj],
                [C6_proj_left, C6_proj_right],
            )
            new_C6_list.append(_post_process_CTM_tensors(new_C6, config))

            T1_proj_left = corner_90_projectors.get_projector(x, y, 0, -1)[0]
            T1_proj_right = corner_90_projectors.get_projector(x, y, 0, 0)[1]
            new_T1 = apply_contraction_jitted(
                "triangular_ctmrg_T1_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T1_proj_left, T1_proj_right],
            )
            new_T1 /= jnp.linalg.norm(new_T1)

            new_T1_matrix = new_T1.reshape(
                new_T1.shape[0] * new_T1.shape[1] * new_T1.shape[2],
                new_T1.shape[3] * new_T1.shape[4] * new_T1.shape[5],
            )
            new_T1_U, new_T1_S, new_T1_Vh = gauge_fixed_svd(new_T1_matrix)
            new_T1_S = jnp.where(
                new_T1_S / new_T1_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T1_S / new_T1_S[0] >= truncation_eps, new_T1_S, 1)
                ),
                0,
            )
            new_T1a = new_T1_S[:, jnp.newaxis] * new_T1_Vh
            new_T1a = new_T1a.reshape(
                new_T1a.shape[0], new_T1.shape[3], new_T1.shape[4], new_T1.shape[5]
            )
            new_T1b = new_T1_U * new_T1_S[jnp.newaxis, :]
            new_T1b = new_T1b.reshape(
                new_T1.shape[0], new_T1.shape[1], new_T1.shape[2], new_T1b.shape[-1]
            )
            new_T1a_list.append(_post_process_CTM_tensors(new_T1a, config))
            new_T1b_list.append(_post_process_CTM_tensors(new_T1b, config))

            T2_proj_left = corner_30_projectors.get_projector(x, y, -1, -1)[0]
            T2_proj_right = corner_30_projectors.get_projector(x, y, 0, 0)[1]
            new_T2 = apply_contraction_jitted(
                "triangular_ctmrg_T2_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T2_proj_left, T2_proj_right],
            )
            new_T2 /= jnp.linalg.norm(new_T2)

            new_T2_matrix = new_T2.reshape(
                new_T2.shape[0] * new_T2.shape[1] * new_T2.shape[2],
                new_T2.shape[3] * new_T2.shape[4] * new_T2.shape[5],
            )
            new_T2_U, new_T2_S, new_T2_Vh = gauge_fixed_svd(new_T2_matrix)
            new_T2_S = jnp.where(
                new_T2_S / new_T2_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T2_S / new_T2_S[0] >= truncation_eps, new_T2_S, 1)
                ),
                0,
            )
            new_T2a = new_T2_S[:, jnp.newaxis] * new_T2_Vh
            new_T2a = new_T2a.reshape(
                new_T2a.shape[0], new_T2.shape[3], new_T2.shape[4], new_T2.shape[5]
            )
            new_T2b = new_T2_U * new_T2_S[jnp.newaxis, :]
            new_T2b = new_T2b.reshape(
                new_T2.shape[0], new_T2.shape[1], new_T2.shape[2], new_T2b.shape[-1]
            )
            new_T2a_list.append(_post_process_CTM_tensors(new_T2a, config))
            new_T2b_list.append(_post_process_CTM_tensors(new_T2b, config))

            T3_proj_left = corner_330_projectors.get_projector(x, y, -1, 0)[0]
            T3_proj_right = corner_330_projectors.get_projector(x, y, 0, 0)[1]
            new_T3 = apply_contraction_jitted(
                "triangular_ctmrg_T3_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T3_proj_left, T3_proj_right],
            )
            new_T3 /= jnp.linalg.norm(new_T3)

            new_T3_matrix = new_T3.reshape(
                new_T3.shape[0] * new_T3.shape[1] * new_T3.shape[2],
                new_T3.shape[3] * new_T3.shape[4] * new_T3.shape[5],
            )
            new_T3_U, new_T3_S, new_T3_Vh = gauge_fixed_svd(new_T3_matrix)
            new_T3_S = jnp.where(
                new_T3_S / new_T3_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T3_S / new_T3_S[0] >= truncation_eps, new_T3_S, 1)
                ),
                0,
            )
            new_T3a = new_T3_S[:, jnp.newaxis] * new_T3_Vh
            new_T3a = new_T3a.reshape(
                new_T3a.shape[0], new_T3.shape[3], new_T3.shape[4], new_T3.shape[5]
            )
            new_T3b = new_T3_U * new_T3_S[jnp.newaxis, :]
            new_T3b = new_T3b.reshape(
                new_T3.shape[0], new_T3.shape[1], new_T3.shape[2], new_T3b.shape[-1]
            )
            new_T3a_list.append(_post_process_CTM_tensors(new_T3a, config))
            new_T3b_list.append(_post_process_CTM_tensors(new_T3b, config))

            T4_proj_left = corner_270_projectors.get_projector(x, y, 0, 0)[0]
            T4_proj_right = corner_270_projectors.get_projector(x, y, 0, -1)[1]
            new_T4 = apply_contraction_jitted(
                "triangular_ctmrg_T4_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T4_proj_left, T4_proj_right],
            )
            new_T4 /= jnp.linalg.norm(new_T4)

            new_T4_matrix = new_T4.reshape(
                new_T4.shape[0] * new_T4.shape[1] * new_T4.shape[2],
                new_T4.shape[3] * new_T4.shape[4] * new_T4.shape[5],
            )
            new_T4_U, new_T4_S, new_T4_Vh = gauge_fixed_svd(new_T4_matrix)
            new_T4_S = jnp.where(
                new_T4_S / new_T4_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T4_S / new_T4_S[0] >= truncation_eps, new_T4_S, 1)
                ),
                0,
            )
            new_T4a = new_T4_S[:, jnp.newaxis] * new_T4_Vh
            new_T4a = new_T4a.reshape(
                new_T4a.shape[0], new_T4.shape[3], new_T4.shape[4], new_T4.shape[5]
            )
            new_T4b = new_T4_U * new_T4_S[jnp.newaxis, :]
            new_T4b = new_T4b.reshape(
                new_T4.shape[0], new_T4.shape[1], new_T4.shape[2], new_T4b.shape[-1]
            )
            new_T4a_list.append(_post_process_CTM_tensors(new_T4a, config))
            new_T4b_list.append(_post_process_CTM_tensors(new_T4b, config))

            T5_proj_left = corner_210_projectors.get_projector(x, y, 0, 0)[0]
            T5_proj_right = corner_210_projectors.get_projector(x, y, -1, -1)[1]
            new_T5 = apply_contraction_jitted(
                "triangular_ctmrg_T5_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T5_proj_left, T5_proj_right],
            )
            new_T5 /= jnp.linalg.norm(new_T5)

            new_T5_matrix = new_T5.reshape(
                new_T5.shape[0] * new_T5.shape[1] * new_T5.shape[2],
                new_T5.shape[3] * new_T5.shape[4] * new_T5.shape[5],
            )
            new_T5_U, new_T5_S, new_T5_Vh = gauge_fixed_svd(new_T5_matrix)
            new_T5_S = jnp.where(
                new_T5_S / new_T5_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T5_S / new_T5_S[0] >= truncation_eps, new_T5_S, 1)
                ),
                0,
            )
            new_T5a = new_T5_S[:, jnp.newaxis] * new_T5_Vh
            new_T5a = new_T5a.reshape(
                new_T5a.shape[0], new_T5.shape[3], new_T5.shape[4], new_T5.shape[5]
            )
            new_T5b = new_T5_U * new_T5_S[jnp.newaxis, :]
            new_T5b = new_T5b.reshape(
                new_T5.shape[0], new_T5.shape[1], new_T5.shape[2], new_T5b.shape[-1]
            )
            new_T5a_list.append(_post_process_CTM_tensors(new_T5a, config))
            new_T5b_list.append(_post_process_CTM_tensors(new_T5b, config))

            T6_proj_left = corner_150_projectors.get_projector(x, y, 0, 0)[0]
            T6_proj_right = corner_150_projectors.get_projector(x, y, -1, 0)[1]
            new_T6 = apply_contraction_jitted(
                "triangular_ctmrg_T6_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T6_proj_left, T6_proj_right],
            )
            new_T6 /= jnp.linalg.norm(new_T6)

            new_T6_matrix = new_T6.reshape(
                new_T6.shape[0] * new_T6.shape[1] * new_T6.shape[2],
                new_T6.shape[3] * new_T6.shape[4] * new_T6.shape[5],
            )
            new_T6_U, new_T6_S, new_T6_Vh = gauge_fixed_svd(new_T6_matrix)
            new_T6_S = jnp.where(
                new_T6_S / new_T6_S[0] >= truncation_eps,
                jnp.sqrt(
                    jnp.where(new_T6_S / new_T6_S[0] >= truncation_eps, new_T6_S, 1)
                ),
                0,
            )
            new_T6a = new_T6_S[:, jnp.newaxis] * new_T6_Vh
            new_T6a = new_T6a.reshape(
                new_T6a.shape[0], new_T6.shape[3], new_T6.shape[4], new_T6.shape[5]
            )
            new_T6b = new_T6_U * new_T6_S[jnp.newaxis, :]
            new_T6b = new_T6b.reshape(
                new_T6.shape[0], new_T6.shape[1], new_T6.shape[2], new_T6b.shape[-1]
            )
            new_T6a_list.append(_post_process_CTM_tensors(new_T6a, config))
            new_T6b_list.append(_post_process_CTM_tensors(new_T6b, config))

    i = 0
    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            new_t = view[1, 1][0][0].copy()
            new_t.C1 = new_C1_list[i]
            new_t.T1a = new_T1a_list[i]
            new_t.T6b = new_T6b_list[i]
            view[1, 1] = new_t

            new_t = view[1, 0][0][0].copy()
            new_t.C2 = new_C2_list[i]
            new_t.T1b = new_T1b_list[i]
            new_t.T2a = new_T2a_list[i]
            view[1, 0] = new_t

            new_t = view[0, -1][0][0].copy()
            new_t.C3 = new_C3_list[i]
            new_t.T2b = new_T2b_list[i]
            new_t.T3a = new_T3a_list[i]
            view[0, -1] = new_t

            new_t = view[-1, -1][0][0].copy()
            new_t.C4 = new_C4_list[i]
            new_t.T3b = new_T3b_list[i]
            new_t.T4a = new_T4a_list[i]
            view[-1, -1] = new_t

            new_t = view[-1, 0][0][0].copy()
            new_t.C5 = new_C5_list[i]
            new_t.T4b = new_T4b_list[i]
            new_t.T5a = new_T5a_list[i]
            view[-1, 0] = new_t

            new_t = view[0, 1][0][0].copy()
            new_t.C6 = new_C6_list[i]
            new_t.T5b = new_T5b_list[i]
            new_t.T6a = new_T6a_list[i]
            view[0, 1] = new_t

            i += 1

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            proj_90, proj_210, proj_330, smallest_S_T = calc_T_90_210_330_projectors(
                *_get_triangular_ctmrg_2x2_structure(peps_tensors, view), config, state
            )

            T_90_projectors[(x, y)] = proj_90
            T_210_projectors[(x, y)] = proj_210
            T_330_projectors[(x, y)] = proj_330

            smallest_S_list.append(smallest_S_T)

            proj_30, proj_150, proj_270, smallest_S_T = calc_T_30_150_270_projectors(
                *_get_triangular_ctmrg_2x2_structure(peps_tensors, view), config, state
            )

            T_30_projectors[(x, y)] = proj_30
            T_150_projectors[(x, y)] = proj_150
            T_270_projectors[(x, y)] = proj_270

            smallest_S_list.append(smallest_S_T)

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            new_t = view[0, 0][0][0].copy()

            new_t.T1a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_90_projectors.get_projector(x, y, 0, -1)[0],
                    new_t.T1a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T1b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T1b,
                    T_90_projectors.get_projector(x, y, 0, 0)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            new_t.T2a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_30_projectors.get_projector(x, y, -1, -1)[0],
                    new_t.T2a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T2b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T2b,
                    T_30_projectors.get_projector(x, y, 0, 0)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            new_t.T3a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_330_projectors.get_projector(x, y, -1, 0)[0],
                    new_t.T3a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T3b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T3b,
                    T_330_projectors.get_projector(x, y, 0, 0)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            new_t.T4a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_270_projectors.get_projector(x, y, 0, 0)[0],
                    new_t.T4a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T4b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T4b,
                    T_270_projectors.get_projector(x, y, 0, -1)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            new_t.T5a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_210_projectors.get_projector(x, y, 0, 0)[0],
                    new_t.T5a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T5b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T5b,
                    T_210_projectors.get_projector(x, y, -1, -1)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            new_t.T6a = _post_process_CTM_tensors(
                jnp.tensordot(
                    T_150_projectors.get_projector(x, y, 0, 0)[0],
                    new_t.T6a,
                    ((1,), (0,)),
                ),
                config,
            )
            new_t.T6b = _post_process_CTM_tensors(
                jnp.tensordot(
                    new_t.T6b,
                    T_150_projectors.get_projector(x, y, -1, 0)[1],
                    ((3,), (0,)),
                ),
                config,
            )

            view[0, 0] = new_t

    smallest_S = jnp.linalg.norm(
        jnp.asarray([j for i in smallest_S_list for j in i]), ord=jnp.inf
    )

    return working_unitcell, smallest_S
