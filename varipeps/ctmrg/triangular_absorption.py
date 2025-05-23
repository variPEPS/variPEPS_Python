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
    calc_split_projectors_phase_1,
    calc_split_projectors_phase_2,
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


def decompose_new_T(new_T, chi, truncation_eps, config):
    new_T /= jnp.linalg.norm(new_T)

    new_T_matrix = new_T.reshape(
        new_T.shape[0] * new_T.shape[1] * new_T.shape[2],
        new_T.shape[3] * new_T.shape[4] * new_T.shape[5],
    )
    new_T_U, new_T_S, new_T_Vh = gauge_fixed_svd(new_T_matrix)
    new_T_S = jnp.where(
        new_T_S / new_T_S[0] >= truncation_eps,
        jnp.sqrt(jnp.where(new_T_S / new_T_S[0] >= truncation_eps, new_T_S, 1)),
        0,
    )

    new_Ta = new_T_S[:, jnp.newaxis] * new_T_Vh
    new_Ta = new_Ta.reshape(
        new_Ta.shape[0], new_T.shape[3], new_T.shape[4], new_T.shape[5]
    )
    new_Tb = new_T_U * new_T_S[jnp.newaxis, :]
    new_Tb = new_Tb.reshape(
        new_T.shape[0], new_T.shape[1], new_T.shape[2], new_Tb.shape[-1]
    )

    new_Ta_trunc = new_T_S[:chi, jnp.newaxis] * new_T_Vh[:chi, :]
    new_Ta_trunc = new_Ta_trunc.reshape(
        new_Ta_trunc.shape[0], new_T.shape[3], new_T.shape[4], new_T.shape[5]
    )
    new_Tb_trunc = new_T_U[:, :chi] * new_T_S[jnp.newaxis, :chi]
    new_Tb_trunc = new_Tb_trunc.reshape(
        new_T.shape[0], new_T.shape[1], new_T.shape[2], new_Tb_trunc.shape[-1]
    )

    return (
        _post_process_CTM_tensors(new_Ta, config),
        _post_process_CTM_tensors(new_Tb, config),
        _post_process_CTM_tensors(new_Ta_trunc, config),
        _post_process_CTM_tensors(new_Tb_trunc, config),
    )


def do_absorption_step_triangular(
    peps_tensors: Sequence[jnp.ndarray],
    unitcell: PEPS_Unit_Cell,
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> PEPS_Unit_Cell:
    if config.triangular_ctmrg_use_split:
        return do_absorption_step_triangular_split(
            peps_tensors, unitcell, config, state
        )

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
    new_T1a_trunc_list = []
    new_T1b_trunc_list = []
    new_T2a_trunc_list = []
    new_T2b_trunc_list = []
    new_T3a_trunc_list = []
    new_T3b_trunc_list = []
    new_T4a_trunc_list = []
    new_T4b_trunc_list = []
    new_T5a_trunc_list = []
    new_T5b_trunc_list = []
    new_T6a_trunc_list = []
    new_T6b_trunc_list = []
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
            new_T1a, new_T1b, new_T1a_trunc, new_T1b_trunc = decompose_new_T(
                new_T1, working_tensor_obj.chi, truncation_eps, config
            )
            new_T1a_list.append(new_T1a)
            new_T1b_list.append(new_T1b)
            new_T1a_trunc_list.append(new_T1a_trunc)
            new_T1b_trunc_list.append(new_T1b_trunc)

            T2_proj_left = corner_30_projectors.get_projector(x, y, -1, -1)[0]
            T2_proj_right = corner_30_projectors.get_projector(x, y, 0, 0)[1]
            new_T2 = apply_contraction_jitted(
                "triangular_ctmrg_T2_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T2_proj_left, T2_proj_right],
            )
            new_T2a, new_T2b, new_T2a_trunc, new_T2b_trunc = decompose_new_T(
                new_T2, working_tensor_obj.chi, truncation_eps, config
            )
            new_T2a_list.append(new_T2a)
            new_T2b_list.append(new_T2b)
            new_T2a_trunc_list.append(new_T2a_trunc)
            new_T2b_trunc_list.append(new_T2b_trunc)

            T3_proj_left = corner_330_projectors.get_projector(x, y, -1, 0)[0]
            T3_proj_right = corner_330_projectors.get_projector(x, y, 0, 0)[1]
            new_T3 = apply_contraction_jitted(
                "triangular_ctmrg_T3_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T3_proj_left, T3_proj_right],
            )
            new_T3a, new_T3b, new_T3a_trunc, new_T3b_trunc = decompose_new_T(
                new_T3, working_tensor_obj.chi, truncation_eps, config
            )
            new_T3a_list.append(new_T3a)
            new_T3b_list.append(new_T3b)
            new_T3a_trunc_list.append(new_T3a_trunc)
            new_T3b_trunc_list.append(new_T3b_trunc)

            T4_proj_left = corner_270_projectors.get_projector(x, y, 0, 0)[0]
            T4_proj_right = corner_270_projectors.get_projector(x, y, 0, -1)[1]
            new_T4 = apply_contraction_jitted(
                "triangular_ctmrg_T4_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T4_proj_left, T4_proj_right],
            )
            new_T4a, new_T4b, new_T4a_trunc, new_T4b_trunc = decompose_new_T(
                new_T4, working_tensor_obj.chi, truncation_eps, config
            )
            new_T4a_list.append(new_T4a)
            new_T4b_list.append(new_T4b)
            new_T4a_trunc_list.append(new_T4a_trunc)
            new_T4b_trunc_list.append(new_T4b_trunc)

            T5_proj_left = corner_210_projectors.get_projector(x, y, 0, 0)[0]
            T5_proj_right = corner_210_projectors.get_projector(x, y, -1, -1)[1]
            new_T5 = apply_contraction_jitted(
                "triangular_ctmrg_T5_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T5_proj_left, T5_proj_right],
            )
            new_T5a, new_T5b, new_T5a_trunc, new_T5b_trunc = decompose_new_T(
                new_T5, working_tensor_obj.chi, truncation_eps, config
            )
            new_T5a_list.append(new_T5a)
            new_T5b_list.append(new_T5b)
            new_T5a_trunc_list.append(new_T5a_trunc)
            new_T5b_trunc_list.append(new_T5b_trunc)

            T6_proj_left = corner_150_projectors.get_projector(x, y, 0, 0)[0]
            T6_proj_right = corner_150_projectors.get_projector(x, y, -1, 0)[1]
            new_T6 = apply_contraction_jitted(
                "triangular_ctmrg_T6_absorption",
                [working_tensor],
                [working_tensor_obj],
                [T6_proj_left, T6_proj_right],
            )
            new_T6a, new_T6b, new_T6a_trunc, new_T6b_trunc = decompose_new_T(
                new_T6, working_tensor_obj.chi, truncation_eps, config
            )
            new_T6a_list.append(new_T6a)
            new_T6b_list.append(new_T6b)
            new_T6a_trunc_list.append(new_T6a_trunc)
            new_T6b_trunc_list.append(new_T6b_trunc)

    i = 0
    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            new_t = view[1, 1][0][0].copy_including_trunc()
            new_t.C1 = new_C1_list[i]
            new_t.T1a = new_T1a_list[i]
            new_t.T6b = new_T6b_list[i]
            new_t.T1a_trunc = new_T1a_trunc_list[i]
            new_t.T6b_trunc = new_T6b_trunc_list[i]
            view[1, 1] = new_t

            new_t = view[1, 0][0][0].copy_including_trunc()
            new_t.C2 = new_C2_list[i]
            new_t.T1b = new_T1b_list[i]
            new_t.T2a = new_T2a_list[i]
            new_t.T1b_trunc = new_T1b_trunc_list[i]
            new_t.T2a_trunc = new_T2a_trunc_list[i]
            view[1, 0] = new_t

            new_t = view[0, -1][0][0].copy_including_trunc()
            new_t.C3 = new_C3_list[i]
            new_t.T2b = new_T2b_list[i]
            new_t.T3a = new_T3a_list[i]
            new_t.T2b_trunc = new_T2b_trunc_list[i]
            new_t.T3a_trunc = new_T3a_trunc_list[i]
            view[0, -1] = new_t

            new_t = view[-1, -1][0][0].copy_including_trunc()
            new_t.C4 = new_C4_list[i]
            new_t.T3b = new_T3b_list[i]
            new_t.T4a = new_T4a_list[i]
            new_t.T3b_trunc = new_T3b_trunc_list[i]
            new_t.T4a_trunc = new_T4a_trunc_list[i]
            view[-1, -1] = new_t

            new_t = view[-1, 0][0][0].copy_including_trunc()
            new_t.C5 = new_C5_list[i]
            new_t.T4b = new_T4b_list[i]
            new_t.T5a = new_T5a_list[i]
            new_t.T4b_trunc = new_T4b_trunc_list[i]
            new_t.T5a_trunc = new_T5a_trunc_list[i]
            view[-1, 0] = new_t

            new_t = view[0, 1][0][0].copy_including_trunc()
            new_t.C6 = new_C6_list[i]
            new_t.T5b = new_T5b_list[i]
            new_t.T6a = new_T6a_list[i]
            new_t.T5b_trunc = new_T5b_trunc_list[i]
            new_t.T6a_trunc = new_T6a_trunc_list[i]
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

            view[0, 0] = new_t

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
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


def do_absorption_step_triangular_split(
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

    corner_30_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_30_bra_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_90_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_90_bra_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_150_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_150_bra_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_210_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_210_bra_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_270_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_270_bra_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_330_ket_projectors = Projector_Dict_Triangular(
        view=unitcell, max_x=max_x, max_y=max_y
    )
    corner_330_bra_projectors = Projector_Dict_Triangular(
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
                proj_30_ket,
                proj_90_ket,
                proj_150_ket,
                proj_210_bra,
                proj_270_bra,
                proj_330_bra,
                smallest_S_corner,
            ) = calc_split_projectors_phase_1(
                *_get_triangular_ctmrg_2x2_structure(peps_tensors, view), config, state
            )

            corner_30_ket_projectors[(x, y)] = proj_30_ket
            corner_90_ket_projectors[(x, y)] = proj_90_ket
            corner_150_ket_projectors[(x, y)] = proj_150_ket
            corner_210_bra_projectors[(x, y)] = proj_210_bra
            corner_270_bra_projectors[(x, y)] = proj_270_bra
            corner_330_bra_projectors[(x, y)] = proj_330_bra

            smallest_S_list.append(smallest_S_corner)

    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            (
                proj_30_bra,
                proj_90_bra,
                proj_150_bra,
                proj_210_ket,
                proj_270_ket,
                proj_330_ket,
                smallest_S_corner,
            ) = calc_split_projectors_phase_2(
                *_get_triangular_ctmrg_2x2_structure(peps_tensors, view),
                *corner_30_ket_projectors.get_projector(x, y, 0, 0),
                *corner_90_ket_projectors.get_projector(x, y, 0, 0),
                *corner_150_ket_projectors.get_projector(x, y, 0, 0),
                *corner_210_bra_projectors.get_projector(x, y, 0, 0),
                *corner_270_bra_projectors.get_projector(x, y, 1, 0),
                *corner_330_bra_projectors.get_projector(x, y, 0, 1),
                config,
                state,
            )

            corner_30_bra_projectors[(x, y)] = proj_30_bra
            corner_90_bra_projectors[(x, y)] = proj_90_bra
            corner_150_bra_projectors[(x, y)] = proj_150_bra
            corner_210_ket_projectors[(x, y)] = proj_210_ket
            corner_270_ket_projectors[(x + 1, y)] = proj_270_ket
            corner_330_ket_projectors[(x, y + 1)] = proj_330_ket

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
    new_T1a_trunc_list = []
    new_T1b_trunc_list = []
    new_T2a_trunc_list = []
    new_T2b_trunc_list = []
    new_T3a_trunc_list = []
    new_T3b_trunc_list = []
    new_T4a_trunc_list = []
    new_T4b_trunc_list = []
    new_T5a_trunc_list = []
    new_T5b_trunc_list = []
    new_T6a_trunc_list = []
    new_T6b_trunc_list = []
    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
            working_tensor_obj = view[0, 0][0][0]

            C1_proj_left_bra = corner_150_bra_projectors.get_projector(x, y, 0, 0)[0]
            C1_proj_left_ket = corner_150_ket_projectors.get_projector(x, y, 0, 0)[0]
            C1_proj_right_ket = corner_90_ket_projectors.get_projector(x, y, 0, 0)[1]
            C1_proj_right_bra = corner_90_bra_projectors.get_projector(x, y, 0, 0)[1]
            new_C1 = apply_contraction_jitted(
                "triangular_ctmrg_split_C1_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C1_proj_left_bra,
                    C1_proj_left_ket,
                    C1_proj_right_ket,
                    C1_proj_right_bra,
                ],
            )
            new_C1_list.append(_post_process_CTM_tensors(new_C1, config))

            C2_proj_left_bra = corner_90_bra_projectors.get_projector(x, y, 0, -1)[0]
            C2_proj_left_ket = corner_90_ket_projectors.get_projector(x, y, 0, -1)[0]
            C2_proj_right_ket = corner_30_ket_projectors.get_projector(x, y, 0, 0)[1]
            C2_proj_right_bra = corner_30_bra_projectors.get_projector(x, y, 0, 0)[1]
            new_C2 = apply_contraction_jitted(
                "triangular_ctmrg_split_C2_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C2_proj_left_bra,
                    C2_proj_left_ket,
                    C2_proj_right_ket,
                    C2_proj_right_bra,
                ],
            )
            new_C2_list.append(_post_process_CTM_tensors(new_C2, config))

            C3_proj_left_bra = corner_30_bra_projectors.get_projector(x, y, -1, -1)[0]
            C3_proj_left_ket = corner_30_ket_projectors.get_projector(x, y, -1, -1)[0]
            C3_proj_right_bra = corner_330_bra_projectors.get_projector(x, y, 0, 0)[1]
            C3_proj_right_ket = corner_330_ket_projectors.get_projector(x, y, 0, 0)[1]
            new_C3 = apply_contraction_jitted(
                "triangular_ctmrg_split_C3_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C3_proj_left_bra,
                    C3_proj_left_ket,
                    C3_proj_right_bra,
                    C3_proj_right_ket,
                ],
            )
            new_C3_list.append(_post_process_CTM_tensors(new_C3, config))

            C4_proj_left_ket = corner_330_ket_projectors.get_projector(x, y, -1, 0)[0]
            C4_proj_left_bra = corner_330_bra_projectors.get_projector(x, y, -1, 0)[0]
            C4_proj_right_bra = corner_270_bra_projectors.get_projector(x, y, 0, -1)[1]
            C4_proj_right_ket = corner_270_ket_projectors.get_projector(x, y, 0, -1)[1]
            new_C4 = apply_contraction_jitted(
                "triangular_ctmrg_split_C4_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C4_proj_left_ket,
                    C4_proj_left_bra,
                    C4_proj_right_bra,
                    C4_proj_right_ket,
                ],
            )
            new_C4_list.append(_post_process_CTM_tensors(new_C4, config))

            C5_proj_left_ket = corner_270_ket_projectors.get_projector(x, y, 0, 0)[0]
            C5_proj_left_bra = corner_270_bra_projectors.get_projector(x, y, 0, 0)[0]
            C5_proj_right_bra = corner_210_bra_projectors.get_projector(x, y, -1, -1)[1]
            C5_proj_right_ket = corner_210_ket_projectors.get_projector(x, y, -1, -1)[1]
            new_C5 = apply_contraction_jitted(
                "triangular_ctmrg_split_C5_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C5_proj_left_ket,
                    C5_proj_left_bra,
                    C5_proj_right_bra,
                    C5_proj_right_ket,
                ],
            )
            new_C5_list.append(_post_process_CTM_tensors(new_C5, config))

            C6_proj_left_ket = corner_210_ket_projectors.get_projector(x, y, 0, 0)[0]
            C6_proj_left_bra = corner_210_bra_projectors.get_projector(x, y, 0, 0)[0]
            C6_proj_right_ket = corner_150_ket_projectors.get_projector(x, y, -1, 0)[1]
            C6_proj_right_bra = corner_150_bra_projectors.get_projector(x, y, -1, 0)[1]
            new_C6 = apply_contraction_jitted(
                "triangular_ctmrg_split_C6_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    C6_proj_left_ket,
                    C6_proj_left_bra,
                    C6_proj_right_ket,
                    C6_proj_right_bra,
                ],
            )
            new_C6_list.append(_post_process_CTM_tensors(new_C6, config))

            T1_proj_left_bra = corner_90_bra_projectors.get_projector(x, y, 0, -1)[0]
            T1_proj_left_ket = corner_90_ket_projectors.get_projector(x, y, 0, -1)[0]
            T1_proj_right_ket = corner_90_ket_projectors.get_projector(x, y, 0, 0)[1]
            T1_proj_right_bra = corner_90_bra_projectors.get_projector(x, y, 0, 0)[1]
            new_T1 = apply_contraction_jitted(
                "triangular_ctmrg_split_T1_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T1_proj_left_bra,
                    T1_proj_left_ket,
                    T1_proj_right_ket,
                    T1_proj_right_bra,
                ],
            )
            new_T1a, new_T1b, new_T1a_trunc, new_T1b_trunc = decompose_new_T(
                new_T1, working_tensor_obj.chi, truncation_eps, config
            )
            new_T1a_list.append(new_T1a)
            new_T1b_list.append(new_T1b)
            new_T1a_trunc_list.append(new_T1a_trunc)
            new_T1b_trunc_list.append(new_T1b_trunc)

            T2_proj_left_bra = corner_30_bra_projectors.get_projector(x, y, -1, -1)[0]
            T2_proj_left_ket = corner_30_ket_projectors.get_projector(x, y, -1, -1)[0]
            T2_proj_right_ket = corner_30_ket_projectors.get_projector(x, y, 0, 0)[1]
            T2_proj_right_bra = corner_30_bra_projectors.get_projector(x, y, 0, 0)[1]
            new_T2 = apply_contraction_jitted(
                "triangular_ctmrg_split_T2_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T2_proj_left_bra,
                    T2_proj_left_ket,
                    T2_proj_right_ket,
                    T2_proj_right_bra,
                ],
            )
            new_T2a, new_T2b, new_T2a_trunc, new_T2b_trunc = decompose_new_T(
                new_T2, working_tensor_obj.chi, truncation_eps, config
            )
            new_T2a_list.append(new_T2a)
            new_T2b_list.append(new_T2b)
            new_T2a_trunc_list.append(new_T2a_trunc)
            new_T2b_trunc_list.append(new_T2b_trunc)

            T3_proj_left_ket = corner_330_ket_projectors.get_projector(x, y, -1, 0)[0]
            T3_proj_left_bra = corner_330_bra_projectors.get_projector(x, y, -1, 0)[0]
            T3_proj_right_bra = corner_330_bra_projectors.get_projector(x, y, 0, 0)[1]
            T3_proj_right_ket = corner_330_ket_projectors.get_projector(x, y, 0, 0)[1]
            new_T3 = apply_contraction_jitted(
                "triangular_ctmrg_split_T3_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T3_proj_left_ket,
                    T3_proj_left_bra,
                    T3_proj_right_bra,
                    T3_proj_right_ket,
                ],
            )
            new_T3a, new_T3b, new_T3a_trunc, new_T3b_trunc = decompose_new_T(
                new_T3, working_tensor_obj.chi, truncation_eps, config
            )
            new_T3a_list.append(new_T3a)
            new_T3b_list.append(new_T3b)
            new_T3a_trunc_list.append(new_T3a_trunc)
            new_T3b_trunc_list.append(new_T3b_trunc)

            T4_proj_left_ket = corner_270_ket_projectors.get_projector(x, y, 0, 0)[0]
            T4_proj_left_bra = corner_270_bra_projectors.get_projector(x, y, 0, 0)[0]
            T4_proj_right_bra = corner_270_bra_projectors.get_projector(x, y, 0, -1)[1]
            T4_proj_right_ket = corner_270_ket_projectors.get_projector(x, y, 0, -1)[1]
            new_T4 = apply_contraction_jitted(
                "triangular_ctmrg_split_T4_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T4_proj_left_ket,
                    T4_proj_left_bra,
                    T4_proj_right_bra,
                    T4_proj_right_ket,
                ],
            )
            new_T4a, new_T4b, new_T4a_trunc, new_T4b_trunc = decompose_new_T(
                new_T4, working_tensor_obj.chi, truncation_eps, config
            )
            new_T4a_list.append(new_T4a)
            new_T4b_list.append(new_T4b)
            new_T4a_trunc_list.append(new_T4a_trunc)
            new_T4b_trunc_list.append(new_T4b_trunc)

            T5_proj_left_ket = corner_210_ket_projectors.get_projector(x, y, 0, 0)[0]
            T5_proj_left_bra = corner_210_bra_projectors.get_projector(x, y, 0, 0)[0]
            T5_proj_right_bra = corner_210_bra_projectors.get_projector(x, y, -1, -1)[1]
            T5_proj_right_ket = corner_210_ket_projectors.get_projector(x, y, -1, -1)[1]
            new_T5 = apply_contraction_jitted(
                "triangular_ctmrg_split_T5_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T5_proj_left_ket,
                    T5_proj_left_bra,
                    T5_proj_right_bra,
                    T5_proj_right_ket,
                ],
            )
            new_T5a, new_T5b, new_T5a_trunc, new_T5b_trunc = decompose_new_T(
                new_T5, working_tensor_obj.chi, truncation_eps, config
            )
            new_T5a_list.append(new_T5a)
            new_T5b_list.append(new_T5b)
            new_T5a_trunc_list.append(new_T5a_trunc)
            new_T5b_trunc_list.append(new_T5b_trunc)

            T6_proj_left_bra = corner_150_bra_projectors.get_projector(x, y, 0, 0)[0]
            T6_proj_left_ket = corner_150_ket_projectors.get_projector(x, y, 0, 0)[0]
            T6_proj_right_ket = corner_150_ket_projectors.get_projector(x, y, -1, 0)[1]
            T6_proj_right_bra = corner_150_bra_projectors.get_projector(x, y, -1, 0)[1]
            new_T6 = apply_contraction_jitted(
                "triangular_ctmrg_split_T6_absorption",
                [working_tensor],
                [working_tensor_obj],
                [
                    T6_proj_left_bra,
                    T6_proj_left_ket,
                    T6_proj_right_ket,
                    T6_proj_right_bra,
                ],
            )
            new_T6a, new_T6b, new_T6a_trunc, new_T6b_trunc = decompose_new_T(
                new_T6, working_tensor_obj.chi, truncation_eps, config
            )
            new_T6a_list.append(new_T6a)
            new_T6b_list.append(new_T6b)
            new_T6a_trunc_list.append(new_T6a_trunc)
            new_T6b_trunc_list.append(new_T6b_trunc)

    i = 0
    for x, iter_rows in working_unitcell.iter_all_rows(only_unique=True):
        for y, view in iter_rows:
            new_t = view[1, 1][0][0].copy_including_trunc()
            new_t.C1 = new_C1_list[i]
            new_t.T1a = new_T1a_list[i]
            new_t.T6b = new_T6b_list[i]
            new_t.T1a_trunc = new_T1a_trunc_list[i]
            new_t.T6b_trunc = new_T6b_trunc_list[i]
            view[1, 1] = new_t

            new_t = view[1, 0][0][0].copy_including_trunc()
            new_t.C2 = new_C2_list[i]
            new_t.T1b = new_T1b_list[i]
            new_t.T2a = new_T2a_list[i]
            new_t.T1b_trunc = new_T1b_trunc_list[i]
            new_t.T2a_trunc = new_T2a_trunc_list[i]
            view[1, 0] = new_t

            new_t = view[0, -1][0][0].copy_including_trunc()
            new_t.C3 = new_C3_list[i]
            new_t.T2b = new_T2b_list[i]
            new_t.T3a = new_T3a_list[i]
            new_t.T2b_trunc = new_T2b_trunc_list[i]
            new_t.T3a_trunc = new_T3a_trunc_list[i]
            view[0, -1] = new_t

            new_t = view[-1, -1][0][0].copy_including_trunc()
            new_t.C4 = new_C4_list[i]
            new_t.T3b = new_T3b_list[i]
            new_t.T4a = new_T4a_list[i]
            new_t.T3b_trunc = new_T3b_trunc_list[i]
            new_t.T4a_trunc = new_T4a_trunc_list[i]
            view[-1, -1] = new_t

            new_t = view[-1, 0][0][0].copy_including_trunc()
            new_t.C5 = new_C5_list[i]
            new_t.T4b = new_T4b_list[i]
            new_t.T5a = new_T5a_list[i]
            new_t.T4b_trunc = new_T4b_trunc_list[i]
            new_t.T5a_trunc = new_T5a_trunc_list[i]
            view[-1, 0] = new_t

            new_t = view[0, 1][0][0].copy_including_trunc()
            new_t.C6 = new_C6_list[i]
            new_t.T5b = new_T5b_list[i]
            new_t.T6a = new_T6a_list[i]
            new_t.T5b_trunc = new_T5b_trunc_list[i]
            new_t.T6a_trunc = new_T6a_trunc_list[i]
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
