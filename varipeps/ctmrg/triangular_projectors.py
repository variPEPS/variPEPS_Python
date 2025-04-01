import jax
import jax.numpy as jnp

from varipeps.peps import PEPS_Tensor_Triangular
from varipeps.contractions import apply_contraction_jitted
from varipeps.utils.svd import gauge_fixed_svd
from varipeps.config import Projector_Method, VariPEPS_Config
from varipeps.global_state import VariPEPS_Global_State
from .projectors import _check_chi, _truncated_SVD

from typing import Sequence


def _corner_workhorse(
    tensor_left_1: jnp.ndarray,
    tensor_left_2: jnp.ndarray,
    tensor_right: jnp.ndarray,
    chi: int,
    truncation_eps: float,
    fishman: bool,
):
    if tensor_left_1 is not None:
        left_matrix_1 = tensor_left_1.reshape(
            tensor_left_1.shape[0] * tensor_left_1.shape[1] * tensor_left_1.shape[2],
            tensor_left_1.shape[3] * tensor_left_1.shape[4] * tensor_left_1.shape[5],
        )

    left_matrix_2 = tensor_left_2.reshape(
        tensor_left_2.shape[0] * tensor_left_2.shape[1] * tensor_left_2.shape[2],
        tensor_left_2.shape[3] * tensor_left_2.shape[4] * tensor_left_2.shape[5],
    )

    right_matrix = tensor_right.reshape(
        tensor_right.shape[0] * tensor_right.shape[1] * tensor_right.shape[2],
        tensor_right.shape[3] * tensor_right.shape[4] * tensor_right.shape[5],
    )

    if tensor_left_1 is not None:
        left_matrix = left_matrix_1 @ left_matrix_2
    else:
        left_matrix = left_matrix_2

    if fishman:
        _, left_S, left_Vh = gauge_fixed_svd(left_matrix)
        left_S = jnp.where(
            left_S / left_S[0] < truncation_eps,
            0,
            jnp.sqrt(jnp.where(left_S / left_S[0] < truncation_eps, 1, left_S)),
        )
        left_matrix = left_S[:, jnp.newaxis] * left_Vh

        right_U, right_S, _ = gauge_fixed_svd(right_matrix)
        right_S = jnp.where(
            right_S / right_S[0] < truncation_eps,
            0,
            jnp.sqrt(jnp.where(right_S / right_S[0] < truncation_eps, 1, right_S)),
        )
        right_matrix = right_U * right_S[jnp.newaxis, :]

    left_matrix /= jnp.linalg.norm(left_matrix)
    right_matrix /= jnp.linalg.norm(right_matrix)

    product_matrix = left_matrix @ right_matrix
    product_matrix /= jnp.linalg.norm(product_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_left = jnp.dot(
        S_inv_sqrt[:, jnp.newaxis] * U.transpose().conj(),
        left_matrix,
    )

    projector_right = jnp.dot(
        right_matrix, Vh.transpose().conj() * S_inv_sqrt[jnp.newaxis, :]
    )

    projector_left = projector_left.reshape(
        projector_left.shape[0],
        tensor_left_2.shape[3],
        tensor_left_2.shape[4],
        tensor_left_2.shape[5],
    )

    projector_right = projector_right.reshape(
        tensor_right.shape[0],
        tensor_right.shape[1],
        tensor_right.shape[2],
        projector_right.shape[1],
    )

    return (
        projector_left,
        projector_right,
        smallest_S,
    )


def calc_corner_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor_Triangular]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
):
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for triangular CTMRG approach."
        )

    chi = _check_chi(peps_tensor_objs)

    projector_method = (
        config.ctmrg_full_projector_method
        if state.ctmrg_projector_method is None
        else state.ctmrg_projector_method
    )
    truncation_eps = (
        config.ctmrg_truncation_eps
        if state.ctmrg_effective_truncation_eps is None
        else state.ctmrg_effective_truncation_eps
    )

    corner_90 = apply_contraction_jitted(
        "triangular_ctmrg_corner_90",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    corner_210 = apply_contraction_jitted(
        "triangular_ctmrg_corner_210",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )

    corner_330 = apply_contraction_jitted(
        "triangular_ctmrg_corner_330",
        [peps_tensors[1][1]],
        [peps_tensor_objs[1][1]],
        [],
    )

    corner_30 = apply_contraction_jitted(
        "triangular_ctmrg_corner_30",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )

    corner_150 = apply_contraction_jitted(
        "triangular_ctmrg_corner_150",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    corner_270 = apply_contraction_jitted(
        "triangular_ctmrg_corner_270",
        [peps_tensors[1][1]],
        [peps_tensor_objs[1][1]],
        [],
    )

    projector_30_left, projector_30_right, smallest_S_30 = _corner_workhorse(
        corner_210 if projector_method is not Projector_Method.HALF else None,
        corner_90,
        corner_330,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_150_left, projector_150_right, smallest_S_150 = _corner_workhorse(
        corner_330 if projector_method is not Projector_Method.HALF else None,
        corner_210,
        corner_90,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_270_left, projector_270_right, smallest_S_270 = _corner_workhorse(
        corner_90 if projector_method is not Projector_Method.HALF else None,
        corner_330,
        corner_210,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_90_left, projector_90_right, smallest_S_90 = _corner_workhorse(
        corner_270 if projector_method is not Projector_Method.HALF else None,
        corner_150,
        corner_30,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_210_left, projector_210_right, smallest_S_210 = _corner_workhorse(
        corner_30 if projector_method is not Projector_Method.HALF else None,
        corner_270,
        corner_150,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_330_left, projector_330_right, smallest_S_330 = _corner_workhorse(
        corner_150 if projector_method is not Projector_Method.HALF else None,
        corner_30,
        corner_270,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    return (
        (projector_30_left, projector_30_right),
        (projector_90_left, projector_90_right),
        (projector_150_left, projector_150_right),
        (projector_210_left, projector_210_right),
        (projector_270_left, projector_270_right),
        (projector_330_left, projector_330_right),
        (
            smallest_S_30,
            smallest_S_90,
            smallest_S_150,
            smallest_S_210,
            smallest_S_270,
            smallest_S_330,
        ),
    )


def _T_workhorse(
    tensor_left: jnp.ndarray,
    tensor_right: jnp.ndarray,
    chi: int,
    truncation_eps: float,
    fishman: bool,
):
    left_matrix = tensor_left.reshape(
        tensor_left.shape[0] * tensor_left.shape[1] * tensor_left.shape[2],
        tensor_left.shape[3],
    )

    right_matrix = tensor_right.reshape(
        tensor_right.shape[0],
        tensor_right.shape[1] * tensor_right.shape[2] * tensor_right.shape[3],
    )

    if fishman:
        _, left_S, left_Vh = gauge_fixed_svd(left_matrix)
        left_S = jnp.where(
            left_S / left_S[0] < truncation_eps,
            0,
            jnp.sqrt(jnp.where(left_S / left_S[0] < truncation_eps, 1, left_S)),
        )
        left_matrix = left_S[:, jnp.newaxis] * left_Vh

        right_U, right_S, _ = gauge_fixed_svd(right_matrix)
        right_S = jnp.where(
            right_S / right_S[0] < truncation_eps,
            0,
            jnp.sqrt(jnp.where(right_S / right_S[0] < truncation_eps, 1, right_S)),
        )
        right_matrix = right_U * right_S[jnp.newaxis, :]

    left_matrix /= jnp.linalg.norm(left_matrix)
    right_matrix /= jnp.linalg.norm(right_matrix)

    product_matrix = left_matrix @ right_matrix
    product_matrix /= jnp.linalg.norm(product_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_left = jnp.dot(
        S_inv_sqrt[:, jnp.newaxis] * U.transpose().conj(),
        left_matrix,
    )

    projector_right = jnp.dot(
        right_matrix, Vh.transpose().conj() * S_inv_sqrt[jnp.newaxis, :]
    )

    projector_left = projector_left.reshape(
        projector_left.shape[0],
        tensor_left.shape[3],
    )

    projector_right = projector_right.reshape(
        tensor_right.shape[0], projector_right.shape[1]
    )

    return (
        projector_left,
        projector_right,
        smallest_S,
    )


def calc_T_30_150_270_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor_Triangular]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
):
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for triangular CTMRG approach."
        )

    chi = _check_chi(peps_tensor_objs)

    projector_method = (
        config.ctmrg_full_projector_method
        if state.ctmrg_projector_method is None
        else state.ctmrg_projector_method
    )
    truncation_eps = (
        config.ctmrg_truncation_eps
        if state.ctmrg_effective_truncation_eps is None
        else state.ctmrg_effective_truncation_eps
    )

    T_30_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_30_left",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    T_30_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_30_right",
        [peps_tensors[1][1]],
        [peps_tensor_objs[1][1]],
        [],
    )

    T_150_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_150_left",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )
    T_150_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_150_right",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    T_270_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_270_left",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )
    T_270_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_270_right",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    projector_T_30_left, projector_T_30_right, smallest_S_30 = _T_workhorse(
        T_30_left,
        T_30_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_T_150_left, projector_T_150_right, smallest_S_150 = _T_workhorse(
        T_150_left,
        T_150_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_T_270_left, projector_T_270_right, smallest_S_270 = _T_workhorse(
        T_270_left,
        T_270_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    return (
        (projector_T_30_left, projector_T_30_right),
        (projector_T_150_left, projector_T_150_right),
        (projector_T_270_left, projector_T_270_right),
        (smallest_S_30, smallest_S_150, smallest_S_270),
    )


def calc_T_90_210_330_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor_Triangular]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
):
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for triangular CTMRG approach."
        )

    chi = _check_chi(peps_tensor_objs)

    projector_method = (
        config.ctmrg_full_projector_method
        if state.ctmrg_projector_method is None
        else state.ctmrg_projector_method
    )
    truncation_eps = (
        config.ctmrg_truncation_eps
        if state.ctmrg_effective_truncation_eps is None
        else state.ctmrg_effective_truncation_eps
    )

    T_90_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_90_left",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    T_90_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_90_right",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )

    T_210_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_210_left",
        [peps_tensors[1][1]],
        [peps_tensor_objs[1][1]],
        [],
    )
    T_210_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_210_right",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    T_330_left = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_330_left",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    T_330_right = apply_contraction_jitted(
        "triangular_ctmrg_T_proj_330_right",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )

    projector_T_90_left, projector_T_90_right, smallest_S_90 = _T_workhorse(
        T_90_left,
        T_90_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_T_210_left, projector_T_210_right, smallest_S_210 = _T_workhorse(
        T_210_left,
        T_210_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    projector_T_330_left, projector_T_330_right, smallest_S_330 = _T_workhorse(
        T_330_left,
        T_330_right,
        chi,
        truncation_eps,
        projector_method is Projector_Method.FISHMAN,
    )

    return (
        (projector_T_90_left, projector_T_90_right),
        (projector_T_210_left, projector_T_210_right),
        (projector_T_330_left, projector_T_330_right),
        (smallest_S_90, smallest_S_210, smallest_S_330),
    )
