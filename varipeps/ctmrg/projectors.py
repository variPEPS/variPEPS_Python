import enum
from functools import partial

import jax.numpy as jnp
from jax import jit, checkpoint

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps import varipeps_config
from varipeps.utils.func_cache import Checkpointing_Cache
from varipeps.utils.svd import gauge_fixed_svd
from varipeps.utils.projector_dict import (
    Left_Projectors,
    Right_Projectors,
    Top_Projectors,
    Bottom_Projectors,
    Left_Projectors_Split_Transfer,
    Right_Projectors_Split_Transfer,
    Top_Projectors_Split_Transfer,
    Bottom_Projectors_Split_Transfer,
)
from varipeps.config import Projector_Method, VariPEPS_Config
from varipeps.global_state import VariPEPS_Global_State

from typing import Sequence, Tuple, TypeVar, Optional

from varipeps.utils.debug_print import debug_print
import jax.debug


class _Projectors_Func_Cache:
    _left = None
    _right = None
    _top = None
    _bottom = None

    def __class_getitem__(cls, name: str) -> Checkpointing_Cache:
        name = f"_{name}"
        obj = getattr(cls, name)
        if obj is None:
            obj = Checkpointing_Cache(varipeps_config.checkpointing_projectors)
            setattr(cls, name, obj)
        return obj


def _check_chi(peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]]) -> int:
    chi = int(
        peps_tensor_objs[0][0].chi
    )  # in the backward pass chi is a traced ConcreteArray
    if not all(int(j.chi) == chi for i in peps_tensor_objs for j in i):
        raise ValueError(
            "Environment bond dimension not the same over the whole network."
        )
    return chi


def _calc_ctmrg_quarters(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if peps_tensor_objs[0][0].d > peps_tensor_objs[0][0].chi:
        top_left = apply_contraction(
            "ctmrg_top_left_large_d", [peps_tensors[0][0]], [peps_tensor_objs[0][0]], []
        )

        top_right = apply_contraction(
            "ctmrg_top_right_large_d",
            [peps_tensors[0][1]],
            [peps_tensor_objs[0][1]],
            [],
        )

        bottom_left = apply_contraction(
            "ctmrg_bottom_left_large_d",
            [peps_tensors[1][0]],
            [peps_tensor_objs[1][0]],
            [],
        )

        bottom_right = apply_contraction(
            "ctmrg_bottom_right_large_d",
            [peps_tensors[1][1]],
            [peps_tensor_objs[1][1]],
            [],
        )
    else:
        top_left = apply_contraction(
            "ctmrg_top_left", [peps_tensors[0][0]], [peps_tensor_objs[0][0]], []
        )

        top_right = apply_contraction(
            "ctmrg_top_right", [peps_tensors[0][1]], [peps_tensor_objs[0][1]], []
        )

        bottom_left = apply_contraction(
            "ctmrg_bottom_left", [peps_tensors[1][0]], [peps_tensor_objs[1][0]], []
        )

        bottom_right = apply_contraction(
            "ctmrg_bottom_right", [peps_tensors[1][1]], [peps_tensor_objs[1][1]], []
        )

    return top_left, top_right, bottom_left, bottom_right


def _truncated_SVD(
    matrix: jnp.ndarray, chi: int, truncation_eps: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    U, S, Vh = gauge_fixed_svd(matrix)

    # Truncate the singular values
    S = S[:chi]
    U = U[:, :chi]
    Vh = Vh[:chi, :]

    relevant_S_values = (S / S[0]) > truncation_eps
    S_inv_sqrt = jnp.where(
        relevant_S_values, 1 / jnp.sqrt(jnp.where(relevant_S_values, S, 1)), 0
    )

    matrix_norm = jnp.sum(jnp.abs(matrix) ** 2)
    S_norm = jnp.sum(S**2)
    trunc_error = 1 - S_norm / matrix_norm
    trunc_error = jnp.where(
        trunc_error < truncation_eps**2,
        0,
        jnp.sqrt(jnp.where(trunc_error < truncation_eps**2, 1, trunc_error)),
    )

    return S_inv_sqrt, U, Vh, trunc_error


def _quarter_tensors_to_matrix(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    top_left_matrix = top_left.reshape(
        top_left.shape[0] * top_left.shape[1] * top_left.shape[2],
        top_left.shape[3] * top_left.shape[4] * top_left.shape[5],
    )
    top_right_matrix = top_right.reshape(
        top_right.shape[0] * top_right.shape[1] * top_right.shape[2],
        top_right.shape[3] * top_right.shape[4] * top_right.shape[5],
    )
    bottom_left_matrix = bottom_left.reshape(
        bottom_left.shape[0] * bottom_left.shape[1] * bottom_left.shape[2],
        bottom_left.shape[3] * bottom_left.shape[4] * bottom_left.shape[5],
    )
    bottom_right_matrix = bottom_right.reshape(
        bottom_right.shape[0] * bottom_right.shape[1] * bottom_right.shape[2],
        bottom_right.shape[3] * bottom_right.shape[4] * bottom_right.shape[5],
    )

    return top_left_matrix, top_right_matrix, bottom_left_matrix, bottom_right_matrix


def _horizontal_cut(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (
        top_left_matrix,
        top_right_matrix,
        bottom_left_matrix,
        bottom_right_matrix,
    ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)

    top_matrix = jnp.dot(top_left_matrix, top_right_matrix)
    bottom_matrix = jnp.dot(bottom_right_matrix, bottom_left_matrix)

    return top_matrix / jnp.linalg.norm(top_matrix), bottom_matrix / jnp.linalg.norm(
        bottom_matrix
    )


def _vertical_cut(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (
        top_left_matrix,
        top_right_matrix,
        bottom_left_matrix,
        bottom_right_matrix,
    ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)

    left_matrix = jnp.dot(bottom_left_matrix, top_left_matrix)
    right_matrix = jnp.dot(top_right_matrix, bottom_right_matrix)

    return left_matrix / jnp.linalg.norm(left_matrix), right_matrix / jnp.linalg.norm(
        right_matrix
    )


def _fishman_horizontal_cut(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    (
        top_left_matrix,
        top_right_matrix,
        bottom_left_matrix,
        bottom_right_matrix,
    ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)

    top_matrix = jnp.dot(top_left_matrix, top_right_matrix)
    bottom_matrix = jnp.dot(bottom_right_matrix, bottom_left_matrix)

    top_U, top_S, top_Vh = gauge_fixed_svd(top_matrix)
    top_S = jnp.where((top_S / top_S[0]) >= truncation_eps, top_S, 0)

    bottom_U, bottom_S, bottom_Vh = gauge_fixed_svd(bottom_matrix)
    bottom_S = jnp.where((bottom_S / bottom_S[0]) >= truncation_eps, bottom_S, 0)

    return top_U, top_S, top_Vh, bottom_U, bottom_S, bottom_Vh


def _fishman_vertical_cut(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    (
        top_left_matrix,
        top_right_matrix,
        bottom_left_matrix,
        bottom_right_matrix,
    ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)

    left_matrix = jnp.dot(bottom_left_matrix, top_left_matrix)
    right_matrix = jnp.dot(top_right_matrix, bottom_right_matrix)

    left_U, left_S, left_Vh = gauge_fixed_svd(left_matrix)
    left_S = jnp.where((left_S / left_S[0]) >= truncation_eps, left_S, 0)

    right_U, right_S, right_Vh = gauge_fixed_svd(right_matrix)
    right_S = jnp.where((right_S / right_S[0]) >= truncation_eps, right_S, 0)

    return left_U, left_S, left_Vh, right_U, right_S, right_Vh


def _split_transfer_fishman(first_tensor, second_tensor, truncation_eps):
    if first_tensor.ndim == 5:
        first_tensor_ketbra = first_tensor.reshape(
            first_tensor.shape[0] * first_tensor.shape[1],
            first_tensor.shape[2] * first_tensor.shape[3] * first_tensor.shape[4],
        )

        second_tensor_ketbra = second_tensor.reshape(
            second_tensor.shape[0] * second_tensor.shape[1] * second_tensor.shape[2],
            second_tensor.shape[3] * second_tensor.shape[4],
        )
    elif first_tensor.ndim == 6:
        first_tensor_ketbra = first_tensor.reshape(
            first_tensor.shape[0] * first_tensor.shape[1] * first_tensor.shape[2],
            first_tensor.shape[3] * first_tensor.shape[4] * first_tensor.shape[5],
        )

        second_tensor_ketbra = second_tensor.reshape(
            second_tensor.shape[0] * second_tensor.shape[1] * second_tensor.shape[2],
            second_tensor.shape[3] * second_tensor.shape[4] * second_tensor.shape[5],
        )
    else:
        first_tensor_ketbra = first_tensor.reshape(
            first_tensor.shape[0] * first_tensor.shape[1],
            first_tensor.shape[2] * first_tensor.shape[3],
        )

        second_tensor_ketbra = second_tensor.reshape(
            second_tensor.shape[0] * second_tensor.shape[1],
            second_tensor.shape[2] * second_tensor.shape[3],
        )

    first_ketbra_U, first_ketbra_S, first_ketbra_Vh = gauge_fixed_svd(
        first_tensor_ketbra
    )
    first_ketbra_S = jnp.where(
        (first_ketbra_S / first_ketbra_S[0]) >= truncation_eps, first_ketbra_S, 0
    )
    first_ketbra_S /= jnp.sum(first_ketbra_S)
    first_ketbra_S = jnp.where(
        first_ketbra_S == 0,
        0,
        jnp.sqrt(jnp.where(first_ketbra_S == 0, 1, first_ketbra_S)),
    )
    if first_tensor.ndim == 5:
        first_ketbra_U = first_ketbra_U.reshape(
            first_tensor.shape[0], first_tensor.shape[1], first_ketbra_U.shape[-1]
        )
        first_ketbra_Vh = first_ketbra_Vh.reshape(
            first_ketbra_Vh.shape[0],
            first_tensor.shape[2],
            first_tensor.shape[3],
            first_tensor.shape[4],
        )
    elif first_tensor.ndim == 6:
        first_ketbra_U = first_ketbra_U.reshape(
            first_tensor.shape[0],
            first_tensor.shape[1],
            first_tensor.shape[2],
            first_ketbra_U.shape[-1],
        )
        first_ketbra_Vh = first_ketbra_Vh.reshape(
            first_ketbra_Vh.shape[0],
            first_tensor.shape[3],
            first_tensor.shape[4],
            first_tensor.shape[5],
        )
    else:
        first_ketbra_U = first_ketbra_U.reshape(
            first_tensor.shape[0], first_tensor.shape[1], first_ketbra_U.shape[-1]
        )
        first_ketbra_Vh = first_ketbra_Vh.reshape(
            first_ketbra_Vh.shape[0], first_tensor.shape[2], first_tensor.shape[3]
        )

    second_ketbra_U, second_ketbra_S, second_ketbra_Vh = gauge_fixed_svd(
        second_tensor_ketbra
    )
    second_ketbra_S = jnp.where(
        (second_ketbra_S / second_ketbra_S[0]) >= truncation_eps, second_ketbra_S, 0
    )
    second_ketbra_S /= jnp.sum(second_ketbra_S)
    second_ketbra_S = jnp.where(
        second_ketbra_S == 0,
        0,
        jnp.sqrt(jnp.where(second_ketbra_S == 0, 1, second_ketbra_S)),
    )
    if second_tensor.ndim == 5:
        second_ketbra_U = second_ketbra_U.reshape(
            second_tensor.shape[0],
            second_tensor.shape[1],
            second_tensor.shape[2],
            second_ketbra_U.shape[-1],
        )
        second_ketbra_Vh = second_ketbra_Vh.reshape(
            second_ketbra_Vh.shape[0], second_tensor.shape[3], second_tensor.shape[4]
        )
    elif first_tensor.ndim == 6:
        second_ketbra_U = second_ketbra_U.reshape(
            second_tensor.shape[0],
            second_tensor.shape[1],
            second_tensor.shape[2],
            second_ketbra_U.shape[-1],
        )
        second_ketbra_Vh = second_ketbra_Vh.reshape(
            second_ketbra_Vh.shape[0],
            second_tensor.shape[3],
            second_tensor.shape[4],
            second_tensor.shape[5],
        )
    else:
        second_ketbra_U = second_ketbra_U.reshape(
            second_tensor.shape[0], second_tensor.shape[1], second_ketbra_U.shape[-1]
        )
        second_ketbra_Vh = second_ketbra_Vh.reshape(
            second_ketbra_Vh.shape[0], second_tensor.shape[2], second_tensor.shape[3]
        )

    return (
        first_ketbra_U,
        first_ketbra_S,
        first_ketbra_Vh,
        second_ketbra_U,
        second_ketbra_S,
        second_ketbra_Vh,
    )


def _horizontal_cut_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    fishman: bool = False,
    truncation_eps: Optional[float] = None,
    offset: int = 0,
):
    top_tensor = apply_contraction_jitted(
        "ctmrg_split_transfer_top",
        [peps_tensors[0][0 + offset]],
        [peps_tensor_objs[0][0 + offset]],
        [],
    )

    bottom_tensor = apply_contraction_jitted(
        "ctmrg_split_transfer_bottom",
        [peps_tensors[1][0 + offset]],
        [peps_tensor_objs[1][0 + offset]],
        [],
    )

    top_tensor /= jnp.linalg.norm(top_tensor)
    bottom_tensor /= jnp.linalg.norm(bottom_tensor)

    if fishman:
        return _split_transfer_fishman(top_tensor, bottom_tensor, truncation_eps)

    return (
        top_tensor,
        bottom_tensor,
    )


def _vertical_cut_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    fishman: bool = False,
    truncation_eps: Optional[float] = None,
    offset: int = 0,
):
    left_tensor = apply_contraction_jitted(
        "ctmrg_split_transfer_left",
        [peps_tensors[0 + offset][0]],
        [peps_tensor_objs[0 + offset][0]],
        [],
    )

    right_tensor = apply_contraction_jitted(
        "ctmrg_split_transfer_right",
        [peps_tensors[0 + offset][1]],
        [peps_tensor_objs[0 + offset][1]],
        [],
    )

    left_tensor /= jnp.linalg.norm(left_tensor)
    right_tensor /= jnp.linalg.norm(right_tensor)

    if fishman:
        return _split_transfer_fishman(left_tensor, right_tensor, truncation_eps)

    return (
        left_tensor,
        right_tensor,
    )


@partial(jit, static_argnums=(4, 5, 6), inline=True)
def _left_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
    projector_method: Projector_Method,
    chi: int,
) -> Left_Projectors:
    if projector_method is Projector_Method.FULL:
        top_matrix, bottom_matrix = _horizontal_cut(
            top_left, top_right, bottom_left, bottom_right
        )
    elif projector_method is Projector_Method.HALF:
        (
            top_matrix,
            _,
            bottom_matrix,
            _,
        ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)
        top_matrix /= jnp.linalg.norm(top_matrix)
        bottom_matrix /= jnp.linalg.norm(bottom_matrix)
    elif projector_method is Projector_Method.FISHMAN:
        top_U, top_S, _, _, bottom_S, bottom_Vh = _fishman_horizontal_cut(
            top_left, top_right, bottom_left, bottom_right, truncation_eps
        )
        top_matrix = top_U * jnp.sqrt(top_S)[jnp.newaxis, :]
        bottom_matrix = jnp.sqrt(bottom_S)[:, jnp.newaxis] * bottom_Vh
        top_matrix /= jnp.linalg.norm(top_matrix)
        bottom_matrix /= jnp.linalg.norm(bottom_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(bottom_matrix, top_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_left_top = jnp.dot(top_matrix, Vh.transpose().conj() * S_inv_sqrt)
    projector_left_bottom = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], bottom_matrix
    )

    projector_left_top = projector_left_top.reshape(
        top_left.shape[0],
        top_left.shape[1],
        top_left.shape[2],
        projector_left_top.shape[1],
    )
    projector_left_bottom = projector_left_bottom.reshape(
        projector_left_bottom.shape[0],
        bottom_left.shape[3],
        bottom_left.shape[4],
        bottom_left.shape[5],
    )

    return (
        Left_Projectors(top=projector_left_top, bottom=projector_left_bottom),
        smallest_S,
    )


def calc_left_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Left_Projectors:
    """
    Calculate the left projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.utils.projector_dict.Left_Projectors`:
        The left top and bottom projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if config.checkpointing_projectors:
        f = checkpoint(_left_projectors_workhorse, static_argnums=(4, 5, 6))
    else:
        f = _left_projectors_workhorse

    return f(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        (
            config.ctmrg_truncation_eps
            if state.ctmrg_effective_truncation_eps is None
            else state.ctmrg_effective_truncation_eps
        ),
        (
            config.ctmrg_full_projector_method
            if state.ctmrg_projector_method is None
            else state.ctmrg_projector_method
        ),
        chi,
    )


@partial(jit, static_argnums=(4, 5, 6), inline=True)
def _right_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
    projector_method: Projector_Method,
    chi: int,
) -> Right_Projectors:
    if projector_method is Projector_Method.FULL:
        top_matrix, bottom_matrix = _horizontal_cut(
            top_left, top_right, bottom_left, bottom_right
        )
    elif projector_method is Projector_Method.HALF:
        (
            _,
            top_matrix,
            _,
            bottom_matrix,
        ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)
        top_matrix /= jnp.linalg.norm(top_matrix)
        bottom_matrix /= jnp.linalg.norm(bottom_matrix)
    elif projector_method is Projector_Method.FISHMAN:
        _, top_S, top_Vh, bottom_U, bottom_S, _ = _fishman_horizontal_cut(
            top_left, top_right, bottom_left, bottom_right, truncation_eps
        )
        top_matrix = jnp.sqrt(top_S)[:, jnp.newaxis] * top_Vh
        bottom_matrix = bottom_U * jnp.sqrt(bottom_S)[jnp.newaxis, :]
        top_matrix /= jnp.linalg.norm(top_matrix)
        bottom_matrix /= jnp.linalg.norm(bottom_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(top_matrix, bottom_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_right_top = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], top_matrix
    )
    projector_right_bottom = jnp.dot(bottom_matrix, Vh.transpose().conj() * S_inv_sqrt)

    projector_right_top = projector_right_top.reshape(
        projector_right_top.shape[0],
        top_right.shape[3],
        top_right.shape[4],
        top_right.shape[5],
    )
    projector_right_bottom = projector_right_bottom.reshape(
        bottom_right.shape[0],
        bottom_right.shape[1],
        bottom_right.shape[2],
        projector_right_bottom.shape[1],
    )

    return (
        Right_Projectors(top=projector_right_top, bottom=projector_right_bottom),
        smallest_S,
    )


def calc_right_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Right_Projectors:
    """
    Calculate the right projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.utils.projector_dict.Right_Projectors`:
        The right top and bottom projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if config.checkpointing_projectors:
        f = checkpoint(_right_projectors_workhorse, static_argnums=(4, 5, 6))
    else:
        f = _right_projectors_workhorse

    return f(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        (
            config.ctmrg_truncation_eps
            if state.ctmrg_effective_truncation_eps is None
            else state.ctmrg_effective_truncation_eps
        ),
        (
            config.ctmrg_full_projector_method
            if state.ctmrg_projector_method is None
            else state.ctmrg_projector_method
        ),
        chi,
    )


@partial(jit, static_argnums=(4, 5, 6), inline=True)
def _top_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
    projector_method: Projector_Method,
    chi: int,
) -> Top_Projectors:
    if projector_method is Projector_Method.FULL:
        left_matrix, right_matrix = _vertical_cut(
            top_left, top_right, bottom_left, bottom_right
        )
    elif projector_method is Projector_Method.HALF:
        (
            left_matrix,
            right_matrix,
            _,
            _,
        ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)
        left_matrix /= jnp.linalg.norm(left_matrix)
        right_matrix /= jnp.linalg.norm(right_matrix)
    elif projector_method is Projector_Method.FISHMAN:
        _, left_S, left_Vh, right_U, right_S, _ = _fishman_vertical_cut(
            top_left, top_right, bottom_left, bottom_right, truncation_eps
        )
        left_matrix = jnp.sqrt(left_S)[:, jnp.newaxis] * left_Vh
        right_matrix = right_U * jnp.sqrt(right_S)[jnp.newaxis, :]
        left_matrix /= jnp.linalg.norm(left_matrix)
        right_matrix /= jnp.linalg.norm(right_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(left_matrix, right_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_top_left = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], left_matrix
    )
    projector_top_right = jnp.dot(right_matrix, Vh.transpose().conj() * S_inv_sqrt)

    projector_top_left = projector_top_left.reshape(
        projector_top_left.shape[0],
        top_left.shape[3],
        top_left.shape[4],
        top_left.shape[5],
    )
    projector_top_right = projector_top_right.reshape(
        top_right.shape[0],
        top_right.shape[1],
        top_right.shape[2],
        projector_top_right.shape[1],
    )

    return (
        Top_Projectors(left=projector_top_left, right=projector_top_right),
        smallest_S,
    )


def calc_top_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Top_Projectors:
    """
    Calculate the top projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.utils.projector_dict.Top_Projectors`:
        The top left and right projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if config.checkpointing_projectors:
        f = checkpoint(_top_projectors_workhorse, static_argnums=(4, 5, 6))
    else:
        f = _top_projectors_workhorse

    return f(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        (
            config.ctmrg_truncation_eps
            if state.ctmrg_effective_truncation_eps is None
            else state.ctmrg_effective_truncation_eps
        ),
        (
            config.ctmrg_full_projector_method
            if state.ctmrg_projector_method is None
            else state.ctmrg_projector_method
        ),
        chi,
    )


@partial(jit, static_argnums=(4, 5, 6), inline=True)
def _bottom_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    truncation_eps: float,
    projector_method: Projector_Method,
    chi: int,
) -> Bottom_Projectors:
    if projector_method is Projector_Method.FULL:
        left_matrix, right_matrix = _vertical_cut(
            top_left, top_right, bottom_left, bottom_right
        )
    elif projector_method is Projector_Method.HALF:
        (
            _,
            _,
            left_matrix,
            right_matrix,
        ) = _quarter_tensors_to_matrix(top_left, top_right, bottom_left, bottom_right)
        left_matrix /= jnp.linalg.norm(left_matrix)
        right_matrix /= jnp.linalg.norm(right_matrix)
    elif projector_method is Projector_Method.FISHMAN:
        left_U, left_S, _, _, right_S, right_Vh = _fishman_vertical_cut(
            top_left, top_right, bottom_left, bottom_right, truncation_eps
        )
        left_matrix = left_U * jnp.sqrt(left_S)[jnp.newaxis, :]
        right_matrix = jnp.sqrt(right_S)[:, jnp.newaxis] * right_Vh
        left_matrix /= jnp.linalg.norm(left_matrix)
        right_matrix /= jnp.linalg.norm(right_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(right_matrix, left_matrix)

    S_inv_sqrt, U, Vh, smallest_S = _truncated_SVD(product_matrix, chi, truncation_eps)

    projector_bottom_left = jnp.dot(left_matrix, Vh.transpose().conj() * S_inv_sqrt)
    projector_bottom_right = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], right_matrix
    )

    projector_bottom_left = projector_bottom_left.reshape(
        bottom_left.shape[0],
        bottom_left.shape[1],
        bottom_left.shape[2],
        projector_bottom_left.shape[1],
    )
    projector_bottom_right = projector_bottom_right.reshape(
        projector_bottom_right.shape[0],
        bottom_right.shape[3],
        bottom_right.shape[4],
        bottom_right.shape[5],
    )

    return (
        Bottom_Projectors(left=projector_bottom_left, right=projector_bottom_right),
        smallest_S,
    )


def calc_bottom_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Bottom_Projectors:
    """
    Calculate the bottom projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
      state (:obj:`~varipeps.global_state.VariPEPS_Global_State`):
        Global state object of the variPEPS library. It is used to transport
        a common state across different parts of the framework. Please see its
        class definition for details.
    Returns:
      :obj:`~varipeps.utils.projector_dict.Bottom_Projectors`:
        The bottom left and right projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if config.checkpointing_projectors:
        f = checkpoint(_bottom_projectors_workhorse, static_argnums=(4, 5, 6))
    else:
        f = _bottom_projectors_workhorse

    return f(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        (
            config.ctmrg_truncation_eps
            if state.ctmrg_effective_truncation_eps is None
            else state.ctmrg_effective_truncation_eps
        ),
        (
            config.ctmrg_full_projector_method
            if state.ctmrg_projector_method is None
            else state.ctmrg_projector_method
        ),
        chi,
    )


def _split_transfer_workhorse(
    first_ketbra: jnp.ndarray,
    second_ketbra: jnp.ndarray,
    chi: int,
    truncation_eps: float,
    fishman_input: bool = False,
):
    if fishman_input and first_ketbra.ndim == 4:
        first_ketbra_matrix = first_ketbra.reshape(
            first_ketbra.shape[0],
            first_ketbra.shape[1] * first_ketbra.shape[2] * first_ketbra.shape[3],
        )

        second_ketbra_matrix = second_ketbra.reshape(
            second_ketbra.shape[0] * second_ketbra.shape[1] * second_ketbra.shape[2],
            second_ketbra.shape[3],
        )
    elif first_ketbra.ndim == 4:
        first_ketbra_matrix = first_ketbra.reshape(
            first_ketbra.shape[0] * first_ketbra.shape[1],
            first_ketbra.shape[2] * first_ketbra.shape[3],
        )

        second_ketbra_matrix = second_ketbra.reshape(
            second_ketbra.shape[0] * second_ketbra.shape[1],
            second_ketbra.shape[2] * second_ketbra.shape[3],
        )
    elif first_ketbra.ndim == 3:
        first_ketbra_matrix = first_ketbra.reshape(
            first_ketbra.shape[0],
            first_ketbra.shape[1] * first_ketbra.shape[2],
        )

        second_ketbra_matrix = second_ketbra.reshape(
            second_ketbra.shape[0] * second_ketbra.shape[1],
            second_ketbra.shape[2],
        )
    elif first_ketbra.ndim == 5:
        first_ketbra_matrix = first_ketbra.reshape(
            first_ketbra.shape[0] * first_ketbra.shape[1] * first_ketbra.shape[2],
            first_ketbra.shape[3] * first_ketbra.shape[4],
        )
        second_ketbra_matrix = second_ketbra.reshape(
            second_ketbra.shape[0] * second_ketbra.shape[1],
            second_ketbra.shape[2] * second_ketbra.shape[3] * second_ketbra.shape[4],
        )
    elif first_ketbra.ndim == 6:
        first_ketbra_matrix = first_ketbra.reshape(
            first_ketbra.shape[0] * first_ketbra.shape[1] * first_ketbra.shape[2],
            first_ketbra.shape[3] * first_ketbra.shape[4] * first_ketbra.shape[5],
        )
        second_ketbra_matrix = second_ketbra.reshape(
            second_ketbra.shape[0] * second_ketbra.shape[1] * second_ketbra.shape[2],
            second_ketbra.shape[3] * second_ketbra.shape[4] * second_ketbra.shape[5],
        )
    else:
        raise ValueError("Invalid dimension of the input tensor")

    product_matrix_ketbra = jnp.dot(first_ketbra_matrix, second_ketbra_matrix)
    S_inv_sqrt_ketbra, U_ketbra, Vh_ketbra, smallest_S_ketbra = _truncated_SVD(
        product_matrix_ketbra, chi, truncation_eps
    )

    projector_first_ketbra = jnp.dot(
        U_ketbra.transpose().conj() * S_inv_sqrt_ketbra[:, jnp.newaxis],
        first_ketbra_matrix,
    )
    projector_second_ketbra = jnp.dot(
        second_ketbra_matrix, Vh_ketbra.transpose().conj() * S_inv_sqrt_ketbra
    )

    if first_ketbra.ndim == 6:
        projector_first_ketbra = projector_first_ketbra.reshape(
            projector_first_ketbra.shape[0],
            first_ketbra.shape[3],
            first_ketbra.shape[4],
            first_ketbra.shape[5],
        )
        projector_second_ketbra = projector_second_ketbra.reshape(
            second_ketbra.shape[0],
            second_ketbra.shape[1],
            second_ketbra.shape[2],
            projector_second_ketbra.shape[1],
        )
    else:
        if fishman_input and first_ketbra.ndim == 4:
            projector_first_ketbra = projector_first_ketbra.reshape(
                projector_first_ketbra.shape[0],
                first_ketbra.shape[1],
                first_ketbra.shape[2],
                first_ketbra.shape[3],
            )
        elif first_ketbra.ndim == 4:
            projector_first_ketbra = projector_first_ketbra.reshape(
                projector_first_ketbra.shape[0],
                first_ketbra.shape[2],
                first_ketbra.shape[3],
            )
        elif first_ketbra.ndim == 5:
            projector_first_ketbra = projector_first_ketbra.reshape(
                projector_first_ketbra.shape[0],
                first_ketbra.shape[3],
                first_ketbra.shape[4],
            )
        else:
            projector_first_ketbra = projector_first_ketbra.reshape(
                projector_first_ketbra.shape[0],
                first_ketbra.shape[1],
                first_ketbra.shape[2],
            )

        if fishman_input and first_ketbra.ndim == 4:
            projector_second_ketbra = projector_second_ketbra.reshape(
                second_ketbra.shape[0],
                second_ketbra.shape[1],
                second_ketbra.shape[2],
                projector_second_ketbra.shape[1],
            )
        else:
            projector_second_ketbra = projector_second_ketbra.reshape(
                second_ketbra.shape[0],
                second_ketbra.shape[1],
                projector_second_ketbra.shape[1],
            )

    return (
        projector_first_ketbra,
        projector_second_ketbra,
        smallest_S_ketbra,
    )


def calc_left_projectors_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Left_Projectors_Split_Transfer:
    """
    Calculate the left projectors for the CTMRG method. This functions uses the
    CTMRG method with split transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The left top and bottom projectors for both layer.
    """
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for split transfer matrices approach."
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

    if projector_method is Projector_Method.FULL:
        (
            top_tensor_ketbra_left,
            bottom_tensor_ketbra_left,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.HALF:
        (
            top_tensor_ketbra_left,
            bottom_tensor_ketbra_left,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            left_tensor_ketbra_bottom,
            right_tensor_ketbra_bottom,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.FISHMAN:
        (
            top_ketbra_U,
            top_ketbra_S,
            _,
            _,
            bottom_ketbra_S,
            bottom_ketbra_Vh,
        ) = _horizontal_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps
        )

        top_tensor_ketbra_left = (
            top_ketbra_U * top_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
        bottom_tensor_ketbra_left = (
            bottom_ketbra_S[:, jnp.newaxis, jnp.newaxis] * bottom_ketbra_Vh
        )

        (
            _,
            top_ketbra_S,
            top_ketbra_Vh,
            bottom_ketbra_U,
            bottom_ketbra_S,
            _,
        ) = _horizontal_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps, offset=1
        )
        top_tensor_ketbra_right = (
            top_ketbra_S[:, jnp.newaxis, jnp.newaxis] * top_ketbra_Vh
        )
        bottom_tensor_ketbra_right = (
            bottom_ketbra_U * bottom_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
    else:
        raise ValueError("Invalid projector method!")

    (
        projector_left_bottom_ket,
        projector_left_top_ket,
        smallest_S_ket,
    ) = _split_transfer_workhorse(
        bottom_tensor_ketbra_left,
        top_tensor_ketbra_left,
        chi,
        truncation_eps,
    )

    if (
        projector_method is Projector_Method.FULL
        or projector_method is Projector_Method.FISHMAN
    ):
        (
            projector_right_top_bra,
            projector_right_bottom_bra,
            _,
        ) = _split_transfer_workhorse(
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
            chi,
            truncation_eps,
        )

        top_tensor_bra_left = apply_contraction_jitted(
            "ctmrg_split_transfer_top_full",
            [peps_tensors[0][0], peps_tensors[0][1]],
            [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
            [projector_left_bottom_ket, projector_right_bottom_bra],
        )

        bottom_tensor_bra_left = apply_contraction_jitted(
            "ctmrg_split_transfer_bottom_full",
            [peps_tensors[1][0], peps_tensors[1][1]],
            [peps_tensor_objs[1][0], peps_tensor_objs[1][1]],
            [projector_left_top_ket, projector_right_top_bra],
        )

        top_tensor_bra_left /= jnp.linalg.norm(top_tensor_bra_left)
        bottom_tensor_bra_left /= jnp.linalg.norm(bottom_tensor_bra_left)
    elif projector_method is Projector_Method.HALF:
        (
            _,
            projector_top_right_ket,
            _,
        ) = _split_transfer_workhorse(
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
            chi,
            truncation_eps,
        )

        (
            projector_bottom_right_bra,
            _,
            _,
        ) = _split_transfer_workhorse(
            right_tensor_ketbra_bottom,
            left_tensor_ketbra_bottom,
            chi,
            truncation_eps,
        )

        top_tensor_bra_left = apply_contraction_jitted(
            "ctmrg_split_transfer_left_top_half",
            [peps_tensors[0][0]],
            [peps_tensor_objs[0][0]],
            [projector_left_bottom_ket, projector_top_right_ket],
        )

        bottom_tensor_bra_left = apply_contraction_jitted(
            "ctmrg_split_transfer_left_bottom_half",
            [peps_tensors[1][0]],
            [peps_tensor_objs[1][0]],
            [projector_left_top_ket, projector_bottom_right_bra],
        )

        top_tensor_bra_left /= jnp.linalg.norm(top_tensor_bra_left)
        bottom_tensor_bra_left /= jnp.linalg.norm(bottom_tensor_bra_left)

    if projector_method is Projector_Method.FISHMAN:
        (
            top_bra_U,
            top_bra_S,
            _,
            _,
            bottom_bra_S,
            bottom_bra_Vh,
        ) = _split_transfer_fishman(
            top_tensor_bra_left, bottom_tensor_bra_left, truncation_eps
        )

        top_tensor_bra_left = top_bra_U * top_bra_S[jnp.newaxis, jnp.newaxis, :]
        bottom_tensor_bra_left = (
            bottom_bra_S[:, jnp.newaxis, jnp.newaxis] * bottom_bra_Vh
        )

    (
        projector_left_bottom_bra,
        projector_left_top_bra,
        smallest_S_bra,
    ) = _split_transfer_workhorse(
        bottom_tensor_bra_left,
        top_tensor_bra_left,
        chi,
        truncation_eps,
    )

    top_tensor_phys_ket_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    bottom_tensor_phys_ket_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    top_tensor_phys_bra_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )
    bottom_tensor_phys_bra_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )
    top_tensor_phys_bra_right = top_tensor_phys_bra_right.transpose(0, 1, 4, 2, 3)
    bottom_tensor_phys_bra_right = bottom_tensor_phys_bra_right.transpose(0, 1, 4, 2, 3)

    top_tensor_phys_ket_left /= jnp.linalg.norm(top_tensor_phys_ket_left)
    bottom_tensor_phys_ket_left /= jnp.linalg.norm(bottom_tensor_phys_ket_left)
    top_tensor_phys_bra_right /= jnp.linalg.norm(top_tensor_phys_bra_right)
    bottom_tensor_phys_bra_right /= jnp.linalg.norm(bottom_tensor_phys_bra_right)

    if (
        projector_method is Projector_Method.FISHMAN
        or projector_method is Projector_Method.HALF
    ):
        (
            top_phys_ket_U,
            top_phys_ket_S,
            _,
            _,
            bottom_phys_ket_S,
            bottom_phys_ket_Vh,
        ) = _split_transfer_fishman(
            top_tensor_phys_ket_left, bottom_tensor_phys_ket_left, truncation_eps
        )

        top_tensor_phys_ket_left = (
            top_phys_ket_U * top_phys_ket_S[jnp.newaxis, jnp.newaxis, :]
        )
        bottom_tensor_phys_ket_left = (
            bottom_phys_ket_S[:, jnp.newaxis, jnp.newaxis] * bottom_phys_ket_Vh
        )

        (
            bottom_phys_bra_U,
            bottom_phys_bra_S,
            _,
            _,
            top_phys_bra_S,
            top_phys_bra_Vh,
        ) = _split_transfer_fishman(
            bottom_tensor_phys_bra_right, top_tensor_phys_bra_right, truncation_eps
        )
        top_tensor_phys_bra_right = (
            top_phys_bra_S[:, jnp.newaxis, jnp.newaxis] * top_phys_bra_Vh
        )
        bottom_tensor_phys_bra_right = (
            bottom_phys_bra_U * bottom_phys_bra_S[jnp.newaxis, jnp.newaxis, :]
        )

    (
        projector_left_bottom_phys_ket,
        projector_left_top_phys_ket,
        smallest_S_phys_ket,
    ) = _split_transfer_workhorse(
        bottom_tensor_phys_ket_left,
        top_tensor_phys_ket_left,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
    )

    (
        projector_right_top_phys_bra,
        projector_right_bottom_phys_bra,
        _,
    ) = _split_transfer_workhorse(
        top_tensor_phys_bra_right,
        bottom_tensor_phys_bra_right,
        peps_tensor_objs[0][1].interlayer_chi,
        truncation_eps,
    )

    top_tensor_phys_bra_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top_full",
        [peps_tensors[0][0], peps_tensors[0][1]],
        [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
        [projector_left_bottom_phys_ket, projector_right_bottom_phys_bra],
    )
    bottom_tensor_phys_bra_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom_full",
        [peps_tensors[0][0], peps_tensors[0][1]],
        [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
        [projector_left_top_phys_ket, projector_right_top_phys_bra],
    )
    top_tensor_phys_bra_left /= jnp.linalg.norm(top_tensor_phys_bra_left)
    bottom_tensor_phys_bra_left /= jnp.linalg.norm(bottom_tensor_phys_bra_left)

    if projector_method is Projector_Method.FISHMAN:
        (
            top_phys_bra_U,
            top_phys_bra_S,
            _,
            _,
            bottom_phys_bra_S,
            bottom_phys_bra_Vh,
        ) = _split_transfer_fishman(
            top_tensor_phys_bra_left, bottom_tensor_phys_bra_left, truncation_eps
        )

        top_tensor_phys_bra_left = (
            top_phys_bra_U * top_phys_bra_S[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )
        bottom_tensor_phys_bra_left = (
            bottom_phys_bra_S[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            * bottom_phys_bra_Vh
        )

    (
        projector_left_bottom_phys_bra,
        projector_left_top_phys_bra,
        smallest_S_phys_bra,
    ) = _split_transfer_workhorse(
        bottom_tensor_phys_bra_left,
        top_tensor_phys_bra_left,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
        fishman_input=projector_method is Projector_Method.FISHMAN,
    )

    return (
        Left_Projectors_Split_Transfer(
            top_ket=projector_left_top_ket,
            bottom_ket=projector_left_bottom_ket,
            top_bra=projector_left_top_bra,
            bottom_bra=projector_left_bottom_bra,
            top_phys_ket=projector_left_top_phys_ket,
            bottom_phys_ket=projector_left_bottom_phys_ket,
            top_phys_bra=projector_left_top_phys_bra,
            bottom_phys_bra=projector_left_bottom_phys_bra,
        ),
        smallest_S_ket,
        smallest_S_bra,
        smallest_S_phys_ket,
        smallest_S_phys_bra,
    )


def calc_right_projectors_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Right_Projectors_Split_Transfer:
    """
    Calculate the right projectors for the CTMRG method. This functions uses the
    CTMRG method with split transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The left top and bottom projectors for both layer.
    """
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for split transfer matrices approach."
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

    if projector_method is Projector_Method.FULL:
        (
            top_tensor_ketbra_left,
            bottom_tensor_ketbra_left,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.HALF:
        (
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)

        (
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            left_tensor_ketbra_bottom,
            right_tensor_ketbra_bottom,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.FISHMAN:
        (
            top_ketbra_U,
            top_ketbra_S,
            _,
            _,
            bottom_ketbra_S,
            bottom_ketbra_Vh,
        ) = _horizontal_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps
        )

        top_tensor_ketbra_left = (
            top_ketbra_U * top_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
        bottom_tensor_ketbra_left = (
            bottom_ketbra_S[:, jnp.newaxis, jnp.newaxis] * bottom_ketbra_Vh
        )

        (
            _,
            top_ketbra_S,
            top_ketbra_Vh,
            bottom_ketbra_U,
            bottom_ketbra_S,
            _,
        ) = _horizontal_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps, offset=1
        )
        top_tensor_ketbra_right = (
            top_ketbra_S[:, jnp.newaxis, jnp.newaxis] * top_ketbra_Vh
        )
        bottom_tensor_ketbra_right = (
            bottom_ketbra_U * bottom_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
    else:
        raise ValueError("Invalid projector method!")

    (
        projector_right_top_bra,
        projector_right_bottom_bra,
        smallest_S_bra,
    ) = _split_transfer_workhorse(
        top_tensor_ketbra_right,
        bottom_tensor_ketbra_right,
        chi,
        truncation_eps,
    )

    if (
        projector_method is Projector_Method.FULL
        or projector_method is Projector_Method.FISHMAN
    ):
        (
            projector_left_bottom_ket,
            projector_left_top_ket,
            _,
        ) = _split_transfer_workhorse(
            bottom_tensor_ketbra_left,
            top_tensor_ketbra_left,
            chi,
            truncation_eps,
        )

        top_tensor_ket_right = apply_contraction_jitted(
            "ctmrg_split_transfer_top_full",
            [peps_tensors[0][0], peps_tensors[0][1]],
            [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
            [projector_left_bottom_ket, projector_right_bottom_bra],
        )

        bottom_tensor_ket_right = apply_contraction_jitted(
            "ctmrg_split_transfer_bottom_full",
            [peps_tensors[1][0], peps_tensors[1][1]],
            [peps_tensor_objs[1][0], peps_tensor_objs[1][1]],
            [projector_left_top_ket, projector_right_top_bra],
        )

        top_tensor_ket_right /= jnp.linalg.norm(top_tensor_ket_right)
        bottom_tensor_ket_right /= jnp.linalg.norm(bottom_tensor_ket_right)
    elif projector_method is Projector_Method.HALF:
        (
            projector_top_left_ket,
            _,
            _,
        ) = _split_transfer_workhorse(
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
            chi,
            truncation_eps,
        )

        (
            _,
            projector_bottom_left_bra,
            _,
        ) = _split_transfer_workhorse(
            right_tensor_ketbra_bottom,
            left_tensor_ketbra_bottom,
            chi,
            truncation_eps,
        )

        top_tensor_ket_right = apply_contraction_jitted(
            "ctmrg_split_transfer_right_top_half",
            [peps_tensors[0][1]],
            [peps_tensor_objs[0][1]],
            [projector_top_left_ket, projector_right_bottom_bra],
        )

        bottom_tensor_ket_right = apply_contraction_jitted(
            "ctmrg_split_transfer_right_bottom_half",
            [peps_tensors[1][1]],
            [peps_tensor_objs[1][1]],
            [projector_bottom_left_bra, projector_right_top_bra],
        )

        top_tensor_ket_right /= jnp.linalg.norm(top_tensor_ket_right)
        bottom_tensor_ket_right /= jnp.linalg.norm(bottom_tensor_ket_right)

    if projector_method is Projector_Method.FISHMAN:
        (
            _,
            top_ket_S,
            top_ket_Vh,
            bottom_ket_U,
            bottom_ket_S,
            _,
        ) = _split_transfer_fishman(
            top_tensor_ket_right, bottom_tensor_ket_right, truncation_eps
        )

        top_tensor_ket_right = top_ket_S[:, jnp.newaxis, jnp.newaxis] * top_ket_Vh
        bottom_tensor_ket_right = (
            bottom_ket_U * bottom_ket_S[jnp.newaxis, jnp.newaxis, :]
        )

    (
        projector_right_top_ket,
        projector_right_bottom_ket,
        smallest_S_ket,
    ) = _split_transfer_workhorse(
        top_tensor_ket_right,
        bottom_tensor_ket_right,
        chi,
        truncation_eps,
    )

    top_tensor_phys_ket_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    bottom_tensor_phys_ket_left = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    top_tensor_phys_bra_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )
    bottom_tensor_phys_bra_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom",
        [peps_tensors[0][1]],
        [peps_tensor_objs[0][1]],
        [],
    )
    top_tensor_phys_bra_right = top_tensor_phys_bra_right.transpose(0, 1, 4, 2, 3)
    bottom_tensor_phys_bra_right = bottom_tensor_phys_bra_right.transpose(0, 1, 4, 2, 3)

    top_tensor_phys_ket_left /= jnp.linalg.norm(top_tensor_phys_ket_left)
    bottom_tensor_phys_ket_left /= jnp.linalg.norm(bottom_tensor_phys_ket_left)
    top_tensor_phys_bra_right /= jnp.linalg.norm(top_tensor_phys_bra_right)
    bottom_tensor_phys_bra_right /= jnp.linalg.norm(bottom_tensor_phys_bra_right)

    if (
        projector_method is Projector_Method.FISHMAN
        or projector_method is Projector_Method.HALF
    ):
        (
            top_phys_ket_U,
            top_phys_ket_S,
            _,
            _,
            bottom_phys_ket_S,
            bottom_phys_ket_Vh,
        ) = _split_transfer_fishman(
            top_tensor_phys_ket_left, bottom_tensor_phys_ket_left, truncation_eps
        )

        top_tensor_phys_ket_left = (
            top_phys_ket_U * top_phys_ket_S[jnp.newaxis, jnp.newaxis, :]
        )
        bottom_tensor_phys_ket_left = (
            bottom_phys_ket_S[:, jnp.newaxis, jnp.newaxis] * bottom_phys_ket_Vh
        )

        (
            bottom_phys_bra_U,
            bottom_phys_bra_S,
            _,
            _,
            top_phys_bra_S,
            top_phys_bra_Vh,
        ) = _split_transfer_fishman(
            bottom_tensor_phys_bra_right, top_tensor_phys_bra_right, truncation_eps
        )
        top_tensor_phys_bra_right = (
            top_phys_bra_S[:, jnp.newaxis, jnp.newaxis] * top_phys_bra_Vh
        )
        bottom_tensor_phys_bra_right = (
            bottom_phys_bra_U * bottom_phys_bra_S[jnp.newaxis, jnp.newaxis, :]
        )

    (
        projector_left_bottom_phys_ket,
        projector_left_top_phys_ket,
        _,
    ) = _split_transfer_workhorse(
        bottom_tensor_phys_ket_left,
        top_tensor_phys_ket_left,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
    )

    (
        projector_right_top_phys_bra,
        projector_right_bottom_phys_bra,
        smallest_S_phys_bra,
    ) = _split_transfer_workhorse(
        top_tensor_phys_bra_right,
        bottom_tensor_phys_bra_right,
        peps_tensor_objs[0][1].interlayer_chi,
        truncation_eps,
    )

    top_tensor_phys_ket_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_top_full",
        [peps_tensors[0][0], peps_tensors[0][1]],
        [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
        [projector_left_bottom_phys_ket, projector_right_bottom_phys_bra],
    )
    bottom_tensor_phys_ket_right = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_bottom_full",
        [peps_tensors[0][0], peps_tensors[0][1]],
        [peps_tensor_objs[0][0], peps_tensor_objs[0][1]],
        [projector_left_top_phys_ket, projector_right_top_phys_bra],
    )
    top_tensor_phys_ket_right /= jnp.linalg.norm(top_tensor_phys_ket_right)
    bottom_tensor_phys_ket_right /= jnp.linalg.norm(bottom_tensor_phys_ket_right)

    if projector_method is Projector_Method.FISHMAN:
        (
            bottom_phys_ket_U,
            bottom_phys_ket_S,
            _,
            _,
            top_phys_ket_S,
            top_phys_ket_Vh,
        ) = _split_transfer_fishman(
            bottom_tensor_phys_ket_right, top_tensor_phys_ket_right, truncation_eps
        )

        top_tensor_phys_ket_right = (
            top_phys_ket_S[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * top_phys_ket_Vh
        )
        bottom_tensor_phys_ket_right = (
            bottom_phys_ket_U
            * bottom_phys_ket_S[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )

    (
        projector_right_top_phys_ket,
        projector_right_bottom_phys_ket,
        smallest_S_phys_ket,
    ) = _split_transfer_workhorse(
        top_tensor_phys_ket_right,
        bottom_tensor_phys_ket_right,
        peps_tensor_objs[0][1].interlayer_chi,
        truncation_eps,
        fishman_input=projector_method is Projector_Method.FISHMAN,
    )

    return (
        Right_Projectors_Split_Transfer(
            top_ket=projector_right_top_ket,
            bottom_ket=projector_right_bottom_ket,
            top_bra=projector_right_top_bra,
            bottom_bra=projector_right_bottom_bra,
            top_phys_ket=projector_right_top_phys_ket,
            bottom_phys_ket=projector_right_bottom_phys_ket,
            top_phys_bra=projector_right_top_phys_bra,
            bottom_phys_bra=projector_right_bottom_phys_bra,
        ),
        smallest_S_ket,
        smallest_S_bra,
        smallest_S_phys_ket,
        smallest_S_phys_bra,
    )


def calc_top_projectors_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Top_Projectors_Split_Transfer:
    """
    Calculate the top projectors for the CTMRG method. This functions uses the
    CTMRG method with split transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The left top and bottom projectors for both layer.
    """
    if config.checkpointing_projectors:
        raise NotImplementedError(
            "Checkpointing not implemented for split transfer matrices approach."
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

    if projector_method is Projector_Method.FULL:
        (
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            left_tensor_ketbra_bottom,
            right_tensor_ketbra_bottom,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.HALF:
        (
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            top_tensor_ketbra_left,
            bottom_tensor_ketbra_left,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.FISHMAN:
        (
            _,
            left_ketbra_S,
            left_ketbra_Vh,
            right_ketbra_U,
            right_ketbra_S,
            _,
        ) = _vertical_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps
        )

        left_tensor_ketbra_top = (
            left_ketbra_S[:, jnp.newaxis, jnp.newaxis] * left_ketbra_Vh
        )
        right_tensor_ketbra_top = (
            right_ketbra_U * right_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )

        (
            left_ketbra_U,
            left_ketbra_S,
            _,
            _,
            right_ketbra_S,
            right_ketbra_Vh,
        ) = _vertical_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps, offset=1
        )
        left_tensor_ketbra_bottom = (
            left_ketbra_U * left_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
        right_tensor_ketbra_bottom = (
            right_ketbra_S[:, jnp.newaxis, jnp.newaxis] * right_ketbra_Vh
        )
    else:
        raise ValueError("Invalid projector method!")

    (
        projector_top_left_ket,
        projector_top_right_ket,
        smallest_S_ket,
    ) = _split_transfer_workhorse(
        left_tensor_ketbra_top,
        right_tensor_ketbra_top,
        chi,
        truncation_eps,
    )

    if (
        projector_method is Projector_Method.FULL
        or projector_method is Projector_Method.FISHMAN
    ):
        (
            projector_bottom_right_bra,
            projector_bottom_left_bra,
            _,
        ) = _split_transfer_workhorse(
            right_tensor_ketbra_bottom,
            left_tensor_ketbra_bottom,
            chi,
            truncation_eps,
        )

        left_tensor_bra_top = apply_contraction_jitted(
            "ctmrg_split_transfer_left_full",
            [peps_tensors[0][0], peps_tensors[1][0]],
            [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
            [projector_top_right_ket, projector_bottom_right_bra],
        )

        right_tensor_bra_top = apply_contraction_jitted(
            "ctmrg_split_transfer_right_full",
            [peps_tensors[0][1], peps_tensors[1][1]],
            [peps_tensor_objs[0][1], peps_tensor_objs[1][1]],
            [projector_top_left_ket, projector_bottom_left_bra],
        )

        left_tensor_bra_top /= jnp.linalg.norm(left_tensor_bra_top)
        right_tensor_bra_top /= jnp.linalg.norm(right_tensor_bra_top)
    elif projector_method is Projector_Method.HALF:
        (
            projector_left_bottom_ket,
            _,
            _,
        ) = _split_transfer_workhorse(
            bottom_tensor_ketbra_left,
            top_tensor_ketbra_left,
            chi,
            truncation_eps,
        )

        (
            _,
            projector_right_bottom_bra,
            _,
        ) = _split_transfer_workhorse(
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
            chi,
            truncation_eps,
        )

        left_tensor_bra_top = apply_contraction_jitted(
            "ctmrg_split_transfer_left_top_half",
            [peps_tensors[0][0]],
            [peps_tensor_objs[0][0]],
            [projector_left_bottom_ket, projector_top_right_ket],
        )

        right_tensor_bra_top = apply_contraction_jitted(
            "ctmrg_split_transfer_right_top_half",
            [peps_tensors[0][1]],
            [peps_tensor_objs[0][1]],
            [projector_top_left_ket, projector_right_bottom_bra],
        )

        left_tensor_bra_top /= jnp.linalg.norm(left_tensor_bra_top)
        right_tensor_bra_top /= jnp.linalg.norm(right_tensor_bra_top)

    if projector_method is Projector_Method.FISHMAN:
        (
            _,
            left_bra_S,
            left_bra_Vh,
            right_bra_U,
            right_bra_S,
            _,
        ) = _split_transfer_fishman(
            left_tensor_bra_top, right_tensor_bra_top, truncation_eps
        )

        left_tensor_bra_top = left_bra_S[:, jnp.newaxis, jnp.newaxis] * left_bra_Vh
        right_tensor_bra_top = right_bra_U * right_bra_S[jnp.newaxis, jnp.newaxis, :]

    (
        projector_top_left_bra,
        projector_top_right_bra,
        smallest_S_bra,
    ) = _split_transfer_workhorse(
        left_tensor_bra_top,
        right_tensor_bra_top,
        chi,
        truncation_eps,
    )

    left_tensor_phys_ket_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    right_tensor_phys_ket_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    left_tensor_phys_bra_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )
    right_tensor_phys_bra_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )
    left_tensor_phys_bra_bottom = left_tensor_phys_bra_bottom.transpose(0, 1, 4, 2, 3)
    right_tensor_phys_bra_bottom = right_tensor_phys_bra_bottom.transpose(0, 1, 4, 2, 3)

    left_tensor_phys_ket_top /= jnp.linalg.norm(left_tensor_phys_ket_top)
    right_tensor_phys_ket_top /= jnp.linalg.norm(right_tensor_phys_ket_top)
    left_tensor_phys_bra_bottom /= jnp.linalg.norm(left_tensor_phys_bra_bottom)
    right_tensor_phys_bra_bottom /= jnp.linalg.norm(right_tensor_phys_bra_bottom)

    if (
        projector_method is Projector_Method.FISHMAN
        or projector_method is Projector_Method.HALF
    ):
        (
            right_phys_ket_U,
            right_phys_ket_S,
            _,
            _,
            left_phys_ket_S,
            left_phys_ket_Vh,
        ) = _split_transfer_fishman(
            right_tensor_phys_ket_top, left_tensor_phys_ket_top, truncation_eps
        )

        right_tensor_phys_ket_top = (
            right_phys_ket_U * right_phys_ket_S[jnp.newaxis, jnp.newaxis, :]
        )
        left_tensor_phys_ket_top = (
            left_phys_ket_S[:, jnp.newaxis, jnp.newaxis] * left_phys_ket_Vh
        )

        (
            left_phys_bra_U,
            left_phys_bra_S,
            _,
            _,
            right_phys_bra_S,
            right_phys_bra_Vh,
        ) = _split_transfer_fishman(
            left_tensor_phys_bra_bottom, right_tensor_phys_bra_bottom, truncation_eps
        )
        left_tensor_phys_bra_bottom = (
            left_phys_bra_U * left_phys_bra_S[jnp.newaxis, jnp.newaxis, :]
        )
        right_tensor_phys_bra_bottom = (
            right_phys_bra_S[:, jnp.newaxis, jnp.newaxis] * right_phys_bra_Vh
        )

    (
        projector_top_left_phys_ket,
        projector_top_right_phys_ket,
        smallest_S_phys_ket,
    ) = _split_transfer_workhorse(
        left_tensor_phys_ket_top,
        right_tensor_phys_ket_top,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
    )

    (
        projector_bottom_right_phys_bra,
        projector_bottom_left_phys_bra,
        _,
    ) = _split_transfer_workhorse(
        right_tensor_phys_bra_bottom,
        left_tensor_phys_bra_bottom,
        peps_tensor_objs[1][0].interlayer_chi,
        truncation_eps,
    )

    left_tensor_phys_bra_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left_full",
        [peps_tensors[0][0], peps_tensors[1][0]],
        [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
        [projector_top_right_phys_ket, projector_bottom_right_phys_bra],
    )
    right_tensor_phys_bra_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right_full",
        [peps_tensors[0][0], peps_tensors[1][0]],
        [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
        [projector_top_left_phys_ket, projector_bottom_left_phys_bra],
    )
    left_tensor_phys_bra_top /= jnp.linalg.norm(left_tensor_phys_bra_top)
    right_tensor_phys_bra_top /= jnp.linalg.norm(right_tensor_phys_bra_top)

    if projector_method is Projector_Method.FISHMAN:
        (
            right_phys_bra_U,
            right_phys_bra_S,
            _,
            _,
            left_phys_bra_S,
            left_phys_bra_Vh,
        ) = _split_transfer_fishman(
            right_tensor_phys_bra_top, left_tensor_phys_bra_top, truncation_eps
        )

        right_tensor_phys_bra_top = (
            right_phys_bra_U
            * right_phys_bra_S[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )
        left_tensor_phys_bra_top = (
            left_phys_bra_S[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * left_phys_bra_Vh
        )

    (
        projector_top_left_phys_bra,
        projector_top_right_phys_bra,
        smallest_S_phys_bra,
    ) = _split_transfer_workhorse(
        left_tensor_phys_bra_top,
        right_tensor_phys_bra_top,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
        fishman_input=projector_method is Projector_Method.FISHMAN,
    )

    return (
        Top_Projectors_Split_Transfer(
            left_ket=projector_top_left_ket,
            right_ket=projector_top_right_ket,
            left_bra=projector_top_left_bra,
            right_bra=projector_top_right_bra,
            left_phys_ket=projector_top_left_phys_ket,
            right_phys_ket=projector_top_right_phys_ket,
            left_phys_bra=projector_top_left_phys_bra,
            right_phys_bra=projector_top_right_phys_bra,
        ),
        smallest_S_ket,
        smallest_S_bra,
        smallest_S_phys_ket,
        smallest_S_phys_bra,
    )


def calc_bottom_projectors_split_transfer(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
    config: VariPEPS_Config,
    state: VariPEPS_Global_State,
) -> Bottom_Projectors_Split_Transfer:
    """
    Calculate the bottom projectors for the CTMRG method. This functions uses the
    CTMRG method with split transfer matrices for the bra and ket layer.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The left top and bottom projectors for both layer.
    """
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

    if projector_method is Projector_Method.FULL:
        (
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            left_tensor_ketbra_bottom,
            right_tensor_ketbra_bottom,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.HALF:
        (
            left_tensor_ketbra_bottom,
            right_tensor_ketbra_bottom,
        ) = _vertical_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)

        (
            top_tensor_ketbra_left,
            bottom_tensor_ketbra_left,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs)

        (
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
        ) = _horizontal_cut_split_transfer(peps_tensors, peps_tensor_objs, offset=1)
    elif projector_method is Projector_Method.FISHMAN:
        (
            _,
            left_ketbra_S,
            left_ketbra_Vh,
            right_ketbra_U,
            right_ketbra_S,
            _,
        ) = _vertical_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps
        )

        left_tensor_ketbra_top = (
            left_ketbra_S[:, jnp.newaxis, jnp.newaxis] * left_ketbra_Vh
        )
        right_tensor_ketbra_top = (
            right_ketbra_U * right_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )

        (
            left_ketbra_U,
            left_ketbra_S,
            _,
            _,
            right_ketbra_S,
            right_ketbra_Vh,
        ) = _vertical_cut_split_transfer(
            peps_tensors, peps_tensor_objs, True, truncation_eps, offset=1
        )
        left_tensor_ketbra_bottom = (
            left_ketbra_U * left_ketbra_S[jnp.newaxis, jnp.newaxis, :]
        )
        right_tensor_ketbra_bottom = (
            right_ketbra_S[:, jnp.newaxis, jnp.newaxis] * right_ketbra_Vh
        )
    else:
        raise ValueError("Invalid projector method!")

    (
        projector_bottom_right_bra,
        projector_bottom_left_bra,
        smallest_S_bra,
    ) = _split_transfer_workhorse(
        right_tensor_ketbra_bottom,
        left_tensor_ketbra_bottom,
        chi,
        truncation_eps,
    )

    if (
        projector_method is Projector_Method.FULL
        or projector_method is Projector_Method.FISHMAN
    ):
        (
            projector_top_left_ket,
            projector_top_right_ket,
            _,
        ) = _split_transfer_workhorse(
            left_tensor_ketbra_top,
            right_tensor_ketbra_top,
            chi,
            truncation_eps,
        )

        left_tensor_ket_bottom = apply_contraction_jitted(
            "ctmrg_split_transfer_left_full",
            [peps_tensors[0][0], peps_tensors[1][0]],
            [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
            [projector_top_right_ket, projector_bottom_right_bra],
        )

        right_tensor_ket_bottom = apply_contraction_jitted(
            "ctmrg_split_transfer_right_full",
            [peps_tensors[0][1], peps_tensors[1][1]],
            [peps_tensor_objs[0][1], peps_tensor_objs[1][1]],
            [projector_top_left_ket, projector_bottom_left_bra],
        )

        left_tensor_ket_bottom /= jnp.linalg.norm(left_tensor_ket_bottom)
        right_tensor_ket_bottom /= jnp.linalg.norm(right_tensor_ket_bottom)
    elif projector_method is Projector_Method.HALF:
        (
            _,
            projector_left_top_ket,
            _,
        ) = _split_transfer_workhorse(
            bottom_tensor_ketbra_left,
            top_tensor_ketbra_left,
            chi,
            truncation_eps,
        )

        (
            projector_right_top_bra,
            _,
            _,
        ) = _split_transfer_workhorse(
            top_tensor_ketbra_right,
            bottom_tensor_ketbra_right,
            chi,
            truncation_eps,
        )

        left_tensor_ket_bottom = apply_contraction_jitted(
            "ctmrg_split_transfer_bottom_full",
            [peps_tensors[1][0], peps_tensors[1][1]],
            [peps_tensor_objs[1][0], peps_tensor_objs[1][1]],
            [projector_left_top_ket, projector_right_top_bra],
        )

        right_tensor_ket_bottom = apply_contraction_jitted(
            "ctmrg_split_transfer_right_bottom_half",
            [peps_tensors[1][1]],
            [peps_tensor_objs[1][1]],
            [projector_bottom_left_bra, projector_right_top_bra],
        )

        left_tensor_ket_bottom /= jnp.linalg.norm(left_tensor_ket_bottom)
        right_tensor_ket_bottom /= jnp.linalg.norm(right_tensor_ket_bottom)

    if projector_method is Projector_Method.FISHMAN:
        (
            left_ket_U,
            left_ket_S,
            _,
            _,
            right_ket_S,
            right_ket_Vh,
        ) = _split_transfer_fishman(
            left_tensor_ket_bottom, right_tensor_ket_bottom, truncation_eps
        )

        left_tensor_ket_bottom = left_ket_U * left_ket_S[jnp.newaxis, jnp.newaxis, :]
        right_tensor_ket_bottom = (
            right_ket_S[:, jnp.newaxis, jnp.newaxis] * right_ket_Vh
        )

    (
        projector_bottom_right_ket,
        projector_bottom_left_ket,
        smallest_S_ket,
    ) = _split_transfer_workhorse(
        right_tensor_ket_bottom,
        left_tensor_ket_bottom,
        chi,
        truncation_eps,
    )

    left_tensor_phys_ket_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )
    right_tensor_phys_ket_top = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right",
        [peps_tensors[0][0]],
        [peps_tensor_objs[0][0]],
        [],
    )

    left_tensor_phys_bra_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )
    right_tensor_phys_bra_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right",
        [peps_tensors[1][0]],
        [peps_tensor_objs[1][0]],
        [],
    )
    left_tensor_phys_bra_bottom = left_tensor_phys_bra_bottom.transpose(0, 1, 4, 2, 3)
    right_tensor_phys_bra_bottom = right_tensor_phys_bra_bottom.transpose(0, 1, 4, 2, 3)

    left_tensor_phys_ket_top /= jnp.linalg.norm(left_tensor_phys_ket_top)
    right_tensor_phys_ket_top /= jnp.linalg.norm(right_tensor_phys_ket_top)
    left_tensor_phys_bra_bottom /= jnp.linalg.norm(left_tensor_phys_bra_bottom)
    right_tensor_phys_bra_bottom /= jnp.linalg.norm(right_tensor_phys_bra_bottom)

    if (
        projector_method is Projector_Method.FISHMAN
        or projector_method is Projector_Method.HALF
    ):
        (
            right_phys_ket_U,
            right_phys_ket_S,
            _,
            _,
            left_phys_ket_S,
            left_phys_ket_Vh,
        ) = _split_transfer_fishman(
            right_tensor_phys_ket_top, left_tensor_phys_ket_top, truncation_eps
        )

        right_tensor_phys_ket_top = (
            right_phys_ket_U * right_phys_ket_S[jnp.newaxis, jnp.newaxis, :]
        )
        left_tensor_phys_ket_top = (
            left_phys_ket_S[:, jnp.newaxis, jnp.newaxis] * left_phys_ket_Vh
        )

        (
            left_phys_bra_U,
            left_phys_bra_S,
            _,
            _,
            right_phys_bra_S,
            right_phys_bra_Vh,
        ) = _split_transfer_fishman(
            left_tensor_phys_bra_bottom, right_tensor_phys_bra_bottom, truncation_eps
        )
        left_tensor_phys_bra_bottom = (
            left_phys_bra_U * left_phys_bra_S[jnp.newaxis, jnp.newaxis, :]
        )
        right_tensor_phys_bra_bottom = (
            right_phys_bra_S[:, jnp.newaxis, jnp.newaxis] * right_phys_bra_Vh
        )

    (
        projector_top_left_phys_ket,
        projector_top_right_phys_ket,
        _,
    ) = _split_transfer_workhorse(
        left_tensor_phys_ket_top,
        right_tensor_phys_ket_top,
        peps_tensor_objs[0][0].interlayer_chi,
        truncation_eps,
    )

    (
        projector_bottom_right_phys_bra,
        projector_bottom_left_phys_bra,
        smallest_S_phys_bra,
    ) = _split_transfer_workhorse(
        right_tensor_phys_bra_bottom,
        left_tensor_phys_bra_bottom,
        peps_tensor_objs[1][0].interlayer_chi,
        truncation_eps,
    )

    left_tensor_phys_ket_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_left_full",
        [peps_tensors[0][0], peps_tensors[1][0]],
        [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
        [projector_top_right_phys_ket, projector_bottom_right_phys_bra],
    )
    right_tensor_phys_ket_bottom = apply_contraction_jitted(
        "ctmrg_split_transfer_phys_right_full",
        [peps_tensors[0][0], peps_tensors[1][0]],
        [peps_tensor_objs[0][0], peps_tensor_objs[1][0]],
        [projector_top_left_phys_ket, projector_bottom_left_phys_bra],
    )
    left_tensor_phys_ket_bottom /= jnp.linalg.norm(left_tensor_phys_ket_bottom)
    right_tensor_phys_ket_bottom /= jnp.linalg.norm(right_tensor_phys_ket_bottom)

    if projector_method is Projector_Method.FISHMAN:
        (
            left_phys_bra_U,
            left_phys_bra_S,
            _,
            _,
            right_phys_bra_S,
            right_phys_bra_Vh,
        ) = _split_transfer_fishman(
            left_tensor_phys_ket_bottom, right_tensor_phys_ket_bottom, truncation_eps
        )

        left_tensor_phys_ket_bottom = (
            left_phys_bra_U * left_phys_bra_S[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )
        right_tensor_phys_ket_bottom = (
            right_phys_bra_S[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            * right_phys_bra_Vh
        )

    (
        projector_bottom_right_phys_ket,
        projector_bottom_left_phys_ket,
        smallest_S_phys_ket,
    ) = _split_transfer_workhorse(
        right_tensor_phys_ket_bottom,
        left_tensor_phys_ket_bottom,
        peps_tensor_objs[1][0].interlayer_chi,
        truncation_eps,
        fishman_input=projector_method is Projector_Method.FISHMAN,
    )

    return (
        Bottom_Projectors_Split_Transfer(
            left_ket=projector_bottom_left_ket,
            right_ket=projector_bottom_right_ket,
            left_bra=projector_bottom_left_bra,
            right_bra=projector_bottom_right_bra,
            left_phys_ket=projector_bottom_left_phys_ket,
            right_phys_ket=projector_bottom_right_phys_ket,
            left_phys_bra=projector_bottom_left_phys_bra,
            right_phys_bra=projector_bottom_right_phys_bra,
        ),
        smallest_S_ket,
        smallest_S_bra,
        smallest_S_phys_ket,
        smallest_S_phys_bra,
    )
