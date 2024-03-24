import enum
from functools import partial

import jax.numpy as jnp
from jax import jit, checkpoint

from varipeps.peps import PEPS_Tensor
from varipeps.contractions import apply_contraction
from varipeps import varipeps_config
from varipeps.utils.func_cache import Checkpointing_Cache
from varipeps.utils.svd import gauge_fixed_svd
from varipeps.utils.projector_dict import (
    Left_Projectors,
    Right_Projectors,
    Top_Projectors,
    Bottom_Projectors,
)
from varipeps.config import Projector_Method, VariPEPS_Config
from varipeps.global_state import VariPEPS_Global_State

from typing import Sequence, Tuple, TypeVar


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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
