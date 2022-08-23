from collections import namedtuple
import enum
from functools import partial

import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor
from peps_ad.contractions import apply_contraction
from peps_ad import peps_ad_config, peps_ad_global_state
from peps_ad.utils.func_cache import Checkpointing_Cache
from peps_ad.utils.svd import gauge_fixed_svd
from peps_ad.config import Projector_Method

from typing import Sequence, Tuple, TypeVar


Left_Projectors = namedtuple("Left_Projectors", ("top", "bottom"))
Right_Projectors = namedtuple("Right_Projectors", ("top", "bottom"))
Top_Projectors = namedtuple("Top_Projectors", ("left", "right"))
Bottom_Projectors = namedtuple("Bottom_Projectors", ("left", "right"))

T_Projector = TypeVar(
    "T_Projector", Left_Projectors, Right_Projectors, Top_Projectors, Bottom_Projectors
)


class _Projectors_Func_Cache:
    _left = None
    _right = None
    _top = None
    _bottom = None

    def __class_getitem__(cls, name: str) -> Checkpointing_Cache:
        name = f"_{name}"
        obj = getattr(cls, name)
        if obj is None:
            obj = Checkpointing_Cache(peps_ad_config.checkpointing_projectors)
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

    S_sqrt = jnp.sqrt(S)
    relevant_S_values = S_sqrt > truncation_eps
    S_inv_sqrt = jnp.where(
        relevant_S_values, 1 / jnp.where(relevant_S_values, S_sqrt, 1), 0
    )

    return S_inv_sqrt, U, Vh


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


@partial(jit, static_argnums=(4, 5, 6))
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
        bottom_matrix /= jnp.linalg.norm(top_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(bottom_matrix, top_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi, truncation_eps)

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

    return Left_Projectors(top=projector_left_top, bottom=projector_left_bottom)


def calc_left_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Left_Projectors:
    """
    Calculate the left projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The left top and bottom projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if chi not in _Projectors_Func_Cache["left"]:
        _Projectors_Func_Cache["left"][chi] = partial(
            _left_projectors_workhorse, chi=chi
        )

    return _Projectors_Func_Cache["left"][chi](
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        peps_ad_config.ctmrg_truncation_eps
        if peps_ad_global_state.ctmrg_effective_truncation_eps is None
        else peps_ad_global_state.ctmrg_effective_truncation_eps,
        Projector_Method.FULL
        if peps_ad_global_state.ctmrg_projector_method is None
        else peps_ad_global_state.ctmrg_projector_method,
    )


@partial(jit, static_argnums=(4, 5, 6))
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
        bottom_matrix /= jnp.linalg.norm(top_matrix)
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(top_matrix, bottom_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi, truncation_eps)

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

    return Right_Projectors(top=projector_right_top, bottom=projector_right_bottom)


def calc_right_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Right_Projectors:
    """
    Calculate the right projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The right top and bottom projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if chi not in _Projectors_Func_Cache["right"]:
        _Projectors_Func_Cache["right"][chi] = partial(
            _right_projectors_workhorse, chi=chi
        )

    return _Projectors_Func_Cache["right"][chi](
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        peps_ad_config.ctmrg_truncation_eps
        if peps_ad_global_state.ctmrg_effective_truncation_eps is None
        else peps_ad_global_state.ctmrg_effective_truncation_eps,
        Projector_Method.FULL
        if peps_ad_global_state.ctmrg_projector_method is None
        else peps_ad_global_state.ctmrg_projector_method,
    )


@partial(jit, static_argnums=(4, 5, 6))
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
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(left_matrix, right_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi, truncation_eps)

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

    return Top_Projectors(left=projector_top_left, right=projector_top_right)


def calc_top_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Top_Projectors:
    """
    Calculate the top projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The top left and right projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if chi not in _Projectors_Func_Cache["top"]:
        _Projectors_Func_Cache["top"][chi] = partial(_top_projectors_workhorse, chi=chi)

    return _Projectors_Func_Cache["top"][chi](
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        peps_ad_config.ctmrg_truncation_eps
        if peps_ad_global_state.ctmrg_effective_truncation_eps is None
        else peps_ad_global_state.ctmrg_effective_truncation_eps,
        Projector_Method.FULL
        if peps_ad_global_state.ctmrg_projector_method is None
        else peps_ad_global_state.ctmrg_projector_method,
    )


@partial(jit, static_argnums=(4, 5, 6))
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
    else:
        raise ValueError("Invalid projector method!")

    product_matrix = jnp.dot(right_matrix, left_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi, truncation_eps)

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

    return Bottom_Projectors(left=projector_bottom_left, right=projector_bottom_right)


def calc_bottom_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Bottom_Projectors:
    """
    Calculate the bottom projectors for the CTMRG method.

    Args:
      peps_tensors (:term:`sequence` of :term:`sequence` of :obj:`jax.numpy.ndarray`):
        Nested list of the PEPS tensor arrays. The row (first) index corresponds
        to the x axis, the column (second) index to y.
      peps_tensor_objs (:term:`sequence` of :term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        Nested list of the PEPS tensor objects. Same format as for `peps_tensors`.
    Returns:
      :obj:`tuple`\ (:obj:`jax.numpy.ndarray`, :obj:`jax.numpy.ndarray`):
        The bottom left and right projectors.
    """
    chi = _check_chi(peps_tensor_objs)

    top_left, top_right, bottom_left, bottom_right = _calc_ctmrg_quarters(
        peps_tensors, peps_tensor_objs
    )

    if chi not in _Projectors_Func_Cache["bottom"]:
        _Projectors_Func_Cache["bottom"][chi] = partial(
            _bottom_projectors_workhorse, chi=chi
        )

    return _Projectors_Func_Cache["bottom"][chi](
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        peps_ad_config.ctmrg_truncation_eps
        if peps_ad_global_state.ctmrg_effective_truncation_eps is None
        else peps_ad_global_state.ctmrg_effective_truncation_eps,
        Projector_Method.FULL
        if peps_ad_global_state.ctmrg_projector_method is None
        else peps_ad_global_state.ctmrg_projector_method,
    )
