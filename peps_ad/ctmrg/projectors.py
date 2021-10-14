import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor
from peps_ad.contractions import apply_contraction

from typing import Sequence, Tuple


def _check_chi(peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]]) -> int:
    chi = peps_tensor_objs[0][0].chi
    if not all(j.chi == chi for i in peps_tensor_objs for j in i):
        raise ValueError(
            "Environment bond dimension not the same over the whole network."
        )
    return chi


def _calc_ctmrg_quarters(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    matrix: jnp.ndarray, chi: int, eps: float = 1e-8
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Truncate the singular values
    newChi = jnp.sum((S[:chi] / S[0]) > eps)
    S = S[:newChi]
    U = U[:, :newChi]
    Vh = Vh[:newChi, :]

    S_inv_sqrt = 1 / jnp.sqrt(S)

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


@jit
def _left_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    chi: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    top_matrix, bottom_matrix = _horizontal_cut(
        top_left, top_right, bottom_left, bottom_right
    )

    product_matrix = jnp.dot(bottom_matrix, top_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi)

    rho_left_top = jnp.dot(top_matrix, Vh.transpose().conj() * S_inv_sqrt)
    rho_left_bottom = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], bottom_matrix
    )

    rho_left_top = rho_left_top.reshape(
        top_left.shape[0], top_left.shape[1], top_left.shape[2], chi
    )
    rho_left_bottom = rho_left_bottom.reshape(
        chi, bottom_left.shape[3], bottom_left.shape[4], bottom_left.shape[5]
    )

    return rho_left_bottom, rho_left_top


def calc_left_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    return _left_projectors_workhorse(
        top_left, top_right, bottom_left, bottom_right, chi
    )


@jit
def _right_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    chi: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    top_matrix, bottom_matrix = _horizontal_cut(
        top_left, top_right, bottom_left, bottom_right
    )

    product_matrix = jnp.dot(top_matrix, bottom_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi)

    rho_right_top = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], top_matrix
    )
    rho_right_bottom = jnp.dot(bottom_matrix, Vh.transpose().conj() * S_inv_sqrt)

    rho_right_top = rho_right_top.reshape(
        chi, top_right.shape[3], top_right.shape[4], top_right.shape[5]
    )
    rho_right_bottom = rho_right_bottom.reshape(
        bottom_right.shape[0], bottom_right.shape[1], bottom_right.shape[2], chi
    )

    return rho_right_bottom, rho_right_top


def calc_right_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    return _right_projectors_workhorse(
        top_left, top_right, bottom_left, bottom_right, chi
    )


@jit
def _top_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    chi: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_matrix, right_matrix = _vertical_cut(
        top_left, top_right, bottom_left, bottom_right
    )

    product_matrix = jnp.dot(left_matrix, right_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi)

    rho_top_left = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], left_matrix
    )
    rho_top_right = jnp.dot(right_matrix, Vh.transpose().conj() * S_inv_sqrt)

    rho_top_left = rho_top_left.reshape(
        chi, top_left.shape[3], top_left.shape[4], top_left.shape[5]
    )
    rho_top_right = rho_top_right.reshape(
        top_right.shape[0], top_right.shape[1], top_right.shape[2], chi
    )

    return rho_top_left, rho_top_right


def calc_top_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    return _top_projectors_workhorse(
        top_left, top_right, bottom_left, bottom_right, chi
    )


@jit
def _bottom_projectors_workhorse(
    top_left: jnp.ndarray,
    top_right: jnp.ndarray,
    bottom_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    chi: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_matrix, right_matrix = _vertical_cut(
        top_left, top_right, bottom_left, bottom_right
    )

    product_matrix = jnp.dot(right_matrix, left_matrix)

    S_inv_sqrt, U, Vh = _truncated_SVD(product_matrix, chi)

    rho_bottom_left = jnp.dot(left_matrix, Vh.transpose().conj() * S_inv_sqrt)
    rho_bottom_right = jnp.dot(
        U.transpose().conj() * S_inv_sqrt[:, jnp.newaxis], right_matrix
    )

    rho_bottom_left = rho_bottom_left.reshape(
        bottom_left.shape[0], bottom_left.shape[1], bottom_left.shape[2], chi
    )
    rho_bottom_right = rho_bottom_right.reshape(
        chi, bottom_right.shape[3], bottom_right.shape[4], bottom_right.shape[5]
    )

    return rho_bottom_left, rho_bottom_right


def calc_bottom_projectors(
    peps_tensors: Sequence[Sequence[jnp.ndarray]],
    peps_tensor_objs: Sequence[Sequence[PEPS_Tensor]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    return _bottom_projectors_workhorse(
        top_left, top_right, bottom_left, bottom_right, chi
    )
