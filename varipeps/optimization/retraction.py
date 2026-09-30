"""
Utilities for the norm-preserving (Riemannian) optimization of the PEPS
tensors.

Each tensor :math:`A` which is optimized is restricted to the manifold of
tensors with fixed norm :math:`\\|A\\|`. Since the expectation value is
invariant under a rescaling of each individual tensor, this does not restrict
the variational space. Tangent vectors :math:`B` at :math:`A` satisfy
:math:`\\mathrm{Re}\\langle A, B \\rangle = 0`, a step along a tangent vector
is performed by a retraction and tangent vectors from previous steps (as needed
by the CG and (L-)BFGS methods) are moved to the new point by a vector
transport.

The implementation follows the functions ``norm_preserving_retract`` and
``norm_preserving_transport!`` of the PEPSKit.jl package.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from typing import Sequence, List, Tuple, Union


def _real_inner(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.vdot(a, b))


def _norm(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(_real_inner(a, a))


def _project_single(vec, tensor):
    return vec - (_real_inner(tensor, vec) / _real_inner(tensor, tensor)) * tensor


def _retract_single(tensor, direction, alpha):
    norm_tensor = _norm(tensor)
    norm_direction = _norm(direction)
    norm_direction_safe = jnp.where(norm_direction > 0, norm_direction, 1)

    theta = alpha * norm_direction / norm_tensor
    sn = jnp.sin(theta)
    cs = jnp.cos(theta)

    new_tensor = cs * tensor + (sn * norm_tensor / norm_direction_safe) * direction
    new_direction = cs * direction - (sn * norm_direction / norm_tensor) * tensor

    return new_tensor, new_direction


def _transport_single(vec, tensor, direction, alpha):
    norm_tensor = _norm(tensor)
    norm_direction = _norm(direction)
    norm_direction_safe = jnp.where(norm_direction > 0, norm_direction, 1)

    theta = alpha * norm_direction / norm_tensor
    sn = jnp.sin(theta)
    cs = jnp.cos(theta)

    overlap = _real_inner(direction, vec) / norm_direction_safe

    return (
        vec
        + ((cs - 1) * overlap / norm_direction_safe) * direction
        - (sn * overlap / norm_tensor) * tensor
    )


@partial(jit, static_argnums=(2,))
def project_to_tangent_space(
    vectors: Sequence[jnp.ndarray],
    peps_tensors: Sequence[jnp.ndarray],
    skip_indices: Tuple[int, ...] = (),
) -> List[jnp.ndarray]:
    """
    Orthogonal projection of vectors onto the tangent space of the manifold
    of fixed-norm tensors at the point `peps_tensors`:

    .. math::

      B \\to B - \\frac{\\mathrm{Re}\\langle A, B \\rangle}{\\|A\\|^2} A

    Args:
      vectors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the vectors (e.g. gradient or descent direction) which
        should be projected.
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors defining the point on the manifold.
      skip_indices (:obj:`tuple` of :obj:`int`):
        Indices of the elements which are not restricted to fixed norm (e.g.
        the wave vectors of spiral iPEPS). These elements are returned
        unchanged.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        The projected vectors.
    """
    return [
        v if i in skip_indices else _project_single(v, t)
        for i, (v, t) in enumerate(zip(vectors, peps_tensors, strict=True))
    ]


@partial(jit, static_argnums=(3,))
def norm_preserving_retract(
    peps_tensors: Sequence[jnp.ndarray],
    descent_dir: Sequence[jnp.ndarray],
    alpha: Union[float, jnp.ndarray],
    skip_indices: Tuple[int, ...] = (),
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Perform a norm-preserving retraction of each tensor `A` along the direction
    `η` with step size `α`, giving a new tensor

    .. math::

      A' = \\cos(\\alpha \\|\\eta\\| / \\|A\\|) A
           + \\sin(\\alpha \\|\\eta\\| / \\|A\\|) \\|A\\| \\eta / \\|\\eta\\|

    and the corresponding directional derivative

    .. math::

      \\xi = \\frac{d A'}{d \\alpha}
           = \\cos(\\alpha \\|\\eta\\| / \\|A\\|) \\eta
           - \\sin(\\alpha \\|\\eta\\| / \\|A\\|) \\|\\eta\\| A / \\|A\\|

    such that :math:`\\|A'\\| = \\|A\\|` and
    :math:`\\mathrm{Re}\\langle A', \\xi \\rangle = 0`. The direction `η` is
    expected to be a tangent vector, i.e.
    :math:`\\mathrm{Re}\\langle A, \\eta \\rangle = 0`.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the current tensors.
      descent_dir (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the descent direction for each tensor.
      alpha (:obj:`float` or :obj:`jax.numpy.ndarray`):
        The step size.
      skip_indices (:obj:`tuple` of :obj:`int`):
        Indices of the elements which are not restricted to fixed norm (e.g.
        the wave vectors of spiral iPEPS). For these elements the usual
        linear update is used.
    Returns:
      :obj:`tuple`\\ (:obj:`list`\\ (:obj:`jax.numpy.ndarray`), :obj:`list`\\ (:obj:`jax.numpy.ndarray`)):
        Tuple with the new tensors and the directional derivative at the new
        tensors.
    """
    new_tensors = []
    new_descent_dir = []

    for i, (t, d) in enumerate(zip(peps_tensors, descent_dir, strict=True)):
        if i in skip_indices:
            new_tensors.append(t + alpha * d)
            new_descent_dir.append(d)
        else:
            new_t, new_d = _retract_single(t, d, alpha)
            new_tensors.append(new_t)
            new_descent_dir.append(new_d)

    return new_tensors, new_descent_dir


@partial(jit, static_argnums=(4,))
def norm_preserving_transport(
    vectors: Sequence[jnp.ndarray],
    peps_tensors: Sequence[jnp.ndarray],
    descent_dir: Sequence[jnp.ndarray],
    alpha: Union[float, jnp.ndarray],
    skip_indices: Tuple[int, ...] = (),
) -> List[jnp.ndarray]:
    """
    Transport tangent vectors `ξ` at the tensors `A` to valid tangent vectors
    at the new tensors `A'` obtained by the norm-preserving retraction
    (see :obj:`norm_preserving_retract`) of `A` along `η` with step size `α`.

    Decomposing
    :math:`\\xi = \\langle \\eta / \\|\\eta\\|, \\xi \\rangle \\eta / \\|\\eta\\| + \\Delta\\xi`
    with :math:`\\langle \\Delta\\xi, A \\rangle = \\langle \\Delta\\xi, \\eta \\rangle = 0`,
    the result is

    .. math::

      \\xi(\\alpha) = \\langle \\eta / \\|\\eta\\|, \\xi \\rangle
      \\left( \\cos(\\alpha \\|\\eta\\| / \\|A\\|) \\eta / \\|\\eta\\|
      - \\sin(\\alpha \\|\\eta\\| / \\|A\\|) A / \\|A\\| \\right) + \\Delta\\xi

    such that :math:`\\|\\xi(\\alpha)\\| = \\|\\xi\\|` and
    :math:`\\mathrm{Re}\\langle A', \\xi(\\alpha) \\rangle = 0`.

    Args:
      vectors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tangent vectors which should be transported.
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors before the retraction.
      descent_dir (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the descent direction used in the retraction.
      alpha (:obj:`float` or :obj:`jax.numpy.ndarray`):
        The step size used in the retraction.
      skip_indices (:obj:`tuple` of :obj:`int`):
        Indices of the elements which are not restricted to fixed norm (e.g.
        the wave vectors of spiral iPEPS). These elements are returned
        unchanged.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        The transported vectors.
    """
    return [
        v if i in skip_indices else _transport_single(v, t, d, alpha)
        for i, (v, t, d) in enumerate(
            zip(vectors, peps_tensors, descent_dir, strict=True)
        )
    ]


@partial(jit, static_argnums=(1,))
def normalize_tensors(
    peps_tensors: Sequence[jnp.ndarray],
    skip_indices: Tuple[int, ...] = (),
) -> List[jnp.ndarray]:
    """
    Normalize each tensor to norm 1. Used as starting point of the
    norm-preserving optimization since the retraction keeps the norm of each
    tensor fixed and strongly differing norms lead to a badly conditioned
    optimization. This follows the function ``peps_normalize`` of the
    PEPSKit.jl package.

    Args:
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence of the tensors which should be normalized.
      skip_indices (:obj:`tuple` of :obj:`int`):
        Indices of the elements which are not restricted to fixed norm (e.g.
        the wave vectors of spiral iPEPS). These elements are returned
        unchanged.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        The normalized tensors.
    """
    return [
        t if i in skip_indices else t / _norm(t) for i, t in enumerate(peps_tensors)
    ]
