import math

import jax
import jax.numpy as jnp
from jax import custom_jvp, lax

from varipeps import varipeps_config

from jax._src.lax.linalg import _tril

from .svd import _T, _H


@custom_jvp
def thin_qr(a):
    *_, m, n = a.shape

    if m < n:
        raise NotImplementedError

    return lax.linalg.qr(a, full_matrices=False, pivoting=False)


@thin_qr.defjvp
def _thin_qr_jvp(primals, tangents):
    """JVP for QR decompositions of [..., m, n] matrices with m >= n."""
    # Adapted from code in https://github.com/jax-ml/jax/blob/c33b9275c99a761ed0ce7efad1c8592f694d3102/jax/_src/lax/linalg.py
    # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.

    (x,) = primals
    (dx,) = tangents
    q, r = thin_qr(x)

    # dx_rinv = lax.linalg.triangular_solve(r, dx)  # Right side solve by default
    r_u, r_s, r_vh = jnp.linalg.svd(r, full_matrices=False)
    r_s_inv = jnp.where(r_s / r_s[0] >= 1e-12, 1 / r_s, 0)
    dx_rinv = dx @ ((r_vh.T.conj() * r_s_inv[jnp.newaxis, :]) @ r_u.T.conj())

    qt_dx_rinv = _H(q) @ dx_rinv
    qt_dx_rinv_lower = _tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric

    # The following correction is necessary for complex inputs
    n = r.shape[-1]
    I = lax.expand_dims(jnp.eye(n, dtype=do.dtype), range(qt_dx_rinv.ndim - 2))
    do = do + I * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))

    dq = q @ (do - qt_dx_rinv) + dx_rinv
    dr = (qt_dx_rinv - do) @ r

    return (q, r), (dq, dr)


def qr_oversampled_dim(chi: int, *dims: int) -> int:
    """
    Number of columns of the FULL_QR subspace isometries for a target bond
    dimension ``chi``. Clipped to the dimensions of the matrices the isometries
    act on, since the thin QR needs at least as many rows as columns.
    """
    p = max(
        math.ceil(chi * (1 + varipeps_config.ctmrg_qr_oversampling_factor)),
        chi + varipeps_config.ctmrg_qr_min_oversampling,
    )
    if len(dims) > 0:
        p = min(p, *dims)
    return int(p)


def qr_subspace_iteration(mats, qr_left, qr_right, n_iter: int):
    """
    Subspace iteration for the dominant singular subspaces of
    ``M = mats[0] @ mats[1] @ ... @ mats[-1]`` without forming M.

    The left and right isometries are updated in alternation
    (``Q_R <- orth(M^H Q_L^H)``, ``Q_L <- orth(M Q_R)``), so one iteration
    applies M and M^H once each. The former independent updates
    ``Q_L <- orth(Q_L M M^H)`` and ``Q_R <- orth(M^H M Q_R)`` together with
    ``Q_L M Q_R`` needed five applications of M per CTMRG step.

    Args:
      mats: Sequence of matrices whose product is M.
      qr_left: Left isometry with orthonormal rows, shape (p, rows of M).
      qr_right: Right isometry with orthonormal columns, shape (cols of M, p).
        Only used if ``n_iter == 0``.
      n_iter: Number of alternating iterations (static).
    Returns:
      Tuple of the new left (p, rows) and right (cols, p) isometries and the
      product ``M @ Q_R`` (rows, p), which can be reused to build the small
      matrix ``Q_L @ M @ Q_R`` without applying M again.
    """

    def apply_M(x):
        for m in reversed(mats):
            x = m @ x
        return x

    def apply_MH(x):
        for m in mats:
            x = m.T.conj() @ x
        return x

    if n_iter < 1:
        return qr_left, qr_right, apply_M(qr_right)

    for _ in range(n_iter):
        y = apply_MH(qr_left.T.conj())
        qr_right, _ = thin_qr(y / jnp.linalg.norm(y))

        mq_right = apply_M(qr_right)
        q, _ = thin_qr(mq_right / jnp.linalg.norm(mq_right))
        qr_left = q.T.conj()

    return qr_left, qr_right, mq_right
