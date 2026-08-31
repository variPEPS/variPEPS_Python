import jax
import jax.numpy as jnp
from jax import custom_jvp, lax

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
