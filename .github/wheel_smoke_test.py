# Test of a built wheel (cibuildwheel test-command): variPEPS imports, and the
# tblis backend (not on Windows) agrees with jnp.einsum, also in a gradient.
from io import BytesIO
import sys

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from varipeps.config import Contraction_Backend
from varipeps.contractions import tblis
from varipeps.peps import PEPS_Unit_Cell

with h5py.File(BytesIO(), "w") as f:
    config = f.create_group("config")
    for backend in Contraction_Backend:
        config.attrs["contraction_backend"] = backend
        loaded = PEPS_Unit_Cell.load_from_group(
            f, return_unitcell=False, return_config=True
        )
        assert loaded.contraction_backend is backend

if sys.platform == "win32":
    assert tblis._tblis_contract_lib is None
    print("Windows: variPEPS imported, tblis not built")
    sys.exit(0)

keys = jax.random.split(jax.random.PRNGKey(0), 3)
for dtype in (jnp.float64, jnp.complex128):
    a = jax.random.normal(keys[0], (6, 5, 7), dtype)
    b = jax.random.normal(keys[1], (7, 5, 8, 3), dtype)
    for spec in ("ijk,kjlm->ilm", "ijk,kjlm->mli", "ijk,kjlm->jilm"):
        err = jnp.max(jnp.abs(tblis.einsum(spec, a, b) - jnp.einsum(spec, a, b)))
        assert err < 1e-12, (dtype, spec, err)

x = jnp.array([2e20], dtype=jnp.float32)
y = jnp.ones(1, dtype=jnp.float64)
path = [(0, 1), (0, 1)]
assert tblis.einsum("i,i,i->", x, x, y, optimize=path) == jnp.einsum(
    "i,i,i->", x, x, y, optimize=path
)
x = jnp.array([[1e8], [1], [-1e8]], dtype=jnp.float32)
assert tblis.einsum("ij,j->", x, y) == jnp.einsum("ij,j->", x, y)

a = jax.random.normal(keys[2], (40, 30))
grad = jax.grad(lambda x: jnp.sum(tblis.einsum("ij,jk->ik", x, x.T) ** 2))(a)
grad_ref = jax.grad(lambda x: jnp.sum(jnp.einsum("ij,jk->ik", x, x.T) ** 2))(a)
assert jnp.max(jnp.abs(grad - grad_ref)) < 1e-9 * jnp.max(jnp.abs(grad_ref))
print("tblis backend OK, cost model:", tblis.cost_model(jnp.float64))
