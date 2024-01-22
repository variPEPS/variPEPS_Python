from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

from typing import Sequence, Union


@partial(jit, static_argnums=(5, 6, 7))
def apply_unitary(
    gate: jnp.ndarray,
    delta_r: jnp.ndarray,
    q: Sequence[jnp.ndarray],
    unitary_operator_D: jnp.ndarray,
    unitary_operator_sigma: jnp.ndarray,
    phys_d: int,
    number_sites: int,
    apply_to_index: Sequence[int],
):
    if len(q) != len(apply_to_index):
        raise ValueError("Length mismatch!")

    working_gate = gate.reshape((phys_d,) * 2 * number_sites)

    for index, i in enumerate(apply_to_index):
        w_q = q[index]

        if w_q.ndim == 0:
            w_q = jnp.array((w_q, w_q))
        elif w_q.size == 1:
            w_q = jnp.array((w_q[0], w_q[0]))

        w_q = w_q % 4 - 2

        # U = jsp.linalg.expm(1j * jnp.pi * jnp.dot(w_q, delta_r) * unitary_operator)
        U = jnp.exp(1j * jnp.pi * jnp.dot(w_q, delta_r) * unitary_operator_D)
        U = jnp.dot(
            unitary_operator_sigma * U[jnp.newaxis, :], unitary_operator_sigma.T.conj()
        )

        working_gate = jnp.tensordot(U, working_gate, ((1,), (i,)))
        working_gate = jnp.tensordot(
            U.conj(), working_gate, ((1,), (number_sites + i,))
        )

        new_i_list = list(range(2, 2 * number_sites))
        new_i_list.insert(i, 1)
        new_i_list.insert(number_sites + i, 0)

        working_gate = working_gate.transpose(new_i_list)

    working_gate = working_gate.reshape(phys_d**number_sites, phys_d**number_sites)

    return working_gate
