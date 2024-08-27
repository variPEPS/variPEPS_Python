from functools import partial

from varipeps.config import Wavevector_Type
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

from typing import Sequence, Union


@partial(jit, static_argnums=(5, 6, 7, 8))
def apply_unitary(
    gate: jnp.ndarray,
    delta_r: jnp.ndarray,
    q: Sequence[jnp.ndarray],
    unitary_operator_D: jnp.ndarray,
    unitary_operator_sigma: jnp.ndarray,
    phys_d: int,
    number_sites: int,
    apply_to_index: Sequence[int],
    wavevector_type: Wavevector_Type,
) -> jnp.ndarray:
    """
    Apply the unitary of the spiral iPEPS approach to a gate. The function
    calculates the relative unitary gate from the operator, the spatial
    difference and the wavevector.

    Args:
      gate (:obj:`jax.numpy.ndarray`):
        The gate which should be updated with the unitary operator.
      delta_r (:obj:`jax.numpy.ndarray`):
        Vector for the spatial difference. Can be a sequence if the spatial
        difference are different for the single indices.
      q (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the relevant wavevector for the different indices of
        the gate.
      unitary_operator_D (:obj:`jax.numpy.ndarray`):
        Array with the eigenvalues of the operator from which the unitary
        is generated.
      unitary_operator_sigma (:obj:`jax.numpy.ndarray`):
        Array with the eigenvectors of the operator from which the unitary
        is generated.
      phys_d (:obj:`int`):
        Physical dimension of the indices of the gate.
      number_sites (:obj:`int`):
        Number of sites the gate is applied to.
      apply_to_index (:term:`sequence` of :obj:`int`):
        The indices of the gate which should be modified by the unitary
        generated to the same ordered sequence of wavevectors.
      wavevector_type (:obj:`~varipeps.config.Wavevector_Type`):
        Type of the wavevector (see type definition for details).
    Returns:
      :obj:`jax.numpy.ndarray`:
        The updated gate with the unitary applied.
    """
    if isinstance(delta_r, jnp.ndarray):
        delta_r = (delta_r,) * len(apply_to_index)

    if len(q) != len(apply_to_index) or len(q) != len(delta_r):
        raise ValueError("Length mismatch!")

    working_gate = gate.reshape((phys_d,) * 2 * number_sites)

    for index, i in enumerate(apply_to_index):
        w_q = q[index]
        w_r = delta_r[index]

        if w_q.ndim == 0:
            w_q = jnp.array((w_q, w_q))
        elif w_q.size == 1:
            w_q = jnp.array((w_q[0], w_q[0]))

        if wavevector_type is Wavevector_Type.TWO_PI_POSITIVE_ONLY:
            w_q = w_q % 2
        elif wavevector_type is Wavevector_Type.TWO_PI_SYMMETRIC:
            w_q = w_q % 4 - 2
        else:
            raise ValueError("Unknown wavevector type!")

        # U = jsp.linalg.expm(1j * jnp.pi * jnp.dot(w_q, w_r) * unitary_operator)
        U = jnp.exp(1j * jnp.pi * jnp.dot(w_q, w_r) * unitary_operator_D)
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
