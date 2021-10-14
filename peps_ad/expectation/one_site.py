import jax.numpy as jnp
from jax import jit

from peps_ad.peps import PEPS_Tensor
from peps_ad.contractions import apply_contraction

from typing import Sequence, List, Tuple


def _one_site_workhorse_body(
    density_matrix: jnp.ndarray,
    gates: Tuple[jnp.ndarray, ...],
    real_result: bool = False,
) -> List[jnp.ndarray]:
    norm = jnp.trace(density_matrix)

    if real_result:
        return [
            jnp.real(jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm)
            for g in gates
        ]
    else:
        return [
            jnp.tensordot(density_matrix, g, ((0, 1), (0, 1))) / norm for g in gates
        ]


_one_site_workhorse = jit(_one_site_workhorse_body, static_argnums=(2,))


def calc_one_site_multi_gates(
    peps_tensor: jnp.ndarray, peps_tensor_obj: PEPS_Tensor, gates: Sequence[jnp.ndarray]
) -> List[jnp.ndarray]:
    """
    Calculate the one site expectation values for a PEPS tensor and its
    environment.

    Args:
      peps_tensor (:obj:`jax.numpy.ndarray`):
        The PEPS tensor array. Have to be the same object as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_obj (:obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor object.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates which should be applied to the PEPS tensor.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        List with the calculated expectation values of each gate.
    """
    density_matrix = apply_contraction(
        "density_matrix_one_site", [peps_tensor], [peps_tensor_obj], []
    )

    real_result = all(jnp.allclose(g, jnp.real(g)) for g in gates)

    return _one_site_workhorse(density_matrix, tuple(gates), real_result)


def calc_one_site_single_gate(
    peps_tensor: jnp.ndarray, peps_tensor_obj: PEPS_Tensor, gate: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the one site expectation values for a PEPS tensor and its
    environment.

    This function just wraps :obj:`~peps_ad.expectation.calc_one_site_multi_gates`.

    Args:
      peps_tensor (:obj:`jax.numpy.ndarray`):
        The PEPS tensor array. Have to be the same object as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_obj (:obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor object.
      gates (:obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensor.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Expectation value for the gate.
    """
    return calc_one_site_multi_gates(peps_tensor, peps_tensor_obj, [gate])[0]


def calc_one_site_multi_gates_obj(
    peps_tensor_obj: PEPS_Tensor, gates: Sequence[jnp.ndarray]
) -> List[jnp.ndarray]:
    """
    Calculate the one site expectation values for a PEPS tensor and its
    environment.

    This function just wraps :obj:`~peps_ad.expectation.calc_one_site_multi_gates`.

    Args:
      peps_tensor_obj (:obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor object.
      gates (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Sequence with the gates which should be applied to the PEPS tensor.
    Returns:
      :obj:`list` of :obj:`jax.numpy.ndarray`:
        List with the calculated expectation values of each gate.
    """
    return calc_one_site_multi_gates(
        jnp.asarray(peps_tensor_obj.tensor), peps_tensor_obj, gates
    )


def calc_one_site_single_gate_obj(
    peps_tensor_obj: PEPS_Tensor, gate: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the one site expectation values for a PEPS tensor and its
    environment.

    This function just wraps :obj:`~peps_ad.expectation.calc_one_site_single_gate`.

    Args:
      peps_tensor_obj (:obj:`~peps_ad.peps.PEPS_Tensor`):
        PEPS tensor object.
      gates (:obj:`jax.numpy.ndarray`):
        Gate which should be applied to the PEPS tensor.
    Returns:
      :obj:`jax.numpy.ndarray`:
        Expectation value for the gate.
    """
    return calc_one_site_single_gate(
        jnp.asarray(peps_tensor_obj.tensor), peps_tensor_obj, gate
    )
