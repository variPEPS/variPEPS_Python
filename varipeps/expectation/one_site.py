from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction
from .model import Expectation_Model

from typing import Sequence, List, Tuple, Union


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
      peps_tensor_obj (:obj:`~varipeps.peps.PEPS_Tensor`):
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

    real_result = all(jnp.allclose(g, g.T.conj()) for g in gates)

    return _one_site_workhorse(density_matrix, tuple(gates), real_result)


def calc_one_site_single_gate(
    peps_tensor: jnp.ndarray, peps_tensor_obj: PEPS_Tensor, gate: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the one site expectation values for a PEPS tensor and its
    environment.

    This function just wraps :obj:`~varipeps.expectation.calc_one_site_multi_gates`.

    Args:
      peps_tensor (:obj:`jax.numpy.ndarray`):
        The PEPS tensor array. Have to be the same object as the tensor
        attribute of the `peps_tensor_obj` argument.
      peps_tensor_obj (:obj:`~varipeps.peps.PEPS_Tensor`):
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

    This function just wraps :obj:`~varipeps.expectation.calc_one_site_multi_gates`.

    Args:
      peps_tensor_obj (:obj:`~varipeps.peps.PEPS_Tensor`):
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

    This function just wraps :obj:`~varipeps.expectation.calc_one_site_single_gate`.

    Args:
      peps_tensor_obj (:obj:`~varipeps.peps.PEPS_Tensor`):
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


@dataclass
class One_Site_Expectation_Value(Expectation_Model):
    gates: Sequence[jnp.ndarray]

    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result_type = (
            jnp.float64
            if all(jnp.allclose(g, jnp.real(g)) for g in self.gates)
            else jnp.complex128
        )
        result = [jnp.array(0, dtype=result_type) for _ in range(len(self.gates))]

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                working_tensor = peps_tensors[view.get_indices((0, 0))[0][0]]
                working_tensor_obj = view[0, 0][0][0]

                step_result = calc_one_site_multi_gates(
                    working_tensor, working_tensor_obj, self.gates
                )

                for sr_i, sr in enumerate(step_result):
                    result[sr_i] += sr

        if normalize_by_size:
            if only_unique:
                size = unitcell.get_len_unique_tensors()
            else:
                size = unitcell.get_size()[0] * unitcell.get_size()[1]
            result = [r / size for r in result]

        if len(result) == 1:
            return result[0]
        else:
            return result
