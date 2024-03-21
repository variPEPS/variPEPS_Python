from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from varipeps.peps import PEPS_Unit_Cell

from typing import Sequence, Union, List, Optional


@dataclass
class Expectation_Model(ABC):
    """
    Abstract model of an general expectation value calculated for the complete
    unit cell.

    Parameters:
      is_spiral_peps (:obj:`bool`):
        Flag if the expectation value is for a spiral iPEPS ansatz.
    """

    @abstractmethod
    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        spiral_vectors: Optional[Union[jnp.ndarray, Sequence[jnp.ndarray]]] = None,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        """
        Calculate the expectation value for PEPS unitcell depending on the gates
        set in the class.

        Args:
          peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
            The sequence of unique PEPS tensors in the unitcell.
          unitcell (:obj:`~varipeps.peps.PEPS_Unit_Cell`):
            The PEPS unitcell.
          spiral_vectors (single or :term:`sequence` of :obj:`jax.numpy.ndarray`):
            If the expectation value is for a spiral iPEPS ansatz, in this
            argument the wavevectors are expected.
        Keyword args:
          normalize_by_size (:obj:`bool`):
            Flag if the expectation value should be normalized by the number
            of tensors in the unitcell.
          only_unique (:obj:`bool`):
            Flag if the expectation value should be calculated just once for
            each unique PEPS tensor in the unitcell.
        Returns:
          :obj:`jax.numpy.ndarray` or :obj:`list` of :obj:`jax.numpy.ndarray`:
            The expectation values for all gates. Single tensor if only one gate
            is applied.
        """
        pass
