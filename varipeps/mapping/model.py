from abc import ABC, abstractmethod

import jax.numpy as jnp

from varipeps.peps import PEPS_Unit_Cell

from typing import Sequence, Union, List, Tuple


class Map_To_PEPS_Model(ABC):
    """
    Abstract model of an general callable class to map a sequence of tensors to
    a PEPS unitcell.
    """

    @abstractmethod
    def __call__(
        self,
        input_tensors: Sequence[jnp.ndarray],
        *,
        generate_unitcell: bool = True,
    ) -> Union[List[jnp.ndarray], Tuple[List[jnp.ndarray], PEPS_Unit_Cell]]:
        """
        Calculate the PEPS unitcell out of a list of input tensors depending on
        the original systems.

        Args:
          input_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
            The sequence of input tensors that should be converted.
        Keyword args:
          generate_unitcell (:obj:`bool`):
            Flag if the only the PEPS tensors or additionally the unitcell
            should be calculated.
        Returns:
          :obj:`list` of :obj:`jax.numpy.ndarray` or :obj:`tuple` of :obj:`list` of :obj:`jax.numpy.ndarray` and :obj:`~varipeps.peps.PEPS_Unit_Cell`:
            The mapped PEPS tensors and if ``generate_unitcell = True`` the
            unitcell for these tensors.
        """
        pass
