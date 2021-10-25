from abc import ABC, abstractmethod

import jax.numpy as jnp

from peps_ad.peps import PEPS_Unit_Cell

from typing import Sequence, Union, List


class Expectation_Model(ABC):
    """
    Abstract model of an general expectation value calculated for the complete
    unit cell.
    """

    @abstractmethod
    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        pass
