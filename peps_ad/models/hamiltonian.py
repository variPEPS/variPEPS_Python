"""
Design ideas for gates and Hamiltonians
"""

import abc
import collections
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from typing import Union, Sequence


@dataclass
class PEPS_Gate(abc.ABC):
    """
    Abstract base class for a PEPS gate.

    Args:
      size (int or sequence(int)): Size of the gate. Either single integer
                                   describing the length in x-direction or a
                                   sequence of integers with each element the
                                   length in x-direction and len(sequence) the
                                   length of the gate in y-direction.
    """

    size: Union[int, Sequence[int]]

    def __post_init__(self) -> None:
        if not isinstance(self.size, (int, collections.abc.Sequence)):
            raise ValueError("Invalid type of argument size.")

        if (isinstance(self.size, int) and self.size >= 1) or all(
            i >= 1 for i in self.size
        ):
            raise ValueError(
                "Negative or zero values encountered in specification of gate size"
            )

    @abc.abstractmethod
    def get_gate(self) -> Union[np.ndarray, jnp.ndarray]:
        """
        Return tensor for the PEPS gate.

        The convention of the indices is that first indices for the PEPS layer
        are returned and then the indices for the conjugated layer.
        For each layer

        Returns:
          numpy or jax tensor for the gate with the above discussed convention.
        """
        pass


class PEPS_Hamiltonian(PEPS_Gate):
    """
    Base class for a PEPS Hamiltonian.
    """

    def get_H(self) -> Union[np.ndarray, jnp.ndarray]:
        """
        Return the tensor for the Hamiltonian.

        Normally just a wrapper around get_gate(). See there for details.

        Returns:
          numpy or jax tensor for the Hamiltonian.
        """
        return self.get_gate()


class PEPS_Two_Body_Gate(PEPS_Gate):
    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.Lx == 2 and self.Ly == 1 or self.Lx == 1 and self.Ly == 2):
            raise ValueError("Not a two-body gate")


@dataclass
class PEPS_Hamiltonian_Ising(PEPS_Hamiltonian, PEPS_Two_Body_Gate):
    """
    Implementation of a two-body Ising Hamiltonian.

    Args:
      J (float): Interaction strength
    """

    J: float

    def get_gate(self) -> jnp.ndarray:
        Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.int8)

        return self.J * jnp.kron(Z, Z)
