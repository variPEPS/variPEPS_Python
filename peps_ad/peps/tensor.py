"""
Implementation of a single PEPS tensor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from peps_ad.utils.random import PEPS_Random

import typing
from typing import TypeVar, Type, Union, Optional, Sequence
from peps_ad.typing import Tensor

T_PEPS_Tensor = TypeVar("T_PEPS_Tensor", bound="PEPS_Tensor")


@dataclass
class PEPS_Tensor:
    """
    Class to model a single a PEPS tensor with the corresponding CTM tensors.

    Args:
      tensor (numpy.ndarray or jax.ndarray): PEPS tensor
      C1 (numpy.ndarray or jax.ndarray): C1 tensor
      C2 (numpy.ndarray or jax.ndarray): C2 tensor
      C3 (numpy.ndarray or jax.ndarray): C3 tensor
      C4 (numpy.ndarray or jax.ndarray): C4 tensor
      T1 (numpy.ndarray or jax.ndarray): T1 tensor
      T2 (numpy.ndarray or jax.ndarray): T2 tensor
      T3 (numpy.ndarray or jax.ndarray): T3 tensor
      T4 (numpy.ndarray or jax.ndarray): T4 tensor
      d (int): Physical dimension of the PEPS tensor
      D (sequence(int)): Sequence of the bond dimensions of the PEPS tensor
      chi (int): Bond dimension for the CTM tensors
    """

    tensor: Tensor

    C1: Tensor
    C2: Tensor
    C3: Tensor
    C4: Tensor

    T1: Tensor
    T2: Tensor
    T3: Tensor
    T4: Tensor

    d: int
    D: Sequence[int]
    chi: int

    def __post_init__(self) -> None:
        # Copied from https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses
        for field_name, field_def in self.__dataclass_fields__.items(): # type: ignore
            evaled_type = eval(field_def.type)
            if isinstance(evaled_type, typing._SpecialForm):
                # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
                continue
            actual_type = typing.get_origin(evaled_type) or evaled_type
            actual_value = getattr(self, field_name)
            if isinstance(actual_type, typing._SpecialForm):
                # case of typing.Union[…] or typing.ClassVar[…]
                actual_type = typing.get_args(evaled_type)
            if not isinstance(actual_value, actual_type):
                raise ValueError(
                    f"Invalid type for field '{field_name}'. Expected '{field_def.type}'"
                )

        if (
            self.tensor.shape[0] != self.D[0]
            or self.tensor.shape[1] != self.D[1]
            or self.tensor.shape[3] != self.D[2]
            or self.tensor.shape[4] != self.D[3]
        ):
            raise ValueError("Bond dimension sequence mismatches tensor.")

        if not (
            self.T1.shape[1] == self.T1.shape[2] == self.D[3]
            and self.T2.shape[0] == self.T2.shape[1] == self.D[2]
            and self.T3.shape[2] == self.T3.shape[3] == self.D[1]
            and self.T4.shape[1] == self.T4.shape[2] == self.D[0]
        ):
            raise ValueError(
                "At least one transfer tensors mismatch bond dimensions of PEPS tensor."
            )

    @classmethod
    def random(
        cls: Type[T_PEPS_Tensor],
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        dtype: Union[Type[np.number], Type[jnp.number]],
        *,
        normalize: bool = True,
        seed: Optional[int] = None,
        backend: str = "jax",
    ) -> T_PEPS_Tensor:
        """
        Randomly initialize a PEPS tensor with CTM tensors.

        Args:
          d (int): Physical dimension
          D (int or sequence(int)): Bond dimensions for the PEPS tensor
          chi (int): Bond dimension for the environment tensors
          dtype (numpy or jax.numpy.dtype): Dtype of the generated tensors
        Keyword args:
          normalize (bool, optional): Flag if the generated tensors are
                                      normalized. Default: True
          seed (int, optional): Seed for the random number generator.
          backend (str, optional): Backend for the generated tensors (may be
                                   "jax" or "numpy"). Default: "jax"
        Returns:
          Instance of PEPS_Tensor with the randomly initialized tensors.
        """
        if isinstance(D, int):
            D = [D] * 4

        if not all(isinstance(i, int) for i in D) or not len(D) == 4:
            raise ValueError("Invalid argument for D.")

        rng = PEPS_Random.get_generator(seed, backend=backend)

        tensor = rng.block((D[0], D[1], d, D[2], D[3]), dtype, normalize=normalize)

        C1 = rng.block((chi, chi), dtype, normalize=normalize)
        C2 = rng.block((chi, chi), dtype, normalize=normalize)
        C3 = rng.block((chi, chi), dtype, normalize=normalize)
        C4 = rng.block((chi, chi), dtype, normalize=normalize)

        T1 = rng.block((chi, D[3], D[3], chi), dtype, normalize=normalize)
        T2 = rng.block((D[2], D[2], chi, chi), dtype, normalize=normalize)
        T3 = rng.block((chi, chi, D[1], D[1]), dtype, normalize=normalize)
        T4 = rng.block((chi, D[0], D[0], chi), dtype, normalize=normalize)

        return cls(
            tensor=tensor,
            C1=C1,
            C2=C2,
            C3=C3,
            C4=C4,
            T1=T1,
            T2=T2,
            T3=T3,
            T4=T4,
            d=d,
            D=D,
            chi=chi,
        )
