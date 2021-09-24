"""
Utility functions for initialize tensors randomly
"""

from __future__ import annotations

import abc
import random
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from typing import Type, Sequence, Optional
from peps_ad.typing import Tensor


class PEPS_Random_Number_Generator:
    """
    Class to maintain a global instance of random number generators
    """

    __instance_jax: Optional[PEPS_Random_Impl] = None
    __instance_np: Optional[PEPS_Random_Impl] = None

    @classmethod
    def get_generator(
        cls, seed: Optional[int] = None, *, backend: str = "jax"
    ) -> PEPS_Random_Impl:
        """
        Class method to obtain the instance of the random number generator for
        the specific backend

        Args:
          seed (int, optional): Seed for the random number generator. If there
                                is already a instance of the generator, this
                                parameter is ignored.
        Keyword args:
          backend (str, optional): Backend which should be used as random number
                                   generator. May be "jax" or "numpy".
                                   Default: "jax".
        Returns:
          Instance of random number generator implementation.
        """
        if backend == "jax":
            if cls.__instance_jax is None:
                cls.__instance_jax = PEPS_Jax_Random(seed)
            elif seed is not None:
                warnings.warn(
                    "There is already a random generator instance. The seed parameter is ignored. "
                    "If you want to set a new seed, please call the destroy_state() method before."
                )
            return cls.__instance_jax
        elif backend == "numpy":
            if cls.__instance_np is None:
                cls.__instance_np = PEPS_Numpy_Random(seed)
            elif seed is not None:
                warnings.warn(
                    "There is already a random generator instance. The seed parameter is ignored. "
                    "If you want to set a new seed, please call the destroy_state() method before."
                )
            return cls.__instance_np
        else:
            raise ValueError("Unknown backend")

    @classmethod
    def destroy_state(cls) -> None:
        """
        Destroy the current state of the random number generator.
        """
        cls.__instance_jax = None
        cls.__instance_np = None


class PEPS_Random_Impl(abc.ABC):
    """
    Abstract base class for random number generator implementation.
    """

    @abc.abstractmethod
    def block(
        self, dim: Sequence[int], dtype: Type[np.number], *, normalize: bool = True
    ) -> Tensor:
        """
        Generate a tensor with random numbers.

        Args:
          dim (sequence(int)): Sequence with the dimensions of the tensor
          dtype (numpy or jax.numpy.dtype): Dtype of the generated tensors
        Keyword args:
          normalize (bool, optional): Flag if the generated tensors are
                                      normalized. Default: True
        """
        pass


class PEPS_Numpy_Random(PEPS_Random_Impl):
    """
    Numpy implementation for the random number generator.

    Args:
      seed (int, optional): Seed for the generator.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def block(
        self, dim: Sequence[int], dtype: Type[np.number], *, normalize: bool = True
    ) -> np.ndarray:
        if (
            np.dtype(dtype) is np.dtype(np.complex64)
            or np.dtype(dtype) is np.dtype(np.complex128)
            or np.dtype(dtype) is np.dtype(np.complex256)
        ):
            block = self.rng.uniform(-1, 1, dim).astype(dtype) + 1j * self.rng.uniform(
                -1, 1, dim
            ).astype(dtype)
        else:
            block = self.rng.uniform(-1, 1, dim, dtype=dtype)  # type: ignore
        if normalize:
            block /= np.linalg.norm(block)
        return block


class PEPS_Jax_Random(PEPS_Random_Impl):
    """
    Jax implementation for the random number generator.

    Args:
      seed (int, optional): Seed for the generator.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = jnp.uint64(random.randint(0, 2**64-1))

        self.key = jax.random.PRNGKey(seed)

    def block(
        self, dim: Sequence[int], dtype: Type[np.number], *, normalize: bool = True
    ) -> jnp.ndarray:
        if jnp.dtype(dtype) is jnp.dtype(jnp.complex64) or jnp.dtype(
            dtype
        ) is jnp.dtype(jnp.complex128):
            self.key, key1, key2 = jax.random.split(self.key, 3)
            block = jax.random.uniform(key1, dim, minval=-1, maxval=1).astype(
                dtype
            ) + 1j * jax.random.uniform(key2, dim, minval=-1, maxval=1).astype(dtype)
        else:
            self.key, key1 = jax.random.split(self.key, 2)
            block = jax.random.uniform(key1, dim, dtype=dtype, minval=-1, maxval=1)
        if normalize:
            block /= jnp.linalg.norm(block)
        return block
