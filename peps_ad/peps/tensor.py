"""
Implementation of a single PEPS tensor
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from peps_ad.utils.random import PEPS_Random_Number_Generator

import typing
from typing import TypeVar, Type, Union, Optional, Sequence, Tuple, Any
from peps_ad.typing import Tensor, is_tensor

T_PEPS_Tensor = TypeVar("T_PEPS_Tensor", bound="PEPS_Tensor")


@dataclass(frozen=True)
@register_pytree_node_class
class PEPS_Tensor:
    """
    Class to model a single a PEPS tensor with the corresponding CTM tensors.

    Args:
      tensor (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        PEPS tensor
      C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C1 tensor
      C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C2 tensor
      C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C3 tensor
      C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C4 tensor
      T1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T1 tensor
      T2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T2 tensor
      T3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T3 tensor
      T4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T4 tensor
      d (:obj:`int`): Physical dimension of the PEPS tensor
      D (:term:`sequence` of :obj:`int`):
        Sequence of the bond dimensions of the PEPS tensor
      chi (:obj:`int`): Bond dimension for the CTM tensors
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
    D: Tuple[int, int, int, int]
    chi: int

    sanity_checks: bool = True

    def __post_init__(self) -> None:
        if not self.sanity_checks:
            return

        # Copied from https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses
        for field_name, field_def in self.__dataclass_fields__.items():  # type: ignore
            actual_value = getattr(self, field_name)
            if isinstance(actual_value, jax.core.Tracer):
                continue
            evaled_type = eval(field_def.type)
            if isinstance(evaled_type, typing._SpecialForm):
                # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
                continue
            try:
                actual_type = evaled_type.__origin__
            except AttributeError:
                actual_type = evaled_type
            if isinstance(actual_type, typing._SpecialForm):
                # case of typing.Union[…] or typing.ClassVar[…]
                actual_type = evaled_type.__args__
            if not isinstance(actual_value, actual_type):
                raise ValueError(
                    f"Invalid type for field '{field_name}'. Expected '{field_def.type}', got '{type(field_name)}.'"
                )

        if not len(self.D) == 4:
            raise ValueError(
                "The bond dimension of the PEPS tensor has to be a tuple of four entries."
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
    def from_tensor(
        cls: Type[T_PEPS_Tensor],
        tensor: Tensor,
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        *,
        ctm_tensors_are_identities: bool = True,
        normalize: bool = True,
        seed: Optional[int] = None,
        backend: str = "jax",
    ) -> T_PEPS_Tensor:
        """
        Initialize a PEPS tensor object with a given tensor and new CTM tensors.

        Args:
          tensor (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            PEPS tensor to initialize the object with
          d (:obj:`int`):
            Physical dimension
          D (:obj:`int` or :term:`sequence` of :obj:`int`):
            Bond dimensions for the PEPS tensor
          chi (:obj:`int`):
            Bond dimension for the environment tensors
        Keyword args:
          ctm_tensors_are_identities (:obj:`bool`, optional):
            Flag if the CTM tensors are initialized as identities. Otherwise,
            they are initialized randomly. Defaults to True.
          normalize (:obj:`bool`, optional):
            Flag if the generated tensors are normalized. Defaults to True.
          seed (:obj:`int`, optional):
            Seed for the random number generator.
          backend (:obj:`str`, optional):
            Backend for the generated tensors (may be ``jax`` or ``numpy``).
            Defaults to ``jax``.
        Returns:
          PEPS_Tensor:
            Instance of PEPS_Tensor with the randomly initialized tensors.
        """
        if not is_tensor(tensor):
            raise ValueError("Invalid argument for tensor.")

        if isinstance(D, int):
            D = (D,) * 4
        elif isinstance(D, collections.abc.Sequence) and not isinstance(D, tuple):
            D = tuple(D)

        if not all(isinstance(i, int) for i in D) or not len(D) == 4:
            raise ValueError("Invalid argument for D.")

        if (
            tensor.shape[0] != D[0]
            or tensor.shape[1] != D[1]
            or tensor.shape[3] != D[2]
            or tensor.shape[4] != D[3]
            or tensor.shape[2] != d
        ):
            raise ValueError("Tensor dimensions mismatch the dimension arguments.")

        dtype = tensor.dtype

        if ctm_tensors_are_identities:
            C1 = jnp.ones((1, 1), dtype=dtype)
            C2 = jnp.ones((1, 1), dtype=dtype)
            C3 = jnp.ones((1, 1), dtype=dtype)
            C4 = jnp.ones((1, 1), dtype=dtype)

            T1 = jnp.eye(D[3], dtype=dtype).reshape(1, D[3], D[3], 1)
            T2 = jnp.eye(D[2], dtype=dtype).reshape(D[2], D[2], 1, 1)
            T3 = jnp.eye(D[1], dtype=dtype).reshape(1, 1, D[1], D[1])
            T4 = jnp.eye(D[0], dtype=dtype).reshape(1, D[0], D[0], 1)
        else:
            rng = PEPS_Random_Number_Generator.get_generator(seed, backend=backend)

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
            D=D,  # type: ignore
            chi=chi,
        )

    @classmethod
    def random(
        cls: Type[T_PEPS_Tensor],
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        dtype: Union[Type[np.number], Type[jnp.number]],
        *,
        ctm_tensors_are_identities: bool = True,
        normalize: bool = True,
        seed: Optional[int] = None,
        backend: str = "jax",
    ) -> T_PEPS_Tensor:
        """
        Randomly initialize a PEPS tensor with CTM tensors.

        Args:
          d (:obj:`int`):
            Physical dimension
          D (:obj:`int` or :term:`sequence` of :obj:`int`):
            Bond dimensions for the PEPS tensor
          chi (:obj:`int`):
            Bond dimension for the environment tensors
          dtype (:obj:`numpy.dtype` or :obj:`jax.numpy.dtype`):
            Dtype of the generated tensors
        Keyword args:
          ctm_tensors_are_identities (:obj:`bool`, optional):
            Flag if the CTM tensors are initialized as identities. Otherwise,
            they are initialized randomly. Defaults to True.
          normalize (:obj:`bool`, optional):
            Flag if the generated tensors are normalized. Defaults to True.
          seed (:obj:`int`, optional):
            Seed for the random number generator.
          backend (:obj:`str`, optional):
            Backend for the generated tensors (may be ``jax`` or ``numpy``).
            Defaults to ``jax``.
        Returns:
          PEPS_Tensor:
            Instance of PEPS_Tensor with the randomly initialized tensors.
        """
        if isinstance(D, int):
            D = (D,) * 4
        elif isinstance(D, collections.abc.Sequence) and not isinstance(D, tuple):
            D = tuple(D)

        if not all(isinstance(i, int) for i in D) or not len(D) == 4:
            raise ValueError("Invalid argument for D.")

        rng = PEPS_Random_Number_Generator.get_generator(seed, backend=backend)

        tensor = rng.block((D[0], D[1], d, D[2], D[3]), dtype, normalize=normalize)

        if ctm_tensors_are_identities:
            C1 = jnp.ones((1, 1), dtype=dtype)
            C2 = jnp.ones((1, 1), dtype=dtype)
            C3 = jnp.ones((1, 1), dtype=dtype)
            C4 = jnp.ones((1, 1), dtype=dtype)

            T1 = jnp.eye(D[3], dtype=dtype).reshape(1, D[3], D[3], 1)
            T2 = jnp.eye(D[2], dtype=dtype).reshape(D[2], D[2], 1, 1)
            T3 = jnp.eye(D[1], dtype=dtype).reshape(1, 1, D[1], D[1])
            T4 = jnp.eye(D[0], dtype=dtype).reshape(1, D[0], D[0], 1)
        else:
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
            D=D,  # type: ignore
            chi=chi,
        )

    def replace_tensor(self: T_PEPS_Tensor, new_tensor: Tensor) -> T_PEPS_Tensor:
        """
        Replace the PEPS tensor and returns new object of the class.

        Args:
          new_tensor (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New PEPS tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor`:
            New instance of the class with the tensor replaced.
        """
        return type(self)(
            tensor=new_tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
        )

    def replace_left_env_tensors(
        self: T_PEPS_Tensor, new_C1: Tensor, new_T4: Tensor, new_C4: Tensor
    ) -> T_PEPS_Tensor:
        """
        Replace the left CTMRG tensors and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
          new_T4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T4 tensor.
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
            C2=self.C2,
            C3=self.C3,
            C4=new_C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=new_T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
        )

    def replace_right_env_tensors(
        self: T_PEPS_Tensor, new_C2: Tensor, new_T2: Tensor, new_C3: Tensor
    ) -> T_PEPS_Tensor:
        """
        Replace the right CTMRG tensors and returns new object of the class.

        Args:
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
          new_T2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T2 tensor.
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=new_C2,
            C3=new_C3,
            C4=self.C4,
            T1=self.T1,
            T2=new_T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
        )

    def replace_top_env_tensors(
        self: T_PEPS_Tensor, new_C1: Tensor, new_T1: Tensor, new_C2: Tensor
    ) -> T_PEPS_Tensor:
        """
        Replace the top CTMRG tensors and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
          new_T1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T1 tensor.
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
            C2=new_C2,
            C3=self.C3,
            C4=self.C4,
            T1=new_T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
        )

    def replace_bottom_env_tensors(
        self: T_PEPS_Tensor, new_C4: Tensor, new_T3: Tensor, new_C3: Tensor
    ) -> T_PEPS_Tensor:
        """
        Replace the bottom CTMRG tensors and returns new object of the class.

        Args:
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
          new_T3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T3 tensor.
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=new_C3,
            C4=new_C4,
            T1=self.T1,
            T2=self.T2,
            T3=new_T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
        )

    def tree_flatten(self) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
        field_names = tuple(
            i for i in self.__dataclass_fields__.keys() if i != "sanity_checks"  # type: ignore
        )
        field_values = tuple(getattr(self, name) for name in field_names)

        return (field_values, field_names)

    @classmethod
    def tree_unflatten(
        cls: Type[T_PEPS_Tensor], aux_data: Tuple[str, ...], children: Sequence[Any]
    ) -> T_PEPS_Tensor:
        return cls(**dict(jax.util.safe_zip(aux_data, children)), sanity_checks=False)
