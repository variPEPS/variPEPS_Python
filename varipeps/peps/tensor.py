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
import h5py

from varipeps.utils.random import PEPS_Random_Number_Generator
from varipeps.utils.svd import gauge_fixed_svd

import typing
from typing import TypeVar, Type, Union, Optional, Sequence, Tuple, Any
from varipeps.typing import Tensor, is_tensor

T_PEPS_Tensor = TypeVar("T_PEPS_Tensor", bound="PEPS_Tensor")
T_PEPS_Tensor_Split_Transfer = TypeVar(
    "T_PEPS_Tensor_Split_Transfer", bound="PEPS_Tensor_Split_Transfer"
)


@dataclass
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
      d (:obj:`int`):
        Physical dimension of the PEPS tensor
      D (:term:`sequence` of :obj:`int`):
        Sequence of the bond dimensions of the PEPS tensor
      chi (:obj:`int`):
        Bond dimension for the CTM tensors
      max_chi (:obj:`int`):
        Maximal allowed bond dimension of environment tensors.
    """

    tensor: Tensor

    C1: Tensor
    C2: Tensor
    C3: Tensor
    C4: Tensor

    d: int
    D: Tuple[int, int, int, int]
    chi: int
    max_chi: int

    T1: Tensor
    T2: Tensor
    T3: Tensor
    T4: Tensor

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

    @property
    def left_upper_transfer_shape(self) -> Tensor:
        return self.T4.shape[3]

    @property
    def left_lower_transfer_shape(self) -> Tensor:
        return self.T4.shape[0]

    @property
    def right_upper_transfer_shape(self) -> Tensor:
        return self.T2.shape[3]

    @property
    def right_lower_transfer_shape(self) -> Tensor:
        return self.T2.shape[2]

    @property
    def top_left_transfer_shape(self) -> Tensor:
        return self.T1.shape[0]

    @property
    def top_right_transfer_shape(self) -> Tensor:
        return self.T1.shape[3]

    @property
    def bottom_left_transfer_shape(self) -> Tensor:
        return self.T3.shape[0]

    @property
    def bottom_right_transfer_shape(self) -> Tensor:
        return self.T3.shape[1]

    @classmethod
    def from_tensor(
        cls: Type[T_PEPS_Tensor],
        tensor: Tensor,
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        max_chi: Optional[int] = None,
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
          max_chi (:obj:`int`):
            Maximal allowed bond dimension for the environment tensors
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

        if max_chi is None:
            max_chi = chi

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
            max_chi=max_chi,
        )

    @classmethod
    def random(
        cls: Type[T_PEPS_Tensor],
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        dtype: Union[Type[np.number], Type[jnp.number]],
        max_chi: Optional[int] = None,
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
          max_chi (:obj:`int`):
            Maximal allowed bond dimension for the environment tensors
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

        if max_chi is None:
            max_chi = chi

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
            max_chi=max_chi,
        )

    def replace_tensor(
        self: T_PEPS_Tensor,
        new_tensor: Tensor,
        *,
        reinitialize_env_as_identities: bool = True,
        new_D: Optional[Tuple[int, int, int, int]] = None,
    ) -> T_PEPS_Tensor:
        """
        Replace the PEPS tensor and returns new object of the class.

        Args:
          new_tensor (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New PEPS tensor.
        Keyword args:
          reinitialize_env_as_identities (:obj:`bool`):
            Reinitialize the CTM tensors as identities.
          new_D (:obj:`tuple` of four :obj:`int`, optional):
            Tuple of new iPEPS bond dimensions if tensor has changed dimensions
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensor replaced.
        """
        if new_D is None:
            new_D = self.D
        elif not isinstance(new_D, tuple) and len(new_D) != 4:
            raise ValueError("Invalid argument for parameter new_D")

        if reinitialize_env_as_identities:
            return type(self)(
                tensor=new_tensor,
                C1=jnp.ones((1, 1), dtype=self.C1.dtype),
                C2=jnp.ones((1, 1), dtype=self.C2.dtype),
                C3=jnp.ones((1, 1), dtype=self.C3.dtype),
                C4=jnp.ones((1, 1), dtype=self.C4.dtype),
                T1=jnp.eye(new_D[3], dtype=self.T1.dtype).reshape(
                    1, new_D[3], new_D[3], 1
                ),
                T2=jnp.eye(new_D[2], dtype=self.T2.dtype).reshape(
                    new_D[2], new_D[2], 1, 1
                ),
                T3=jnp.eye(new_D[1], dtype=self.T3.dtype).reshape(
                    1, 1, new_D[1], new_D[1]
                ),
                T4=jnp.eye(new_D[0], dtype=self.T4.dtype).reshape(
                    1, new_D[0], new_D[0], 1
                ),
                d=self.d,
                D=new_D,
                chi=self.chi,
                max_chi=self.max_chi,
            )
        else:
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
                D=new_D,
                chi=self.chi,
                max_chi=self.max_chi,
            )

    def change_chi(
        self: T_PEPS_Tensor,
        new_chi: int,
        *,
        reinitialize_env_as_identities: bool = True,
        reset_max_chi: bool = False,
    ) -> T_PEPS_Tensor:
        """
        Change the environment bond dimension and returns new object of the class.

        Args:
          new_chi (:obj:`int`):
            New value for environment bond dimension.
        Keyword args:
          reinitialize_env_as_identities (:obj:`bool`):
            Reinitialize the CTM tensors as identities if decreasing the dimension.
          reset_max_chi (:obj:`bool`):
            Set maximal bond dimension to the same new value.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the increased value.
        """
        new_max_chi = new_chi if reset_max_chi else self.max_chi

        if new_chi > new_max_chi:
            raise ValueError(
                "Increase above the max value for environment bond dimension."
            )

        if new_chi < self.chi and reinitialize_env_as_identities:
            return type(self)(
                tensor=self.tensor,
                C1=jnp.ones((1, 1), dtype=self.C1.dtype),
                C2=jnp.ones((1, 1), dtype=self.C2.dtype),
                C3=jnp.ones((1, 1), dtype=self.C3.dtype),
                C4=jnp.ones((1, 1), dtype=self.C4.dtype),
                T1=jnp.eye(self.D[3], dtype=self.T1.dtype).reshape(
                    1, self.D[3], self.D[3], 1
                ),
                T2=jnp.eye(self.D[2], dtype=self.T2.dtype).reshape(
                    self.D[2], self.D[2], 1, 1
                ),
                T3=jnp.eye(self.D[1], dtype=self.T3.dtype).reshape(
                    1, 1, self.D[1], self.D[1]
                ),
                T4=jnp.eye(self.D[0], dtype=self.T4.dtype).reshape(
                    1, self.D[0], self.D[0], 1
                ),
                d=self.d,
                D=self.D,
                chi=new_chi,
                max_chi=new_max_chi,
            )
        else:
            return type(self)(
                tensor=self.tensor,
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
                chi=new_chi,
                max_chi=new_max_chi,
            )

    def increase_max_chi(
        self: T_PEPS_Tensor,
        new_max_chi: int,
    ) -> T_PEPS_Tensor:
        """
        Change the maximal environment bond dimension and returns new object of the class.

        Args:
          new_max_chi (:obj:`int`):
            New value for maximal environment bond dimension.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the increased value.
        """
        if new_max_chi < self.max_chi:
            raise ValueError(
                "Decrease below the old max value for environment bond dimension."
            )

        return type(self)(
            tensor=self.tensor,
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
            max_chi=new_max_chi,
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
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            max_chi=self.max_chi,
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
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            max_chi=self.max_chi,
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
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            max_chi=self.max_chi,
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
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            max_chi=self.max_chi,
        )

    def replace_C1(self: T_PEPS_Tensor, new_C1: Tensor) -> T_PEPS_Tensor:
        """
        Replace the C1 and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
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
            max_chi=self.max_chi,
        )

    def replace_C2(self: T_PEPS_Tensor, new_C2: Tensor) -> T_PEPS_Tensor:
        """
        Replace the C2 and returns new object of the class.

        Args:
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=new_C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_C3(self: T_PEPS_Tensor, new_C3: Tensor) -> T_PEPS_Tensor:
        """
        Replace the C3 and returns new object of the class.

        Args:
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=new_C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_C4(self: T_PEPS_Tensor, new_C4: Tensor) -> T_PEPS_Tensor:
        """
        Replace the C4 and returns new object of the class.

        Args:
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=new_C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_T1(self: T_PEPS_Tensor, new_T1: Tensor) -> T_PEPS_Tensor:
        """
        Replace the T1 and returns new object of the class.

        Args:
          new_T1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T1 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=new_T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_T2(self: T_PEPS_Tensor, new_T2: Tensor) -> T_PEPS_Tensor:
        """
        Replace the T2 and returns new object of the class.

        Args:
          new_T2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T2 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=new_T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_T3(self: T_PEPS_Tensor, new_T3: Tensor) -> T_PEPS_Tensor:
        """
        Replace the T3 and returns new object of the class.

        Args:
          new_T3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T3 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=new_T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_T4(self: T_PEPS_Tensor, new_T4: Tensor) -> T_PEPS_Tensor:
        """
        Replace the T4 and returns new object of the class.

        Args:
          new_T4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T4 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=new_T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_C1_C3(
        self: T_PEPS_Tensor, new_C1: Tensor, new_C3: Tensor
    ) -> T_PEPS_Tensor:
        """
        Replace C1 and C3 tensors and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
            C2=self.C2,
            C3=new_C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_T1_C2_T2_T3_C4_T4(
        self: T_PEPS_Tensor,
        new_T1: Tensor,
        new_C2: Tensor,
        new_T2: Tensor,
        new_T3: Tensor,
        new_C4: Tensor,
        new_T4: Tensor,
    ) -> T_PEPS_Tensor:
        """
        Replace all but C1 and C3 tensors and returns new object of the class.

        Args:
          new_T1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T1 tensor.
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
          new_T2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T2 tensor.
          new_T3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T3 tensor.
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
          new_T4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T4 tensor.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=new_C2,
            C3=self.C3,
            C4=new_C4,
            T1=new_T1,
            T2=new_T2,
            T3=new_T3,
            T4=new_T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def __add__(
        self: T_PEPS_Tensor, other: T_PEPS_Tensor, *, checks: bool = True
    ) -> T_PEPS_Tensor:
        """
        Add the environment tensors of two PEPS tensors.

        Args:
          other (:obj:`~varipeps.peps.PEPS_Tensor`):
            Other PEPS tensor object which should be added to this one.
        Keyword args:
          checks (:obj:`bool`):
            Enable checks that the addition of the two tensor objects makes
            sense. Maybe disabled for jax transformations.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance with the added env tensors.
        """
        if checks and (
            self.tensor is not other.tensor
            or self.d != other.d
            or self.D != other.D
            or self.chi != other.chi
        ):
            raise ValueError(
                "Both PEPS tensors must have the same tensor, d, D and chi values."
            )

        return PEPS_Tensor(
            tensor=self.tensor,
            C1=self.C1 + other.C1,
            C2=self.C2 + other.C2,
            C3=self.C3 + other.C3,
            C4=self.C4 + other.C4,
            T1=self.T1 + other.T1,
            T2=self.T2 + other.T2,
            T3=self.T3 + other.T3,
            T4=self.T4 + other.T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    @classmethod
    def zeros_like(cls: Type[T_PEPS_Tensor], t: T_PEPS_Tensor) -> T_PEPS_Tensor:
        """
        Create a PEPS tensor with same shape as another one but with zeros
        everywhere.

        Args:
          t (:obj:`~varipeps.peps.PEPS_Tensor`):
            Other PEPS tensor object whose shape should be copied.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance with the zero initialized tensors.
        """
        return cls(
            tensor=jnp.zeros_like(t.tensor),
            C1=jnp.zeros_like(t.C1),
            C2=jnp.zeros_like(t.C2),
            C3=jnp.zeros_like(t.C3),
            C4=jnp.zeros_like(t.C4),
            T1=jnp.zeros_like(t.T1),
            T2=jnp.zeros_like(t.T2),
            T3=jnp.zeros_like(t.T3),
            T4=jnp.zeros_like(t.T4),
            d=t.d,
            D=t.D,
            chi=t.chi,
            max_chi=t.max_chi,
        )

    def zeros_like_self(self: T_PEPS_Tensor) -> T_PEPS_Tensor:
        """
        Wrapper around :obj:`~varipeps.peps.PEPS_Tensor.zeros_like` with the
        self object as argument. For details see there.
        """
        return type(self).zeros_like(self)

    def save_to_group(self, grp: h5py.Group) -> None:
        """
        Store the PEPS tensor into a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to save the data into.
        """
        grp.attrs["d"] = self.d
        grp.attrs["D"] = self.D
        grp.attrs["chi"] = self.chi
        grp.attrs["max_chi"] = self.max_chi
        grp.create_dataset(
            "tensor", data=self.tensor, compression="gzip", compression_opts=6
        )
        grp.create_dataset("C1", data=self.C1, compression="gzip", compression_opts=6)
        grp.create_dataset("C2", data=self.C2, compression="gzip", compression_opts=6)
        grp.create_dataset("C3", data=self.C3, compression="gzip", compression_opts=6)
        grp.create_dataset("C4", data=self.C4, compression="gzip", compression_opts=6)
        grp.create_dataset("T1", data=self.T1, compression="gzip", compression_opts=6)
        grp.create_dataset("T2", data=self.T2, compression="gzip", compression_opts=6)
        grp.create_dataset("T3", data=self.T3, compression="gzip", compression_opts=6)
        grp.create_dataset("T4", data=self.T4, compression="gzip", compression_opts=6)

    @classmethod
    def load_from_group(cls: Type[T_PEPS_Tensor], grp: h5py.Group) -> T_PEPS_Tensor:
        """
        Load the PEPS tensor from a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
        """
        d = int(grp.attrs["d"])
        D = tuple(int(i) for i in grp.attrs["D"])
        chi = int(grp.attrs["chi"])
        try:
            max_chi = int(grp.attrs["max_chi"])
        except KeyError:
            max_chi = chi

        tensor = jnp.asarray(grp["tensor"])
        C1 = jnp.asarray(grp["C1"])
        C2 = jnp.asarray(grp["C2"])
        C3 = jnp.asarray(grp["C3"])
        C4 = jnp.asarray(grp["C4"])
        T1 = jnp.asarray(grp["T1"])
        T2 = jnp.asarray(grp["T2"])
        T3 = jnp.asarray(grp["T3"])
        T4 = jnp.asarray(grp["T4"])

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
            max_chi=max_chi,
        )

    @property
    def is_split_transfer(self: T_PEPS_Tensor) -> bool:
        return False

    def convert_to_split_transfer(
        self: T_PEPS_Tensor, interlayer_chi: Optional[int] = None
    ) -> T_PEPS_Tensor_Split_Transfer:
        if interlayer_chi is None:
            interlayer_chi = self.chi

        return PEPS_Tensor_Split_Transfer(
            tensor=self.tensor,
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
            max_chi=self.max_chi,
            interlayer_chi=interlayer_chi,
        )

    def convert_to_full_transfer(self: T_PEPS_Tensor) -> T_PEPS_Tensor:
        return self

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        data = (
            self.tensor,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.T1,
            self.T2,
            self.T3,
            self.T4,
        )
        aux_data = (self.d, self.D, self.chi, self.max_chi)

        return (data, aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_PEPS_Tensor], aux_data: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> T_PEPS_Tensor:
        tensor, C1, C2, C3, C4, T1, T2, T3, T4 = children
        d, D, chi, max_chi = aux_data

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
            max_chi=max_chi,
            sanity_checks=False,
        )


@dataclass
@register_pytree_node_class
class PEPS_Tensor_Structure_Factor(PEPS_Tensor):
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
      C1_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C1 tensor including structure factor phase
      C2_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C2 tensor including structure factor phase
      C3_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C3 tensor including structure factor phase
      C4_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        C4 tensor including structure factor phase
      T1_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T1 tensor including structure factor phase
      T2_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T2 tensor including structure factor phase
      T3_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T3 tensor including structure factor phase
      T4_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
        T4 tensor including structure factor phase
      d (:obj:`int`):
        Physical dimension of the PEPS tensor
      D (:term:`sequence` of :obj:`int`):
        Sequence of the bond dimensions of the PEPS tensor
      chi (:obj:`int`):
        Bond dimension for the CTM tensors
      max_chi (:obj:`int`):
        Maximal allowed bond dimension of environment tensors.
    """

    C1_phase: Tensor = None
    C2_phase: Tensor = None
    C3_phase: Tensor = None
    C4_phase: Tensor = None

    T1_phase: Tensor = None
    T2_phase: Tensor = None
    T3_phase: Tensor = None
    T4_phase: Tensor = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not (
            self.T1_phase.shape[1] == self.T1_phase.shape[2] == self.D[3]
            and self.T2_phase.shape[0] == self.T2_phase.shape[1] == self.D[2]
            and self.T3_phase.shape[2] == self.T3_phase.shape[3] == self.D[1]
            and self.T4_phase.shape[1] == self.T4_phase.shape[2] == self.D[0]
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
        max_chi: Optional[int] = None,
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
          max_chi (:obj:`int`):
            Maximal allowed bond dimension for the environment tensors
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

        if max_chi is None:
            max_chi = chi

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

            C1_phase = jnp.ones((1, 1), dtype=dtype)
            C2_phase = jnp.ones((1, 1), dtype=dtype)
            C3_phase = jnp.ones((1, 1), dtype=dtype)
            C4_phase = jnp.ones((1, 1), dtype=dtype)

            T1_phase = jnp.eye(D[3], dtype=dtype).reshape(1, D[3], D[3], 1)
            T2_phase = jnp.eye(D[2], dtype=dtype).reshape(D[2], D[2], 1, 1)
            T3_phase = jnp.eye(D[1], dtype=dtype).reshape(1, 1, D[1], D[1])
            T4_phase = jnp.eye(D[0], dtype=dtype).reshape(1, D[0], D[0], 1)
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

            C1_phase = rng.block((chi, chi), dtype, normalize=normalize)
            C2_phase = rng.block((chi, chi), dtype, normalize=normalize)
            C3_phase = rng.block((chi, chi), dtype, normalize=normalize)
            C4_phase = rng.block((chi, chi), dtype, normalize=normalize)

            T1_phase = rng.block((chi, D[3], D[3], chi), dtype, normalize=normalize)
            T2_phase = rng.block((D[2], D[2], chi, chi), dtype, normalize=normalize)
            T3_phase = rng.block((chi, chi, D[1], D[1]), dtype, normalize=normalize)
            T4_phase = rng.block((chi, D[0], D[0], chi), dtype, normalize=normalize)

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
            C1_phase=C1_phase,
            C2_phase=C2_phase,
            C3_phase=C3_phase,
            C4_phase=C4_phase,
            T1_phase=T1_phase,
            T2_phase=T2_phase,
            T3_phase=T3_phase,
            T4_phase=T4_phase,
            d=d,
            D=D,  # type: ignore
            chi=chi,
            max_chi=max_chi,
        )

    def change_chi(
        self: T_PEPS_Tensor,
        new_chi: int,
        *,
        reinitialize_env_as_identities: bool = True,
        reset_max_chi: bool = False,
    ) -> T_PEPS_Tensor:
        """
        Change the environment bond dimension and returns new object of the class.

        Args:
          new_chi (:obj:`int`):
            New value for environment bond dimension.
        Keyword args:
          reinitialize_env_as_identities (:obj:`bool`):
            Reinitialize the CTM tensors as identities if decreasing the dimension.
          reset_max_chi (:obj:`bool`):
            Set maximal bond dimension to the same new value.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the increased value.
        """
        new_max_chi = new_chi if reset_max_chi else self.max_chi

        if new_chi > new_max_chi:
            raise ValueError(
                "Increase above the max value for environment bond dimension."
            )

        if new_chi < self.chi and reinitialize_env_as_identities:
            return type(self)(
                tensor=self.tensor,
                C1=jnp.ones((1, 1), dtype=self.C1.dtype),
                C2=jnp.ones((1, 1), dtype=self.C2.dtype),
                C3=jnp.ones((1, 1), dtype=self.C3.dtype),
                C4=jnp.ones((1, 1), dtype=self.C4.dtype),
                T1=jnp.eye(self.D[3], dtype=self.T1.dtype).reshape(
                    1, self.D[3], self.D[3], 1
                ),
                T2=jnp.eye(self.D[2], dtype=self.T2.dtype).reshape(
                    self.D[2], self.D[2], 1, 1
                ),
                T3=jnp.eye(self.D[1], dtype=self.T3.dtype).reshape(
                    1, 1, self.D[1], self.D[1]
                ),
                T4=jnp.eye(self.D[0], dtype=self.T4.dtype).reshape(
                    1, self.D[0], self.D[0], 1
                ),
                C1_phase=jnp.ones((1, 1), dtype=self.C1_phase.dtype),
                C2_phase=jnp.ones((1, 1), dtype=self.C2_phase.dtype),
                C3_phase=jnp.ones((1, 1), dtype=self.C3_phase.dtype),
                C4_phase=jnp.ones((1, 1), dtype=self.C4_phase.dtype),
                T1_phase=jnp.eye(self.D[3], dtype=self.T1_phase.dtype).reshape(
                    1, self.D[3], self.D[3], 1
                ),
                T2_phase=jnp.eye(self.D[2], dtype=self.T2_phase.dtype).reshape(
                    self.D[2], self.D[2], 1, 1
                ),
                T3_phase=jnp.eye(self.D[1], dtype=self.T3_phase.dtype).reshape(
                    1, 1, self.D[1], self.D[1]
                ),
                T4_phase=jnp.eye(self.D[0], dtype=self.T4_phase.dtype).reshape(
                    1, self.D[0], self.D[0], 1
                ),
                d=self.d,
                D=self.D,
                chi=new_chi,
                max_chi=new_max_chi,
            )
        else:
            return type(self)(
                tensor=self.tensor,
                C1=self.C1,
                C2=self.C2,
                C3=self.C3,
                C4=self.C4,
                T1=self.T1,
                T2=self.T2,
                T3=self.T3,
                T4=self.T4,
                C1_phase=self.C1_phase,
                C2_phase=self.C2_phase,
                C3_phase=self.C3_phase,
                C4_phase=self.C4_phase,
                T1_phase=self.T1_phase,
                T2_phase=self.T2_phase,
                T3_phase=self.T3_phase,
                T4_phase=self.T4_phase,
                d=self.d,
                D=self.D,
                chi=new_chi,
                max_chi=new_max_chi,
            )

    def increase_max_chi(
        self: T_PEPS_Tensor,
        new_max_chi: int,
    ) -> T_PEPS_Tensor:
        """
        Change the maximal environment bond dimension and returns new object of the class.

        Args:
          new_max_chi (:obj:`int`):
            New value for maximal environment bond dimension.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
            New instance of the class with the increased value.
        """
        if new_max_chi < self.max_chi:
            raise ValueError(
                "Decrease below the old max value for environment bond dimension."
            )

        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=self.T1,
            T2=self.T2,
            T3=self.T3,
            T4=self.T4,
            C1_phase=self.C1_phase,
            C2_phase=self.C2_phase,
            C3_phase=self.C3_phase,
            C4_phase=self.C4_phase,
            T1_phase=self.T1_phase,
            T2_phase=self.T2_phase,
            T3_phase=self.T3_phase,
            T4_phase=self.T4_phase,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=new_max_chi,
        )

    def replace_left_env_tensors(
        self: T_PEPS_Tensor,
        new_C1: Tensor,
        new_T4: Tensor,
        new_C4: Tensor,
        new_C1_phase: Tensor,
        new_T4_phase: Tensor,
        new_C4_phase: Tensor,
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
          new_C1_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor including structure factor phase.
          new_T4_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T4 tensor including structure factor phase.
          new_C4_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor including structure factor phase.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            C1_phase=new_C1_phase,
            C2_phase=self.C2_phase,
            C3_phase=self.C3_phase,
            C4_phase=new_C4_phase,
            T1_phase=self.T1_phase,
            T2_phase=self.T2_phase,
            T3_phase=self.T3_phase,
            T4_phase=new_T4_phase,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_right_env_tensors(
        self: T_PEPS_Tensor,
        new_C2: Tensor,
        new_T2: Tensor,
        new_C3: Tensor,
        new_C2_phase: Tensor,
        new_T2_phase: Tensor,
        new_C3_phase: Tensor,
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
          new_C2_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor including structure factor phase.
          new_T2_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T2 tensor including structure factor phase.
          new_C3_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor including structure factor phase.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            C1_phase=self.C1_phase,
            C2_phase=new_C2_phase,
            C3_phase=new_C3_phase,
            C4_phase=self.C4_phase,
            T1_phase=self.T1_phase,
            T2_phase=new_T2_phase,
            T3_phase=self.T3_phase,
            T4_phase=self.T4_phase,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_top_env_tensors(
        self: T_PEPS_Tensor,
        new_C1: Tensor,
        new_T1: Tensor,
        new_C2: Tensor,
        new_C1_phase: Tensor,
        new_T1_phase: Tensor,
        new_C2_phase: Tensor,
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
          new_C1_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor including structure factor phase.
          new_T1_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T1 tensor including structure factor phase.
          new_C2_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor including structure factor phase.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            C1_phase=new_C1_phase,
            C2_phase=new_C2_phase,
            C3_phase=self.C3_phase,
            C4_phase=self.C4_phase,
            T1_phase=new_T1_phase,
            T2_phase=self.T2_phase,
            T3_phase=self.T3_phase,
            T4_phase=self.T4_phase,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def replace_bottom_env_tensors(
        self: T_PEPS_Tensor,
        new_C4: Tensor,
        new_T3: Tensor,
        new_C3: Tensor,
        new_C4_phase: Tensor,
        new_T3_phase: Tensor,
        new_C3_phase: Tensor,
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
          new_C4_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor including structure factor phase.
          new_T3_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New T3 tensor including structure factor phase.
          new_C3_phase (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor including structure factor phase.
        Returns:
          :obj:`~varipeps.peps.PEPS_Tensor`:
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
            C1_phase=self.C1_phase,
            C2_phase=self.C2_phase,
            C3_phase=new_C3_phase,
            C4_phase=new_C4_phase,
            T1_phase=self.T1_phase,
            T2_phase=self.T2_phase,
            T3_phase=new_T3_phase,
            T4_phase=self.T4_phase,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def save_to_group(self, grp: h5py.Group) -> None:
        """
        Store the PEPS tensor into a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to save the data into.
        """
        grp.attrs["d"] = self.d
        grp.attrs["D"] = self.D
        grp.attrs["chi"] = self.chi
        grp.attrs["max_chi"] = self.max_chi
        grp.create_dataset(
            "tensor", data=self.tensor, compression="gzip", compression_opts=6
        )
        grp.create_dataset("C1", data=self.C1, compression="gzip", compression_opts=6)
        grp.create_dataset("C2", data=self.C2, compression="gzip", compression_opts=6)
        grp.create_dataset("C3", data=self.C3, compression="gzip", compression_opts=6)
        grp.create_dataset("C4", data=self.C4, compression="gzip", compression_opts=6)
        grp.create_dataset("T1", data=self.T1, compression="gzip", compression_opts=6)
        grp.create_dataset("T2", data=self.T2, compression="gzip", compression_opts=6)
        grp.create_dataset("T3", data=self.T3, compression="gzip", compression_opts=6)
        grp.create_dataset("T4", data=self.T4, compression="gzip", compression_opts=6)
        grp.create_dataset(
            "C1_phase", data=self.C1_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "C2_phase", data=self.C2_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "C3_phase", data=self.C3_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "C4_phase", data=self.C4_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T1_phase", data=self.T1_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T2_phase", data=self.T2_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T3_phase", data=self.T3_phase, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T4_phase", data=self.T4_phase, compression="gzip", compression_opts=6
        )

    @classmethod
    def load_from_group(cls: Type[T_PEPS_Tensor], grp: h5py.Group) -> T_PEPS_Tensor:
        """
        Load the PEPS tensor from a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
        """
        d = int(grp.attrs["d"])
        D = tuple(int(i) for i in grp.attrs["D"])
        chi = int(grp.attrs["chi"])
        try:
            max_chi = int(grp.attrs["max_chi"])
        except KeyError:
            max_chi = chi

        tensor = jnp.asarray(grp["tensor"])
        C1 = jnp.asarray(grp["C1"])
        C2 = jnp.asarray(grp["C2"])
        C3 = jnp.asarray(grp["C3"])
        C4 = jnp.asarray(grp["C4"])
        T1 = jnp.asarray(grp["T1"])
        T2 = jnp.asarray(grp["T2"])
        T3 = jnp.asarray(grp["T3"])
        T4 = jnp.asarray(grp["T4"])
        C1_phase = jnp.asarray(grp["C1_phase"])
        C2_phase = jnp.asarray(grp["C2_phase"])
        C3_phase = jnp.asarray(grp["C3_phase"])
        C4_phase = jnp.asarray(grp["C4_phase"])
        T1_phase = jnp.asarray(grp["T1_phase"])
        T2_phase = jnp.asarray(grp["T2_phase"])
        T3_phase = jnp.asarray(grp["T3_phase"])
        T4_phase = jnp.asarray(grp["T4_phase"])

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
            C1_phase=C1_phase,
            C2_phase=C2_phase,
            C3_phase=C3_phase,
            C4_phase=C4_phase,
            T1_phase=T1_phase,
            T2_phase=T2_phase,
            T3_phase=T3_phase,
            T4_phase=T4_phase,
            d=d,
            D=D,
            chi=chi,
            max_chi=max_chi,
        )

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        data = (
            self.tensor,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.T1,
            self.T2,
            self.T3,
            self.T4,
            self.C1_phase,
            self.C2_phase,
            self.C3_phase,
            self.C4_phase,
            self.T1_phase,
            self.T2_phase,
            self.T3_phase,
            self.T4_phase,
        )
        aux_data = (self.d, self.D, self.chi, self.max_chi)

        return (data, aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_PEPS_Tensor], aux_data: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> T_PEPS_Tensor:
        (
            tensor,
            C1,
            C2,
            C3,
            C4,
            T1,
            T2,
            T3,
            T4,
            C1_phase,
            C2_phase,
            C3_phase,
            C4_phase,
            T1_phase,
            T2_phase,
            T3_phase,
            T4_phase,
        ) = children
        d, D, chi, max_chi = aux_data

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
            C1_phase=C1_phase,
            C2_phase=C2_phase,
            C3_phase=C3_phase,
            C4_phase=C4_phase,
            T1_phase=T1_phase,
            T2_phase=T2_phase,
            T3_phase=T3_phase,
            T4_phase=T4_phase,
            d=d,
            D=D,
            chi=chi,
            max_chi=max_chi,
            sanity_checks=False,
        )


@dataclass
@register_pytree_node_class
class PEPS_Tensor_Split_Transfer(PEPS_Tensor):
    T1: Optional[Tensor] = None
    T2: Optional[Tensor] = None
    T3: Optional[Tensor] = None
    T4: Optional[Tensor] = None

    T1_ket: Optional[Tensor] = None
    T1_bra: Optional[Tensor] = None
    T2_ket: Optional[Tensor] = None
    T2_bra: Optional[Tensor] = None
    T3_ket: Optional[Tensor] = None
    T3_bra: Optional[Tensor] = None
    T4_ket: Optional[Tensor] = None
    T4_bra: Optional[Tensor] = None

    interlayer_chi: Optional[int] = None

    def __post_init__(self) -> None:
        if any(
            e is None
            for e in (
                self.T1_ket,
                self.T1_bra,
                self.T2_ket,
                self.T2_bra,
                self.T3_ket,
                self.T3_bra,
                self.T4_ket,
                self.T4_bra,
            )
        ):
            self._split_up_T1(self.T1)
            self._split_up_T2(self.T2)
            self._split_up_T3(self.T3)
            self._split_up_T4(self.T4)
            self.T1 = None
            self.T2 = None
            self.T3 = None
            self.T4 = None

        if self.interlayer_chi is None:
            self.interlayer_chi = self.chi

        if not self.sanity_checks:
            return

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

        # if not (
        #     self.T1.shape[1] == self.T1.shape[2] == self.D[3]
        #     and self.T2.shape[0] == self.T2.shape[1] == self.D[2]
        #     and self.T3.shape[2] == self.T3.shape[3] == self.D[1]
        #     and self.T4.shape[1] == self.T4.shape[2] == self.D[0]
        # ):
        #     raise ValueError(
        #         "At least one transfer tensors mismatch bond dimensions of PEPS tensor."
        #     )

    @classmethod
    def from_tensor(
        cls: Type[T_PEPS_Tensor],
        tensor: Tensor,
        d: int,
        D: Union[int, Sequence[int]],
        chi: int,
        interlayer_chi: Optional[int] = None,
        max_chi: Optional[int] = None,
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
          interlayer_chi (:obj:`int`):
            Bond dimension for the interlayer bonds of the environment tensors
          max_chi (:obj:`int`):
            Maximal allowed bond dimension for the environment tensors
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

        if interlayer_chi is None:
            interlayer_chi = chi
        if max_chi is None:
            max_chi = chi

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
                interlayer_chi=interlayer_chi,
                max_chi=max_chi,
            )
        else:
            rng = PEPS_Random_Number_Generator.get_generator(seed, backend=backend)

            C1 = rng.block((chi, chi), dtype, normalize=normalize)
            C2 = rng.block((chi, chi), dtype, normalize=normalize)
            C3 = rng.block((chi, chi), dtype, normalize=normalize)
            C4 = rng.block((chi, chi), dtype, normalize=normalize)

            T1_ket = rng.block((chi, D[3], interlayer_chi), dtype, normalize=normalize)
            T1_bra = rng.block((interlayer_chi, D[3], chi), dtype, normalize=normalize)
            T2_ket = rng.block((interlayer_chi, D[2], chi), dtype, normalize=normalize)
            T2_bra = rng.block((chi, D[2], interlayer_chi), dtype, normalize=normalize)
            T3_ket = rng.block((chi, D[1], interlayer_chi), dtype, normalize=normalize)
            T3_bra = rng.block((interlayer_chi, D[1], chi), dtype, normalize=normalize)
            T4_ket = rng.block((interlayer_chi, D[0], chi), dtype, normalize=normalize)
            T4_bra = rng.block((chi, D[0], interlayer_chi), dtype, normalize=normalize)

            return cls(
                tensor=tensor,
                C1=C1,
                C2=C2,
                C3=C3,
                C4=C4,
                T1_ket=T1_ket,
                T1_bra=T1_bra,
                T2_ket=T2_ket,
                T2_bra=T2_bra,
                T3_ket=T3_ket,
                T3_bra=T3_bra,
                T4_ket=T4_ket,
                T4_bra=T4_bra,
                d=d,
                D=D,  # type: ignore
                chi=chi,
                interlayer_chi=interlayer_chi,
                max_chi=max_chi,
            )

    @property
    def left_upper_transfer_shape(self) -> Tensor:
        return self.T4_ket.shape[2]

    @property
    def left_lower_transfer_shape(self) -> Tensor:
        return self.T4_bra.shape[0]

    @property
    def right_upper_transfer_shape(self) -> Tensor:
        return self.T2_ket.shape[2]

    @property
    def right_lower_transfer_shape(self) -> Tensor:
        return self.T2_bra.shape[0]

    @property
    def top_left_transfer_shape(self) -> Tensor:
        return self.T1_ket.shape[0]

    @property
    def top_right_transfer_shape(self) -> Tensor:
        return self.T1_bra.shape[2]

    @property
    def bottom_left_transfer_shape(self) -> Tensor:
        return self.T3_ket.shape[0]

    @property
    def bottom_right_transfer_shape(self) -> Tensor:
        return self.T3_bra.shape[2]

    def _split_up_T1(self, T1):
        tmp_T1 = T1.reshape(T1.shape[0] * T1.shape[1], T1.shape[2] * T1.shape[3])

        self.T1_ket, T1_S, self.T1_bra = gauge_fixed_svd(tmp_T1)

        self.T1_ket = self.T1_ket[:, : self.interlayer_chi]
        T1_S = jnp.sqrt(T1_S[: self.interlayer_chi])
        self.T1_bra = self.T1_bra[: self.interlayer_chi, :]

        self.T1_ket = self.T1_ket * T1_S[jnp.newaxis, :]
        self.T1_bra = T1_S[:, jnp.newaxis] * self.T1_bra

        self.T1_ket = self.T1_ket.reshape(
            T1.shape[0], T1.shape[1], self.T1_ket.shape[1]
        )
        self.T1_bra = self.T1_bra.reshape(
            self.T1_bra.shape[0], T1.shape[2], T1.shape[3]
        )

    def _split_up_T2(self, T2):
        tmp_T2 = T2.transpose(2, 1, 0, 3)
        tmp_T2 = tmp_T2.reshape(
            tmp_T2.shape[0] * tmp_T2.shape[1], tmp_T2.shape[2] * tmp_T2.shape[3]
        )

        self.T2_bra, T2_S, self.T2_ket = gauge_fixed_svd(tmp_T2)

        self.T2_bra = self.T2_bra[:, : self.interlayer_chi]
        T2_S = jnp.sqrt(T2_S[: self.interlayer_chi])
        self.T2_ket = self.T2_ket[: self.interlayer_chi, :]

        self.T2_bra = self.T2_bra * T2_S[jnp.newaxis, :]
        self.T2_ket = T2_S[:, jnp.newaxis] * self.T2_ket

        self.T2_bra = self.T2_bra.reshape(
            T2.shape[2], T2.shape[1], self.T2_bra.shape[1]
        )
        self.T2_ket = self.T2_ket.reshape(
            self.T2_ket.shape[0], T2.shape[0], T2.shape[3]
        )

    def _split_up_T3(self, T3):
        tmp_T3 = T3.transpose(0, 3, 2, 1)
        tmp_T3 = tmp_T3.reshape(
            tmp_T3.shape[0] * tmp_T3.shape[1], tmp_T3.shape[2] * tmp_T3.shape[3]
        )

        self.T3_ket, T3_S, self.T3_bra = gauge_fixed_svd(tmp_T3)

        self.T3_ket = self.T3_ket[:, : self.interlayer_chi]
        T3_S = jnp.sqrt(T3_S[: self.interlayer_chi])
        self.T3_bra = self.T3_bra[: self.interlayer_chi, :]

        self.T3_ket = self.T3_ket * T3_S[jnp.newaxis, :]
        self.T3_bra = T3_S[:, jnp.newaxis] * self.T3_bra

        self.T3_ket = self.T3_ket.reshape(
            T3.shape[0], T3.shape[3], self.T3_ket.shape[1]
        )
        self.T3_bra = self.T3_bra.reshape(
            self.T3_bra.shape[0], T3.shape[2], T3.shape[1]
        )

    def _split_up_T4(self, T4):
        tmp_T4 = T4.reshape(T4.shape[0] * T4.shape[1], T4.shape[2] * T4.shape[3])

        self.T4_bra, T4_S, self.T4_ket = gauge_fixed_svd(tmp_T4)

        self.T4_bra = self.T4_bra[:, : self.interlayer_chi]
        T4_S = jnp.sqrt(T4_S[: self.interlayer_chi])
        self.T4_ket = self.T4_ket[: self.interlayer_chi, :]

        self.T4_bra = self.T4_bra * T4_S[jnp.newaxis, :]
        self.T4_ket = T4_S[:, jnp.newaxis] * self.T4_ket

        self.T4_bra = self.T4_bra.reshape(
            T4.shape[0], T4.shape[1], self.T4_bra.shape[1]
        )
        self.T4_ket = self.T4_ket.reshape(
            self.T4_ket.shape[0], T4.shape[2], T4.shape[3]
        )

    def replace_tensor(
        self: T_PEPS_Tensor_Split_Transfer,
        new_tensor: Tensor,
        *,
        reinitialize_env_as_identities: bool = True,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Replace the PEPS tensor and returns new object of the class.

        Args:
          new_tensor (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New PEPS tensor.
        Keyword args:
          reinitialize_env_as_identities (:obj:`bool`):
            Reinitialize the CTM tensors as identities.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the tensor replaced.
        """
        if reinitialize_env_as_identities:
            return type(self)(
                tensor=new_tensor,
                C1=jnp.ones((1, 1), dtype=self.C1.dtype),
                C2=jnp.ones((1, 1), dtype=self.C2.dtype),
                C3=jnp.ones((1, 1), dtype=self.C3.dtype),
                C4=jnp.ones((1, 1), dtype=self.C4.dtype),
                T1=jnp.eye(self.D[3], dtype=self.T1_ket.dtype).reshape(
                    1, self.D[3], self.D[3], 1
                ),
                T2=jnp.eye(self.D[2], dtype=self.T2_ket.dtype).reshape(
                    self.D[2], self.D[2], 1, 1
                ),
                T3=jnp.eye(self.D[1], dtype=self.T3_ket.dtype).reshape(
                    1, 1, self.D[1], self.D[1]
                ),
                T4=jnp.eye(self.D[0], dtype=self.T4_ket.dtype).reshape(
                    1, self.D[0], self.D[0], 1
                ),
                d=self.d,
                D=self.D,
                chi=self.chi,
                max_chi=self.max_chi,
                interlayer_chi=self.interlayer_chi,
            )
        else:
            return type(self)(
                tensor=new_tensor,
                C1=self.C1,
                C2=self.C2,
                C3=self.C3,
                C4=self.C4,
                T1_ket=self.T1_ket,
                T1_bra=self.T1_bra,
                T2_ket=self.T2_ket,
                T2_bra=self.T2_bra,
                T3_ket=self.T3_ket,
                T3_bra=self.T3_bra,
                T4_ket=self.T4_ket,
                T4_bra=self.T4_bra,
                d=self.d,
                D=self.D,
                chi=self.chi,
                max_chi=self.max_chi,
                interlayer_chi=self.interlayer_chi,
            )

    def change_chi(
        self: T_PEPS_Tensor_Split_Transfer,
        new_chi: int,
        *,
        reinitialize_env_as_identities: bool = True,
        reset_max_chi: bool = False,
        reset_interlayer_chi: bool = True,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Change the environment bond dimension and returns new object of the class.

        Args:
          new_chi (:obj:`int`):
            New value for environment bond dimension.
        Keyword args:
          reinitialize_env_as_identities (:obj:`bool`):
            Reinitialize the CTM tensors as identities if decreasing the dimension.
          reset_max_chi (:obj:`bool`):
            Set maximal bond dimension to the same new value.
          reset_interlayer_chi (:obj:`bool`):
            Set interlayer bond dimension to the same new value.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the increased value.
        """
        new_max_chi = new_chi if reset_max_chi else self.max_chi

        if new_chi > new_max_chi:
            raise ValueError(
                "Increase above the max value for environment bond dimension."
            )

        new_interlayer_chi = new_chi if reset_interlayer_chi else self.interlayer_chi

        if new_chi < self.chi and reinitialize_env_as_identities:
            return type(self)(
                tensor=self.tensor,
                C1=jnp.ones((1, 1), dtype=self.C1.dtype),
                C2=jnp.ones((1, 1), dtype=self.C2.dtype),
                C3=jnp.ones((1, 1), dtype=self.C3.dtype),
                C4=jnp.ones((1, 1), dtype=self.C4.dtype),
                T1=jnp.eye(self.D[3], dtype=self.T1_ket.dtype).reshape(
                    1, self.D[3], self.D[3], 1
                ),
                T2=jnp.eye(self.D[2], dtype=self.T2_ket.dtype).reshape(
                    self.D[2], self.D[2], 1, 1
                ),
                T3=jnp.eye(self.D[1], dtype=self.T3_ket.dtype).reshape(
                    1, 1, self.D[1], self.D[1]
                ),
                T4=jnp.eye(self.D[0], dtype=self.T4_ket.dtype).reshape(
                    1, self.D[0], self.D[0], 1
                ),
                d=self.d,
                D=self.D,
                chi=new_chi,
                max_chi=new_max_chi,
                interlayer_chi=new_interlayer_chi,
            )
        else:
            return type(self)(
                tensor=self.tensor,
                C1=self.C1,
                C2=self.C2,
                C3=self.C3,
                C4=self.C4,
                T1_ket=self.T1_ket,
                T1_bra=self.T1_bra,
                T2_ket=self.T2_ket,
                T2_bra=self.T2_bra,
                T3_ket=self.T3_ket,
                T3_bra=self.T3_bra,
                T4_ket=self.T4_ket,
                T4_bra=self.T4_bra,
                d=self.d,
                D=self.D,
                chi=new_chi,
                max_chi=new_max_chi,
                interlayer_chi=new_interlayer_chi,
            )

    def increase_max_chi(
        self: T_PEPS_Tensor_Split_Transfer,
        new_max_chi: int,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Change the maximal environment bond dimension and returns new object of the class.

        Args:
          new_max_chi (:obj:`int`):
            New value for maximal environment bond dimension.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the increased value.
        """
        if new_max_chi < self.max_chi:
            raise ValueError(
                "Decrease below the old max value for environment bond dimension."
            )

        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1_ket=self.T1_ket,
            T1_bra=self.T1_bra,
            T2_ket=self.T2_ket,
            T2_bra=self.T2_bra,
            T3_ket=self.T3_ket,
            T3_bra=self.T3_bra,
            T4_ket=self.T4_ket,
            T4_bra=self.T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=new_max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    def replace_left_env_tensors(
        self: T_PEPS_Tensor_Split_Transfer,
        new_C1: Tensor,
        new_T4_ket: Tensor,
        new_T4_bra: Tensor,
        new_C4: Tensor,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Replace the left CTMRG tensors and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
          new_T4_ket (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New ket T4 tensor.
          new_T4_bra (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New bra T4 tensor.
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
            C2=self.C2,
            C3=self.C3,
            C4=new_C4,
            T1_ket=self.T1_ket,
            T1_bra=self.T1_bra,
            T2_ket=self.T2_ket,
            T2_bra=self.T2_bra,
            T3_ket=self.T3_ket,
            T3_bra=self.T3_bra,
            T4_ket=new_T4_ket,
            T4_bra=new_T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    def replace_right_env_tensors(
        self: T_PEPS_Tensor_Split_Transfer,
        new_C2: Tensor,
        new_T2_ket: Tensor,
        new_T2_bra: Tensor,
        new_C3: Tensor,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Replace the right CTMRG tensors and returns new object of the class.

        Args:
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
          new_T2_ket (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New ket T2 tensor.
          new_T2_bra (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New bra T2 tensor.
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=new_C2,
            C3=new_C3,
            C4=self.C4,
            T1_ket=self.T1_ket,
            T1_bra=self.T1_bra,
            T2_ket=new_T2_ket,
            T2_bra=new_T2_bra,
            T3_ket=self.T3_ket,
            T3_bra=self.T3_bra,
            T4_ket=self.T4_ket,
            T4_bra=self.T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    def replace_top_env_tensors(
        self: T_PEPS_Tensor_Split_Transfer,
        new_C1: Tensor,
        new_T1_ket: Tensor,
        new_T1_bra: Tensor,
        new_C2: Tensor,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Replace the top CTMRG tensors and returns new object of the class.

        Args:
          new_C1 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C1 tensor.
          new_T1_ket (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New ket T1 tensor.
          new_T1_bra (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New bra T1 tensor.
          new_C2 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C2 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=new_C1,
            C2=new_C2,
            C3=self.C3,
            C4=self.C4,
            T1_ket=new_T1_ket,
            T1_bra=new_T1_bra,
            T2_ket=self.T2_ket,
            T2_bra=self.T2_bra,
            T3_ket=self.T3_ket,
            T3_bra=self.T3_bra,
            T4_ket=self.T4_ket,
            T4_bra=self.T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    def replace_bottom_env_tensors(
        self: T_PEPS_Tensor_Split_Transfer,
        new_C4: Tensor,
        new_T3_ket: Tensor,
        new_T3_bra: Tensor,
        new_C3: Tensor,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Replace the bottom CTMRG tensors and returns new object of the class.

        Args:
          new_C4 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C4 tensor.
          new_T3_ket (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New ket T3 tensor.
          new_T3_bra (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New bra T3 tensor.
          new_C3 (:obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`):
            New C3 tensor.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance of the class with the tensors replaced.
        """
        return type(self)(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=new_C3,
            C4=new_C4,
            T1_ket=self.T1_ket,
            T1_bra=self.T1_bra,
            T2_ket=self.T2_ket,
            T2_bra=self.T2_bra,
            T3_ket=new_T3_ket,
            T3_bra=new_T3_bra,
            T4_ket=self.T4_ket,
            T4_bra=self.T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    def __add__(
        self: T_PEPS_Tensor_Split_Transfer,
        other: T_PEPS_Tensor_Split_Transfer,
        *,
        checks: bool = True,
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Add the environment tensors of two PEPS tensors.

        Args:
          other (:obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`):
            Other PEPS tensor object which should be added to this one.
        Keyword args:
          checks (:obj:`bool`):
            Enable checks that the addition of the two tensor objects makes
            sense. Maybe disabled for jax transformations.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance with the added env tensors.
        """
        if checks and (
            self.tensor is not other.tensor
            or self.d != other.d
            or self.D != other.D
            or self.chi != other.chi
        ):
            raise ValueError(
                "Both PEPS tensors must have the same tensor, d, D and chi values."
            )

        return type(self)(
            tensor=self.tensor,
            C1=self.C1 + other.C1,
            C2=self.C2 + other.C2,
            C3=self.C3 + other.C3,
            C4=self.C4 + other.C4,
            T1_ket=self.T1_ket + other.T1_ket,
            T1_bra=self.T1_bra + other.T1_bra,
            T2_ket=self.T2_ket + other.T2_ket,
            T2_bra=self.T2_bra + other.T2_bra,
            T3_ket=self.T3_ket + other.T3_ket,
            T3_bra=self.T3_bra + other.T3_bra,
            T4_ket=self.T4_ket + other.T4_ket,
            T4_bra=self.T4_bra + other.T4_bra,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
            interlayer_chi=self.interlayer_chi,
        )

    @classmethod
    def zeros_like(
        cls: Type[T_PEPS_Tensor_Split_Transfer], t: T_PEPS_Tensor_Split_Transfer
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Create a PEPS tensor with same shape as another one but with zeros
        everywhere.

        Args:
          t (:obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`):
            Other PEPS tensor object whose shape should be copied.
        Returns:
          :obj:`~peps_ad.peps.PEPS_Tensor_Split_Transfer`:
            New instance with the zero initialized tensors.
        """
        return cls(
            tensor=jnp.zeros_like(t.tensor),
            C1=jnp.zeros_like(t.C1),
            C2=jnp.zeros_like(t.C2),
            C3=jnp.zeros_like(t.C3),
            C4=jnp.zeros_like(t.C4),
            T1_ket=jnp.zeros_like(t.T1_ket),
            T1_bra=jnp.zeros_like(t.T1_bra),
            T2_ket=jnp.zeros_like(t.T2_ket),
            T2_bra=jnp.zeros_like(t.T2_bra),
            T3_ket=jnp.zeros_like(t.T3_ket),
            T3_bra=jnp.zeros_like(t.T3_bra),
            T4_ket=jnp.zeros_like(t.T4_ket),
            T4_bra=jnp.zeros_like(t.T4_bra),
            d=t.d,
            D=t.D,
            chi=t.chi,
            max_chi=t.max_chi,
            interlayer_chi=t.interlayer_chi,
        )

    def save_to_group(self, grp: h5py.Group) -> None:
        """
        Store the PEPS tensor into a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to save the data into.
        """
        grp.attrs["d"] = self.d
        grp.attrs["D"] = self.D
        grp.attrs["chi"] = self.chi
        grp.attrs["max_chi"] = self.max_chi
        grp.attrs["interlayer_chi"] = self.interlayer_chi
        grp.create_dataset(
            "tensor", data=self.tensor, compression="gzip", compression_opts=6
        )
        grp.create_dataset("C1", data=self.C1, compression="gzip", compression_opts=6)
        grp.create_dataset("C2", data=self.C2, compression="gzip", compression_opts=6)
        grp.create_dataset("C3", data=self.C3, compression="gzip", compression_opts=6)
        grp.create_dataset("C4", data=self.C4, compression="gzip", compression_opts=6)
        grp.create_dataset(
            "T1_ket", data=self.T1_ket, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T1_bra", data=self.T1_bra, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T2_ket", data=self.T2_ket, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T2_bra", data=self.T2_bra, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T3_ket", data=self.T3_ket, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T3_bra", data=self.T3_bra, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T4_ket", data=self.T4_ket, compression="gzip", compression_opts=6
        )
        grp.create_dataset(
            "T4_bra", data=self.T4_bra, compression="gzip", compression_opts=6
        )

    @classmethod
    def load_from_group(
        cls: Type[T_PEPS_Tensor_Split_Transfer], grp: h5py.Group
    ) -> T_PEPS_Tensor_Split_Transfer:
        """
        Load the PEPS tensor from a HDF5 group.

        Args:
          grp (:obj:`h5py.Group`):
            HDF5 group object to load the data from.
        """
        d = int(grp.attrs["d"])
        D = tuple(int(i) for i in grp.attrs["D"])
        chi = int(grp.attrs["chi"])
        max_chi = int(grp.attrs["max_chi"])
        interlayer_chi = int(grp.attrs["interlayer_chi"])

        tensor = jnp.asarray(grp["tensor"])
        C1 = jnp.asarray(grp["C1"])
        C2 = jnp.asarray(grp["C2"])
        C3 = jnp.asarray(grp["C3"])
        C4 = jnp.asarray(grp["C4"])
        T1_ket = jnp.asarray(grp["T1_ket"])
        T1_bra = jnp.asarray(grp["T1_bra"])
        T2_ket = jnp.asarray(grp["T2_ket"])
        T2_bra = jnp.asarray(grp["T2_bra"])
        T3_ket = jnp.asarray(grp["T3_ket"])
        T3_bra = jnp.asarray(grp["T3_bra"])
        T4_ket = jnp.asarray(grp["T4_ket"])
        T4_bra = jnp.asarray(grp["T4_bra"])

        return cls(
            tensor=tensor,
            C1=C1,
            C2=C2,
            C3=C3,
            C4=C4,
            T1_ket=T1_ket,
            T1_bra=T1_bra,
            T2_ket=T2_ket,
            T2_bra=T2_bra,
            T3_ket=T3_ket,
            T3_bra=T3_bra,
            T4_ket=T4_ket,
            T4_bra=T4_bra,
            d=d,
            D=D,
            chi=chi,
            max_chi=max_chi,
            interlayer_chi=interlayer_chi,
        )

    @property
    def is_split_transfer(self: T_PEPS_Tensor_Split_Transfer) -> bool:
        return True

    def convert_to_split_transfer(
        self: T_PEPS_Tensor_Split_Transfer,
    ) -> T_PEPS_Tensor_Split_Transfer:
        return self

    def convert_to_full_transfer(self: T_PEPS_Tensor_Split_Transfer) -> T_PEPS_Tensor:
        T1 = jnp.tensordot(self.T1_ket, self.T1_bra, ((2,), (0,)))
        T2 = jnp.tensordot(self.T2_bra, self.T2_ket, ((2,), (0,)))
        T2 = T2.transpose(2, 1, 0, 3)
        T3 = jnp.tensordot(self.T3_ket, self.T3_bra, ((2,), (0,)))
        T3 = T3.transpose(0, 3, 2, 1)
        T4 = jnp.tensordot(self.T4_bra, self.T4_ket, ((2,), (0,)))

        return PEPS_Tensor(
            tensor=self.tensor,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
            C4=self.C4,
            T1=T1,
            T2=T2,
            T3=T3,
            T4=T4,
            d=self.d,
            D=self.D,
            chi=self.chi,
            max_chi=self.max_chi,
        )

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        data = (
            self.tensor,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.T1_ket,
            self.T1_bra,
            self.T2_ket,
            self.T2_bra,
            self.T3_ket,
            self.T3_bra,
            self.T4_ket,
            self.T4_bra,
        )
        aux_data = (self.d, self.D, self.chi, self.max_chi, self.interlayer_chi)

        return (data, aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_PEPS_Tensor_Split_Transfer],
        aux_data: Tuple[Any, ...],
        children: Tuple[Any, ...],
    ) -> T_PEPS_Tensor_Split_Transfer:
        (
            tensor,
            C1,
            C2,
            C3,
            C4,
            T1_ket,
            T1_bra,
            T2_ket,
            T2_bra,
            T3_ket,
            T3_bra,
            T4_ket,
            T4_bra,
        ) = children
        d, D, chi, max_chi, interlayer_chi = aux_data

        return cls(
            tensor=tensor,
            C1=C1,
            C2=C2,
            C3=C3,
            C4=C4,
            T1_ket=T1_ket,
            T1_bra=T1_bra,
            T2_ket=T2_ket,
            T2_bra=T2_bra,
            T3_ket=T3_ket,
            T3_bra=T3_bra,
            T4_ket=T4_ket,
            T4_bra=T4_bra,
            d=d,
            D=D,
            chi=chi,
            max_chi=max_chi,
            interlayer_chi=interlayer_chi,
            sanity_checks=False,
        )
