"""
Helpers to apply contractions.
"""

import collections

import jax
import jax.numpy as jnp
import tensornetwork as tn

from peps_ad.peps import PEPS_Tensor

from .definitions import Definitions

from typing import Sequence, List, Tuple


def apply_contraction(
    name: str,
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    additional_tensors: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    """
    Apply a contraction to a list of tensors.

    For details on the contractions and their definition see
    :class:`peps_ad.contractions.Definitions`.

    Args:
      name (:obj:`str`):
        Name of the contraction. Must be a class attribute of the class
        :class:`peps_ad.contractions.Definitions`.
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays that should be contracted.
      peps_tensor_objs (:term:`sequence` of :obj:`~peps_ad.peps.PEPS_Tensor`):
        The PEPS tensor objects corresponding the the arrays. These arguments are
        split up due to limitation of the jax library.
      additional_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Additional non-PEPS tensors which should be contracted (e.g. gates).
    Returns:
      jax.numpy.ndarray:
        The contracted tensor.
    """
    if len(peps_tensors) != len(peps_tensor_objs):
        raise ValueError(
            "Number of PEPS tensors have to match number of PEPS tensor objects."
        )

    if not all(isinstance(t, jax.core.Tracer) for t in peps_tensors) and not all(
        peps_tensors[i] is peps_tensor_objs[i].tensor for i in range(len(peps_tensors))
    ):
        raise ValueError(
            "Sequence of PEPS tensors mismatch the objects sequence. Please check your code!"
        )

    contraction = getattr(Definitions, name)

    filter_peps_tensors: List[Sequence[str]] = []
    filter_additional_tensors: List[str] = []

    network_peps_tensors: List[List[Tuple[int, ...]]] = []
    network_additional_tensors: List[Tuple[int, ...]] = []

    for t in contraction["tensors"]:
        if isinstance(t, collections.abc.Sequence):
            if len(filter_additional_tensors) != 0:
                raise ValueError("Invalid specification for contraction.")

            filter_peps_tensors.append(t)
        else:
            filter_additional_tensors.append(t)

    for n in contraction["network"]:
        if isinstance(n, collections.abc.Sequence) and all(
            isinstance(ni, collections.abc.Sequence) for ni in n
        ):
            if len(network_additional_tensors) != 0:
                raise ValueError("Invalid specification for contraction.")

            network_peps_tensors.append(n)  # type: ignore
        elif isinstance(n, collections.abc.Sequence) and all(
            isinstance(ni, int) for ni in n
        ):
            network_additional_tensors.append(n)  # type: ignore
        else:
            raise ValueError("Invalid specification for contraction.")

    if len(filter_peps_tensors) != len(peps_tensors):
        raise ValueError(
            "Number of PEPS tensor objects does not fit the expected number."
        )

    if len(filter_additional_tensors) != len(additional_tensors):
        raise ValueError(
            "Number of additional tensor objects does not fit the expected number."
        )

    if len(network_peps_tensors) != len(filter_peps_tensors) or not all(
        len(network_peps_tensors[i]) == len(filter_peps_tensors[i])
        for i in range(len(filter_peps_tensors))
    ):
        raise ValueError("Invalid specification for contraction.")

    if not isinstance(additional_tensors, list):
        additional_tensors = list(additional_tensors)

    tensors = []

    for ti, t_filter in enumerate(filter_peps_tensors):
        for f in t_filter:
            if f == "tensor":
                tensors.append(peps_tensors[ti])
            elif f == "tensor_conj":
                tensors.append(peps_tensors[ti].conj())
            else:
                tensors.append(getattr(peps_tensor_objs[ti], f))

    tensors += additional_tensors

    network = [j for i in network_peps_tensors for j in i] + network_additional_tensors

    return tn.ncon(tensors, network, backend="jax")
