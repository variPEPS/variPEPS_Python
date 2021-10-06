"""
Helpers to apply contractions.
"""

import collections

import jax.numpy as jnp
import tensornetwork as tn

from peps_ad.peps import PEPS_Tensor

from .definitions import Definitions

from typing import Sequence


def apply_contraction(
    name: str,
    peps_tensors: Sequence[PEPS_Tensor],
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
      peps_tensors (:term:`sequence` of :obj:`PEPS_Tensor`):
        The PEPS tensor objects that should be contracted.
      additional_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Additional non-PEPS tensors which should be contracted (e.g. gates).
    Returns:
      jax.numpy.ndarray:
        The contracted tensor.
    """
    contraction = getattr(Definitions, name)

    filter_peps_tensors = []
    filter_additional_tensors = []

    network_peps_tensors = []
    network_additional_tensors = []

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

            network_peps_tensors.append(n)
        elif isinstance(n, collections.abc.Sequence) and all(
            isinstance(ni, int) for ni in n
        ):
            network_additional_tensors.append(n)
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
            if f == "tensor_conj":
                tensors.append(getattr(peps_tensors[ti], "tensor").conj())
            else:
                tensors.append(getattr(peps_tensors[ti], f))

    tensors += additional_tensors

    network = [j for i in network_peps_tensors for j in i] + network_additional_tensors

    return tn.ncon(tensors, network, backend="jax")
