"""
Helpers to apply contractions.
"""

from functools import partial

import jax
import jax.numpy as jnp

from varipeps.peps import PEPS_Tensor

from .definitions import Definitions, Definition

from typing import Sequence, List, Tuple, Dict, Union, Optional


@partial(
    jax.jit, static_argnames=("name", "disable_identity_check")
)
def apply_contraction(
    name: str,
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    additional_tensors: Sequence[jnp.ndarray],
    *,
    disable_identity_check: bool = True,
) -> jnp.ndarray:
    """
    Apply a contraction to a list of tensors.

    For details on the contractions and their definition see
    :class:`varipeps.contractions.Definitions`.

    Args:
      name (:obj:`str`):
        Name of the contraction. Must be a class attribute of the class
        :class:`varipeps.contractions.Definitions`.
      peps_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS tensor arrays that should be contracted.
      peps_tensor_objs (:term:`sequence` of :obj:`~varipeps.peps.PEPS_Tensor`):
        The PEPS tensor objects corresponding the the arrays. These arguments are
        split up due to limitation of the jax library.
      additional_tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        Additional non-PEPS tensors which should be contracted (e.g. gates).
    Keyword args:
      disable_identity_check (:obj:`bool`):
        Disable the check if the tensor is identical to the one of the
        corresponding object.
    Returns:
      jax.numpy.ndarray:
        The contracted tensor.
    """
    if len(peps_tensors) != len(peps_tensor_objs):
        raise ValueError(
            "Number of PEPS tensors have to match number of PEPS tensor objects."
        )

    if (
        not disable_identity_check
        and not all(isinstance(t, jax.core.Tracer) for t in peps_tensors)
        and not all(isinstance(to.tensor, jax.core.Tracer) for to in peps_tensor_objs)
        and not all(
            jnp.allclose(peps_tensors[i], peps_tensor_objs[i].tensor)
            for i in range(len(peps_tensors))
        )
    ):
        raise ValueError(
            "Sequence of PEPS tensors mismatch the objects sequence. Please check your code!"
        )

    contraction = getattr(Definitions, name)

    if len(contraction["filter_peps_tensors"]) != len(peps_tensors):
        raise ValueError(
            f"Number of PEPS tensor ({len(peps_tensors)}) objects does not fit the expected number ({len(contraction['filter_peps_tensors'])})."
        )

    if len(contraction["filter_additional_tensors"]) != len(additional_tensors):
        raise ValueError(
            f"Number of additional tensor ({len(additional_tensors)}) objects does not fit the expected number ({len(contraction['filter_additional_tensors'])})."
        )

    if not isinstance(additional_tensors, list):
        additional_tensors = list(additional_tensors)

    tensors = []

    for ti, t_filter in enumerate(contraction["filter_peps_tensors"]):
        for f in t_filter:
            if f == "tensor":
                tensors.append(peps_tensors[ti])
            elif f == "tensor_conj":
                if (
                    hasattr(peps_tensor_objs[ti], "tensor_conj")
                    and peps_tensor_objs[ti].tensor_conj is not None
                ):
                    tensors.append(peps_tensor_objs[ti].tensor_conj)
                else:
                    tensors.append(peps_tensors[ti].conj())
            else:
                tensors.append(getattr(peps_tensor_objs[ti], f))

    tensors += additional_tensors

    tensor_shapes = tuple(tuple(e.shape) for e in tensors)

    return jnp.einsum(
        contraction["einsum_network"],
        *tensors,
        optimize="optimal" if len(tensors) < 10 else "dp",
    )


apply_contraction_jitted = apply_contraction
