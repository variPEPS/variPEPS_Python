"""
Helpers to apply contractions.
"""

import collections
from functools import partial

import jax
import jax.numpy as jnp
import tensornetwork as tn
from tensornetwork.ncon_interface import _jittable_ncon

from varipeps.peps import PEPS_Tensor
from varipeps import varipeps_config
from varipeps.config import VariPEPS_Config
from varipeps.utils.func_cache import Checkpointing_Cache

from .definitions import Definitions, Definition

from typing import Sequence, List, Tuple, Dict, Union, Optional


class _Contraction_Cache:
    _cache = None

    def __class_getitem__(cls, name: str) -> Checkpointing_Cache:
        name = f"_{name}"
        obj = getattr(cls, name)
        if obj is None:
            obj = Checkpointing_Cache(varipeps_config.checkpointing_ncon)
            setattr(cls, name, obj)
        return obj


_ncon_jitted = jax.jit(_jittable_ncon, static_argnums=(1, 2, 3, 4, 5), inline=True)


def apply_contraction(
    name: str,
    peps_tensors: Sequence[jnp.ndarray],
    peps_tensor_objs: Sequence[PEPS_Tensor],
    additional_tensors: Sequence[jnp.ndarray],
    *,
    disable_identity_check: bool = False,
    custom_definition: Optional[Definition] = None,
    config: VariPEPS_Config = varipeps_config,
    _jitable: bool = False,
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
      custom_definition (:obj:`~varipeps.contractions.apply.Definition`, optional):
        Use a custom definition for the contraction which is not defined in the
        :class:`varipeps.contractions.Definitions` class.
      config (:obj:`~varipeps.config.VariPEPS_Config`):
        Global configuration object of the variPEPS library. Please see its
        class definition for details.
    Returns:
      jax.numpy.ndarray:
        The contracted tensor.
    """
    if len(peps_tensors) != len(peps_tensor_objs):
        raise ValueError(
            "Number of PEPS tensors have to match number of PEPS tensor objects."
        )

    if (
        not _jitable
        and not disable_identity_check
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

    if custom_definition is not None:
        contraction = custom_definition
    else:
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
                tensors.append(peps_tensors[ti].conj())
            else:
                tensors.append(getattr(peps_tensor_objs[ti], f))

    tensors += additional_tensors

    if _jitable:
        if config.checkpointing_ncon:
            f = jax.checkpoint(_ncon_jitted, static_argnums=(1, 2, 3, 4, 5))
        else:
            f = _ncon_jitted

        return f(
            tensors,
            contraction["ncon_flat_network"],
            contraction["ncon_sizes"],
            contraction["ncon_con_order"],
            contraction["ncon_out_order"],
            tn.backends.backend_factory.get_backend("jax"),
        )

    if config.checkpointing_ncon:
        f = jax.checkpoint(
            partial(
                tn.ncon, network_structure=contraction["ncon_network"], backend="jax"
            )
        )
    else:
        f = partial(
            tn.ncon, network_structure=contraction["ncon_network"], backend="jax"
        )

    return f(tensors)


apply_contraction_jitted = jax.jit(
    partial(apply_contraction, _jitable=True),
    static_argnames=("name", "disable_identity_check", "custom_definition"),
)
