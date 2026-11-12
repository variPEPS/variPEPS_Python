"""
Helpers to apply contractions.
"""

from functools import lru_cache, partial
import hashlib
import json
import math
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import opt_einsum

from varipeps import varipeps_config
from varipeps.peps import PEPS_Tensor

from .definitions import Definitions, Definition

from typing import Sequence, List, Tuple, Dict, Union, Optional


def _path(network, shapes):
    return opt_einsum.contract_path(
        network,
        *shapes,
        shapes=True,
        optimize="optimal" if len(shapes) < 10 else "dp",
    )


def _search_slicing_plan(network, shapes, itemsize, max_bytes):
    """
    Bond and number of chunks for a contraction whose largest intermediate
    exceeds ``max_bytes``.

    The candidates are the bonds that connect exactly two input tensors. For
    each, the number of chunks is increased until every intermediate stays
    within ``max_bytes``. This includes the intermediates that do not depend
    on the bond: they are the same for all chunks, so XLA moves them out of
    the loop and computes them once in full. Among the plans examined, choose
    the lowest flop count that fits. If none fits, minimize the largest
    intermediate and break ties by flop count.

    Args:
      network (:obj:`str`):
        The einsum network.
      shapes (:obj:`tuple` of :obj:`tuple` of :obj:`int`):
        Shapes of the input tensors.
      itemsize (:obj:`int`):
        Bytes per entry.
      max_bytes (:obj:`int`):
        Largest size of an intermediate without slicing.
    Returns:
      :obj:`tuple` or :obj:`None`:
        Bond, number of chunks and contraction path of a chunk, or
        :obj:`None` if the contraction is not sliced.
    """
    if max_bytes <= 0:
        return None
    lhs, output = network.split("->")
    inputs = lhs.split(",")
    _, info = _path(network, shapes)
    largest = itemsize * int(info.largest_intermediate)
    if largest <= max_bytes:
        return None

    fitting = None
    smallest = None
    for bond in sorted(set(lhs) - set(output) - {","}):
        owners = tuple(
            (idx, s) for idx, s in zip(inputs, shapes, strict=True) if bond in idx
        )
        if len(owners) != 2 or any(idx.count(bond) != 1 for idx, _ in owners):
            continue
        dim, other_dim = (s[idx.index(bond)] for idx, s in owners)
        if dim != other_dim or dim <= 1:
            continue
        chunks = min(dim, math.ceil(largest / max_bytes))
        while True:
            width = -(-dim // chunks)
            chunks = -(-dim // width)
            chunk_shapes = tuple(
                tuple(width if i == bond else d for i, d in zip(idx, s, strict=True))
                for idx, s in zip(inputs, shapes, strict=True)
            )
            path, info = _path(network, chunk_shapes)
            peak = itemsize * int(info.largest_intermediate)
            plan = (chunks * float(info.opt_cost), peak, bond, width, tuple(path))
            if peak <= max_bytes:
                fitting = min(fitting or plan, plan, key=lambda p: p[:2] + (-p[3],))
                break
            smallest = min(smallest or plan, plan, key=lambda p: (p[1], p[0]))
            if chunks == dim:
                break
            chunks = -(-dim // (width - 1))

    plan = fitting or smallest
    if plan is None or plan[1] >= largest:
        return None
    return plan[2:]


@lru_cache(maxsize=None)
def _slicing_plan(network, shapes, itemsize, max_bytes, plan_dir=""):
    """
    :func:`_search_slicing_plan`, read from and saved to ``plan_dir`` if set.

    Each plan is one JSON file ``<hash[:2]>/<hash>.json``, named by the SHA-256
    of its inputs, which the file also stores and which are compared on
    reading, so a stale or foreign file is never used. A plan of ``None`` (do
    not slice) is saved too. A missing or unreadable file means the plan is
    searched, a file that cannot be written is skipped. Files are written to a
    temporary name and renamed (removed on failure), so concurrent jobs never
    read a partial file.

    Args:
      network (:obj:`str`):
        Einsum subscripts of the contraction.
      shapes (:obj:`tuple`):
        Shapes of the tensors.
      itemsize (:obj:`int`):
        Size of one element in bytes.
      max_bytes (:obj:`int`):
        Threshold for the largest intermediate above which to slice.
      plan_dir (:obj:`str`):
        Directory of saved slicing plans; ``""`` keeps them in memory only.
    Returns:
      :obj:`tuple` or :obj:`None`:
        Bond, width and contraction path of the sliced contraction, or
        ``None`` if it is not sliced.
    """
    if not plan_dir:
        return _search_slicing_plan(network, shapes, itemsize, max_bytes)

    key = {
        "format": 1,
        "network": network,
        "shapes": [list(s) for s in shapes],
        "itemsize": itemsize,
        "max_bytes": max_bytes,
    }
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    path = Path(plan_dir) / digest[:2] / digest[2:4] / f"{digest}.json"

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["key"] == key:
            plan = stored["plan"]
            if plan is None:
                return None
            return plan["bond"], plan["width"], tuple(map(tuple, plan["path"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass

    plan = _search_slicing_plan(network, shapes, itemsize, max_bytes)
    record = {
        "key": key,
        "plan": (
            None
            if plan is None
            else {"bond": plan[0], "width": plan[1], "path": [list(c) for c in plan[2]]}
        ),
    }
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump(record, f, ensure_ascii=False, indent=1)
        tmp.replace(path)
    except OSError:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return plan


def _einsum(network, tensors, max_bytes, checkpoint, plan_dir=""):
    """
    Contract an einsum network, as a sum over chunks of one bond if its
    largest intermediate exceeds ``max_bytes`` (see :func:`_slicing_plan`).

    The chunks are summed in a :func:`jax.lax.scan`. With ``checkpoint``,
    the contraction of a chunk is checkpointed: reverse-mode AD keeps only
    the inputs of the chunks and recomputes the intermediates.

    Args:
      network (:obj:`str`):
        The einsum network.
      tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The input tensors.
      max_bytes (:obj:`int`):
        Largest size of an intermediate without slicing; ``0`` disables
        slicing.
      checkpoint (:obj:`bool`):
        Recompute the intermediates of a chunk in reverse-mode AD instead of
        storing them.
      plan_dir (:obj:`str`):
        Directory of saved slicing plans; ``""`` keeps them in memory only.
    Returns:
      :obj:`jax.numpy.ndarray`:
        The contracted tensor.
    """
    optimize = "optimal" if len(tensors) < 10 else "dp"
    plan = None
    if max_bytes > 0 and all(isinstance(d, int) for t in tensors for d in t.shape):
        plan = _slicing_plan(
            network,
            tuple(t.shape for t in tensors),
            jnp.result_type(*tensors).itemsize,
            max_bytes,
            plan_dir,
        )
    if plan is None:
        return jnp.einsum(network, *tensors, optimize=optimize)

    bond, width, path = plan
    inputs = network.split("->")[0].split(",")
    owners = [k for k, i in enumerate(inputs) if bond in i]
    dim = tensors[owners[0]].shape[inputs[owners[0]].index(bond)]
    chunks = -(-dim // width)

    sliced = []
    for k in owners:
        t = tensors[k]
        axis = inputs[k].index(bond)
        t = jnp.pad(
            t, [(0, chunks * width - dim if a == axis else 0) for a in range(t.ndim)]
        )
        t = t.reshape(t.shape[:axis] + (chunks, width) + t.shape[axis + 1 :])
        sliced.append(jnp.moveaxis(t, axis, 0))
    others = [t for k, t in enumerate(tensors) if k not in owners]

    def contract_chunk(chunk):
        ts = list(others)
        for k, c in zip(owners, chunk, strict=True):
            ts.insert(k, c)
        return jnp.einsum(network, *ts, optimize=list(path))

    if checkpoint:
        contract_chunk = jax.checkpoint(contract_chunk)

    result = jax.eval_shape(partial(jnp.einsum, network, optimize=optimize), *tensors)
    return jax.lax.scan(
        lambda acc, chunk: (acc + contract_chunk(chunk), None),
        jnp.zeros(result.shape, result.dtype),
        tuple(sliced),
    )[0]


@partial(jax.jit, static_argnames=("name", "disable_identity_check"))
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

    return _einsum(
        contraction["einsum_network"],
        tensors,
        varipeps_config.contraction_slicing_max_bytes,
        varipeps_config.contraction_slicing_checkpoint,
        varipeps_config.contraction_slicing_plan_dir,
    )


apply_contraction_jitted = apply_contraction
