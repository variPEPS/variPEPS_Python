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
from varipeps.config import Contraction_Backend
from varipeps.peps import PEPS_Tensor

from . import tblis
from .definitions import Definitions, Definition

from typing import Sequence, List, Tuple, Dict, Union, Optional


@lru_cache(maxsize=1024)
def _path(network, shapes):
    path, _ = opt_einsum.contract_path(
        network,
        *shapes,
        shapes=True,
        optimize="optimal" if len(shapes) < 10 else "dp",
    )
    return tuple(map(tuple, path))


def _steps(inputs, output, path, sliced=frozenset()):
    """
    Pairwise steps of a contraction path: the index strings of the nodes (the
    inputs, then one node per step), whether a node depends on one of the
    ``sliced`` bonds, and the steps as (operand nodes, result node). The
    result of the last step carries the output indices (in its own order).
    """
    idx = list(inputs)
    dep = [any(b in i for b in sliced) for i in inputs]
    current, steps = list(range(len(inputs))), []
    for contraction in path:
        operands = tuple(current[k] for k in contraction)
        for k in sorted(contraction, reverse=True):
            del current[k]
        keep = set(output).union(*(idx[n] for n in current))
        union = "".join(dict.fromkeys("".join(idx[n] for n in operands)))
        idx.append("".join(c for c in union if c in keep))
        dep.append(any(dep[n] for n in operands))
        steps.append((operands, len(idx) - 1))
        current.append(len(idx) - 1)
    return idx, dep, steps


# Cost of reading or writing an entry, in multiply-adds
_MOVE = 32


def _cost(idx, dep, steps, dims, chunks, resident=None):
    """
    Cost, peak memory (entries) and multiply-adds of a sliced contraction
    with the extents ``dims`` of a chunk, as :func:`_einsum` runs it: first
    the steps independent of the sliced bonds, once, then the others once per
    chunk. The cost adds ``_MOVE`` per entry of the operands and the result of
    each step and of the summation of the output per chunk to the
    multiply-adds (of equal multiply-adds, fewer and larger chunks win).
    The peak is the largest intermediate (or output) if ``resident`` is
    ``None``, else the intermediates allocated at the same time (without the
    inputs), where the loops also keep the hoisted results they use, the
    summed output and ``resident`` entries (the reordered sliced inputs).
    """
    size = lambda n: math.prod(dims[c] for c in idx[n])
    cost, flops, peak, base, live = 0.0, 0.0, 0, 0, {}
    for loop in (False, True):
        if loop:
            base, live = sum(live.values()) + size(steps[-1][1]) + (resident or 0), {}
        for operands, n in steps:
            if dep[n] != loop:
                continue
            work = math.prod(dims[c] for c in set("".join(idx[o] for o in operands)))
            moves = size(n) + sum(size(o) for o in operands)
            flops += work * (chunks if loop else 1)
            cost += (work + _MOVE * moves) * (chunks if loop else 1)
            peak = max(
                peak,
                size(n) if resident is None else base + sum(live.values()) + size(n),
            )
            for o in operands:
                live.pop(o, None)
            live[n] = size(n)
    if chunks > 1:
        cost += 3 * _MOVE * chunks * size(steps[-1][1])
    return cost, peak, flops


def _largest(network, shapes):
    """Entries of the largest intermediate of the unsliced contraction."""
    lhs, output = network.split("->")
    inputs = lhs.split(",")
    extents = {c: d for idx, s in zip(inputs, shapes) for c, d in zip(idx, s)}
    return _cost(*_steps(inputs, output, _path(network, shapes)), extents, 1)[1]


def _search_slicing_plan(
    network, shapes, itemsize, max_bytes, live=False, max_flops=2.0
):
    """
    Bonds and chunk widths for a contraction whose peak memory (see
    :func:`_cost`) exceeds ``max_bytes``.

    The candidates are the bonds that connect exactly two input tensors. One
    bond: for each, the number of chunks is increased until the peak stays
    within ``max_bytes``; the largest width dividing the extent is examined as
    well (no padding). Several bonds: starting from the unsliced contraction,
    the chunk width of one bond is halved per step, choosing the step with the
    fewest flops (once with, once without hoisting) among those that lower the
    peak (else, at most twice in a row, among those that keep it), until it
    fits. Flops (the cost of :func:`_cost`, which includes memory traffic,
    except for ``max_flops``) and peak are those of :func:`_einsum`: the
    steps that do not depend on the sliced bonds are computed once, before the
    loops; for each slicing, the best order for the chunk shapes (one path
    search each, the expensive part: in total up to 3-4 minutes for a
    network of 20 tensors, see ``varipeps_config.contraction_slicing_plan_dir``)
    and the unsliced order are compared. Plans with more than ``max_flops``
    times the flops of the unsliced contraction are discarded, and so are
    plans whose peak exceeds both ``max_bytes`` and 3/4 of the unsliced peak.
    Among the others, choose the fewest flops that fit; if none fits, the
    fewest flops among those within 25% of the smallest peak; if none is left,
    do not slice.

    Args:
      network (:obj:`str`):
        The einsum network.
      shapes (:obj:`tuple` of :obj:`tuple` of :obj:`int`):
        Shapes of the input tensors.
      itemsize (:obj:`int`):
        Bytes per entry.
      max_bytes (:obj:`int`):
        Peak memory in bytes allowed without slicing.
      live (:obj:`bool`):
        Peak of all intermediates allocated at the same time (for checkpointed
        chunks, whose gradient keeps no intermediates of the chunks) instead
        of the largest intermediate.
      max_flops (:obj:`float`):
        Largest multiply-adds of a plan relative to the unsliced contraction.
    Returns:
      :obj:`tuple` or :obj:`None`:
        Tuple of (bond, chunk width) pairs and the contraction path of a chunk,
        or :obj:`None` if the contraction is not sliced.
    """
    if max_bytes <= 0:
        return None
    lhs, output = network.split("->")
    inputs = lhs.split(",")
    unsliced = _path(network, shapes)
    extents = {
        c: d
        for idx, s in zip(inputs, shapes, strict=True)
        for c, d in zip(idx, s, strict=True)
    }
    _, full, flops0 = _cost(
        *_steps(inputs, output, unsliced), extents, 1, 0 if live else None
    )
    full *= itemsize
    if full <= max_bytes:
        return None

    dims = {}
    for bond in sorted(set(lhs) - set(output) - {","}):
        owners = tuple(
            (idx, s) for idx, s in zip(inputs, shapes, strict=True) if bond in idx
        )
        if len(owners) != 2 or any(idx.count(bond) != 1 for idx, _ in owners):
            continue
        dim, other_dim = (s[idx.index(bond)] for idx, s in owners)
        if dim == other_dim and dim > 1:
            dims[bond] = dim

    seen = []

    def examine(widths, rank=0):
        """Cost, peak bytes, the plan for the chunk widths ``widths`` (bond ->
        width), the cost without hoisting and the multiply-adds (cost: of
        :func:`_cost` and ``2 * _MOVE`` per entry of the reordered sliced
        inputs), for the better (by ``rank``) of the best contraction order of
        a chunk (a path search, the expensive part) and the unsliced order
        (which may share more steps between the chunks); records the plans
        within ``max_flops``."""
        chunk_shapes = tuple(
            tuple(widths.get(i, d) for i, d in zip(idx, s, strict=True))
            for idx, s in zip(inputs, shapes, strict=True)
        )
        chunks = math.prod(-(-dims[b] // w) for b, w in widths.items())
        dims_chunk = {c: widths.get(c, d) for c, d in extents.items()}
        copies = sum(
            math.prod(
                -(-extents[c] // widths[c]) * widths[c] if c in widths else extents[c]
                for c in i
            )
            for i in inputs
            if set(i) & set(widths)
        )
        plans = []
        for path in dict.fromkeys((_path(network, chunk_shapes), unsliced)):
            idx, dep, steps = _steps(inputs, output, path, frozenset(widths))
            cost, peak, flops = _cost(
                idx, dep, steps, dims_chunk, chunks, copies if live else None
            )
            cost += 2 * _MOVE * copies
            plain = (
                _cost(idx, [True] * len(dep), steps, dims_chunk, chunks)[0]
                + 2 * _MOVE * copies
            )
            plans.append(
                (
                    cost,
                    itemsize * peak,
                    tuple(sorted(widths.items())),
                    path,
                    plain,
                    flops,
                )
            )
        seen.extend(p for p in plans if p[5] <= max_flops * flops0)
        return min(
            plans,
            key=lambda p: (p[5] > max_flops * flops0, p[1] > max_bytes, p[rank], p[1]),
        )

    for bond, dim in dims.items():
        chunks = min(dim, math.ceil(full / max_bytes))
        while True:
            width = -(-dim // chunks)
            chunks = -(-dim // width)
            plan = examine({bond: width})
            if plan[1] <= max_bytes or plan[5] > max_flops * flops0 or chunks == dim:
                break
            chunks = -(-dim // (width - 1))
        divisor = max(w for w in range(1, width + 1) if dim % w == 0)
        if divisor < width:  # no padded chunks
            examine({bond: divisor})

    # Greedy halving, once with the steps ranked by the flops with and once
    # without hoisting: two trajectories, the plans of both are examined.
    for rank in (0, 4):
        widths, peak, flat = {}, full, 0
        while peak > max_bytes:
            steps = [
                examine({**widths, b: -(-widths.get(b, d) // 2)}, rank)
                for b, d in dims.items()
                if widths.get(b, d) > 1
            ]
            steps = [p for p in steps if p[5] <= max_flops * flops0]
            # Steps that lower the peak, else (at most twice in a row) those
            # that keep it: intermediates of equal size may need different
            # bonds sliced.
            lower = [p for p in steps if p[1] < peak]
            flat = 0 if lower else flat + 1
            lower = lower or [p for p in steps if p[1] == peak]
            if not lower or flat > 2:
                break
            step = min(lower, key=lambda p: (p[rank], p[1]))
            widths, peak = dict(step[2]), step[1]

    def select():
        """The plans to choose from, best first: those that fit, else those
        within 25% of the smallest peak; only peaks of at most 3/4 of the
        unsliced one (a real gain)."""
        ok = [p for p in seen if p[1] <= max(max_bytes, 0.75 * full)]
        fits = [p for p in ok if p[1] <= max_bytes]
        if not fits and ok:
            low = min(p[1] for p in ok)
            fits = [p for p in ok if p[1] <= 1.25 * low]
        return sorted(fits, key=lambda p: p[:2])

    ranked = select()
    return ranked[0][2:4] if ranked else None


@lru_cache(maxsize=None)
def _slicing_plan(network, shapes, itemsize, max_bytes, live=False, plan_dir=""):
    """
    :func:`_search_slicing_plan`, read from and saved to ``plan_dir`` if set.

    Each plan is one JSON file ``<hash[:2]>/<hash[2:4]>/<hash>.json``, named by the SHA-256
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
        Threshold for the peak memory above which to slice.
      live (:obj:`bool`):
        Count all intermediates allocated at the same time (see
        :func:`_search_slicing_plan`).
      plan_dir (:obj:`str`):
        Directory of saved slicing plans; ``""`` keeps them in memory only.
    Returns:
      :obj:`tuple` or :obj:`None`:
        Tuple of (bond, chunk width) pairs and contraction path of a chunk,
        or ``None`` if the contraction is not sliced.
    """
    if not plan_dir:
        return _search_slicing_plan(network, shapes, itemsize, max_bytes, live)

    key = {
        "format": 2,
        "network": network,
        "shapes": [list(s) for s in shapes],
        "itemsize": itemsize,
        "max_bytes": max_bytes,
        "live": live,
    }
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    path = Path(plan_dir) / digest[:2] / digest[2:4] / f"{digest}.json"

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["key"] == key:
            plan = stored["plan"]
            if plan is None:
                return None
            return tuple(map(tuple, plan["bonds"])), tuple(map(tuple, plan["path"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass

    plan = _search_slicing_plan(network, shapes, itemsize, max_bytes, live)
    record = {
        "key": key,
        "plan": (
            None
            if plan is None
            else {
                "bonds": [list(b) for b in plan[0]],
                "path": [list(c) for c in plan[1]],
            }
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
    Contract an einsum network, as a sum over chunks of one or more bonds if
    its largest intermediate exceeds ``max_bytes``, with ``checkpoint`` if all
    intermediates allocated at the same time do (see
    :func:`_search_slicing_plan`).

    The chunks of each bond are summed in a :func:`jax.lax.scan`, nested for
    several bonds. The steps of the contraction path that do not depend on the
    sliced bonds are computed once before the loops, also in reverse-mode AD.
    With ``checkpoint``, the contraction of a chunk is checkpointed:
    reverse-mode AD keeps only the inputs of the chunks and recomputes the
    intermediates. Without it, reverse-mode AD stores the intermediates of all
    chunks, so slicing lowers the memory of a gradient only if one huge
    intermediate dominates; on CPU the contraction is then sliced only if its
    largest intermediate exceeds ``varipeps_config.contraction_slicing_min_bytes``. The einsum
    implementation is selected by ``varipeps_config.contraction_backend``.

    Args:
      network (:obj:`str`):
        The einsum network.
      tensors (:term:`sequence` of :obj:`jax.numpy.ndarray`):
        The input tensors.
      max_bytes (:obj:`int`):
        Peak memory in bytes allowed without slicing; ``0`` disables slicing,
        negative: 256 MiB on CPU, 128 MiB on other backends.
      checkpoint (:obj:`bool`):
        Recompute the intermediates of a chunk in reverse-mode AD instead of
        storing them.
      plan_dir (:obj:`str`):
        Directory of saved slicing plans; ``""`` keeps them in memory only.
    Returns:
      :obj:`jax.numpy.ndarray`:
        The contracted tensor.
    """
    if max_bytes < 0:
        max_bytes = 2**28 if jax.default_backend() == "cpu" else 2**27
    backend = varipeps_config.contraction_backend
    use_tblis = backend is Contraction_Backend.TBLIS or (
        backend is Contraction_Backend.TBLIS_COMPLEX
        and jnp.issubdtype(jnp.result_type(*tensors), jnp.complexfloating)
    )
    einsum = tblis.einsum if use_tblis else jnp.einsum
    optimize = "optimal" if len(tensors) < 10 else "dp"
    plan = None
    shapes = tuple(t.shape for t in tensors)
    itemsize = jnp.result_type(*tensors).itemsize
    if max_bytes > 0 and all(isinstance(d, int) for s in shapes for d in s):
        # On CPU, chunks without checkpoints cost time and seldom lower the
        # memory of gradients: slice only huge intermediates.
        if (
            checkpoint
            or jax.default_backend() != "cpu"
            or itemsize * _largest(network, shapes)
            > varipeps_config.contraction_slicing_min_bytes
        ):
            plan = _slicing_plan(
                network, shapes, itemsize, max_bytes, checkpoint, plan_dir
            )
    if plan is None:
        return einsum(network, *tensors, optimize=optimize)

    bonds, path = plan
    lhs, output = network.split("->")
    inputs = lhs.split(",")
    result = jax.eval_shape(partial(jnp.einsum, network, optimize=optimize), *tensors)
    idx, dep, steps = _steps(inputs, output, path, frozenset(b for b, _ in bonds))
    producer = {n: s for s in steps for n in s[1:]}

    def subtree(n):
        if n not in producer:
            return []
        return [s for o in producer[n][0] for s in subtree(o)] + [producer[n]]

    def contract_steps(values, todo, out):
        """One einsum of the steps ``todo``, with their order as path."""
        made = {n for _, n in todo}
        leaves = [o for ops, _ in todo for o in ops if o not in made]
        current, sub = list(leaves), []
        for ops, n in todo:
            sub.append(tuple(current.index(o) for o in ops))
            current = [c for c in current if c not in ops] + [n]
        return einsum(
            ",".join(idx[o] for o in leaves) + "->" + out,
            *(values[o] for o in leaves),
            optimize=sub,
        )

    # Subtrees that do not depend on the sliced bonds: once, before the loops.
    todo = [s for s in steps if dep[s[1]]]
    roots = {o for ops, _ in todo for o in ops if not dep[o] and o in producer}
    hoisted = {
        r: contract_steps(dict(enumerate(tensors)), subtree(r), idx[r]) for r in roots
    }

    def contract_chunk(ts):
        return contract_steps({**hoisted, **dict(enumerate(ts))}, todo, output)

    if checkpoint:
        contract_chunk = jax.checkpoint(contract_chunk)

    def contract(ts, bonds):
        if not bonds:
            return contract_chunk(ts)
        (bond, width), rest = bonds[0], bonds[1:]
        owners = [k for k, i in enumerate(inputs) if bond in i]
        dim = ts[owners[0]].shape[inputs[owners[0]].index(bond)]
        chunks = -(-dim // width)
        sliced = []
        for k in owners:
            t = ts[k]
            axis = inputs[k].index(bond)
            t = jnp.pad(
                t,
                [(0, chunks * width - dim if a == axis else 0) for a in range(t.ndim)],
            )
            t = t.reshape(t.shape[:axis] + (chunks, width) + t.shape[axis + 1 :])
            sliced.append(jnp.moveaxis(t, axis, 0))

        def step(acc, chunk):
            parts = list(ts)
            for k, c in zip(owners, chunk, strict=True):
                parts[k] = c
            return acc + contract(parts, rest), None

        return jax.lax.scan(step, jnp.zeros(result.shape, result.dtype), tuple(sliced))[
            0
        ]

    return contract(list(tensors), bonds)


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
