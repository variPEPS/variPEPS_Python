"""
Tensor contractions with the tblis library through the TAPP interface.

Pairwise contractions are done by tblis directly on the row-major arrays of
jax, i.e. without explicit transposes of the operands. Use
:obj:`varipeps.config.contraction_backend` to use these functions for the
contractions in :func:`~varipeps.contractions.apply_contraction`. tblis
itself runs single-threaded; large contractions are split into independent
parts which run on the intra-op thread pool of XLA (no extra threads).
``TBLIS_NUM_THREADS`` limits the number of parts (default: size of the pool).
The split uses a cost model which is calibrated on this machine at
the first contraction, see :func:`cost_model`.
"""

from functools import partial
import math
import os

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
import numpy as np
import opt_einsum

try:
    import varipeps.utils.extensions._tblis_contract as _tblis_contract_lib
except ModuleNotFoundError:  # built without tblis (Windows): JAX contractions only
    _tblis_contract_lib = None

from typing import Sequence, Tuple, Union


def _set_blis_arch():
    """
    Choose the BLIS kernels on AMD CPUs which the bundled BLIS does not detect
    well: it takes the AVX2 kernels (``zen3``) on Zen 4 and the slow reference
    kernels (``generic``) on Zen 5 and later (CPU family 0x1A and above). With
    AVX-512 this sets ``BLIS_ARCH_TYPE=skx`` (AVX-512 kernels), otherwise
    ``zen3`` from family 0x1A on. A value set by the user is kept. BLIS reads
    the variable at the first contraction.
    """
    if "BLIS_ARCH_TYPE" in os.environ:
        return
    cpu = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if not line.strip():
                    break
                key, _, value = line.partition(":")
                cpu[key.strip()] = value.strip()
        family = int(cpu.get("cpu family", 0))
    except (OSError, ValueError):
        return
    if cpu.get("vendor_id") != "AuthenticAMD" or family < 0x19:
        return
    if {"avx512f", "avx512dq", "avx512bw", "avx512vl"} <= set(
        cpu.get("flags", "").split()
    ):
        os.environ["BLIS_ARCH_TYPE"] = "skx"
    elif family > 0x19:
        os.environ["BLIS_ARCH_TYPE"] = "zen3"


_set_blis_arch()

_DTYPE_SUFFIX = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}

for _suffix in _DTYPE_SUFFIX.values() if _tblis_contract_lib is not None else ():
    jax.ffi.register_ffi_target(
        f"tblis_contract_{_suffix}",
        getattr(_tblis_contract_lib, f"tblis_contract_{_suffix}")(),
        platform="cpu",
    )


_cost_models = {}


def cost_model(dtype=jnp.float64):
    """
    Get the cost model that decides how a contraction is split into parts
    that run in parallel.

    The model is set up once per data type, at the first call: a small
    matrix product measures the multiply-add rate of this machine, which gives
    the smallest part worth running separately (about 20 microseconds of work). The other
    values are fixed. Environment variables override them, mainly for
    benchmarks: ``VARIPEPS_TBLIS_CALIBRATE=0`` skips the measurement
    (reproducible splits); ``VARIPEPS_TBLIS_MIN_PART_WORK``,
    ``VARIPEPS_TBLIS_MIN_PART_SCALE``, ``VARIPEPS_TBLIS_MOVE_COST``,
    ``VARIPEPS_TBLIS_REDUCE_COST`` and ``VARIPEPS_TBLIS_MAX_PARTIAL_BYTES`` set
    the returned values; ``VARIPEPS_TBLIS_FOLD_PENALTY`` and
    ``VARIPEPS_TBLIS_EVEN_PARTS`` change which index is split;
    ``VARIPEPS_TBLIS_VERBOSE=1`` prints the model. The first call also tests
    whether the parts of a split should map the pages of a new output before
    writing it (Linux; it depends on the kernel), ``VARIPEPS_TBLIS_PREFAULT=0``
    or ``1`` sets this without the test.

    Args:
      dtype (:obj:`jax.numpy.dtype`):
        Data type of the contraction (float32, float64, complex64 or
        complex128).
    Returns:
      :obj:`tuple`\\ (:obj:`float`, :obj:`float`, :obj:`float`, :obj:`float`):
        Smallest work of a part (multiply-adds), cost of packing one entry of
        an operand and of adding one entry of a partial result (both in
        multiply-adds), and the memory limit of the partial results of a
        split over a summed index (bytes).
    """
    if _tblis_contract_lib is None:
        raise RuntimeError(
            "variPEPS was built without the tblis backend (not available on Windows)."
        )
    suffix = _DTYPE_SUFFIX[jnp.dtype(dtype)]
    if suffix not in _cost_models:
        _cost_models[suffix] = _tblis_contract_lib.cost_model(suffix)
    return _cost_models[suffix]


# The pairwise contraction is a bilinear primitive, so that jax can linearize
# and transpose it (needed e.g. by the linear solver of the CTMRG gradient).
_contract_p = Primitive("tblis_contract")


@partial(jax.jit, static_argnames=("idx_a", "idx_b", "idx_out"))
def _contract_impl(a, b, *, idx_a, idx_b, idx_out):
    return _contract_p.bind(a, b, idx_a=idx_a, idx_b=idx_b, idx_out=idx_out)


_contract_p.def_impl(_contract_impl)


@_contract_p.def_abstract_eval
def _contract_abstract_eval(a, b, *, idx_a, idx_b, idx_out):
    extents = dict(zip(idx_a, a.shape, strict=True))
    extents.update(zip(idx_b, b.shape, strict=True))
    return jax.core.ShapedArray(tuple(extents[i] for i in idx_out), a.dtype)


def _contract_lowering(ctx, a, b, *, idx_a, idx_b, idx_out):
    suffix = _DTYPE_SUFFIX[ctx.avals_in[0].dtype]
    return jax.ffi.ffi_lowering(f"tblis_contract_{suffix}")(
        ctx,
        a,
        b,
        idx_a=np.asarray(idx_a, dtype=np.int64),
        idx_b=np.asarray(idx_b, dtype=np.int64),
        idx_d=np.asarray(idx_out, dtype=np.int64),
    )


mlir.register_lowering(_contract_p, _contract_lowering, platform="cpu")

# Derivatives. ad.defbilinear registers for p(a, b), linear in a and in b:
# forward mode p(da, b) + p(a, db) (zero tangents skipped), and reverse mode the
# two rules below, called with the cotangent ct of the output when the cotangent
# of a (first rule) or of b (second rule) is needed; the operand that is
# differentiated arrives as a placeholder (only its shape is known). The rules
# are again contractions (no conjugation, as for jnp.einsum) and go through
# contract(), so that its output layout applies to gradients too.
ad.defbilinear(
    _contract_p,
    lambda ct, a, b, *, idx_a, idx_b, idx_out: contract(ct, b, idx_out, idx_b, idx_a),
    lambda ct, a, b, *, idx_a, idx_b, idx_out: contract(a, ct, idx_a, idx_out, idx_b),
)


def _contract_batch(args, dims, *, idx_a, idx_b, idx_out):
    # The vmapped axis becomes an additional (batch) label.
    new = max(idx_a + idx_b + idx_out, default=0) + 1
    a, b = args
    if dims[0] is not None:
        a, idx_a = jnp.moveaxis(a, dims[0], 0), (new,) + idx_a
    if dims[1] is not None:
        b, idx_b = jnp.moveaxis(b, dims[1], 0), (new,) + idx_b
    out = _contract_p.bind(a, b, idx_a=idx_a, idx_b=idx_b, idx_out=(new,) + idx_out)
    return out, 0


batching.primitive_batchers[_contract_p] = _contract_batch


def _out_layout_order(idx_a, idx_b, idx_out, shape_a, shape_b, dtype):
    """
    Order of the output labels in which :func:`contract` lets tblis compute the
    result before XLA transposes it, or ``None`` to compute ``idx_out``
    directly. The order (batch labels, Y, X, X') is used if

    * the free (non-batch) labels of the output form exactly three blocks
      X, Y, X' (X and X' from one operand, Y from the other),
    * the labels of Y are in another order in their operand than in the output
      (in the same order, tblis writes the result about as fast),
    * the last output label is free and has 2 to 7 values (complex: not 4, the
      block size of tblis for complex numbers), and
    * every output entry takes at least ``VARIPEPS_TBLIS_OUT_LAYOUT``
      multiply-adds (default 1024, so that the transpose is cheap in
      comparison; 0 disables the rule; read when :func:`contract` is traced).
    """
    free = [lab for lab in idx_out if lab not in idx_a or lab not in idx_b]
    # positions in free where the operand changes (two for X, Y, X')
    breaks = [
        k for k in range(1, len(free)) if (free[k] in idx_a) != (free[k - 1] in idx_a)
    ]
    if len(breaks) != 2 or free[-1] != idx_out[-1]:
        return None
    k1, k2 = breaks  # X = free[:k1], Y = free[k1:k2], X' = free[k2:]
    y = free[k1:k2]
    extent = dict(zip(idx_a, shape_a, strict=True))
    extent.update(zip(idx_b, shape_b, strict=True))
    last = extent[free[-1]]
    madds_per_entry = math.prod(extent.values()) / max(
        1, math.prod(extent[lab] for lab in idx_out)
    )
    min_madds = float(os.environ.get("VARIPEPS_TBLIS_OUT_LAYOUT") or 1024)
    if (
        [lab for lab in (idx_a if y[0] in idx_a else idx_b) if lab in y] != y
        and 2 <= last < 8
        and not (jnp.issubdtype(dtype, jnp.complexfloating) and last % 4 == 0)
        and 0 < min_madds <= madds_per_entry
    ):
        return tuple(
            [lab for lab in idx_out if lab not in free] + y + free[:k1] + free[k2:]
        )
    return None


def contract(
    a: jnp.ndarray,
    b: jnp.ndarray,
    idx_a: Tuple[int, ...],
    idx_b: Tuple[int, ...],
    idx_out: Tuple[int, ...],
) -> jnp.ndarray:
    """
    Contract two tensors with tblis, similar to
    ``jnp.einsum(a, idx_a, b, idx_b, idx_out)``.

    Labels shared by ``a`` and ``b`` but not in ``idx_out`` are summed over,
    labels in all three are batch labels. Every label of ``a`` and ``b`` has
    to occur in the other operand or in ``idx_out`` and labels must not repeat
    within one tensor (use :func:`einsum` for these cases).

    Args:
      a (:obj:`jax.numpy.ndarray`):
        First operand.
      b (:obj:`jax.numpy.ndarray`):
        Second operand.
      idx_a (:obj:`tuple` of :obj:`int`):
        Labels of the axes of ``a``.
      idx_b (:obj:`tuple` of :obj:`int`):
        Labels of the axes of ``b``.
      idx_out (:obj:`tuple` of :obj:`int`):
        Labels of the axes of the result.
    Returns:
      :obj:`jax.numpy.ndarray`:
        The contracted tensor.
    """
    dtype = jnp.result_type(a, b)
    if dtype not in _DTYPE_SUFFIX:
        raise ValueError(f"Unsupported dtype {dtype} for tblis contraction.")
    cost_model(dtype)  # measure now (trace time), not during an XLA execution

    # Output layout. tblis writes the result slowly in some layouts, e.g. in
    # purity_ctrmg_proj_phys_1x1_top
    #   a[c,r,s,h,l,q] b[m,c,h,n,t,l,o,p] -> out[m,n,o,p, q,r,s, t]  (t: 2 values)
    # with m,n,o,p,t from b and q,r,s from a. Then the result is computed as
    # out[q,r,s, m,n,o,p,t] and XLA transposes it, see _out_layout_order for
    # when. The gradient contractions (transpose rules above) use this function
    # too; the forward-mode and vmap rules do not.
    order = _out_layout_order(idx_a, idx_b, idx_out, jnp.shape(a), jnp.shape(b), dtype)
    if order is not None:
        out = contract(a, b, idx_a, idx_b, order)
        return jnp.transpose(out, [order.index(lab) for lab in idx_out])
    return _contract_p.bind(
        jnp.asarray(a, dtype),
        jnp.asarray(b, dtype),
        idx_a=tuple(idx_a),
        idx_b=tuple(idx_b),
        idx_out=tuple(idx_out),
    )


def _pairwise(einsum_str, x, y):
    inputs, output = einsum_str.split("->")
    ix, iy = inputs.split(",")
    # Sum out labels tblis cannot handle (only in one operand or repeated).
    kept_x = "".join(dict.fromkeys(c for c in ix if c in iy or c in output))
    kept_y = "".join(dict.fromkeys(c for c in iy if c in ix or c in output))
    if kept_x != ix:
        x = jnp.einsum(f"{ix}->{kept_x}", x)
    if kept_y != iy:
        y = jnp.einsum(f"{iy}->{kept_y}", y)
    return contract(
        x,
        y,
        tuple(map(ord, kept_x)),
        tuple(map(ord, kept_y)),
        tuple(map(ord, output)),
    )


def einsum(
    subscripts: str,
    *operands: jnp.ndarray,
    optimize: Union[str, Sequence[Tuple[int, ...]]] = "optimal",
) -> jnp.ndarray:
    """
    Drop-in replacement for :func:`jax.numpy.einsum` with explicit subscripts
    which does the pairwise contractions of the path with tblis.

    Args:
      subscripts (:obj:`str`):
        The einsum network, e.g. ``"ab,bc->ac"``.
      operands (:obj:`jax.numpy.ndarray`):
        The input tensors.
    Keyword args:
      optimize (:obj:`str` or :term:`sequence` of :obj:`tuple`):
        Path optimizer of :mod:`opt_einsum` or an explicit contraction path.
    Returns:
      :obj:`jax.numpy.ndarray`:
        The contracted tensor.
    """
    dtype = jnp.result_type(*operands)
    operands = [
        x if jnp.result_type(x) == dtype else jnp.asarray(x, dtype) for x in operands
    ]
    _, contractions = opt_einsum.contract_path(
        subscripts, *operands, einsum_call=True, optimize=optimize
    )
    for inds, _, einsum_str, _, _ in contractions:
        tensors = [operands.pop(i) for i in inds]
        if len(tensors) == 2:
            operands.append(_pairwise(einsum_str, *tensors))
        else:
            operands.append(jnp.einsum(einsum_str, *tensors))
    return operands[0]


def tensordot(
    a: jnp.ndarray,
    b: jnp.ndarray,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
) -> jnp.ndarray:
    """
    Drop-in replacement for :func:`jax.numpy.tensordot` using tblis. For
    matrices ``tensordot(a, b, 1)`` is the same as ``jnp.dot(a, b)``.

    Args:
      a (:obj:`jax.numpy.ndarray`):
        First operand.
      b (:obj:`jax.numpy.ndarray`):
        Second operand.
      axes (:obj:`int` or :obj:`tuple` of two sequences of :obj:`int`):
        Number of trailing axes of ``a`` contracted with the leading axes of
        ``b``, or the two sequences of axes to contract.
    Returns:
      :obj:`jax.numpy.ndarray`:
        The contracted tensor with the free axes of ``a`` followed by the free
        axes of ``b``.
    """
    if isinstance(axes, int):
        axes = (range(a.ndim - axes, a.ndim), range(axes))
    axes_a = [i % a.ndim for i in np.atleast_1d(axes[0])]
    axes_b = [i % b.ndim for i in np.atleast_1d(axes[1])]

    idx_a = tuple(range(a.ndim))
    idx_b = list(range(a.ndim, a.ndim + b.ndim))
    for i, j in zip(axes_a, axes_b, strict=True):
        idx_b[j] = idx_a[i]
    idx_out = tuple(i for k, i in enumerate(idx_a) if k not in axes_a) + tuple(
        i for k, i in enumerate(idx_b) if k not in axes_b
    )
    return contract(a, b, idx_a, tuple(idx_b), idx_out)
