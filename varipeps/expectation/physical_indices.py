from typing import Sequence, Tuple


def build_physical_contraction_indices(
    site_open_physical_indices: Sequence[int],
    num_coarse_grained_physical_indices: int,
    start_contraction_at: int,
    negative_offset: int = 0,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build the contraction index tuples `(tensor, tensor_conj)` for the physical
    indices of one coarse-grained physical site of a partially-traced density
    matrix. Traced ket-bra physical sites are assigned a matching positive
    dummy label. Open or dangling links are assigned a decreasing negative
    output label in the order provided in ``site_open_physical_indices``,
    which need not be ascending a priori.

    Example:

    .. code-block:: python

        site_open_physical_indices = (4, 1)
        num_coarse_grained_physical_indices = 4
        start_contraction_at = 5
        negative_offset = -3

    gives the contraction tuples

    .. code-block:: python

        ((-5, 5, 6, -4), (-7, 5, 6, -6))

    Args:
      site_open_physical_indices (:term:`sequence` of :obj:`int`):
        Physical site indices (1-based) that remain open (untraced) for this
        specific tensor. Must be duplicate-free and within
        ``[1, num_coarse_grained_physical_indices]``.
      num_coarse_grained_physical_indices (:obj:`int`):
        The total number of physical sites within the coarse-grained tensor.
      start_contraction_at (:obj:`int`):
        The first positive label used for the dummy tracing indices. The
        surrounding network must already use exactly ``1..K``, so pass
        ``K + 1``.
      negative_offset (:obj:`int`, optional):
        Offset to be considered before we place the first open physical leg.
        Must be non-positive, and the default ``0`` means the first open index
        sits at ``-1``. A negative value of ``-M``, would sit at ``-M-1``.

    Returns:
      :obj:`tuple` of two :obj:`tuple` of :obj:`int`:
        The contraction indices for the ket tensor and the bra tensor.
    """
    if any(
        not isinstance(site_id, int) or isinstance(site_id, bool)
        for site_id in site_open_physical_indices
    ):
        raise TypeError(
            f"`site_open_physical_indices` must contain plain ints, got "
            f"{site_open_physical_indices!r}"
        )
    if negative_offset > 0:
        raise ValueError(
            f"`negative_offset` must be non-positive, got {negative_offset}"
        )

    n_open = len(site_open_physical_indices)
    if n_open > num_coarse_grained_physical_indices:
        raise ValueError(
            f"`site_open_physical_indices` {site_open_physical_indices!r} holds more "
            f"entries than the {num_coarse_grained_physical_indices} coarse-grained "
            f"physical sites available"
        )
    if len(set(site_open_physical_indices)) != n_open:
        raise ValueError(
            f"`site_open_physical_indices` {site_open_physical_indices!r} contains "
            f"duplicates; each physical site can be opened at most once"
        )

    out_of_range = [
        site_id
        for site_id in site_open_physical_indices
        if not 1 <= site_id <= num_coarse_grained_physical_indices
    ]
    if out_of_range:
        raise ValueError(
            f"`site_open_physical_indices` {site_open_physical_indices!r} names sites "
            f"{out_of_range} outside the 1..{num_coarse_grained_physical_indices} range "
            f"of coarse-grained physical sites"
        )

    position_map = {
        site_id: pos for pos, site_id in enumerate(site_open_physical_indices)
    }
    tensor = []
    tensor_conj = []
    traced = start_contraction_at

    for phys in range(1, num_coarse_grained_physical_indices + 1):
        if phys in position_map:
            pos = position_map[phys]
            tensor.append(negative_offset - (pos + 1))
            tensor_conj.append(negative_offset - n_open - (pos + 1))
        else:
            tensor.append(traced)
            tensor_conj.append(traced)
            traced += 1

    return tuple(tensor), tuple(tensor_conj)
