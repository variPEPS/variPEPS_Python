import numpy as np

from typing import Sequence, Tuple, Union


def calculate_periodic_indices(
    key: Tuple[Union[int, slice], Union[int, slice]],
    structure: Tuple[Tuple[int, ...], ...],
    real_ix: int,
    real_iy: int,
) -> Sequence[Sequence[int]]:
    """
    Get indices according to unit cell structure.

    Args:
      key (:obj:`tuple` of 2 :obj:`int` or :obj:`slice` elements):
        x and y coordinates to select. Can be either integers or slices.
        Negative numbers as selectors are supported.
      structure (2d :obj:`jax.numpy.ndarray`):
        Two dimensional array modeling the structure of the unit cell. For
        details see the description of :obj:`~varipeps.peps.PEPS_Unit_Cell`.
      real_ix (:obj:`int`):
        Which real x index of structure correspond to x = 0.
      real_iy (:obj:`int`):
        Which real y index of structure correspond to y = 0.
    Returns:
      :term:`sequence` of :term:`sequence` of :obj:`int`:
        2d sequence with the indices of the selected objects.
    """
    if not isinstance(key, tuple) or not len(key) == 2:
        raise TypeError("Expect a tuple with coordinates x and y.")

    x, y = key

    if isinstance(x, int):
        x = slice(x, x + 1, None)
    if isinstance(y, int):
        y = slice(y, y + 1, None)

    if not isinstance(x, slice) or not isinstance(y, slice):
        raise TypeError("Key elements have to be integers or slices.")

    unit_cell_len_x = len(structure)
    unit_cell_len_y = len(structure[0])

    if x.start is not None and x.start < 0:
        shift = (-x.start // unit_cell_len_x + 1) * unit_cell_len_x
        x = slice(x.start + shift, x.stop + shift, x.step)

    if x.start is None and x.stop < 0:
        x = slice(None, x.stop + unit_cell_len_x, x.step)

    if y.start is not None and y.start < 0:
        shift = (-y.start // unit_cell_len_y + 1) * unit_cell_len_y
        y = slice(y.start + shift, y.stop + shift, y.step)

    if y.start is None and y.stop < 0:
        y = slice(None, y.stop + unit_cell_len_y, y.step)

    xarr = np.arange(
        x.start or 0, x.stop if x.stop is not None else unit_cell_len_x, x.step
    )
    yarr = np.arange(
        y.start or 0, y.stop if y.stop is not None else unit_cell_len_y, y.step
    )

    xarr = (real_ix + xarr) % unit_cell_len_x
    yarr = (real_iy + yarr) % unit_cell_len_y

    return tuple(tuple(structure[xi][yi] for yi in yarr) for xi in xarr)
