"""
Typing helper for the module
"""

from __future__ import annotations

from typing import Union, Any

import numpy as np
import jax
import jax.numpy as jnp

Tensor = Union[np.ndarray, jax.numpy.ndarray]  #: Typing object for a numpy or jax tensor


def is_tensor(a: Any) -> bool:
    """
    Test if object is a numpy or jax tensor

    Args:
      a: Object to be tested
    Returns:
      bool:
        True if object is a numpy or jax.ndarray. False otherwise.
    """
    return isinstance(a, (np.ndarray, jnp.ndarray))
