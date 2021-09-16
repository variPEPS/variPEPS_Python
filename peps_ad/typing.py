"""
Typing helper for the module
"""

from typing import Union

import numpy as np
import jax.numpy as jnp

Tensor = Union[np.ndarray, jnp.ndarray]


def is_tensor(a: Tensor) -> bool:
    return isinstance(a, (np.ndarray, jnp.ndarray))
