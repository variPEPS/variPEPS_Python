from . import contractions
from . import models
from . import peps
from . import typing
from . import utils

from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
