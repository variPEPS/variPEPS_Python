# First import config so it is usable in other parts of the module
from . import config
from .config import config as peps_ad_config
from .global_state import global_state as peps_ad_global_state

from . import contractions
from . import ctmrg
from . import expectation
from . import mapping
from . import optimization
from . import peps
from . import typing
from . import utils

from .__version__ import __version__

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime

tqdm_logging.set_log_rate(datetime.timedelta(seconds=60))

del datetime
del tqdm_logging
