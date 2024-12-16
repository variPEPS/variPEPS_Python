# First import config so it is usable in other parts of the module
from .config import wrapper as config
from .config import config as varipeps_config
from .global_state import global_state as varipeps_global_state

from . import contractions
from . import corrlength
from . import ctmrg
from . import expectation
from . import mapping
from . import optimization
from . import peps
from . import typing
from . import utils

from .__version__ import __version__, git_commit, git_tag

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime

tqdm_logging.set_log_rate(datetime.timedelta(seconds=60))

del datetime
del tqdm_logging
del jax_config
