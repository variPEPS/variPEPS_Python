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
from . import overlap
from . import peps
from . import typing
from . import utils

from .__version__ import __version__, git_commit, git_tag

import psutil

from jax import config as jax_config

jax_cache_max_memory = -1
if (slurm_data := utils.slurm.SlurmUtils.get_own_job_data()) is not None:
    try:
        slurm_mem = slurm_data["MinMemoryNode"]
        slurm_mem_factor = 1
    except KeyError:
        try:
            slurm_mem = slurm_data["MinMemoryCPU"]
            slurm_mem_factor = slurm_data["NumCPUs"]
        except KeyError:
            slurm_mem = None
    if slurm_mem is not None:
        try:
            slurm_mem, mem_unit = int(slurm_mem[:-1]), slurm_mem[-1]
            slurm_mem *= slurm_mem_factor
            del slurm_mem_factor

            if mem_unit == "G":
                slurm_mem *= 1000 * 1000 * 1000
            elif mem_unit == "M":
                slurm_mem *= 1000 * 1000
            elif mem_unit == "T":
                slurm_mem *= 1000 * 1000 * 1000 * 1000
            else:
                del slurm_mem
                del mem_unit
                raise NameError

            del mem_unit

            jax_cache_max_memory = int(
                varipeps_config.jax_compilation_cache_memory_factor * slurm_mem
            )

            del slurm_mem
        except NameError:
            del slurm_mem
            del slurm_mem_factor
else:
    jax_cache_max_memory = int(
        varipeps_config.jax_compilation_cache_memory_factor
        * psutil.virtual_memory().total
    )

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_compilation_cache_max_size", jax_cache_max_memory)

from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime

tqdm_logging.set_log_rate(datetime.timedelta(seconds=60))

del datetime
del psutil
del tqdm_logging
del jax_config
del jax_cache_max_memory
