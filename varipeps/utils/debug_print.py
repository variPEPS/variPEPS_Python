import functools
import logging


import jax.debug as jdebug
from jax._src.debugging import formatter

logger = logging.getLogger("varipeps.ctmrg")


def _format_print_callback(fmt: str, *args, **kwargs):
    # Send to logger (respects per-module levels/handlers)
    logger.debug(fmt.format(*args, **kwargs))


def debug_print(fmt: str, *args, ordered: bool = True, **kwargs) -> None:
    """
    Prints values and works in staged out JAX functions.

    Function adapted from :obj:`jax.debug.print` to work with logger. See there
    for original authors and function.

    Args:
      fmt (:obj:`str`):
        A format string, e.g. ``"hello {x}"``, that will be used to format
        input arguments.
      *args: A list of positional arguments to be formatted.

    Keyword args:
      ordered (:obj:`bool`): :
        A keyword only argument used to indicate whether or not the
        staged out computation will enforce ordering of this ``debug_print``
        w.r.t. other ordered ``debug_print`` calls.
      **kwargs:
        Additional keyword arguments to be formatted.
    """
    # Check that we provide the correct arguments to be formatted
    formatter.format(fmt, *args, **kwargs)

    jdebug.callback(
        functools.partial(_format_print_callback, fmt), *args, **kwargs, ordered=ordered
    )
