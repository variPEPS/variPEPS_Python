from dataclasses import dataclass
from os import PathLike

import numpy
import scipy.optimize as spo

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from varipeps import varipeps_config, varipeps_global_state
from varipeps.peps import PEPS_Unit_Cell
from varipeps.expectation import Expectation_Model
from varipeps.mapping import Map_To_PEPS_Model
from varipeps.ctmrg import CTMRGNotConvergedError, CTMRGGradientNotConvergedError

from .line_search import NoSuitableStepSizeError
from .optimizer import optimize_peps_network, autosave_function

from typing import List, Union, Tuple, cast, Sequence, Callable, Optional


@dataclass
class VariPEPS_Basinhopping:
    """
    Class to wrap the basinhopping algorithm for the variational update
    of PEPS or mapped structures.

    The parameters of the class initialization are the same as for
    :obj:`~varipeps.optimization.optimize_peps_network`.

    Args:
      initial_guess (:obj:`~varipeps.peps.PEPS_Unit_Cell` or :term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS unitcell to work on or the tensors which should be mapped by
        `convert_to_unitcell_func` to a PEPS unitcell.
      expectation_func (:obj:`~varipeps.expectation.Expectation_Model`):
        Callable to calculate one expectation value which is used as loss
        loss function of the model. Likely the function to calculate the energy.
      convert_to_unitcell_func (:obj:`~varipeps.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the first input parameter.
      autosave_filename (:obj:`os.PathLike`):
        Filename where intermediate results are automatically saved.
      autosave_func (:term:`callable`):
        Function which is called to autosave the intermediate results.
        The function has to accept the arguments `(filename, tensors, unitcell)`.data (:obj:`Unit_Cell_Data`):
        Instance of unit cell data class
    """

    initial_guess: Union[PEPS_Unit_Cell, Sequence[jnp.ndarray]]
    expectation_func: Expectation_Model
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None
    autosave_filename: PathLike = "data/autosave.hdf5"
    autosave_func: Callable[[PathLike, Sequence[jnp.ndarray], PEPS_Unit_Cell], None] = (
        autosave_function
    )

    def __post_init__(self):
        if isinstance(self.initial_guess, PEPS_Unit_Cell):
            initial_guess_tensors = [
                i.tensor for i in self.initial_guess.get_unique_tensors()
            ]
        else:
            initial_guess_tensors = list(self.initial_guess)

        initial_guess_flatten_tensors, self._map_pytree_func = ravel_pytree(
            initial_guess_tensors
        )

        initial_guess_tensors_numpy = numpy.asarray(initial_guess_flatten_tensors)

        if numpy.iscomplexobj(initial_guess_tensors_numpy):
            self._initial_guess_tensors_numpy = numpy.concatenate(
                (
                    numpy.real(initial_guess_tensors_numpy),
                    numpy.imag(initial_guess_tensors_numpy),
                )
            )
            self._iscomplex = True
            self._initial_guess_complex_length = initial_guess_flatten_tensors.size
        else:
            self._initial_guess_tensors_numpy = initial_guess_tensors_numpy
            self._iscomplex = False

    def _wrapper_own_optimizer(
        self,
        fun,
        x0,
        *args,
        **kwargs,
    ):
        varipeps_global_state.basinhopping_disable_half_projector = not self._first_step

        if self._iscomplex:
            x0_jax = jnp.asarray(
                x0[: self._initial_guess_complex_length]
                + 1j * x0[self._initial_guess_complex_length :]
            )
        else:
            x0_jax = jnp.asarray(x0)
        x0_jax = self._map_pytree_func(x0_jax)

        if isinstance(self.initial_guess, PEPS_Unit_Cell):
            input_obj = PEPS_Unit_Cell.from_tensor_list(
                x0_jax, self.initial_guess.data.structure
            )
        else:
            input_obj = x0_jax

        opt_result = optimize_peps_network(
            input_obj,
            self.expectation_func,
            self.convert_to_unitcell_func,
            self.autosave_filename,
            self.autosave_func,
        )

        result_tensors, _ = ravel_pytree(opt_result.x)
        result_tensors_numpy = numpy.asarray(result_tensors)
        if self._iscomplex:
            result_tensors_numpy = numpy.concatenate(
                (numpy.real(result_tensors_numpy), numpy.imag(result_tensors_numpy))
            )

        opt_result["x"] = result_tensors_numpy
        if opt_result.fun is not None:
            opt_result["fun"] = numpy.asarray(opt_result.fun)
        else:
            opt_result["fun"] = numpy.inf

        self._first_step = False

        return opt_result

    @staticmethod
    def _dummy_func(x, *args, **kwargs):
        return x

    def run(self) -> spo.OptimizeResult:
        """
        Run the basinhopping algorithm for the setup initialized in the class
        object.

        For details see :obj:`scipy.optimize.basinhopping`.

        Returns:
          :obj:`scipy.optimize.OptimizeResult`:
            Result from the basinhopping algorithm with additional fields
            ``unitcell`` and ``result_tensors`` for the result tensors and
            unitcell in the normal format of this library.
        """
        self._first_step = True

        result = spo.basinhopping(
            self._dummy_func,
            self._initial_guess_tensors_numpy,
            niter=varipeps_config.basinhopping_niter,
            T=varipeps_config.basinhopping_T,
            niter_success=varipeps_config.basinhopping_niter_success,
            disp=True,
            minimizer_kwargs={"method": self._wrapper_own_optimizer},
        )

        result["unitcell"] = result.lowest_optimization_result.unitcell

        if self._iscomplex:
            x_jax = jnp.asarray(
                result.x[: self._initial_guess_complex_length]
                + 1j * result.x[self._initial_guess_complex_length :]
            )
        else:
            x_jax = jnp.asarray(result.x)
        x_jax = self._map_pytree_func(x_jax)

        result["result_tensors"] = x_jax

        varipeps_global_state.basinhopping_disable_half_projector = None

        return result
