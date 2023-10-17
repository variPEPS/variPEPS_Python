from collections import deque
from functools import partial
from os import PathLike

from scipy.optimize import OptimizeResult

from tqdm_loggable.auto import tqdm

from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from jax.flatten_util import ravel_pytree

from peps_ad import peps_ad_config, peps_ad_global_state
from peps_ad.config import Optimizing_Methods
from peps_ad.peps import PEPS_Unit_Cell
from peps_ad.expectation import Expectation_Model
from peps_ad.config import Projector_Method
from peps_ad.mapping import Map_To_PEPS_Model
from peps_ad.ctmrg import CTMRGNotConvergedError, CTMRGGradientNotConvergedError
from peps_ad.utils.random import PEPS_Random_Number_Generator

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)
from .line_search import line_search, NoSuitableStepSizeError, _scalar_descent_grad

from typing import List, Union, Tuple, cast, Sequence, Callable, Optional


@jit
def _cg_workhorse(new_gradient, old_gradient, old_descent_dir):
    new_grad_vec, new_grad_unravel = ravel_pytree(new_gradient)
    old_grad_vec, old_grad_unravel = ravel_pytree(old_gradient)
    old_des_dir_vec, old_des_dir_unravel = ravel_pytree(old_descent_dir)

    new_grad_len = new_grad_vec.size
    iscomplex = jnp.iscomplexobj(new_grad_vec)

    if iscomplex:
        new_grad_vec = jnp.concatenate((jnp.real(new_grad_vec), jnp.imag(new_grad_vec)))
        old_grad_vec = jnp.concatenate((jnp.real(old_grad_vec), jnp.imag(old_grad_vec)))
        old_des_dir_vec = jnp.concatenate(
            (jnp.real(old_des_dir_vec), jnp.imag(old_des_dir_vec))
        )

    grad_diff = new_grad_vec - old_grad_vec

    # dx = -new_grad_vec
    # dx_old = -old_grad_vec
    # dx_real = jnp.concatenate((jnp.real(dx), jnp.imag(dx)))
    # dx_old_real = jnp.concatenate((jnp.real(dx_old), jnp.imag(dx_old)))
    # old_des_dir_real = jnp.concatenate(
    #     (jnp.real(old_descent_dir), jnp.imag(old_descent_dir))
    # )
    # PRP
    # beta = jnp.sum(dx_real * (dx_real - dx_old_real)) / jnp.sum(dx_old_real * dx_old_real)
    # LS parameter
    # beta = jnp.sum(dx_real * (dx_old_real - dx_real)) / jnp.sum(
    #     old_des_dir_real * dx_old_real
    # )

    # Hager-Zhang
    eta = 0.4
    eta_k = -1 / (
        jnp.linalg.norm(old_des_dir_vec) * jnp.fmin(eta, jnp.linalg.norm(old_grad_vec))
    )
    old_des_grad_diff = jnp.dot(old_des_dir_vec, grad_diff)
    beta = (
        grad_diff - 2 * jnp.linalg.norm(grad_diff) * old_des_dir_vec / old_des_grad_diff
    )
    beta = jnp.dot(beta, new_grad_vec) / old_des_grad_diff
    beta = jnp.fmax(eta_k, beta)

    beta = jnp.fmax(0, beta)

    result = -new_grad_vec + beta * old_des_dir_vec

    if iscomplex:
        result = result[:new_grad_len] + 1j * result[new_grad_len:]

    return new_grad_unravel(result), beta


@partial(jit, static_argnums=(5,))
def _bfgs_workhorse(
    new_gradient, old_gradient, old_descent_dir, old_alpha, B_inv, calc_new_B_inv
):
    new_grad_vec, new_grad_unravel = ravel_pytree(new_gradient)
    new_grad_len = new_grad_vec.size

    iscomplex = jnp.iscomplexobj(new_grad_vec)
    if iscomplex:
        new_grad_vec = jnp.concatenate((jnp.real(new_grad_vec), jnp.imag(new_grad_vec)))

    if calc_new_B_inv:
        old_grad_vec, old_grad_unravel = ravel_pytree(old_gradient)
        old_descent_dir_vec, old_descent_dir_unravel = ravel_pytree(old_descent_dir)
        if iscomplex:
            old_grad_vec = jnp.concatenate(
                (jnp.real(old_grad_vec), jnp.imag(old_grad_vec))
            )
            old_descent_dir_vec = jnp.concatenate(
                (jnp.real(old_descent_dir_vec), jnp.imag(old_descent_dir_vec))
            )

        sk = old_alpha * old_descent_dir_vec
        yk = new_grad_vec - old_grad_vec

        skyk_scalar = jnp.dot(sk, yk)
        B_inv_yk = jnp.dot(B_inv, yk)

        new_B_inv = (
            B_inv
            + ((skyk_scalar + jnp.dot(yk, B_inv_yk)) / (skyk_scalar**2))
            * jnp.outer(sk, sk)
            - (jnp.outer(B_inv_yk, sk) + jnp.outer(sk, B_inv_yk)) / skyk_scalar
        )
    else:
        new_B_inv = B_inv

    result = -jnp.dot(new_B_inv, new_grad_vec)

    if iscomplex:
        result = result[:new_grad_len] + 1j * result[new_grad_len:]

    return new_grad_unravel(result), new_B_inv


@jit
def _l_bfgs_workhorse(value_tuple, gradient_tuple):
    gradient_elem_0, gradient_unravel = ravel_pytree(gradient_tuple[0])
    gradient_len = gradient_elem_0.size

    iscomplex = jnp.iscomplexobj(gradient_elem_0)

    def _make_1d(x):
        x_1d, _ = ravel_pytree(x)
        if iscomplex:
            return jnp.concatenate((jnp.real(x_1d), jnp.imag(x_1d)))
        return x_1d

    value_arr = jnp.asarray([_make_1d(e) for e in value_tuple])
    gradient_arr = jnp.asarray([_make_1d(e) for e in gradient_tuple])

    s_arr = -jnp.diff(value_arr, axis=0)
    y_arr = -jnp.diff(gradient_arr, axis=0)
    pho_arr = 1 / jnp.sum(y_arr * s_arr, axis=1)

    def first_loop(q, x):
        pho_s, y = x
        alpha_i = jnp.sum(pho_s * q)
        return q - alpha_i * y, alpha_i

    q, alpha_arr = scan(
        first_loop,
        gradient_arr[0],
        (pho_arr[:, jnp.newaxis] * s_arr, y_arr),
    )

    gamma = jnp.sum(s_arr[0] * y_arr[0]) / jnp.sum(y_arr[0] * y_arr[0])

    z_result = gamma * q

    def second_loop(z, x):
        pho_y, s, alpha_i = x
        beta_i = jnp.sum(pho_y * z)
        return z + s * (alpha_i - beta_i), None

    z_result, _ = scan(
        second_loop,
        z_result,
        (pho_arr[:, jnp.newaxis] * y_arr, s_arr, alpha_arr),
    )

    z_result = -z_result
    if iscomplex:
        z_result = z_result[:gradient_len] + 1j * z_result[gradient_len:]
    return gradient_unravel(z_result)


def autosave_function(
    filename: PathLike,
    tensors: jnp.ndarray,
    unitcell: PEPS_Unit_Cell,
    counter: Optional[int] = None,
    max_trunc_error_list: Optional[float] = None,
) -> None:
    if counter is not None:
        unitcell.save_to_file(
            f"{str(filename)}.{counter}", max_trunc_error_list=max_trunc_error_list
        )
    else:
        unitcell.save_to_file(filename, max_trunc_error_list=max_trunc_error_list)


def optimize_peps_network(
    input_tensors: Union[PEPS_Unit_Cell, Sequence[jnp.ndarray]],
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None,
    autosave_filename: PathLike = "data/autosave.hdf5",
    autosave_func: Callable[
        [PathLike, Sequence[jnp.ndarray], PEPS_Unit_Cell], None
    ] = autosave_function,
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell, Union[float, jnp.ndarray]]:
    """
    Optimize a PEPS unitcell using a variational method.

    As convergence criterion the norm of the gradient is used.

    Args:
      input_tensors (:obj:`~peps_ad.peps.PEPS_Unit_Cell` or :term:`sequence` of :obj:`jax.numpy.ndarray`):
        The PEPS unitcell to work on or the tensors which should be mapped by
        `convert_to_unitcell_func` to a PEPS unitcell.
      expectation_func (:obj:`~peps_ad.expectation.Expectation_Model`):
        Callable to calculate one expectation value which is used as loss
        loss function of the model. Likely the function to calculate the energy.
      convert_to_unitcell_func (:obj:`~peps_ad.mapping.Map_To_PEPS_Model`):
        Function to convert the `input_tensors` to a PEPS unitcell. If ommited,
        it is assumed that a PEPS unitcell is the first input parameter.
      autosave_filename (:obj:`os.PathLike`):
        Filename where intermediate results are automatically saved.
      autosave_func (:term:`callable`):
        Function which is called to autosave the intermediate results.
        The function has to accept the arguments `(filename, tensors, unitcell)`.
    Returns:
      :obj:`scipy.optimize.OptimizeResult`:
        OptimizeResult object with the optimized tensors, network and the
        final expectation value. See the type definition for other possible
        fields.
    """
    rng = PEPS_Random_Number_Generator.get_generator(backend="jax")

    def random_noise(a):
        return a + a * rng.block(a.shape, dtype=a.dtype) * 1e-3

    if isinstance(input_tensors, PEPS_Unit_Cell):
        working_tensors = cast(
            List[jnp.ndarray], [i.tensor for i in input_tensors.get_unique_tensors()]
        )
        working_unitcell = input_tensors
    else:
        working_tensors = input_tensors
        working_unitcell = None

    old_gradient = None
    old_descent_dir = None
    descent_dir = None
    working_value = None

    max_trunc_error = jnp.nan

    best_value = jnp.inf
    best_tensors = None
    best_unitcell = None

    random_noise_retries = 0

    signal_reset_descent_dir = False

    if peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
        bfgs_prefactor = 2 if any(jnp.iscomplexobj(t) for t in working_tensors) else 1
        bfgs_B_inv = jnp.eye(bfgs_prefactor * sum([t.size for t in working_tensors]))
    elif peps_ad_config.optimizer_method is Optimizing_Methods.L_BFGS:
        l_bfgs_x_cache = deque(maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1)
        l_bfgs_grad_cache = deque(maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1)

    count = 0
    linesearch_step: Union[
        float, jnp.ndarray
    ] = peps_ad_config.line_search_initial_step_size
    working_value: Union[float, jnp.ndarray]
    max_trunc_error_list = []

    if (
        peps_ad_config.optimizer_preconverge_with_half_projectors
        and not peps_ad_global_state.basinhopping_disable_half_projector
    ):
        peps_ad_global_state.ctmrg_projector_method = Projector_Method.HALF
    else:
        peps_ad_global_state.ctmrg_projector_method = None

    with tqdm(desc="Optimizing PEPS state") as pbar:
        while count < peps_ad_config.optimizer_max_steps:
            try:
                if peps_ad_config.ad_use_custom_vjp:
                    (
                        working_value,
                        (working_unitcell, _),
                    ), working_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                        working_tensors,
                        working_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                    )
                else:
                    (
                        working_value,
                        (working_unitcell, _),
                    ), working_gradient_seq = calc_preconverged_ctmrg_value_and_grad(
                        working_tensors,
                        working_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        calc_preconverged=(count == 0),
                    )
            except (CTMRGNotConvergedError, CTMRGGradientNotConvergedError) as e:
                peps_ad_global_state.ctmrg_projector_method = None

                return OptimizeResult(
                    success=False,
                    message=str(type(e)),
                    x=working_tensors,
                    fun=working_value,
                    unitcell=working_unitcell,
                    nit=count,
                    max_trunc_error_list=max_trunc_error_list,
                )

            working_gradient = [elem.conj() for elem in working_gradient_seq]

            if signal_reset_descent_dir:
                if peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
                    bfgs_prefactor = (
                        2 if any(jnp.iscomplexobj(t) for t in working_tensors) else 1
                    )
                    bfgs_B_inv = jnp.eye(
                        bfgs_prefactor * sum([t.size for t in working_tensors])
                    )
                elif peps_ad_config.optimizer_method is Optimizing_Methods.L_BFGS:
                    l_bfgs_x_cache = deque(
                        maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1
                    )
                    l_bfgs_grad_cache = deque(
                        maxlen=peps_ad_config.optimizer_l_bfgs_maxlen + 1
                    )

            if peps_ad_config.optimizer_method is Optimizing_Methods.STEEPEST:
                descent_dir = [-elem for elem in working_gradient]
            elif peps_ad_config.optimizer_method is Optimizing_Methods.CG:
                if count == 0 or signal_reset_descent_dir:
                    descent_dir = [-elem for elem in working_gradient]
                else:
                    descent_dir, beta = _cg_workhorse(
                        working_gradient, old_gradient, old_descent_dir
                    )
            elif peps_ad_config.optimizer_method is Optimizing_Methods.BFGS:
                if count == 0 or signal_reset_descent_dir:
                    descent_dir, _ = _bfgs_workhorse(
                        working_gradient, None, None, None, bfgs_B_inv, False
                    )
                else:
                    descent_dir, bfgs_B_inv = _bfgs_workhorse(
                        working_gradient,
                        old_gradient,
                        old_descent_dir,
                        linesearch_step,
                        bfgs_B_inv,
                        True,
                    )
            elif peps_ad_config.optimizer_method is Optimizing_Methods.L_BFGS:
                l_bfgs_x_cache.appendleft(tuple(working_tensors))
                l_bfgs_grad_cache.appendleft(tuple(working_gradient))

                if count == 0 or signal_reset_descent_dir:
                    descent_dir = [-elem for elem in working_gradient]
                else:
                    descent_dir = _l_bfgs_workhorse(
                        tuple(l_bfgs_x_cache), tuple(l_bfgs_grad_cache)
                    )
            else:
                raise ValueError("Unknown optimization method.")

            signal_reset_descent_dir = False

            if _scalar_descent_grad(descent_dir, working_gradient) > 0:
                tqdm.write("Found bad descent dir. Reset to negative gradient!")
                descent_dir = [-elem for elem in working_gradient]

            conv = jnp.linalg.norm(ravel_pytree(working_gradient)[0])

            try:
                (
                    working_tensors,
                    working_unitcell,
                    working_value,
                    linesearch_step,
                    signal_reset_descent_dir,
                    max_trunc_error,
                ) = line_search(
                    working_tensors,
                    working_unitcell,
                    expectation_func,
                    working_gradient,
                    descent_dir,
                    working_value,
                    linesearch_step,
                    convert_to_unitcell_func,
                )
            except NoSuitableStepSizeError:
                if peps_ad_config.optimizer_fail_if_no_step_size_found:
                    raise
                else:
                    if (
                        (
                            conv > peps_ad_config.optimizer_random_noise_eps
                            or working_value > best_value
                        )
                        and random_noise_retries
                        < peps_ad_config.optimizer_random_noise_max_retries
                    ):
                        tqdm.write(
                            "Convergence is not sufficient. Retry with some random noise on best result."
                        )

                        if working_value < best_value:
                            best_value = working_value
                            best_tensors = working_tensors
                            best_unitcell = working_unitcell

                        if isinstance(input_tensors, PEPS_Unit_Cell):
                            working_tensors = cast(
                                List[jnp.ndarray],
                                [i.tensor for i in best_unitcell.get_unique_tensors()],
                            )

                            working_tensors = [random_noise(i) for i in working_tensors]

                            working_tensors_obj = [
                                e.replace_tensor(working_tensors[i])
                                for i, e in enumerate(
                                    best_unitcell.get_unique_tensors()
                                )
                            ]

                            working_unitcell = best_unitcell.replace_unique_tensors(
                                working_tensors_obj
                            )
                        else:
                            working_tensors = [random_noise(i) for i in best_tensors]
                            working_unitcell = None

                        descent_dir = None
                        working_gradient = None
                        signal_reset_descent_dir = True
                        count = -1
                        random_noise_retries += 1
                        pbar.reset()
                        pbar.refresh()
                    else:
                        conv = 0

            max_trunc_error_list.append(max_trunc_error)

            if conv < peps_ad_config.optimizer_convergence_eps:
                working_value, (
                    working_unitcell,
                    max_trunc_error,
                ) = calc_ctmrg_expectation(
                    working_tensors,
                    working_unitcell,
                    expectation_func,
                    convert_to_unitcell_func,
                    enforce_elementwise_convergence=peps_ad_config.ad_use_custom_vjp,
                )
                peps_ad_global_state.ctmrg_projector_method = None
                max_trunc_error_list[-1] = max_trunc_error
                break

            if (
                peps_ad_config.optimizer_preconverge_with_half_projectors
                and not peps_ad_global_state.basinhopping_disable_half_projector
                and conv < peps_ad_config.optimizer_preconverge_with_half_projectors_eps
            ):
                peps_ad_global_state.ctmrg_projector_method = (
                    peps_ad_config.ctmrg_full_projector_method
                )

                working_value, working_unitcell, _ = calc_ctmrg_expectation(
                    working_tensors,
                    working_unitcell,
                    expectation_func,
                    convert_to_unitcell_func,
                    enforce_elementwise_convergence=peps_ad_config.ad_use_custom_vjp,
                )
                descent_dir = None
                working_gradient = None
                signal_reset_descent_dir = True

            old_descent_dir = descent_dir
            old_gradient = working_gradient

            count += 1

            pbar.update()
            pbar.set_postfix(
                {
                    "Energy": f"{working_value:0.10f}",
                    "Retries": random_noise_retries,
                    "Convergence": f"{conv:0.8f}",
                    "Line search step": f"{linesearch_step:0.8f}",
                    "Max. trunc. err.": f"{max_trunc_error:0.8g}",
                }
            )
            pbar.refresh()

            if count % peps_ad_config.optimizer_autosave_step_count == 0:
                autosave_func(
                    autosave_filename,
                    working_tensors,
                    working_unitcell,
                    counter=random_noise_retries,
                    max_trunc_error_list=max_trunc_error_list,
                )

    if working_value < best_value:
        best_value = working_value
        best_tensors = working_tensors
        best_unitcell = working_unitcell

    print(f"Best energy result found: {best_value}")

    return OptimizeResult(
        success=True,
        x=best_tensors,
        fun=best_value,
        unitcell=best_unitcell,
        nit=count,
        max_trunc_error_list=max_trunc_error_list,
    )
