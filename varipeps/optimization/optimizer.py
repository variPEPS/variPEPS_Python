import collections
from collections import deque
from functools import partial
from os import PathLike
import time

from scipy.optimize import OptimizeResult

from tqdm_loggable.auto import tqdm

from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from jax.flatten_util import ravel_pytree
from jax.util import safe_zip

from varipeps import varipeps_config, varipeps_global_state
from varipeps.config import Optimizing_Methods
from varipeps.peps import PEPS_Unit_Cell
from varipeps.expectation import Expectation_Model
from varipeps.config import Projector_Method
from varipeps.mapping import Map_To_PEPS_Model
from varipeps.ctmrg import CTMRGNotConvergedError, CTMRGGradientNotConvergedError
from varipeps.utils.random import PEPS_Random_Number_Generator

from .inner_function import (
    calc_ctmrg_expectation,
    calc_preconverged_ctmrg_value_and_grad,
    calc_ctmrg_expectation_custom_value_and_grad,
)
from .line_search import line_search, NoSuitableStepSizeError, _scalar_descent_grad

from typing import List, Union, Tuple, cast, Sequence, Callable, Optional, Dict, Any


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
    counter: Optional[Union[int, str]] = None,
    auxiliary_data: Optional[Dict[str, Any]] = None,
) -> None:
    if counter is not None:
        unitcell.save_to_file(
            f"{str(filename)}.{counter}", auxiliary_data=auxiliary_data
        )
    else:
        unitcell.save_to_file(filename, auxiliary_data=auxiliary_data)


def _autosave_wrapper(
    autosave_func,
    autosave_filename,
    working_tensors,
    working_unitcell,
    working_value,
    counter,
    best_run,
    max_trunc_error_list,
    step_energies,
    step_chi,
    step_conv,
    step_runtime,
    spiral_indices,
    additional_input,
):
    auxiliary_data = {
        "best_run": jnp.array(best_run if best_run is not None else 0),
        "current_energy": working_value,
    }

    for k in sorted(max_trunc_error_list.keys()):
        auxiliary_data[f"max_trunc_error_list_{k:d}"] = max_trunc_error_list[k]
        auxiliary_data[f"step_energies_{k:d}"] = step_energies[k]
        auxiliary_data[f"step_chi_{k:d}"] = step_chi[k]
        auxiliary_data[f"step_conv_{k:d}"] = step_conv[k]
        auxiliary_data[f"step_runtime_{k:d}"] = step_runtime[k]

    spiral_vectors = None
    if spiral_indices is not None:
        spiral_vectors = [working_tensors[spiral_i] for spiral_i in spiral_indices]

        if any(i.size == 1 for i in spiral_vectors):
            spiral_vectors_x = additional_input.get("spiral_vectors_x")
            spiral_vectors_y = additional_input.get("spiral_vectors_y")
            if spiral_vectors_x is not None:
                if isinstance(spiral_vectors_x, jnp.ndarray):
                    spiral_vectors_x = (spiral_vectors_x,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors_x, spiral_vectors)
                )
            elif spiral_vectors_y is not None:
                if isinstance(spiral_vectors_y, jnp.ndarray):
                    spiral_vectors_y = (spiral_vectors_y,)
                spiral_vectors = tuple(
                    jnp.array((sx, sy))
                    for sx, sy in safe_zip(spiral_vectors, spiral_vectors_y)
                )
    elif additional_input.get("spiral_vectors") is not None:
        spiral_vectors = additional_input.get("spiral_vectors")
        if isinstance(spiral_vectors, jnp.ndarray):
            spiral_vectors = (spiral_vectors,)

    if spiral_vectors is not None:
        spiral_vectors = [
            e if e.size == 2 else jnp.array((e, e)).reshape(2) for e in spiral_vectors
        ]

        if len(spiral_vectors) == 1:
            auxiliary_data["spiral_vector"] = spiral_vectors[0]
        else:
            for spiral_i, vec in enumerate(spiral_vectors):
                spiral_i += 1
                auxiliary_data[f"spiral_vector_{spiral_i:d}"] = vec

    autosave_func(
        autosave_filename,
        working_tensors,
        working_unitcell,
        counter=counter,
        auxiliary_data=auxiliary_data,
    )


def optimize_peps_network(
    input_tensors: Union[PEPS_Unit_Cell, Sequence[jnp.ndarray]],
    expectation_func: Expectation_Model,
    convert_to_unitcell_func: Optional[Map_To_PEPS_Model] = None,
    autosave_filename: PathLike = "data/autosave.hdf5",
    autosave_func: Callable[
        [PathLike, Sequence[jnp.ndarray], PEPS_Unit_Cell], None
    ] = autosave_function,
    additional_input: Dict[str, jnp.ndarray] = {},
) -> Tuple[Sequence[jnp.ndarray], PEPS_Unit_Cell, Union[float, jnp.ndarray]]:
    """
    Optimize a PEPS unitcell using a variational method.

    As convergence criterion the norm of the gradient is used.

    Args:
      input_tensors (:obj:`~varipeps.peps.PEPS_Unit_Cell` or :term:`sequence` of :obj:`jax.numpy.ndarray`):
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
        The function has to accept the arguments `(filename, tensors, unitcell)`.
      additional_input (:obj:`dict` of :obj:`str` to :obj:`jax.numpy.ndarray` mapping):
        Dict with additional inputs which should be considered in the
        calculation of the expectation value.
    Returns:
      :obj:`scipy.optimize.OptimizeResult`:
        OptimizeResult object with the optimized tensors, network and the
        final expectation value. See the type definition for other possible
        fields.
    """
    rng = PEPS_Random_Number_Generator.get_generator(backend="jax")

    def random_noise(a):
        return (
            a
            + a
            * rng.block(a.shape, dtype=a.dtype)
            * varipeps_config.optimizer_random_noise_relative_amplitude
        )

    if isinstance(input_tensors, PEPS_Unit_Cell):
        working_tensors = cast(
            List[jnp.ndarray], [i.tensor for i in input_tensors.get_unique_tensors()]
        )
        working_unitcell = input_tensors
        generate_unitcell = True
    else:
        if isinstance(input_tensors, collections.abc.Sequence) and isinstance(
            input_tensors[0], PEPS_Unit_Cell
        ):
            if len(input_tensors[0].get_unique_tensors()) != 1:
                raise ValueError(
                    "You want to use spiral PEPS but you use a unit cell with more than one site. Seems wrong to me!"
                )
            working_tensors = cast(
                List[jnp.ndarray],
                [i.tensor for i in input_tensors[0].get_unique_tensors()],
            ) + list(input_tensors[1:])
            working_unitcell = input_tensors[0]
            generate_unitcell = True
        else:
            working_tensors = input_tensors
            working_unitcell = None
            generate_unitcell = False

    old_gradient = None
    old_descent_dir = None
    descent_dir = None
    working_value = None

    max_trunc_error = jnp.nan

    best_value = jnp.inf
    best_tensors = None
    best_unitcell = None
    best_run = None

    random_noise_retries = 0

    signal_reset_descent_dir = False

    spiral_indices = None
    if (
        hasattr(expectation_func, "is_spiral_peps")
        and expectation_func.is_spiral_peps
        and additional_input.get("spiral_vectors") is None
    ):
        if isinstance(input_tensors, collections.abc.Sequence) and isinstance(
            input_tensors[0], PEPS_Unit_Cell
        ):
            spiral_indices = list(range(1, len(input_tensors)))
        else:
            raise NotImplementedError("Only support spiral PEPS for unitcell input yet")

    if varipeps_config.optimizer_method is Optimizing_Methods.BFGS:
        bfgs_prefactor = 2 if any(jnp.iscomplexobj(t) for t in working_tensors) else 1
        bfgs_B_inv = jnp.eye(bfgs_prefactor * sum([t.size for t in working_tensors]))
    elif varipeps_config.optimizer_method is Optimizing_Methods.L_BFGS:
        l_bfgs_x_cache = deque(maxlen=varipeps_config.optimizer_l_bfgs_maxlen + 1)
        l_bfgs_grad_cache = deque(maxlen=varipeps_config.optimizer_l_bfgs_maxlen + 1)

    count = 0
    linesearch_step: Optional[Union[float, jnp.ndarray]] = None
    working_value: Union[float, jnp.ndarray]
    max_trunc_error_list = {random_noise_retries: []}
    step_energies = {random_noise_retries: []}
    step_chi = {random_noise_retries: []}
    step_conv = {random_noise_retries: []}
    step_runtime = {random_noise_retries: []}

    if (
        varipeps_config.optimizer_preconverge_with_half_projectors
        and not varipeps_global_state.basinhopping_disable_half_projector
    ):
        varipeps_global_state.ctmrg_projector_method = Projector_Method.HALF
    else:
        varipeps_global_state.ctmrg_projector_method = None

    with tqdm(desc="Optimizing PEPS state") as pbar:
        while count < varipeps_config.optimizer_max_steps:
            runtime_start = time.perf_counter()

            try:
                if varipeps_config.ad_use_custom_vjp:
                    (
                        working_value,
                        (working_unitcell, _),
                    ), working_gradient_seq = calc_ctmrg_expectation_custom_value_and_grad(
                        working_tensors,
                        working_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        additional_input,
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
                        additional_input,
                        calc_preconverged=(count == 0),
                    )
            except (CTMRGNotConvergedError, CTMRGGradientNotConvergedError) as e:
                varipeps_global_state.ctmrg_projector_method = None

                if random_noise_retries == 0:
                    return OptimizeResult(
                        success=False,
                        message=str(type(e)),
                        x=working_tensors,
                        fun=working_value,
                        unitcell=working_unitcell,
                        nit=count,
                        max_trunc_error_list=max_trunc_error_list,
                        step_energies=step_energies,
                        step_chi=step_chi,
                        step_conv=step_conv,
                        step_runtime=step_runtime,
                        best_run=0,
                    )
                elif (
                    random_noise_retries
                    >= varipeps_config.optimizer_random_noise_max_retries
                ):
                    working_value = jnp.inf
                    break
                else:
                    if isinstance(input_tensors, PEPS_Unit_Cell) or (
                        isinstance(input_tensors, collections.abc.Sequence)
                        and isinstance(input_tensors[0], PEPS_Unit_Cell)
                    ):
                        working_tensors = (
                            cast(
                                List[jnp.ndarray],
                                [i.tensor for i in best_unitcell.get_unique_tensors()],
                            )
                            + best_tensors[best_unitcell.get_len_unique_tensors() :]
                        )

                        working_tensors = [random_noise(i) for i in working_tensors]

                        working_tensors_obj = [
                            e.replace_tensor(working_tensors[i])
                            for i, e in enumerate(best_unitcell.get_unique_tensors())
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
                    count = 0
                    random_noise_retries += 1
                    old_descent_dir = descent_dir
                    old_gradient = working_gradient

                    step_energies[random_noise_retries] = []
                    step_chi[random_noise_retries] = []
                    step_conv[random_noise_retries] = []
                    max_trunc_error_list[random_noise_retries] = []

                    pbar.reset()
                    pbar.refresh()

                    continue

            working_gradient = [elem.conj() for elem in working_gradient_seq]

            if signal_reset_descent_dir:
                if varipeps_config.optimizer_method is Optimizing_Methods.BFGS:
                    bfgs_prefactor = (
                        2 if any(jnp.iscomplexobj(t) for t in working_tensors) else 1
                    )
                    bfgs_B_inv = jnp.eye(
                        bfgs_prefactor * sum([t.size for t in working_tensors])
                    )
                elif varipeps_config.optimizer_method is Optimizing_Methods.L_BFGS:
                    l_bfgs_x_cache = deque(
                        maxlen=varipeps_config.optimizer_l_bfgs_maxlen + 1
                    )
                    l_bfgs_grad_cache = deque(
                        maxlen=varipeps_config.optimizer_l_bfgs_maxlen + 1
                    )

            if varipeps_config.optimizer_method is Optimizing_Methods.STEEPEST:
                descent_dir = [-elem for elem in working_gradient]
            elif varipeps_config.optimizer_method is Optimizing_Methods.CG:
                if count == 0 or signal_reset_descent_dir:
                    descent_dir = [-elem for elem in working_gradient]
                else:
                    descent_dir, beta = _cg_workhorse(
                        working_gradient, old_gradient, old_descent_dir
                    )
            elif varipeps_config.optimizer_method is Optimizing_Methods.BFGS:
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
            elif varipeps_config.optimizer_method is Optimizing_Methods.L_BFGS:
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
            step_conv[random_noise_retries].append(conv)

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
                    generate_unitcell,
                    spiral_indices,
                    additional_input,
                    conv > varipeps_config.optimizer_reuse_env_eps,
                )
            except NoSuitableStepSizeError:
                runtime = time.perf_counter() - runtime_start
                step_runtime[random_noise_retries].append(runtime)

                if varipeps_config.optimizer_fail_if_no_step_size_found:
                    raise
                else:
                    if (
                        (
                            conv > varipeps_config.optimizer_random_noise_eps
                            or working_value > best_value
                        )
                        and random_noise_retries
                        < varipeps_config.optimizer_random_noise_max_retries
                        and not (
                            varipeps_config.optimizer_preconverge_with_half_projectors
                            and not varipeps_global_state.basinhopping_disable_half_projector
                            and varipeps_global_state.ctmrg_projector_method
                            is Projector_Method.HALF
                        )
                    ):
                        tqdm.write(
                            "Convergence is not sufficient. Retry with some random noise on best result."
                        )

                        if working_value < best_value:
                            best_value = working_value
                            best_tensors = working_tensors
                            best_unitcell = working_unitcell
                            best_run = random_noise_retries

                            _autosave_wrapper(
                                autosave_func,
                                autosave_filename,
                                working_tensors,
                                working_unitcell,
                                working_value,
                                "best",
                                best_run,
                                max_trunc_error_list,
                                step_energies,
                                step_chi,
                                step_conv,
                                step_runtime,
                                spiral_indices,
                                additional_input,
                            )

                        if isinstance(input_tensors, PEPS_Unit_Cell) or (
                            isinstance(input_tensors, collections.abc.Sequence)
                            and isinstance(input_tensors[0], PEPS_Unit_Cell)
                        ):
                            working_tensors = (
                                cast(
                                    List[jnp.ndarray],
                                    [
                                        i.tensor
                                        for i in best_unitcell.get_unique_tensors()
                                    ],
                                )
                                + best_tensors[best_unitcell.get_len_unique_tensors() :]
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
                        count = 0
                        random_noise_retries += 1
                        old_descent_dir = descent_dir
                        old_gradient = working_gradient

                        step_energies[random_noise_retries] = []
                        step_chi[random_noise_retries] = []
                        step_conv[random_noise_retries] = []
                        max_trunc_error_list[random_noise_retries] = []
                        step_runtime[random_noise_retries] = []

                        pbar.reset()
                        pbar.refresh()

                        continue
                    else:
                        conv = 0
            else:
                runtime = time.perf_counter() - runtime_start
                step_runtime[random_noise_retries].append(runtime)
                max_trunc_error_list[random_noise_retries].append(max_trunc_error)
                step_energies[random_noise_retries].append(working_value)
                step_chi[random_noise_retries].append(
                    working_unitcell.get_unique_tensors()[0].chi
                )

            if (
                varipeps_config.optimizer_preconverge_with_half_projectors
                and not varipeps_global_state.basinhopping_disable_half_projector
                and varipeps_global_state.ctmrg_projector_method
                is Projector_Method.HALF
                and conv
                < varipeps_config.optimizer_preconverge_with_half_projectors_eps
            ):
                varipeps_global_state.ctmrg_projector_method = (
                    varipeps_config.ctmrg_full_projector_method
                )

                working_value, (working_unitcell, max_trunc_error) = (
                    calc_ctmrg_expectation(
                        working_tensors,
                        working_unitcell,
                        expectation_func,
                        convert_to_unitcell_func,
                        additional_input,
                        enforce_elementwise_convergence=varipeps_config.ad_use_custom_vjp,
                    )
                )
                descent_dir = None
                working_gradient = None
                signal_reset_descent_dir = True
                conv = jnp.inf
                linesearch_step = None

            if conv < varipeps_config.optimizer_convergence_eps:
                working_value, (
                    working_unitcell,
                    max_trunc_error,
                ) = calc_ctmrg_expectation(
                    working_tensors,
                    working_unitcell,
                    expectation_func,
                    convert_to_unitcell_func,
                    additional_input,
                    enforce_elementwise_convergence=varipeps_config.ad_use_custom_vjp,
                )

                try:
                    max_trunc_error_list[random_noise_retries][-1] = max_trunc_error
                except IndexError:
                    max_trunc_error_list[random_noise_retries].append(max_trunc_error)

                try:
                    step_energies[random_noise_retries][-1] = working_value
                except IndexError:
                    step_energies[random_noise_retries].append(working_value)

                break

            old_descent_dir = descent_dir
            old_gradient = working_gradient

            count += 1

            pbar.update()
            pbar.set_postfix(
                {
                    "Energy": f"{working_value:0.10f}",
                    "Retries": random_noise_retries,
                    "Convergence": f"{conv:0.8f}",
                    "Line search step": (
                        f"{linesearch_step:0.8f}"
                        if linesearch_step is not None
                        else "0"
                    ),
                    "Max. trunc. err.": f"{max_trunc_error:0.8g}",
                }
            )
            pbar.refresh()

            if count % varipeps_config.optimizer_autosave_step_count == 0:
                _autosave_wrapper(
                    autosave_func,
                    autosave_filename,
                    working_tensors,
                    working_unitcell,
                    working_value,
                    random_noise_retries,
                    best_run,
                    max_trunc_error_list,
                    step_energies,
                    step_chi,
                    step_conv,
                    step_runtime,
                    spiral_indices,
                    additional_input,
                )

                if working_value < best_value and not (
                    varipeps_config.optimizer_preconverge_with_half_projectors
                    and not varipeps_global_state.basinhopping_disable_half_projector
                    and varipeps_global_state.ctmrg_projector_method
                    is Projector_Method.HALF
                ):
                    _autosave_wrapper(
                        autosave_func,
                        autosave_filename,
                        working_tensors,
                        working_unitcell,
                        working_value,
                        "best",
                        random_noise_retries,
                        max_trunc_error_list,
                        step_energies,
                        step_chi,
                        step_conv,
                        step_runtime,
                        spiral_indices,
                        additional_input,
                    )

            if working_value < best_value and not (
                varipeps_config.optimizer_preconverge_with_half_projectors
                and not varipeps_global_state.basinhopping_disable_half_projector
                and varipeps_global_state.ctmrg_projector_method
                is Projector_Method.HALF
            ):
                best_value = working_value
                best_tensors = working_tensors
                best_unitcell = working_unitcell
                best_run = random_noise_retries

    if working_value < best_value:
        best_value = working_value
        best_tensors = working_tensors
        best_unitcell = working_unitcell
        best_run = random_noise_retries

        if not (
            varipeps_config.optimizer_preconverge_with_half_projectors
            and not varipeps_global_state.basinhopping_disable_half_projector
            and varipeps_global_state.ctmrg_projector_method is Projector_Method.HALF
        ):
            _autosave_wrapper(
                autosave_func,
                autosave_filename,
                working_tensors,
                working_unitcell,
                working_value,
                "best",
                best_run,
                max_trunc_error_list,
                step_energies,
                step_chi,
                step_conv,
                step_runtime,
                spiral_indices,
                additional_input,
            )

    varipeps_global_state.ctmrg_projector_method = None

    print(f"Best energy result found: {best_value}")

    return OptimizeResult(
        success=True,
        x=best_tensors,
        fun=best_value,
        unitcell=best_unitcell,
        nit=count,
        max_trunc_error_list=max_trunc_error_list,
        step_energies=step_energies,
        step_chi=step_chi,
        step_conv=step_conv,
        step_runtime=step_runtime,
        best_run=best_run,
    )
