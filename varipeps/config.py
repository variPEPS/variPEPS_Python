from dataclasses import dataclass
from enum import Enum, IntEnum, auto, unique

from jax.tree_util import register_pytree_node_class

from typing import TypeVar, Tuple, Any, Type, NoReturn

T_VariPEPS_Config = TypeVar("T_VariPEPS_Config", bound="VariPEPS_Config")


@unique
class Optimizing_Methods(IntEnum):
    STEEPEST = auto()  #: Steepest gradient descent
    CG = auto()  #: Conjugate gradient method
    BFGS = auto()  #: BFGS method
    L_BFGS = auto()  #: L-BFGS method


@unique
class Line_Search_Methods(IntEnum):
    SIMPLE = auto()  #: Simple line search method
    ARMIJO = auto()  #: Armijo line search method
    WOLFE = auto()  #: Wolfe line search method
    HAGERZHANG = auto()  #: Hager-Zhang line search method


@unique
class Projector_Method(IntEnum):
    HALF = auto()  #: Use only half network for projector calculation
    FULL = auto()  #: Use full network for projector calculation
    FISHMAN = auto()  #: Use the Fishman method for projector calculation


@unique
class Wavevector_Type(IntEnum):
    TWO_PI_POSITIVE_ONLY = auto()  #: Use interval [0, 2pi) for q vectors
    TWO_PI_SYMMETRIC = auto()  #: Use interval [-2pi, 2pi) for q vectors


@dataclass
@register_pytree_node_class
class VariPEPS_Config:
    """
    Config class for varipeps module. Normally only the blow created instance
    :obj:`config` is used.

    Parameters:
      ad_use_custom_vjp (:obj:`bool`):
        Use custom VJP rule for the CTMRG routine during AD calculation.
      ad_custom_print_steps (:obj:`bool`):
        Print steps of fix-point iteration in custom VJP function.
      ad_custom_verbose_output (:obj:`bool`):
        Print verbose output in custom VJP function.
      ad_custom_convergence_eps (:obj:`float`):
        Convergence criterion for the custom VJP function.
      ad_custom_max_steps (:obj:`int`):
        Maximal number of steps for fix-pointer iteration of the custom VJP
        function.
      checkpointing_ncon (:obj:`bool`):
        Enable AD checkpointing for the ncon calls.
      checkpointing_projectors (:obj:`bool`):
        Enable AD checkpointing for the the calculation of the proejctors.
      ctmrg_convergence_eps (:obj:`float`):
        Convergence criterion for the CTMRG routine.
      ctmrg_enforce_elementwise_convergence (:obj:`bool`):
        Enforce elementwise convergence of the CTM tensors instead of only
        convergence of the singular values of the corners.
      ctmrg_max_steps (:obj:`int`):
        Maximal number of steps for fix-pointer iteration of the CTMRG routine.
      ctmrg_print_steps (:obj:`bool`):
        Print steps of fix-point iteration in CTMRG routine.
      ctmrg_verbose_output (:obj:`bool`):
        Print verbose output in CTMRG routine.
      ctmrg_truncation_eps (:obj:`float`):
        Value for cut off of the singular values compared to the
        biggest one. Used in the calculation of the CTMRG projectors.
      ctmrg_fail_if_not_converged (:obj:`bool`):
        Flag if the CTMRG routine should fail with an error if no convergence
        can be reached within the maximal number of steps.
        If disabled, the result converged so far is returned.
      ctmrg_full_projector_method (:obj:`~varipeps.config.Projector_Method`):
        Set which projector method should be used as default (full) projector
        method during the CTMRG routine. Sensible values are
        :obj:`~varipeps.config.Projector_Method.FULL` or
        :obj:`~varipeps.config.Projector_Method.FISHMAN`.
      ctmrg_increase_truncation_eps (:obj:`bool`):
        Flag if the CTMRG routine should try higher truncation thresholds for
        the SVD based projector methods if the routine does not converge in the
        maximum number of steps.
      ctmrg_increase_truncation_eps_factor (:obj:`float`):
        Factor by which the truncation threshold should be increased.
      ctmrg_increase_truncation_eps_max_value (:obj:`float`):
        Maximal value for the truncation threshold. Do not increase higher than
        this value.
      ctmrg_heuristic_increase_chi (:obj:`bool`):
        Flag if the CTMRG routine should try higher environment bond dimension
        for if the routine found singular values above a threshold during the
        projector calculation of the last absorption step.
      ctmrg_heuristic_increase_chi_threshold (:obj:`float`):
        Threshold for the heuristic environment bond dimension increase.
      ctmrg_heuristic_increase_chi_step_size (:obj:`int`):
        Step size for the heuristic environment bond dimension increase.
      ctmrg_heuristic_decrease_chi (:obj:`bool`):
        Flag if the CTMRG routine should try lower environment bond dimension
        for if the routine found singular values below the SVD threshold during
        the projector calculation of the last absorption step.
      ctmrg_heuristic_decrease_chi_step_size (:obj:`int`):
        Step size for the heuristic environment bond dimension decrease.
      svd_sign_fix_eps (:obj:`float`):
        Value for numerical stability threshold in sign-fixed SVD.
      optimizer_method (:obj:`Optimizing_Methods`):
        Method used for variational optimization of the PEPS network.
      optimizer_max_steps (:obj:`int`):
        Maximal number of steps for fix-pointer iteration in optimization routine.
      optimizer_convergence_eps (:obj:`float`):
        Convergence criterion for the optimization routine.
      optimizer_ctmrg_preconverged_eps (:obj:`float`):
        Convergence criterion for the optimization routine using the gradient
        calculations with the preconverged environment.
      optimizer_fail_if_no_step_size_found (:obj:`bool`):
        Flag if the optimizer routine should fail with an error if no step size
        can be found before the gradient norm is below the convergence
        threshold. If disabled, the result converged so far is returned.
      optimizer_l_bfgs_maxlen (:obj:`int`):
        Maximal number of previous steps used for the L-BFGS method.
      optimizer_preconverge_with_half_projectors (:obj:`bool`):
        Flag if the optimizer should use only CTM half projectors for the steps
        till some converge is reached.
      optimizer_preconverge_with_half_projectors_eps (:obj:`float`):
        Convergence criterion for the preconvergence with only the half
        CTM projectors.
      optimizer_autosave_step_count (:obj:`int`):
        Step count after which the optimizer result is automatically saved.
      optimizer_random_noise_eps (:obj:`float`):
        Optimizer should try best state sofar with some random noise if
        gradient norm is below this threshold.
      optimizer_random_noise_max_retries (:obj:`int`):
        Maximal retries for optimization with random noise.
      optimizer_random_noise_relative_amplitude (:obj:`float`):
        Relative amplitude used for random noise.
      optimizer_reuse_env_eps (:obj:`float`):
        Reuse CTMRG environment of previous step if norm of gradient is below
        this threshold.
      line_search_method (:obj:`Line_Search_Methods`):
        Method used for the line search routine.
      line_search_initial_step_size (:obj:`float`):
        Initial step size for the line search routine.
      line_search_reduction_factor (:obj:`float`):
        Reduction factor between two line search steps.
      line_search_max_steps (:obj:`int`):
        Maximal number of steps in the line search routine.
      line_search_armijo_const (:obj:`float`):
        Constant used in Armijo line search method.
      line_search_wolfe_const (:obj:`float`):
        Constant used in Wolfe line search method.
      line_search_use_last_step_size (:obj:`bool`):
        Flag if the line search should start from the step size of the
        previous optimizer step.
      line_search_hager_zhang_quad_step (:obj:`bool`):
        Use QuadStep method in Hager-Zhang line search to find initial
        step size.
      line_search_hager_zhang_delta (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_sigma (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_psi_0 (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_psi_1 (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_psi_2 (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_eps (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_theta (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_gamma (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      line_search_hager_zhang_rho (:obj:`float`):
        Constant used in Hager-Zhang line search method.
      basinhopping_niter (:obj:`int`):
        Value for parameter `niter` of :obj:`scipy.optimize.basinhopping`.
        See this function for details.
      basinhopping_T (:obj:`int`):
        Value for parameter `T` of :obj:`scipy.optimize.basinhopping`.
        See this function for details.
      basinhopping_niter_success (:obj:`int`):
        Value for parameter `niterniter_success` of
        :obj:`scipy.optimize.basinhopping`. See this function for details.
      spiral_wavevector_type (:obj:`Wavevector_Type`):
        Type of wavevector to be used (only positive/symmetric interval/...).
    """

    # AD config
    ad_use_custom_vjp: bool = True
    ad_custom_print_steps: bool = False
    ad_custom_verbose_output: bool = False
    ad_custom_convergence_eps: float = 1e-7
    ad_custom_max_steps: int = 75
    checkpointing_ncon: bool = False
    checkpointing_projectors: bool = False

    # CTMRG routine
    ctmrg_convergence_eps: float = 1e-8
    ctmrg_enforce_elementwise_convergence: bool = True
    ctmrg_max_steps: int = 75
    ctmrg_print_steps: bool = False
    ctmrg_verbose_output: bool = False
    ctmrg_truncation_eps: float = 1e-12
    ctmrg_fail_if_not_converged: bool = True
    ctmrg_full_projector_method: Projector_Method = Projector_Method.FISHMAN
    ctmrg_increase_truncation_eps: bool = True
    ctmrg_increase_truncation_eps_factor: float = 100
    ctmrg_increase_truncation_eps_max_value: float = 1e-6
    ctmrg_heuristic_increase_chi: bool = True
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6
    ctmrg_heuristic_increase_chi_step_size: float = 2
    ctmrg_heuristic_decrease_chi: bool = True
    ctmrg_heuristic_decrease_chi_step_size: float = 1

    # SVD
    svd_sign_fix_eps: float = 1e-1

    # Optimizer
    optimizer_method: Optimizing_Methods = Optimizing_Methods.BFGS
    optimizer_max_steps: int = 300
    optimizer_convergence_eps: float = 1e-5
    optimizer_ctmrg_preconverged_eps: float = 1e-5
    optimizer_fail_if_no_step_size_found: bool = False
    optimizer_l_bfgs_maxlen: int = 50
    optimizer_preconverge_with_half_projectors: bool = True
    optimizer_preconverge_with_half_projectors_eps: float = 1e-3
    optimizer_autosave_step_count: int = 2
    optimizer_random_noise_eps: float = 1e-4
    optimizer_random_noise_max_retries: int = 5
    optimizer_random_noise_relative_amplitude: float = 1e-1
    optimizer_reuse_env_eps: float = 1e-3

    # Line search
    line_search_method: Line_Search_Methods = Line_Search_Methods.HAGERZHANG
    line_search_initial_step_size: float = 1.0
    line_search_reduction_factor: float = 0.5
    line_search_max_steps: int = 40
    line_search_armijo_const: float = 1e-4
    line_search_wolfe_const: float = 0.9
    line_search_use_last_step_size: bool = False

    line_search_hager_zhang_quad_step: bool = True
    line_search_hager_zhang_delta: float = 0.1
    line_search_hager_zhang_sigma: float = 0.9
    line_search_hager_zhang_psi_0: float = 0.01
    line_search_hager_zhang_psi_1: float = 0.1
    line_search_hager_zhang_psi_2: float = 2.0
    line_search_hager_zhang_eps: float = 1e-6
    line_search_hager_zhang_theta: float = 0.5
    line_search_hager_zhang_gamma: float = 0.66
    line_search_hager_zhang_rho: float = 5

    # Basinhopping
    basinhopping_niter: int = 20
    basinhopping_T: float = 0.001
    basinhopping_niter_success: int = 5

    # Spiral PEPS
    spiral_wavevector_type: Wavevector_Type = Wavevector_Type.TWO_PI_POSITIVE_ONLY

    def update(self, name: str, value: Any) -> NoReturn:
        self.__setattr__(name, value)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        try:
            field = self.__dataclass_fields__[name]
        except KeyError as e:
            raise KeyError(f"Unknown config option '{name}'.") from e

        if not type(value) is field.type:
            if field.type is float and type(value) is int:
                pass
            else:
                raise TypeError(
                    f"Type mismatch for option '{name}', got '{type(value)}', expected '{field.type}'."
                )

        super().__setattr__(name, value)

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        aux_data = (
            {name: getattr(self, name) for name in self.__dataclass_fields__.keys()},
        )

        return ((), aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_VariPEPS_Config],
        aux_data: Tuple[Any, ...],
        children: Tuple[Any, ...],
    ) -> T_VariPEPS_Config:
        (data_dict,) = aux_data

        return cls(**data_dict)


config = VariPEPS_Config()


class ConfigModuleWrapper:
    __slots__ = {
        "Optimizing_Methods",
        "Line_Search_Methods",
        "Projector_Method",
        "Wavevector_Type",
        "VariPEPS_Config",
        "config",
    }

    def __init__(self):
        for e in self.__slots__:
            setattr(self, e, globals()[e])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in self.__slots__:
            return super().__getattr__(name)
        else:
            return getattr(self.config, name)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        if not name.startswith("__") and name not in self.__slots__:
            setattr(self.config, name, value)
        elif not hasattr(self, name):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Attribute '{name}' is write-protected.")


wrapper = ConfigModuleWrapper()
