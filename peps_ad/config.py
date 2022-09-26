from dataclasses import dataclass
from enum import Enum, IntEnum, auto, unique


class Optimizing_Methods(Enum):
    STEEPEST = auto()  #: Steepest gradient descent
    CG = auto()  #: Conjugate gradient method
    BFGS = auto()  #: BFGS method
    L_BFGS = auto()  #: L-BFGS method


class Line_Search_Methods(Enum):
    SIMPLE = auto()  #: Simple line search method
    ARMIJO = auto()  #: Armijo line search method
    WOLFE = auto()  #: Wolfe line search method


@unique
class Projector_Method(IntEnum):
    HALF = auto()  #: Use only half network for projector calculation
    FULL = auto()  #: Use full network for projector calculation
    FISHMAN = auto()  #: Use the Fishman method for projector calculation


@dataclass
class PEPS_AD_Config:
    """
    Config class for peps-ad module. Normally only the blow created instance
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
        Value for cut off of singular values compared to the biggest one. Used
        in calculation of CTMRG projectors.
      ctmrg_fail_if_not_converged (:obj:`bool`):
        Flag if the CTMRG routine should fail with an error if no convergence
        can be reached within the maximal number of steps.
        If disabled, the result converged so far is returned.
      ctmrg_full_projector_method (:obj:`~peps_ad.config.Projector_Method`):
        Set which projector method should be used as default (full) projector
        method during the CTMRG routine. Sensible values are
        :obj:`~peps_ad.config.Projector_Method.FULL` or
        :obj:`~peps_ad.config.Projector_Method.FISHMAN`.
      ctmrg_increase_truncation_eps (:obj:`bool`):
        Flag if the CTMRG routine should try higher truncation thresholds for
        the SVD based projector methods if the routine does not converge in the
        maximum number of steps.
      ctmrg_increase_truncation_eps_factor (:obj:`float`):
        Factor by which the truncation threshold should be increased.
      ctmrg_increase_truncation_eps_max_value (:obj:`float`):
        Maximal value for the truncation threshold. Do not increase higher than
        this value.
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
    ctmrg_truncation_eps: float = 1e-7
    ctmrg_fail_if_not_converged: bool = True
    ctmrg_full_projector_method: Projector_Method = Projector_Method.FULL
    ctmrg_increase_truncation_eps: bool = True
    ctmrg_increase_truncation_eps_factor: float = 10
    ctmrg_increase_truncation_eps_max_value: float = 1e-4

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

    # Line search
    line_search_method: Line_Search_Methods = Line_Search_Methods.WOLFE
    line_search_initial_step_size: float = 1.0
    line_search_reduction_factor: float = 0.5
    line_search_max_steps: int = 20
    line_search_armijo_const: float = 1e-4
    line_search_wolfe_const: float = 0.9
    line_search_use_last_step_size: bool = False


config = PEPS_AD_Config()
