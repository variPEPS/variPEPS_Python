from dataclasses import dataclass
from enum import Enum, auto


class Optimizing_Methods(Enum):
    STEEPEST = auto()  #: Steepest gradient descent
    CG = auto()  #: Conjugate gradient method
    BFGS = auto()  #: BFGS method


class Line_Search_Methods(Enum):
    SIMPLE = auto()  #: Simple line search method
    ARMIJO = auto()  #: Armijo line search method
    WOLFE = auto()  #: Wolfe line search method


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
    """

    # AD config
    ad_use_custom_vjp: bool = True
    ad_custom_print_steps: bool = False
    ad_custom_verbose_output: bool = False
    ad_custom_convergence_eps: float = 1e-7
    ad_custom_max_steps: int = 100
    checkpointing_ncon: bool = False
    checkpointing_projectors: bool = False

    # CTMRG routine
    ctmrg_convergence_eps: float = 1e-8
    ctmrg_enforce_elementwise_convergence: bool = True
    ctmrg_max_steps: int = 150
    ctmrg_print_steps: bool = False
    ctmrg_verbose_output: bool = False
    ctmrg_truncation_eps: float = 1e-8

    # SVD
    svd_sign_fix_eps: float = 1e-4

    # Optimizer
    optimizer_method: Optimizing_Methods = Optimizing_Methods.BFGS
    optimizer_max_steps: int = 300
    optimizer_convergence_eps: float = 1e-4
    optimizer_ctmrg_preconverged_eps: float = 1e-5

    # Line search
    line_search_method: Line_Search_Methods = Line_Search_Methods.ARMIJO
    line_search_initial_step_size: float = 1.0
    line_search_reduction_factor: float = 0.5
    line_search_max_steps: int = 20
    line_search_armijo_const: float = 1e-4
    line_search_wolfe_const: float = 0.1


config = PEPS_AD_Config()
