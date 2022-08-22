from dataclasses import dataclass

from typing import Optional


@dataclass
class PEPS_AD_Global_State:
    """
    Class to track internal global state. Values of the instance of this
    class should not be modified by users.
    """

    ctmrg_effective_truncation_eps: Optional[float] = None


global_state = PEPS_AD_Global_State()
