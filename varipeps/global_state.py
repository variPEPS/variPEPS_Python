from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class

from typing import TypeVar, Tuple, Any, Type, Optional

from .config import Projector_Method

T_VariPEPS_Global_State = TypeVar(
    "T_VariPEPS_Global_State", bound="VariPEPS_Global_State"
)


@dataclass
@register_pytree_node_class
class VariPEPS_Global_State:
    """
    Class to track internal global state. Values of the instance of this
    class should not be modified by users.
    """

    ctmrg_effective_truncation_eps: Optional[float] = None
    ctmrg_projector_method: Optional[Projector_Method] = None
    basinhopping_disable_half_projector: Optional[bool] = None

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        aux_data = (
            {name: getattr(self, name) for name in self.__dataclass_fields__.keys()},
        )

        return ((), aux_data)

    @classmethod
    def tree_unflatten(
        cls: Type[T_VariPEPS_Global_State],
        aux_data: Tuple[Any, ...],
        children: Tuple[Any, ...],
    ) -> T_VariPEPS_Global_State:
        (data_dict,) = aux_data

        return cls(**data_dict)


global_state = VariPEPS_Global_State()
