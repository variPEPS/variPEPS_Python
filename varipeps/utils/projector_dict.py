from collections import namedtuple
import collections.abc
from dataclasses import dataclass, field

from typing import Sequence, Tuple, List, Dict, TypeVar

Left_Projectors = namedtuple("Left_Projectors", ("top", "bottom"))
Right_Projectors = namedtuple("Right_Projectors", ("top", "bottom"))
Top_Projectors = namedtuple("Top_Projectors", ("left", "right"))
Bottom_Projectors = namedtuple("Bottom_Projectors", ("left", "right"))

Unit_Cell_Bond_Dim_Projectors = namedtuple(
    "Unit_Cell_Bond_Dim_Projectors", ("left", "right", "top", "bottom")
)

T_Projector = TypeVar(
    "T_Projector", Left_Projectors, Right_Projectors, Top_Projectors, Bottom_Projectors
)


@dataclass
class Projector_Dict(collections.abc.MutableMapping):
    max_x: int
    max_y: int
    projector_dict: Dict[Tuple[int, int], T_Projector] = field(default_factory=dict)  # type: ignore

    def __getitem__(self, key: Tuple[int, int]) -> T_Projector:
        return self.projector_dict[key]

    def __setitem__(self, key: Tuple[int, int], value: T_Projector) -> None:
        self.projector_dict[key] = value

    def __delitem__(self, key: Tuple[int, int]) -> None:
        self.projector_dict.__delitem__(key)

    def __iter__(self):
        return self.projector_dict.__iter__()

    def __len__(self):
        return self.projector_dict.__len__()

    def get_projector(
        self,
        current_x: int,
        current_y: int,
        relative_x: int,
        relative_y: int,
    ) -> T_Projector:
        select_x = (current_x + relative_x) % self.max_x
        select_y = (current_y + relative_y) % self.max_y

        return self.projector_dict[(select_x, select_y)]
