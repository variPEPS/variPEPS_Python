import collections
from dataclasses import dataclass, field

import jax

from typing import Any, Dict


@dataclass
class Checkpointing_Cache(collections.abc.MutableMapping):
    enable_checkpointing: bool
    cache: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if self.enable_checkpointing:
            self.cache[key] = jax.checkpoint(value)
        else:
            self.cache[key] = value

    def __delitem__(self, key: str) -> None:
        self.cache.__delitem__(key)

    def __iter__(self):
        return self.cache.__iter__()

    def __len__(self):
        return self.cache.__len__()
