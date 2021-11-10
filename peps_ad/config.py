from dataclasses import dataclass


@dataclass
class PEPS_AD_Config:
    checkpointing_ncon: bool = True
    checkpointing_projectors: bool = False


config = PEPS_AD_Config()
