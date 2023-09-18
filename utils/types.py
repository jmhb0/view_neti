import enum
from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    input_ids_placeholder_object: torch.Tensor
    input_ids_placeholder_view: torch.Tensor
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None

@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float
    sigma_theta: Optional[float] = float
    sigma_phi: Optional[float] = float
    sigma_r: Optional[float] = float
    sigma_dtu12: Optional[float] = float

@dataclass 
class MapperOutput:
   word_embedding: torch.Tensor
   bypass_output: torch.Tensor 
   bypass_unconstrained: bool
   output_bypass_alpha: float