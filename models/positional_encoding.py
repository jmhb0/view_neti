import ipdb

from typing import Union, List

import torch
from torch import nn
from torch import Tensor


class NeTIPositionalEncoding(nn.Module):
    """ Inherited from NeTI, but we don't use this class"""

    def __init__(self, sigma_t: float, sigma_l: float, num_w: int = 1024):
        super().__init__()
        self.sigma_t = sigma_t
        self.sigma_l = sigma_l
        self.num_w = num_w
        self.w = torch.randn((num_w, 2))
        self.w[:, 0] *= sigma_t
        self.w[:, 1] *= sigma_l
        self.w = nn.Parameter(self.w).cuda()

    def encode(self,
               t: Union[int, torch.Tensor],
               l: Union[int, torch.Tensor],
               debug=False):
        """ Maps the given time and layer input into a 2048-dimensional vector. """
        if type(t) == int or t.ndim == 0:
            x = torch.tensor([t, l]).float()
        else:
            x = torch.stack([t, l], dim=1).T  # (2,bs)
        x = x.cuda()  # (2,)
        v = torch.cat(
            [torch.sin(self.w.detach() @ x),
             torch.cos(self.w.detach() @ x)])  # (2048,bs)
        if type(t) == int:
            v_norm = v / v.norm()
        else:
            v_norm = v / v.norm(dim=0)  # (2048,bs)
            v_norm = v_norm.T  # (bs,2048)
        return v_norm

    def init_layer(self, num_time_anchors: int,
                   num_layers: int) -> torch.Tensor:
        """ Computes the weights for the positional encoding layer of size 160x2048."""
        anchor_vectors = []
        for t_anchor in range(0, 1000, 1000 // num_time_anchors):
            for l_anchor in range(0, num_layers):
                anchor_vectors.append(self.encode(t_anchor, l_anchor).float())
        A = torch.stack(anchor_vectors)
        return A


class BasicEncoder(nn.Module):
    """ Simply normalizes the given timestep and unet layer to be between -1 and 1. """

    def __init__(self,
                 num_denoising_timesteps: int = 1000,
                 num_unet_layers: int = 16):
        super().__init__()
        self.normalized_timesteps = (torch.arange(num_denoising_timesteps) /
                                     (num_denoising_timesteps - 1)) * 2 - 1
        self.normalized_unet_layers = (torch.arange(num_unet_layers) /
                                       (num_unet_layers - 1)) * 2 - 1
        self.normalized_timesteps = nn.Parameter(
            self.normalized_timesteps).cuda()
        self.normalized_unet_layers = nn.Parameter(
            self.normalized_unet_layers).cuda()

    def encode(self, timestep: torch.Tensor,
               unet_layer: torch.Tensor) -> torch.Tensor:
        normalized_input = torch.stack([
            self.normalized_timesteps[timestep.long()],
            self.normalized_unet_layers[unet_layer.long()]
        ]).T
        return normalized_input


import math

class PositionalEncoding(nn.Module):
    """ not used """

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        device = x.device
        freqs = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) *
            -(math.log(10000.0) / self.d_model))
        angles = x.unsqueeze(-1) * freqs.unsqueeze(0).to(x.device)
        encoding = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1)  # (bs,1,d_model)
        return encoding[:, 0]


class FourierPositionalEncoding(nn.Module):
    """
    Not used
    Implementation of 1d Fourier mapping from 
    https://github.com/tancik/fourier-feature-networks 
    """

    def __init__(self,
                 sigma_x: float,
                 dim: int = 128,
                 normalize=False,
                 seed=0):
        super().__init__()
        self.sigma_x = sigma_x
        self.dim = dim
        torch.manual_seed(seed)
        self.w = torch.randn((dim // 2, 1))
        self.w[:, 0] *= sigma_x
        self.w = nn.Parameter(self.w).cuda()
        self.normalize = normalize

    def forward(
        self,
        x: torch.Tensor,
    ):
        """ 
        Maps the given time and layer input into a 2048-dimensional vector. 
        The neti pos encoding does normalization, but the OG
        """
        # check its in range [-1,1]
        # assert all(x >= -1) and all(x <= 1), "inputs should be in [-1,1]"

        if x.ndim == 1:
            x = x.unsqueeze(1)  # (bs,1)
        x = x.T  # (1,bs)
        x = x.cuda()
        v = torch.cat(
            [torch.sin(self.w.detach() @ x),
             torch.cos(self.w.detach() @ x)])  # (dim, bs)

        if self.normalize:
            v = v / v.norm(dim=0)

        v = v.T  # (bs,dim)
        return v


class FourierPositionalEncodingNDims(nn.Module):
    """ 
    Implementation of n-dim Fourier mapping from 
    https://github.com/tancik/fourier-feature-networks 

    This is the class we use.
    """

    def __init__(self,
                 sigmas: List[float],
                 dim: int = 128,
                 normalize=False,
                 seed=0):
        super().__init__()
        # some config
        self.sigmas = sigmas
        self.dim = dim
        nfeats = len(sigmas)
        torch.manual_seed(seed)

        # generate the random features
        self.w = torch.randn((dim // 2, nfeats))
        for i in range(nfeats):
            self.w[:, i] *= sigmas[i]

        self.w = nn.Parameter(self.w).cuda()
        self.normalize = normalize

    def forward(self, x: torch.Tensor):
        """ 
        Maps the given time and layer input into a 2048-dimensional vector. 
        The neti pos encoding does normalization, but the OG
        """
        # check its in range [-1,1]
        # assert torch.all(x >= -1) and torch.all(x <= 1), "inputs should be in [-1,1]"

        if x.ndim == 1:
            x = x.unsqueeze(1)  # (bs,1)

        x = x.T  # (1,bs)
        x = x.cuda()
        v = torch.cat(
            [torch.sin(self.w.detach() @ x),
             torch.cos(self.w.detach() @ x)])  # (dim, bs)

        if self.normalize:
            v = v / v.norm(dim=0)

        v = v.T  # (bs,dim)
        return v


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    sys.path.append("..")
    from utils.types import PESigmas

    # phi = torch.arange(-1, 1, 0.1)
    bs = 8
    data = torch.randn((bs, 3))  # 3 dims - (t,l,phi)
    pe_sigmas = PESigmas(**{'sigma_t': 0.03, 'sigma_l': 2.0, 'sigma_phi': 1.0})
    pos_enc = FourierPositionalEncodingNDims(
        dim=128,
        sigmas=[pe_sigmas.sigma_t, pe_sigmas.sigma_l, pe_sigmas.sigma_phi])
    enc = pos_enc(data)
