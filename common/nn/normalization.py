from torchtyping import TensorType

import torch
from torch import nn

import einops as eo

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(d))

    def forward(self, x : TensorType["b", "n", "d"]):
        gain = (1 + self.g)[None,None,:] # Add a batch and sequence dim

        rms = (x.pow(2).mean(-1)).sqrt() # [b, n]
        rms = rms[:,:,None]

        x = (x / rms)
        x = x * gain

        return x