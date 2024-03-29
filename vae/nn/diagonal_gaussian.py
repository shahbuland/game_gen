from typing import Tuple, Optional

import torch 
from torch import nn

import einops as eo

# Modified version of diffusers gaussian that works with any shape mean/logvar
# source: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/autoencoders/vae.py

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Tuple[torch.Tensor, torch.Tensor], deterministic: bool = False):
        self.parameters = parameters[0] # For device/dtype stuff

        self.mean, self.logvar = parameters
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype

        sample = torch.randn(
            self.mean.shape,
            generator = generator,
            device = self.mean.device,
            dtype = self.mean.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                term = torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
                return 0.5 * eo.reduce(term, 'b ... -> b', reduction = 'sum')
            else:
                term = torch.pow(self.mean - other.mean, 2) / other.var \
                    + self.var / other.var \
                    - 1.0 \
                    - self.logvar \
                    + other.logvar 
                return 0.5 * eo.reduce(term, 'b ... -> b', reduction = 'sum')

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)

        term = logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var
        return 0.5 * eo.reduce(term, 'b ... -> b', reduction = 'sum')

    def mode(self) -> torch.Tensor:
        return self.mean
