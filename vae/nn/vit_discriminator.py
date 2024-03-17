"""
Discriminator for adversarial objectives
"""

from .vit_vae import ViTEncoder

from torch import nn
import torch

import einops as eo

class ViTPatchDiscriminator(ViTEncoder):
    """
    Patch based discriminator that essentially determines if each patch
    of the image 

    :param mixing_ratio: Probability of a fake patch replacing a real one
    """
    def __init__(self, mixing_ratio = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proj_out = nn.Linear(self.hidden_size, 1)
        self.loss = nn.BCEWithLogitsLoss()

        self.mixing_ratio = mixing_ratio

    def classify(self, x, patchify = False):
        if patchify:
            x = self.patchify(x)
        x = self.proj_in(x)
        x = self.blocks(x, output_hidden_states = False)
        x = self.norm(x)
        x = self.proj_out(x)
        return x

    def forward(self, real, fake):
        # Patchify both
        real = self.patchify(real) # [B, N, D]
        fake = self.patchify(fake) # [B, N, D]
        b,n,d = real.shape

        # Create random mask to splice fake patches into real ones
        mask = torch.rand(b, n, device = real.device)
        patch_mask = eo.repeat(mask, 'b n -> b n d', d = d)
        
        splice = torch.where(patch_mask <= self.mixing_ratio, fake, real)
        labels = self.classify(splice).squeeze(-1) # [B, N]

        true_labels =  torch.where(
            mask <= self.mixing_ratio,
            torch.zeros_like(mask),
            torch.ones_like(mask)
        )

        return self.loss(labels, true_labels)
        
if __name__ == "__main__":
    model = ViTPatchDiscriminator(
        0.5,
        (32, 32), (3, 224, 224), (4, 96, 96),
        4, 8, 256
    ).cuda()

    x = torch.randn(1, 3, 224, 224, device = 'cuda')
    y = model(x, x)
    print(y)