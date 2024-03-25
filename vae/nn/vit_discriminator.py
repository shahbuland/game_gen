"""
Discriminator for adversarial objectives
"""

from .vit_vae import ViTEncoder

from torch import nn
import torch

torch.autograd.set_detect_anomaly(True)

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
            torch.zeros_like(labels),#labels, # fakes should be minimized
            torch.ones_like(labels)#-1 * labels # reals should be maximized
        )

        loss = self.loss(labels, true_labels)
        adv_loss = self.loss(labels, torch.ones_like(labels))
        return loss, adv_loss

class MultiDiscriminator(nn.Module):
    """
    Same as above but does various patch scales (doubling).
    Passed patching is used as a base and successive discriminators
    double the size
    """
    def __init__(
        self,
        mixing_ratio = 0.5, scales = 2,
        patching = None, input_shape = None, latent_shape = None,
        n_layers = None, n_heads = None, hidden_size = None
    ):
        super().__init__()

        self.discs = nn.ModuleList([])

        for i in range(scales):
            patching = (patching[0] * 2 ** i, patching[1] * 2 ** i)
            self.discs.append(ViTPatchDiscriminator(
                mixing_ratio, patching, input_shape, latent_shape,
                n_layers, n_heads, hidden_size
            ))
    
    def forward(self, real, fake):
        losses, adv_losses = 0.0, 0.0
        for disc in self.discs:
            loss, adv = disc(real, fake)
            losses = losses + loss
            adv_losses = adv_losses + adv
        
        return losses, adv_losses

if __name__ == "__main__":
    model = MultiDiscriminator(
        0.5, 2,
        (32, 32), (3, 512, 512), (4, 64, 64),
        4, 8, 256
    ).cuda()

    x = torch.randn(1, 3, 512, 512, device = 'cuda')
    y = model(x, x)
    print(y)

