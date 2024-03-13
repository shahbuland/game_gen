"""
ViT autoencoder
"""

from .vit_modules import (
    PositionalEncoding,
    Transformer,
    StackedTransformer,
    MLP
)

from torch import nn
import torch.nn.functional as F
import torch
import einops as eo
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class ViTEncoder(nn.Module):
    """
    ViT image encoder

    :param patching: Tuple for how big the patches are in each dim
    (i.e. p_y = 4, p_x = 4 -> (4, 4))

    :param input_shape: Tuple for input shape (i.e. (3, 768, 768))
    :param latent_shape: Tuple for latent shape (i.e. (4, 96, 96))
    """
    def __init__(self, patching, input_shape, latent_shape, n_layers, n_heads, hidden_size):
        super().__init__()

        self.p_y, self.p_x = patching
        c, h, w = input_shape
        l_c, l_h, l_w = latent_shape

        # Now to find out latent patch sizes we look at n_patches
        n_patches = (h // self.p_y) * (w // self.p_x)
        # = (l_h // self.p_l_y) * (l_w // self.p_l_x)
        # Assuming square images, this can easily be done
        # = (l_h * l_w) / p_l**2
        self.p_l = round(((l_h * l_w) / n_patches)**.5)
        self.n_p_l = l_h//self.p_l

        patch_content = c * self.p_y * self.p_x
        latent_content = l_c * self.p_l**2

        self.proj_in = nn.Linear(patch_content, hidden_size)
        self.blocks = StackedTransformer(n_layers, n_heads, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, 2*latent_content)

        self.d = latent_content

    def patchify(self, x):
        return eo.rearrange(
            x,
            'b c (n_p_y p_y) (n_p_x p_x) -> b (n_p_y n_p_x) (p_y p_x c)',
            p_y = self.p_y,
            p_x = self.p_x
        )
    
    def depatchify_latent(self, z):
        return eo.rearrange(
            z,
            'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
            n_p_y = self.n_p_l,
            n_p_x = self.n_p_l,
            p_y = self.p_l,
            p_x = self.p_l
        )
    
    def forward(self, pixel_values, output_hidden_states = True):
        x = self.patchify(pixel_values)
        x = self.proj_in(x)
        
        h = []
        x = self.blocks(x, output_hidden_states = output_hidden_states)
        if output_hidden_states:
            x, h = x

        
        x = self.norm(x)
        x = self.proj_out(x)

        mu, logvar = x[...,:self.d], x[...,self.d:]
        mu = self.depatchify_latent(mu)
        logvar = self.depatchify_latent(logvar)

        # They have to cat on channels for this object to work
        dist = DiagonalGaussianDistribution(torch.cat([mu, logvar], dim = 1))

        if output_hidden_states:
            return dist.sample(), dist, h
        else:
            return dist.sample(), dist

class ViTDecoder(nn.Module):
    def __init__(self, patching, input_shape, latent_shape, n_layers, n_heads, hidden_size):
        super().__init__()

        # Patching info
        self.p_y, self.p_x = patching
        c, h, w = input_shape
        l_c, l_h, l_w = latent_shape

        self.n_p_y = h // self.p_y
        self.n_p_x = w // self.p_x
        n_patches = self.n_p_x * self.n_p_y
        self.p_l = round(((l_h * l_w) / n_patches)**.5)

        patch_content = c * self.p_y * self.p_x
        latent_content = l_c * self.p_l**2

        # Model core
        self.proj_in = nn.Linear(latent_content, hidden_size)
        self.blocks = StackedTransformer(n_layers, n_heads, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_content)
    
    def patchify_latent(self, z):
        return eo.rearrange(
            z,
            'b c (n_p_y p_y) (n_p_x p_x) -> b (n_p_y n_p_x) (p_y p_x c)',
            p_y = self.p_l,
            p_x = self.p_l
        )

    def depatchify(self, x):
        return eo.rearrange(
            x,
            'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
            n_p_x = self.n_p_x,
            n_p_y = self.n_p_y,
            p_y = self.p_y,
            p_x = self.p_x
        )
    
    def forward(self, latent, output_hidden_states = True):
        x = self.patchify_latent(latent)
        x = self.proj_in(x)
        
        h = []
        x = self.blocks(x, output_hidden_states = output_hidden_states)
        if output_hidden_states:
            x, h = x
        
        x = self.norm(x)
        x = self.proj_out(x)

        return self.depatchify(x)

class ViTVAE(nn.Module):
    def __init__(self, patching, input_shape, latent_shape, n_layers, n_heads, hidden_size):
        super().__init__()

        self.encoder = ViTEncoder(
            patching, input_shape, latent_shape,
            n_layers, n_heads, hidden_size
        )

        self.decoder = ViTDecoder(
            patching, input_shape, latent_shape,
            n_layers, n_heads, hidden_size
        )

    def encode(self, pixel_values, output_hidden_states = True):
        return self.encoder(pixel_values, output_hidden_states = output_hidden_states)

    def decode(self, latent, output_hidden_states = True):
        return self.decoder(latent, output_hidden_states = output_hidden_states)

    def forward(self, pixel_values):
        latent, dist = self.encode(pixel_values, output_hidden_states = False)
        rec = self.decode(latent, output_hidden_states = False)

        rec_term = F.mse_loss(rec, pixel_values)
        kl_term = dist.kl()

        return 0.000001 * kl_term + rec_term

if __name__ == "__main__":
    model = ViTVAE(
        (32, 32), (3, 768, 768), (4, 96, 96),
        4, 8, 256
    ).cuda()

    x = torch.randn(1, 3, 768, 768, device = 'cuda')

    loss = model(x)
