"""
This script adds all the main modules from game_gen vae in one compressed script
"""

from common.nn.vit_modules import (
    PositionalEncoding,
    Transformer,
    StackedTransformer,
    MLP,
)

from common.modeling import MixIn
from common.configs import ViTConfig

from torch import nn
import torch.nn.functional as F
import torch
import einops as eo

from vae.nn.diagonal_gaussian import DiagonalGaussianDistribution

class ViTEncoder(MixIn):
    """
    ViT video encoder

    :param spatial_downsample: How much of a spacial downsample should we enforce? (i.e. 512 -> 64, f = 8)
    :param temporal_downsample: How much of a temporal downsample? (i.e. 100 -> 10, f = 10)
    :param latent_channels: Not really "channels" but adds more dimensionality to the latents

    :param vit_config: Config for ViT details (see common.configs docs)
    """
    def __init__(
        self,
        spatial_downsample = 8, temporal_downsample = 8, latent_channels = 4,
        vit_config = ViTConfig()
    ):
        super().__init__()

        self.p_y, self.p_x, self.p_t = vit_config.patching
        t, c, h, w = vit_config.input_shape
        
        hidden_size = vit_config.hidden_size
        patch_content = c * self.p_y * self.p_x * self.p_t
        latent_content = latent_channels * self.p_y * self.p_x * self.p_t // (spatial_downsample ** 2) // temporal_downsample

        self.proj_in = nn.Linear(patch_content, hidden_size)
        self.blocks = StackedTransformer(
            vit_config.n_layers,
            vit_config.n_heads,
            vit_config.hidden_size
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, 2 * latent_content)

        self.d = latent_content
        self.hidden_size = hidden_size
    
    def patchify(self, x):
        return eo.rearrange(
            x,
            'b (n_p_t p_t) c (n_p_y p_y) (n_p_x p_x) -> b (n_p_t n_p_y n_p_x) (p_t p_y p_x c)',
            p_y = self.p_y,
            p_x = self.p_x,
            p_t = self.p_t
        )
    
    def forward(self, pixel_values, output_hidden_states = False):
        x = self.patchify(pixel_values)
        x = self.proj_in(x)

        h = []
        x = self.blocks(x, output_hidden_states = output_hidden_states)
        if output_hidden_states:
            x, h = x

        x = self.norm(x)
        x = self.proj_out(x)

        mu, logvar = x[...,:self.d], x[...,self.d:]
        dist = DiagonalGaussianDistribution((mu, logvar))

        if output_hidden_states:
            return dist, h
        return dist

class ViTDecoder(MixIn):
    """
    Decoder analagous to above encoder.
    """
    def __init__(
        self,
        spatial_downsample = 8, temporal_downsample = 8, latent_channels = 4,
        vit_config = ViTConfig()
    ):
        super().__init__()

        self.p_y, self.p_x, self.p_t = vit_config.patching
        t, c, h, w = vit_config.input_shape
        
        hidden_size = vit_config.hidden_size
        patch_content = c * self.p_y * self.p_x * self.p_t
        latent_content = latent_channels * self.p_y * self.p_x * self.p_t // (spatial_downsample ** 2) // temporal_downsample

        self.proj_in = nn.Linear(latent_content, hidden_size)
        self.blocks = StackedTransformer(
            vit_config.n_layers,
            vit_config.n_heads,
            vit_config.hidden_size
            )
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_content)

        self.n_p_y = h // self.p_y
        self.n_p_x = w // self.p_x
        self.n_p_t = t // self.p_t
    
    def depatchify(self, x):
        return eo.rearrange(
            x,
            'b (n_p_t n_p_y n_p_x) (p_t p_y p_x c) -> b (n_p_t p_t) c (n_p_y p_y) (n_p_x p_x)',
            p_y = self.p_y,
            p_x = self.p_x,
            p_t = self.p_t,
            n_p_y = self.n_p_y,
            n_p_x = self.n_p_x,
            n_p_t = self.n_p_t   
        )

    def forward(self, latent, output_hidden_states = False):
        x = self.proj_in(latent)
        
        h = []
        x = self.blocks(x, output_hidden_states = output_hidden_states)
        if output_hidden_states:
            x, h = x
        
        x = self.norm(x)
        x = self.proj_out(x)
        x = self.depatchify(x)

        if output_hidden_states:
            return x, h
        else:
            return x

class VideoVAE(MixIn):
    """
    Encoder and decoder with forward that computes KL loss
    See encoder documentation

    :param kl_weight: Weight of KL loss term (KL w.r.t N(0,1))
    """
    def __init__(
        self,
        vit_config,
        spatial_downsample = 8, temporal_downsample = 8,
        kl_weight = 1.0e-6
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            spatial_downsample, temporal_downsample,
            vit_config = vit_config
        )

        self.decoder = ViTDecoder(
            spatial_downsample, temporal_downsample,
            vit_config = vit_config
        )

        self.kl_weight = kl_weight

        self.n_patches = vit_config.num_patches
        self.hidden_size = vit_config.hidden_size

    def encode(self, pixel_values, output_hidden_states = False):
        return self.encoder(pixel_values, output_hidden_states = output_hidden_states)

    def decode(self, latent, output_hidden_states = False):
        return self.decoder(latent, output_hidden_states = output_hidden_states)
    
    def forward(self, pixel_values):
        dist = self.encode(pixel_values, output_hidden_states = False)
        z = dist.sample()
        rec = self.decode(z, output_hidden_states = False)

        rec_term = F.mse_loss(rec, pixel_values)
        kl_term = dist.kl().mean()

        loss = self.kl_weight * kl_term + rec_term
        return loss

# ==== ADVERSARIAL STUFF =====

class VideoDiscriminator(ViTEncoder):
    """
    Patch based discriminator that classifies individual patches of
    video as being real or spliced in from generated fake

    :param mixing_ratio: Probability of a fake patch replacing a real one
    :param *args: See encoder doc
    :param **kwargs: See encoder doc
    """
    def __init__(self, mixing_ratio = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proj_out = nn.Linear(self.hidden_size, 1)
        self.loss = nn.BCEWithLogitsLoss()

        self.mixing_ratio = mixing_ratio

    def classify(self, x):
        x = self.proj_in(x)
        x = self.blocks(x, output_hidden_states = False)
        x = self.norm(x)
        x = self.proj_out(x)
        return x

    def forward(self, real, fake):
        real = self.patchify(real)
        fake = self.patchify(fake)
        b,n,d = real.shape

        mask = torch.rand(b, n, device = real.device)
        patch_mask = eo.repeat(mask, 'b n -> b n d', d = d)

        splice = torch.where(patch_mask <= self.mixing_ratio, fake, real)
        labels = self.classify(splice).squeeze(-1) # [B,N]

        true_labels = torch.where(
            mask <= self.mixing_ratio,
            torch.zeros_like(labels),
            torch.ones_like(labels)
        )

        loss = self.loss(labels, true_labels)
        adv_loss = self.loss(labels, torch.ones_like(labels))

        return loss, adv_loss

class MultiDiscriminator(MixIn):
    """
    Multiple stacked discriminators that operate at different patch sizes.
    Careful with video patching.

    :param mixing_ratio: See discriminator docs
    :param scales: How many different discriminators to use
    :param args: See encoder docs
    """
    def __init__(
        self,
        mixing_ratio = 0.5, scales = 2,
        spatial_downsample = 8, temporal_downsample = 8,
        vit_config = ViTConfig()
    ):
        super().__init__()

        self.discs = nn.ModuleList([])

        patching = vit_config.patching

        for i in range(scales):
            patching = (
                patching[0] * 2 ** i,
                patching[1] * 2 ** i,
                patching[2] * 2 ** i,
            )
            self.discs.append(VideoDiscriminator(
                mixing_ratio,
                spatial_downsample, temporal_downsample,
                vit_config = vit_config
            ))
    
    def forward(self, real, fake):
        losses, adv_losses = 0.0, 0.0
        for disc in self.discs:
            loss, adv = disc(real, fake)
            losses = losses + loss
            adv_losses = adv_losses + adv
        
        return losses, adv_losses

from vae.nn.adversarial_vae import AdversarialVAE

    