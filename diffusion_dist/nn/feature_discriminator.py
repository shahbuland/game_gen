from common.nn.vit_modules import StackedTransformer

import torch
from torch import nn
import einops as eo

class DiscriminatorHead(nn.Module):
    """
    Head for feature discriminator. Forward mode allows for only fakes to be passed in.

    :param d_model: Dim for the features who are being input into this model
    :param d_inner: Feature dim for the inner layers of this discriminator
    """
    def __init__(self, d_model, vit_config):
        super.__init__()

        self.proj_in = nn.Linear(d_model, vit_config.hidden_size)
        self.norm = nn.LayerNorm(vit_config.hidden_size)
        self.blocks = StackedTransformer(
            vit_config.n_layers,
            vit_config.n_heads,
            vit_config.hidden_size
        )
        self.proj_out = nn.Linear(vit_config.hidden_size, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def classify(self, x):
        x = self.proj_in(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj_out(x)
        return x

    def forward(self, real_features, fake_features = None):
        # If only one arg is passed, assume its fake images for adv loss
        if fake_features is None:
            x = None
            y = real_features
        else:
            x = real_features
            y = fake_features

        if x is not None:
            real_labels = self.classify(x)
            real_loss = self.loss(real_labels, torch.ones_like(real_labels))
        else:
            real_loss = 0.0

        fake_labels = self.classify(y)
        fake_loss = self.loss(fake_labels, torch.zeros_like(fake_labels))
        
        loss = real_loss + fake_loss
        adv_loss = -1 * fake_labels.mean() # Maximize fake labels

        return loss, adv_loss
    
class MixingDiscriminatorHead(DiscriminatorHead):
    """
    Same as above but for mixing patches approach
    """
    def __init__(self, d_model, vit_config, mixing_ratio = 0.5):
        super().__init__(d_model, vit_config)

        self.mixing_ratio = mixing_ratio

    def forward(self, real_features, fake_features):
        b,n,d = real_features.shape

        mask = torch.rand(b, n, device = real_features.device)
        patch_mask = eo.repeat(mask, 'b n -> b n d', d = d)

        splice = torch.where(patch_mask <= self.mixing_ratio, fake_features, real_features)
        labels = self.classify(splice).squeeze(-1) # [B, N]

        true_labels = torch.where(
            mask <= self.mixing_ratio,
            torch.zeros_like(labels),
            torch.ones_like(labels)
        )

        loss = self.loss(labels, true_labels)
        adv_loss = -1 * labels.mean()

        return loss, adv_loss



