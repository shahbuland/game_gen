import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from dit_videos.nn.mlp import MLP

class ViTFeatureProjector(nn.Module):
    """
    Projects ViT features into conv feature space
    """
    def __init__(self, vit_n_patches, vit_d, conv_f, conv_h, conv_w):
        super().__init__()
        # [b, vit_n_patches, vit_d]
        # [b, conv_f, conv_h, conv_w]

        # Assume square shapes
        # Going backwards,
        # (conv_h // p) * (conv_w // p) = n_patches
        # => p = round(((conv_h * conv_w)/n_patches)**.5)

        self.p = round(((conv_h*conv_w)/vit_n_patches)**.5)
        self.d = conv_f * self.p**2

        self.proj = MLP(vit_d, d_out = self.d)

        self.n_p_y = conv_h // self.p
        self.n_p_x = conv_w // self.p

        self.conv_info = (conv_f, conv_h, conv_w)

    def depatchify(self, x):
        return eo.rearrange(
            x,
            'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
            n_p_y = self.n_p_y,
            n_p_x = self.n_p_x,
            p_y = self.p,
            p_x = self.p
        )

    def forward(self, x, conv_features = None):
        x = self.proj(x)
        x = self.depatchify(x)

        if conv_features is not None:
            return F.l1_loss(x, conv_features)
        else:
            return x
