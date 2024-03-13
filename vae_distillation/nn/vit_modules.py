from typing import Tuple

import torch
from torch import nn

from dit_videos.nn.mlp import MLP

class PositionalEncoding(nn.Module):
    def __init__(self, n_patches : Tuple, dim : int):
        super().__init__()

        if len(n_patches) == 3:
            n_patches_h, n_patches_w, n_patches_t = n_patches
            prod = n_patches_h * n_patches_w * n_patches_t
        else:
            n_patches_h, n_patches_w = n_patches
            prod = n_patches_h * n_patches_w

        self.embedding = nn.Parameter(
            torch.randn(prod, dim)
        )

    def forward(self, patch_embeds):
        return patch_embeds + self.embedding[None,:]

class Transformer(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, 3 * dim)

        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first = True)
        self.d = dim
        self.ffn = MLP(dim, d_middle = 4 * dim)
    
    def forward(self, x):
        resid_1 = x.clone()
        x = self.norm1(x)

        qkv = self.qkv(x)
        q = qkv[...,:self.d]
        k = qkv[...,self.d:2*self.d]
        v = qkv[...,2*self.d:]

        attn_out = self.attn(q, k, v)[0]

        x = attn_out + resid_1
        resid_2 = x.clone()

        x = self.norm2(x)
        x = self.ffn(x)

        return x + resid_2

class StackedTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, dim):
        super().__init__()
        self.layers = nn.ModuleList([Transformer(n_heads, dim) for _ in range(n_layers)])

    def forward(self, x, output_hidden_states=False):
        h = []
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                h.append(x)

        if output_hidden_states:
            return x, h
        return x


