from typing import Tuple

import torch
from torch import nn

from .mlp import MLP
from .normalization import RMSNorm
from ..utils import mimetic_init

import einops as eo

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
    def __init__(self, n_heads, dim, flash : bool = False):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias = False)
        mimetic_init(self.qkv, n_heads)

        if flash:
            from flash_attn import flash_attn_qkvpacked_func
            self.attn = flash_attn_qkvpacked_func
        else:
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first = True)
        self.flash = flash

        self.n_heads = n_heads
        
        self.d = dim
        self.ffn = MLP(dim, d_middle = 4 * dim)

        self.qk_norm = RMSNorm(2 * dim)
    
    def forward(self, x, attention_mask = None):
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        resid_1 = x.clone()
        x = self.norm1(x)

        qkv = self.qkv(x)

        if not self.flash:
            qk = qkv[...,:2*self.d].contiguous()
            v = qkv[...,2*self.d:].contiguous()
            qk = self.qk_norm(qk)

            q = qk[...,:self.d].contiguous()
            k = qk[...,self.d:].contiguous()
            attn_out = self.attn(q, k, v, key_padding_mask = attention_mask)[0]
        else:
            qk = qkv[...,:2*self.d].contiguous()
            qk = self.qk_norm(qk)
            qkv[...,:2*self.d] = qk

            qkv = eo.rearrange(qkv, 'b n (c h d) -> b n c h d', c = 3, h = self.n_heads, d = self.d//self.n_heads)
            attn_out = self.attn(qkv)
            attn_out = eo.rearrange(attn_out, 'b n h h_d -> b n (h h_d)')

        x = attn_out + resid_1
        resid_2 = x.clone()

        x = self.norm2(x)
        x = self.ffn(x)

        return x + resid_2

class StackedTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, dim, flash : bool = True):
        super().__init__()
        self.layers = nn.ModuleList([Transformer(n_heads, dim, flash = flash) for _ in range(n_layers)])

    def forward(self, x, attention_mask = None, output_hidden_states=False):
        h = []
        for layer in self.layers:
            x = layer(x, attention_mask)
            if output_hidden_states:
                h.append(x)

        if output_hidden_states:
            return x, h
        return x


