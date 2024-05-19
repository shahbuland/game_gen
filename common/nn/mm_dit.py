from torch import nn
import torch
import einops as eo

from ..configs import ViTConfig
from .modulation import MMDiTModulation, ModulationLayer
from .normalization import RMSNorm
from .mlp import MLP
from .vit_modules import PositionalEncoding
from .embeddings import TimestepEmbedding

from flash_attn import flash_attn_qkvpacked_func
from common.utils import mimetic_init

class MMDiTBlock(nn.Module):
    """
    Main block for MMDiT architecture. See the SD3 paper for more details.
    """
    def __init__(self, n_heads, hidden_size, flash : bool = True):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine = False)
        self.act = nn.SiLU()

        # c => context (text embeddings)
        # i => media (image) features
        self.c_mod = MMDiTModulation(hidden_size)
        self.i_mod = MMDiTModulation(hidden_size)

        self.c_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias = False)
        self.i_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias = False)

        mimetic_init(self.c_qkv, n_heads)
        mimetic_init(self.i_qkv, n_heads)
        
        self.c_fc = nn.Linear(hidden_size, hidden_size)
        self.i_fc = nn.Linear(hidden_size, hidden_size)

        self.c_mlp = MLP(hidden_size, d_middle = 4 * hidden_size)
        self.i_mlp = MLP(hidden_size, d_middle = 4 * hidden_size) 

        self.d = hidden_size
        self.n_heads = n_heads

        if flash:
            self.attn = flash_attn_qkvpacked_func
        else:
            self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first = True)
        self.flash = flash

        self.q_norm = RMSNorm(hidden_size)
        self.k_norm = RMSNorm(hidden_size)
    
    def forward(self, pixel_values, encoder_hidden_states, conditioning):
        # variable names from SD3 paper
        x = pixel_values
        c = encoder_hidden_states
        y = conditioning

        b,n,d = x.shape
        _, n_cross, _ = c.shape

        c_resid = c.clone()
        x_resid = x.clone()

        c = self.norm(c)
        x = self.norm(x)

        # Modulation parameters given t for both image and context
        t_emb_c = self.c_mod(y)
        t_emb_i = self.i_mod(y)

        c = self.c_mod.mod(c, t_emb_c, 0)
        x = self.i_mod.mod(x, t_emb_i, 0)

        qkv = torch.cat([self.c_qkv(c), self.i_qkv(x)], dim = -2) # cat on sequence dim

        if not self.flash:
            q, k, v = qkv.chunk(3, dim = -1)
            # Normalize the querys and keys
            q = self.q_norm(q)
            k = self.k_norm(k)
            attn_out = self.attn(q, k, v)[0]
        else:
            #q = qkv[...,:self.d].contiguous()
            #k = qkv[...,self.d:2*self.d].contiguous()
            #q = self.q_norm(q)
            #k = self.k_norm(k)
            #qkv[...,:self.d] = q
            #qkv[...,self.d:2*self.d] = k

            qkv = eo.rearrange(qkv, 'b n (c h d) -> b n c h d', c = 3, h = self.n_heads, d = self.d//self.n_heads)
            attn_out = flash_attn_qkvpacked_func(qkv)
            attn_out = eo.rearrange(attn_out, 'b n h h_d -> b n (h h_d)')
        
        c = attn_out[:,:n_cross]
        x = attn_out[:,n_cross:]

        c = self.c_fc(c)
        x = self.i_fc(x)

        c = self.c_mod.mod(c, t_emb_c, 1) + c_resid
        x = self.i_mod.mod(x, t_emb_i, 1) + x_resid

        c = self.norm(c)
        x = self.norm(x)

        c = self.c_mod.mod(c, t_emb_c, 2)
        x = self.i_mod.mod(x, t_emb_i, 2)

        c = self.c_mlp(c)
        x = self.i_mlp(x)

        c = self.c_mod.mod(c, t_emb_c, 3) + c_resid
        x = self.i_mod.mod(x, t_emb_i, 3) + x_resid

        return x, c, y

class MMDiT(nn.Module):
    """
    MMDiT with all blocks/layers. See SD3 paper for more details.

    :param encoder_hidden_size: hidden_size for text encoder typically
    """
    def __init__(self, encoder_hidden_size, config : ViTConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.t_embed = TimestepEmbedding(self.hidden_size)
        self.cond_proj = nn.Linear(encoder_hidden_size, self.hidden_size)
        self.blocks = nn.ModuleList([MMDiTBlock(config.n_heads, config.hidden_size, config.flash) for _ in range(config.n_layers)])

        self.final_norm = nn.LayerNorm(self.hidden_size, elementwise_affine = False)
        self.modulation = ModulationLayer(self.hidden_size)
    
    def forward(self, patched_images, encoder_hidden_states, t_cond, output_hidden_states = False):
        x = patched_images
        c = encoder_hidden_states

        c = self.cond_proj(c)
        y = self.t_embed(t_cond)

        h = []
        for block in self.blocks:
            x, c, y = block(x, c, y)
            if output_hidden_states:
                h.append(x)
        
        x = self.final_norm(x)
        x = self.modulation(x, y)

        if output_hidden_states:
            return x, h
        else:
            return x

