from torch import nn
import torch

from .embeddings import TimestepEmbedding
from .modulation import MultipleModulationLayer, ModulationLayer
from .mlp import MLP

class DiTBlock(nn.Module):
    def __init__(self, n_heads, hidden_size):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape = hidden_size, elementwise_affine=False)
        self.act = nn.SiLU()

        self.context_modulation = MultipleModulationLayer(hidden_size)
        self.image_modulation = MultipleModulationLayer(hidden_size)

        self.i_fc_1 = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_fc_1 = nn.Linear(hidden_size, 3 * hidden_size)

        self.i_fc_2 = nn.Linear(hidden_size, hidden_size)
        self.c_fc_2 = nn.Linear(hidden_size, hidden_size)

        self.i_mlp = MLP(hidden_size, d_middle = 4 * hidden_size)
        self.c_mlp = MLP(hidden_size, d_middle = 4 * hidden_size)

        self.d = hidden_size
        
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first = True)

    def forward(self, pixel_values, encoder_hidden_states, conditioning):
        x = pixel_values
        c = encoder_hidden_states
        y = conditioning

        b, n, d = x.shape
        _, n_cross, _ = c.shape
        
        x_resid = x
        c_resid = c

        x = self.norm(x)
        c = self.norm(c)

        t_emb_i = self.image_modulation(y)
        t_emb_c = self.context_modulation(y)

        x = self.image_modulation.mod(x, t_emb_i, 0)
        c = self.context_modulation.mod(c, t_emb_c, 0)

        x = self.i_fc_1(x)
        c = self.c_fc_1(c)

        attention_inputs = torch.cat([x, c], dim = 1)
        query = attention_inputs[:,:,:self.d]
        key = attention_inputs[:,:,self.d:self.d*2]
        value = attention_inputs[:,:,self.d*2:]

        attn_out = self.attn(query, key, value)[0]

        x = self.i_fc_2(attn_out[:,:n])
        c = self.c_fc_2(attn_out[:,n:])

        x = self.image_modulation.mod(x, t_emb_i, 1) + x_resid
        c = self.context_modulation.mod(c, t_emb_c, 1) + c_resid

        x = self.norm(x)
        c = self.norm(c)

        x = self.image_modulation.mod(x, t_emb_i, 2)
        c = self.context_modulation.mod(c, t_emb_c, 2)

        x = self.i_mlp(x)
        c = self.c_mlp(c)

        x = self.image_modulation.mod(x, t_emb_i, 3) + x_resid
        c = self.context_modulation.mod(c, t_emb_c, 3) + c_resid

        return x, c, y

class DiTModelBase(nn.Module):
    def __init__(
            self,
            encoder_hidden_size, n_heads, hidden_size, n_layers
        ):
        super().__init__()
        
        self.cond_embed = TimestepEmbedding(hidden_size)

        self.first_cond_fc = nn.Linear(encoder_hidden_size, hidden_size)

        self.blocks = nn.ModuleList([DiTBlock(n_heads, hidden_size) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(normalized_shape = hidden_size, elementwise_affine=False)
        self.modulation = ModulationLayer(hidden_size)

    def forward(self, patched_images, encoder_hidden_states, cond):
        x = patched_images
        c = encoder_hidden_states

        c = self.first_cond_fc(c)
        y = self.cond_embed(cond)

        for block in self.blocks:
            x, c, y = block(x, c, y)

        x = self.final_norm(x)
        x = self.modulation(x, y)
        return x
