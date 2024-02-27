from torch import nn
import torch

# Timesteps

class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512):
        super().__init__()

        self.mlp = MLP(d_in, d_out)
        self.d = d_in # Assume this is even

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # t is [B] tensor of timesteps ()
        max_period = 10000 # This seems to always be assumed in all repos
        half = self.d // 2

        inds = torch.arange(half, device = t.device, dtype = t.dtype)
        freqs = (
            -math.log(max_period) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim = -1)

        return self.mlp(embs)

# Positional encoding for videos

class PositionalEncoding3D(nn.Module):
    def __init__(self, n_patches : Tuple, dim : int):
        super().__init__()

        n_patches_h, n_patches_w, n_patches_t = n_patches
        self.spatial_embeddings = nn.Parameter(
            torch.randn(n_patches_h * n_patches_w, dim)
        ) # These will embed spatial information

        self.temporal_embeddings = nn.Parameter(
            torch.randn(n_patches_t, dim)
        ) # These will embed temporal information

        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.n_patches_t = n_patches_t

    def forward(self, patch_embeds : TensorType["batch", "sequence", "dim"]):
        # Repeat space embeddings in a % fashion (1 2 3 4, 1 2 3 4, 1 2 3 4, etc.)
        space = eo.repeat(self.spatial_embeddings, 'hw d -> (t hw) d', t = self.n_patches_t)

        # Repeat time embeddings in a // fashion (1 1 1 1, 2 2 2 2, 3 3 3 3, etc.)
        time = eo.repeat(self.temporal_embeddings, 't d -> (t hw) d', hw = self.n_patches_h * self.n_patches_w)

        space_time = space + time
        space_time = space_time[None,:] # batch dim

        return patch_embeds + space_time