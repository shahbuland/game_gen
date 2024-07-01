from torch import nn
import torch
import torch.nn.functional as F
from torchtyping import TensorType

import einops as eo

from common.configs import ViTConfig
from common.nn.vit_modules import StackedTransformer

class LinearPool(nn.Module):
    """
    :param k: Compression/pooling factor
    :param d: Dimensionality of input embeddings
    """
    def __init__(self, k, d):
        super().__init__()

        self.k = k
        self.pool_layer = nn.Linear(k * d, d)
    
    def forward(self, seq : TensorType['b', 'n', 'd'], mask : TensorType['b', 'n']):
        """
        Sequence here is interchangable with either rows OR columns OR temporal patches.
        I.e. it's a sequence of patches under some specific view
        :param seq: Seq of patches with pad tokens, i.e. seq[i,j,k] is i-th batch element, j-th seq, k-th patch embedding in that seq
        :param mask: Attention mask
        """

        # Zero out any embeddings from the mask so they don't influence pool
        seq_len_old = seq.shape[1]
        seq = seq * mask.unsqueeze(-1)
        seq = eo.rearrange(seq, 'b (n k) d -> b n (k d)', k = self.k)
        seq = self.pool_layer(seq)
        seq_len_new = seq.shape[1]
        
        # NOTE: Factoring padding into the compression is likely non-trivial
        # If we don't factor it in, pooling approaches a limit, but we need to figure out which embeddings to mask out in the future
        # We will assume the mask ends in terminal 0's
        # If overall sequence shrunk by half [1 1 0 0] -> []
        # Overall number of 1's in the mask should be divided by self.k
        mask_sum = torch.sum(mask, dim=1)
        new_mask = torch.arange(mask.shape[1], device = mask.device)[None, :] < (mask_sum / self.k)[:, None]
        mask = new_mask.long()

        # Append pad tokens so overall size is not changed
        pad_emb = torch.zeros_like(seq[:, 0, :])
        pad_emb = eo.repeat(pad_emb, 'b d -> b n d', n = seq_len_old - seq_len_new)
        seq = torch.cat([seq, pad_emb], dim = 1)
        seq = seq * mask.unsqueeze(-1) # Zero out embeddings to ensure size is actually compressed
        
        return seq, mask

class LinearUpsample(nn.Module):
    """
    Opposite of downsample
    """
    def __init__(self, k, d):
        super().__init__()

        self.k = k
        self.pool_layer = nn.Linear(d, k * d)

    def forward(self, seq : TensorType['b', 'n', 'd'], mask : TensorType['b', 'n']):
        seq_len_old = seq.shape[1]
        seq = seq * mask.unsqueeze(-1)
        seq = self.pool_layer(seq)
        seq = eo.rearrange(seq, 'b n (k d) -> b (n k) d', k = self.k)

        # Double the ones in each sequence element without changing the size of the mask tensor
        mask_sum = torch.sum(mask, dim=1)
        new_mask = torch.arange(mask.shape[1], device = mask.device)[None, :] < (mask_sum * self.k)[:, None]
        mask = new_mask.long()

        seq = seq[:, :seq_len_old, :]
        mask = mask[:, :seq_len_old]

        return seq, mask

def unstack_patches(patches, vit_config):
    n_patches_each_dim = vit_config.n_patches_each
    if len(n_patches_each_dim) == 2:
        return eo.rearrange(
            patches,
            'b (n_y n_x) d -> b n_y n_x d',
            n_y = n_patches_each_dim[0],
            n_x = n_patches_each_dim[1]
        )
    elif len(n_patches_each_dim) == 3:
        return eo.rearrange(
            patches,
            'b (n_y n_x n_t) d -> b n_y n_x n_t d',
            n_y = n_patches_each_dim[0],
            n_x = n_patches_each_dim[1],
            n_t = n_patches_each_dim[2]
        )

def stack_patches(unstacked_patches, vit_config):
    n_patches_each_dim = vit_config.n_patches_each
    if len(n_patches_each_dim) == 2:
        return eo.rearrange(
            unstacked_patches,
            'b n_y n_x d -> b (n_y n_x) d',
            n_y = n_patches_each_dim[0],
            n_x = n_patches_each_dim[1]
        )
    elif len(n_patches_each_dim) == 3:
        return eo.rearrange(
            unstacked_patches,
            'b n_y n_x n_t d -> b (n_y n_x n_t) d',
            n_y = n_patches_each_dim[0],
            n_x = n_patches_each_dim[1],
            n_t = n_patches_each_dim[2]
        )

def focus_patch_dim(unstacked_patches : TensorType["b", "...", "d"], dim):
    """
    Given unstacked patches [b, n_patches_1, ..., n_patches_k, d]
    and some dim i to index n_patches_i,
    rolls all dimensions except i into the batch axis
    """
    n_dims = len(unstacked_patches.shape)-2

    # For extra info for einops
    dims_dict = {f'n_{i}' : unstacked_patches.shape[i+1] for i in range(n_dims)}

    pattern_left = 'b'
    for i in range(n_dims):
        pattern_left += f' n_{i}'
    pattern_left += ' d'

    # dim = 0 is batch, so realistically it'll start at 1,
    # but to make pattern work we'd want it to start at 0, so shift down one
    pattern_right = '(b'
    for i in range(n_dims):
        if i == dim-1:
            continue
        pattern_right += f' n_{i}'
    pattern_right += f') n_{dim-1} d'
    
    pattern = f'{pattern_left} -> {pattern_right}'
    return eo.rearrange(unstacked_patches, pattern), dims_dict

def unfocus_patch_dim(x, dim, dims_dict):
    """
    undoes the above operation. Needs dims_dict outputted from the above operation
    """
    n_dims = len(dims_dict)

    pattern_left = 'b'
    for i in range(n_dims):
        pattern_left += f' n_{i}'
    pattern_left += ' d'

    pattern_right = '(b'
    for i in range(n_dims):
        if i == dim-1:
            continue
        pattern_right += f' n_{i}'
    pattern_right += f') n_{dim-1} d'

    pattern = f'{pattern_right} -> {pattern_left}'
    return eo.rearrange(x, pattern, **dims_dict)

def masked_pool(x : TensorType["b","n","d"], mask : TensorType["b","n"]) -> TensorType["b", "d"]:
    """
    Mean pool of x while ignoring pad/masked tokens
    """
    # After sum we have [b,d]
    return (x * mask[...,None]).sum(1) / mask.sum(-1)[:,None]

class RoutedDimPool(nn.Module):
    """
    :param vit_config: Config for the overall ViT this router is going into.
    :param router_config: Config for the router network transformer.
    :param factors: Possible factors for compression
    """
    def __init__(self, vit_config : ViTConfig, router_config : ViTConfig = None, factors = [2,4], dim = 1):
        super().__init__()

        self.config = vit_config
        self.router_config = router_config
        self.factors = factors
        self.dim = dim

        self.poolers = nn.ModuleList([LinearPool(factor, vit_config.hidden_size) for factor in factors])
        self.router = StackedTransformer(
            router_config.n_layers,
            router_config.n_heads,
            router_config.hidden_size,
            flash = router_config.flash
        )
        self.router_final = nn.Linear(router_config.hidden_size, len(factors)+1)

        self.factor_embeddings = nn.Parameter(torch.randn(len(factors),vit_config.hidden_size))
        self.rescale_prob = 0.0 # Probability of doing rescaling
        # I speculate having no rescaling earlier in training is better, and we train further scales on a
        # cirriculum:
        # 1. No rescaling
        # 2. Rescaling with heavy bias towards lowest factors (factors are sampled from distribution that tails towards 0)
        # 3. Smart rescaling randomly trades off random rescaling
        # 4. Eventually we should shift towards always doing smart rescaling
        # 5. This is kind of an RL problem but we directly optimize? Is this possible? We will find out

    def forward(self, x : TensorType["b", "n", "d"], attention_mask : TensorType["b", "n"]):
        x = unstack_patches(x, self.config) # [B, n_y, n_x, d]
        mask = unstack_patches(attention_mask[:,:,None], self.config) # [B, n_y, n_x, 1]

        n_dims = len(x.shape)-2
        assert self.dim <= n_dims, 'Dim is outside valid dims for given patch setup'
        dim = self.dim + 1
 
        x, info = focus_patch_dim(x, dim) # [B, n, d]
        mask, _ = focus_patch_dim(mask, dim)

        mask = mask.squeeze(-1) # Remove the 1 from end
        router_logits = self.router(x, mask) 
        chosen_scales = self.router_final(router_logits) # [B, n, n_scales] # Choosing a scale for *every* row
        chosen_scales = chosen_scales.argmax(-1) # [B, n] labels of 0...n_scales-1

        def pooler_out(pooler, x, mask, emb):
            out, mask = pooler(x, mask)
            return out+emb, mask

        outputs_per_scale = [(x, mask)] + [
            pooler_out(self.poolers[i], x, mask, self.factor_embeddings[i])
            for i in range(len(self.factors))
        ]
        outputs_per_scale, masks_per_scale = zip(*outputs_per_scale)
        outputs_per_scale = torch.stack(outputs_per_scale, dim = 1) # [B, n_scales, n, d]
        masks_per_scale = torch.stack(masks_per_scale, dim = 1) # [B, n_scales, n]
        b, n_scales, n, d = outputs_per_scale.shape

        # Each number in this represents how much each batch element was downsampled
        # The normalized sum tells us how much compression this forward pass did,
        # where 0 is no compression, and 1 is complete compression
        # Should be divided by batch size elsewhere
        compression = chosen_scales / n_scales # score 0 -> 1 compression
        compression = masked_pool(compression.unsqueeze(-1), mask).squeeze(-1) # [B] average compression for each row

        chosen_scales = eo.repeat(chosen_scales, 'b n -> b 1 n d', d = d)

        outputs = outputs_per_scale.gather(1, chosen_scales).squeeze(1)
        masks = masks_per_scale.gather(1, chosen_scales[...,0]).squeeze(1)

        outputs = unfocus_patch_dim(outputs, dim, info)
        masks = unfocus_patch_dim(masks.unsqueeze(-1), dim, info)

        outputs = stack_patches(outputs, self.config)
        masks = stack_patches(masks, self.config).squeeze(-1)

        return outputs, masks, compression

class RoutedDimUpsample(nn.Module):
    """
    :param vit_config: Config for the overall ViT this router is going into.
    :param router_config: Config for the router network transformer.
    :param factors: Possible factors for compression
    """
    def __init__(self, vit_config : ViTConfig, router_config : ViTConfig = None, factors = [2,4], dim = 1):
        super().__init__()

        self.config = vit_config
        self.router_config = router_config
        self.factors = factors
        self.dim = dim

        self.upsamplers = nn.ModuleList([LinearUpsample(factor, vit_config.hidden_size) for factor in factors])

        self.router = StackedTransformer(
            router_config.n_layers,
            router_config.n_heads,
            router_config.hidden_size,
            flash = router_config.flash
        )
        self.router_final = nn.Linear(router_config.hidden_size, len(factors)+1)

        self.factor_embeddings = nn.Parameter(torch.randn(len(factors),vit_config.hidden_size))
        self.rescale_prob = 0.0 # Probability of doing rescaling

    def forward(self, x : TensorType["b", "n", "d"], attention_mask : TensorType["b", "n"]):
        x = unstack_patches(x, self.config) # [B, n_y, n_x, d]
        mask = unstack_patches(attention_mask.unsqueeze(-1), self.config) # [B, n_y, n_x, 1]

        n_dims = len(x.shape)-2
        assert self.dim <= n_dims, 'Dim is outside valid dims for given patch setup'
        dim = self.dim + 1
 
        x, info = focus_patch_dim(x, dim) # [B, n, d]
        mask, _ = focus_patch_dim(mask, dim)

        mask = mask.squeeze(-1) # Remove the 1 from end
        router_logits = self.router(x, mask) 
        chosen_scales = self.router_final(router_logits) # [B, n, n_scales] # Choosing a scale for *every* row
        chosen_scales = chosen_scales.argmax(-1) # [B, n] labels of 0...n_scales-1

        def upsampler_out(upsampler, x, mask, emb):
            out, mask = upsampler(x, mask)
            return out+emb, mask

        outputs_per_scale = [(x, mask)] + [
            upsampler_out(self.upsamplers[i], x, mask, self.factor_embeddings[i])
            for i in range(len(self.factors))
        ]
        outputs_per_scale, masks_per_scale = zip(*outputs_per_scale)
        outputs_per_scale = torch.stack(outputs_per_scale, dim = 1) # [B, n_scales, n, d]
        masks_per_scale = torch.stack(masks_per_scale, dim = 1) # [B, n_scales, n]
        b, n_scales, n, d = outputs_per_scale.shape

        # Each number in this represents how much each batch element was downsampled
        # The normalized sum tells us how much compression this forward pass did,
        # where 0 is no compression, and 1 is complete compression
        # Should be divided by batch size elsewhere
        expansion = chosen_scales / n_scales # score 0 -> 1 expansion
        expansion = masked_pool(expansion.unsqueeze(-1), mask).squeeze(-1) # [B] average compression for each row

        chosen_scales = eo.repeat(chosen_scales, 'b n -> b 1 n d', d = d)

        outputs = outputs_per_scale.gather(1, chosen_scales).squeeze(1)
        masks = masks_per_scale.gather(1, chosen_scales[...,0]).squeeze(1)

        outputs = unfocus_patch_dim(outputs, dim, info)
        masks = unfocus_patch_dim(masks.unsqueeze(-1), dim, info)

        outputs = stack_patches(outputs, self.config)
        masks = stack_patches(masks, self.config).squeeze(-1)

        return outputs, masks, expansion

class UpsampleIntoPadding(nn.Module):
    """
    Layer to upsample along a given dim to fill out any pad tokens.
    Dim order should be reverse of what pooling/upsampling is done along
    """
    def __init__(self, vit_config, dim):
        super().__init__()

        self.config = vit_config
        self.dim = dim
    
    def forward(self, x : TensorType["b", "n", "d"], attention_mask : TensorType["b", "n"]):
        # Get  focus on the dim that is being modified
        x = unstack_patches(x, self.config)
        mask = unstack_patches(attention_mask.unsqueeze(-1), self.config)

        n_dims = len(x.shape)-2
        assert self.dim <= n_dims, 'Dim is outside valid dims for given patch setup'
        dim = self.dim + 1
 
        x, info = focus_patch_dim(x, dim) # [B, n, d]
        mask, _ = focus_patch_dim(mask, dim)

        mask = mask.squeeze(-1)

        b, n, d = x.shape

        for i in range(b):
            for j in range(n):
                curr_size = mask[i,j].sum()
                req_size = n
                old_t = x[i,j,:curr_size]
                new_t = F.interpolate(old_t, size = [n, d], mode = 'linear', align_corners = False).squeeze(0)
                x[i,j] = new_t
        
        return x
        
if __name__ == "__main__":
    config = ViTConfig(flash=False)

    router_config = ViTConfig(
        2, flash = False
    )

    model = RoutedDimPool(config, router_config)
    model_2 = RoutedDimUpsample(config, router_config)
    final_up = UpsampleIntoPadding(config, 1)

    model.to('cuda')
    model_2.to('cuda')

    x = torch.randn(4, 64, 768, device = 'cuda')
    mask = torch.ones(4, 64, device = 'cuda').long()
    print(mask.sum())
    y, new_mask, compression = model(x, mask)
    print(compression)
    print(new_mask.sum())
    print(y.shape)

    z, final_mask, exp = model_2(y, new_mask)
    print(exp)
    print(final_mask.sum())
    print(z.shape)

    z *= final_mask[...,None]
    z = final_up(z, final_mask)

    print(z.shape)