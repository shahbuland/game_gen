from torchtyping import TensorType

from .dit_blocks import DiTBlock, DiTModelBase
from .embeddings import PositionalEncoding3D

from torch import nn
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
import einops as eo

class DiTVideo(nn.Module):
    """
    Difference between this and the base is that assumes a video input directly. Above works like LLM.
    """
    def __init__(self, input_size, patch_size, encoder_hidden_size, n_heads, hidden_size, n_layers):
        super().__init__()

        # For videos, input size is assumed to be 
        # (n, c, h, w)

        # patch_size should be
        # (height_patch, width_patch, temporal_patch) 
        # We are gonna assume height_patch = width_patch

        # Compute n_patches first

        self.scheduler = DDIMScheduler(num_train_timesteps = 1000)#, prediction_type = "v_prediction")

        n_temporal_patches = input_size[0] // patch_size[-1]
        n_image_patches = input_size[2] // patch_size[0]
        self.n_patches = n_image_patches**2 * n_temporal_patches

        # Now find dimensionality of each patch (i.e. how many pixels in it?)
        self.patch_content = patch_size[0] * patch_size[1] * patch_size[2] * 3

        self.first_image_fc = nn.Linear(self.patch_content, hidden_size)
        self.pos_enc_3d = PositionalEncoding3D((n_image_patches, n_image_patches, n_temporal_patches), hidden_size)

        self.model = DiTModelBase(
            encoder_hidden_size = encoder_hidden_size,
            n_heads = n_heads,
            hidden_size = hidden_size,
            n_layers = n_layers
        )

        #self.final_fc = nn.Linear(hidden_size, 2 * self.patch_content)
        self.final_fc = nn.Linear(hidden_size, self.patch_content)
        self.n_patches_each = [n_image_patches, n_image_patches, n_temporal_patches]
        self.patch_content_each = patch_size

        #self.vb_loss = VBLoss(self.scheduler)

    @property
    def device(self):
        return self.first_image_fc.weight.device
    
    def patchify(self, x : TensorType["batch", "n_frames", "channels", "height", "width"]):
        # This is a bit tricky
        x = eo.rearrange(
            x,
            'b (t p_t) c (h p_h) (w p_w) -> b (t h w) (p_t p_h p_w c)',
            t = self.n_patches_each[-1],
            h = self.n_patches_each[0],
            w = self.n_patches_each[1]
        )
        return x

    def depatchify(self, x : TensorType["batch", "seq", "patch_content"]):
        x = eo.rearrange(
            x,
            'b (t h w) (p_t p_h p_w c) -> b (t p_t) c (h p_h) (w p_w)',
            t = self.n_patches_each[-1],
            h = self.n_patches_each[0],
            w = self.n_patches_each[1],
            p_t = self.patch_content_each[-1],
            p_h = self.patch_content_each[0],
            p_w = self.patch_content_each[1]
        )
        return x

    def denoise(self, pixel_values, encoder_hidden_states, cond):
        patches = self.patchify(pixel_values)
        x = self.first_image_fc(patches)
        x = self.pos_enc_3d(x)

        x = self.model(x, encoder_hidden_states, cond)
        x = self.final_fc(x)

        #sigma = x[:,:,self.patch_content:]
        #x = x[:,:,:self.patch_content]

        x = self.depatchify(x)
        #sigma = self.depatchify(sigma)

        return x#, sigma
    
    def predict(self, sample, t, embeds):
        return self.denoise(sample, embeds, t)
    
    def sample_t(self, batch_size): 
        return torch.randint(0, self.scheduler.config.num_train_timesteps, size = (batch_size,),)

        # Choosing values closer to center (ignore for now)
        t = torch.randn(batch_size) # ~ N(0,1)
        t = t * (0.25/2) # Makes the approximate range [-0.5, 0.5]
        t += 0.5 # [0, 1]
        t = t.clamp(0,1)
        t = (t * 1000).long()
        return t

    def forward(self, pixel_values, encoder_hidden_states):
        # Generate timesteps randomly
        cond = self.sample_t(len(pixel_values)).to(pixel_values.device)

        # Use them to noise the pixel_values
        z = torch.randn_like(pixel_values)
        noisy_sample = self.scheduler.add_noise(
            pixel_values, z, cond
        )
        pred = self.denoise(noisy_sample, encoder_hidden_states, cond)

        #target = self.scheduler.get_velocity(pixel_values, z, cond)
        target = z
        simple_loss = F.mse_loss(pred, target)
        
        #frozen_out = torch.cat([pred.detach()
        return simple_loss
