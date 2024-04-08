from ..modeling import MixIn
from .mm_dit import MMDiT
from .vit_modules import PositionalEncoding
from ..utils import sample_lognorm_timesteps, rectflow_lerp, list_prod

import torch
import torch.nn.functional as F

class ConditionedRectFlowTransformer(MixIn):
    """
    Conditioned rectified flow transformer on MMDiT architecture

    :param vit_config: Config used to create MMDiT
    :param text_encoder: Generally just assumed to be something like CLIP. It will be assumed it has .config.hidden_size or .hidden_size
    :param vae: Optionally give a vae. If given, will use for predictions/decoding/encoding/etc.
    """
    def __init__(self, vit_config, text_encoder, vae = None):
        super().__init__()

        encoder_hidden = None
        try:
            encoder_hidden = text_encoder.config.hidden_size
        except AttributeError:
            try:
                text_encoder = text_encoder.hidden_size
            except AttributeError:
                raise ValueError("Can't figure out text encoder hidden size for ConditionedDenoiser")

        self.model = MMDiT(encoder_hidden, vit_config)
        self.text_encoder = text_encoder
        self.vae = vae
        self.config = vit_config

        # Image -> Transformer stuff
        patch_content = list_prod(vit_config.patching) * vit_config.input_shape[-3] # * channels
        self.project_patches = nn.Linear(patch_content, vit_config.hidden_size)
        self.pos_enc = PositionalEncoding(vit_config.patching, vit_config.hidden_size)
        self.unproject_patches = nn.Linear(vit_config.hidden_size, patch_content)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        embeds = self.text_encoder(input_ids, attention_mask, output_hidden_states = True)[0]
        return embeds
    
    def predict(self, perturbed, input_ids, attention_mask, timesteps):
        """
        Gets a prediction for the velocity given some parameters

        :param perturbed: Pixels for the perturbed image/video/media
        :param input_ids: Input ids for text conditioning
        :param attention_mask: Attention mask for text conditioning
        :param timesteps: Diffusion timesteps
        """
        encoder_hidden_states = self.encode_text(input_ids, attention_mask)

        x = self.patchify(perturbed)
        x = self.project_patches(x)
        x = self.pos_enc(x)

        x = self.model(x, encoder_hidden_states, timesteps)
        x = self.unproject_patches(x)
        x = self.depatchify(x)

        return x

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Call this for training. Samples timesteps and computes loss for given samples and input_ids/attention_mask
        """
        timesteps = sample_lognorm_timesteps(pixel_values)

        z = torch.randn_like(pixel_values)
        x = rectflow_lerp(pixel_values, z, timesteps)

        pred = self.predict(x, input_ids, attention_mask, timesteps)
        loss = F.mse_loss(pred, pixel_values - z)
        
        return loss

    def patchify(self, x):
        patching = self.config.patching

        if len(patching) == 2: # Images
            p_y, p_x = patching
            return eo.rearrange(
                x,
                'b c (n_p_y p_y) (n_p_x p_x) -> b (n_p_y n_p_x) (p_y p_x c)',
                p_y = p_y,
                p_x = p_x
            )
        elif len(patching) == 3: # Videos
            p_y, p_x, p_t = patching
            return eo.rearrange(
                x,
                'b (n_p_t p_t) c (n_p_y p_y) (n_p_x p_x) -> b (n_p_t n_p_y n_p_x) (p_t p_y p_x c)',
                p_y = p_y,
                p_x = p_x,
                p_t = p_t
            )
        else:
            raise ValueError("Undefined patching dimensions (must be (Y,X) for images and (Y,X,T) for videos)")

    def depatchify(self, x):
        input_shape = self.config.input_shape
        patching = self.config.patching

        if len(patching) == 2: # Images
            p_y, p_x = patching
            h, w = input_shape[-2:]
            n_p_y = h // p_y
            n_P_x = w // p_x

            return eo.rearrange(
                x,
                'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
                p_y = p_y,
                p_x = p_x,
                n_p_y = n_p_y,
                n_p_x = n_p_x
            )
        elif len(patching) == 3: # Videos
            p_y, p_x, p_t = patching
            h, w = input_shape[-2:]
            t = input_shape[0]
            n_p_y = h // p_y
            n_p_x = w // p_x
            n_p_t = t // p_t

            return eo.rearrange(
                x,
                'b (n_p_t n_p_y n_p_x) (p_t p_y p_x c) -> b (n_p_t p_t) c (n_p_y p_y) (n_p_x p_x)',
                p_y = p_y,
                p_x = p_x,
                p_t = p_t,
                n_p_y = n_p_y,
                n_p_x = n_p_x,
                n_p_t = n_p_t
            )
        else:
            raise ValueError("Undefined patching dimensions (must be (Y,X) for images and (Y,X,T) for videos)")
