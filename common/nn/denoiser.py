from abc import abstractmethod

from ..modeling import MixIn
from .mm_dit import MMDiT
from .vit_modules import PositionalEncoding
from ..utils import sample_lognorm_timesteps, rectflow_lerp, list_prod, n_patches

import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

def peek_hidden_size(model):
    try:
        hidden_size = model.hidden_size
    except:
        try:
            hidden_size = model.config.hidden_size
        except:
            raise ValueError(f"Couldn't figure out hidden size from {model}")
    return hidden_size

class CLIPConditioner(nn.Module):
    """
    CLIP-like conditioning model

    :param core_model: Core model whose hidden states we use for conditioning
    :param tokenizer: Optional tokenizer for the core model
    :param layer_skip: Returns this index from the hidden layers. Defaults to last layer (-1), sometimes second last is better (-2)
    :param hidden_size: Optional hidden size for the core model. Tries to infer from the model if this isn't given.
    """
    def __init__(self, core_model, tokenizer = None, layer_skip = -1, hidden_size = None):
        super().__init__()

        self.layer_skip = layer_skip
        self.tokenizer = tokenizer
        self.model = core_model
        self.hidden_size = hidden_size

        if self.hidden_size is None:
            self.hidden_size = peek_hidden_size(self.model)

    @torch.no_grad()
    def forward(self, input_ids = None, attention_mask = None, text = None):
        if text is not None:
            assert self.tokenizer is not None, "Can't encode text directly unless conditioner was initialized with a tokenizer."
            tok_out = self.tokenizer(text, return_tensors = "pt", padding = 'max_length', max_length = 77, truncation = True).to(self.model.device)
            return self.forward(**tok_out)
        else:
            _, _, h = self.model(input_ids, attention_mask, output_hidden_states = True, return_dict = False)
            return h[self.layer_skip]

class ConditionedDenoiser(MixIn):
    """
    Base class for any conditioned denoiser

    :param config: Generic config to store any info about model
    :param text_encoder: Text encoder for conditioning signal
    :param vae: Optional VAE to use for encoding/decoding
    """
    def __init__(self, config, text_encoder, vae = None):
        super().__init__()

        self.text_encoder = text_encoder
        self.encoder_hidden_size = peek_hidden_size(self.text_encoder)
        self.config = config
        self.vae = vae

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        """
        Encode text with text encoder from tokenizer output into embeddings for conditioning signal
        """
        return self.text_encoder(input_ids, attention_mask)

    @abstractmethod
    def predict(self, noisy, text_features, timesteps):
        """
        Get model prediction from noisy/perturbed sample, text encoder inputs and timesteps
        """
        pass

    @abstractmethod
    def forward(self, pixel_values, input_ids, attention_mask):
        pass

class ConditionedRectFlowTransformer(ConditionedDenoiser):
    """
    Conditioned rectified flow transformer on MMDiT architecture

    :param config: vit_config used to create MMDiT
    :param text_encoder: Generally just assumed to be something like CLIP. It will be assumed it has .config.hidden_size or .hidden_size
    :param vae: Optionally give a vae. If given, will use for predictions/decoding/encoding/etc.
    """
    def __init__(self, config, text_encoder, vae = None):
        super().__init__(config, text_encoder, vae)

        self.model = MMDiT(self.encoder_hidden_size, self.config)

        # Image -> Transformer stuff
        patch_content = list_prod(self.config.patching) * self.config.input_shape[-3] # * channels
        self.project_patches = nn.Linear(patch_content, self.config.hidden_size)
        self.pos_enc = PositionalEncoding(n_patches(config), self.config.hidden_size)
        self.unproject_patches = nn.Linear(self.config.hidden_size, patch_content)
    
    def predict(self, perturbed, text_features, timesteps, output_hidden_states = False):
        """
        Gets a prediction for the velocity given some parameters

        :param perturbed: Pixels for the perturbed image/video/media
        :param input_ids: Input ids for text conditioning
        :param attention_mask: Attention mask for text conditioning
        :param timesteps: Diffusion timesteps
        """

        x = self.patchify(perturbed)
        x = self.project_patches(x)
        x = self.pos_enc(x)

        if output_hidden_states:
            x, h = self.model(x, text_features, timesteps, output_hidden_states = output_hidden_states)
        else:
            x = self.model(x, text_features, timesteps, output_hidden_states = output_hidden_states)
        x = self.unproject_patches(x)
        x = self.depatchify(x)

        if output_hidden_states:
            return x, h
        else:
            return x

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Call this for training. Samples timesteps and computes loss for given samples and input_ids/attention_mask
        """
        timesteps = sample_lognorm_timesteps(pixel_values)

        z = torch.randn_like(pixel_values)
        x = rectflow_lerp(pixel_values, z, timesteps)

        text_features = self.encode_text(input_ids, attention_mask)

        pred = self.predict(x, text_features, timesteps)
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
            n_p_x = w // p_x

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

if __name__ == "__main__":
    # Forward pass test
    
    import einops as eo

    input_ids = torch.arange(77)
    input_ids = eo.repeat(input_ids, 'n -> b n', b = 4).cuda()
    attention_mask = torch.randint(2, (77,), dtype=torch.long)
    attention_mask = eo.repeat(attention_mask, 'n -> b n', b = 4).cuda()
    samples = torch.randn(4, 3, 32, 32).cuda()

    from common.configs import ViTConfig
    from transformers import CLIPTokenizer, CLIPTextModel

    clip_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    clip_lm = CLIPTextModel.from_pretrained(clip_id)

    text_encoder = CLIPConditioner(clip_lm, tokenizer, layer_skip = -2, hidden_size = 512)

    config = ViTConfig(
        n_layers = 12,
        n_heads = 12,
        hidden_size = 768,
        input_shape = (3, 32, 32),
        patching = (4, 4)
    )

    denoiser = ConditionedRectFlowTransformer(
        config,
        text_encoder
    )
    denoiser.cuda()

    loss = denoiser(samples, input_ids, attention_mask)
    print(loss.item())
