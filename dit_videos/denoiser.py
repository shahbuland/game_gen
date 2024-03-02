from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
import torch
import einops as eo

class Denoiser(nn.Module):
    """
    Core Denoiser object. Wraps around any other model that goes inside it.
    """
    def __init__(self, base_model_cls, *args, **kwargs):
        super().__init__()
        clip_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.embedder = CLIPTextModel.from_pretrained(clip_id)
        self.embedder.requires_grad = False

        self.core_model = base_model_cls(*args, **kwargs)
        self.scheduler = self.core_model.scheduler

        # precompute negative embedding
        self.negative_embed = self.encode_text(text="")

    def get_negative(self, batch_size):
        return eo.repeat(self.negative_embed, '1 ... -> b ...', b = batch_size)

    @property
    def device(self):
        return self.core_model.device

    def save(self, path):
        torch.save(self.core_model.state_dict(), path)
    
    def load(self, path):
        self.core_model.load_state_dict(torch.load(path))
    
    def forward(self, *args, **kwargs):
        return self.core_model(*args, **kwargs)
    
    def predict(self, sample, t, embeds):
        return self.core_model.predict(sample, t, embeds)
    
    @torch.no_grad()
    def encode_text(self, input_ids = None, attention_mask = None, text = None):
        if text is not None:
            tok_out = self.tokenizer(text, return_tensors = "pt", padding = "max_length", max_length = 77)
            input_ids = tok_out.input_ids
            attention_mask = tok_out.attention_mask

        input_ids = input_ids.to(self.embedder.device)
        attention_mask = attention_mask.to(self.embedder.device)
        embeds = self.embedder(input_ids, attention_mask, output_hidden_states = True)[0]
        return embeds
