"""
This script tests reflow model
"""

from .nn.reflow import Reflow

from diffusers import StableDiffusionPipeline
from datasets import load_dataset

from torch.utils.data import DataLoader
from torch import nn
import einops as eo
import torch

from common.configs import ViTConfig

if __name__ == "__main__":
    from diffusers import StableDiffusionPipeline
    from copy import deepcopy

    pipe_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_id)
    pipe.to('cuda')

    def encode_text(text):
        tok_out = pipe.tokenizer(
            text,
            padding = 'max_length',
            max_length = 77,
            truncation = True,
            return_tensors = 'pt'
        ).to('cuda')
        text_features = pipe.text_encoder(**tok_out, output_hidden_states=True, return_dict = True).hidden_states[-1]
        text_features = pipe.text_encoder.text_model.final_layer_norm(text_features)
        return text_features
    
    neg_embed = encode_text([""]) # [1, 77, ...]

    def call_pipe_fn(text_features):
        # Call the standard pipeline __fn__ with text prompts (the features aren't used)
        noise = pipe.prepare_latents(
            len(text_features),
            4, 96, 96,
            text_features.dtype,
            text_features.device,
            None # no generator to pass
        )
        neg_embeds = eo.repeat(neg_embed, '1 ... -> b ...', b = len(text_features))

        # diffusers pipelines scale the noise like this, we should unscale before inputting
        noise_unscaled = noise / pipe.scheduler.init_noise_sigma
        pipe_out = pipe(
            latents = noise_unscaled, prompt_embeds = text_features, negative_prompt_embeds = neg_embeds,
            output_type = 'latent'
        ).images

        return pipe_out, noise

    # Simple wrapper for pipe unet to make it work here
    class UNetWrapper(nn.Module):
        def __init__(self, pipe):
            super().__init__()

            self.model = deepcopy(pipe.unet)
        
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs).sample

    # prompt dataset from hf hub
    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split = 'train')
    loader = DataLoader(ds, collate_fn = lambda x: encode_text([x_i['Prompt'] for x_i in x]), batch_size = 8)

    model = Reflow(
        UNetWrapper(pipe),
        pipe.unet,
        ViTConfig(
            input_shape = (4, 96, 96)
        ),
        n_steps_loss = 25,
        teacher_sample_fn = call_pipe_fn
    )
    opt = torch.optim.AdamW(model.parameters(), lr = 1.0e-4)

    for i, batch in enumerate(loader):
        opt.zero_grad()
        loss, metric = model(batch)
        loss.backward()
        opt.step()

        print(loss.item())
        print(metric)

        if i > 100:
            break

    