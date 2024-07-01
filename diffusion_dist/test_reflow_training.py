"""
This script tests reflow model
"""

from .nn.reflow import Reflow
from common.configs import (
    ProjectConfig,
    TrainConfig,
    LoggingConfig
)
from common.trainer import Trainer
from common.sampling import Text2ImageSampler

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
    pipe.set_progress_bar_config(disable=True)

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
    dc = lambda x: {'text_features' : encode_text([x_i['Prompt'] for x_i in x])}

    model = Reflow(
        UNetWrapper(pipe),
        pipe.unet,
        ViTConfig(
            input_shape = (4, 96, 96)
        ),
        n_steps_loss = 25,
        teacher_sample_fn = call_pipe_fn
    )

    def model_sampling_fn(model, text):
        new_pipe = deepcopy(pipe)
        new_pipe.unet = model.student.model
        gen_1 = torch.Generator('cuda').manual_seed(0)
        gen_2 = torch.Generator('cuda').manual_seed(0)

        # Get both original images and images with RF model to compare directly
        return pipe(text[:len(text)//2], generator = gen_1).images + new_pipe(text[:len(text)//2], generator = gen_2).images
    
    def sample_prompts(size):
        return [ds[i]['Prompt'] for i in range(size)]

    config = ProjectConfig(
        train=TrainConfig(
            batch_size = 8,
            checkpoint_dir = "./diffusion_dist/checkpoints/sd2-2-rf",
            train_state_checkpoint = "./diffusion_dist/checkpoints/sd2-2-rf-train",
            sample_every = 50
        ),
        logging=LoggingConfig(
            "SD2.1 2-RF",
            "shahbuland",
            "reflow"
        )
    )
    
    trainer = Trainer(
        model,
        ds, dc,
        config = config,
        sampler = Text2ImageSampler(
            sample_prompts(4)*2,
            preprocessor = lambda x: x,
            postprocessor = lambda x: x
        ),
        model_sample_fn = model_sampling_fn
    )

    trainer.train()

    