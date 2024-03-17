from typing import Tuple
from dataclasses import dataclass

from common.trainer import Trainer
from common.configs import ProjectConfig

from .teacher_student import TeacherStudent
from .nn.vit_vae import ViTVAE
from .adapt_diffusers import HiddenStateAutoencoderKL
import torch


from diffusers import StableDiffusionPipeline

@dataclass
class ViTVAEConfig:
    n_layers : int = 8
    n_heads : int = 8
    hidden_size : int = 256

    patching : Tuple[int] = (32, 32)

if __name__ == "__main__":
    original_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionPipeline.from_pretrained(original_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    
    input_shape = (pipe.vae.config.in_channels, pipe.vae.config.sample_size, pipe.vae.config.sample_size)
    latent_shape = (pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)

    model_config = ViTVAEConfig()
    train_config = ProjectConfig()

    student = ViTVAE(
        model_config.patching, input_shape, latent_shape,
        model_config.n_layers, model_config.n_heads, model_config.hidden_size
    )

    teacher = pipe.vae
    teacher.float()

    model = TeacherStudent(
        teacher, student,
        input_shape, latent_shape
    )

    ds = None # TODO
    dc = None # TODO

    trainer = Trainer(
        model, ds, dc, config = train_config
    )





