from typing import Tuple
from dataclasses import dataclass

from game_gen.trainer import Trainer
from game_gen.configs import ProjectConfig

from .nn.vit_vae import ViTVAE
import torch

@dataclass
class ViTVAEConfig:
    n_layers : int = 8
    n_heads : int = 8
    hidden_size : int = 256

    patching : Tuple[int] = (32, 32)

if __name__ == "__main__":

    input_shape = (3, 512, 512)
    latent_shape = (4, 64, 64)

    model_config = ViTVAEConfig()
    train_config = ProjectConfig()

    model = ViTVAE(
        model_config.patching, input_shape, latent_shape,
        model_config.n_layers, model_config.n_heads, model_config.hidden_size
    )
    
    ds = None # TODO
    dc = None # TODO

    trainer = Trainer(
        model, ds, dc, config = train_config
    )