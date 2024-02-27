from typing import Tuple, Dict

from dataclasses import dataclass, field

@dataclass
class DiTConfig:
    n_layers : int = 24
    hidden_size : int = 1024
    heads : int = 16
    
    context_dim : int = 768
    patch_size : Tuple[int] = (4, 4, 10) # (h,w, video)
    
    channels : int = 3
    img_size : int = 32
    total_frames : int = 100
    fps : int = 10

@dataclass
class TrainConfig:
    batch_size : int = 1
    target_batch : int = 128
    num_workers : int = 1
    epochs : int = 10

    save_every : int = 1000
    sample_every : int = 1000
    sample_prompt : str = "A bright blue sky"

    checkpoint_dir : str = "./dit_out"
    train_state_checkpoint : str = "./trainer_state"
    resume = False

    ds_mode : str = "train" # train (or val for debug)
    ds_path : str = "../webvid"

    cfg_prob : float = 0.1

    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 2.0e-4,
        "weight_decay": 0.05,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    })

@dataclass
class LoggingConfig:
    run_name : str = None
    wandb_entity : str = None
    wandb_project : str = None

@dataclass
class ProjectConfig:
    model : DiTConfig = DiTConfig()
    logging : LoggingConfig = LoggingConfig()
    train : TrainConfig = TrainConfig()