from typing import Tuple, Dict, List

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
    batch_size : int = 16
    target_batch : int = 256# // 8 # Divide by num_processes
    num_workers : int = 1 # Using more than 1 results in repeated batches
    epochs : int = 10

    save_every : int = 1000
    sample_every : int = 100

    checkpoint_dir : str = "./dit_out"
    train_state_checkpoint : str = "./trainer_state"
    resume = True

    ds_mode : str = "train" # train (or val for debug)
    ds_path : str = "../webvid"

    cfg_prob : float = 0.1

    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    })

    sample_prompts : List = field(default_factory = lambda : [
        "a warm sunny day",
        "a man fighting an alligator",
        "a gopro recording of a snowboarding trick",
        "a pyramid in the desert",
        "people in the park",
        "supernova in space",
        "the man is eating a hot dog",
        "the dog is playing in the snow"
    ])

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

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as Config.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert Config to dictionary.
        """
        data = {
            "model": self.model.__dict__,
            "train": self.train.__dict__,
            "logging": self.logging.__dict__
        }

        return data
