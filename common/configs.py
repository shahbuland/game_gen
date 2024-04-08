from typing import Tuple, Dict, List, Any

from dataclasses import dataclass, field

@dataclass
class ViTConfig:
    """
    Generic config that is useful for any transformer model

    :param n_layers: Number of transformer layers
    :param n_heads: Number of attention heads
    :param hidden_size: d_model

    :param input_shape: Shape of input to ViT (length 3 for images, 4 for videos)
    :param patching: Patch size for inputs (length 2 for images, 3 for videos)
    """
    n_layers : int = 12
    n_heads : int = 12
    hidden_size : int = 768

    input_shape : Tuple[Any] = (3, 256, 256)
    patching : Tuple[Any] = (32, 32)

    flash : bool = False

    @property
    def num_patches(self):
        if len(self.patching) == 2: # Image patching
            return (self.input_shape[1] // self.patching[0]) * (self.input_shape[2] // self.patching[1])
        elif len(self.patching) == 3: # Video patching
            return (self.input_shape[2] // self.patching[0]) * (self.input_shape[3] // self.patching[1]) * (self.input_shape[0] // self.patching[2])


@dataclass
class TrainConfig:
    batch_size : int = 16
    target_batch : int = 256# // 8 # Divide by num_processes
    num_workers : int = 2 # Using more than 1 results in repeated batches unless shard shuffle used
    epochs : int = 10
    prepare_loader : bool = True # Prepare loader with accelerate?
    
    save_every : int = 200
    sample_every : int = 100
    eval_every : int = 1000

    checkpoint_dir : str = "./dit_out"
    train_state_checkpoint : str = "./trainer_state"
    resume = False

    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "weight_decay": 0.0,
        "betas": (0.8, 0.99),
        "eps": 1e-8
    })

    # scheduler
    scheduler : str = "ExponentialLR"
    scheduler_kwargs : Dict = field(default_factory = lambda : {
        "gamma" : 0.999996
    })

@dataclass
class LoggingConfig:
    run_name : str = None
    wandb_entity : str = None
    wandb_project : str = None

@dataclass
class ProjectConfig:
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
            "train": self.train.__dict__,
            "logging": self.logging.__dict__
        }

        return data
