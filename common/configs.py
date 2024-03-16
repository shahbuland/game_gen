from typing import Tuple, Dict, List

from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    batch_size : int = 16
    target_batch : int = 256# // 8 # Divide by num_processes
    num_workers : int = 1 # Using more than 1 results in repeated batches
    epochs : int = 10

    save_every : int = 1000
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
