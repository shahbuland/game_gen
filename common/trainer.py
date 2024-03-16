import wandb
import torch
from tqdm import tqdm
from accelerate import Accelerator

from torch.utils.data import DataLoader
from .configs import ProjectConfig
from .sampling import GenModelSampler

from abc import abstractmethod
from functools import partial

class Trainer:
    """
    :param model: Model we are training. The forward should return loss
    :param sampler: Used to sample generations with gen models for logging
    :param model_sample_fn: Function to call to get generations with models
        (since forward returns loss, this probably isn't the same thing)
        (note that this should take model AND inputs and then return desired output)
    """
    def __init__(
        self,
        model,
        train_dataset, data_collator, eval_dataset = None,
        config : ProjectConfig = ProjectConfig(),
        sampler : GenModelSampler = None, model_sample_fn = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset

        self.config = config

        if self.config.train.target_batch is None:
            self.accum_steps = self.config.train.target_batch // self.config.train.batch_size
        else:
            self.accum_steps = 1

        # For video datasets, decord sucks at multiproc
        # As such it makes more sense to have dataloader purely on one device,
        # then to dispatch batches after collation
        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps = self.accum_steps,
            split_batches = True,
            dispatch_batches = True
        )

        tracker_kwargs = {}
        self.use_wandb = not (config.logging.wandb_project is None)
        if self.use_wandb:
            log = config.logging
            tracker_kwargs["wandb"] = {
                "name" : log.run_name,
                "entity" : log.wandb_entity,
                "mode" : "online"
            }

            self.accelerator.init_trackers(
                project_name = log.wandb_project,
                config = config.to_dict(),
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes

        self.sampler = sampler
        self.model_sample_fn = model_sample_fn

    def setup_loader(self):
        loader = DataLoader(
            self.train_dataset,
            collate_fn = self.data_collator,
            batch_size = self.config.train.batch_size
        )
        return loader

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def sample_fn(self):
        return self.sampler(
            partial(self.model_sample_fn, self.unwrapped_model),
            self.accelerator.device
        )

    @abstractmethod
    def evaluate_fn(self):
        """
        Should return a dictionary of values to log to wandb
        """
        pass

    def get_should(self, idx):
        # Returns dictionary expressing if we "should" do certain things

        keys = ["save", "sample", "eval"]
        intervals = [self.config.train.save_every, self.config.train.sample_every, self.config.train.eval_every]

        should = {}
        for key, interval in zip(keys, intervals):
            should[key] = idx % interval == 0 and self.accelerator.is_main_process

        should['eval'] = should['eval'] and self.eval_dataset is not None
        should['sample'] = should['sample'] and \
            self.model_sample_fn is not None and \
            self.sampler is not None and \
            self.use_wandb

        return should

    def train(self):
        loader = self.setup_loader()
        opt_class = getattr(torch.optim, self.config.train.opt)
        opt = opt_class(self.model.parameters(), **self.config.train.opt_kwargs)

        self.model, opt, loader = self.accelerator.prepare(self.model, opt, loader)

        if self.config.train.resume:
            try:
                self.accelerator.load_state(self.config.train.train_state_checkpoint)
            except:
                print("Called with resume but checkpoint could not be loaded. Terminating...")
                exit()

        for epoch in range(self.config.train.epochs):
            for idx, batch in enumerate(loader):
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    loss = self.model(**batch)
                    opt.zero_grad()
                    self.accelerator.backward(loss)
                    opt.step()

                    if self.use_wandb:
                        self.accelerator.log({
                            "loss" : loss.item()
                        })
                    else:
                        self.accelerator.print(f"{idx} Loss : {loss.item()}")

                    should = self.get_should(idx)

                    if should["save"]:
                        self.accelerator.save_state(self.config.train.train_state_checkpoint)
                        self.accelerator.unwrap_model(self.model).save(self.config.train.checkpoint_dir)
                    
                    if should["sample"]:
                        samples = self.sample_fn()
                        wandb.log(samples)
                    
                    if should["eval"]:
                        metrics = self.evaluate_fn()
                        wandb.log(metrics)


