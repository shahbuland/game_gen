import wandb
import torch
import torch_optimizer
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration

from torch.utils.data import DataLoader
from .configs import ProjectConfig
from .sampling import GenModelSampler
from .utils import Timer

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
            dataloader_config = DataLoaderConfiguration(
                dispatch_batches = True,
                split_batches = True
            )
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

        # For ema
        self.ema_accum = 0

        # To account for when we don't want to prepare the loader
        self.prepare_loader = config.train.prepare_loader

    def setup_loader(self):
        mult = 1
        if self.accelerator.split_batches and self.prepare_loader:
            mult = self.world_size
        else:
            # Logic is that any collator not being prepared needs a process index
            self.data_collator.process_idx = self.accelerator.process_index
            
        loader = DataLoader(
            self.train_dataset,
            collate_fn = self.data_collator,
            batch_size = self.config.train.batch_size * mult
        )
        return loader

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @torch.no_grad()
    def sample_fn(self):
        return self.sampler(
            partial(self.model_sample_fn, self.unwrapped_model),
            self.accelerator.device
        )
    
    def handle_ema(self):
        ema_every = self.accum_steps

        if self.ema_accum % ema_every == 0:
            self.unwrapped_model.update_ema() # If model has the mixin
        
        self.ema_accum += 1

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

        should['time'] = self.accelerator.is_main_process
        return should

    def train(self):
        loader = self.setup_loader()

        try:
            opt_class = getattr(torch.optim, self.config.train.opt)
        except:
            opt_class = getattr(torch_optimizer, self.config.train.opt)

        scheduler_class = getattr(torch.optim.lr_scheduler, self.config.train.scheduler)

        opt = opt_class(self.model.parameters(), **self.config.train.opt_kwargs)
        scheduler = scheduler_class(opt, **self.config.train.scheduler_kwargs)

        self.model, opt, scheduler = self.accelerator.prepare(self.model, opt, scheduler)
        if self.prepare_loader:
            loader = self.accelerator.prepare(loader)

        if self.config.train.resume:
            try:
                self.accelerator.load_state(self.config.train.train_state_checkpoint)
            except:
                print("Called with resume but checkpoint could not be loaded. Terminating...")
                exit()

        timer = Timer()
        accum = 0
        for epoch in range(self.config.train.epochs):
            for idx, batch in enumerate(loader):
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    output = self.model(**batch)
                    if isinstance(output, tuple):
                        loss, metrics = output
                        if self.use_wandb:
                            self.accelerator.log(
                                metrics
                            )
                    else:
                        loss = output
                        metrics = {
                            "loss" : loss.item()
                        }

                    self.accelerator.backward(loss)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    self.handle_ema()

                    if self.use_wandb:
                        self.accelerator.log(
                            metrics
                        )
                    else:
                        self.accelerator.print(f"{idx} Loss : {loss.item()}")

                    should = self.get_should(accum)

                    if should["save"]:
                        self.accelerator.save_state(self.config.train.train_state_checkpoint)
                        self.unwrapped_model.save(self.config.train.checkpoint_dir)
                    
                    if should["sample"]:
                        samples = self.sample_fn()
                        self.accelerator.log(samples)
                    
                    if should["eval"]:
                        metrics = self.evaluate_fn()
                        self.accelerator.log(metrics)

                    if should["time"]:
                        self.accelerator.log({
                            "throughput (samples/sec)" : timer.log(self.config.train.batch_size * self.world_size)
                        })
                    
                    accum += 1

                    


