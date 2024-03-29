"""
Adversarial Trainer (since there's some important differences)
"""

from .trainer import Trainer
from .utils import Timer

import torch
import wandb

class AdversarialTrainer(Trainer):
    """
    :param ratios: How many steps to train main model, then how many steps to train discriminator
        (model is assumed to have focus_main() and focus_disc() methods)
    """
    def train(self, ratios = (1,1)):
        # Expand ratios in terms of accumulation steps
        ratios = (ratios[0] * self.accum_steps, ratios[1] * self.accum_steps)

        loader = self.setup_loader()

        opt_class = getattr(torch.optim, self.config.train.opt)
        scheduler_class = getattr(torch.optim.lr_scheduler, self.config.train.scheduler)

        opt = opt_class(self.model.parameters(), **self.config.train.opt_kwargs)
        scheduler = scheduler_class(opt, **self.config.train.scheduler_kwargs)

        self.model, opt, loader, scheduler = self.accelerator.prepare(self.model, opt, loader, scheduler)

        if self.config.train.resume:
            try:
                self.accelerator.load_state(self.config.train.train_state_checkpoint)
            except:
                print("Called with resume but checkpoint could not be loaded. Terminating...")
                exit()

        timer = Timer()
        for epoch in range(self.config.train.epochs):
            for idx, batch in enumerate(loader):
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    if (idx % sum(ratios)) == 0:
                        self.unwrapped_model.focus_main()
                    elif (idx % sum(ratios)) == ratios[0]:
                        self.unwrapped_model.focus_disc()

                    # Adversarial model loss is such a small part of it
                    # Makes sense to use metrics for logging
                    # Metrics is assumed to be a dict of "str" : "float" that can be passed to wandb
                    loss, metrics = self.model(**batch)

                    self.handle_ema()

                    self.accelerator.backward(loss)
                    opt.step()
                    #scheduler.step()
                    opt.zero_grad()

                    if self.use_wandb:
                        self.accelerator.log(
                            metrics
                        )
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

                    if should["time"]:
                        self.accelerator.log({
                            "throughput (samples/sec)" : timer.log(self.config.config.train.batch_size * self.world_size)
                        })                    

