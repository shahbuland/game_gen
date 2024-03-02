

from .inference import wandb_sample
from .configs import ProjectConfig
from .denoiser import Denoiser
from .nn.dit import DiTVideo
from .data_sd import VideoDataset, DataCollator

import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

class Trainer:
    def __init__(self, config : ProjectConfig = ProjectConfig()):
        self.config = config

        if self.config.train.target_batch is None:
            self.accum_steps = self.config.train.target_batch // self.config.train.batch_size
        else:
            self.accum_steps = 1

        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps =   self.accum_steps
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

        self.model = self.setup_model()
    
    def setup_model(self):
        model_config = self.config.model
        n_frames = model_config.total_frames
        img_size = model_config.img_size
        return Denoiser(
            DiTVideo,
            (n_frames, model_config.channels, img_size, img_size),
            model_config.patch_size,
            model_config.context_dim,
            model_config.heads, model_config.hidden_size, model_config.n_layers
        )
    
    def setup_loader(self):
        ds = VideoDataset(
            fps = self.config.model.fps,
            size = (self.config.model.img_size,)*2
        )
            #webvid_path = self.config.train.ds_path,
            #mode = self.config.train.ds_mode
        #)
        collator = DataCollator(
            self.model.tokenizer,
            fps = self.config.model.fps,
            target_frames = self.config.model.total_frames,
            size = self.config.model.img_size,
            cfg_prob = self.config.train.cfg_prob
        )
        return DataLoader(
            ds,
            batch_size = self.config.train.batch_size,
            num_workers = self.config.train.num_workers,
            pin_memory = True,
            collate_fn = collator
        )

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

        @torch.no_grad()
        def encode_text(batch):
            return self.accelerator.unwrap_model(self.model).encode_text(
                batch['input_ids'],
                batch['attention_mask']
            )
            
        for epoch in range(self.config.train.epochs):
            for idx, batch in enumerate(loader):
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    if batch == "BATCH_ERROR":
                        continue
                    print(batch["frame_rates"])
                    
                    embeds = encode_text(batch)
                    loss = self.model(batch['pixel_values'], embeds)

                    opt.zero_grad()
                    self.accelerator.backward(loss)
                    opt.step()

                    self.accelerator.wait_for_everyone()

                    if self.use_wandb:
                        self.accelerator.log({
                            "loss" : loss.item()
                        })
                    else:
                        self.accelerator.print(f"{idx} Loss: {loss.item()}")
                    
                    if idx % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                        self.accelerator.save_state(self.config.train.train_state_checkpoint)
                        self.accelerator.unwrap_model(self.model).save(self.config.train.checkpoint_dir)

                    if idx % self.config.train.sample_every == 0 and self.accelerator.is_main_process:
                        with torch.no_grad():
                            wandb_videos = wandb_sample(self.accelerator.unwrap_model(self.model), self.config.train.sample_prompts)
                            if self.use_wandb:
                                wandb.log({
                                    "videos" : wandb_videos
                                })
                    self.accelerator.wait_for_everyone()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
