from .denoiser import Denoiser
from .utils import rescale_noise_cfg

from wandb import Video
import numpy as np
import einops as eo
import torch
from tqdm import tqdm
import wandb

def to_np_video(video : torch.Tensor):
    x = eo.rearrange(video, 'n c h w -> n h w c')
    x = x + 1
    x = x * 127.5
    x = x.clamp(0,255)
    x = x.to(torch.uint8)
    x = x.detach().cpu().numpy()
    return x

def to_wandb_video(video : torch.Tensor, captions, fps = 10):
    x = video + 1
    x = x * 127.5
    x = x.clamp(0,255)
    x = x.to(torch.uint8)
    x = x.detach().cpu().numpy()
    return [wandb.Video(x_i, caption = caption, fps = fps) for (caption, x_i) in zip(captions, x)]

def inference(denoiser, init_noise, prompt : str, num_inference_steps = 50, device = 'cuda'):
    b = len(prompt)
    embeds = denoiser.encode_text(text = prompt).to(device)
    # cfg embeddings
    negative_embeds = denoiser.get_negative(b).to(device)
    embeds = torch.cat([embeds, negative_embeds])

    scheduler = denoiser.scheduler
    prev_timesteps = len(scheduler.timesteps)
    scheduler.set_timesteps(num_inference_steps, device = device)
    timesteps = scheduler.timesteps
    sample = init_noise.to(denoiser.device)

    for i,t in enumerate(tqdm(timesteps)):
        sample = scheduler.scale_model_input(sample, t)
        model_input = eo.repeat(sample, 'b ... -> (2 b) ...')

        model_pred = denoiser.predict(
            model_input, t, embeds
        )

        # cfg
        pred_text, pred_uncond = model_pred[:b], model_pred[b:]
        noise_pred = pred_uncond + 7.5 * (pred_text - pred_uncond)
        #noise_pred = rescale_noise_cfg(noise_pred, pred_text, pred_uncond)[None,:]

        sample = scheduler.step(noise_pred, t, sample).prev_sample
    
    # Undo whatever we did
    scheduler.set_timesteps(prev_timesteps, device = device)
    return sample

def wandb_sample(denoiser, prompt, device = 'cuda:0'):
    if type(prompt) is str:
        prompt = [prompt]
    # Defaults to deterministic sampling

    torch.manual_seed(0)
    init_noise = torch.randn(len(prompt), 100, 3, 32, 32)
    num_inference_steps = 50
    sample = inference(denoiser, init_noise, prompt, num_inference_steps, device = device)
    return to_wandb_video(sample, captions = prompt)
