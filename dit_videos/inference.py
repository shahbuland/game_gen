from .denoiser import Denoiser
from .utils import rescale_noise_cfg

from wandb import Video
import numpy as np
import einops as eo
import torch
from tqdm import tqdm
import wandb
from copy import deepcopy

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

def inference(denoiser, init_noise, prompt, num_inference_steps = 50, device = 'cuda'):
    b = len(prompt)
    embeds = denoiser.encode_text(text = prompt).to(device)
    # cfg embeddings
    negative_embeds = denoiser.get_negative(b).to(device)
    embeds = torch.cat([embeds, negative_embeds])

    scheduler = deepcopy(denoiser.scheduler)
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
        noise_pred = rescale_noise_cfg(noise_pred, pred_text, pred_uncond)

        sample = scheduler.step(noise_pred, t, sample).prev_sample
    
    return sample

@torch.no_grad()
def ode_infernce(denoiser, init_noise, prompt, num_inference_steps = 100, device = 'cuda:0'):
    """
    Euler ODE solver
    """
    eps = 1e-3
    dt = 1./num_inference_steps
    b = len(prompt)

    x = init_noise.to(device)
    embeds = denoiser.encode_text(text = prompt).to(device)
    # cfg embeddings
    negative_embeds = denoiser.get_negative(b).to(device)
    embeds = torch.cat([embeds, negative_embeds])

    for i in range(num_inference_steps):
        t_i = i/num_inference_steps * (1 - eps) + eps
        t_i = torch.ones(b, device = device) * t_i

        model_input = eo.repeat(x, 'b ... -> (2 b) ...')
        t_i = eo.repeat(t_i, 'b ... -> (2 b) ...')
        model_pred = denoiser.predict(
            model_input,
            t_i,
            embeds
        )

        # cfg
        pred_text, pred_uncond = model_pred[:b], model_pred[b:]
        final_pred = pred_uncond + 7.5 * (pred_text - pred_uncond)
        #noise_pred = rescale_noise_cfg(final_pred, pred_text, pred_uncond)

        x = x.detach().clone() + final_pred * dt

    return x

def wandb_sample(denoiser, prompt, device = 'cuda:0', mode = "ddim"):
    if type(prompt) is str:
        prompt = [prompt]
    # Defaults to deterministic sampling

    torch.manual_seed(0)
    init_noise = torch.randn(len(prompt), 100, 3, 32, 32)
    num_inference_steps = 100

    if mode == "ddim":
        sample = inference(denoiser, init_noise, prompt, num_inference_steps, device = device)
    elif mode == "ode":
        sample = ode_infernce(denoiser, init_noise, prompt, num_inference_steps, device = device)
    else:
        raise ValueError("Invalid inference mode")

    return to_wandb_video(sample, captions = prompt)
