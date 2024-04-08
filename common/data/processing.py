from typing import List, Tuple
from torchtyping import TensorType

import torchvision.transforms as TF
import torch

from PIL import Image
import math
import numpy as np

# These values seeem to work best (not entirely sure why)
clip_means = [0.48145466, 0.4578275, 0.40821073]
clip_stds = [0.26862954, 0.26130258, 0.27577711]

def common_image_preprocessor(images = None, img_size=224, resampling = 3):
    """
    Simple image preprocessor that can be used directly as callable or to create
    a preprocessing function. If no images are passed, this returns a callable.
    """
    transform = TF.Compose([
        TF.Resize((img_size, img_size), interpolation = resampling),
        TF.ToTensor(),
        TF.Normalize(
            clip_means,
            clip_stds
        )
    ])

    if images is None:
        return lambda x : torch.stack([transform(x_i) for x_i in x])
    return torch.stack([transform(img) for img in images])

@torch.no_grad()
def common_image_postprocessor(model_out, to_pil = False):
    for i in range(3):
        model_out[:,i] = model_out[:,i] * clip_stds[i] + clip_means[i]
    model_out = model_out.clamp(0, 1)
    model_out = model_out.permute(0, 2, 3, 1)
    model_out = (model_out.float() * 255).detach().cpu().numpy().astype(np.uint8)
    return [Image.fromarray(img) if to_pil else img for img in model_out]

def common_video_preprocessor(videos : TensorType["b", "t", "c", "h", "w"] = None):
    """
    Input is list of videos as [0,255] uint8 tensors of equal length
    """

    if videos is None:
        return lambda videos: common_video_preprocessor(videos)

    videos = videos.float() / 255
    return videos

def common_video_postprocessor(model_out):
    model_out = model_out.clamp(0,1)
    model_out = (model_out.float() * 255).detach().cpu().numpy().astype(np.uint8)
    return model_out