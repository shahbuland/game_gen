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

def resample_video(video : TensorType["n_frames", "c", "h", "w"], target_frames : int = 100) -> Tuple[torch.tensor, float]:
    """
    Resamples videos of varying frame lengths to a fixed frame length

    :param video: Video tensor
    :param target_frames: Final frame rate

    :return: Resampled video and factor by which its length was modified to achieve resample
        (i.e. factor of 2.0 indicates it was doubled)
    """

    factor = target_frames / len(video)

    def downsample_video(video, target_frames):
        # Downsample by skipping frames
        N = len(video)
        skip = N / target_frames
        inds = [min(N, round(i * skip)) for i in range(target_frames)]
        return video[inds]
    
    def upsample_video(video, target_frames):
        # Upsample via frame interpolation
        res = torch.empty(
            (target_frames,) + video.shape[1:],
            device = video.device,
            dtype = video.dtype
        )

        N = len(video)

        for i in range(target_frames):
            t = (i / target_frames) * N

            min_ind = math.floor(t)
            max_ind = math.ceil(t)
            t = t % 1

            if max_ind == 0: # Start of video
                res[i] = video[max_ind]
            elif min_ind == (N - 1): # End of video
                res[i] = video[min_ind]
            else:
                res[i] = (1 - t) * video[min_ind] + t * video[max_ind]
        
        return res

    if factor == 1.0:
        return video, factor
    elif factor < 1.0: # video too long
        return downsample_video(video, target_frames), factor
    elif factor > 1.0: # Too short 
        return upsample_video(video, target_frames), factor

def common_video_preprocessor(videos : List[torch.tensor] = None, target_frames = 100):
    """
    Input is list of videos as [0,255] uint8 tensors of varying lengths

    Since we assume working with decord, this assumes decord video output. This is a bit complicated
    so bear with me:
    - Decord loader iteration gives two tensors
    - The first is x:[n_frames, c, h, w], second is y:[n_frames, 2]
        - y[i,0] is which video x[i] frame was from
        - y[i,1] is which frame in that video the frame was from (frame index)
    - Decord does the resizing for us but returns [0,255] uint8 tensors
    """

    if videos is None:
        return lambda videos: common_video_preprocessor(videos, target_frames = target_frames)

    new_videos = []
    factors = []

    """
    for video in videos:
        new_vid, f = resample_video(video, target_frames)
        new_videos.append(new_vid)
        factors.append(f)
    videos = torch.stack(new_videos)
    """

    videos = videos.float()
    for i in range(3):
        videos[:,:,i] = (videos[:,:,i] - clip_means[i]) / clip_stds[i]

    return videos

    
