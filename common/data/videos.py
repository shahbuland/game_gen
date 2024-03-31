from typing import List

from .processing import common_video_preprocessor

from io import BytesIO
import os
import torch
import einops as eo

import decord
from decord import VideoLoader, gpu, cpu
decord.bridge.set_bridge('torch')

# Alternate version of the above that saves the video to a tmp file, then reads and deletes
def read_video(video_bytes : BytesIO, target_frames = 100, img_size = 256):
    """
    Using decord, read a video from a file path in a given size and FPS
    Retains duration of the original video and uses FPS given to skip frames rather than as an assumption of duration.

    :param path: File path to video
    :param fps: Video is temporally downsampled to be this FPS
    :size: Tuple for image size of each frame
    """
    vr = decord.VideoReader(video_bytes, width = img_size, height = img_size, 
        ctx = cpu(),
        num_threads = 2
    )

    max_frames = len(vr)

    scale_factor = max_frames / target_frames
    inds = list(range(target_frames))
    if max_frames != target_frames:
        inds = [int(i * scale_factor) for i in inds]

    frames = vr.get_batch(inds)
    
    return frames

class VideoCollator:
    """
    Given paths to videos, loads a batch of videos
    Should be sped up by passing number of CPUs as number of workers
    """
    def __init__(
        self,
        processor = None,
        img_size : int = None,
        target_frames : int = None,
    ):
        if processor is None:
            assert target_frames is not None, "Default processor needs target_frames to be passed to collator"
            self.processor = common_video_preprocessor

        self.img_size = img_size
        self.target_frames = target_frames

    def __call__(self, video_bytes : List[BytesIO]):
        video_list = []
        videos = torch.empty(len(video_bytes), self.target_frames, self.img_size, self.img_size, 3, dtype = torch.uint8)

        for i, vid in enumerate(video_bytes):
            videos[i] = read_video(vid, target_frames = self.target_frames, img_size = self.img_size)
        
        videos = eo.rearrange(videos, 'b t h w c -> b t c h w')
        videos = self.processor(videos)
        return {
            "pixel_values" : videos
        }