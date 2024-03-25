from typing import List

from .processing import common_video_preprocessor

from io import BytesIO
import os
import shutil
import torch
from tinygrad.helpers import Timing
import math
import random

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
import decord
from decord import VideoLoader, VideoReader, gpu, cpu
#import multiprocessing
import torch.multiprocessing as mp

decord.bridge.set_bridge('torch')

class BadVideoException(Exception):
    pass

def decord_load_single_video(fp, img_size, target_frames, gpu_idx):
    reader = VideoReader(
        fp, ctx = gpu(gpu_idx),
        width = img_size, height = img_size
    )

    n_frames = len(reader)

    # Downsample or upsample with indices
    inds = [i for i in range(target_frames)]
    if n_frames > target_frames:
        skip = n_frames / target_frames
        inds = [min(n_frames, round(i * skip)) for i in range(target_frames)]
    elif n_frames < target_frames:
        inds = [int(i * (n_frames / target_frames)) for i in inds]
    
    try:
        video = reader.get_batch(inds)
    except:
        raise BadVideoException

    return video

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
            self.processor = common_video_preprocessor(target_frames = target_frames)

        self.img_size = img_size
        self.target_frames = target_frames

    def __call__(self, video_bytes : List[BytesIO]):
        #if not ":" in str(self.device):
        #    gpu_idx = 0
        #else:
        #    gpu_idx = int(str(self.device).split(":")[-1])

        gpu_idx = random.randint(0, 7)
        print(f"{gpu_idx} started a job")

        video_list = []
        for vid in video_bytes:
            try:
                video = decord_load_single_video(vid, self.img_size, self.target_frames, gpu_idx)
                video_list.append(video)
            except BadVideoException:
                continue
        videos = torch.stack(video_list)
        print(f"{gpu_idx} finished a job")


        return common_video_preprocessor(videos)