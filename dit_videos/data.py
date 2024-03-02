from typing import Tuple, List, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import einops as eo

import pandas as pd
import os
import random

import decord
from decord import VideoLoader, gpu, cpu
decord.bridge.set_bridge('torch')

def read_video(path, fps = 30, size = (64, 64)):
    """
    Using decord, read a video from a file path in a given size and FPS
    Retains duration of the original video and uses FPS given to skip frames rather than as an assumption of duration.

    :param path: File path to video
    :param fps: Video is temporally downsampled to be this FPS
    :size: Tuple for image size of each frame
    """
    vr = decord.VideoReader(path, width = size[0], height = size[1])
    true_fps = vr.get_avg_fps()
    skip = round(true_fps/fps)
    max_frames = len(vr)

    inds = range(0, max_frames, skip)
    frames = vr.get_batch(inds)
    frames = frames.permute(0, 3, 1, 2)
    frames = frames.float() / 127.5 - 1
    return frames

# Data utilities

def downsample_video(n_frames: int, fps: float, target_frames: int) -> Tuple[List[int], float]:
    """
    Temporally downsample a video via frame sampling to have target_frames number of frames.

    :param n_frames: Number of frames in the original video
    :param fps: Frames per second in the original video
    :param target_frames: Desired number of frames in the downsampled video
    :return: A tuple containing a list of indices for the downsampled video and the corresponding fps
    """
    skip = n_frames / target_frames
    inds = [min(n_frames, round(i * skip)) for i in range(target_frames)]
    new_fps = fps / skip
    return inds, new_fps

def upsample_video(n_frames: int, fps : float, target_frames: int) -> List[int]:
    """
    Temporally upsample a video via frame sampling to have target_frames number of frames.

    :param n_frames: Number of frames in the original video
    :param target_frames: Desired number of frames in the upsampled video
    :return: A list of indices for the upsampled video
    """
    if target_frames <= n_frames:
        raise ValueError("Target frames must be greater than the number of original frames for upsampling.")

    # Calculate the upsampling factor
    factor = target_frames / n_frames

    # Generate the upsampled indices
    inds = []
    for i in range(target_frames):
        # Find the nearest frame in the original video
        original_frame_index = int(round(i / factor))
        # Ensure the index is within the bounds of the original video
        original_frame_index = min(original_frame_index, n_frames - 1)
        inds.append(original_frame_index)


    # Calculate the new frame rate after upsampling
    new_fps = fps * factor

    return inds, new_fps

def resample_video(video, target_frames, fps):
    n_frames = len(video)
    if n_frames < target_frames:
        inds, fps = upsample_video(n_frames, fps, target_frames)
        return video[inds], fps
    elif n_frames > target_frames:
        inds, fps = downsample_video(n_frames, fps, target_frames)
        return video[inds], fps
    else:
        return video, fps
    
class VideoDataset(Dataset):
    """
    Class wrapping WebVid. Returns paths and video titles. Can do the train or val split.

    :param mode: "val" or "train". Train is huge, val is small. For debugging or inference use val.
    """
    def __init__(self, webvid_path = "./webvid", mode = "val"):
        if mode == "train":
            df_path = os.path.join(webvid_path, "/results_10M_train.csv")
        elif mode == "val":
            df_path = os.path.join(webvid_path, "/results_2M_val.csv")
        self.df = pd.read_csv(df_path)
        self.data_root = os.path.join(webvid_path, "data/videos")

        print(f"Initial dataframe size: {len(self.df)}")
        
        # Filter the dataframe for videos that exist
        self.df = self.df[self.df.apply(lambda row: os.path.exists(os.path.join(self.data_root, f"{row['page_dir']}/{row['videoid']}.mp4")), axis=1)]
        self.df = self.df[self.df.apply(lambda row: os.path.getsize(os.path.join(self.data_root, f"{row['page_dir']}/{row['videoid']}.mp4")) >= 10240, axis=1)]
        
        print(f"Dataframe size after filtering: {len(self.df)}")
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_path = os.path.join(self.data_root, f"{row['page_dir']}/{row['videoid']}.mp4")
        return video_path, row['name']
    
    def __len__(self):
        return len(self.df)

class DataCollator:
    """
    Produces batched data for training. By default is fault tolerant to account for partial downloads of dataset where files are not all downloaded or some are corrupted.
    This means the batches might not always be as long as you want them to be.

    :param fps: Temporal downsample to this FPS
    :param size: Size of frames from videos
    """
    def __init__(self, tokenizer, fps = 10, size = 32, target_frames = 100, cfg_prob = 0.1):
        self.fps = fps # This serves as a maximum fps
        self.size = size
        self.tokenizer = tokenizer
        self.cfg_prob = 0

        self.target_frames = target_frames
    
    def __call__(self, batch : Iterable[Tuple[str, str]]):
        b = len(batch)
        videos = [[] for _ in range(b)]
        vid_paths = []
        all_text = []

        for vid, text in batch:
            vid_paths.append(vid)

            if random.random() < self.cfg_prob:
                text = ""
            all_text.append(text)

        # Handle all the text stuff first
        tokenizer_out = self.tokenizer(all_text, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = tokenizer_out['input_ids']
        attention_mask = tokenizer_out['attention_mask']

        videos = []
        frame_rates = []
        for vid_path in vid_paths:
            try:
                video = read_video(vid_path, fps = self.fps, size = (self.size, self.size))
            except:
                continue
            videos.append(video)

        for i in range(len(videos)):
            videos[i], new_fps = resample_video(videos[i], self.target_frames, self.fps)
            frame_rates.append(new_fps)

        if len(videos) == 0:
            return "BATCH_ERROR"
                
        videos = torch.stack(videos)

        return {
            "pixel_values" : videos,
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "frame_rates" : torch.tensor(frame_rates)
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ds = VideoDataset("val")
    dc = DataCollator(tokenizer)

    loader = DataLoader(ds, collate_fn = dc, batch_size = 4)

    for batch in loader:
        print(batch["frame_rates"])
        print(batch["pixel_values"].shape)
        exit()
