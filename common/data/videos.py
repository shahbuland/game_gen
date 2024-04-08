from typing import List

from .processing import common_video_preprocessor

from io import BytesIO
import os
import torch
import einops as eo

from torchaudio.io import StreamReader

def yuv_2_rgb(videos):
    videos = videos.float()
    y = videos[...,0,:,:] / 255
    u = videos[...,1,:,:] / 255 - 0.5
    v = videos[...,2,:,:] / 255 - 0.5

    res = torch.empty_like(videos)
    res[...,0,:,:] = y + 1.14 * v # r
    res[...,1,:,:] = y + -0.396 * u - 0.581 * v # g
    res[...,2,:,:] = y + 0.2029 * u # b

    return res

def read_video(video_bytes : BytesIO, gpu_idx : int = 0, img_size = 256, chunk_size = 50):
    """
    Read video from BytesIO wih nvdec

    :param video_bytes: BytesIO object for MP4 video
    :param gpu_idx: Index for GPU to use
    :target_frames
    """
    config = {
        'decoder' : 'h264_cuvid',
        'hw_accel' : f'cuda:{gpu_idx}',
        'decoder_option' : {
            'resize' : f'{img_size}x{img_size}'
        }
    }

    s = StreamReader(video_bytes.getvalue())
    s.add_video_stream(chunk_size, **config)
    chunks = []
    for i, (chunk, ) in enumerate(s.stream()):
        chunks.append(chunk)

    chunks = torch.cat(chunks)

    return chunks

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
        gpu_idx : int = 0
    ):
        if processor is None:
            assert target_frames is not None, "Default processor needs target_frames to be passed to collator"
            self.processor = common_video_preprocessor

        self.img_size = img_size
        self.target_frames = target_frames
        self.process_idx = gpu_idx

    def temporal_resample(self, video):
        n_frames, c, h, w = video.shape
        ratio = n_frames / self.target_frames

        res = torch.empty(self.target_frames, c, h, w, device = video.device, dtype = video.dtype)
        for i in range(self.target_frames):
            res[i] = video[min(round(i*ratio), n_frames - 1)]
        
        return res

    def __call__(self, video_bytes : List[BytesIO]):
        video_list = []
        videos = []

        for i, vid in enumerate(video_bytes):
            try:
                res_vid = read_video(vid, self.process_idx, img_size = self.img_size)
                res_vid = self.temporal_resample(res_vid)
                videos.append(res_vid)
            except:
                print("Bad video detected. Ignoring.")
            
        
        videos = yuv_2_rgb(torch.stack(videos))
        return {
            "pixel_values" : videos
        }
