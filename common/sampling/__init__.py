from typing import Iterable, Callable
from abc import abstractmethod

import torch
import wandb

from ..data.processing import (
    common_image_preprocessor,
    common_image_postprocessor,
    common_video_preprocessor,
    common_video_postprocessor
)

class GenModelSampler:
    """
    Generic class for any sampling function. Takes some set of default prompts to use every time.
    This is only for training visualizations

    :param example_inputs: Inputs used for visualization (i.e. prompts or original images)
    :param preprocessor: Convert example_inputs into model inputs
    :param postprocessor: Convert model outputs into something to pass to logger
    :param decode_fn: For latent models, if this is given it is applied directly to output of model before postprocess occurs
    """
    def __init__(
        self,
        example_inputs : Iterable,
        preprocessor : Callable,
        postprocessor : Callable,
        decode_fn : Callable = None
    ):
        self.example_inputs = example_inputs
        self.model_inputs = preprocessor(example_inputs) if example_inputs is not None else None
        self.preproc = preprocessor
        self.postproc = postprocessor
        self.decode_fn = decode_fn

    @abstractmethod
    def __call__(self, model_fn : Callable, device):
        """
        Call the sampler with a given model, device

        :param model_fn: A kind of "pipeline" that directly takes processed input
        """
        pass

class ReconstructedImageSampler(GenModelSampler):
    """
    This sampler assumes the example_inputs provided were all PIL images
    """
    def __init__(
        self,
        example_inputs : Iterable = None,
        preprocessor = common_image_preprocessor(img_size = 224),
        postprocessor = common_image_postprocessor
    ):
        super().__init__(
            example_inputs,
            preprocessor,
            postprocessor
        )

    def __call__(self, model_fn : Callable, device, new_inputs = None):
        if new_inputs is not None and self.example_inputs is None:
            self.model_inputs = self.preproc(new_inputs)

        model_inputs = self.model_inputs.to(device)
        rec = model_fn(model_inputs)

        orig = self.postproc(model_inputs)
        rec = self.postproc(rec)

        res = []
        for i in range(len(rec)):
            res += [
                wandb.Image(orig[i], caption = f"{i} Original"),
                wandb.Image(rec[i], caption = f"{i} Reconstruction")
            ]

        return {
            "Image Reconstructions" : res
        }

class ReconstructedVideoSampler(GenModelSampler):
    def __init__(
        self,
        example_inputs : Iterable = None,
        preprocessor = lambda x : x,#common_video_preprocessor,
        postprocessor = common_video_postprocessor
    ):
        super().__init__(
            example_inputs,
            preprocessor,
            postprocessor
        )
    
    def __call__(self, model_fn : Callable, device, new_inputs = None):
        if new_inputs is not None and self.example_inputs is None:
            self.model_inputs = self.preproc(new_inputs)

        model_inputs = self.model_inputs.to(device)
        rec = model_fn(model_inputs)

        orig = self.postproc(model_inputs)
        rec = self.postproc(rec)

        res = []
        for i in range(len(rec)):
            res += [
                wandb.Video(orig[i], caption = f"{i} Original"),
                wandb.Video(rec[i], caption = f"{i} Reconstruction")
            ]

        return {
            "Video Reconstructions" : res
        }

class Text2ImageSampler(GenModelSampler):
    """
    This sampler assumes example_inputs are text prompts
    """
    def __call__(self, model_fn : Callable, device = None):
        if isinstance(self.model_inputs, torch.Tensor):
            model_inputs = self.model_inputs.to(device)
        else:
            model_inputs = self.model_inputs

        imgs = model_fn(model_inputs)
        if self.decode_fn is not None:
            imgs = self.decode_fn(imgs)
        imgs = self.postproc(imgs)

        res = [
            wandb.Image(img, caption = prompt) for (img, prompt) in zip(imgs, self.example_inputs)
        ]

        return {
            "Generated Images" : res
        }






