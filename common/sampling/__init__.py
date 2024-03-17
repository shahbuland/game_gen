from typing import Iterable, Callable
from abc import abstractmethod

import torch
import wandb

from ..data.processing import common_image_preprocessor, common_image_postprocessor

class GenModelSampler:
    """
    Generic class for any sampling function. Takes some set of default prompts to use every time.
    This is only for training visualizations

    :param example_inputs: Inputs used for visualization (i.e. prompts or original images)
    :param preprocessor: Convert example_inputs into model inputs
    :param postprocessor: Convert model outputs into something to pass to logger
    """
    def __init__(
        self,
        example_inputs : Iterable,
        preprocessor : Callable,
        postprocessor : Callable
    ):
        self.example_inputs = example_inputs
        self.model_inputs = preprocessor(example_inputs)
        self.postproc = postprocessor

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
        example_inputs : Iterable,
        preprocessor = common_image_preprocessor(img_size = 224),
        postprocessor = common_image_postprocessor
    ):
        super().__init__(
            example_inputs,
            preprocessor,
            postprocessor
        )

    def __call__(self, model_fn : Callable, device):
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

class Text2ImageSampler(GenModelSampler):
    """
    This sampler assumes example_inputs are text prompts
    """
    def __call__(self, model_fn : Callable, device):
        model_inputs = self.model_inputs.to(device)
        imgs = model_fn(model_inputs)
        imgs = self.postproc(imgs)

        res = [
            wandb.Image(img, caption = prompt) for (img, prompt) in zip(imgs, self.example_inputs)
        ]

        return {
            "Generated Images" : res
        }






