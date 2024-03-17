"""
This script contains common/generic collators for model inputs
"""

from typing import Iterable, Union, Tuple

from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, CLIPFeatureExtractor
import torchvision.transforms as TF

from .processing import common_image_preprocessor

class ImageCollator:
    def __init__(self, processor = None, device = None, img_size = None):
        if processor is None:
            assert img_size is not None, "Default processor needs img_size to be passed to collator"
            self.processor = common_image_preprocessor(img_size = img_size)
        else:
            self.processor = processor

        self.device = device

    def __call__(self, image_batch: Iterable[Image.Image]):
        batch = self.processor(image_batch)
        if self.device is not None:
            batch = batch.to(self.device)

        return {
            "pixel_values" : batch
        }

class ImageTextCollator:
    def __init__(self, tokenizer = None):
        self.img_c = ImageCollator

        clip_id = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        if isinstance(tokenizer, str): # Interpret is as an identifier
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) 
        elif tokenizer is not None: # It was passed
            self.tokenizer = tokenizer
        else: # reasonable random init
            self.tokenizer = AutoTokenizer.from_pretrained(clip_id)

    def __call__(self, batch : Iterable[Tuple]):
        """
        Image/text collator that works agnostic to whether batches are presented (image, str) or (str, image)
        """
        shift = 0
        if not isinstance(batch[0][0], Image.Image):
            shift = 1 # shift if caption first

        images = [item[shift] for item in batch]
        texts = [item[1-shift] for item in batch]

        images_out = self.img_c(images)
        tok_out = self.tokenizer(
            texts,
            return_tensors = 'pt',
            padding = "max_length", # Just standard diffusion settings
            truncation = True,
            max_length = 77
        )

        return images_out + {
            "input_ids" : tok_out.input_ids,
            "attention_mask" : tok_out.attention_mask
        }
TextImageCollator = ImageTextCollator # Alias lol

        