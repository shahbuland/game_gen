"""
This script contains common/generic collators for model inputs
"""

from typing import Iterable, Union, Tuple

from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, CLIPFeatureExtractor

class ImageCollator:
    def __init__(self, processor = None):
        clip_id = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        if processor is None:
            self.processor = CLIPFeatureExtractor.from_pretrained(clip_id)
        else:
            self.processor = processor
            
    def __call__(self, image_batch: Iterable[Image.Image]):
        return {
            "pixel_values" : self.processor(image_batch, return_tensors = "pt").pixel_values
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

        