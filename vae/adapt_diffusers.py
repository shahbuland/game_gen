"""
This script is to adapt diffusers AutoencoderKL models to a format that can be trained
"""

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder, Decoder
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

import torch
from torch import nn

class HiddenStateEncoder(Encoder):
    def forward(self, sample, output_hidden_states = False):
        h = []

        sample = self.conv_in(sample)
        h.append(sample.clone())

        if self.training and self.gradient_checkpointing:
            raise ValueError("Gradient checkpointing not supported")
        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)
                h.append(sample.clone())

            # middle
            sample = self.mid_block(sample)
            h.append(sample.clone())

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if output_hidden_states:
            return sample, h
        else:
            return sample

    @classmethod
    def cast(cls, model : Encoder):
        # Make a dummy instance of HiddenStateEncoder then fill all the attributes after
        model_new = cls()

        model_new.layers_per_block = model.layers_per_block
        model_new.conv_in = model.conv_in
        model_new.mid_block = model.mid_block
        model_new.down_blocks = model.down_blocks
        model_new.conv_norm_out = model.conv_norm_out
        model_new.conv_act = model.conv_act
        model_new.conv_out = model.conv_out

        return model_new

class HiddenStateDecoder(Decoder):
    def forward(self, sample, output_hidden_states = False):
        h = []

        sample = self.conv_in(sample)
        h.append(sample.clone())

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:
            raise ValueError("Gradient checkpointing not supported")
        else:
            # middle
            sample = self.mid_block(sample)
            h.append(sample.clone())
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)
                h.append(sample.clone())

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if output_hidden_states:
            return sample, h
        else:
            return sample

    @classmethod
    def cast(cls, model : Decoder):
        # Make a dummy instance of HiddenStateDecoder then fill all the attributes after
        model_new = cls()

        model_new.layers_per_block = model.layers_per_block
        model_new.conv_in = model.conv_in
        model_new.mid_block = model.mid_block
        model_new.up_blocks = model.up_blocks
        model_new.conv_norm_out = model.conv_norm_out
        model_new.conv_act = model.conv_act
        model_new.conv_out = model.conv_out

        return model_new

class HiddenStateAutoencoderKL(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        quant_conv,
        post_quant_conv,
        config = None
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_path):
        model = AutoencoderKL.from_pretrained(model_path)
        return HiddenStateAutoencoderKL.cast(model)

    @classmethod
    def from_single_file(cls, model_path):
        model = AutoencoderKL.from_single_file(model_path)
        return HiddenStateAutoencoderKL.cast(model)
    
    @classmethod
    def cast(cls, model : AutoencoderKL):
        encoder = HiddenStateEncoder.cast(model.encoder)
        decoder = HiddenStateDecoder.cast(model.decoder)

        quant_conv = model.quant_conv
        post_quant_conv = model.post_quant_conv
        config = model.config

        return cls(encoder, decoder, quant_conv, post_quant_conv, config)     

    def encode(self, sample):
        x, h = self.encoder(sample, output_hidden_states = True)
        moments = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(moments)
        
        return posterior, h

    def decode(self, sample):
        sample = self.post_quant_conv(sample)
        x, h = self.decoder(sample, output_hidden_states = True)

        return x, h

if __name__ == "__main__":
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    model = HiddenStateAutoencoderKL.from_single_file(url).cuda()

    x = torch.randn(1, 3, 768, 768).cuda()
    z, h = model.encode(x)
    x_rec, h = model.decode(z.sample())
    print(x_rec.shape)

    print("Done")
