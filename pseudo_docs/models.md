# Models

- `common/modeling.py` contains a mixin class that has some primitives for ema, saving, loading and loading a pretrained model from path (ala HF, granted I use my own model classes which are basically training wrappers, to simplify training)
- I generally follow the same formula with all of my models in that:
    -> They take the absolute bare minimum of what they need to train, and generate intermediate info themselves in their forward call (i.e. diffusion models only take sample pixel_values, generate timesteps themselves)
    -> Models output loss directly for us to call backward on

# ViTs

- There is a heavy emphasis on ViT backbones for most models in this repo
- See `common/nn/vit_modules.py` and `ViTConfig` in `common/configs.py` for an easy method to configure any ViT model
- Flash is currently WIP because not all hardware/torch versions support it so it might be commented out in certain places

# Denoiser

- In `common/nn/mm_dit.py` you will find an implementation of MMDiT from the SD3 paper
- In `common/nn/denoiser.py` you will find a wrapper around MMDiT to use for any rectified flow training. Note that neither of these implement the text embedder