from torchtyping import TensorType

from common.modeling import MixIn
from common.configs import ViTConfig
from common.utils import (
    freeze_module, unfreeze_module,
    sample_lognorm_timesteps,
    rectflow_lerp, rectflow_sampling
)

import torch
from torch import nn
import einops as eo

class Reflow(MixIn):
    """
    Wrapper to train a v-pred denoiser to become a rectified flow denoiser
    
    :param student: Typically starts as a clone of teacher
    :param teacher: Already trained v-pred conditioned denoiser. Input format of pixel_values, input_ids, attention_mask is assumed
    :param vit_config: ViT config for teacher (to infer input sample size)
    :param n_steps_loss: Loss is computed with an integral. This is how many steps to do integral (reimann sum)
    :param teacher_sampling_fn: Use this as opposed to default teacher sample if it is given
    """
    def __init__(self, student, teacher, vit_config, n_steps_loss = 50, teacher_sample_fn = None):
        super().__init__()

        self.student = student
        self.teacher = teacher
        freeze_module(self.teacher)

        self.config = vit_config
        self.n_steps = n_steps_loss

        self.teacher_sample_fn = teacher_sample_fn

    @torch.no_grad()
    def teacher_sample(self, text_features):
        """
        Sample a generation using teacher
        """
        if self.teacher_sample_fn is not None:
            return self.teacher_sample_fn(text_features)
        return rectflow_sampling(self.teacher, self.config.input_shape, text_features, self.t_steps)

    def forward(self, text_features : TensorType["b", "n", "d"]):
        x, z = self.teacher_sample(text_features)

        t = torch.linspace(0, 1, self.n_steps, dtype = text_features.dtype, device = z.device)
        t = eo.repeat(t, 'n -> b n', b = len(x))

        def lerp(a, b, t):
            t = t.view(t.shape[0], *([1] * (a.dim()-1))) # [B, 1, ...]
            t = t.expand_as(a)
            return a*(1-t) + t*b

        # each [B, ...]
        perturbs = [lerp(z, x, t[:,i]) for i in range(self.n_steps)] 

        # [N, B, ...]
        v_preds = torch.stack([self.student(perturbs[i], 999*t[:,i], text_features) for i in range(self.n_steps)])

        dx_dt = (x - z) / self.n_steps
        dx_dt = eo.repeat(dx_dt, '... -> n ...', n = self.n_steps)

        def sum_skip_last(x):
            return eo.reduce(x, 'n ... -> n', reduction = 'sum')

        loss = (dx_dt - v_preds).pow(2)
        loss = sum_skip_last(loss)
        loss = loss.mean() / self.n_steps

        # As a metric let's also compute path straightness
        # We can do this by just computing overall distance taken by the path
        # The ratio should be small but go to 1 if that path is more straight
        with torch.no_grad():
            lowest_distance = (x-z).pow(2)
            lowest_distance = sum_skip_last(lowest_distance) # [B, 1]
            lowest_distance = lowest_distance.sqrt()
            
            actual_distance = 0.
            dt = 1/self.n_steps
            for i in range(self.n_steps):
                displacement = v_preds[i]*dt
                distance = displacement.pow(2)
                distance = sum_skip_last(distance) # [B, 1]
                distance = distance.sqrt()
                actual_distance += distance

            metric = {
                'loss' : loss.item(),
                'distance' : actual_distance.mean().item(),
                'best_distance' : lowest_distance.mean().item()
            }

        return loss, metric

if __name__ == "__main__":
    # Try it with SD2.1 (a v pred model)
    from diffusers import StableDiffusionPipeline
    from copy import deepcopy

    pipe_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_id)
    pipe.to('cuda')

    test_prompt = ["A photo of a cat", "A photo of a dog"]
    neg_prompt = [""] * len(test_prompt)

    def encode_text(text):
        tok_out = pipe.tokenizer(
            text,
            padding = 'longest',
            return_tensors = 'pt'
        ).to('cuda')
        text_features = pipe.text_encoder(**tok_out, output_hidden_states=True, return_dict = True).hidden_states[-1]
        text_features = pipe.text_encoder.text_model.final_layer_norm(text_features)
        return text_features

    embeds = encode_text(test_prompt + neg_prompt)
    prompt_embeds, neg_embeds = embeds[:len(test_prompt)], embeds[len(test_prompt):]

    import random
    random_number = random.randint(-1000, 1000)
    gen_1 = torch.Generator('cuda').manual_seed(random_number)
    gen_2 = torch.Generator('cuda').manual_seed(random_number)
    
    def call_pipe_fn(text_features):
        # Call the standard pipeline __fn__ with text prompts (the features aren't used)
        noise = pipe.prepare_latents(
            len(text_features),
            4, 96, 96,
            text_features.dtype,
            text_features.device,
            None # no generator to pass
        )

        # diffusers pipelines scale the noise like this, we should unscale before inputting
        noise_unscaled = noise / pipe.scheduler.init_noise_sigma
        pipe_out = pipe(
            latents = noise_unscaled, prompt_embeds = text_features, negative_prompt_embeds = neg_embeds,
            output_type = 'latent'
        ).images

        return pipe_out, noise

    # Simple wrapper for pipe unet to make it work here
    class UNetWrapper(nn.Module):
        def __init__(self, pipe):
            super().__init__()

            self.model = deepcopy(pipe.unet)
        
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs).sample

    reflow_model = Reflow(
        UNetWrapper(pipe),
        pipe.unet,
        ViTConfig(
            input_shape = (4, 96, 96)
        ),
        n_steps_loss = 100,
        teacher_sample_fn = call_pipe_fn
    )

    loss, metric = reflow_model(prompt_embeds)
    print(loss.item())
    print(metric)



    