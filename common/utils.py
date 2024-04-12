from typing import Dict, Union, Any, List
from torchtyping import TensorType
import torch

import time

# ========== GENERAL ==============

def dict_to(d : Dict, dest : Union[Any, List[Any]]):
    """
    Allows us to change device/datatype over an arbitrary dictionary of tensors. dest is argument of tensor.to(...)
    Allows for nested types (i.e. if a dict value is itself a dict or a list)
    Skips specific tensor/device/dtype combinations that don't make sense
    """
    
    if not isinstance(dest, list):
        dest = [dest]

    def is_valid_match(t, dest_j):
        if (t.dtype == torch.long or t.dtype == torch.bool) and dest_j == torch.half:
            return False
        return True
    
    # To account for when theres lists of lists or nested data types in the dict
    def recursive_cast(x, dest : List):
        if torch.is_tensor(x):
            for dest_i in dest:
                if is_valid_match(x, dest_i):
                    x = x.to(dest_i)
            return x
        elif isinstance(x, list):
            return [recursive_cast(x_i, dest) for x_i in x]
        elif isinstance(x, dict):
            return {k: recursive_cast(v, dest) for k, v in x.items()}
        else:
            return x

    return recursive_cast(d, dest)

class Timer:
    """
    Measures training through-put in terms of samples/sec
    """
    def __init__(self):
        self.total_samples = 0 # Total samples seen so far
        self.time_start = time.time()

    def log(self, new_n_samples : int):
        self.total_samples += new_n_samples
        total_time = time.time() - self.time_start

        return self.total_samples / total_time

def list_prod(L):
    res = 1
    if len(L) == 0:
        return 0
    for x in L:
        res *= x
    return res

# ========= ADVERSARIAL TRAINING ========

def freeze_module(module : torch.nn.Module):
    for p in module.parameters():
        p.require_grad = False

def unfreeze_module(module : torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

# ========== DIFFUSION/RECT FLOW ================

def sample_lognorm_timesteps(x : TensorType["b", "..."]):
    """
    Logit normal for diffusion timesteps. Uses some input tensor for computing batch, device and dtype
    """
    t = torch.randn(x.shape[0], device = x.device, dtype = x.dtype).sigmoid()
    return t

def rectflow_sampling(model, input_shape, text_features, num_inference_steps):
    """
    Basic rectified flow sampling

    :param model: Model that we will sample with. Assumed forward format is (input, text_features, timestep)
    :param input_shape: Tuple for input shape to model
    :param text_features: Features from some text encoder
    :param num_inference_steps: Number of steps for inference
    """
    b, _, _ = text_features.shape

    dt = 1./num_inference_steps
    eps = 1e-3
    x = torch.randn((b,) + input_shape, dtype = text_features.dtype, device = text_features.device)
    
    for i in range(num_inference_steps):
        num_t = i / num_inference_steps * (1 - eps) + eps
        t = torch.ones(b, device = text_features.device, dtype = text_features.dtype) * num_t
        pred = model.predict(x, text_features, t*999) # [0, 1000] scales is better for pos-emb
        x = x.detach().clone() + pred * dt
    
    return x

def rectflow_lerp(x : TensorType["b", "..."], z : TensorType["b", "..."], t : TensorType["b"]):
    """
    Interpolation/destructive process for rectified flow
    """
    t = t.view(t.shape[0], *([1] * (x.dim()-1))) # [B, 1, ...]
    t = t.expand_as(x)

    return t*x + (1.-t)*z