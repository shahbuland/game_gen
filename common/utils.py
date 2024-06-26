from typing import Dict, Union, Any, List
from torchtyping import TensorType
import torch
from torch import nn

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

def n_patches(vit_config):
    """
    Given vit_config, get n_patches tuple for how many patches there are in each dim
    """
    input_shape = vit_config.input_shape
    # Drop channel component
    if len(input_shape) == 3:
        input_shape = input_shape[1:]
    elif len(input_shape) == 4:
        # Order is reversed like this cause in patching temporal patches come last
        input_shape = input_shape[2:] + (input_shape[0],)
    
    patching = vit_config.patching

    res = tuple(shape_i // patch_i for (patch_i, shape_i) in zip(patching, input_shape))
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
    init_noise = x.clone()
    
    for i in range(num_inference_steps):
        num_t = i / num_inference_steps * (1 - eps) + eps
        t = torch.ones(b, device = text_features.device, dtype = text_features.dtype) * num_t
        pred = model.predict(x, text_features, t*999) # [0, 1000] scales is better for pos-emb
        x = x.detach().clone() + pred * dt
    
    return x, init_noise

def rectflow_lerp(x : TensorType["b", "..."], z : TensorType["b", "..."], t : TensorType["b"]):
    """
    Interpolation/destructive process for rectified flow
    """
    t = t.view(t.shape[0], *([1] * (x.dim()-1))) # [B, 1, ...]
    t = t.expand_as(x)

    return t*x + (1.-t)*z

def soft_timestep_remap(timesteps : TensorType["b"]) -> TensorType["b"]:
    """
    Typically timestep embeddings work better when timesteps are larger numbers
    (Since they are essentially the same as pos-enc's in transformers)
    Often times the math is easier if we treat t as a float in [0,1]
    This utility function maps [0,1] to [0,1000] if it has not been done already
    """
    if timesteps.max().item() > 1:
        return timesteps
    else:
        return 999*timesteps

# ======= INIT =========

def mimetic_init_(d, n_heads, alpha_1 = 0.7, beta_1 = 0.7, alpha_2 = 0.4, beta_2 = 0.4):
    """
    Main part to mimetic init which actually generates the weight matrices
    """
    k = d // n_heads
    
    Z = torch.randn(d,d)/(d**.5) # ~ N(0,I/d)
    term = alpha_1 * Z + beta_1 * torch.eye(d)
    u,s,v = torch.linalg.svd(term)
    s = torch.diag(s)
    
    Wv = u @ s
    Wp = v @ torch.pow(s, 0.5)

    def subsample():
        Z = torch.randn(d,d)/(d**.5)
        term = alpha_2 * Z + beta_2 * torch.eye(d)
        u,s,v = torch.linalg.svd(term)
        s = torch.diag(s)
        Wq = u[:,:k] @ torch.pow(s[:k,:k], 0.5)
        Wk = v[:,:k] @ torch.pow(s[:k,:k], 0.5)

        return Wq, Wk

    Wq, Wk = [], []
    for i in range(n_heads):
        Wq_head, Wk_head = subsample()
        Wq.append(Wq_head)
        Wk.append(Wk_head)

    Wq = torch.cat(Wq, -1)
    Wk = torch.cat(Wk, -1)

    return torch.cat([Wq,Wk,Wv], 0)

@torch.no_grad()
def mimetic_init(layer, n_heads):
    """
    Initialize QKV weights using mimetic init: https://arxiv.org/pdf/2305.09828
    """
    
    # Infer d from the layer
    d = layer.weight.shape[1] # Input shape
    layer.weight = nn.Parameter(mimetic_init_(d, n_heads))
