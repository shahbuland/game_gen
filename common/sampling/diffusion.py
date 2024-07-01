"""
To avoid renames, putting anything diffusion sampling related into this script
"""

import torch
import einops as eo

def ode_sampling(model, input_shape, text_features, num_inference_steps):
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
    
    return x

def ode_sampling_cfg(model, input_shape, text_features, num_inference_steps, negative_features, cfg_scale = 7.0):
    """
    Identical to above but with classifier free guidance added in

    :param negative_feature: Negative text features (assumed single batch element [1, ...])
    """
    b, _, _ = text_features.shape

    negative_features = negative_features.to(text_features.device).to(text_features.dtype)
    negative_features = eo.repeat(negative_features, '1 ... -> b ...', b = b)

    dt = 1./num_inference_steps
    eps = 1e-3
    x = torch.randn((b,) + input_shape, dtype = text_features.dtype, device = text_features.device)
    init_noise = x.clone()
    
    for i in range(num_inference_steps):
        num_t = i / num_inference_steps * (1 - eps) + eps
        t = torch.ones(b, device = text_features.device, dtype = text_features.dtype) * num_t
        
        pred = model.predict(x, text_features, t*999) # [0, 1000] scales is better for pos-emb
        pred_neg = model.predict(x, negative_features, t*999)
        pred_cfg = pred_neg + cfg_scale * (pred - pred_neg)

        x = x.detach().clone() + pred_cfg * dt
    
    return x
