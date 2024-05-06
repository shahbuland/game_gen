from torchtyping import TensorType

from common.modeling import MixIn
from common.utils import (
    freeze_module, unfreeze_module,
    sample_lognorm_timesteps,
    rectflow_lerp, rectflow_sampling
)

import einops as eo

class Reflow(MixIn):
    """
    Wrapper to train a v-pred denoiser to become a rectified flow denoiser
    
    :param student: Typically starts as a clone of teacher
    :param teacher: Already trained v-pred conditioned denoiser. Input format of pixel_values, input_ids, attention_mask is assumed
    :param vit_config: ViT config for teacher (to infer input sample size)
    :param n_steps_loss: Loss is computed with an integral. This is how many steps to do integral (reimann sum)
    """
    def __init__(self, student, teacher, vit_config, n_steps_loss = 50):
        self.student = student
        self.teacher = teacher
        freeze_module(self.teacher)

        self.config = vit_config
        self.n_steps = n_steps_loss

    @torch.no_grad()
    def teacher_sample(self, text_features):
        """
        Sample a generation using teacher
        """
        return rectflow_sampling(self.teacher, self.config.input_shape, text_features, self.t_steps)

    def forward(self, text_features : TensorType["b", "n", "d"]):
        x, z = self.teacher_sample(text_features)

        t = torch.linspace(0, 1, self.n_steps, device = z.device)
        t = eo.repeat(t, 'n -> b n', b = len(x))

        def lerp(a, b, t):
            t = t.view(t.shape[0], *([1] * (a.dim()-1))) # [B, 1, ...]
            t = t.expand_as(a)
            return a*(1-t) + t*b

        # each [B, ...]
        perturbs = [lerp(z, x, t[:,i]) for i in range(self.n_steps)] 

        # [N, B, ...]
        v_preds = torch.stack([self.student(perturbs[i], 999*t[:,i]) for i in range(self.n_steps)])

        dx_dt = (x - z) / self.n_steps
        dx_dt = eo.repeat(dx_dt, '... -> n ...', n = self.n_steps)

        loss = (dx_dt - v_preds).pow(2)
        loss = eo.reduce(loss, 'n ... -> n', reduction = 'sum')
        loss = loss.mean() / self.n_steps

        return loss





    
    