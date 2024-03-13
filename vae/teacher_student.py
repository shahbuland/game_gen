from torch import nn
import torch
import torch.nn.functional as F
from einops import reduce

from .nn.projection import ViTFeatureProjector
from .nn.vit_vae import ViTVAE

# For conv backbone VAEs only
class TeacherStudent(nn.Module):
    def __init__(
        self,
        teacher_model,
        student_model,
        input_shape,
        latent_shape
    ):
        super().__init__()

        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student_model

        # We also assume model encode forward passes return:
        #   - latent_dist, hidden_states
        # And decoder returns
        #   - reconstruction, hidden_states
        # NOTE: This probably requires a wrapper around whatever model you want to use
        # Latent dist is diffusers DiagonalGaussianDistribution

        # loss weighting
        self.feature_loss_weight = 1.0
        self.kl_loss_weight = 1.0
        self.z_rec_weight = 1.0
        self.rec_weight = 1.0

        # int(len(h) * frac) becomes indices for which layers to match
        self.feature_frac_enc = [0.75]
        self.feature_frac_dec = [0.25]

        self.encoder_projs = self.compute_shapes(input_shape, self.feature_frac_enc, self.teacher.encode)
        self.decoder_projs = self.compute_shapes(latent_shape, self.feature_frac_dec, self.teacher.decode)

    def compute_shapes(self, input_shape, fracs, model_fn):
        """
        Computes shapes of the required hidden states and creates projection layers
       
        :param input_shape: Tuple input shape of each sample (excluding batch)
        :param model_fn: The function we call that we assume returns [something], hidden_states
        """
        shape = (1,) + input_shape
        x = torch.randn(shape, device = 'cuda')
        teacher_h = model_fn(x)[-1]

        matched_h = []
        proj = []
        n, d = self.student.n_patches, self.student.hidden_size

        for frac in fracs:
            idx = int(frac * len(teacher_h))
            matched_h.append(teacher_h[idx])

            _, c, h, w = teacher_h[idx].shape
            
            proj.append(ViTFeatureProjector(n, d, c, h, w))

        return nn.ModuleList(proj)

    def forward(self, pixel_values):
        # Encoding
        dist_teacher, h_teacher = self.teacher.encode(pixel_values)
        dist, h = self.student.encode(pixel_values)

        loss = 0

        for frac, proj in zip(self.feature_frac_enc, self.encoder_projs):
            teacher_idx = int(frac * len(h_teacher))
            student_idx = int(frac * len(h))

            loss += proj(h[student_idx], h_teacher[teacher_idx])

        # Sample from both distributions 
        z_true = dist_teacher.sample()
        z = dist.sample()

        # Align the latent distribution

        # KL divergence between two diagonal gaussians
        kl_div = self.kl_loss_weight * dist.kl(dist_teacher)

        z_rec_loss = self.z_rec_weight * F.mse_loss(z, z_true)

        loss += kl_div.squeeze()
        loss += z_rec_loss

        # Decoding
        rec_teacher, h_teacher = self.teacher.decode(z_true)
        rec, h = self.student.decode(z)

        for frac, proj in zip(self.feature_frac_dec, self.decoder_projs):
            teacher_idx = int(frac * len(h_teacher))
            student_idx = int(frac * len(h))

            loss += proj(h[student_idx], h_teacher[teacher_idx])

        rec_loss = self.rec_weight * F.mse_loss(rec, rec_teacher)

        loss += rec_loss

        return loss

if __name__ == "__main__":
    from .adapt_diffusers import HiddenStateAutoencoderKL
    from diffusers import AutoencoderKL
    import torch

    sd_vae_path = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

    teacher = HiddenStateAutoencoderKL.from_single_file(sd_vae_path).cuda()
    c, h, w = teacher.config.out_channels, teacher.config.sample_size, teacher.config.sample_size

    input_size = (teacher.config.out_channels, teacher.config.sample_size, teacher.config.sample_size)
    latent_size = (4, 64, 64)

    model = TeacherStudent(
        teacher_model = teacher,
        student_model = ViTVAE(
            (32, 32), input_size, latent_size,
            8, 4, 256
        ),
        input_shape=input_size, latent_shape=latent_size
    ).cuda()

    rand_image = torch.randn((1,) + input_size).cuda()
    loss = model(rand_image)
    print(loss)
