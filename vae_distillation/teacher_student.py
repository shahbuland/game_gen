from torch import nn
import torch
import torch.nn.functional as F
from einops import reduce

# NOTE: Rudimentary code for now. Need something to actually experiment on

# For conv backbone VAEs only
class TeacherStudent(nn.Module):
    def __init__(self, teacher_model, student_model):
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

        # Boolean lists
        #  should given layer have features distilled?
        self.student_distill_features = {
            "encoder" : [True] * 4,
            "decoder" : [True] * 4
        }

        # loss weighting
        self.feature_loss_weight = 1.0
        self.kl_loss_weight = 1.0
        self.z_rec_weight = 1.0
        self.rec_weight = 1.0

        # Which features in teacher are mapped to which features in student?
        # You should know before hand how many layers are in the encoder
        # and the decoder
        self.feature_mapping_encoder = {

        }
        self.feature_mapping_decoder = None
    
    @torch.no_grad()
    def mapping_help(self, sample):
        """
        Pretty prints all hidden shapes so you can find best features to match.
        Reccomended to match two layers, and matching middle + later layers is 
        more effective
        """
        print("===================================")
        
        teacher_dist, teacher_h = self.teacher.encode(sample)
        latent_dist, h = self.student.encode(sample)

        print("Teacher hidden layer shapes:")
        for i, h_i in enumerate(teacher_h):
            print(f"Layer {i}: {list(h_i.shape)}")
        
        print("Student hidden layer shapes:")
        for i, h_i in enumerate(h):
            print(f"Layer {i}: {list(h_i.shape)}")

        z_teacher = teacher_dist.sample()
        z_student = latent_dist.sample()

        print("z_teacher shape:", z_teacher.shape)
        print("z_student shape:", z_student.shape)

        x_rec_teacher, h_teacher = self.teacher.decode(z_teacher)
        x_rec_student, h_student = self.student.decode(z_student)

        print("Teacher hidden layer shapes:")
        for i, h_i in enumerate(h_teacher):
            print(f"Layer {i}: {list(h_i.shape)}")

        print("Student hidden layer shapes:")
        for i, h_i in enumerate(h_student):
            print(f"Layer {i}: {list(h_i.shape)}")

        print("Teacher reconstructed x shape:", x_rec_teacher.shape)
        print("Student reconstructed x shape:", x_rec_student.shape)

        print("===================================")
        exit()
    
    def forward(self, pixel_values):
        # Encoding
        latent_dist_true, h_teacher = self.teacher.encode(pixel_values)
        latent_dist, h = self.student.encode(pixel_values)
        
        h_preds = []
        h_targets = []

        for key in self.feature_mapping_encoder:
            h_preds.append(h[key])
            h_targets.append(h_teacher[self.feature_mapping_encoder[key]])

        h_targets = [h_true[idx] for idx in self.feature_mapping_encoder]
        print(len(h))
        print(len(h_targets))
        exit()

        loss = 0

        for (h_i, h_i_target) in zip(h_preds, h_targets):
            loss += self.feature_loss_weight * F.l1_loss(h[idx], h_targets[idx])
        
        # Sample from both distributions 
        z_true = latent_dist_true.sample()
        z = latent_dist.sample

        # Align the latent distribution

        # KL divergence between two diagonal gaussians
        kl_div = self.kl_loss_weight * latent_dist.kl(latent_dist_true)

        z_rec_loss = self.z_rec_weight * F.mse_loss(z, z_true)

        loss += kl_div
        loss += z_rec_loss

        # Decoding
        rec_true, h_true = self.teacher.decode(z_true)
        rec, h = self.student.decode(z)

        if self.feature_mapping_decoder is None:
            self.feature_mapping_decoder = self.lazy_find_mapping(
                h, h_true
            )
        h_targets = [h_true[idx] for idx in self.feature_mapping_decoder]

        for idx in range(len(h)):
            if self.student_distill_features['decoder'][idx]:
                loss += self.feature_loss_weight * F.l1_loss(h[idx], h_targets[idx])

        rec_loss = self.rec_weight * self.rec_loss(rec, rec_true)

        loss += rec_loss

        return loss

if __name__ == "__main__":
    from .adapt_diffusers import HiddenStateAutoencoderKL
    from diffusers import AutoencoderKL
    import torch

    sd_vae_path = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

    model = TeacherStudent(
        teacher_model = HiddenStateAutoencoderKL.from_single_file(sd_vae_path),
        student_model = HiddenStateAutoencoderKL.cast(AutoencoderKL(
            down_block_types = ('DownEncoderBlock2D',)*2,
            block_out_channels = (128, 512),
            up_block_types = ('UpDecoderBlock2D',)*2,
            sample_size = 768, layers_per_block = 1,
            latent_channels = 4
        ))
    ).cuda()
    model.mapping_help(torch.randn(1,3,768,768).cuda())
    exit()

    rand_image = torch.randn(1, 3, 768, 768).cuda()
    loss = model(rand_image)
    print(loss)
