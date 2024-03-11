from torch import nn
import torch.nn.functional as F
from einops import reduce

# NOTE: Rudimentary code for now. Need something to actually experiment on

# For conv backbone VAEs only
class TeacherStudent(nn.Module):
    def __init__(self, teacher_model, student_model):

        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student_model

        # For simplicities sake we will make heavy assumptions about architectures
        # Specifically, we will assume VAEs

        # We will also assume the teacher model can output hidden states
        # the channel lists below should correspond to layers
        # the outputted hidden states should correspond to layers represented by channel counts

        # We also assume the model encode forward passes return:
        #   - latent, mu, logvar, hidden_states
        # And decoder returns
        #   - reconstruction, hidden_states
        # NOTE: This probably requires a wrapper

        # Boolean lists
        #  should given layer have features distilled?
        self.student_distill_features = {
            "encoder" : []
            "decoder" : []
        }

        # loss weighting
        self.feature_loss_weight = 0.1
        self.kl_loss_weight = 0.1
        self.z_rec_weight = 0.1
        self.rec_weight = 0.1
    
    def forward(self, pixel_values):
        # Encoding
        z_true, mu_true, logvar_true, h_true = self.teacher.encode(pixel_values, output_hidden_states = True)
        z, mu, logvar, h = self.student.encode(pixel_values, output_hidden_states = True)
        
        step_size = len(h_true) // len(h)
        h_true_sampled = h_true[::step_size][:len(h)]

        loss = 0

        for idx in range(len(h)):
            if self.student_distill_features['encoder'][idx]:
                loss += self.feature_loss_weight * F.l1_loss(h[idx], h_true_sampled[idx])
        
        # Align the latent distribution
        loss += F.mse_loss()

        # KL divergence between two diagonal gaussians
        d = torch.prod(torch.tensor(z.shape[1:]))
        def d_sum(x):
            return reduce(x, 'b ... -> b', reduction = 'sum')

        kl_div = self.kl_loss_weight * 0.5 * (
            d_sum(logar - logvar_true) - \
            d + \
            d_sum(logvar - logvar_true).exp() + \
            d_sum((mu_true - mu).pow(2) * (1./logvar_true))
        )

        z_rec_loss = self.z_rec_weight * F.mse_loss(z, z_true)

        loss += kl_div
        loss += z_rec_loss

        # Decoding
        rec_true, h_true = self.teacher.decode(z_true, output_hidden_states = True)
        rec, h = self.student.decode(z, output_hidden_states = True)

        step_size = len(h_true) // len(h)
        h_true_sampled = h_true[::step_size][:len(h)]

        for idx in range(len(h)):
            if self.student_distill_features['decoder'][idx]:
                loss += self.feature_loss_weight * F.l1_loss(h[idx], h_true_sampled[idx])

        rec_loss = self.rec_weight * self.rec_loss(rec, rec_true)

        loss += rec_loss

        return loss


