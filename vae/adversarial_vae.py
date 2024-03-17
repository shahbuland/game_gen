from common.modeling import MixIn

import torch.nn.functional as F
import torch

class AdversarialVAE(MixIn):
    def __init__(self, vae, discriminator, adv_weight = 0.1):
        self.vae = vae
        self.discriminator = discriminator

        self.adv_weight = adv_weight
        self.train_disc = False
        self.focus_main()
    
    def focus_main(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = True
        self.train_disc = False
    
    def focus_disc(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True
        for param in self.vae.parameters():
            param.requires_grad = False
        self.train_disc = True

    def forward(pixel_values):
        if train_disc:
            with torch.no_grad():
                dist = self.vae.encode(pixel_values)
                z = dist.sample()
                rec = self.vae.decode(z)
            disc_loss = self.discriminator(pixel_values, rec)
            return disc_loss
        else:
            dist = self.vae.encode(pixel_values)
            z = dist.sample()
            rec = self.vae.decode(z)

            rec_term = F.mse(rec, pixel_values)
            kl_term = dist.kl().mean()

            # TLDR to train gen you maximize discriminator score directly
            adv_term = self.discriminator.classify(rec, patchify = True) * -1

            return rec_term + self.vae.kl_weight * kl_term + self.adv_weight * adv_term


    