from common.modeling import MixIn

import torch.nn.functional as F
import torch

class AdversarialVAE(MixIn):
    def __init__(self, vae, discriminator, adv_weight = 0.1):
        super().__init__()

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

    def forward(self, pixel_values):
        if self.train_disc:
            with torch.no_grad():
                dist = self.vae.encode(pixel_values)
                z = dist.sample()
                rec = self.vae.decode(z)
            disc_loss, _ = self.discriminator(pixel_values, rec)
            return disc_loss, {"discriminator_loss" : disc_loss.item()}
        else:
            dist = self.vae.encode(pixel_values)
            z = dist.sample()
            rec = self.vae.decode(z)

            rec_term = F.mse_loss(rec, pixel_values)
            kl_term = dist.kl().mean()

            # adv_signal directly gives the loss the for the decoder
            _, adv_term = self.discriminator(pixel_values, rec).clamp(-1, 1)

            return rec_term + self.vae.kl_weight * kl_term + self.adv_weight * adv_term, {"loss" : rec_term.item(), "adv_loss" : adv_term.item()}


if __name__ == "__main__":
    from .nn.vit_discriminator import ViTPatchDiscriminator
    from .nn.vit_vae import ViTVAE

    IMG_SIZE = 1024

    vae = ViTVAE(
        (32, 32), (3, 1024, 1024), (8, 32, 32),
        4, 8, 256
    ).cuda()

    disc = ViTPatchDiscriminator(
        0.5,
        (32, 32), (3, 1024, 1024), (4, 32, 32),
        4, 8, 256
    ).cuda()

    adv_model = AdversarialVAE(
        vae,
        disc
    )

    x = torch.randn(1, 3, 1024, 1024).cuda()

    adv_model.focus_main()

    print(adv_model(x))

    adv_model.focus_disc()

    print(adv_model(x))
