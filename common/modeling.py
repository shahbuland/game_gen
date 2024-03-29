from torch import nn
import torch

from ema_pytorch import EMA

class MixIn(nn.Module):
    """
    MixIn that can be passed to any model to add some basic common features
    """
    def __init__(self):
        super().__init__()

        self.ema = None

    def init_ema(self, model):
        """
        Initialzies the EMA module for a given model
        """
        self.ema = EMA(
            model,
            beta = 0.9999,
            power = 3/4,
            update_every = 1,
            update_after_step = 1,
            include_online_model = False
        )
    
    def update_ema(self):
        if self.ema is not None:
            self.ema.update()

    def save(self, path):
        torch.save(self, path)

    def load(self, path):
        self.load_state_dict(torch.load(path).state_dict())
    
    def from_pretrained(cls, path):
        return torch.load(path)