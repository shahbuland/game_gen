from torch import nn
import torch

class MMDiTModulation(nn.Module):
    """
    mm-dit multi-modulation layer for applying modulation at multiple parts of forward pass
    """
    def __init__(self, d_cond, d_out = None):
        super().__init__()

        if d_out is None: d_out = d_cond

        self.fc = nn.Linear(d_cond, d_out * 6)
        self.d = d_out
        self.act = nn.SiLU()

    def forward(self, cond):
        """
        Forward pass of the conditioning to get all parameters for all modulations
        This needs to be carried forward when calling mod
        """
        cond = self.act(cond)
        return self.fc(cond)
    
    def get_i(self, my_prev_output, index):
        """
        i-th modulation
        """
        # [B, 6*D] needs to broadcast to [B, N, D]
        chunk = torch.chunk(my_prev_output, 6, dim = -1)[index] # 6 chunks on D dim, get the index-th one
        return chunk[:,None] # Add the N dim

    def mod(self, x, my_prev_output, index : int):
        """
        index-th modulation step on x given previous output from self.forward
        """
        if index == 0:
            a_1, b_1 = self.get_i(my_prev_output, 0), self.get_i(my_prev_output, 1)
            return x * (1 + a_1) + b_1
        elif index == 1:
            c = self.get_i(my_prev_output, 2)
            return x * c
        if index == 2:
            a_2, b_2 = self.get_i(my_prev_output, 3), self.get_i(my_prev_output, 4)
            return x * (1 + a_2) + b_2
        elif index == 3:
            d = self.get_i(my_prev_output, 5)
            return x * d

class ModulationLayer(nn.Module):
    """
    AdaLN_0 from DiT
    """
    def __init__(self, d_cond, d_out = None):
        super().__init__()

        if d_out is None:
            d_out = d_cond
        
        self.fc = nn.Linear(d_cond, d_out * 2)
        self.d = d_out
    
    def forward(self, x, cond):
        params = self.fc(cond)[:,None] # [B, 2*D] -> [B, 1, 2*D]
        a, b = torch.chunk(params, 2, dim = -1)
        return x * (1 + a) + b