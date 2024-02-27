from torch import nn 

# Modulation layers for norms/etc.

class MultipleModulationLayer(nn.Module):
    def __init__(self, d_cond, d_out = None):
        super().__init__()

        if d_out is None:
            d_out = d_cond

        self.fc = nn.Linear(d_cond, d_out * 6)
        self.d = d_out
        self.act = nn.SiLU()

    def forward(self, cond):
        cond = self.act(cond)
        return self.fc(cond)
    
    def get_i(self, my_prev_output, index):
        """
        Given previous output from this layer, get the index-th parameter
        """
        return my_prev_output[:,None,index*self.d:(index+1)*self.d]
    
    def mod(self, x, my_prev_output, index : int):
        """
        Does the index-th modulation step on x given previous output from this layer
        """
        if index == 0:
            a_1, b_1 = self.get_i(my_prev_output, 0), self.get_i(my_prev_output, 1)
            return x * (1 + a_1) + b_1
        elif index == 1:
            c = self.get_i(my_prev_output, 2)
            return x * c
        elif index == 2:
            a_2, b_2 = self.get_i(my_prev_output, 3), self.get_i(my_prev_output, 4)
            return x * (1 + a_2) + b_2
        elif index == 3:
            d = self.get_i(my_prev_output, 5)
            return x * d

class ModulationLayer(nn.Module):
    def __init__(self, d_cond, d_out = None):
        super().__init__()

        if d_out is None:
            d_out = d_cond
        
        self.fc = nn.Linear(d_cond, d_out * 2)
        self.d = d_out

    def forward(self, x, cond):
        params = self.fc(cond)
        a, b = params[:,None,:self.d], params[:,None,self.d:]
        return x * (1 + a) + b
