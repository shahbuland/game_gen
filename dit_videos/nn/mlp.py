from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_out = None, d_middle = None):
        super().__init__()

        if d_out is None:
            d_out = d_in

        if d_middle is None:
            self.fc1 = nn.Linear(d_in, d_out)
            self.fc2 = nn.Linear(d_out, d_out)
        else:
            self.fc1 = nn.Linear(d_in, d_middle)
            self.fc2 = nn.Linear(d_middle, d_out)

        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x