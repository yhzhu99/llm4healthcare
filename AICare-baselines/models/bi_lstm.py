from torch import nn
import einops
from models.utils import get_last_visit

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x, mask, **kwargs):
        bs, visit_length, input_dim = x.shape
        # import pdb;pdb.set_trace()
        output, (h_n, c_n) = self.lstm(x)
        output = einops.rearrange(output, 'b l (d h) -> b d l h', d=2)
        output = output.mean(dim=1)
        out = get_last_visit(output, mask)
        return out