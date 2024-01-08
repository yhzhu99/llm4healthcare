import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
import einops

from models.utils import get_last_visit


class MTRHN(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=64, num_layers=1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.x_proj = nn.Linear(input_dim, input_dim)
        self.stats_proj = nn.Linear(6, 6)

        self.stats_fusion = nn.Linear(6,1)
        
        self.rnn = nn.RNN(input_dim*2, hidden_dim, num_layers=self.num_layers)

        self.out_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)


    def compute_stats(self, x):
        # x: (batch_size, time_step, feature_dim)
        # calculate each features' stats in batch, should return (batch_size, feature_dim, 6)
        # 6: mean, median, std, var, min, max
        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)
        stats = torch.zeros((batch_size, feature_dim, 6)).to(x.device)
        for i in range(feature_dim):
            stats[:, i, 0] = torch.mean(x[:, :, i])
            stats[:, i, 1] = torch.median(x[:, :, i])
            stats[:, i, 2] = torch.std(x[:, :, i])
            stats[:, i, 3] = torch.var(x[:, :, i])
            stats[:, i, 4] = torch.min(x[:, :, i])
            stats[:, i, 5] = torch.max(x[:, :, i])
        return stats


    def forward(self, x, mask, **kwargs):
        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        stats = self.compute_stats(x)
        # import pdb; pdb.set_trace()

        # cur_x = x[:,:,:self.cur_x_dim]
        # his_x = x[:,:,self.cur_x_dim:]

        embed_x = x+self.x_proj(x)
        embed_stats = self.stats_fusion(stats+self.stats_proj(stats)).squeeze(dim=-1)

        embed_stats = einops.repeat(embed_stats, 'b d -> b t d', t=time_step)

        embed_cat = torch.cat((embed_x, embed_stats), dim=-1)

        x = embed_cat

        output, _ = self.rnn(x)

        out = get_last_visit(output, mask)
        return out

if __name__ == "__main__":
    x = torch.rand(32, 100, 140)
    model = MTRHN()
    out = model(x)
    print(out.shape)
