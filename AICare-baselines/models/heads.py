import torch
from torch import nn


class MultitaskHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, act_layer=nn.GELU, drop=0.0):
        super(MultitaskHead, self).__init__()
        self.hidden_dim = (hidden_dim,)
        self.output_dim = (output_dim,)
        self.act = act_layer()
        self.outcome_task_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop),
            nn.Sigmoid(),
        )
        self.los_task_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = self.act(x)
        outcome = self.outcome_task_head(x)
        los = self.los_task_head(x)
        return torch.cat([outcome, los], dim=2)