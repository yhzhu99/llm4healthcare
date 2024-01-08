from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import weight_norm

from models.utils import get_last_visit


# From TCN original paper https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNLayer(nn.Module):
    """Temporal Convolutional Networks layer.

    Shaojie Bai et al. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.

    This layer wraps the PyTorch TCN layer with masking and dropout support. It is
    used in the TCN model. But it can also be used as a standalone layer.

    Args:
        input_dim: input feature size.
        hidden_dim: int or list of ints. If int, the depth will be automatically decided by the max_seq_length. If list, number of channels in each layer.
        max_seq_length: max sequence length. Used to compute the depth of the TCN.
        kernel_size: kernel size of the TCN.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            TCN blocks. Default is 0.5.

    Examples:
        >>> from pyhealth.models import TCNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = TCNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        max_seq_length: int = 20,
        kernel_size: int = 2,
        dropout: float = 0.5,
    ):
        super(TCNLayer, self).__init__()
        self.hidden_dim = hidden_dim

        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(hidden_dim, int) and max_seq_length:
            hidden_dim = [hidden_dim] * int(
                np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size))
            )
        elif isinstance(hidden_dim, int) and not max_seq_length:
            raise Exception(
                "a maximum sequence length needs to be provided if hidden_dim is int"
            )
        else:
            pass

        num_levels = len(hidden_dim)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else hidden_dim[i - 1]
            out_channels = hidden_dim[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_out: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
            out: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
        """
        out = self.network(x.permute(0, 2, 1)).permute(0, 2, 1)
        last_out = get_last_visit(out, mask)
        return last_out, out

class TCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        max_seq_length: int = 10,
        kernel_size: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super(TCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.tcn_layer = TCNLayer(input_dim, hidden_dim, max_seq_length, kernel_size, dropout)
    def forward(self, x, mask):
        batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out, _ = self.tcn_layer(cur_x, cur_mask)
        #     out[:, cur_time, :] = cur_out
        # return out
        out, _ = self.tcn_layer(x, mask)
        return out