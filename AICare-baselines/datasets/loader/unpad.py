import torch
from torch.nn.utils.rnn import unpad_sequence
import numpy as np

def unpad_y(y_pred, y_true, lens):
    raw_device = y_pred.device
    device = torch.device("cpu")
    y_pred, y_true, lens = y_pred.to(device), y_true.to(device), lens.to(device)
    # y_pred_unpad = unpad_sequence(y_pred, batch_first=True, lengths=lens)
    # y_pred_stack = torch.vstack(y_pred_unpad).squeeze(dim=-1)
    # y_true_unpad = unpad_sequence(y_true, batch_first=True, lengths=lens)
    # y_true_stack = torch.vstack(y_true_unpad).squeeze(dim=-1)
    # return y_pred_stack.to(raw_device), y_true_stack.to(raw_device)
    y_true = y_true[:, 0, :].unsqueeze(dim=-1)
    return y_pred.to(raw_device), y_true.to(raw_device)


def unpad_batch(x, y, lens):
    x = x.detach().cpu()
    y = y.detach().cpu()
    lens = lens.detach().cpu()
    x_unpad = unpad_sequence(x, batch_first=True, lengths=lens)
    x_last = []
    for x in x_unpad:
        # print('x:', len(x[-1]))
        x_last.append(x[-1].numpy())
    y_unpad = unpad_sequence(y, batch_first=True, lengths=lens)
    y_last = []
    for y in y_unpad:
        # print('y:', len(y[-1]))
        y_last.append(y[-1].numpy())
    # print('len', len(x_last), len(y_last), y_last)
    return np.array(x_last), np.array(y_last)
