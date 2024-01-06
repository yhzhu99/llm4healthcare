import torch

from .binary_classification_metrics import get_binary_metrics
from .regression_metrics import get_regression_metrics
from .metrics_utils import check_metric_is_better


def reverse_los(y, los_info):
    return y * los_info["los_std"] + los_info["los_mean"]

def get_all_metrics(preds, labels, task, los_info):
    # convert preds and labels to tensor if they are ndarray type
    if isinstance(preds, torch.Tensor) == False:
        preds = torch.tensor(preds)
    if isinstance(labels, torch.Tensor) == False:
        labels = torch.tensor(labels)
    
    if task == "outcome":
        return get_binary_metrics(preds, labels)
    elif task == "los":
        return get_regression_metrics(reverse_los(preds, los_info), reverse_los(labels[:, 1], los_info))
    elif task == "multitask":
        return get_binary_metrics(preds[:, 0], labels[:, 0]) | get_regression_metrics(reverse_los(preds[:, 1], los_info), reverse_los(labels[:, 1], los_info))
    else:
        raise ValueError("Task not supported")
    