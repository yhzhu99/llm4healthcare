import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision

threshold = 0.5

def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=threshold)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")

    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    
    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
    }
