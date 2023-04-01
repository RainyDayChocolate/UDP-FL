# AUC for binary classification, with sklearn
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def auc(y_true_proba: np.ndarray, y_pred: np.ndarray):
    return roc_auc_score(y_true_proba, y_pred)


def binary_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    return accuracy_score(y_true, y_pred)

def multiclass_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """_summary_

    Args:
        y_true (torch.Tensor): Should be just a list of integer, 1-d dimension
        y_pred (torch.Tensor): Should be a matrix of probability, 2-d dimension

    Examples:
        y_true = torch.tensor([0, 2, 2, 2]) # 4 samples
        y_pred = torch.tensor([
            [0.1, 0.2, 0.7], # False
            [0.4, 0.5, 0.1], # False
            [0.7, 0.2, 0.1], # True
            [1, 0, 0] # True
        ])
        Returns: 0.5   # 2 correct predictions

    Returns:
        _type_: _description_
    """
    _, predicted = torch.max(y_pred, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()

    return correct, total


