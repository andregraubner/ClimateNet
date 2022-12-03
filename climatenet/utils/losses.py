import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F


def inputs(logits, true):
    """Turns inputs of logits & true values into computable values.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
    Returns:
        probas: softmax probabilities of predicted class
        true_1_hot: true classifications in a one-hot-encoded vector
        dims: dimensions of the true vector
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))

    return probas, true_1_hot, dims


def get_cardinality(probas, true_1_hot, dims):
    """Returns the cardinality of the inputs"""
    return torch.sum(probas + true_1_hot, dims)


def get_intersection(probas, true_1_hot, dims):
    """Returns the intersection of the inputs"""
    return torch.sum(probas * true_1_hot, dims)


def get_union(cardinality, intersection):
    """Returns the union of the inputs"""
    return cardinality - intersection


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    probas, true_1_hot, dims = inputs(logits, true)
    intersection = get_intersection(probas, true_1_hot, dims)
    cardinality = get_cardinality(probas, true_1_hot, dims)
    union = get_union(cardinality, intersection)

    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def dice_coefficient(logits, true, eps=1e-7):
    """Computes the Dice Coefficient, a.k.a the Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Dice loss.
    """
    probas, true_1_hot, dims = inputs(logits, true)
    intersection = get_intersection(probas, true_1_hot, dims)
    cardinality = get_cardinality(probas, true_1_hot, dims)

    dice_loss = (2 * intersection / (2 * cardinality + eps)).mean()
    return (1 - dice_loss)

def cross_entropy_loss_pytorch(y_logit, y_true):
    '''
    Multi-label cross-entropy with pytorch
    y_true: true value
    y_logit: predicted value
    https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    '''

    loss = nn.CrossEntropyLoss(y_logit,y_true)
    return loss(y_logit,y_true)
