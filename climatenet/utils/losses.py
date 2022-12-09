import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F

def loss_function(logits, true, config_loss='jaccard'):
    loss = 0.0
    if config_loss == "jaccard":
        loss = jaccard_loss(logits, true)
    elif config_loss == "weighted_jaccard":
        loss = weighted_jaccard_loss(logits, true)
    elif config_loss == "dice":
        loss = dice_loss(logits, true)
    elif config_loss == "cross_entropy":
        loss = cross_entropy_loss(logits, true)
    elif config_loss == "weighted_cross_entropy":
        loss = weighted_cross_entropy_loss(logits, true)
    elif config_loss == "focal_tversky":
        loss = focal_tversky_loss(logits, true)

    return loss

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

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes, device = device)[true.squeeze(1)]
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

def dice_loss(logits, true, eps=1e-7):
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

def cross_entropy_loss(y_logit, y_true):
    '''
    Multi-label cross-entropy with pytorch
    y_true: true value
    y_logit: predicted value
    https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    '''

    loss = nn.CrossEntropyLoss()
    return loss(y_logit,y_true)

def weighted_cross_entropy_loss(logits, true):
    """Computes the weighted cross entropy loss .
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weights: a tensor of shape [C]. Corresponds to the relative
            weights we want to assign to each class.
    Returns:
        wce_loss: the weighted cross-entropy loss.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    wce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.355, 72.171, 5.875], device=device))
    return wce_loss(logits, true)

def weighted_jaccard_loss(logits, true, eps=1e-7):
    """Computes the weighted Jaccard loss using 'weighted IoUs'.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the weighted Jaccard loss.
    """
    probas, true_1_hot, dims = inputs(logits, true)
    intersection = get_intersection(probas, true_1_hot, dims)
    cardinality = get_cardinality(probas, true_1_hot, dims)
    union = get_union(cardinality, intersection)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    weights = torch.tensor([0.355, 72.171, 5.875], device=device) # Tensor of size [ 3 ]

    jacc_loss = (intersection / (union + eps)) # Tensor of size [ 3 x 1152 ]
    jacc_loss = jacc_loss.mean(1) # Tensor of size [ 3 ]
    jacc_loss = (jacc_loss * weights).mean() # Tensor of size [ 1 ]
    return (1 - jacc_loss)

def focal_tversky_loss(logits, true, alpha=0.7, beta=0.3, gamma=4, eps=1e-7):
    """Computes the focal Tversky loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        FT_loss: the focas Tversky loss.
    """
    probas, true_1_hot, dims = inputs(logits, true)
    TP = get_intersection(probas, true_1_hot, dims)
    FP = torch.sum(probas, dims) - TP
    FN = torch.sum(true_1_hot, dims) - TP

    FT_loss = (TP / (TP + alpha*FN + beta*FP + eps)) # Tensor of size [ 3 x 1152 ]
    FT_loss = FT_loss.mean(1) # Tensor of size [ 3 ]
    FT_loss = (FT_loss).mean() # Tensor of size [ 1 ]
    return (1 - FT_loss)**gamma