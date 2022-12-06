import numpy as np
import torch

def get_dice_perClass(confM):
    """
    Takes a confusion matrix confM and returns the dice per class
    """
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
    dicePerClass = np.zeros(3)
    for i in range(0,3):
        if unionPerClass[i] == 0:
            dicePerClass[i] = 1
        else:
            dicePerClass[i] = (2 * confM.diagonal()[i]) / (unionPerClass[i] + confM.diagonal()[i])
    return dicePerClass

def get_iou_perClass(confM):
    """
    Takes a confusion matrix confM and returns the IoU per class
    """
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
    iouPerClass = np.zeros(3)
    for i in range(0,3):
        if unionPerClass[i] == 0:
            iouPerClass[i] = 1
        else:
            iouPerClass[i] = confM.diagonal()[i] / unionPerClass[i]
    return iouPerClass
        
def get_cm(pred, gt, n_classes=3):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        gt_tmp = gt[i].int()

        for actual in range(n_classes):
            for predicted in range(n_classes):
                is_actual = torch.eq(gt_tmp, actual)
                is_pred = torch.eq(pred_tmp, predicted)
                cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))
            
    return cm

def get_confusion_metrics(confusion_matrix):
    """
    Takes a confusion matrix confusion_matrix and returns confusion metrics
    """
    # Compute true positives for each class
    true_positives = np.diagonal(confusion_matrix)

    # Compute false positives for each class
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives

    # Compute false negatives for each class
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    # Compute true negatives for each class
    true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)

    # Compute precision, recall, specificity and sensitivity for each class
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    sensitivity = true_positives / (true_positives + false_negatives)

    # Print precision, recall, specificity and sensitivity for each class
    return precision, recall, specificity, sensitivity