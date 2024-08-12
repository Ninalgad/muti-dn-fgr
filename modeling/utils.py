# %%writefile /content/src/utils.py
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt as eucl_distance


def select_fetal_abdomen_mask_and_frame(segmentation_masks: np.ndarray):
    """
    Select the fetal abdomen mask and the corresponding frame number from the segmentation masks
    """
    # Initialize variables to keep track of the largest area and the corresponding 2D image
    largest_area = 0
    selected_image = None

    # Iterate over the 2D images in the 3D array
    for frame in range(len(segmentation_masks)):
        area = np.sum(segmentation_masks[frame])

        if area > largest_area:
            largest_area = area
            selected_image = segmentation_masks[frame]
            fetal_abdomen_frame_number = frame

    # If no 2D image with a positive area was found, provide an empty segmentation mask and set the frame number to -1
    if selected_image is None:
        selected_image = np.zeros_like(segmentation_masks[0])
        fetal_abdomen_frame_number = -1

    return selected_image, fetal_abdomen_frame_number


def load_mask(path):
    if not path:
        return np.zeros((1, 372, 281), dtype='uint8')

    y = np.load(path)
    y = np.minimum(y, 1)
    y = np.expand_dims(y, axis=0)
    return y


def load_ultra(path):
    x = np.load(path)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 256.
    x = np.concatenate([x, x, x], axis=0)
    return x


def one_hot2dist(seg: np.ndarray, resolution=(1, 1),
                 dtype='float32') -> np.ndarray:
    # assert one_hot(torch.tensor(seg), axis=0)
    K = seg.shape[0]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k, :, :].astype('bool')

        if posmask.any():
            negmask = ~posmask
            res[k, :, :] = eucl_distance(negmask, sampling=resolution) * negmask \
                           - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def dice(y_pred, y_true, k=1):
    y_pred = y_pred.astype('float32')
    y_true = y_true.astype('float32')
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + k)


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceBCELoss(nn.Module):
    def __init__(self, from_logits=False, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, smooth=0.1):
        targets = targets.float()

        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class FocalLoss(nn.Module):
    def __init__(self, from_logits=False, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class FocalTverskyLoss(nn.Module):
    def __init__(self, from_logits=False, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, smooth=1, alpha=.5, beta=.5, gamma=1):
        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky
