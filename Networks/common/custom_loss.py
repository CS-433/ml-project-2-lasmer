import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, beta=1.0, alpha=0.5, gamma=2.0):
        super(CustomLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    def binary_cross_entropy(self, y_true, y_pred, reduction="mean", weight=None):
        loss = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight)
        return loss(y_pred, y_true)

    def weighted_cross_entropy(self, y_true, y_pred):
        with torch.no_grad():
            weights = self.beta * y_true + (1 - y_true)
        return self.binary_cross_entropy(y_pred, y_true, weight=weights)

    def balanced_cross_entropy(self, y_true, y_pred):
        with torch.no_grad():
            weights = self.beta * y_true + (1 - self.beta) * (1 - y_true)
        return self.binary_cross_entropy(y_pred, y_true, weight=weights)

    def focal_loss(self, y_true, y_pred):
        bce = self.binary_cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-bce)
        focal = (
            self.alpha * (1 - pt) ** self.gamma * y_true
            + (1 - self.alpha) * pt**self.gamma * (1 - y_true)
        ) * bce
        return focal.mean()

    def jaccard_loss(self, y_true, y_pred, smooth=1):
        intersection = (y_true * y_pred).sum()
        total = y_true.sum() + y_pred.sum()
        union = total - intersection
        return 1 - (intersection + smooth) / (union + smooth)

    def dice_loss(self, y_true, y_pred, smooth=1):
        intersection = (y_true * y_pred).sum()
        return 1 - (2.0 * intersection + smooth) / (
            y_true.sum() + y_pred.sum() + smooth
        )

    def squared_dice_loss(self, y_true, y_pred, smooth=1):
        intersection = (y_true * y_pred).sum()
        return 1 - (2.0 * intersection + smooth) / (
            torch.pow(y_true, 2).sum() + torch.pow(y_pred, 2).sum() + smooth
        )

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)

    def tversky_loss(self, y_true, y_pred, smooth=1):
        intersection = (y_true * y_pred).sum()
        false_negatives = (y_true * (1 - y_pred)).sum()
        false_positives = ((1 - y_true) * y_pred).sum()
        return 1 - (intersection + smooth) / (
            intersection
            + self.alpha * false_negatives
            + (1 - self.alpha) * false_positives
            + smooth
        )

    def focal_tversky_loss(self, y_true, y_pred, smooth=1):
        tversky = self.tversky_loss(y_true, y_pred, smooth)
        return torch.pow((1 - tversky), self.gamma)

    def bce_dice_loss(self, y_true, y_pred):
        return 0.3 * self.binary_cross_entropy(y_true, y_pred) + 0.7 * self.dice_loss(
            y_true, y_pred
        )

    def combo_loss(self, y_true, y_pred):
        return self.weighted_cross_entropy(y_true, y_pred) + self.dice_loss(
            y_true, y_pred
        )

    def forward(self, y_true, y_pred, loss_type):
        if loss_type == "bce":
            return self.binary_cross_entropy(y_true, y_pred)
        elif loss_type == "weighted_bce":
            return self.weighted_cross_entropy(y_true, y_pred)
        elif loss_type == "balanced_bce":
            return self.balanced_cross_entropy(y_true, y_pred)
        elif loss_type == "focal":
            return self.focal_loss(y_true, y_pred)
        elif loss_type == "jaccard":
            return self.jaccard_loss(y_true, y_pred)
        elif loss_type == "dice":
            return self.dice_loss(y_true, y_pred)
        elif loss_type == "squared_dice":
            return self.squared_dice_loss(y_true, y_pred)
        elif loss_type == "log_cosh_dice":
            return self.log_cosh_dice_loss(y_true, y_pred)
        elif loss_type == "tversky":
            return self.tversky_loss(y_true, y_pred)
        elif loss_type == "focal_tversky":
            return self.focal_tversky_loss(y_true, y_pred)
        elif loss_type == "bce_dice":
            return self.bce_dice_loss(y_true, y_pred)
        elif loss_type == "combo":
            return self.combo_loss(y_true, y_pred)
        else:
            raise ValueError("Invalid loss type specified: {}".format(loss_type))
