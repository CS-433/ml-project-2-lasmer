import numpy as np
import matplotlib.pyplot as plt
import os

THRESHOLD = 0.5  # Threshold for converting predictions to binary values


def IoU(pred, target):
    """Calculate the intersection over union (IoU) score.
    Args:
        pred (numpy.ndarray): Predicted binary values
        target (numpy.ndarray): Ground truth binary values
    Returns:
        iou_score (float): Intersection over union score"""
    pred = pred > THRESHOLD
    target = target > THRESHOLD
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union


def plot(train_losses, val_losses):
    """Plot the training and validation losses
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
    Returns:
        None"""

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def save_losses(train_losses, val_losses, f1_scores, savepath):
    """Save the training and validation losses to a csv file
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        savepath (str): Path to save the csv file
    Returns:
        None"""

    losses_path = savepath + ".csv"
    os.makedirs(os.path.dirname(losses_path), exist_ok=True)
    losses = np.array([train_losses, val_losses, f1_scores])
    np.savetxt(losses_path, losses, delimiter=",")
