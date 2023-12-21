# load the data
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.image as mpimg

SIZE = 416  # Resize images to this size if necessary. Must be multiple of 32.


class SatelliteDataset(Dataset):
    def __init__(self, images_dir, ground_truth_dir, transform=None, resize=False):
        """
        Args:
            images_dir (string): Directory with all the images.
            ground_truth_dir (string): Directory with all the ground truths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            resize (bool, optional): If True, resize images and ground truths to 416x416. Default: False
        """
        self.images_dir = images_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.resize = resize
        # Filter out non-image files when listing
        self.images = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        ## Load Image
        image = mpimg.imread(img_name)

        if self.resize:  # Resize Image if necessary
            if image.shape[0] != SIZE or image.shape[1] != SIZE:
                image = Image.open(img_name)
                image = image.resize((SIZE, SIZE))
                image = np.array(image)

        ## Load Ground truth
        ground_truth_name = os.path.join(self.ground_truth_dir, self.images[idx])
        try:
            ground_truth = mpimg.imread(ground_truth_name)
        except FileNotFoundError:
            ground_truth_name = ground_truth_name.replace("_image", "_labels")
            ground_truth = mpimg.imread(ground_truth_name)

        if self.resize:  # Resize ground truth if necessary
            if ground_truth.shape[0] != SIZE or ground_truth.shape[1] != SIZE:
                ground_truth = Image.open(ground_truth_name)
                ground_truth = ground_truth.resize((SIZE, SIZE))
                ground_truth = np.array(ground_truth)

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth


class testDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        # Filter out non-image files when listing
        self.images = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, f"test_{idx+1}", f"test_{idx+1}.png")
        image = mpimg.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
