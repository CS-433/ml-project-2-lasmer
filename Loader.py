# load the data
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.image as mpimg

class SatelliteDataset(Dataset):
    def __init__(self, images_dir, ground_truth_dir, transform=None):
        self.images_dir = images_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        # self.images = os.listdir(images_dir)
        # Filter out non-image files when listing
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = mpimg.imread(img_name)
        if image.shape[0] != 416 or image.shape[1] != 416:
            image = Image.open(img_name)
            image = image.resize((416, 416))
            image = np.array(image) 

        ground_truth_name = os.path.join(self.ground_truth_dir, self.images[idx])
        try:
            ground_truth = mpimg.imread(ground_truth_name)
        except FileNotFoundError:
            try :
                ground_truth_name = ground_truth_name.replace("sat", "mask")
                ground_truth = mpimg.imread(ground_truth_name)
            except :
                try : 
                    ground_truth_name = ground_truth_name.replace("image", "ground_truth", 1)
                    ground_truth = mpimg.imread(ground_truth_name)
                except :
                    ground_truth_name = ground_truth_name.replace("sat", "label")
                    ground_truth = mpimg.imread(ground_truth_name)

        if ground_truth.shape[0] != 416 or ground_truth.shape[1] != 416:
            ground_truth = Image.open(ground_truth_name)
            ground_truth = ground_truth.resize((416, 416))
            ground_truth = np.array(ground_truth)

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth
    
    