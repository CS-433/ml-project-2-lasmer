from helpers import *
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from NLLinkNet.loss import *
from NLLinkNet.networks.dinknet import *
from NLLinkNet.networks.unet import *
from NLLinkNet.networks.nllinknet_location import *
from NLLinkNet.networks.nllinknet_pairwise_func import *

from Loader import *
import time
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
import torch.nn.functional as F


def train(batch_size=8, epochs=50, lr=1e-4):
    
    ## Define device for training
    device = torch.device("mps" )
    print("Using device: {}".format(device))
    current_time = time.strftime("%Y_%m_%d_%H:%M:%S")
    savepath = "models/"+str(current_time)+".pt"
    ########################################################################################################################################
    ## Create dataset
    transform = transforms.Compose([transforms.ToTensor(), ]) # Convert PIL Images to tensors # Add any other transforms you need here
    dataset = SatelliteDataset("data/training/images", "data/training/groundtruth", transform=transform)
    dataset_mit = SatelliteDataset("data/MIT/training/images", "data/MIT/training/groundtruth", transform=transform)
    dataset_mit = torch.utils.data.Subset(dataset_mit, range(400))
    
    
    ########################################################################################################################################
    ## Splitting dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset_mit])
    print("Training set size: {}".format(len(train_dataset)))

    ########################################################################################################################################
    ## Create DataLoaders for train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    ########################################################################################################################################
    ## Create model
    # model = DinkNet34(num_classes=1).to(device)
    # model = LinkNet34(num_classes=1).to(device)
    # model = Baseline(num_classes=1).to(device)
    # model = NL3_LinkNet(num_classes=1).to(device)
    # model = NL34_LinkNet(num_classes=1).to(device)
    model = NL_LinkNet_EGaussian(num_classes=1).to(device)
    # model = NL_LinkNet_Gaussian(num_classes=1).to(device)
    
    
    ########################################################################################################################################
    #Optimoizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    calc_loss = dice_bce_loss()

    ########################################################################################################################################
    best_loss = 1e10
    best_f1_score = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    # Add variables to store all labels and predictions for F1 calculation
    val_labels_all, val_preds_all = [], []
    for epoch in tqdm(range(epochs),desc="Training"):
        print('-' * 100,'Epoch {}/{}\n'.format(epoch, epochs - 1))
        since = time.time()
        ########################################################################################################################################
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            #Models expect input images to be divisible by 32 so we pad the input images
            inputs = F.pad(inputs, (8, 8, 8, 8))  # Pad inputs
            labels = F.pad(labels, (8, 8, 8, 8))  # Crop labels
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)  # Pad inputs
                loss = calc_loss(outputs, labels)  # Crop labels
                loss.backward()
                optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_samples += inputs.size(0)

        train_epoch_loss = train_loss / train_samples
        train_losses.append(train_epoch_loss)
        print("Training Loss: {:.4f}".format(train_epoch_loss))

        ########################################################################################################################################
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        for inputs, labels in val_dataloader:
            inputs = F.pad(inputs, (8, 8, 8, 8),value=0)  # Pad inputs
            labels = F.pad(labels, (8, 8, 8, 8),value=0)  # Crop labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels > 0.5
            labels = labels.float()
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        val_samples += inputs.size(0)
        # Store predictions and labels
        preds = (outputs > 0.5).view(-1).cpu().numpy()  # Flatten and threshold predictions
        labels_flat = labels.view(-1).cpu().numpy()  # Flatten labels
        val_f1_score = f1_score(labels_flat, preds, average='binary')
        print("IoU score: {:.4f}".format(IoU(preds,labels_flat)))
        print("F1 score: {:.4f}".format(val_f1_score))

        val_labels_all.extend(labels_flat)
        val_preds_all.extend(preds)

        scheduler.step()
        val_epoch_loss = val_loss / val_samples
        val_losses.append(val_epoch_loss)
        print("Validation Loss: {:.4f}".format(val_epoch_loss))
    

        # Check if this is the best model so far
        if  best_f1_score < val_f1_score:
            best_f1_score = val_f1_score
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model, savepath=savepath)
            print("New best model {} saved with f1 score: {:.4f}".format(savepath, best_f1_score))
        # Print time elapsed for this epoch
        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
     
    
    return model, train_losses, val_losses

def plot(train_losses,val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for road segmentation.')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    args = parser.parse_args()

    model,train_losses,val_losses = train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    plot(train_losses,val_losses)
    