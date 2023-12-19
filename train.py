from helpers import *
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from Networks.common.custom_loss import *
from Networks.dinknet import *
from Networks.UNet import *
from Networks.GCDCNN import *
from Networks.nllinknet_location import *
from Networks.nllinknet_pairwise_func import *

from Loader import *
import time
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import torch.nn.functional as F
from PIL import Image


def train(model,batch_size=8, epochs=50, lr=1e-4,nb_samples_mit=0,nb_samples = 0 ,loss_name="combo"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("Using device: {}".format(device))
    current_time = time.strftime("%Y_%m_%d_%H:%M:%S")
    savepath = "models/"+str(current_time)+".pt"
    ########################################################################################################################################
    ## Create dataset
    
    transform = transforms.Compose([ transforms.ToTensor(), ]) # Convert PIL Images to tensors # Add any other transforms you need here
    dataset = SatelliteDataset("data/training/images", "data/training/groundtruth", transform=transform)
    print("Samples from dataset :", len(dataset))

    ########################################################################################################################################

    ## Splitting dataset into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    print("Training set size: {}".format(train_size))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    ########################################################################################################################################
    ## Create DataLoaders for train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    ########################################################################################################################################
    ## Create model
    model = model.to(device)
    
    ########################################################################################################################################
    #Optimoizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    calc_loss = CustomLoss(beta=0.8)

    ########################################################################################################################################
    best_loss = 1e10
    best_f1_score = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    f1_scores = []  
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

            with torch.set_grad_enabled(True):
                outputs = model(inputs)  # Pad inputs
                loss = calc_loss(outputs, labels,loss_name)  # Crop labels
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
        val_preds,val_targets = []
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels,loss_name)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                val_preds.append(outputs > 0.5)  # Threshold predictions
                val_targets.append(labels >0.5)

        # Store predictions and labels

        val_epoch_loss = val_loss / val_samples
        val_losses.append(val_epoch_loss)
        val_preds = torch.cat(val_preds).view(-1).cpu().numpy()
        val_targets = torch.cat(val_targets).view(-1).cpu().numpy()
        val_f1_score = f1_score(val_targets, val_preds, average='binary')
        f1_scores.append(val_f1_score)

        print("IoU score: {:.4f}".format(IoU(val_targets, val_preds)))
        print("F1 score: {:.4f}".format(val_f1_score))

        val_labels_all.extend(val_targets)
        val_preds_all.extend(val_preds)

        scheduler.step()
        val_epoch_loss = val_loss / val_samples
        val_losses.append(val_epoch_loss)
        print("Validation Loss: {:.4f}".format(val_epoch_loss))
    

        # Check if this is the best model so far
        if  best_f1_score < val_f1_score:
            best_f1_score = val_f1_score
            save_model(model, savepath=savepath)
            print("New best model {} saved with f1 score: {:.4f}".format(savepath, best_f1_score))
        # Print time elapsed for this epoch
        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
     
    # save training and validation losses and f1 scores to csv file
    save_losses(train_losses, val_losses, f1_scores, savepath=savepath)
    return model, train_losses, val_losses

def plot(train_losses,val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def save_losses(train_losses, val_losses, f1_scores, savepath):
    losses = np.array([train_losses, val_losses, f1_scores])
    np.savetxt(savepath + ".csv", losses, delimiter=",")
    
# Define a dictionary mapping model type names to model classes
model_dict = {
    'dinknet34': DinkNet34,
    'linknet34': LinkNet34,
    'baseline': Baseline,
    'nl3_linknet': NL3_LinkNet,
    'nl34_linknet': NL34_LinkNet,
    'nl_linknet_egaussian': NL_LinkNet_EGaussian,
    'nl_linknet_gaussian': NL_LinkNet_Gaussian,
    'UNet': UNet,
    'GCDCNN': GCDCNN
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model for road segmentation.')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train (default: 70)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument("--model", type=str, default="nl34_linknet", choices=model_dict.keys(), help="Model to train: e.g: dinknet / linknet / baseline / nl3_linknet / nl34_linknet / nl_linknet_egaussian / nl_linknet_gaussian")
    parser.add_argument("--mit", type=int, default=0, help="Number of samples from MIT dataset to add to training set")
    parser.add_argument("--deepglobe", type=int, default=0, help="Number of samples from DeepGlobe dataset to add to training set")
    parser.add_argument("--loss", type=str, default="dice", help="Loss function to use: e.g: dice_bce / focal_loss / dice_focal_loss")

    
    args = parser.parse_args()
    
    # Instantiate the selected model
    ModelClass = model_dict[args.model]
    model = ModelClass(num_classes=1)

    model,train_losses,val_losses = train(model,epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,nb_samples = args.deepglobe,loss_name=args.loss,nb_samples_mit=args.mit)
    plot(train_losses,val_losses)
    