from helpers import *
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from NLLinkNet.loss import *
from NLLinkNet.custom_loss import *
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
from sklearn.model_selection import KFold

import torch.nn.functional as F
from PIL import Image

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()

def train(model,batch_size=8, epochs=50, lr=1e-4,nb_samples_mit=0,nb_samples = 0 ,loss_name="combo",k_folds=5):
    
    ## Define device for training
    device = torch.device("mps" )
    print("Using device: {}".format(device))
    ########################################################################################################################################
    model = model.to(device) ## Create mode 
    calc_loss = CustomLoss(beta=0.8) # Define loss function
    current_time = time.strftime("%Y_%m_%d_%H:%M:%S")
    savepath = os.path.join( "./models", str(current_time))
    os.makedirs(savepath, exist_ok=True)

    ########################################################################################################################################
    ## Create dataset
    
    transform = transforms.Compose([ transforms.ToTensor(), ]) # Convert PIL Images to tensors # Add any other transforms you need here
    dataset = SatelliteDataset("data/training/images", "data/training/groundtruth", transform=transform)
    print("Samples from original dataset :", len(dataset))
    augmented_dataset1 = SatelliteDataset("data/augmented/images", "data/augmented/ground_truth", transform=transform)
    print("Samples from augmented dataset :", len(augmented_dataset1))
    augmented_dataset2 = SatelliteDataset("data/augmented/augmented_images", "data/augmented/augmented_ground_truth", transform=transform)
    print("Samples from augmented dataset :", len(augmented_dataset2))
    
    # dataset_mit = SatelliteDataset("data/MIT/training/images", "data/MIT/training/groundtruth", transform=transform)
    # dataset_mit = torch.utils.data.Subset(dataset_mit, range(nb_samples_mit))
    # print("Samples from MIT dataset :", nb_samples_mit)
    # dataset_deepglobe = SatelliteDataset("data/DeepGlobe/training/images", "data/DeepGlobe/training/groundtruth", transform=transform)
    # dataset_deepglobe = torch.utils.data.Subset(dataset_deepglobe, range(nb_samples))
    # print("Samples from DeepGlobe dataset :", nb_samples)
    # dataset_drive = SatelliteDataset("training/trainingImages", "training/trainingGroundtruth", transform=transform)
    # print("Samples from Drive dataset :", len(dataset_drive))
    # dataset_final = SatelliteDataset("final_data/images", "final_data/groundtruths", transform=transform)


    # dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset1])
    # dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset2])

     # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}
  
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}/{k_folds-1}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_dataloader = DataLoader(dataset, batch_size=batch_size,sampler=train_subsampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size,sampler=test_subsampler)
        
        # Init the neural network
        network = model.to(device) # Move network to GPU if available
        network.apply(reset_weights)
        network.train() # Set network in training mode
        # Initialize optimizer
        
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)     # Optimizer 
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # Scheduler for learning rate decay
        
        # Run the training loop for defined number of epochs
        train_losses = []
        for epoch in tqdm(range(epochs),desc ="Training"):
            print(f'Starting epoch {epoch+1}')
            start_time = time.time()
            current_loss = 0.0 # Set current loss value

            # Iterate over the DataLoader for training data
            for i, data in tqdm(enumerate(train_dataloader, 0)):
                inputs, targets = data # Get inputs
                inputs, targets = inputs.to(device), targets.to(device) # Transfer to GPU
                optimizer.zero_grad() # Zero the gradients
                outputs = network(inputs) # Perform forward pass
                loss = calc_loss(outputs, targets,loss_name) # Compute loss
                loss.backward() # Perform backward pass
                optimizer.step() # Perform optimization
            
                # Print statistics
                current_loss += loss.item()
                ('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            train_epoch_loss = current_loss / len(train_dataloader)
            train_losses.append(train_epoch_loss)
            print(f"Epoch {epoch}/{epochs - 1} - Training Loss: {train_epoch_loss:.4f}")            
            current_loss = 0.0
            print('Time for epoch: %.3f' % (time.time() - start_time))
            
            # Perform validation with eval mode
            network.eval()
            val_losses = []
            val_loss = 0.0
            val_preds = []
            val_targets = []
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                val_targets.append((labels>0.5).float())
                val_preds.append((outputs > 0.5).float())

            # Calculate and print F1 score at the end of each validation phase
            val_f1_score = f1_score(
                torch.cat(val_targets).view(-1).cpu().numpy(),
                torch.cat(val_preds).view(-1).cpu().numpy(),
                average='binary'
            )
            val_epoch_loss = val_loss / len(val_dataloader)
            val_losses.append(val_epoch_loss)
            print(f"Epoch {epoch}/{epochs - 1} - Validation Loss: {val_epoch_loss:.4f}, F1 Score: {val_f1_score:.4f}")
        # Saving the model
        save_path = os.path.join(savepath, 'fold-{fold}.pth')
        torch.save(network.state_dict(), save_path) 

        # Evaluationfor this fold
        network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in tqdm(enumerate(val_dataloader, 0)):
                inputs, targets = data # Get inputs
                inputs, targets = inputs.to(device), targets.to(device) # Transfer to GPU
                outputs = network(inputs) # Generate outputs
                outputs = (outputs > 0.5).float() * 1 # Threshold the outputs to obtain binary predictions
                targets = (targets > 0.5).float() * 1 # Convert targets to binary values
                _, predicted = torch.max(outputs.data, 1) # Set total and correct
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # f1 = f1_score(targets, predicted, average='binary')
            f1 = f1_score(
                targets.view(-1).cpu().numpy(),
                predicted.view(-1).cpu().numpy(),
                average='binary'
            )
            print('F1 Score for fold %d: %.4f' % (fold, f1))
            print('--------------------------------')
            results[fold] = f1
        scheduler.step() # Perform learning rate decay

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    return network
    

    


# Define a dictionary mapping model type names to model classes
model_dict = {
    'dinknet34': DinkNet34,
    'linknet34': LinkNet34,
    'baseline': Baseline,
    'nl3_linknet': NL3_LinkNet,
    'nl34_linknet': NL34_LinkNet,
    'nl_linknet_egaussian': NL_LinkNet_EGaussian,
    'nl_linknet_gaussian': NL_LinkNet_Gaussian
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model for road segmentation.')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train (default: 70)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument("--model", type=str, default="nl34_linknet", choices=model_dict.keys(), help="Model to train: e.g: dinknet / linknet / baseline / nl3_linknet / nl34_linknet / nl_linknet_egaussian / nl_linknet_gaussian")
    parser.add_argument("--mit", type=int, default=0, help="Number of samples from MIT dataset to add to training set")
    parser.add_argument("--deepglobe", type=int, default=0, help="Number of samples from DeepGlobe dataset to add to training set")
    parser.add_argument("--loss", type=str, default="combo", help="Loss function to use: e.g: dice_bce / focal_loss / dice_focal_loss")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for k-fold cross validation")
    
    args = parser.parse_args()
    
    # Instantiate the selected model
    ModelClass = model_dict[args.model]
    model = ModelClass(num_classes=1)

    train(model,epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,nb_samples = args.deepglobe,loss_name=args.loss,nb_samples_mit=args.mit,k_folds=args.k_folds)
    
    