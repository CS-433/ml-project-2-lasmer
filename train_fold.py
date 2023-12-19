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
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

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

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train_kfold(model_class, batch_size=8, epochs=50, lr=1e-4, loss_name="combo", k_folds=5):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    
    # Create dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SatelliteDataset("training/608by608Images", "training/608by608LabelsBW", transform=transform)
    print("Samples from dataset:", len(dataset))

    # K-fold Cross Validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # DataLoaders for the current fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Initialize model for current fold
        model = model_class(num_classes=1).to(device)
        model.apply(reset_weights)

        # Loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        calc_loss = CustomLoss(beta=0.8)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for inputs, labels in tqdm(train_dataloader, desc="Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, loss_name)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_epoch_loss = train_loss / len(train_ids)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, loss_name)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.append(outputs > 0.5)
                    val_labels.append(labels > 0.5)

            val_epoch_loss = val_loss / len(test_ids)
            val_f1_score = f1_score(
                torch.cat(val_labels).view(-1).cpu().numpy(),
                torch.cat(val_preds).view(-1).cpu().numpy(),
                average='binary'
            )
            
            print(f"Fold {fold}, Epoch {epoch}, Training Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1: {val_f1_score:.4f}")

            scheduler.step()

        # Save the model for the current fold
        fold_savepath = os.path.join("models", f"model_fold_{fold}.pt")
        torch.save(model.state_dict(), fold_savepath)

        # Record the performance of this fold
        fold_results[fold] = val_f1_score

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    for key, value in fold_results.items():
        print(f'Fold {key}: F1 Score {value}')
    avg_f1 = sum(fold_results.values()) / len(fold_results)
    print(f'Average F1 Score: {avg_f1}')

    return fold_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with k-fold cross validation for road segmentation.')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train (default: 70)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument("--model", type=str, default="nl34_linknet", choices=model_dict.keys(), help="Model to train")
    parser.add_argument("--loss", type=str, default="combo", help="Loss function to use")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for k-fold cross validation")
    
    args = parser.parse_args()
    
    ModelClass = model_dict[args.model]
    train_kfold(ModelClass, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, loss_name=args.loss, k_folds=args.k_folds)