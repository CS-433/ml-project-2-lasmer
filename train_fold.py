from helpers import *
import os
from Networks.common.custom_loss import *
from Networks.dinknet import *
from Networks.nllinknet_location import *
from Networks.nllinknet_pairwise_func import *
from Networks.UNet import *
from Networks.GCDCNN import *
from Loader import *
import time
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

THRESHOLD = 0.5  # Threshold for converting predictions to binary values
BETA = 0.8  # Beta parameter for the loss (defines the weight of the dice loss if hybrid with BCE is used)


def reset_weights(model):
    """Reset model weights to avoid weight leakage.
    Args:
        m: model
    Returns:
        None"""
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def train(model, batch_size=8, epochs=50, lr=1e-4, loss_name="combo", k_folds=5):
    """Train the model"""
    # Define device for training
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available(): # Uncomment the following two line if you work with M1 chip Mac
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device: {}".format(device))

    savepath = "models"
    model_name = "trained_model_" + str(model) + ".pt"
    ########################################################################################################################################
    ## Create dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )  # Convert PIL Images to tensors
    resize = False if model in ["UNet", "GCDCNN"] else True
    dataset = SatelliteDataset(
        "data/training/images",
        "data/training/labels",
        transform=transform,
        resize=resize,
    )
    print("Total training dataset :", len(dataset))

    ########################################################################################################################################
    # Create the selected model
    ModelClass = model_dict[model]
    model = ModelClass(num_classes=1)
    model = model.to(device)

    ########################################################################################################################################
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    calc_loss = CustomLoss(beta=0.8)

    ########################################################################################################################################
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f"FOLD {fold}/{k_folds-1}")
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler
        )
        val_dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler
        )

        # Init the neural network
        network = model.to(device)
        network.apply(reset_weights)
        network.train()  # Set network in training mode

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)  # Optimizer
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.9
        )  # Scheduler for learning rate decay

        ########################################################################################################################################
        # Run the training loop for defined number of epochs
        train_losses = []
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}")
            print("-" * 20)
            start_time = time.time()
            current_loss = 0.0  # Set current loss value

            for i, data in tqdm(enumerate(train_dataloader, 0)):
                inputs, targets = data  # Get inputs
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = calc_loss(outputs, targets, loss_name)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            train_epoch_loss = current_loss / len(train_dataloader)
            train_losses.append(train_epoch_loss)
            print(f"Epoch {epoch}/{epochs - 1} - Training Loss: {train_epoch_loss:.4f}")
            current_loss = 0.0
            print("Time for epoch: %.3fs" % (time.time() - start_time))
            ########################################################################################################################################
            # Perform validation for epoch with eval mode (Can be commented out to speed up training)
            network.eval()
            val_losses = []
            val_loss, val_samples = 0.0, 0
            val_preds = []
            val_targets = []

            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.set_grad_enabled(False):
                    outputs = network(inputs)
                    loss = calc_loss(outputs, labels, loss_name)
                    val_loss += loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)
                    val_targets.append((labels > THRESHOLD))
                    val_preds.append((outputs > THRESHOLD))

            # Calculate and print F1 score at the end of each validation phase
            val_f1_score = f1_score(
                torch.cat(val_targets).view(-1).cpu().numpy(),
                torch.cat(val_preds).view(-1).cpu().numpy(),
                average="binary",
            )
            val_epoch_loss = val_loss / val_samples
            val_losses.append(val_epoch_loss)
            iou_score = IoU(
                torch.cat(val_targets).view(-1).cpu().numpy(),
                torch.cat(val_preds).view(-1).cpu().numpy(),
            )

            print(
                f"Epoch {epoch}/{epochs - 1} - Validation Loss: {val_epoch_loss:.4f}, F1 Score: {val_f1_score:.4f}, IoU score: { iou_score:.4f}"
            )

        ########################################################################################################################################
        # Saving the model
        save_model(model, savepath=savepath, model_name=f"fold{fold}_" + model_name)
        network.eval()
        targets, predicted = [], []

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_dataloader, 0)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                predicted.append(
                    (outputs > THRESHOLD)
                )  # Threshold the outputs to obtain binary predictions
                targets.append((labels > THRESHOLD))  # Convert targets to binary values

            f1 = f1_score(
                torch.cat(targets).view(-1).cpu().numpy(),
                torch.cat(predicted).view(-1).cpu().numpy(),
                average="binary",
            )
            print("F1 Score for fold %d: %.4f" % (fold, f1))
            print("--------------------------------")
            results[fold] = f1
        scheduler.step()  # Perform learning rate decay

    # Print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        sum += value
    print(f"Average F1: {sum/len(results.items())} %")

    return network


# Define a dictionary mapping model type names to model classes
model_dict = {
    "dinknet34": DinkNet34,
    "linknet34": LinkNet34,
    "baseline": Baseline,
    "nl3_linknet": NL3_LinkNet,
    "nl34_linknet": NL34_LinkNet,
    "nl_linknet_egaussian": NL_LinkNet_EGaussian,
    "nl_linknet_gaussian": NL_LinkNet_Gaussian,
    "UNet": UNet,
    "GCDCNN": GCDCNN,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for road segmentation.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs", type=int, default=70, help="number of epochs to train (default: 70)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        choices=model_dict.keys(),
        help="Model to train: e.g: dinknet / linknet / baseline / nl3_linknet / nl34_linknet / nl_linknet_egaussian / nl_linknet_gaussian",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="combo",
        help="Loss function to use: e.g: dice_bce / focal_loss / dice_focal_loss",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross validation",
    )

    args = parser.parse_args()
    print(
        f"Training model {args.model} with {args.loss} loss for {args.epochs} epochs with learning rate {args.lr} and batch size {args.batch_size}"
    )
    train(
        args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        loss_name=args.loss,
        k_folds=args.k_folds,
    )
