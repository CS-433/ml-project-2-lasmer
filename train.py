from helpers import *
import os
import numpy as np
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


THRESHOLD = 0.5  # Threshold for converting predictions to binary values


def train(model, batch_size=8, epochs=50, lr=1e-4, loss_name="dice"):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():  # Uncomment the following two line if you work with M1 chip Macbook
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
    train_dataset = SatelliteDataset(
        "data/training/images",
        "data/training/labels",
        transform=transform,
        resize=resize,
    )
    
    val_dataset = SatelliteDataset(
        "data/validation/images",
        "data/validation/labels",
        transform=transform,
        resize=resize,
    )

    print("length of the training dataset :", len(train_dataset))
    print("length of the validation dataset :", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    best_f1_score = 0.0
    train_losses = []
    val_losses = []
    f1_scores = []
    val_labels_all, val_preds_all = [], []

    for epoch in range(epochs):
        print("-" * 20, "Epoch {}/{}\n".format(epoch, epochs - 1))
        since = time.time()
        ########################################################################################################################################
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training Batches"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, loss_name)
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
        val_preds = []
        val_targets = []
        for inputs, labels in tqdm(val_dataloader, desc="Validation Batches"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, loss_name)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                val_preds.append(outputs > THRESHOLD)  # Threshold predictions
                val_targets.append(labels > THRESHOLD)

        # Store predictions and labels
        val_epoch_loss = val_loss / val_samples
        val_preds = torch.cat(val_preds).view(-1).cpu().numpy()
        val_targets = torch.cat(val_targets).view(-1).cpu().numpy()
        val_f1_score = f1_score(val_targets, val_preds, average="binary")
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
        if best_f1_score < val_f1_score:
            best_f1_score = val_f1_score
            save_model(model, savepath=savepath, model_name=model_name)
            print(
                "New best model {} saved with f1 score: {:.4f}".format(
                    os.path.join(savepath, model_name), best_f1_score
                )
            )

        time_elapsed = time.time() - since
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    # save training and validation losses and f1 scores to csv file
    save_losses(
        train_losses,
        val_losses,
        f1_scores,
        savepath=os.path.join(savepath, model_name[:-3]),
    )  # remove .pt from model_name
    return model, train_losses, val_losses


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
        default=10,
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
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
        default="dice",
        help="Loss function to use: e.g: dice_bce / focal_loss / dice_focal_loss",
    )
    args = parser.parse_args()

    print(
        f"Training model {args.model} with {args.loss} loss for {args.epochs} epochs with learning rate {args.lr} and batch size {args.batch_size}"
    )
    model, train_losses, val_losses = train(
        args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        loss_name=args.loss,
    )
    plot(train_losses, val_losses)
