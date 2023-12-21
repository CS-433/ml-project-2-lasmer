from train import train
from predict import predict
import argparse


if __name__ == "__main__":
    # Change the parameters depending on which model and hyermarameter you want to use
    batch_size = 50
    epochs = 1
    lr = 3e-4
    model_name = "UNet"
    loss_name = "dice"
    
    print(f"Training model {model_name} with {loss_name} loss for {epochs} epochs with learning rate {lr} and batch size {batch_size}")
    model, train_losses, val_losses = train(
        model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        loss_name=loss_name,
    )
    print("Finished training!")
    print("Predicting test images...")

    # Change the parameters depending on which models you want to use
    args = {
        "use_unet": True,
        "use_GCDCNN": False,
        "use_linknet": True,
        "use_crop": False,
        "use_TTA": False,
    }
    
    # Call the predict function with your args
    args = argparse.Namespace(**args)
    predict(args)