from helpers import *
import os
import matplotlib.pyplot as plt
from Networks.common.custom_loss import *
from Networks.dinknet import *
from Networks.UNet import *
from Networks.GCDCNN import *
from Networks.nllinknet_location import *
from Networks.nllinknet_pairwise_func import *
from Loader import *
import argparse
import ttach as tta
from tqdm import tqdm

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


def predict(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    def load_unet():
        """Load the UNet model

        Returns:
            model: trained model
        """
        model = UNet(n_channels=3, num_classes=1).to(device)
        model.load_state_dict(torch.load("models/trained_model_UNet.pt"))
        if args.use_TTA:
            return tta.SegmentationTTAWrapper(
                model, tta.aliases.d4_transform(), merge_mode="mean"
            )
        else:
            return model

    def load_gcdcnn():
        """Load the GCDCNN model

        Returns:
            model: trained model
        """
        model = GCDCNN(n_channels=3, num_classes=1).to(device)
        model.load_state_dict(torch.load("models/trained_model_GCDCNN.pt"))
        if args.use_TTA:
            return tta.SegmentationTTAWrapper(
                model, tta.aliases.d4_transform(), merge_mode="mean"
            )
        else:
            return model

    def load_linknet():
        """Load the LinkNet model

        Returns:
            model: trained model
        """
        model = NL_LinkNet_EGaussian().to(device)
        model.load_state_dict(
            torch.load("models/trained_model_nl_linknet_egaussian.pt")
        )
        if args.use_TTA:
            return tta.SegmentationTTAWrapper(
                model, tta.aliases.d4_transform(), merge_mode="mean"
            )
        else:
            return model

    def output_croped(model, image, step=400):
        """Predict the mask for an image by cropping it into patches of size
           stepxstep and then predicting the mask for each patch

        Args:
            model: trained model
            image: image to predict the mask for
            step: size of the patches

        Returns:
            full_mask: predicted mask for the image
        """
        original_size = image.shape
        # Create an empty mask for the entire image and a counter for the number of overlapping patches
        full_mask = torch.zeros(1, 1, original_size[2], original_size[3]).to(device)
        overlap_count = torch.zeros(1, 1, original_size[2], original_size[3]).to(device)

        # Iterate over the image in patches of size stepxstep and predict the mask for each patch
        for x in range(0, original_size[2], step):
            for y in range(0, original_size[3], step):
                patch = image[:, :, y : y + step, x : x + step]
                output = model(patch)
                full_mask[:, :, y : y + step, x : x + step] += output
                overlap_count[:, :, y : y + step, x : x + step] += 1

        # Normalize the mask by dividing each pixel by the number of overlapping patches
        full_mask /= overlap_count
        return full_mask

    models = {}
    print(args)
    if args.use_unet:
        models["unet"] = load_unet().eval()
    if args.use_GCDCNN:
        models["GCDCNN"] = load_gcdcnn().eval()
    if args.use_linknet:
        models["linknet"] = load_linknet().eval()

    # load the test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # check if the test_set_masks folder exists, if not create it
    if not os.path.exists("test_set_masks"):
        os.makedirs("test_set_masks")
    else:
        # delete the contents of the test_set_masks folder
        for file in os.listdir("test_set_masks"):
            os.remove(os.path.join("test_set_masks", file))

    # load the test dataset

    test_dataset = testDataset("test_set_images", transform=transform)

    for i in tqdm(range(len(test_dataset)), desc="Predicting test images"):
        image = test_dataset[i].unsqueeze(0).to(device)
        outputs = []
        with torch.no_grad():
            if args.use_unet:
                if args.use_crop:
                    outputs.append(output_croped(models["unet"], image))
                else:
                    outputs.append(models["unet"](image))
            if args.use_GCDCNN:
                if args.use_crop:
                    outputs.append(output_croped(models["GCDCNN"], image))
                else:
                    outputs.append(models["GCDCNN"](image))
            if args.use_linknet:
                if args.use_crop:
                    outputs.append(output_croped(models["linknet"], image))
                else:
                    outputs.append(models["linknet"](image))

        output = torch.mean(torch.stack(outputs), dim=0)
        output = output > 0.5
        output = output.squeeze().cpu().numpy()

        plt.imsave(f"test_set_masks/test_{i+1}_mask.png", output, cmap="gray")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict the test images using the trained model(s).if you want to use multiple models, use --use_unet, --use_GCDCNN, --use_linknet and set them to true"
    )
    parser.add_argument(
        "--use_unet",
        type=bool,
        default=False,
        help="use true if you want to use unet (default: True)",
    )
    parser.add_argument(
        "--use_GCDCNN",
        type=bool,
        default=False,
        help="use true if you want to use GCDCNN (default: False)",
    )
    parser.add_argument(
        "--use_linknet",
        type=bool,
        default=True,
        help="use true if you want to use linknet (default: False)",
    )
    parser.add_argument(
        "--use_crop",
        type=bool,
        default=False,
        help="use true if you want to use crop (default: False)",
    )
    parser.add_argument(
        "--use_TTA",
        type=bool,
        default=False,
        help="use true if you want to use TTA (default: False)",
    )

    args = parser.parse_args()

    predict(args)
