import os
import imageio.v2 as imageio  # Make sure to use imageio v2 to avoid deprecation warnings
from imgaug import augmenters as iaa
from tqdm import tqdm

NB_AUGMENTATIONS = 3  # Number of times to augment each image


def augment_and_save(
    image_path, ground_truth_path, index, aug_index, output_image_dir, output_gt_dir
):
    """Augment an image and ground truth and save them to the specified output directories
    Args:
        image_path (str): Path to the image to augment
        ground_truth_path (str): Path to the ground truth to augment
        index (int): Index of the image
        aug_index (int): Index of the augmentation
        output_image_dir (str): Path to the output directory for augmented images
        output_gt_dir (str): Path to the output directory for augmented ground truths
    Returns:
        None
    """
    # Load image and ground truth
    image = imageio.imread(image_path)
    ground_truth = imageio.imread(ground_truth_path)

    # Define augmentations
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips with probability 0.5
            iaa.Flipud(0.5),  # vertical flips with probability 0.5
            iaa.Affine(
                rotate=(-180, 180),  # random rotations between -180 and 180 degrees
                scale={
                    "x": (0.7, 1.3),
                    "y": (0.7, 1.3),
                },  # random scaling between 0.7 and 1.3
                mode="symmetric",
            ),  # use symmetric padding for image borders after rotation
        ]
    )

    # Define augmentations for brightness and contrast ( will not be applied to ground truth )
    seq_plus = iaa.Sequential(
        [
            iaa.Multiply((0.9, 1.1)),  # change brightness
            iaa.LinearContrast((0.9, 1.1)),  # improve or worsen the contrast
        ]
    )

    # Apply transformations
    seq_det = seq.to_deterministic()
    transformed_image = seq_det.augment_image(image)
    transformed_image = seq_plus.augment_image(transformed_image)

    transformed_ground_truth = seq_det.augment_image(ground_truth)

    # Create the output directories if they do not exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)

    # Filenames for augmented images and ground truths
    augmented_image_filename = f"augmented_image_{index}_{aug_index}.png"
    augmented_ground_truth_filename = f"augmented_ground_truth_{index}_{aug_index}.png"

    # Save paths for augmented images and ground truths in their specified output directories
    augmented_image_path = os.path.join(output_image_dir, augmented_image_filename)
    augmented_ground_truth_path = os.path.join(
        output_gt_dir, augmented_ground_truth_filename
    )

    # Save augmented images and ground truths
    imageio.imwrite(augmented_image_path, transformed_image)
    imageio.imwrite(augmented_ground_truth_path, transformed_ground_truth)


def main():
    # Define paths
    images_dir = "data/training/images"
    ground_truth_dir = "data/training/labels"
    output_dir = "data/augmented"

    # Specific output directories for images and ground truths
    output_image_dir = os.path.join(output_dir, "images")
    output_gt_dir = os.path.join(output_dir, "labels")
    filenames = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    # Process each image
    for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        for aug_index in range(
            NB_AUGMENTATIONS
        ):  # Augment each image randomly multiple times( This hyperparameter can be adjusted to include more random rotations )
            augment_and_save(
                image_path,
                ground_truth_path,
                index,
                aug_index,
                output_image_dir,
                output_gt_dir,
            )


if __name__ == "__main__":
    main()
