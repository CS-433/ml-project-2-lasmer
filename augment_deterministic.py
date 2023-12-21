import os
import imageio.v2 as imageio  # Updated import statement
from imgaug import augmenters as iaa
from tqdm import tqdm

GAUSSIAN_NOISE = 0.05  # Standard deviation of gaussian noise to add to images
SALT_PEPPER_NOISE = 0.01  # Proportion of salt and pepper noise to add to images


def rotate_and_transform(image, ground_truth, rotation_degree):
    """Rotate an image and ground truth by the specified degree and apply transformations
    Args:
        image (numpy.ndarray): Image to rotate and transform
        ground_truth (numpy.ndarray): Ground truth to rotate and transform
        rotation_degree (int): Degree to rotate the image and ground truth by
    Returns:
        transformed_image (numpy.ndarray): Rotated and transformed image
        transformed_ground_truth (numpy.ndarray): Rotated and transformed ground truth
    """

    # Define augmentations
    seq_image = iaa.Sequential(
        [
            iaa.Affine(
                rotate=rotation_degree, mode="symmetric"  # Rotate by specified degree
            ),  # Use symmetric padding for image borders after rotation
            iaa.AdditiveGaussianNoise(
                scale=(0, GAUSSIAN_NOISE * 255)
            ),  # Gaussian noise
            iaa.SaltAndPepper(SALT_PEPPER_NOISE),  # Salt and Pepper noise
        ]
    )

    seq_gt = iaa.Sequential(
        [
            iaa.Affine(
                rotate=rotation_degree, mode="symmetric"  # Rotate by specified degree
            )  # Use symmetric padding for image borders after rotation
        ]
    )

    # Apply transformations
    transformed_image = seq_image.augment_image(image)
    transformed_ground_truth = seq_gt.augment_image(ground_truth)

    return transformed_image, transformed_ground_truth


def augment_and_save(image_path, ground_truth_path, index, rotations, output_dir):
    """Augment an image and ground truth and save them to the specified output directories
    Args:
        image_path (str): Path to the image to augment
        ground_truth_path (str): Path to the ground truth to augment
        index (int): Index of the image
        rotations (list): List of rotations to apply to the image and ground truth
        output_dir (str): Path to the output directory for augmented images and ground truths
    Returns:
        None"""

    if len(rotations) == 0:
        return
    if not image_path.endswith(".png"):
        return

    # Load image and ground truth
    image = imageio.imread(image_path)
    ground_truth = imageio.imread(ground_truth_path)

    # Make sure the output directories exist
    augmented_image_dir = os.path.join(output_dir, "augmented_images")
    augmented_ground_truth_dir = os.path.join(output_dir, "augmented_ground_truth")
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_ground_truth_dir, exist_ok=True)

    for aug_index, rotation_degree in enumerate(rotations):
        transformed_image, transformed_ground_truth = rotate_and_transform(
            image, ground_truth, rotation_degree
        )

        # Filenames for augmented images
        augmented_image_filename = f"new_image_{index}_{aug_index}.png"
        augmented_ground_truth_filename = f"new_ground_truth_{index}_{aug_index}.png"

        # Save paths for augmented images
        augmented_image_path = os.path.join(
            augmented_image_dir, augmented_image_filename
        )
        augmented_ground_truth_path = os.path.join(
            augmented_ground_truth_dir, augmented_ground_truth_filename
        )

        # Save augmented images
        imageio.imwrite(augmented_image_path, transformed_image)
        imageio.imwrite(augmented_ground_truth_path, transformed_ground_truth)


def main():
    # List of rotations
    rotations = [15, 30, 45, 60, 90, 180, 270]

    ########################################################################################################################################
    ## Augment training dataset
    print("Augmenting training dataset...")
    images_dir = "data/training/images"
    ground_truth_dir = "data/training/labels"
    output_dir = "data_augmented/training"  # Specify the directory where augmented images should be saved

    filenames = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        augment_and_save(image_path, ground_truth_path, index, rotations, output_dir)

    ########################################################################################################################################
    ## Augment Validation dataset
    print("Augmenting validation dataset...")
    images_dir = "data/validation/images"
    ground_truth_dir = "data/validation/labels"
    output_dir = "data_augmented/validation"  # Specify the directory where augmented images should be saved

    filenames = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]
    for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        augment_and_save(image_path, ground_truth_path, index, rotations, output_dir)


if __name__ == "__main__":
    main()
