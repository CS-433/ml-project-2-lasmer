import os
import random
import imageio
from imgaug import augmenters as iaa

def augment_and_save(image_path, ground_truth_path, index, aug_index):
    # Load images
    image = imageio.imread(image_path)
    ground_truth = imageio.imread(ground_truth_path)

    # Define augmentations: horizeontal flips ,vertical flips, rotation, brightness and contrast
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  
        iaa.Flipud(0.5),  
        iaa.Affine(
            rotate=(-180, 180),
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},  
            mode='symmetric'
        ),
        iaa.Multiply((0.9, 1.1)),
        iaa.LinearContrast((0.9, 1.1))  
    ])

    # Apply transformations
    seq_det = seq.to_deterministic()
    transformed_image = seq_det.augment_image(image)
    transformed_ground_truth = seq_det.augment_image(ground_truth)

    # save in the same directories
    augmented_image_dir = os.path.dirname(image_path)
    augmented_ground_truth_dir = os.path.dirname(ground_truth_path)

    augmented_image_filename = f'augmented_image_{index}_{aug_index}.png'
    augmented_ground_truth_filename = f'augmented_image_{index}_{aug_index}.png'

    augmented_image_path = os.path.join(augmented_image_dir, augmented_image_filename)
    augmented_ground_truth_path = os.path.join(augmented_ground_truth_dir, augmented_ground_truth_filename)
    
    imageio.imwrite(augmented_image_path, transformed_image)
    imageio.imwrite(augmented_ground_truth_path, transformed_ground_truth)


def main():
    images_dir = "training/images"
    ground_truth_dir = "training/groundtruth"

    # Process each image
    for index, filename in enumerate(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        for aug_index in range(3):  
            augment_and_save(image_path, ground_truth_path, index, aug_index)

if __name__ == "__main__":
    main()

