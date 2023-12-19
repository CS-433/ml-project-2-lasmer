import os
import imageio.v2 as imageio
from imgaug import augmenters as iaa
from tqdm import tqdm
import random


def augment_and_save(image_path, ground_truth_path, aug_index):
    # Load images
    image = imageio.imread(image_path)
    ground_truth = imageio.imread(ground_truth_path)

    # Define augmentations
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  
        iaa.Flipud(0.5),  
        iaa.Affine(
            rotate=(-180, 180),
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},  
            mode='symmetric')
       ] )
    
    seq_plus = iaa.Sequential([
        iaa.Multiply((0.9, 1.1)),
        iaa.LinearContrast((0.9, 1.1))  
    ])

    # Apply transformations
    seq_det = seq.to_deterministic()
    transformed_image = seq_det.augment_image(image)
    transformed_image = seq_plus.augment_image(transformed_image)
    transformed_ground_truth = seq_det.augment_image(ground_truth)

    # Filenames for augmented images and ground truths
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    augmented_image_filename = f'{base_filename}_augmented_{aug_index}.png'
    augmented_ground_truth_filename = f'{base_filename}_augmented_{aug_index}.png'

    # Save paths for augmented images and ground truths
    augmented_image_path = os.path.join(os.path.dirname(image_path), augmented_image_filename)
    augmented_ground_truth_path = os.path.join(os.path.dirname(ground_truth_path), augmented_ground_truth_filename)

    # Save augmented images and ground truths
    imageio.imwrite(augmented_image_path, transformed_image)
    imageio.imwrite(augmented_ground_truth_path, transformed_ground_truth)

def main():
    images_dir = "training/608by608Images"
    ground_truth_dir = "training/608by608LabelsBW"
    filenames = os.listdir(images_dir)

    # Process each image
    for filename in tqdm(filenames, total=len(filenames)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename.replace("_image_", "_labels_"))

        # Augment only 50% of the images
        if random.random() < 0.5:
            for aug_index in range(1):  
                augment_and_save(image_path, ground_truth_path, aug_index)

if __name__ == "__main__":
    main()