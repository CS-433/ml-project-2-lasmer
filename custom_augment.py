import os
import imageio.v2 as imageio  # Updated import statement
from imgaug import augmenters as iaa
from tqdm import tqdm

def rotate_and_transform(image, ground_truth, rotation_degree):
    # Define augmentations
    seq_image = iaa.Sequential([
        iaa.Affine(rotate=rotation_degree,  # Rotate by specified degree
                    scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},  
                    mode='symmetric'),  
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Gaussian noise
        iaa.SaltAndPepper(0.01)  # Salt and Pepper noise
    ])
    
    seq_gt = iaa.Sequential([
        iaa.Affine(rotate=rotation_degree,  # Rotate by specified degree
                    scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},  
                    mode='symmetric')
    ])

    # Apply transformations
    transformed_image = seq_image.augment_image(image)
    transformed_ground_truth = seq_gt.augment_image(ground_truth)

    return transformed_image, transformed_ground_truth

def augment_and_save(image_path, ground_truth_path, index, rotations, output_dir):
    image = imageio.imread(image_path)
    ground_truth = imageio.imread(ground_truth_path)

    # Make sure the output directories exist
    augmented_image_dir = os.path.join(output_dir, 'augmented_images')
    augmented_ground_truth_dir = os.path.join(output_dir, 'augmented_ground_truth')
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_ground_truth_dir, exist_ok=True)

    for aug_index, rotation_degree in enumerate(rotations):
        transformed_image, transformed_ground_truth = rotate_and_transform(image, ground_truth, rotation_degree)

        # Filenames for augmented images
        augmented_image_filename = f'new_image_{index}_{aug_index}.png'
        augmented_ground_truth_filename = f'new_ground_truth_{index}_{aug_index}.png'

        # Save paths for augmented images
        augmented_image_path = os.path.join(augmented_image_dir, augmented_image_filename)
        augmented_ground_truth_path = os.path.join(augmented_ground_truth_dir, augmented_ground_truth_filename)

        # Save augmented images
        imageio.imwrite(augmented_image_path, transformed_image)
        imageio.imwrite(augmented_ground_truth_path, transformed_ground_truth)

def main():
    images_dir = "data/training/images"
    ground_truth_dir = "data/training/groundtruth"
    output_dir = "data/augmented"  # Specify the directory where augmented images should be saved
    
    rotations = [15, 30, 45, 60, 90, 180, 270]  # List of rotations
    filenames = os.listdir(images_dir)

    for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
        image_path = os.path.join(images_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)

        augment_and_save(image_path, ground_truth_path, index, rotations, output_dir)

if __name__ == "__main__":
    main()
