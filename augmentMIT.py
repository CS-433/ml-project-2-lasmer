from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import os
import numpy as np


def crop_and_convert_to_png(image_path, groundtruth_path, output_folder_images, output_folder_groundtruths, crop_size=(400, 400)):
    """
    Crop a TIFF image and its corresponding ground truth into several segments and save them as PNG.

    Parameters:
    image_path (Path): Path to the input TIFF image.
    groundtruth_path (Path): Path to the input TIFF ground truth.
    output_folder_images (Path): Folder where the cropped image PNGs will be saved.
    output_folder_groundtruths (Path): Folder where the cropped ground truth PNGs will be saved.
    crop_size (tuple, optional): Size of each cropped segment, default is (400, 400).
    """
    with Image.open(image_path) as img, Image.open(groundtruth_path) as gt:
        img_width, img_height = img.size
        gt_width, gt_height = gt.size

        # Ensure the image and ground truth sizes match
        if img_width != gt_width or img_height != gt_height:
            raise ValueError(f"Image and ground truth sizes do not match for {image_path.name} and {groundtruth_path.name}")

        # Calculate the number of segments in both dimensions
        x_segments = img_width // crop_size[0]
        y_segments = img_height // crop_size[1]

        # Crop and save each segment
        for x in range(x_segments):
            for y in range(y_segments):
                left = x * crop_size[0]
                upper = y * crop_size[1]
                right = left + crop_size[0]
                lower = upper + crop_size[1]

                cropped_img = img.crop((left, upper, right, lower))
                cropped_gt = gt.crop((left, upper, right, lower))

                cropped_img.save(output_folder_images / f"{image_path.stem}_{x}_{y}.png")
                cropped_gt.save(output_folder_groundtruths / f"{groundtruth_path.stem}_{x}_{y}.png")


def delete_images_with_too_many_whites(image_dir, gt_dir, threshold):
    """
    Deletes images and their corresponding ground truths if the images 
    have more than a specified threshold of white pixels.
    
    :param image_dir: Path to the directory containing the images.
    :param gt_dir: Path to the directory containing the ground truth images.
    :param threshold: The threshold for the proportion of white pixels.
                      For example, 0.1 means 10% of the pixels.
    """
    len_deleted = 0
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        gt_path = gt_path.replace(".tiff", ".tif")
        # Check if the corresponding ground truth file exists
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {filename}, skipping.")
            continue

        # Read the image and convert to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {filename}, skipping.")
            continue

        # Calculate the proportion of white pixels
        white_pixel_ratio = np.sum(image == 255) / image.size

        # Delete the image and ground truth if the condition is met
        if white_pixel_ratio > threshold:
            len_deleted += 1
            os.remove(image_path)
            os.remove(gt_path)
            print(f"Deleted {filename} and its ground truth due to high white pixel ratio.")
    print(f"Deleted {len_deleted} images and ground truths.")

# Directories
input_folder = Path('archive/tiff/train')  # Replace with your input folder path for images
input_folder_gt = Path('archive/tiff/train_labels')  # Replace with your input folder path for ground truths

# Delete images with too many white pixels
delete_images_with_too_many_whites(input_folder, input_folder_gt, 0.1)

output_folder_images = Path('data/MIT/training/images')  # Replace with your output folder path for images
output_folder_groundtruths = Path('data/MIT/training/groundtruth')  # Replace with your output folder path for ground truths



# Ensure output folders exist
output_folder_images.mkdir(parents=True, exist_ok=True)
output_folder_groundtruths.mkdir(parents=True, exist_ok=True)

# # Assuming filenames without extensions are identical for images and groundtruths
# tiff_files = list(input_folder.glob('*.tiff'))
# tiff_files_gt = list(input_folder_gt.glob('*.tif'))

# # Verify that the number of images and groundtruth files are the same
# if len(tiff_files) != len(tiff_files_gt):
#     raise ValueError("The number of images and ground truth files do not match.")

# # Sort the lists to make sure they are in the same order
# tiff_files.sort()
# tiff_files_gt.sort()

# # Processing each image with its corresponding ground truth
# for image_path, groundtruth_path in tqdm(zip(tiff_files, tiff_files_gt), total=len(tiff_files), desc="Processing"):
#     crop_and_convert_to_png(image_path, groundtruth_path, output_folder_images, output_folder_groundtruths)

# print("All images and ground truths have been cropped and converted to PNG.")

# Delete final images with too many white pixels
delete_images_with_too_many_whites(output_folder_images, output_folder_groundtruths, 0.1)


