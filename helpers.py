import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import torch

# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1)) 
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


# Extract features for a given image
def extract_img_features(filename,patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X

def value_to_class(v,foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def img_float_to_uint8_alpha(img):
    # Assuming the image is in float format with values in [0, 1]
    rimg = img.copy() * 255.0
    rimg = np.round(rimg).astype(np.uint8)
    return rimg

def make_img_overlay_alpha(img, predicted_img):
    # Extract the alpha channel from the original image
    alpha_channel = img[:, :, 3]
    
    # Create the color mask for the overlay
    # Assuming the mask is represented by the red channel in predicted_img
    color_mask = np.zeros((400, 400, 4), dtype=np.uint8)  # Change to 4 channels for RGBA
    color_mask[:, :, 0] = predicted_img[:, :, 0]  # Red channel for mask
    color_mask[:, :, 3] = alpha_channel  # Alpha channel from the original image
    
    # Convert the float image to uint8
    img_uint8 = img_float_to_uint8_alpha(img[:, :, :3])  # Use only the first three channels (RGB)
    
    # Create images using PIL
    background = Image.fromarray(img_uint8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGBA")
    
    # Blend the images using the specified alpha
    new_img = Image.blend(background, overlay, 0.2)
    
    return new_img



def load_model( model, savepath="models/"): 
    model.load_state_dict(torch.load(savepath))
    return

def save_model(model, savepath="models", model_name="best_model.pt"):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    full_save_path = os.path.join(savepath, model_name)
    torch.save(model.state_dict(), full_save_path)
    print(f"Model saved at: {full_save_path}")

def img_to_class(img,foreground_threshold):
    img[img > foreground_threshold] = 1
    img[img <= foreground_threshold] = 0
    return img


def IoU(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union