import cv2
import numpy as np
import cv2
import numpy as np
from scipy.interpolate import interp1d


def morphological_postprocessing(binary_mask, close=9, open=9):
    # Convert binary mask to uint8
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Define the structuring element for morphological operations
    # Adjust the kernel size based on expected road width
    # kernel_size = 20
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open, open))
    
    # Closing to close small gaps
    closed_mask = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, close_kernel)
    
    # Opening to remove small false positives
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)

    # Convert back to binary
    processed_mask = opened_mask / 255
    return processed_mask

def connected_components_postprocessing(binary_mask, min_size_threshold=200):
    # Run connected components on the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask * 255).astype(np.uint8), connectivity=8
    )
    
    # Define minimum size threshold
    # min_size_threshold = 500  # Adjust based on expected minimum road size

    # Filter out small components
    for i in range(1, num_labels):  # Start from 1 to skip the background label
        if stats[i, cv2.CC_STAT_AREA] < min_size_threshold:
            labels[labels == i] = 0

    # Convert labels back to binary mask
    processed_mask = (labels > 0).astype(np.uint8)
    return processed_mask



def fill_gaps_with_hough_transform_and_interpolation(mask):
    # Ensure mask is a binary image of type uint8
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask

    # Use HoughLinesP to detect lines
    lines = cv2.HoughLinesP(mask_uint8, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=50)

    # Initialize an empty mask to draw the filled in roads
    filled_mask = np.zeros_like(mask_uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filled_mask, (x1, y1), (x2, y2), (255), 3)

    # Find contours in the filled mask
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour and interpolate between endpoints
    for contour in contours:
        # Fit a line to the contour points and interpolate between the ends
        if len(contour) > 1:
            x = contour[:, 0][:, 0]
            y = contour[:, 0][:, 1]

            # Sort the contour points by x-coordinate for interpolation
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]

            # Perform linear interpolation between the contour points
            for i in range(1, len(x_sorted)):
                x_values = [x_sorted[i - 1], x_sorted[i]]
                y_values = [y_sorted[i - 1], y_sorted[i]]
                interpolate_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
                x_interp = np.arange(x_values[0], x_values[1] + 1)
                y_interp = interpolate_func(x_interp).astype(int)

                # Draw the interpolated points on the filled mask
                for x_i, y_i in zip(x_interp, y_interp):
                    filled_mask[y_i, x_i] = 255

    return filled_mask
