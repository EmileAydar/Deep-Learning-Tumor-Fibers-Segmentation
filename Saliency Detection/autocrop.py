import cv2
import numpy as np
from tifffile import imread, imwrite
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
# Read the 5D TIFF
saliency_map_5d = imread('C:\\Users\\yunus\\Desktop\\Tumor_Control7B_ech13_adj_saliency.tif')
saliency_map_5d = saliency_map_5d.astype(np.float32, copy=False)

# Check the shape
print(saliency_map_5d.shape)  # Should print: (1648, 1, 1, 224, 224)

# Prepare an empty array for the 5D mask
mask_5d = np.empty_like(saliency_map_5d, dtype=np.uint8)

# Loop over each slice
for i in range(saliency_map_5d.shape[0]):
    # Get the saliency map for this slice
    saliency_map = saliency_map_5d[i, 0, 0]
    #saliency_map = saliency_map_5d[i]

    # Normalize to 0-255 and convert to 8-bit unsigned integer
    normalized_saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Invert the normalized saliency map
    inverted_saliency_map = cv2.bitwise_not(normalized_saliency_map)

    # Apply thresholding to create the mask for this slice
    _, mask = cv2.threshold(inverted_saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes in the mask
    mask = binary_fill_holes(mask).astype(np.uint8) * 255

    # Label connected components
    labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

    # If there are multiple blobs, keep only the largest
    if num_labels > 1:
        # Measure the area of each blob
        regions = regionprops(labeled_mask)

        # Find the largest blob
        largest_blob = max(regions, key=lambda region: region.area)

        # Create a new mask with only the largest blob
        mask = (labeled_mask == largest_blob.label).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create an empty mask to draw the filled contour
    filled_mask = np.zeros_like(mask)

    # Draw the largest contour (index 0) on the mask, filled with white (255)
    cv2.drawContours(filled_mask, contours, 0, 255, thickness=-1)

    # Store the filled mask in the 5D mask array
    mask_5d[i, 0, 0] = filled_mask
    mask_5d=mask_5d.astype(np.float16)
    #mask_5d[i] = filled_mask  #Use this line after cropping the original stack


# Now mask_5d is a 5D mask with filled ROI
# Write the 5D mask to a new TIFF file
#imwrite('filled_roi1.tiff', mask_5d)

# Resize the mask to match the dimensions of the original stack
mask_resized = resize(mask_5d, (1648, 1, 1, 2706, 2586))

# Read the original stack
og_stack = imread('path_to_og_stack')

# Overlay the mask on the original stack
og_stack_masked = cv2.bitwise_and(og_stack, mask_resized)

# Crop the original stack around the mask
og_stack_cropped = og_stack[mask_resized > 0]

# Write the cropped original stack to a new TIFF file
imwrite('final_cropped_mask.tif', og_stack_cropped)
