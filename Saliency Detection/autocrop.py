import cv2
import numpy as np
from tifffile import imread, imwrite
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from PIL import Image


# Read the 5D TIFF
og_image = imread('C:\\Users\\aydar\\Desktop\\Tumor_Control7B_ech13_adj.tif')
saliency_map_5d = imread('C:\\Users\\aydar\\Desktop\\Tumor_Control7B_ech13_adj_saliency.tif')
saliency_map_5d = saliency_map_5d.astype(np.float32, copy=False)

# check the shape
print(saliency_map_5d.shape)  # Should print: (slices number, 1, 1, 224, 224)

# empty array for the 5D mask
mask_5d = np.empty_like(saliency_map_5d, dtype=np.uint8)

# loop over each slice
for i in range(saliency_map_5d.shape[0]):
    # get the saliency map for this slice
    saliency_map = saliency_map_5d[i, 0, 0]
    #saliency_map = saliency_map_5d[i]

    # MIN-MAX Normalization to 0-255 and convert to 8-bit unsigned integer
    normalized_saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # invert normalized saliency map
    inverted_saliency_map = cv2.bitwise_not(normalized_saliency_map)

    # thresholding to create the mask for this slice
    _, mask = cv2.threshold(inverted_saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fill holes in the mask
    mask = binary_fill_holes(mask).astype(np.uint8) * 255

    # label connected components
    labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

    # if there are multiple blobs, we only keep the largest one
    if num_labels > 1:
        # measurement of the area of each blob
        regions = regionprops(labeled_mask)

        # largest blob is the biggest region
        largest_blob = max(regions, key=lambda region: region.area)

        # new mask with only the largest blob
        mask = (labeled_mask == largest_blob.label).astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area, descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # empty mask to draw the filled contour
    filled_mask = np.zeros_like(mask)

    # largest contour (index 0) on the mask, filled with white (255)
    cv2.drawContours(filled_mask, contours, 0, 255, thickness=-1)

    # store the filled mask in the 5D mask array
    mask_5d[i, 0, 0] = filled_mask
    mask_5d=mask_5d.astype(bool)
    #mask_5d[i] = filled_mask  #Use this line after cropping the original stack


# now mask_5d is a 5D mask with filled ROI
# save 5D mask to a new TIFF file
#imwrite('filled_roi1.tiff', mask_5d)
# empty array for the resized images
og_image_resized = np.empty((og_image.shape[0], mask_5d.shape[3], mask_5d.shape[4]))

# loop over all images in the original stack
for i in range(og_image.shape[0]):
    # convert the 2D numpy array image to a PIL image
    img_pil = Image.fromarray(og_image[i])

    # resizing
    img_pil_resized = img_pil.resize((mask_5d.shape[3], mask_5d.shape[4]))

    # convert the PIL image back to a numpy array and store in the new array
    og_image_resized[i] = np.array(img_pil_resized)

# squeeze the mask so it has the same number of dimensions as the resized original image
mask_squeezed = np.squeeze(mask_5d)  # shape: (1648, 224, 224)

# set the pixel values on the resized original images that are OUTSIDE the mask region to 0
og_image_resized[~mask_squeezed] = 0

imwrite('og_image_clipped.tif', og_image_resized)
