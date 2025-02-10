import cv2
import numpy as np
from tifffile import imread, imwrite
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from PIL import Image


# 5D TIFF
og_image = imread('C:\\Users\\aydar\\Desktop\\Tumor_Control7B_ech13_adj.tif')
saliency_map_5d = imread('C:\\Users\\aydar\\Desktop\\Tumor_Control7B_ech13_adj_saliency.tif')
saliency_map_5d = saliency_map_5d.astype(np.float32, copy=False)

print(saliency_map_5d.shape)  # Should print: (slices number, 1, 1, 224, 224)

mask_5d = np.empty_like(saliency_map_5d, dtype=np.uint8)

for i in range(saliency_map_5d.shape[0]):
    # get the saliency map for this slice
    saliency_map = saliency_map_5d[i, 0, 0]
    #saliency_map = saliency_map_5d[i]

    # MIN-MAX Normalization
    normalized_saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    inverted_saliency_map = cv2.bitwise_not(normalized_saliency_map)
    _, mask = cv2.threshold(inverted_saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary_fill_holes(mask).astype(np.uint8) * 255
    labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)
    
    if num_labels > 1:
        regions = regionprops(labeled_mask)
        largest_blob = max(regions, key=lambda region: region.area)
        mask = (labeled_mask == largest_blob.label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, 0, 255, thickness=-1)

    mask_5d[i, 0, 0] = filled_mask
    mask_5d=mask_5d.astype(bool)
    #mask_5d[i] = filled_mask 

#imwrite('filled_roi1.tiff', mask_5d)
og_image_resized = np.empty((og_image.shape[0], mask_5d.shape[3], mask_5d.shape[4]))

for i in range(og_image.shape[0]):
    img_pil = Image.fromarray(og_image[i])
    img_pil_resized = img_pil.resize((mask_5d.shape[3], mask_5d.shape[4]))
    og_image_resized[i] = np.array(img_pil_resized)
    
mask_squeezed = np.squeeze(mask_5d)  # shape: (1648, 224, 224)
og_image_resized[~mask_squeezed] = 0

imwrite('og_image_clipped.tif', og_image_resized)
