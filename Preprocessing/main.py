# Emile Aydar
# Fiber Detection in Cold-Plasma treated lung tumors
# LPP/ISIR || Ecole Polytechnique/Sorbonne Université, 2023

# Tridimensional CLAHE processing technique for 3D tiff stacks

import numpy as np
import tifffile as tiff
from skimage.exposure import equalize_adapthist
from windows import get_3d_windowed_view
import argparse


def equalize_3d_histogram(volume, clip_limit):
    """
    Apply 3D histogram equalization with a clip limit to a given volume.

    Parameters:
    - volume: 3D numpy array, the input volume.
    - clip_limit: float, the proportion of total pixel count that is the maximum allowed
      frequency for any intensity value in the histogram (default is 0.01).

    Returns:
    - equalized_volume: 3D numpy array, the histogram-equalized volume.
    """

    hist, bins = np.histogram(volume.flatten(), bins=256)

    total_pixels = volume.size
    actual_clip_limit = int(clip_limit * total_pixels)

    excess = sum([h - actual_clip_limit for h in hist if h > actual_clip_limit])
    avg_inc = int(excess / len(hist))
    for i in range(len(hist)):
        hist[i] = min(hist[i], actual_clip_limit)
    excess -= len(hist) * avg_inc
    for i in range(excess):
        hist[i % len(hist)] += 1

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    equalized_volume = np.interp(volume.flatten(), bins[:-1], cdf_normalized)

    return equalized_volume.reshape(volume.shape)


def normalize_image(image):
    """Normalize image values to [-1, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    if min_val == max_val:
        return image
    normalized = 2 * ((image - min_val) / (max_val - min_val)) - 1
    return normalized


def 3d_clahe(tiff_path, window_shape, step=1, clip_limit=0.01,
                                               save_path='3d_clahe_output.tif', debug=False):
    """
    Enhance the contrast of a 3D TIFF stack using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
    - tiff_path: str, path to the input 3D TIFF stack.
    - window_shape: tuple or int, the shape of the 3D window (tile) for CLAHE.
    - step: tuple or int, step size for rolling window. Default is 1.
    - clip_limit: float, clipping limit for CLAHE. Default is 0.01.
    - save_path: str, path to save the processed TIFF stack. Default is 'output.tif'.
    - debug: bool, if True, prints debugging information. Default is False.

    Returns:
    - None
    """
    tiff_path = input("Please enter the path to your TIFF file: ")
    try:
        stack = tiff.imread(tiff_path)
    except Exception as e:
        raise ValueError(f"Error reading the TIFF file: {e}")

    if not (isinstance(window_shape, (int, tuple)) and (isinstance(window_shape, int) or len(window_shape) == 3)):
        raise TypeError("window_shape must be an integer or a tuple of three integers.")

    if not (isinstance(step, (int, tuple)) and (isinstance(step, int) or len(step) == 3)):
        raise TypeError("step must be an integer or a tuple of three integers.")

    windows = get_3d_windowed_view(stack, window_shape, step)

    sum_array = np.zeros_like(stack, dtype=np.float32)
    count_array = np.zeros_like(stack)

    for index in np.ndindex(windows.shape[:3]):
        window = windows[index+(slice(None), slice(None), slice(None))]

        # Normalize the window
        window = normalize_image(window)

        # Apply true 3D histogram equalization to the entire window
        window = equalize_3d_histogram(window,clip_limit)

        x_start = index[0] * step
        x_end = min(x_start + window_shape[0], stack.shape[0])
        y_start = index[1] * step
        y_end = min(y_start + window_shape[1], stack.shape[1])
        z_start = index[2] * step
        z_end = min(z_start + window_shape[2], stack.shape[2])

        sum_array[x_start:x_end, y_start:y_end, z_start:z_end] += window
        count_array[x_start:x_end, y_start:y_end, z_start:z_end] += 1

        if debug:
            print(f"Processed window at index {index}")

    try:

        clahe_stack = sum_array / count_array
        tiff.imwrite(save_path, clahe_stack.astype(np.uint16))

    except Exception as e:
        raise ValueError(f"Error saving the TIFF file: {e}")

    if debug:
        print(f"Saved enhanced stack to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhance the contrast of a 3D TIFF stack using CLAHE.')

    parser.add_argument('tiff_path', type=str, help='Path to the input 3D TIFF stack.')
    parser.add_argument('window_shape', type=int, nargs=3, help='Shape of the 3D window (tile) for CLAHE.')
    parser.add_argument('--step', type=int, default=1, nargs='?', help='Step size for rolling window. Default is 1.')
    parser.add_argument('--clip_limit', type=float, default=0.01, help='Clipping limit for CLAHE. Default is 0.01.')
    parser.add_argument('--save_path', type=str, default='output.tif',
                        help='Path to save the processed TIFF stack. Default is output.tif.')
    parser.add_argument('--debug', action='store_true', help='Print debugging information.')

    args = parser.parse_args()

    3d_clahe(args.tiff_path, tuple(args.window_shape), args.step, args.clip_limit,
                                               args.save_path,
                                               args.debug)
