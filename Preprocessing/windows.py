# Emile Aydar
# Fiber Detection in Cold-Plasma treated lung tumors
# LPP/LIP6 || Ecole Polytechnique/Sorbonne Université, 2023

# Credits : Code inspired and modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py
# Copyright (C) 2011, the scikit-image team All rights reserved.


# Construction of a 3D sliding window for Tridimensional CLAHE

import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided


def get_3d_windowed_view(input_array, tile_shape, step_size=1):
    """
    Generate a 3D windowed view of an input 3D array.

    Parameters:
    - input_array: numpy.ndarray
        The 3D array for which the windowed view is to be generated.
    - tile_shape: int or tuple of int
        Shape of the 3D window (tile) for generating the view.
    - step_size: int or tuple of int, optional
        Step size for the rolling window. Default is 1.

    Returns:
    - numpy.ndarray
        A windowed view of the input array.

    Raises:
    - TypeError: If input_array is not a numpy ndarray.
    - ValueError: For issues related to the shape or dimension of the input.
    """

    if not isinstance(input_array, np.ndarray):
        raise TypeError("Expected a numpy ndarray for input.")

    if input_array.ndim != 3:
        raise ValueError("Input array must be 3D.")

    if isinstance(tile_shape, numbers.Number):
        tile_shape = (tile_shape, tile_shape, tile_shape)
    elif len(tile_shape) != 3:
        raise ValueError("Tile shape is not compatible with input array shape.")

    if isinstance(step_size, numbers.Number):
        if step_size < 1:
            raise ValueError("Step size should be >= 1.")
        step_size = (step_size, step_size, step_size)
    elif len(step_size) != 3:
        raise ValueError("Step size is not compatible with input array shape.")

    array_shape = np.array(input_array.shape)
    tile_shape_arr = np.array(tile_shape, dtype=int)

    if (array_shape - tile_shape_arr < 0).any():
        raise ValueError("Tile shape exceeds array dimensions.")
    if (tile_shape_arr - 1 < 0).any():
        raise ValueError("Tile shape is too small.")

    window_slices = tuple(slice(None, None, s) for s in step_size)
    tile_strides = np.array(input_array.strides)
    slice_strides = input_array[window_slices].strides
    tile_indices_shape = (array_shape - tile_shape_arr) // np.array(step_size) + 1

    final_shape = tuple(list(tile_indices_shape) + list(tile_shape))
    final_strides = tuple(list(slice_strides) + list(tile_strides))
    windowed_view = as_strided(input_array, shape=final_shape, strides=final_strides)

    return windowed_view