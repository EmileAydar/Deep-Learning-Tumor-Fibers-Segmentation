import numpy as np
import tensorflow as tf
from temp import tf_batch_gather, tf_batch_histogram
from itertools import product
import tifffile
import os


def mclahe(x, kernel_size=None, n_bins=256, clip_limit=0.0001, adaptive_hist_range=True, use_gpu=True):
    """
    Multidimensional contrast limited adaptive histogram equalization implemented in tensorflow.

    Taken from Stimper.and al, adapted for TensorFlow v2 (the original code uses deprecated tensorflow methods and is not usable anymore in its original form)
    and improved for better memory management and synchrotron image processing.

    Adaptation for PyTorch Framework coming too..

    :param x: numpy array to which clahe is applied
    :param kernel_size: tuple of kernel sizes, 1/8 of dimension lengths of x if None
    :param n_bins: number of bins to be used in the histogram
    :param clip_limit: relative intensity limit to be ignored in the histogram equalization
    :param adaptive_hist_range: flag, if true individual range for histogram computation of each block is used
    :param use_gpu: Flag, if true gpu is used for computations if available
    :return: numpy array to which clahe was applied, scaled on interval [0, 1]
    """

    if kernel_size is None:
        kernel_size = tuple(s // 1 for s in x.shape)
    kernel_size = np.array(kernel_size)

    assert len(kernel_size) == len(x.shape)
    dim = len(x.shape)

    # Normalize data
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    x = (x - x_min) / (x_max - x_min)

    # Pad data
    x_shape = np.array(x.shape)
    padding_x_length = kernel_size - 1 - ((x_shape - 1) % kernel_size)
    padding_x = np.column_stack(((padding_x_length + 1) // 2, padding_x_length // 2))
    padding_hist = np.column_stack((kernel_size // 2, (kernel_size + 1) // 2)) + padding_x
    x_hist_padded = tf.pad(x, padding_hist, 'symmetric')

    tf_x_hist_padded = tf.Variable(initial_value=x_hist_padded, trainable=False)
    tf_x_padded = tf.slice(tf_x_hist_padded, kernel_size // 2, x_shape + padding_x_length)

    # Form blocks used for interpolation
    n_blocks = np.ceil(np.array(x.shape) / kernel_size).astype(np.int32)
    new_shape = np.reshape(np.column_stack((n_blocks, kernel_size)), (2 * dim,))
    perm = tuple(2 * i for i in range(dim)) + tuple(2 * i + 1 for i in range(dim))
    tf_x_block = tf.transpose(tf.reshape(tf_x_padded, new_shape), perm=perm)
    shape_x_block = np.concatenate((n_blocks, kernel_size))

    # Form block used for histogram
    n_blocks_hist = n_blocks + np.ones(dim, dtype=np.int32)
    new_shape = np.reshape(np.column_stack((n_blocks_hist, kernel_size)), (2 * dim,))
    perm = tuple(2 * i for i in range(dim)) + tuple(2 * i + 1 for i in range(dim))
    tf_x_hist = tf.transpose(tf.reshape(tf_x_hist_padded, new_shape), perm=perm)

    # Get maps
    # Get histogram
    if adaptive_hist_range:
        hist_ex_shape = np.concatenate((n_blocks_hist, [1] * dim))
        tf_x_hist_min = tf.Variable(initial_value=tf.zeros(n_blocks_hist), dtype=tf.float32, trainable=False)
        tf_x_hist_max = tf.reduce_max(tf_x_hist, axis=np.arange(-dim, 0))
        tf_x_hist_norm = tf.Variable(initial_value=tf.ones(n_blocks_hist), dtype=tf.float32, trainable=False)

        # Directly compute and assign values
        tf_x_hist_min.assign(tf.reduce_min(tf_x_hist, axis=np.arange(-dim, 0)))
        tf_x_hist_norm.assign(tf.where(tf.equal(tf_x_hist_min, tf_x_hist_max),
                                       tf.ones_like(tf_x_hist_min),
                                       tf_x_hist_max - tf_x_hist_min))
        tf_x_hist_scaled = (tf_x_hist - tf.reshape(tf_x_hist_min, hist_ex_shape)) / tf.reshape(tf_x_hist_norm, hist_ex_shape)
    else:
        tf_x_hist_scaled = tf_x_hist

    tf_hist = tf.cast(tf_batch_histogram(tf_x_hist_scaled, [0., 1.], dim, nbins=n_bins), tf.float32)

    # Clip histogram
    tf_n_to_high = tf.reduce_sum(tf.nn.relu(tf_hist - np.prod(kernel_size) * clip_limit), axis=-1, keepdims=True)
    tf_hist_clipped = tf.minimum(tf_hist, np.prod(kernel_size) * clip_limit) + tf_n_to_high / n_bins
    tf_cdf = tf.cumsum(tf_hist_clipped, axis=-1)
    tf_cdf_slice_size = tf.constant(np.concatenate((n_blocks_hist, [1])), dtype=tf.int32)
    tf_cdf_min = tf.slice(tf_cdf, tf.constant([0] * (dim + 1), dtype=tf.int32), tf_cdf_slice_size)
    tf_cdf_max = tf.slice(tf_cdf, tf.constant([0] * dim + [n_bins - 1], dtype=tf.int32), tf_cdf_slice_size)
    tf_cdf_norm = tf.where(tf.equal(tf_cdf_min, tf_cdf_max), tf.ones_like(tf_cdf_max), tf_cdf_max - tf_cdf_min)
    tf_mapping = (tf_cdf - tf_cdf_min) / tf_cdf_norm

    map_shape = np.concatenate((n_blocks_hist, [n_bins]))
    tf_map = tf.Variable(initial_value=tf.zeros(map_shape, dtype=tf.float32), dtype=tf.float32, trainable=False)
    tf_map.assign(tf_mapping)

    # Prepare initializer
    tf_x_block_init = tf.convert_to_tensor(np.zeros(shape_x_block, dtype=np.float32), dtype=tf.float32)

    # Set up slice of data and map
    tf_slice_begin = tf.Variable(initial_value=tf.zeros(shape=(dim,), dtype=tf.int32), dtype=tf.int32, trainable=False)
    tf_map_slice_begin = tf.concat([tf_slice_begin, [0]], 0)
    tf_map_slice_size = tf.constant(np.concatenate((n_blocks, [n_bins])), dtype=tf.int32)
    tf_map_slice = tf.slice(tf_map, tf_map_slice_begin, tf_map_slice_size)

    # Get bins
    if adaptive_hist_range:
        # Local bins
        tf_hist_norm_slice_shape = np.concatenate((n_blocks, [1] * dim))
        tf_x_hist_min_sub = tf.slice(tf_x_hist_min, tf_slice_begin, n_blocks)
        tf_x_hist_norm_sub = tf.slice(tf_x_hist_norm, tf_slice_begin, n_blocks)
        tf_x_block_scaled = (tf_x_block - tf.reshape(tf_x_hist_min_sub, tf_hist_norm_slice_shape)) \
                            / tf.reshape(tf_x_hist_norm_sub, tf_hist_norm_slice_shape)
        tf_bin = tf.histogram_fixed_width_bins(tf_x_block_scaled, [0., 1.], nbins=n_bins)
    else:
        # Global bins
        tf_bin = tf.Variable(initial_value=tf.cast(tf_x_block_init, tf.float32), dtype=tf.float32, trainable=False)
        bins = tf.histogram_fixed_width_bins(tf_x_block, [0., 1.], nbins=n_bins)
        tf_bin.assign(tf.cast(bins, dtype=tf.float32))

    # Apply map

    tf_mapped_sub = tf_batch_gather(tf_map_slice, tf_bin, dim)

    # Apply coefficients
    tf_coeff = tf.Variable(initial_value=tf.constant(1.0, dtype=tf.float32), dtype=tf.float32, trainable=False)
    tf_res_sub = tf.Variable(initial_value=tf_x_block_init, dtype=tf.float32, trainable=False)
    tf_res_sub.assign(tf_mapped_sub)
    tf_res_sub.assign(tf_coeff * tf_res_sub)

    # Update results
    tf_res = tf.Variable(initial_value=tf_x_block_init, dtype=tf.float32, trainable=False)
    tf_res.assign_add(tf_res_sub)

    # Rescaling
    tf_res.assign((tf_res - tf.reduce_min(tf_res)) / (tf.reduce_max(tf_res) - tf.reduce_min(tf_res)))

    # Reshape result
    new_shape = tuple((axis, axis + dim) for axis in range(dim))
    new_shape = tuple(j for i in new_shape for j in i)
    tf_res_transposed = tf.transpose(tf_res, new_shape)
    tf_res_reshaped = tf.reshape(tf_res_transposed, tuple(n_blocks[axis] * kernel_size[axis] for axis in range(dim)))

    # Recover original size
    tf_res_cropped = tf.slice(tf_res_reshaped, padding_x[:, 0], x.shape)

    # If not using GPU, set the device to CPU
    device = '/GPU:0' if use_gpu else '/CPU:0'

    with tf.device(device):
        map_init = tf.zeros(map_shape, dtype=tf.float32)
        print(f"map_init_shape{map_init.shape}")

        # Prepare the list of variables with their initial values
        variables_with_initial_values = [
            (tf_x_hist_padded, tf.convert_to_tensor(x_hist_padded, dtype=tf.float32)),
            (tf_map, map_init)
        ]
        print(f"tf_map_shape{tf_map.shape}")

        # If adaptive histogram range is used, add extra variables to the initialization list
        if adaptive_hist_range:
            x_hist_ex_init = tf.Variable(initial_value=n_blocks_hist, dtype=tf.float32)
            variables_with_initial_values.extend([
                (tf_x_hist_min, tf.convert_to_tensor(x_hist_ex_init, dtype=tf.float32)),
                (tf_x_hist_norm, tf.convert_to_tensor(x_hist_ex_init, dtype=tf.float32))
            ])

        # Initialize the variables
        for var, init_value in variables_with_initial_values:
            print(f"Variable shape: {var.shape}")
            print(f"Init value shape: {init_value.shape}")
            if var.shape != init_value.shape:
                init_value = tf.zeros(var.shape, dtype=tf.float32)  # We initialize to zeros using the shape of var.
            var.assign(init_value)

        # Normalize histogram data if needed
        if adaptive_hist_range:
            # You may need to ensure that `tf_get_hist_min` and `tf_get_hist_norm` are correctly defined functions
            tf_x_hist_min.assign(tf.reduce_min(tf_x_hist, axis=np.arange(-dim, 0)))
            tf_x_hist_max = tf.reduce_max(tf_x_hist, axis=np.arange(-dim, 0))
            tf_x_hist_norm.assign(tf.where(tf.equal(tf_x_hist_min, tf_x_hist_max),
                                       tf.ones_like(tf_x_hist_min),
                                       tf_x_hist_max - tf_x_hist_min))

        # Assign the mapping
        tf_map.assign(tf_mapping)

        # Get global hist bins if needed
        if not adaptive_hist_range:
            bins_result = tf.histogram_fixed_width_bins(tf_x_block, [0., 1.], nbins=n_bins)
            tf_bin.assign(tf.cast(bins_result, tf.float32))

        # Loop over maps
        inds = [list(i) for i in product([0, 1], repeat=dim)]
        for ind_map in inds:
            tf_mapped_sub = tf_batch_gather(tf_map_slice, tf_bin, dim)
            tf_res_sub.assign(tf_mapped_sub)
            # Calculate and apply coefficients
            for axis in range(dim):
                coeff = np.arange(kernel_size[axis], dtype=np.float32) / kernel_size[axis]
                if kernel_size[axis] % 2 == 0:
                    coeff = 0.5 / kernel_size[axis] + coeff
                if ind_map[axis] == 0:
                    coeff = 1. - coeff
                new_shape = [1] * (dim + axis) + [kernel_size[axis]] + [1] * (dim - 1 - axis)
                coeff = np.reshape(coeff, new_shape)
                tf_res_sub.assign(tf.multiply(tf_res_sub, coeff))
            # Update results
            tf_res.assign_add(tf_res_sub)

        # Rescaling
        tf_res.assign((tf_res - tf.reduce_min(tf_res)) / (tf.reduce_max(tf_res) - tf.reduce_min(tf_res)))

        # Get result
        result = tf_res_cropped.numpy()

    return result


def process_in_batches(input_file, batch_size):
    # Load the input 3D stack
    input_stack = tifffile.imread(input_file).astype(np.float32)

    # Determine the number of batches
    num_batches = int(np.ceil(input_stack.shape[0] / batch_size))

    # Create the directory to store intermediate results
    if not os.path.exists("clahe_batch"):
        os.makedirs("clahe_batch")

    # Iterate over each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, input_stack.shape[0])

        # Apply mclahe function
        batch_processed = mclahe(input_stack[start_idx:end_idx])

        # Save the processed batch
        tifffile.imwrite(f"clahe_batch/batch_{i + 1}.tif", batch_processed)

    # After processing all batches, concatenate them to form the final output
    batches = [tifffile.imread(f"clahe_batch/batch_{i + 1}.tif") for i in range(num_batches)]
    final_output = np.concatenate(batches, axis=0)

    # Save the final output
    tifffile.imwrite("final_output.tif", final_output)


# Example usage
input_file = 'C:\\Users\\aydar\\Desktop\\og_image_clipped.tif'
batch_size = 1648
process_in_batches(input_file, batch_size)

