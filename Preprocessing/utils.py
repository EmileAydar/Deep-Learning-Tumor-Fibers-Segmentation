import torch

def pytorch_batch_gather(params, indices, axis):
    # Convert inputs to tensors and get shapes
    if not torch.is_tensor(params):
        params = torch.tensor(params)
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices)
    params_shape = params.shape
    indices_shape = indices.shape

    # Check dimensions
    if indices.dim() is None:
        raise ValueError("batch_gather does not allow indices with unknown shape.")

    # Cast params_shape to indices_dtype for computations
    casted_params_shape = params_shape.to(indices.dtype)

    # Compute flat indices
    batch_indices = indices
    accum_dim_value = torch.ones(()).to(indices.dtype)
    for dim in range(axis, 0, -1):
        dim_value = casted_params_shape[dim - 1]
        accum_dim_value *= casted_params_shape[dim]
        dim_indices = torch.arange(start=0, end=dim_value, step=1, dtype=indices.dtype)
        dim_indices *= accum_dim_value
        dim_shape = torch.stack([1] * (dim - 1) + [dim_value] + [1] * (indices.dim() - dim))
        batch_indices += dim_indices.reshape(dim_shape.tolist())

    # Flatten params and indices
    flat_inner_shape_indices = indices_shape[:(axis + 1)].numel()
    flat_indices = batch_indices.view([flat_inner_shape_indices] + list(indices_shape[(axis + 1):]))
    outer_shape = params_shape[(axis + 1):]
    flat_inner_shape_params = params_shape[:(axis + 1)].numel()
    flat_params = params.view([flat_inner_shape_params] + list(outer_shape))

    # Gather elements
    flat_result = flat_params[flat_indices.long()]
    result = flat_result.view(list(indices_shape) + list(outer_shape))

    # Check and set final shape
    final_shape = list(indices.shape[:axis]) + list(params.shape[:axis]) + list(indices.shape[axis:]) + list(params.shape[(axis + 1):])
    result = result.view(final_shape)

    return result


def pytorch_batch_histogram(values, value_range, axis, nbins=100, dtype=torch.int32, use_map=True):
    # Get shape
    values_shape = list(values.shape)
    batch_dim = values_shape[:axis]
    rest_dim = values_shape[axis:]
    num_batch = torch.prod(torch.tensor(batch_dim))

    if use_map:
        values_reshaped = values.reshape([num_batch.item()] + rest_dim)
        hist_list = [torch.histc(x, bins=nbins, min=value_range[0], max=value_range[1]) for x in values_reshaped]
        hist = torch.stack(hist_list).to(dtype)
    else:
        # Normalize
        values_float = values.float()
        value_range_float = value_range.float()

        # Clip values
        values_norm = (values_float - value_range_float[0]) / (value_range_float[1] - value_range_float[0])
        values_clip1 = torch.clamp(values_norm, min=0.5 / float(nbins))
        values_clip2 = torch.clamp(values_clip1, max=1.0 - 0.5 / float(nbins))

        # Shift values
        values_shift = values_clip2 + torch.arange(num_batch.item(), dtype=torch.float32).reshape(batch_dim + [1]*len(rest_dim))

        # Get histogram
        hist = torch.histc(values_shift, bins=num_batch.item()*nbins, min=0., max=float(num_batch)).to(dtype)

    return hist.reshape(batch_dim + [nbins])
