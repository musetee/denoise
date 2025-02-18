import numpy as np
def crop_numpy(array, target_shape, center_crop=True):
    """
    Crops a NumPy array to the desired shape.

    Parameters:
        array (np.ndarray): The input NumPy array (can be 2D, 3D, etc.).
        target_shape (tuple): The desired shape (must be smaller than the original shape).
        center_crop (bool): Whether to crop from the center (default: True). 
                            If False, crops from the top-left corner.

    Returns:
        np.ndarray: The cropped array.
    """
    original_shape = array.shape

    if any(t > o for t, o in zip(target_shape, original_shape)):
        raise ValueError(f"Target shape {target_shape} must be smaller than the original shape {original_shape}")

    if center_crop:
        # Compute starting and ending indices for center cropping
        start_idx = [(o - t) // 2 for o, t in zip(original_shape, target_shape)]
    else:
        # Crop from the top-left (starting indices are all 0)
        start_idx = [0] * len(original_shape)

    end_idx = [start + t for start, t in zip(start_idx, target_shape)]

    # Crop the array
    cropped_array = array[tuple(slice(start, end) for start, end in zip(start_idx, end_idx))]
    
    return cropped_array