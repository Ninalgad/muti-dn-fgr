from scipy.ndimage import binary_dilation
import numpy as np
from scipy.ndimage import zoom


def get_boundary_points(binary_mask):
    """
    Given a binary mask of an image, mark the boundary points with a value of 2.

    Parameters:
    binary_mask (numpy array): A binary mask of the image (2D array).

    Returns:
    modified_mask (numpy array): The binary mask with boundary points marked as 2.
    """

    # Dilate the binary mask
    dilated_mask = binary_dilation(binary_mask)

    # The boundary is the difference between the dilated mask and the original mask
    boundary = dilated_mask & ~binary_mask

    # return boundary mask
    return boundary


def postprocess_single_probability_map(p_map, config):
    # m: (n, 372, 281)
    binary_map = (p_map > config['threshold'])
    classes = []
    for bin_frame in binary_map:
        annotated_frame = np.zeros_like(bin_frame, dtype="uint8")
        annotated_frame[bin_frame] = 2

        bound_mask = get_boundary_points(bin_frame)
        annotated_frame[bound_mask] = 1

        frame = zoom(annotated_frame, (2, 2))
        classes.append(frame)
    classes = np.transpose(np.array(classes, 'uint8'), axes=(0, 2, 1))
    return classes
