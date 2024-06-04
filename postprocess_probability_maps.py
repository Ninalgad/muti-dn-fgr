from scipy.ndimage import binary_dilation


def mark_boundary_points(binary_mask):
    """
    Given a binary mask of an image, mark the boundary points with a value of 2.

    Parameters:
    binary_mask (numpy array): A binary mask of the image (2D array).

    Returns:
    modified_mask (numpy array): The binary mask with boundary points marked as 2.
    """
    # Ensure the input is a binary mask
    binary_mask = binary_mask.astype(bool)

    # Dilate the binary mask
    dilated_mask = binary_dilation(binary_mask)

    # The boundary is the difference between the dilated mask and the original mask
    boundary = dilated_mask & ~binary_mask

    # Create a copy of the original mask to modify
    modified_mask = binary_mask.astype(int)

    # Set the boundary points to 2
    modified_mask[boundary] = 2

    return modified_mask


def postprocess_single_probability_map(m, config):
    m = (m > config['threshold']).astype('uint8')
    return mark_boundary_points(m)
