from scipy.ndimage import binary_dilation
import numpy as np
from scipy.ndimage import zoom


def postprocess_single_probability_map(p_map, config):
    # m: (840, 372, 281)
    binary_map = (p_map > config['threshold'])
    classes = []
    for bin_frame in binary_map:
        frame = zoom(bin_frame, (2, 2))
        classes.append(frame)
    classes = np.transpose(np.array(classes, 'float16'), axes=(0, 2, 1))
    return classes
