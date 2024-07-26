import numpy as np


def inflate_2d(x, r):
    # grows a 2d array by repeating each element of an array after themselves
    x = np.repeat(x, r, axis=0)
    x = np.repeat(x, r, axis=1)
    return x


def postprocess_single_probability_map(p_map, config):
    # m: (840, 372, 281)
    binary_map = (p_map > config['threshold'])
    classes = []
    for bin_frame in binary_map:
        frame = inflate_2d(bin_frame, 2)
        classes.append(frame)
    classes = np.transpose(np.array(classes, 'float16'), axes=(0, 2, 1))
    return classes
