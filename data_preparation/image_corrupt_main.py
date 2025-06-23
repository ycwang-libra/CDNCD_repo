import numpy as np
from PIL import Image
from image_corrupt_toolbox import *

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur, 
                    motion_blur, zoom_blur, brightness, contrast, elastic_transform, 
                    jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}

def corrupt(x, severity=1, corruption_name=None):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur', 'zoom_blur',
                    'brightness', 'contrast', 'elastic_transform', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """
    x_corrupted = corruption_dict[corruption_name](Image.fromarray(x), severity)
    return np.uint8(x_corrupted)