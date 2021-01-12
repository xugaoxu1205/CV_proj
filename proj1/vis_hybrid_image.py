import numpy as np
from scipy.misc import imresize


def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image[:]
    cur_image = hybrid_image[:]
    for i in range(1, scales):
        output = np.concatenate((output,
                                 np.ones((original_height, padding, num_colors))),
                                axis=1)
        cur_image = imresize(cur_image, scale_factor, 'bilinear').astype(np.float) / 255

        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0],
                                       cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output
