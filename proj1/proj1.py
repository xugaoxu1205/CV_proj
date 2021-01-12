import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image
from gauss2D import gauss2D


def main():
    """ function to create hybrid images """
    # read images and convert to floating point format
    image1 = mpimg.imread(
        './data/dog.bmp')
    image2 = mpimg.imread(
        './data/cat.bmp')
    image1 = image1.astype(np.float32) / 255
    image2 = image2.astype(np.float32) / 255

    cutoff_frequency_1 = 7
    gaussian_filter_1 = gauss2D(shape=(cutoff_frequency_1 * 4 + 1, cutoff_frequency_1 * 4 + 1),
                                sigma=cutoff_frequency_1)

    low_frequencies = my_imfilter(image1, gaussian_filter_1)

    cutoff_frequency_2 = 7
    gaussian_filter_2 = gauss2D(shape=(cutoff_frequency_2 * 4 + 1, cutoff_frequency_2 * 4 + 1),
                                sigma=cutoff_frequency_2)
    low_frequencies_2 = my_imfilter(image2, gaussian_filter_2)
    high_frequencies = image2 - low_frequencies_2

    hybrid_image = low_frequencies + high_frequencies

    plt.figure(1)
    plt.imshow(low_frequencies)
    plt.figure(2)
    plt.imshow(high_frequencies + 0.5)
    plt.figure(3)
    plt.imshow(hybrid_image)
    vis = vis_hybrid_image(hybrid_image)
    plt.figure(4)
    plt.imshow(vis)

    plt.show()


if __name__ == '__main__':
    main()
