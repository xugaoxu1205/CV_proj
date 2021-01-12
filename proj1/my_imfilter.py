import numpy as np


def my_imfilter(image, imfilter):

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    H_img = R.shape[0]
    W_img = R.shape[1]
    H_filter = imfilter.shape[0]
    W_filter = imfilter.shape[1]
    H_pad = int((H_filter - 1) / 2)
    W_pad = int((W_filter - 1) / 2)

    npad = ((H_pad, H_pad), (W_pad, W_pad))
    RGB_pad = []
    RGB_pad.append(np.pad(R, pad_width=npad, mode='reflect'))
    RGB_pad.append(np.pad(G, pad_width=npad, mode='reflect'))
    RGB_pad.append(np.pad(B, pad_width=npad, mode='reflect'))

    output = np.zeros_like(R)

    for each in RGB_pad:
        RGB_new = []
        # convolution of whole matrix
        for m in range(H_img):
            for n in range(W_img):
                # convolution in each small matrix
                total = 0
                total = np.sum(np.multiply(each[m:m + H_filter, n:n + W_filter], imfilter))
                RGB_new.append(total)

        RGB_new = np.asarray(RGB_new)
        RGB_new = RGB_new.reshape(H_img, W_img)

        # combine RGB channel into 3D array
        output = np.dstack((output, RGB_new))

    # remove the zeros array
    output = output[:, :, 1:]

    return output
