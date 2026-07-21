#!/usr/bin/env python3
"""Performs a convolution on images with channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels."""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    for i in range(kh):
        for j in range(kw):
            output += np.sum(
                padded[:, i:i + sh * oh:sh,
                          j:j + sw * ow:sw, :] *
                kernel[i, j, :],
                axis=3
            )

    return output
