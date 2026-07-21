#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    output = np.zeros((m, oh, ow))

    for i in range(kh):
        for j in range(kw):
            output += (kernel[i, j] *
                       padded[:, i:i + oh, j:j + ow])

    return output
