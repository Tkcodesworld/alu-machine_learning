#!/usr/bin/env python3
"""Performs a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = (kh - 1) // 2
    pw = (kw - 1) // 2

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    output = np.zeros((m, h, w))

    for i in range(kh):
        for j in range(kw):
            output += (kernel[i, j] *
                       padded[:, i:i + h, j:j + w])

    return output
