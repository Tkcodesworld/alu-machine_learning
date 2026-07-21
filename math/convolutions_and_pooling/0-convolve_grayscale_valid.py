#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(kh):
        for j in range(kw):
            output += (
                images[:, i:i + output_h, j:j + output_w]
                * kernel[i, j]
            )

    return output
