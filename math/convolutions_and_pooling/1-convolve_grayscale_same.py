#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    ph = kh // 2 if kh % 2 == 1 else kh // 2 - 1
    pw = kw // 2 if kw % 2 == 1 else kw // 2 - 1

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(kh):
        for j in range(kw):
            output += (
                padded[:, i:i + h, j:j + w]
                * kernel[i, j]
            )

    return output
