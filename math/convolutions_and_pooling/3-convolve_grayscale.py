#!/usr/bin/env python3
"""Performs a convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w)
        kernel: numpy.ndarray of shape (kh, kw)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2)
        pw = (((w - 1) * sw + kw - w) // 2)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(kh):
        for j in range(kw):
            output += (
                kernel[i, j] *
                padded[:, i:i + sh * out_h:sh,
                          j:j + sw * out_w:sw]
            )

    return output
