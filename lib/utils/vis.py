from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (22, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
            [224, 224, 192],
        ]
    )

def vis_segments(labels, num_classes):
    colors = get_pascal_labels()
    # Draw the detections.
    height = labels.shape[0]
    width = labels.shape[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))

    img[yv, xv, :] = colors[labels]

    return img
