# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os

# from lib.utils.colormap import colormap
import lib.utils.env as envu
# import lib.utils.keypoints as keypoint_utils
#
# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator
#
#
# _GRAY = (218, 227, 218)
# _GREEN = (18, 127, 15)
# _WHITE = (255, 255, 255)
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
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    # cmap = plt.get_cmap('rainbow')
    # colors = [cmap(i) for i in np.linspace(0, 1, num_classes-1)]
    # colors = np.array([[c[2] * 255, c[1] * 255, c[0] * 255] for c in colors])
    # colors = np.vstack(([[0, 0, 0]], colors))
    colors = get_pascal_labels()
    # Draw the detections.
    height = labels.shape[0]
    width = labels.shape[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))

    img[yv, xv, :] = colors[labels]

    return img
