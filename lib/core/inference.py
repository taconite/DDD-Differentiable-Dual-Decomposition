from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import cv2

import numpy as np

def _softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_final_preds(batch_heatmaps, batch_heatmaps_=None):
    '''
    get predictions from heatmaps (logits)
    batch_heatmaps: numpy.ndarray([batch_size, num_classes, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_probs = _softmax(batch_heatmaps, axis=1)

    if batch_heatmaps_ is not None:
        assert isinstance(batch_heatmaps_, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps_.ndim == 4, 'batch_images should be 4-ndim'

        batch_probs_ = _softmax(batch_heatmaps_, axis=1)
        batch_preds = np.argmax((batch_probs + batch_probs_) / 2.0, axis=1)
    else:
        batch_preds = np.argmax(batch_probs, axis=1)

    return batch_preds
