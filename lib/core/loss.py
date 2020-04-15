from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch.nn as nn

import pdb

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class CrossEntropy2D(nn.Module):
    def __init__(self, ignore_index, reduction='mean', weight=None):
        super(CrossEntropy2D, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target, resize_scores=True):
        _assert_no_grad(target)

        b, c, h, w = output.size()
        tb, th, tw = target.size()

        assert(b == tb)

        # Handle inconsistent size between input and target

        if resize_scores:
# <<<<<<< HEAD
#             if h != th or w != tw:  # downsample labels
#                 output = nn.functional.upsample(output, size=(th, tw), mode="bilinear", align_corners=False)
#                 #output = nn.functional.interpolate(output, size=(th, tw), mode="bilinear", align_corners=False)
#         elif config.MODEL.USE_DUPSAMPLING:
#             pad_size = (h - th)/2.
#             bottom, right =  math.floor(pad_size), math.ceil(pad_size)
#             m = nn.ZeroPad2d((bottom, right, bottom, right))
#             target = m(target)
#         else:
#             assert (h == th and w == tw)
#         pdb.set_trace()
#         loss = nn.functional.cross_entropy(output, target, weight=weight, ignore_index=self.ignore_index, reduce=True if self.reduction == 'mean' else False)
# =======
            if h != th or w != tw:  # upsample logits
                output = nn.functional.interpolate(output, size=(th, tw), mode="bilinear", align_corners=False)
        elif config.MODEL.USE_DUPSAMPLING:
            pad_size = (h - th)/2.
            bottom, right =  math.floor(pad_size), math.ceil(pad_size)
            m = nn.ZeroPad2d((bottom, right, bottom, right))
            target = m(target)
        else:
            if h != th or w != tw:  # downsample labels
                target = nn.functional.interpolate(target.view(b, 1, th, tw).float(), size=(h, w), mode="nearest").view(b, h, w).long()


        loss = nn.functional.cross_entropy(
            output, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )
# >>>>>>> Add dataloader for PASCAL-CONTEXT, CityScapes.

        return loss
