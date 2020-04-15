from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def calc_IoU(label_preds, label_trues, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    # return {
    #     "Pixel Accuracy": acc,
    #     "Mean Accuracy": acc_cls,
    #     "Frequency Weighted IoU": fwavacc,
    #     "Mean IoU": mean_iu,
    #     "Class IoU": cls_iu,
    # }
    return mean_iu

# def calc_IoU(preds, targets, num_classes):
#
#     if isinstance(preds, list):
#         assert isinstance(targets, list)
#
#         inters = np.zeros(20, dtype=np.float32)
#         unions = np.zeros(20, dtype=np.float32)
#
#         for pred, target in zip(preds, targets):
#             for cls in range(1, num_classes):
#                 care = (target != -100)
#                 cls_pred = np.logical_and((pred == cls), care)
#                 cls_target = np.logical_and((target == cls), care)
#                 # if np.all(cls_target == 0):
#                 #     continue
#
#                 intersection = np.logical_and(cls_target, cls_pred)
#                 union = np.logical_or(cls_target, cls_pred)
#                 # iou = np.sum(intersection) / np.sum(union)
#                 # iou_scores[cls-1].append(iou)
#                 inters[cls-1] += intersection.sum()
#                 unions[cls-1] += union.sum()
#
#         iou_scores = [i / u for i, u in zip(inters, unions)]
#         # for cls in range(len(iou_scores)):
#         #     iou_scores[cls] = np.mean(iou_scores[cls]) if len(iou_scores[cls]) > 0 else 0
#
#         avg_iou_score = np.mean(iou_scores)
#
#     elif isinstance(preds, np.ndarray):
#         assert isinstance(targets, np.ndarray)
#
#         iou_scores = []
#
#         for cls in range(1, num_classes):   # class 0 is always background
#             cls_pred = (preds == cls)
#             cls_target = (targets == cls)
#             intersection = np.logical_and(cls_target, cls_pred)
#             union = np.logical_or(cls_target, cls_pred)
#             iou = np.sum(intersection) / np.sum(union)
#             iou_scores.append(iou)
#
#         avg_iou_score = np.mean(iou_scores)
#     else:
#         raise ValueError('Input is not of correct type')
#
#     return avg_iou_score, iou_scores
