from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from lib.core.config import get_model_name
from lib.models.sync_bn.inplace_abn.bn import InPlaceABNSync


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    if cfg.DATASET.SUBSET:
        dataset += '_subset'

    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_1x_params(model, key):
    # For backbone and heads
    if key == "conv_body":
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                for p in m.parameters():
                    yield p
    # For temperature parameters
    elif key == "temperature":
        for name, m in model.named_modules():
            if "fixed_point_iteration" in name and m.gamma is not None:
                yield m.gamma
        if isinstance(model.gamma, nn.Parameter):
            yield model.gamma
    else:
        raise ValueError('Parameter key {} not supported'.format(key))

def get_10x_params(model, key):
    # For backbone
    if key == "base":
        for name, m in model.named_modules():
            if "unary" not in name and "pairwise" not in name and "aspp" not in name and "conv_bottleneck" not in name and "feature_layer" not in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                    for p in m.parameters():
                        yield p
    # For classifiers weights after backbone
    elif key == "head_weight":
        for name, m in model.named_modules():
            if "unary" in name or "pairwise" in name or "aspp" in name or "conv_bottleneck" in name or "feature_layer" in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                    yield m.weight
    # For classifiers biases after backbone
    elif key == "head_bias":
        for name, m in model.named_modules():
            if "unary" in name or "pairwise" in name or "aspp" in name or "conv_bottleneck" in name or "feature_layer" in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                    if m.bias is not None:
                        yield m.bias
    elif key == "temperature":
        for name, m in model.named_modules():
            if "fixed_point_iteration" in name and m.gamma is not None:
                yield m.gamma
        if isinstance(model.gamma, nn.Parameter):
            yield model.gamma
        # for alpha in model.alphas:
        #     yield alpha
    else:
        raise ValueError('Parameter key {} not supported'.format(key))

# def get_01x_params(model, key):
#     # For backbone and unary weights
#     if key == "unary":
#         for m in model.named_modules():
#             if "pairwise" not in m[0]:
#                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
#                     for p in m[1].parameters():
#                         yield p
#     # For pairwise weights
#     elif key == "pairwise":
#         for m in model.named_modules():
#             if "pairwise" in m[0]:
#                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
#                     for p in m[1].parameters():
#                         yield p
#             elif "fixed_point_iteration" in m[0]:
#                 if m[1].gamma is not None:
#                     yield m[1].gamma
#     else:
#         raise ValueError('Parameter key {} not supported'.format(key))
def get_01x_params(model, key):
    # For backbone and unary head
    if key == "unary":
        for name, m in model.named_modules():
            if "pairwise" not in name and "fixed_point_iteration" not in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                    for p in m.parameters():
                        yield p
    # For pairwise head
    elif key == "pairwise":
        for name, m in model.named_modules():
            if "pairwise" in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                    for p in m.parameters():
                        yield p
    # For temperature head
    elif key == "temperature":
        for name, m in model.named_modules():
            if "fixed_point_iteration" in name and m.gamma is not None:
                yield m.gamma
        if isinstance(model.gamma, nn.Parameter):
            yield m
    else:
        raise ValueError('Parameter key {} not supported'.format(key))


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        if cfg.TRAIN.HEAD_LR == '10x':
            optimizer = optim.SGD(
                params = [
                    {
                        "params": get_10x_params(model.module, key="base"),
                        "lr": cfg.TRAIN.LR,
                        "weight_decay": cfg.TRAIN.WD,
                    },
                    {
                        "params": get_10x_params(model.module, key="head_weight"),
                        "lr": 10 * cfg.TRAIN.LR,
                        "weight_decay": cfg.TRAIN.WD,
                    },
                    {
                        "params": get_10x_params(model.module, key="temperature"),
                        "lr": 10 * cfg.TRAIN.LR,
                        "weight_decay": 0.0,
                    },
                    {
                        "params": get_10x_params(model.module, key="head_bias"),
                        "lr": 10 * cfg.TRAIN.LR,
                        "weight_decay": 0.0,
                    }
                ],
                momentum=cfg.TRAIN.MOMENTUM
            )
        elif cfg.TRAIN.HEAD_LR == '0.1x':
            optimizer = optim.SGD(
                params = [
                    {
                        "params": get_01x_params(model.module, key="unary"),
                        "lr": cfg.TRAIN.LR,
                        "weight_decay": cfg.TRAIN.WD,
                    },
                    {
                        "params": get_01x_params(model.module, key="pairwise"),
                        "lr": 0.1 * cfg.TRAIN.LR,
                        "weight_decay": cfg.TRAIN.WD,
                    },
                    {
                        "params": get_01x_params(model.module, key="temperature"),
                        "lr": 0.1 * cfg.TRAIN.LR,
                        "weight_decay": 0.0,
                    },
                ],
                momentum=cfg.TRAIN.MOMENTUM
            )
        elif cfg.TRAIN.HEAD_LR == '1x':
            # optimizer = optim.SGD(
            #     model.parameters(),
            #     lr=cfg.TRAIN.LR,
            #     weight_decay=cfg.TRAIN.WD,
            #     momentum=cfg.TRAIN.MOMENTUM
            # )
            optimizer = optim.SGD(
                params = [
                    {
                        "params": get_1x_params(model.module, key="conv_body"),
                        "lr": cfg.TRAIN.LR,
                        "weight_decay": cfg.TRAIN.WD,
                    },
                    {
                        "params": get_1x_params(model.module, key="temperature"),
                        "lr": cfg.TRAIN.LR,
                        "weight_decay": 0.0,
                    }
                ],
                momentum=cfg.TRAIN.MOMENTUM
            )
        else:
            raise NotImplementedError("Head learning mode {} not implemented".format(cfg.TRAIN.HEAD_LR))
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            betas=(cfg.TRAIN.BETA1, cfg.TRAIN.BETA2),
            weight_decay=cfg.TRAIN.WD
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamax':
        optimizer = optim.Adamax(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            betas=(cfg.TRAIN.BETA1, cfg.TRAIN.BETA2),
            weight_decay=cfg.TRAIN.WD
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

def evalpotential(sites_locations, sources):
    """
    Parameters:
    -----------
    sources: numpy array
    4*N matrix such that first row is the charge of sources and 2, 3 and 4 raws are x, y, z of the sources.
    (N is the number of sources)

    sites_locations: numpy array
    3*M matrix such that first, second and third raws are x, y, z of the sites_locations.
    (M is the number of sites)

    Return:
    -------
    recordings: numpy array
    an array with the values of the recordings for the sites.
    """
    k = sites_locations.shape[1]
    m = sources.shape[1]
    recordings = np.zeros(k);
    diff_x = sources[1, :].reshape((m, 1)) - sites_locations[0, :].reshape((1, k))
    diff_y = sources[2, :].reshape((m, 1)) - sites_locations[1, :].reshape((1, k))
    diff_z = sources[3, :].reshape((m, 1)) - sites_locations[2, :].reshape((1, k))

    dist_sources = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
    recordings = sources[0, :].reshape((m, 1)) / dist_sources
    recordings = recordings.sum(axis=0).reshape(-1)
    # for i in range(k):
    #     dis_sources = deepcopy(sources[1:,:])
    #     dis_sources[0, :] -= sites_locations[0,i]
    #     dis_sources[1, :] -= sites_locations[1,i]
    #     dis_sources[2, :] -= sites_locations[2,i]
    #     dis_sources = np.sqrt(sum(dis_sources**2,0))
    #     recordings[i] = sum(sources[0,:]/dis_sources);
    return recordings

def get_mesh(x_mesh, y_mesh):

    d = x_mesh*y_mesh
    x = np.linspace(0, 1, x_mesh)
    y = np.linspace(0, 1, y_mesh)
    Xsim,Ysim = np.meshgrid(x,y)
    Xsim = np.reshape(Xsim, [d])
    Ysim = np.reshape(Ysim, [d])
    mesh = np.array([Xsim,Ysim,np.zeros(d)]);
    return mesh

# def get_source(n_sources=[40, 60],
#                p_n_sources=[.5, .5],
#                depth=[.1],
#                p_depth=[1],
#                amplitude=[-1,1],
#                p_amplitude=[.5,.5]):
#     """
#     Parameters:
#     -----------
#     n_sources: numpy array (int)
#     the number of sources in the plane
#
#     p_n_sources: numpy array
#     The probabily distribution over the number of sources.
#
#     depth: numpy array
#     the depth of sources
#
#     p_depth: numpy array
#     The probabily distribution over the depth of sources.
#
#     amplitude: numpy array
#     amplitude of sources.
#
#     p_amplitude: numpy array
#     The probabily distribution over the amplitude of sources.
#
#     Returns:
#     -------
#     sources: numpy array of shape (4, n)
#     the bio information of sources.
#     """
#
#     number_of_sources = np.random.choice(a=n_sources, size=1, p=p_n_sources)[0]
#     sources = np.zeros(4, number_of_sources);
#     d = np.expand_dims(np.random.choice(a=depth, size=number_of_sources, p=p_depth),0)
#     loc = np.random.rand(2, number_of_sources)
#     amp = np.expand_dims(np.random.choice(a=amplitude, size=number_of_sources, p=p_amplitude),0)
#     sources = np.append(np.append(amp, loc, 0), d,0)
#     return sources

def get_source(depth, n_sources, var_noise):

    sources = np.random.rand(4, n_sources);
    sources[3, :] = depth
    sources[0, :] = 2*np.floor(2*sources[0, :])-1;
    # image = evalpotential(mesh, sources);
    # image = image.reshape((y_mesh, x_mesh)) + var_noise*np.random.randn(y_mesh, x_mesh)
    return sources

def generate_heatmaps(keypoints, im_height, im_width):
    heatmaps = np.zeros((1, int(im_height), int(im_width)), dtype=np.float32)
    sigma = max(im_height, im_width) / 64.0
    size = 6*sigma + 3
    x = np.arange(0, size, 1, dtype=np.float32)
    y = x[:, np.newaxis]
    x0, y0 = 3*sigma + 1, 3*sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    for idx in range(keypoints.shape[0]):
        pt = keypoints[idx, :]

        x, y = int(pt[0]), int(pt[1])
        if x < 0 or y < 0 or x >= im_width or y >= im_height:
            #print('not in', x, y)
            continue
        # print (x, y, im_width, im_height)
        # assert (x >= 0 and y >= 0 and x < im_width and y < im_height)
        ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
        br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

        c,d = max(0, -ul[0]), min(br[0], im_width) - ul[0]
        a,b = max(0, -ul[1]), min(br[1], im_height) - ul[1]

        cc,dd = max(0, ul[0]), min(br[0], im_width)
        aa,bb = max(0, ul[1]), min(br[1], im_height)
        heatmaps[0, aa:bb,cc:dd] = np.maximum(heatmaps[0, aa:bb,cc:dd], g[a:b,c:d])


    # assert(len(fg_keypoint_locs) > 0)
    # assert(len(instance_inds) > 0)
    # Make sure fg_keypoint_locs and instance_inds are 2D-arrays
    return heatmaps
