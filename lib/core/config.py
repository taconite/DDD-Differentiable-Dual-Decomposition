from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# seg_resnet related params
SEG_RESNET = edict()
SEG_RESNET.NUM_LAYERS = 50
SEG_RESNET.DECONV_WITH_BIAS = False
SEG_RESNET.NUM_DECONV_LAYERS = 3
SEG_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
SEG_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
SEG_RESNET.FINAL_CONV_KERNEL = 1
SEG_RESNET.TARGET_TYPE = 'gaussian'
SEG_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
SEG_RESNET.NUM_CLASSES = 21

SEG_FCN = edict()
SEG_FCN.NUM_CONV_LAYERS = 3
SEG_FCN.NUM_CONV_FILTERS = [64, 128, 256]
SEG_FCN.NUM_CONV_KERNELS = [3, 3, 3]
SEG_FCN.CONV_STRIDES = [2, 1, 1]
SEG_FCN.CONV_PADDINGS = [1, 2, 2]
SEG_FCN.CONV_DILATIONS = [1, 3, 3]
SEG_FCN.NUM_DECONV_LAYERS = 2
SEG_FCN.NUM_DECONV_FILTERS = [256, 256]
SEG_FCN.NUM_DECONV_KERNELS = [4, 4]
SEG_FCN.NUM_CLASSES = 11
SEG_FCN.FINAL_CONV_KERNEL = 1
SEG_FCN.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32

SEG_FPN = edict()
SEG_FPN.NUM_LAYERS = 50
SEG_FPN.DECONV_WITH_BIAS = False
SEG_FPN.NUM_DECONV_LAYERS = 2
SEG_FPN.NUM_DECONV_FILTERS = [256, 256]
SEG_FPN.NUM_DECONV_KERNELS = [4, 4]
SEG_FPN.FINAL_CONV_KERNEL = 1
SEG_FPN.TARGET_TYPE = 'gaussian'
SEG_FPN.NUM_CLASSES = 21

MODEL_EXTRAS = {
    'seg_fcn': SEG_FCN,
    'seg_resnet': SEG_RESNET,
    'seg_fpn': SEG_FPN,
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'seg_fcn'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.INIT_DECONVS = False
config.MODEL.PRETRAINED = ''
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.LEARN_PAIRWISE_TERMS = False
config.MODEL.FPI_ITERS = 5
config.MODEL.PAIRWISE_STEP_SIZE = 1
config.MODEL.NUM_PAIRWISE_TERMS = 2

config.MODEL.NE_GAMMA = 1.0
config.MODEL.PAIRWISE_BN = False
config.MODEL.UNARY_BN = False
config.MODEL.ADAPTIVE_FPI_STEP_SIZE = False
config.MODEL.LEARN_GAMMA = False
config.MODEL.LEARN_OUTPUT_TEMPERATURE = False
config.MODEL.INIT_OUTPUT_TEMPERATURE = 1.0
config.MODEL.OUTPUT_STRIDE = 16
config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]
config.MODEL.USE_DUPSAMPLING = False

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_PAIRWISE_LOSS = True
config.LOSS.USE_UNARY_LOSS = False
config.LOSS.USE_ALL_FPI_LOSS = False
config.LOSS.PAIRWISE_LOSS_WEIGHT = 0.01

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'voc'
config.DATASET.SUBSET = True
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.SBD_PATH = '/home/sfwang/Datasets/benchmark_RELEASE'

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30
config.DATASET.PAD_BORDER = True

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0
config.TRAIN.BETA1 = 0.9
config.TRAIN.BETA2 = 0.999

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

config.TRAIN.LR_SCHEDULER = 'multistep'
config.TRAIN.TRAIN_BN = False
config.TRAIN.HEAD_LR = '1x'

config.TRAIN.NE_GAMMA_U = 1.0
config.TRAIN.NE_GAMMA_L = 0.1
config.TRAIN.NE_GAMMA_EXP = 1.0
config.TRAIN.NE_ITER_RATIO = 1.0

config.TRAIN.UNARY_GAMMA_U = 1.0
config.TRAIN.UNARY_GAMMA_L = 1.0
config.TRAIN.EVAL_INTERVAL = 1

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True

config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0

config.TEST.NUM_SAMPLES = 5e3

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if name in ['seg_block4_resnet', 'seg_aspp_resnet', 'seg_aspp_xception']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        full_name = '{height}x{width}_{name}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
