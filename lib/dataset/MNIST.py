import logging
import os
import json
import glob

import numpy as np
from scipy.io import loadmat, savemat
import copy

from torch.utils.data import Dataset

# from lib.utils.utils import get_source, get_mesh, evalpotential, generate_heatmaps

logger = logging.getLogger(__name__)

class MNIST(Dataset):
    def __init__(self, cfg, root, image_set, is_train):
        self.cfg = cfg
        self.is_train = is_train

        self.root = root
        self.image_set = image_set

        self.is_subset = cfg.DATASET.SUBSET

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(mean, axis=0), axis=-1), axis=-1)
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(std, axis=0), axis=-1), axis=-1)

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        file_name = self.db[idx]

        data = loadmat(file_name)

        if self.is_subset:
            image = data['imgMat'].transpose([3, 2, 0, 1]).astype(np.float32)[:64]
            semantic_mask = data['semanticMaskMat'].transpose([2, 0, 1]).astype(np.int64)[:64]
            instance_mask = data['instanceMaskMat'].transpose([2, 0, 1]).astype(np.int64)[:64]
        else:
            image = data['imgMat'].transpose([3, 2, 0, 1]).astype(np.float32)
            semantic_mask = data['semanticMaskMat'].transpose([2, 0, 1]).astype(np.int64)
            instance_mask = data['instanceMaskMat'].transpose([2, 0, 1]).astype(np.int64)

        image = (image - self.mean) / self.std

        h_semantic_mask = semantic_mask[:, :, :-1] * 11 + semantic_mask[:, :, 1:]
        v_semantic_mask = semantic_mask[:, :-1, :] * 11 + semantic_mask[:, 1:, :]

        return image, semantic_mask, h_semantic_mask, v_semantic_mask, instance_mask

    def _get_db(self):
        gt_db = []

        prefix = 'batch' if self.is_train else 'testset'

        dataset_path = os.path.join('data/mnist', 'toydata_v3' if self.is_subset else 'toydata_v4', '{}*.mat'.format(prefix))

        gt_db = sorted(glob.glob(dataset_path))

        return gt_db

    def __len__(self):
        return self.db_length
