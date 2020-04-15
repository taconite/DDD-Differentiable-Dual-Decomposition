import logging
import os
import json
import numpy as np
# from scipy.io import loadmat, savemat
import copy

import h5py

from torch.utils.data import Dataset

# from lib.utils.utils import get_source, get_mesh, evalpotential, generate_heatmaps

logger = logging.getLogger(__name__)

class RealDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform):
        self.cfg = cfg
        self.is_train = is_train

        self.root = root
        self.image_set = image_set

        if is_train:
            raise ValueError('Real dataset has no labels and thus cannot be trained on')
        # self.is_train = is_train

        self.transform = transform

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        self.x_min = cfg.DATASET.X_MIN
        self.y_min = cfg.DATASET.Y_MIN
        self.x_max = cfg.DATASET.X_MAX
        self.y_max = cfg.DATASET.Y_MAX

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        the_db = copy.deepcopy(self.db[idx])

        image = self.transform(the_db['image'])

        return image

    def _get_db(self):
        gt_db = []

        all_frames = []
        for i, file_name in enumerate(self.image_set):
            dataset = {}
            file_path = os.path.join('data', self.cfg.DATASET.DATASET, file_name)
            with h5py.File(file_path, 'r') as f:
                for k, v in f.items():
                    dataset[k] = np.array(v)

            # assert ('D1' in dataset.keys())
            assert ('D2' in dataset.keys())

            all_frames = np.expand_dims(dataset['D2'].copy().astype(np.float32), axis=-1).repeat(3, axis=-1)
            # all_frames_vis = np.transpose(dataset['D1'], (0, 2, 3, 1)).astype(np.uint8)

            all_frames = all_frames[:, self.y_min:self.y_max, self.x_min:self.x_max, :]
            # all_frames_vis = all_frames_vis[269:270, self.y_min:self.y_max, self.x_min:self.x_max, :]

            for frame in all_frames:
                gt_db.append({
                    'image': frame
                })

        return gt_db

    def __len__(self):
        return self.db_length
