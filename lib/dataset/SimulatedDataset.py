import logging
import os
import json
import numpy as np
# from scipy.io import loadmat, savemat
import copy

from torch.utils.data import Dataset

from lib.utils.utils import get_source, get_mesh, evalpotential, generate_heatmaps

logger = logging.getLogger(__name__)

class SimulatedDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform, n_sources=30):
        self.cfg = cfg
        self.is_train = is_train

        self.root = root
        self.image_set = image_set

        self.is_train = is_train

        self.transform = transform

        self.n_sources = n_sources  # only used at test time

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        self.mesh = get_mesh(self.patch_width, self.patch_height)

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        the_db = copy.deepcopy(self.db[idx])

        image = evalpotential(self.mesh, the_db['sources'])

        image = image.reshape((self.patch_height, self.patch_width)) + \
            self.cfg.MODEL.VAR_NOISE*np.random.randn(self.patch_height, self.patch_width)

        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1).astype(np.float32)

        image = self.transform(image)

        unnormalized_sources = the_db['sources'].transpose()[:, 1:3].copy()
        unnormalized_sources[:, 0] *= self.cfg.MODEL.OUTPUT_SIZE[0]
        unnormalized_sources[:, 1] *= self.cfg.MODEL.OUTPUT_SIZE[1]

        heatmap_target = generate_heatmaps(
            unnormalized_sources,
            self.cfg.MODEL.OUTPUT_SIZE[0],
            self.cfg.MODEL.OUTPUT_SIZE[1]
        )

        valid_source_num = unnormalized_sources.shape[0]

        if valid_source_num < 64:
            unnormalized_sources = np.concatenate(
                (unnormalized_sources,
                -np.ones((64 - valid_source_num, 2))),
                axis=0
            )

        meta = {
            'sources': unnormalized_sources,
            'valid_source_num': valid_source_num
        }

        return image, heatmap_target, meta

    def _get_db(self):
        gt_db = []
        self.db_length = int(self.cfg.TRAIN.NUM_SAMPLES) if self.is_train else int(self.cfg.TEST.NUM_SAMPLES)
        for i in range(self.db_length):
            n_sources = np.random.randint(1, 65) if self.is_train else self.n_sources
            sources = get_source(self.cfg.MODEL.DEPTH, n_sources, self.cfg.MODEL.VAR_NOISE)

            gt_db.append({
                'sources': sources
            })

        return gt_db

    def __len__(self):
        return self.db_length
