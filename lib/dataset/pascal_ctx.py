import os
import math
import torch

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms


class PASCALContext(data.Dataset):
    def __init__(self,
                 cfg,
                 root,
                 image_set,
                 num_classes=59,
                 transform=None,
                 augmentations=None):

        self.cfg = cfg
        self.root = os.path.join(root, 'VOCdevkit/VOC2010')
        # self.root = root
        self.image_set = image_set

        if 'xception' in cfg.MODEL.NAME:
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(mean, axis=0), axis=-1), axis=-1)
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(std, axis=0), axis=-1), axis=-1)

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        self.n_classes = num_classes

        self.output_stride = cfg.MODEL.OUTPUT_STRIDE

        self.tf = transform
        self.augmentations = augmentations

        self._setup_db()
        self.db_length = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        im_name = item['file_name']
        im_path = os.path.join(self.root, "JPEGImages", im_name)
        img_id = item['image_id']

        im = Image.open(im_path)
        lbl = np.asarray(self.masks[img_id], dtype=np.int)
        lbl = self.label_transform(lbl)
        # Map all ignored pixels to class index 255
        lbl[np.logical_or(lbl >= self.n_classes, lbl < 0)] = 255
        lbl = Image.fromarray(lbl)

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)

        im, lbl, lbl_os = self.transform(im, lbl)

        if self.cfg.MODEL.LEARN_PAIRWISE_TERMS:
            # if self.cfg.MODEL.NUM_PAIRWISE_TERMS > 1:
            lbl_hs = []
            lbl_vs = []

            stride = 1
            for _ in range(self.cfg.MODEL.NUM_PAIRWISE_TERMS):
                for i in range(stride):
                    for j in range(stride):
                        lbl_os_ = lbl_os[i::stride, j::stride]

                        lbl_h = lbl_os_[:, :-1] * self.n_classes + lbl_os_[:, 1:]
                        lbl_v = lbl_os_[:-1, :] * self.n_classes + lbl_os_[1:, :]
                        lbl_h[(lbl_os_[:, :-1] >= self.n_classes) | (lbl_os_[:, 1:] >= self.n_classes)] = 255
                        lbl_v[(lbl_os_[:-1, :] >= self.n_classes) | (lbl_os_[1:, :] >= self.n_classes)] = 255

                        lbl_hs.append(lbl_h)
                        lbl_vs.append(lbl_v)

                stride += self.cfg.MODEL.PAIRWISE_STEP_SIZE
            # else:
            #     lbl_h = lbl_os[:, :-1] * self.n_classes + lbl_os[:, 1:]
            #     lbl_v = lbl_os[:-1, :] * self.n_classes + lbl_os[1:, :]
            #     lbl_h[(lbl_os[:, :-1] >= self.n_classes) | (lbl_os[:, 1:] >= self.n_classes)] = 255
            #     lbl_v[(lbl_os[:-1, :] >= self.n_classes) | (lbl_os[1:, :] >= self.n_classes)] = 255

            return im, lbl, lbl_hs, lbl_vs, dict()
        else:
            return im, lbl, dict(), dict(), dict()

    def transform(self, img, lbl):
        if self.tf is not None:
            img = self.tf(img)

        # if self.is_train:
        w, h = lbl.size
        lbl_os = lbl.resize((math.ceil(w / self.output_stride), math.ceil(h / self.output_stride)), Image.NEAREST)

        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl >= self.n_classes] = 255  # ignore pixels

        lbl_os = torch.from_numpy(np.array(lbl_os)).long()
        lbl_os[lbl_os >= self.n_classes] = 255  # ignore pixels

        return img, lbl, lbl_os

    def _setup_db(self):
        # prepare data
        annots = os.path.join(self.root, 'trainval_merged.json')
        img_path = os.path.join(self.root, 'JPEGImages')
        from detail import Detail
        if 'val' in self.image_set:
            self.detail = Detail(annots, img_path, 'val')
            mask_file = os.path.join(self.root, 'val.pth')
        elif 'train' in self.image_set:
            self.mode = 'train'
            self.detail = Detail(annots, img_path, 'train')
            mask_file = os.path.join(self.root, 'train.pth')
        else:
            raise NotImplementedError('only supporting train and val set.')
        self.files = self.detail.getImgs()

        # generate masks
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))

        self._key = np.array(range(len(self._mapping))).astype('uint8')

        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        for i in range(len(self.files)):
            img_id = self.files[i]
            mask = Image.fromarray(self._class_to_index(
                self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
        torch.save(masks, mask_file)
        return masks

    def label_transform(self, label):
        if self.n_classes == 59:
            # background is ignored
            label = np.array(label).astype('int32') - 1
            label[label==-2] = -1
        else:
            label = np.array(label).astype('int32')
        return label

    def get_pascal_labels(self):
        """Load the mapping that associates pascal-context classes with label colors

        Returns:
            np.ndarray with dimensions (60, 3)
        """

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, self.n_classes - 1)]
        colors = np.array([[c[2] * 255, c[1] * 255, c[0] * 255] for c in colors])
        colors = np.vstack(([[0, 0, 0]], colors))

        return colors

# Leave code for debugging purposes
import lib.utils.augmentations as aug
from lib.core.config import config
if __name__ == '__main__':
    bs = 1

    config.MODEL.IMAGE_SIZE = (513, 513)
    config.MODEL.OUTPUT_STRIDE = 16
    config.MODEL.LEARN_PAIRWISE_TERMS = False

    # augs = aug.Compose([aug.AdjustBrightness(0.1),
    #                     aug.AdjustContrast(0.1),
    #                     aug.AdjustSaturation(0.1),
    #                     aug.AdjustHue(0.1),
    #                     aug.RandomScale(0.5, 1.5),
    #                     aug.RandomRotate(10),
    #                     aug.RandomHorizontallyFlip(0.5),
    #                     aug.RandomSizedCrop(config.MODEL.IMAGE_SIZE)])
    augs = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dst = PASCALContext(config,
                        'data/pascal_ctx',
                        'val',
                        59,
                        transforms.Compose([
                            transforms.ToTensor(),
                            # normalize,
                        ]),
                        augmentations=augs)

    trainloader = data.DataLoader(dst,
        batch_size=bs,
        shuffle=False,
        pin_memory=True)

    for i, (imgs, labels, _, _, _) in enumerate(trainloader):
        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[0].imshow(imgs[j])
            axarr[1].imshow(dst.decode_segmap(labels.numpy()[j]))
            plt.show()
            a = input()
            if a == 'ex':
                break
            else:
                plt.close()
