import os
import collections
import json
import math
import torch
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import imageio

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


class pascalVOC(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(self, cfg, root,
        image_set,
        transform,
        augmentations=None
    ):
        self.cfg = cfg

        self.root = root
        self.image_set = image_set

        self.sbd_path = cfg.DATASET.SBD_PATH
        # self.sbd_path = '/home/sfwang/Datasets/benchmark_RELEASE'

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]
        # self.patch_width = 512
        # self.patch_height = 512

        if 'xception' in cfg.MODEL.NAME:
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(mean, axis=0), axis=-1), axis=-1)
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(std, axis=0), axis=-1), axis=-1)

        self.n_classes = 21
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE

        self.files = collections.defaultdict(list)

        self.tf = transform
        self.augmentations = augmentations

        self._setup_db()
        self.db_length = len(self.files[self.image_set])

        # logger.info('=> load {} samples'.format(len(self.db)))

    def __len__(self):
        return len(self.files[self.image_set])

    def __getitem__(self, index):
        im_name = self.files[self.image_set][index]
        im_path = os.path.join(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
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

    def _setup_db(self):

        for image_set in ["train", "val", "trainval"]:
            path = os.path.join(self.root, "ImageSets/Segmentation", image_set + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[image_set] = file_list
        self.setup_annotations()

    def transform(self, img, lbl):
        # if self.patch_width == "same" and self.patch_height ==  "same":
        #     pass
        # else:
        #     img = img.resize((self.patch_height, self.patch_width), Image.BILINEAR)  # uint8 with RGB mode
        #     lbl = lbl.resize((self.patch_height, self.patch_width), Image.NEAREST)

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

    def get_pascal_labels(self):
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

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

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

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = os.path.join(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        sbd_train_path = os.path.join(sbd_path, "dataset/train.txt")
        sbd_val_path = os.path.join(sbd_path, "dataset/val.txt")

        sbd_train_list = tuple(open(sbd_train_path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]

        sbd_val_list = tuple(open(sbd_val_path, "r"))
        sbd_val_list = [id_.rstrip() for id_ in sbd_val_list]

        sbd_all_list = sorted(list(set(sbd_train_list + sbd_val_list)))

        # validation set is pascal val minus sbd train (904 images)
        set_diff = set(self.files["val"]) - set(sbd_train_list)
        self.files["train_aug_val"] = sorted(list(set_diff))
        # training set is all combined (pascal and sbd) minus validation set
        all_list = self.files["trainval"] + sbd_all_list
        set_diff = set(all_list) - set(self.files["val"])
        self.files["train_aug"] = sorted(list(set_diff))

        pre_encoded = glob.glob(os.path.join(target_path, "*.png"))
        expected = np.unique(self.files["train_aug"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_all_list):
                lbl_path = os.path.join(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0]
                assert lbl.max() <= 20
                # lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                # m.imsave(os.path.join(target_path, ii + ".png"), lbl)
                imageio.imwrite(os.path.join(target_path, ii + ".png"), lbl.astype(np.uint8))

            for ii in tqdm(self.files["trainval"]):
                fname = ii + ".png"
                lbl_path = os.path.join(self.root, "SegmentationClass", fname)
                lbl = imageio.imread(lbl_path, pilmode='RGB')

                unique_lbl = np.unique(np.reshape(lbl, (-1, 3)), axis=0)
                pascal_lbl = self.get_pascal_labels()
                for l in unique_lbl:
                    assert (np.all(pascal_lbl == l, axis=-1).sum() == 1)

                lbl = self.encode_segmap(lbl)
                assert(lbl.max() <= 22)
                # lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                # m.imsave(os.path.join(target_path, fname), lbl)
                imageio.imwrite(os.path.join(target_path, fname), lbl.astype(np.uint8))

        assert expected == 12031, "unexpected dataset sizes"


# # Leave code for debugging purposes
# import lib.utils.augmentations as aug
# if __name__ == '__main__':
# # local_path = '/home/meetshah1995/datasets/VOCdevkit/VOC2012/'
#     bs = 1
#     augs = aug.Compose([aug.RandomScale(0.5, 1.5),
#                         aug.RandomRotate(10),
#                         aug.RandomHorizontallyFlip(0.5),
#                         aug.RandomCrop(20),
#                         aug.PadByStride(8)])
#     # augs = None
#     cfg = None
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     dst = pascalVOC(cfg,
#         'data/pascal_voc',
#         'train_aug',
#         transforms.Compose([
#             transforms.ToTensor(),
#             # normalize,
#         ]),
#         augmentations=augs)
#
#     trainloader = data.DataLoader(dst,
#         batch_size=bs,
#         shuffle=True,
#         pin_memory=True)
#
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
#             plt.show()
#             a = input()
#             if a == 'ex':
#                 break
#             else:
#                 plt.close()
