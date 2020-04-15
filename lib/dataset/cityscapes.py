import os
import math
import torch
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.utils import data
from torchvision import transforms

class CityScapes(data.Dataset):
    def __init__(self,
                 cfg,
                 root,
                 list_path,
                 num_classes=19,
                 transform=None,
                 augmentations=None,
                 ignore_label=255):

        self.cfg = cfg
        self.root = root
        self.list_path = list_path

        # self.multi_scale = multi_scale
        # self.flip = flip

        self.img_list = [line.strip().split() for line in open(list_path)]

        self.files = self.read_files()

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

        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}
        self.ignore_label = ignore_label
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843,
                                        1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507]).cuda()

        self.output_stride = cfg.MODEL.OUTPUT_STRIDE

        self.tf = transform
        self.augmentations = augmentations

        self.db_length = len(self.files)

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        # name = item["name"]
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        # size = image.shape

        im_name = item['img']
        im_path = os.path.join(self.root, im_name)

        im = Image.open(im_path)

        # if 'test' in self.list_path:
        #     image = self.input_transform(image)
        #     image = image.transpose((2, 0, 1))

        #     return image.copy(), np.array(size), name

        lbl_name = item['label']
        lbl_path = os.path.join(self.root, lbl_name)

        lbl = cv2.imread(os.path.join(self.root, lbl_name), cv2.IMREAD_GRAYSCALE)
        lbl = self.convert_label(lbl)

        lbl = Image.fromarray(lbl, mode="L")

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)

        im, lbl, lbl_os = self.transform(im, lbl)

        if self.cfg.MODEL.LEARN_PAIRWISE_TERMS:
            lbl_hs = []
            lbl_vs = []

            stride = 1
            for _ in range(self.cfg.MODEL.NUM_PAIRWISE_TERMS):
                for i in range(stride):
                    for j in range(stride):
                        lbl_os_ = lbl_os[i::stride, j::stride]

                        lbl_h = lbl_os_[:, :-1] * self.n_classes + lbl_os_[:, 1:]
                        lbl_v = lbl_os_[:-1, :] * self.n_classes + lbl_os_[1:, :]
                        lbl_h[(lbl_os_[:, :-1] >= self.n_classes) | (lbl_os_[:, 1:] >= self.n_classes)] = self.ignore_label
                        lbl_v[(lbl_os_[:-1, :] >= self.n_classes) | (lbl_os_[1:, :] >= self.n_classes)] = self.ignore_label

                        lbl_hs.append(lbl_h)
                        lbl_vs.append(lbl_v)

                stride += self.cfg.MODEL.PAIRWISE_STEP_SIZE

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
        lbl[lbl >= self.n_classes] = self.ignore_label  # ignore pixels

        lbl_os = torch.from_numpy(np.array(lbl_os)).long()
        lbl_os[lbl_os >= self.n_classes] = self.ignore_label  # ignore pixels

        return img, lbl, lbl_os

    def get_cityscapes_labels(self):
        """Load the mapping that associates cityscapes classes with label colors

        Returns:
            np.ndarray with dimensions (60, 3)
        """

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, self.n_classes - 1)]
        colors = np.array([[c[2] * 255, c[1] * 255, c[0] * 255] for c in colors])
        colors = np.vstack(([[0, 0, 0]], colors))

        return colors

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
        label_colours = self.get_cityscapes_labels()
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

    # def multi_scale_inference(self, model, image, scales=[1], flip=False):
    #     batch, _, ori_height, ori_width = image.size()
    #     assert batch == 1, "only supporting batchsize 1."
    #     image = image.numpy()[0].transpose((1,2,0)).copy()
    #     stride_h = np.int(self.crop_size[0] * 1.0)
    #     stride_w = np.int(self.crop_size[1] * 1.0)
    #     final_pred = torch.zeros([1, self.num_classes,
    #                                 ori_height,ori_width]).cuda()
    #     for scale in scales:
    #         new_img = self.multi_scale_aug(image=image,
    #                                        rand_scale=scale,
    #                                        rand_crop=False)
    #         height, width = new_img.shape[:-1]

    #         if scale <= 1.0:
    #             new_img = new_img.transpose((2, 0, 1))
    #             new_img = np.expand_dims(new_img, axis=0)
    #             new_img = torch.from_numpy(new_img)
    #             preds = self.inference(model, new_img, flip)
    #             preds = preds[:, :, 0:height, 0:width]
    #         else:
    #             new_h, new_w = new_img.shape[:-1]
    #             rows = np.int(np.ceil(1.0 * (new_h -
    #                             self.crop_size[0]) / stride_h)) + 1
    #             cols = np.int(np.ceil(1.0 * (new_w -
    #                             self.crop_size[1]) / stride_w)) + 1
    #             preds = torch.zeros([1, self.num_classes,
    #                                        new_h,new_w]).cuda()
    #             count = torch.zeros([1,1, new_h, new_w]).cuda()

    #             for r in range(rows):
    #                 for c in range(cols):
    #                     h0 = r * stride_h
    #                     w0 = c * stride_w
    #                     h1 = min(h0 + self.crop_size[0], new_h)
    #                     w1 = min(w0 + self.crop_size[1], new_w)
    #                     h0 = max(int(h1 - self.crop_size[0]), 0)
    #                     w0 = max(int(w1 - self.crop_size[1]), 0)
    #                     crop_img = new_img[h0:h1, w0:w1, :]
    #                     crop_img = crop_img.transpose((2, 0, 1))
    #                     crop_img = np.expand_dims(crop_img, axis=0)
    #                     crop_img = torch.from_numpy(crop_img)
    #                     pred = self.inference(model, crop_img, flip)
    #                     preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
    #                     count[:,:,h0:h1,w0:w1] += 1
    #             preds = preds / count
    #             preds = preds[:,:,:height,:width]
    #         preds = F.upsample(preds, (ori_height, ori_width),
    #                                mode='bilinear')
    #         final_pred += preds
    #     return final_pred

    # def get_palette(self, n):
    #     palette = [0] * (n * 3)
    #     for j in range(0, n):
    #         lab = j
    #         palette[j * 3 + 0] = 0
    #         palette[j * 3 + 1] = 0
    #         palette[j * 3 + 2] = 0
    #         i = 0
    #         while lab:
    #             palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
    #             palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
    #             palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
    #             i += 1
    #             lab >>= 3
    #     return palette

    # def save_pred(self, preds, sv_path, name):
    #     palette = self.get_palette(256)
    #     preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    #     for i in range(preds.shape[0]):
    #         pred = self.convert_label(preds[i], inverse=True)
    #         save_img = Image.fromarray(pred)
    #         save_img.putpalette(palette)
    #         save_img.save(os.path.join(sv_path, name[i]+'.png'))


# Leave code for debugging purposes
import lib.utils.augmentations as aug
from lib.core.config import config
if __name__ == '__main__':
    bs = 1

    config.MODEL.IMAGE_SIZE = (769, 769)
    config.MODEL.OUTPUT_STRIDE = 16
    config.MODEL.LEARN_PAIRWISE_TERMS = False

    augs = aug.Compose([aug.AdjustBrightness(0.1),
                        aug.AdjustContrast(0.1),
                        aug.AdjustSaturation(0.1),
                        aug.AdjustHue(0.1),
                        aug.RandomScale(0.75, 2.0),
                        aug.RandomRotate(10),
                        aug.RandomHorizontallyFlip(0.5),
                        aug.RandomSizedCrop(config.MODEL.IMAGE_SIZE)])
    # augs = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dst = CityScapes(config,
                     'data/cityscapes',
                     'data/list/cityscapes/train.lst',
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         # normalize,
                     ]),
                     augmentations=augs)

    trainloader = data.DataLoader(dst,
        batch_size=bs,
        num_workers=0,
        shuffle=True,
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
