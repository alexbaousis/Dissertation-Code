from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
import cv2
import torchvision


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        sample = self.normalize(sample.float())
        return sample

    def __str__(self):
        return 'Normalize([%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f])' % (
            self.mean[0], self.mean[1], self.mean[2],
            self.std[0], self.std[1], self.std[2]
        )


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    Modified so SAM masks / edges are optional.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal, sam_masks=None, sam_edges=None):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        img_ = F.interpolate(
            img[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        label_ = F.interpolate(
            label[None, None, i:i + h, j:j + w],
            size=(height, width),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        depth_ = F.interpolate(
            depth[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='nearest'
        ).squeeze(0)

        normal_ = F.interpolate(
            normal[None, :, i:i + h, j:j + w],
            size=(height, width),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        if sam_masks is not None:
            sam_masks_ = F.interpolate(
                sam_masks[None, :, i:i + h, j:j + w],
                size=(height, width),
                mode='nearest'
            ).squeeze(0)
        else:
            sam_masks_ = None

        if sam_edges is not None:
            sam_edges_ = F.interpolate(
                sam_edges[None, :, i:i + h, j:j + w],
                size=(height, width),
                mode='nearest'
            ).squeeze(0)
        else:
            sam_edges_ = None

        trans_params = torch.tensor([sc, h, w, i, j, height, width])

        return img_, label_, depth_ / sc, normal_, sam_masks_, sam_edges_, trans_params


class NYUv2(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, index=None):
        self.train = train
        self.root = os.path.expanduser(root)

        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        image = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)
        )
        semantic = torch.from_numpy(
            np.load(self.data_path + '/label/{:d}.npy'.format(index))
        )
        depth = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)
        )
        normal = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)
        )

        sam_masks = cv2.imread(self.data_path + '/sam/{:d}.png'.format(index))
        sam_masks = cv2.cvtColor(sam_masks, cv2.COLOR_RGB2GRAY)
        sam_masks = self.normalize(sam_masks)
        sam_masks = torch.from_numpy(sam_masks).unsqueeze(dim=0)

        sam_edges = cv2.imread(self.data_path + '/sam_edge/{:d}.png'.format(index))
        sam_edges = cv2.cvtColor(sam_edges, cv2.COLOR_RGB2GRAY)
        sam_edges = sam_edges / 255.
        sam_edges = torch.from_numpy(sam_edges).unsqueeze(dim=0)

        if self.train:
            return (
                image.type(torch.FloatTensor),
                semantic.type(torch.FloatTensor),
                depth.type(torch.FloatTensor),
                normal.type(torch.FloatTensor),
                sam_masks.type(torch.FloatTensor),
                sam_edges.type(torch.FloatTensor),
                index
            )
        else:
            return (
                image.type(torch.FloatTensor),
                semantic.type(torch.FloatTensor),
                depth.type(torch.FloatTensor),
                normal.type(torch.FloatTensor),
                sam_masks.type(torch.FloatTensor),
                sam_edges.type(torch.FloatTensor)
            )

    def __len__(self):
        return self.data_len

    def normalize(self, image):
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12)
        return image


class NYUv2_crop(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    Supports:
      - plain MTI-Net training (aug_twice=False)
      - two-view training without edges (aug_twice=True, sam_edge=False)
      - two-view training with edges (aug_twice=True, sam_edge=True)
    """
    def __init__(self, root, train=True, index=None, augmentation=False, aug_twice=False, sam_edge=True, sam_iou=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.aug_twice = aug_twice
        self.sam_edge = sam_edge
        self.sam_iou = sam_iou

        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        image = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)
        )
        semantic = torch.from_numpy(
            np.load(self.data_path + '/label/{:d}.npy'.format(index))
        )
        depth = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)
        )
        normal = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)
        )

        sam_masks = cv2.imread(self.data_path + '/sam/{:d}.png'.format(index))
        sam_masks = cv2.cvtColor(sam_masks, cv2.COLOR_RGB2GRAY)
        sam_masks = torch.from_numpy(sam_masks).unsqueeze(dim=0)

        sam_edges = cv2.imread(self.data_path + '/sam_edge/{:d}.png'.format(index))
        sam_edges = cv2.cvtColor(sam_edges, cv2.COLOR_RGB2GRAY)
        sam_edges = sam_edges / 255.
        sam_edges = torch.from_numpy(sam_edges).unsqueeze(dim=0)

        if self.augmentation and not self.aug_twice:
            image, semantic, depth, normal, _, _, _ = RandomScaleCrop()(
                image, semantic, depth, normal
            )
            return (
                image.type(torch.FloatTensor),
                semantic.type(torch.FloatTensor),
                depth.type(torch.FloatTensor),
                normal.type(torch.FloatTensor),
                index
            )

        elif self.augmentation and self.aug_twice and not self.sam_edge:
            image, semantic, depth, normal, sam_masks, _, _ = RandomScaleCrop()(
                image, semantic, depth, normal, sam_masks
            )
            image1, semantic1, depth1, normal1, sam_masks1, _, trans_params = RandomScaleCrop()(
                image, semantic, depth, normal, sam_masks
            )
            return (
                image.type(torch.FloatTensor),
                semantic.type(torch.FloatTensor),
                depth.type(torch.FloatTensor),
                normal.type(torch.FloatTensor),
                sam_masks.type(torch.FloatTensor),
                index,
                image1.type(torch.FloatTensor),
                semantic1.type(torch.FloatTensor),
                depth1.type(torch.FloatTensor),
                normal1.type(torch.FloatTensor),
                sam_masks1.type(torch.FloatTensor),
                trans_params
            )

        elif self.augmentation and self.aug_twice and self.sam_edge:
            image, semantic, depth, normal, sam_masks, sam_edges, _ = RandomScaleCrop()(
                image, semantic, depth, normal, sam_masks, sam_edges
            )
            image1, semantic1, depth1, normal1, sam_masks1, sam_edges1, trans_params = RandomScaleCrop()(
                image, semantic, depth, normal, sam_masks, sam_edges
            )
            return (
                image.type(torch.FloatTensor),
                semantic.type(torch.FloatTensor),
                depth.type(torch.FloatTensor),
                normal.type(torch.FloatTensor),
                sam_masks.type(torch.FloatTensor),
                sam_edges.type(torch.FloatTensor),
                index,
                image1.type(torch.FloatTensor),
                semantic1.type(torch.FloatTensor),
                depth1.type(torch.FloatTensor),
                normal1.type(torch.FloatTensor),
                sam_masks1.type(torch.FloatTensor),
                sam_edges1.type(torch.FloatTensor),
                trans_params
            )

        else:
            # No augmentation path
            if self.sam_edge:
                return (
                    image.type(torch.FloatTensor),
                    semantic.type(torch.FloatTensor),
                    depth.type(torch.FloatTensor),
                    normal.type(torch.FloatTensor),
                    sam_masks.type(torch.FloatTensor),
                    sam_edges.type(torch.FloatTensor),
                    index
                )
            else:
                return (
                    image.type(torch.FloatTensor),
                    semantic.type(torch.FloatTensor),
                    depth.type(torch.FloatTensor),
                    normal.type(torch.FloatTensor),
                    sam_masks.type(torch.FloatTensor),
                    index
                )

    def __len__(self):
        return self.data_len

    def normalize(self, image):
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12)
        return image

    def get_random_sample(self, index):
        pass