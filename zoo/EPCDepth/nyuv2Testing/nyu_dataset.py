# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import copy
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

import time
import h5py

CROP = 16


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.array(img.convert('RGB'))
            h, w, c = img.shape
            return img


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    norm = np.array(h5f['norm'])
    norm = np.transpose(norm, (1, 2, 0))
    valid_mask = np.array(h5f['mask'])

    return rgb, depth, norm, valid_mask


class NYUTestDataset(data.Dataset):
    def __init__(self, data_path, height, width):
        super(NYUTestDataset, self).__init__()
        self.full_res_shape = (640 - CROP * 2, 480 - CROP * 2)
        self.K = self._get_intrinsics()

        self.data_path = data_path
        self.filenames = readlines('nyu_test.txt')
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.loader = h5_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index]
        line = os.path.join(self.data_path, line)
        rgb, depth, norm, valid_mask = self.loader(line)

        rgb = rgb[44: 471, 40: 601, :]
        depth = depth[44: 471, 40: 601]
        norm = norm[44:471, 40:601, :]
        valid_mask = valid_mask[44:471, 40:601]

        rgb = Image.fromarray(rgb)
        rgb = self.to_tensor(self.resize(rgb))

        depth = self.to_tensor(depth)
        norm = self.to_tensor(norm)
        norm_mask = self.to_tensor(valid_mask)

        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        return rgb, depth, norm, norm_mask, K, np.linalg.pinv(K)

    def _get_intrinsics(self):
        # 640, 480
        w, h = self.full_res_shape

        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = 3.2558244941119034e+02 / w
        cy = 2.5373616633400465e+02 / h

        intrinsics = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics
