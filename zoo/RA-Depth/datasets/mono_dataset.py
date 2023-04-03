# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.height_full = 375
        self.width_full = 1242
        # self.num_scales = num_scales
        self.num_scales = 1
        self.interp = Image.ANTIALIAS   #a high-quality downsampling filter

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS, do_crop_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        self.resize_HiS = transforms.Resize((height_re_HiS, width_re_HiS), interpolation=self.interp)
        self.resize_MiS = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.resize_LoS = transforms.Resize((height_re_LoS, width_re_LoS), interpolation=self.interp)
        box_HiS = (dx_HiS, dy_HiS, dx_HiS+self.width, dy_HiS+self.height)
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n + "_HiS", im, i)] = self.resize_HiS(inputs[(n, im, i - 1)]).crop(box_HiS)
                    inputs[(n + "_MiS", im, i)] = self.resize_MiS(inputs[(n, im, i - 1)])
                    inputs[(n + "_LoS", im, i)] = self.resize_LoS(inputs[(n, im, i - 1)])


        for k in list(inputs):
            f = inputs[k]
            if "color_HiS" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
            if "color_MiS" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f) #[3,192,640]
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_LoS" in k:
                n, im, i = k
                LoS_part = self.to_tensor(f)
                point1 = int(2*width_re_LoS-self.width)
                point2 = int(2*height_re_LoS-self.height)
                Tensor_LoS = torch.zeros(3, self.height, self.width)
                Tensor_LoS[:, 0:height_re_LoS, 0:width_re_LoS] = LoS_part
                Tensor_LoS[:, height_re_LoS:self.height, 0:width_re_LoS] = LoS_part[:, point2:height_re_LoS, 0:width_re_LoS]
                Tensor_LoS[:, 0:height_re_LoS, width_re_LoS:self.width] = LoS_part[:, 0:height_re_LoS, point1:width_re_LoS]
                Tensor_LoS[:, height_re_LoS:self.height, width_re_LoS:self.width] = LoS_part[:, point2:height_re_LoS, point1:width_re_LoS]
                inputs[(n, im, i)] = Tensor_LoS

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        do_crop_aug = self.is_train

        # High-Scale
        ra_HiS = 1.1
        rb_HiS = 2.0
        resize_ratio_HiS = (rb_HiS - ra_HiS) * random.random() + ra_HiS
        if do_crop_aug:
            height_re_HiS = int(self.height * resize_ratio_HiS)
            width_re_HiS = int(self.width * resize_ratio_HiS)
        else:
            height_re_HiS = self.height
            width_re_HiS = self.width

        height_d_HiS = height_re_HiS - self.height
        width_d_HiS = width_re_HiS - self.width
        if do_crop_aug:
            dx_HiS = int(width_d_HiS * random.random())
            dy_HiS = int(height_d_HiS*random.random())
        else:
            dx_HiS = 0
            dy_HiS = 0


        # Middle-Scale
        dx_MiS = 0
        dy_MiS = 0


        # Low-Scale
        ra_LoS = 0.7
        rb_LoS = 0.9
        resize_ratio_LoS = (rb_LoS - ra_LoS) * random.random() + ra_LoS
        height_re_LoS = int(self.height * resize_ratio_LoS)
        width_re_LoS = int(self.width * resize_ratio_LoS)

        dx_LoS = 0
        dy_LoS = 0

        inputs[("dxy_HiS")] = torch.Tensor((dx_HiS, dy_HiS))
        inputs[("dxy_MiS")] = torch.Tensor((dx_MiS, dy_MiS))
        inputs[("dxy_LoS")] = torch.Tensor((dx_LoS, dy_LoS))
        inputs[("resize_HiS")] = torch.Tensor((width_re_HiS, height_re_HiS))
        inputs[("resize_LoS")] = torch.Tensor((width_re_LoS, height_re_LoS))


        line = self.filenames[index].split()
        folder = line[0]


        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)



        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K_HiS = self.K.copy()
            K_MiS = self.K.copy()
            K_LoS = self.K.copy()

            K_HiS[0, :] *= width_re_HiS // (2 ** scale)
            K_HiS[1, :] *= height_re_HiS // (2 ** scale)
            inv_K_HiS = np.linalg.pinv(K_HiS)
            inputs[("K_HiS", scale)] = torch.from_numpy(K_HiS)
            inputs[("inv_K_HiS", scale)] = torch.from_numpy(inv_K_HiS)

            K_MiS[0, :] *= self.width // (2 ** scale)
            K_MiS[1, :] *= self.height // (2 ** scale)
            inv_K_MiS = np.linalg.pinv(K_MiS)
            inputs[("K_MiS", scale)] = torch.from_numpy(K_MiS)
            inputs[("inv_K_MiS", scale)] = torch.from_numpy(inv_K_MiS)

            K_LoS[0, :] *= width_re_LoS // (2 ** scale)
            K_LoS[1, :] *= height_re_LoS // (2 ** scale)
            inv_K_LoS = np.linalg.pinv(K_LoS)
            inputs[("K_LoS", scale)] = torch.from_numpy(K_LoS)
            inputs[("inv_K_LoS", scale)] = torch.from_numpy(inv_K_LoS)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS, do_crop_aug)

        #删除原尺寸图像，-1表示原始图像(1242, 375)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, side, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            stereo_T_inv = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            stereo_T_inv[0, 3] = side_sign * baseline_sign * (-0.1)

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            inputs["stereo_T_inv"] = torch.from_numpy(stereo_T_inv)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
