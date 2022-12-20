'''
Seokju Lee

# (+) customized inputs: images (src/tgt), segmentation mask (src/tgt), intrinsics

'''
from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
import cv2
from matplotlib import pyplot as plt
import pdb

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, segms, intrinsics):
        for t in self.transforms:
            images, segms, intrinsics = t(images, segms, intrinsics)
        return images, segms, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, segms, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, segms, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, segms, intrinsics):
        img_tensors = []
        seg_tensors = []
        for im in images:
            im = np.transpose(im, (2, 0, 1))                        # put it from HWC to CHW format
            img_tensors.append(torch.from_numpy(im).float()/255)    # handle numpy array
        for im in segms:
            im = np.transpose(im, (2, 0, 1))
            seg_tensors.append(torch.from_numpy(im).float())
        return img_tensors, seg_tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, segms, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            output_segms = [np.copy(np.fliplr(im)) for im in segms]
            
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_segms = segms
            output_intrinsics = intrinsics
        return output_images, output_segms, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, segms, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]                                       # scipy.misc.imresize는 255 스케일로 변환됨!
        scaled_segms = [cv2.resize(im, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) for im in segms]      # 이 부분에서 1채널 세그먼트 [256 x 832 x 1] >> [256 x 832]로 변환됨!

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        cropped_segms = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_segms]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, cropped_segms, output_intrinsics
