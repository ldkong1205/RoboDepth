# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

""" Script to precompute depth hints using the 'fused SGM' method """

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F


import cv2
cv2.setNumThreads(0)

from utils import *
from layers import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def generate_stereo_matchers():
    """ Instantiate stereo matchers with different hyperparameters to build fused depth hints"""
    numDisparities = [64, 96, 128, 160]
    stereo_matchers = []
    for blockSize in [1, 2, 3]:
        for numDisparity in numDisparities:

            sad_window_size = 3
            stereo_params = dict(
                preFilterCap=63,
                P1=sad_window_size * sad_window_size * 4,
                P2=sad_window_size * sad_window_size * 32,
                minDisparity=0,
                numDisparities=numDisparity,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=16,
                blockSize=blockSize)
            stereo_matcher = cv2.StereoSGBM_create(**stereo_params)
            stereo_matchers.append(stereo_matcher)

    return stereo_matchers


def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """

    ssim = SSIM()
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


class DepthHintDataset:
    """
    Class to load data to precompute depth hints.

    Set up as a pytorch dataset to make use of pytorch DataLoader multithreading.
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height, width,
                 save_path,
                 overwrite):

        self.data_path = data_path
        self.filenames = filenames
        self.save_path = save_path
        self.overwrite = overwrite

        self.height, self.width = height, width

        self.interp = Image.ANTIALIAS
        self.resizer = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.stereo_matchers = generate_stereo_matchers()
        self.data_size = len(self.stereo_matchers)

        # setup intrinsics and extrinsics for reprojection
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0] *= self.width
        self.K[1] *= self.height
        self.invK = np.linalg.pinv(self.K)

        # convert everything to tensors and reshape into a batch
        self.K = \
            torch.from_numpy(self.K).unsqueeze(0).expand(self.data_size, -1, -1).float()
        self.invK = \
            torch.from_numpy(self.invK).unsqueeze(0).expand(self.data_size, -1, -1).float()

        self.baseline = 0.1  # the same baseline in datasets/mono_dataset.py
        self.T = torch.eye(4).unsqueeze(0).float()
        self.T[0, 0, 3] = self.baseline

    def __len__(self):
        return len(self.filenames)

    def compute_depths(self, base_image, lookup_image, reverse=False):
        """ For a given stereo pair, compute multiple depth maps using stereo matching
        (OpenCV Semi-Global Block Matching). Raw pixel disparities are converted to depth using
        focal length and baseline.

        Set reverse flag to be True if base image is on the right and lookup image is on the left
        (OpenCV SGBM computes disparity for the left image)"""

        if reverse:
            base_image = base_image[:, ::-1]
            lookup_image = lookup_image[:, ::-1]

        disps = []
        for matcher in self.stereo_matchers:
            disp = matcher.compute(base_image, lookup_image) / 16  # convert to pixel disparity
            if reverse:
                disp = disp[:, ::-1]
            disps.append(disp)

        disps = np.stack(disps)
        disps = torch.from_numpy(disps).float()

        # convert disp to depth ignoring missing pixels
        depths = self.K[0, 0, 0] * self.baseline / (disps + 1e-7) * (disps > 0).float()

        return depths

    def __getitem__(self, index):
        """ For a given image, get multiple depth maps, intrinsics, extrinsics and images. """
        inputs = {}

        sequence, frame, side = self.filenames[index].split()

        if side == 'l':
            side, otherside = 'image_02', 'image_03'
            baseline_sign = -1
        else:
            side, otherside = 'image_03', 'image_02'
            baseline_sign = 1

        if not self.overwrite:
            # if depth exists, then skip this image
            if os.path.isfile(os.path.join(self.save_path, sequence, side,
                                           '{}.npy'.format(str(frame).zfill(10)))):
                return inputs

        # flip extrinsics if necessary
        T = self.T
        T[0, 0, 3] = baseline_sign * self.baseline

        base_image = pil_loader(os.path.join(self.data_path, sequence, side,
                                       'data/{}.jpg'.format(str(frame).zfill(10))))
        lookup_image = pil_loader(os.path.join(self.data_path, sequence, otherside,
                                       'data/{}.jpg'.format(str(frame).zfill(10))))

        base_image = np.array(self.resizer(base_image))
        lookup_image = np.array(self.resizer(lookup_image))

        depths = self.compute_depths(base_image, lookup_image, reverse=side == 'image_03')

        # convert to tensors and reshape into batch
        base_image = torch.from_numpy(base_image).permute(2, 0, 1).float().unsqueeze(0)\
            .expand(self.data_size, -1, -1, -1) / 255
        lookup_image = torch.from_numpy(lookup_image).permute(2, 0, 1).float().unsqueeze(0)\
            .expand(self.data_size, -1, -1, -1) / 255

        inputs['base_image'] = base_image
        inputs['lookup_image'] = lookup_image
        inputs['K'] = self.K
        inputs['invK'] = self.invK
        inputs['depths'] = depths
        inputs['T'] = T

        return inputs


def run(opt):
    """ Computes depth hints for all files in opt.filenames.

    Makes use of pytorch DataLoader multithreading.
     """

    print('Computing depth hints...')

    if opt.save_path is None:
        opt.save_path = os.path.join(opt.data_path, 'depth_hints')
    print('Saving depth hints to {}'.format(opt.save_path))

    # setup projection mechanism - batch size of 12 as we have 4 x 3 stereo matchers
    cam_to_world = BackprojectDepth(12, opt.height, opt.width).cuda()
    world_to_cam = Project3D(12, opt.height, opt.width).cuda()

    # setup dataloader
    # batch size hardcoded to 1 as each item will contain multiple depth maps for a single image
    filenames = readlines(opt.filenames)
    dataset = DepthHintDataset(opt.data_path, filenames, opt.height, opt.width, opt.save_path,
                               opt.overwrite_saved_depths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=12)

    time_before = time.time()
    for i, data in enumerate(dataloader):

        # log timings
        if i % 50 == 0:
            print('image {} of {}'.format(i, len(dataloader)))
            if i != 0:
                time_taken = time.time() - time_before
                print('time for 50 imgs: {}s'.format(time_taken))
                print('imgs/s: {}'.format(50/time_taken))
                time_before = time.time()

        # check dataloader actually returned something, if not we have skipped an image
        if data:
            for key in data:
                if torch.cuda.is_available():
                    data[key] = data[key].cuda()
                data[key] = data[key][0]  # dataloader returns batch of size 1

            # for each pixel, find 'best' depth which gives the lowest reprojection loss
            world_points = cam_to_world(data['depths'], data['invK'])
            cam_pix = world_to_cam(world_points, data['K'], data['T'])
            sample = F.grid_sample(data['lookup_image'], cam_pix, padding_mode='border')
            losses = compute_reprojection_loss(sample, data['base_image'])
            best_index = torch.argmin(losses, dim=0)
            best_depth = torch.gather(data['depths'], dim=0, index=best_index).cpu().numpy()

            sequence, frame, side = filenames[i].split(' ')
            if side == 'l':
                side = 'image_02'
            else:
                side = 'image_03'

            savepath = os.path.join(opt.save_path, sequence, side)
            os.makedirs(savepath, exist_ok=True)
            np.save(os.path.join(savepath, '{}.npy'.format(str(frame).zfill(10))), best_depth)


def get_opts():
    """ parse command line options """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to images',
                        type=str)
    parser.add_argument('--filenames',
                        help='path to textfile containing list of images. Each line is expected to '
                             'be of the form "sequence_name frame_number side"',
                        type=str,
                        default='splits/eigen_full/all_files.txt')
    parser.add_argument('--save_path',
                        help='path to save resulting depth hints to. If not set will save to '
                             'datapath/depth_hints',
                        type=str)
    parser.add_argument('--height',
                        help='height of computed depth hints',
                        default=320,
                        type=int)
    parser.add_argument('--width',
                        help='width of computed depth hints',
                        default=1024,
                        type=int)
    parser.add_argument('--overwrite_saved_depths',
                        help='if set, will overwrite any existing depth hints rather than skipping',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    opts = get_opts()
    run(opts)
