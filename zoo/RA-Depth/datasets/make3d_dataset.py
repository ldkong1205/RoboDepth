# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .make3d import Make3dDataset


class MAKE3DDataset(Make3dDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(MAKE3DDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_color(self, folder):
        color = self.loader(self.get_image_path(folder))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class Make3dTestDataset(MAKE3DDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(Make3dTestDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder):
        image_path = os.path.join(
            self.data_path,
            folder)
        return image_path

