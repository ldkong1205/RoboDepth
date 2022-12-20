# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader.pt_data_loader.basedataset import BaseDataset
import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.definitions.labels_file as lf


class StandardDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(StandardDataset, self).__init__(*args, **kwargs)

        if self.disable_const_items is False:
            assert self.parameters.K is not None and self.parameters.stereo_T is not None, '''There are no K matrix and
            stereo_T parameter available for this dataset.'''

    def add_const_dataset_items(self, sample):
        K = self.parameters.K.copy()

        native_key = ('color', 0, -1) if (('color', 0, -1) in sample) else ('color_right', 0, -1)
        native_im_shape = sample[native_key].shape

        K[0, :] *= native_im_shape[1]
        K[1, :] *= native_im_shape[0]

        sample["K", -1] = K
        sample["stereo_T"] = self.parameters.stereo_T

        return sample