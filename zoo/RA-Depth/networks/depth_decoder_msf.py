# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


#depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder_MSF(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_MSF, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc        #features in encoder, [64, 18, 36, 72, 144]

        # decoder
        self.convs = OrderedDict()

        self.convs[("parallel_conv"), 0, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 0, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 0, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("parallel_conv"), 0, 4] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
        self.convs[("conv1x1", 0, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 4_3)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[3])
        self.convs[("conv1x1", 0, 4_2)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 4_1)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 1, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("conv1x1", 1, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 1, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 1, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 3, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv1x1", 3, 1_0)] = ConvBlock1x1(self.num_ch_enc[1], self.num_ch_enc[0])

        self.convs[("parallel_conv"), 4, 0] = ConvBlock(self.num_ch_enc[0], 32)
        self.convs[("parallel_conv"), 5, 0] = ConvBlock(32, 16)
        self.convs[("dispconv", 0)] = Conv3x3(16, self.num_output_channels)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features):
        self.outputs = {}

        # features in encoder
        e4 = input_features[4]
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]

        d0_1 = self.convs[("parallel_conv"), 0, 1](e1)
        d0_2 = self.convs[("parallel_conv"), 0, 2](e2)
        d0_3 = self.convs[("parallel_conv"), 0, 3](e3)
        d0_4 = self.convs[("parallel_conv"), 0, 4](e4)

        d0_2_1 = updown_sample(d0_2, 2)
        d0_3_2 = updown_sample(d0_3, 2)
        d0_3_1 = updown_sample(d0_3, 4)
        d0_4_3 = updown_sample(d0_4, 2)
        d0_4_2 = updown_sample(d0_4, 4)
        d0_4_1 = updown_sample(d0_4, 8)

        d0_2_1 = self.convs[("conv1x1", 0, 2_1)](d0_2_1)
        d0_3_2 = self.convs[("conv1x1", 0, 3_2)](d0_3_2)
        d0_3_1 = self.convs[("conv1x1", 0, 3_1)](d0_3_1)
        d0_4_3 = self.convs[("conv1x1", 0, 4_3)](d0_4_3)
        d0_4_2 = self.convs[("conv1x1", 0, 4_2)](d0_4_2)
        d0_4_1 = self.convs[("conv1x1", 0, 4_1)](d0_4_1)

        d0_1_msf = d0_1 + d0_2_1 + d0_3_1 + d0_4_1
        d0_2_msf = d0_2 + d0_3_2 + d0_4_2
        d0_3_msf = d0_3 + d0_4_3


        d1_1 = self.convs[("parallel_conv"), 1, 1](d0_1_msf)
        d1_2 = self.convs[("parallel_conv"), 1, 2](d0_2_msf)
        d1_3 = self.convs[("parallel_conv"), 1, 3](d0_3_msf)

        d1_2_1 = updown_sample(d1_2, 2)
        d1_3_2 = updown_sample(d1_3, 2)
        d1_3_1 = updown_sample(d1_3, 4)

        d1_2_1 = self.convs[("conv1x1", 1, 2_1)](d1_2_1)
        d1_3_2 = self.convs[("conv1x1", 1, 3_2)](d1_3_2)
        d1_3_1 = self.convs[("conv1x1", 1, 3_1)](d1_3_1)

        d1_1_msf = d1_1 + d1_2_1 + d1_3_1
        d1_2_msf = d1_2 + d1_3_2


        d2_1 = self.convs[("parallel_conv"), 2, 1](d1_1_msf)
        d2_2 = self.convs[("parallel_conv"), 2, 2](d1_2_msf)

        d2_2_1 = updown_sample(d2_2, 2)

        d2_2_1 = self.convs[("conv1x1", 2, 2_1)](d2_2_1)

        d2_1_msf = d2_1 + d2_2_1


        d3_0 = self.convs[("parallel_conv"), 3, 0](e0)
        d3_1 = self.convs[("parallel_conv"), 3, 1](d2_1_msf)

        d3_1_0 = updown_sample(d3_1, 2)

        d3_1_0 = self.convs[("conv1x1", 3, 1_0)](d3_1_0)

        d3_0_msf = d3_0 + d3_1_0


        d4_0 = self.convs[("parallel_conv"), 4, 0](d3_0_msf)
        d4_0 = updown_sample(d4_0, 2)
        d5 = self.convs[("parallel_conv"), 5, 0](d4_0)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d5))

        return self.outputs     #single-scale depth






