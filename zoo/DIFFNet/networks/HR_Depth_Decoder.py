from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from hr_layers import *
from layers import upsample

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else: 
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64 , 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18,[feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))
        
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
        
