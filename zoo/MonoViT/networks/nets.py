from __future__ import absolute_import, division, print_function
#from msilib.schema import Class

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
#from layers import *

#from .resnet_encoder import ResnetEncoder
from .hr_decoder import DepthDecoder
#from .pose_decoder import PoseDecoder
from .mpvit import *


class DeepNet(nn.Module):
    def __init__(self,type,weights_init= "pretrained",num_layers=18,num_pose_frames=2,scales=range(4)):
        super(DeepNet, self).__init__()
        self.type = type
        self.num_layers=num_layers
        self.weights_init=weights_init
        self.num_pose_frames=num_pose_frames
        self.scales = scales


        if self.type =='mpvitnet':
            self.encoder = mpvit_small()
            self.decoder = DepthDecoder()
 
        else:
            print("wrong type of the networks, only depthnet and posenet")
    

    def forward(self, inputs):
        if self.type =='mpvitnet': 
            self.outputs = self.decoder(self.encoder(inputs))
        else:
            self.outputs = self.decoder(self.encoder(inputs))
        return self.outputs
