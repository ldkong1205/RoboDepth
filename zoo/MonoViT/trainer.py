
from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:

    #######################
    ####   MonoViT      ##
    ######################
        #self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.params = [ {
            "params":self.parameters_to_train, 
            "lr": 1e-4
            #"weight_decay": 0.01
            },
            {
            "params": list(self.models["encoder"].parameters()), 
           "lr": self.opt.learning_rate
            #"weight_decay": 0.01
            } ]
            
        self.model_optimizer = optim.AdamW(self.params)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
		self.model_optimizer,0.9)
