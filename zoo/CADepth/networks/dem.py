import torch
import torch.nn as nn

class DEM(nn.Module):
    def __init__(self, channel):
        """ Detail Emphasis Module """
        super(DEM, self).__init__()
        
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(channel), 
                                   nn.ReLU(True))
        
        self.global_path = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.ReLU(True),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())

    def forward(self, x):
	    """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : recalibrated feature + input feature
                attention: B X C X 1 X 1
        """
	    out = self.conv1(x)
	    attention = self.global_path(out)      
        
	    return out + out * attention.expand_as(out) 
