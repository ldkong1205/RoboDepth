import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.depth_utils import ConvBlock


def W(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)

    )

class MultiEmbedding(nn.Module):
    def __init__(self, in_channels, num_head, ratio):
        super(MultiEmbedding, self).__init__()
        self.in_channels = in_channels

        self.num_head = num_head
        self.out_channel = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.W = W(int(in_channels * ratio), in_channels)

        self.fuse = nn.Sequential(ConvBlock(in_channels * 2, in_channels),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1))

    def forward(self, key, query):

        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channel)

        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
        else:
            softmax = F.softmax(att, dim=1).unsqueeze(dim=2)

        weighted_value = v_out * softmax
        weighted_value = weighted_value.sum(dim=1)
        out = self.W(weighted_value)

        return self.fuse(torch.cat([key, out], dim=1))
