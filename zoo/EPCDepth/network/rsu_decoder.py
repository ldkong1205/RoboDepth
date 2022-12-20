import torch
import torch.nn as nn
from network.rsu_layer import *


class EncoderDisp(nn.Module):
    def __init__(self, bott_channels, out_channels, bottleneck):
        super(EncoderDisp, self).__init__()
        self.bottleneck = bottleneck
        self.disp = nn.Sequential(
            nn.Conv2d(bott_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        features = self.bottleneck(inputs)
        out = self.disp(features)
        return out


class RSUDecoder(nn.Module):
    def __init__(self, encoder_layer_channels, num_output_channels=1, use_encoder_disp=False):
        super(RSUDecoder, self).__init__()

        self.use_encoder_disp = use_encoder_disp
        decoder_layer_channels = [256, 128, 64, 32, 16]

        # decoder
        self.stage5d = RSU3(encoder_layer_channels[-1] + encoder_layer_channels[-2], 64, decoder_layer_channels[0], False)

        self.stage4d = RSU4(decoder_layer_channels[0] + encoder_layer_channels[-3], 32, decoder_layer_channels[1], False)

        self.stage3d = RSU5(decoder_layer_channels[1] + encoder_layer_channels[-4], 16, decoder_layer_channels[2], False)

        self.stage2d = RSU6(decoder_layer_channels[2] + encoder_layer_channels[-5], 8, decoder_layer_channels[3], False)

        self.stage1d = RSU7(decoder_layer_channels[3], 4, decoder_layer_channels[4], False)

        if use_encoder_disp:
            self.encoder_disps = nn.ModuleList()
            bottlenecks = [RSU7, RSU6, RSU5, RSU4, RSU3]
            mid_channels = [32, 32, 64, 128, 256]
            in_channels = encoder_layer_channels
            out_channels = [64, 64, 128, 256, 512]
            for c, mid_c, bott_c, bottleneck in zip(in_channels, mid_channels, out_channels, bottlenecks):
                self.encoder_disps.append(EncoderDisp(bott_c, num_output_channels, bottleneck(c, mid_c, bott_c, False)))

        self.disps = nn.ModuleList()
        for channel in decoder_layer_channels:
            self.disps.append(nn.Sequential(nn.Conv2d(channel, num_output_channels, 3, 1, 1, padding_mode="reflect"), nn.Sigmoid()))

        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, inputs):

        hx6up = self.upsamp(inputs[-1])
        hx5d = self.stage5d(torch.cat((hx6up, inputs[-2]), 1))

        hx5dup = self.upsamp(hx5d)
        hx4d = self.stage4d(torch.cat((hx5dup, inputs[-3]), 1))

        hx4dup = self.upsamp(hx4d)
        hx3d = self.stage3d(torch.cat((hx4dup, inputs[-4]), 1))

        hx3dup = self.upsamp(hx3d)
        hx2d = self.stage2d(torch.cat((hx3dup, inputs[-5]), 1))

        hx2dup = self.upsamp(hx2d)
        hx1d = self.stage1d(hx2dup)

        disp_features = [hx5d, hx4d, hx3d, hx2d, hx1d]
        disps = []
        for i in range(len(disp_features)):
            disps.append(self.disps[i](disp_features[i]))

        if self.use_encoder_disp:
            encoder_disps = []
            for i in range(len(inputs)):
                encoder_disps.append(self.encoder_disps[i](inputs[i]))
            disps = encoder_disps + disps

        return disps[::-1]
