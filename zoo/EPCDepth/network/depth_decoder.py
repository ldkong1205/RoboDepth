import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, inputs):
        inputs = self.pad(inputs)
        out = self.conv(inputs)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        out = self.upsamp(inputs)
        out = self.conv(out)
        out = self.elu(out)
        return out


class IConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IConv, self).__init__()
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.elu(out)
        return out


class DispConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DispConv, self).__init__()
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.sigmod = nn.Sigmoid()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.sigmod(out)
        return out


class DepthDecoder(nn.Module):
    def __init__(self, encoder_layer_channels, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        decoder_layer_channels = [256, 128, 64, 32, 16]

        self.upconv5 = UpConv(encoder_layer_channels[-1], decoder_layer_channels[0])
        self.iconv5 = IConv(in_channels=decoder_layer_channels[0] + encoder_layer_channels[-2], out_channels=decoder_layer_channels[0])

        self.upconv4 = UpConv(decoder_layer_channels[0], decoder_layer_channels[1])
        self.iconv4 = IConv(in_channels=decoder_layer_channels[1] + encoder_layer_channels[-3], out_channels=decoder_layer_channels[1])

        self.upconv3 = UpConv(decoder_layer_channels[1], decoder_layer_channels[2])
        self.iconv3 = IConv(in_channels=decoder_layer_channels[2] + encoder_layer_channels[-4], out_channels=decoder_layer_channels[2])

        self.upconv2 = UpConv(decoder_layer_channels[2], decoder_layer_channels[3])
        self.iconv2 = IConv(in_channels=decoder_layer_channels[3] + encoder_layer_channels[-5], out_channels=decoder_layer_channels[3])

        self.upconv1 = UpConv(decoder_layer_channels[3], decoder_layer_channels[4])
        self.iconv1 = IConv(in_channels=decoder_layer_channels[4], out_channels=decoder_layer_channels[4])

        self.disps = nn.ModuleList()
        for channel in decoder_layer_channels[1:]:
            self.disps.append(DispConv(channel, num_output_channels))

    def forward(self, inputs):
        # decoder
        up5 = self.upconv5(inputs[-1])
        i5 = self.iconv5(torch.cat([up5, inputs[-2]], dim=1))

        up4 = self.upconv4(i5)
        i4 = self.iconv4(torch.cat([up4, inputs[-3]], dim=1))

        up3 = self.upconv3(i4)
        i3 = self.iconv3(torch.cat([up3, inputs[-4]], dim=1))

        up2 = self.upconv2(i3)
        i2 = self.iconv2(torch.cat([up2, inputs[-5]], dim=1))

        up1 = self.upconv1(i2)
        i1 = self.iconv1(up1)

        disp_features = [i4, i3, i2, i1]
        disps = []
        for i in range(len(disp_features)):
            disps.append(self.disps[i](disp_features[i]))

        return disps[::-1]