import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(ConvBnRelu, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class ConvElu(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(ConvElu, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, padding_mode="reflect")
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.elu(self.conv_s1(hx))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU7, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU6, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU5, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU4, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU4F, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


### RSU-3 ###
class RSU3(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU3, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx3 = self.rebnconv3(hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-2 ###
class RSU2(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU2, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx2 = self.rebnconv2(hx1)

        hx1d = self.rebnconv1d(torch.cat((hx1, hx2), 1))

        return hx1d + hxin