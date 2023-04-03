# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import constant_init, kaiming_init


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def right_from_parameters(axisangle, translation0, translation1):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t0 = translation0.clone()
    t1 = translation1.clone()

    t1 *= -1
    t = t0 + t1

    T = get_translation_matrix(t)

    M = torch.matmul(R, T)

    return M


def left_from_parameters(translation):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    t = translation.clone()

    t *= -1

    T = get_translation_matrix(t)

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock_down(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_down, self).__init__()

        self.conv = Conv3x3_down(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock1x3_3x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x3_3x1, self).__init__()

        self.conv = Conv1x3_3x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Conv1x3_3x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1x3_3x1, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv1x3 = nn.Conv2d(int(in_channels), int(out_channels), (1, 3))
        self.conv3x1 = nn.Conv2d(int(out_channels), int(out_channels), (3, 1))
        # self.elu1 = nn.ELU(inplace=True)
        # self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1x3(out)
        # out = self.elu1(out)
        out = self.conv3x1(out)
        # out = self.elu2(out)
        return out


class Conv3x3_down(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3_down, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, 2)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class DetailGuide(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(DetailGuide, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, height_re, width_re, dxy):

        points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = nn.Parameter(torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0), requires_grad=False).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]

            points_all.append(pix_coords)

        points_all = torch.cat(points_all, 0)   #[12,2,122880]
        points_all = points_all.view(self.batch_size, 2, self.height, self.width)
        points_all = points_all.permute(0, 2, 3, 1) #[12,192,640,2]
        points_all[..., 0] *= (self.width * 1.0 / width_re)
        points_all[..., 1] *= (self.height * 1.0 / height_re)
        points_all[..., 0] /= self.width - 1
        points_all[..., 1] /= self.height - 1
        points_all = (points_all - 0.5) * 2

        return points_all


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                              requires_grad=False)   #[1,1,122880]

        # meshgrid = np.meshgrid(range(dx, dx+640), range(dy, dy+192), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)  #[2,192,640]

        # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
        #                          requires_grad=False)   #[12,1,122880]

        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)    #[1,2,122880]
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  #[12,2,122880]
        # self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)   #[12,3,122880]

    def forward(self, depth, inv_K, dxy):

        cam_points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]
            pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                                           requires_grad=False).cuda()   #[1,3,122880]


            cam_points = torch.matmul(inv_K[i, :3, :3], pix_coords)   #[1,3,122880]
            cam_points = depth[i,0,:,:].view(1, 1, -1) * cam_points   #[1,3,122880]
            cam_points = torch.cat([cam_points, ones], 1)   #[1,4,122880]
            cam_points_all.append(cam_points)

        cam_points_all = torch.cat(cam_points_all, 0)

        return cam_points_all


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, dxy):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        dxy = dxy.unsqueeze(1).unsqueeze(2).expand(-1, self.height, self.width, -1)
        pix_coords -= dxy
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Project3D_poseconsis(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_poseconsis, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, T):
        # P = torch.matmul(K, T)[:, :3, :] #P:[12,3,4]   T:[12,4,4]    points:[12,4,1]

        # cam_points = torch.matmul(P, points)

        cam1 = torch.matmul(T, points)  #[12,4,1]

        return cam1


def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest")


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def downsample(x):
    """Downsample input tensor by a factor of 1/2
    """
    return F.interpolate(x, scale_factor=1.0/2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


################################################ SENet ######################

class SEModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SEModule, self).__init__()

        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        # self.sigmoid = nn.Sigmoid()


    def forward(self, features):

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # y = self.sigmoid(y) * 2.0
        features = features + y.expand_as(features)

        return features





