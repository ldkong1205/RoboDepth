'''
Seokju Lee

'''

from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from rigid_warp import inverse_warp_mof, pose_mof2mat, flow_warp
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import pdb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class SSIM(nn.Module):
    '''
        Layer to compute the SSIM loss between a pair of images
    '''

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



compute_ssim_loss = SSIM().to(device)



def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, motion_fields_fwd, motion_fields_bwd, with_ssim, with_mask, with_auto_mask, padding_mode, with_only_obj, tgt_obj_masks, ref_obj_masks, vmasks_fwd, vmasks_bwd):

    photo_loss = 0
    geometry_loss = 0

    r2t_imgs, t2r_imgs = [], []
    r2t_flows, t2r_flows = [], []
    r2t_diffs, t2r_diffs = [], []
    r2t_vals, t2r_vals = [], []

    for ref_img, ref_depth, mf_fwd, mf_bwd, tgt_obj_mask, ref_obj_mask, vmask_fwd, vmask_bwd in zip(ref_imgs, ref_depths, motion_fields_fwd, motion_fields_bwd, tgt_obj_masks, ref_obj_masks, vmasks_fwd, vmasks_bwd):
        photo_loss1, geometry_loss1, r2t_img, tgt_comp_depth, r2t_flow, r2t_diff, r2t_val = compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, mf_fwd, \
                                                                                                                  intrinsics, with_ssim, with_mask, with_auto_mask, padding_mode, \
                                                                                                                  with_only_obj, tgt_obj_mask, vmask_fwd.detach())
        photo_loss2, geometry_loss2, t2r_img, ref_comp_depth, t2r_flow, t2r_diff, t2r_val = compute_pairwise_loss(ref_img, tgt_img, ref_depth, tgt_depth, mf_bwd, \
                                                                                                                  intrinsics, with_ssim, with_mask, with_auto_mask, padding_mode, \
                                                                                                                  with_only_obj, ref_obj_mask, vmask_bwd.detach())
        r2t_imgs.append(r2t_img)
        t2r_imgs.append(t2r_img)
        r2t_flows.append(r2t_flow)
        t2r_flows.append(t2r_flow)
        r2t_diffs.append(r2t_diff)
        t2r_diffs.append(t2r_diff)
        r2t_vals.append(r2t_val)
        t2r_vals.append(t2r_val)

        photo_loss += (photo_loss1 + photo_loss2)
        geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss, r2t_imgs, t2r_imgs, r2t_flows, t2r_flows, r2t_diffs, t2r_diffs, r2t_vals, t2r_vals



def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, motion_field, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode, with_only_obj, obj_mask, vmask):

    ref_img_warped, valid_mask, projected_depth, computed_depth, r2t_flow = inverse_warp_mof(ref_img, tgt_depth, ref_depth, motion_field, intrinsic, padding_mode)
    
    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)      # hyper-parameter

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    if with_only_obj == True:
        valid_mask = valid_mask * obj_mask

    out_val = valid_mask * vmask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, out_val)
    geometry_consistency_loss = mean_on_mask(diff_depth, out_val)

    return reconstruction_loss, geometry_consistency_loss, ref_img_warped, computed_depth, r2t_flow, diff_depth, out_val



def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """
            Computes the smoothness loss for a disparity image
            The color image is used for edge-aware smoothness
        """
        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp - torch.roll(disp, 1, dims=3))
        grad_disp_y = torch.abs(disp - torch.roll(disp, 1, dims=2))
        grad_disp_x[:,:,:,0] = 0
        grad_disp_y[:,:,0,:] = 0

        grad_img_x = torch.mean(torch.abs(img - torch.roll(img, 1, dims=3)), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img - torch.roll(img, 1, dims=2)), 1, keepdim=True)
        grad_img_x[:,:,:,0] = 0
        grad_img_y[:,:,0,:] = 0

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth, ref_img)

    return loss



def compute_obj_size_constraint_loss(height_prior, tgtD, tgtMs, refDs, refMs, intrinsics, mni, num_insts):
    '''
        Reference: Struct2Depth (AAAI'19), https://github.com/tensorflow/models/blob/archive/research/struct2depth/model.py
        args:
            D_avg, D_obj, H_obj, D_app: tensor([d1, d2, d3, ... dn], device='cuda:0')
            num_inst: [n1, n2, ...]
            intrinsics.shape: torch.Size([B, 3, 3])
    '''
    bs, _, hh, ww = tgtD.size()

    loss = torch.tensor(.0).cuda()

    for tgtM, refD, refM, num_inst in zip(tgtMs, refDs, refMs, num_insts):
        if sum(num_inst) != 0:
            fy_rep = intrinsics[:,1,1].repeat_interleave(mni, dim=0)
            
            ### tgt-frame ###
            tgtD_rep = tgtD.repeat_interleave(mni, dim=0)
            tgtD_avg = tgtD_rep.mean(dim=[1,2,3])
            tgtM_rep = tgtM[:,1:].reshape(-1,1,hh,ww)
            tgtD_obj = (tgtD_rep * tgtM_rep).sum(dim=[1,2,3]) / tgtM_rep.sum(dim=[1,2,3]).clamp(min=1e-9)
            tgtM_idx = np.where(tgtM_rep.detach().cpu().numpy()==1)
            tgtH_obj = torch.tensor([ tgtM_idx[2][tgtM_idx[0]==obj].max() - tgtM_idx[2][tgtM_idx[0]==obj].min() if (tgtM_idx[0]==obj).sum()!=0 else 0 for obj in range(tgtM_rep.size(0)) ]).type_as(tgtD)

            tgt_val = (tgtD_obj > 0) * (tgtH_obj > 0)
            
            tgt_fy = fy_rep[tgt_val]
            tgtD_avg = tgtD_avg[tgt_val].detach()   # d_avg.detach() to prevent increasing depth in the sky.
            # tgtD_avg = tgtD_avg[tgt_val]
            tgtD_obj = tgtD_obj[tgt_val]
            tgtH_obj = tgtH_obj[tgt_val]
            tgtD_app = (tgt_fy * height_prior) / tgtH_obj

            loss_tgt = torch.abs( (tgtD_obj-tgtD_app)/tgtD_avg ).sum() / torch.abs( (tgtD_obj-tgtD_app)/tgtD_avg ).size(0)
            

            ### ref-frame ###
            refD_rep = refD.repeat_interleave(mni, dim=0)
            refD_avg = refD_rep.mean(dim=[1,2,3])
            refM_rep = refM[:,1:].reshape(-1,1,hh,ww)
            refD_obj = (refD_rep * refM_rep).sum(dim=[1,2,3]) / refM_rep.sum(dim=[1,2,3]).clamp(min=1e-9)
            refM_idx = np.where(refM_rep.detach().cpu().numpy()==1)
            refH_obj = torch.tensor([ refM_idx[2][refM_idx[0]==obj].max() - refM_idx[2][refM_idx[0]==obj].min() if (refM_idx[0]==obj).sum()!=0 else 0 for obj in range(refM_rep.size(0)) ]).type_as(refD)

            ref_val = (refD_obj > 0) * (refH_obj > 0)
            
            ref_fy = fy_rep[ref_val]
            refD_avg = refD_avg[ref_val].detach()   # d_avg.detach() to prevent increasing depth in the sky.
            # refD_avg = refD_avg[ref_val]
            refD_obj = refD_obj[ref_val]
            refH_obj = refH_obj[ref_val]
            refD_app = (ref_fy * height_prior) / refH_obj

            loss_ref = torch.abs( (refD_obj-refD_app)/refD_avg ).sum() / torch.abs( (refD_obj-refD_app)/refD_avg ).size(0)

            loss += 1/2 * (loss_tgt + loss_ref)

    return loss



def compute_mof_consistency_loss(tgt_mofs, ref_mofs, r2t_flows, t2r_flows, r2t_diffs, t2r_diffs, r2t_vals, t2r_vals, alpha=10, thresh=0.1):
    '''
        Reference: Depth from Videos in the Wild (ICCV'19)
        Args:
            [DIRECTION]
            tgt_mofs_dir[0]: ref[0] >> tgt
            tgt_mofs_dir[1]:           tgt << ref[1]
            [MAGNITUDE]
            tgt_mofs_mag[0]: ref[0] >> tgt
            tgt_mofs_mag[1]:           tgt << ref[1]
    '''
    bs, _, hh, ww = tgt_mofs[0].size()
    eye = torch.eye(3).reshape(1,1,3,3).repeat(bs,hh*ww,1,1).type_as(tgt_mofs[0])
    
    loss = torch.tensor(.0).cuda()

    for enum, (tgt_mof, ref_mof, r2t_flow, t2r_flow, r2t_diff, t2r_diff, r2t_val, t2r_val) in \
        enumerate(zip(tgt_mofs, ref_mofs, r2t_flows, t2r_flows, r2t_diffs, t2r_diffs, r2t_vals, t2r_vals)):

        tgt_mat = pose_mof2mat(tgt_mof)
        ref_mat = pose_mof2mat(ref_mof)
        
        ### rotation error ###
        tgt_rot = tgt_mat[:,:,:3].reshape(bs,3,3,-1).permute(0,3,1,2)
        ref_rot = ref_mat[:,:,:3].reshape(bs,3,3,-1).permute(0,3,1,2)
        rot_unit = torch.matmul(tgt_rot, ref_rot)

        rot_err = torch.mean(torch.pow(rot_unit - eye, 2), dim=[2,3]).reshape(bs, 1, hh, ww)
        rot1_scale = torch.mean(torch.pow(tgt_rot - eye, 2), dim=[2,3]).reshape(bs, 1, hh, ww)
        rot2_scale = torch.mean(torch.pow(ref_rot - eye, 2), dim=[2,3]).reshape(bs, 1, hh, ww)
        rot_err /= (1e-24 + rot1_scale + rot2_scale)
        cost_r = rot_err.mean()
        # pdb.set_trace()

        ### translation error ###
        r2t_mof, _ = flow_warp(ref_mof, r2t_flow.detach())     # to be compared with "tgt_mof"
        r2t_mask = ( (1- (r2t_diff>thresh).float()) * r2t_val ).detach()
        r2t_mat = pose_mof2mat(r2t_mof)

        r2t_trans = r2t_mat[:,:,-1].reshape(bs,3,-1).permute(0,2,1).unsqueeze(-1)
        tgt_trans = tgt_mat[:,:,-1].reshape(bs,3,-1).permute(0,2,1).unsqueeze(-1)
        trans_zero = torch.matmul(tgt_rot, r2t_trans) + tgt_trans
        trans_zero_norm = torch.pow(trans_zero, 2).sum(dim=2).reshape(bs,1,hh,ww)
        r2t_trans_norm = torch.pow(r2t_trans, 2).sum(dim=2).reshape(bs,1,hh,ww)
        tgt_trans_norm = torch.pow(tgt_trans, 2).sum(dim=2).reshape(bs,1,hh,ww)

        trans_err = trans_zero_norm / (1e-24 + r2t_trans_norm + tgt_trans_norm)
        cost_t = mean_on_mask( trans_err, r2t_mask )

        loss += cost_r + alpha*cost_t
        # pdb.set_trace()

    # pdb.set_trace()
    '''
        r2t_mof, r2t_val0 = flow_warp(ref_mof, r2t_flow)
        t2r_mof, t2r_val0 = flow_warp(tgt_mof, t2r_flow)
        tgt_err = (tgt_mof + r2t_mof).abs()
        ref_err = (ref_mof + t2r_mof).abs()
        bb = 0
        vm = 0.02
        plt.close('all'); ea1 = 7; ea2 = 4; ii = 1;
        fig = plt.figure(99, figsize=(21, 12))   # figsize=(22, 13)
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(tgt_mof[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgt_mof", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(ref_mof[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "ref_mof", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_mof[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_mof", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_mof[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r_mof", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(tgt_err[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgt_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(ref_err[bb,2].detach().cpu(), vmax=+vm, vmin=-vm); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "ref_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_diff[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_diff", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_diff[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r_diff", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_val[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_val", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_val0[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_val0", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_val[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r_val", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_val0[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r_val0", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(fwd_mask[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "fwd_mask", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(bwd_mask[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "bwd_mask", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(fwd_val[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "fwd_val", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_mask[bb,0].detach().cpu(), vmax=1, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_mask", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(rot_err[bb,0].detach().cpu() ); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "rot_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(trans_err[bb,0].detach().cpu(), vmax=0.1, vmin=0 ); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "trans_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
        plt.tight_layout(); plt.ion(); plt.show()

    '''
    return loss / (enum+1)



################################################################################################################################################################################


def mean_on_mask(diff, valid_mask):
    '''
        compute mean value given a binary mask
    '''
    mask = valid_mask.expand_as(diff)
    if mask.sum() == 0:
        return torch.tensor(.0).cuda()
    else:
        return (diff * mask).sum() / mask.sum()
        


@torch.no_grad()
def compute_errors(gt, pred, med_scale=None):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
        crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        construct a mask of False values, with the same size as target
        and then set to True values inside the crop
    '''
    crop_mask = gt[0] != gt[0]
    y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
    x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
    crop_mask[y1:y2,x1:x2] = 1
    max_depth = 80

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        if med_scale is None:
            med_scale = torch.median(valid_gt) / torch.median(valid_pred)
        
        valid_pred = valid_pred * med_scale

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]], med_scale

