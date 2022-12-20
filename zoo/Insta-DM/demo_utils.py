'''
Seokju Lee

'''

import torch
from torch import nn
import torch.nn.functional as F
from rigid_warp import pixel2cam, inverse_warp2, depth2flow, flow_warp, transform_scale_consistent_depth
from flow_reversal import FlowReversal
import numpy as np
from matplotlib import pyplot as plt
import pdb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_obj_translation(r2t_obj_depths, t2r_obj_depths, tgt_obj_depths, ref_obj_depths, num_insts, intrinsics):
    trans_fwd, trans_bwd = [], []

    for r2t_obj_depth, t2r_obj_depth, tgt_obj_depth, ref_obj_depth, num_inst in zip(r2t_obj_depths, t2r_obj_depths, tgt_obj_depths, ref_obj_depths, num_insts):

        if sum(num_inst) == 0: 
            continue;

        Ks = []
        for bb, ni in enumerate(num_inst):
            if ni == 0: continue;
            Ks.append( intrinsics[bb].unsqueeze(0).repeat(ni,1,1) )
        Ks = torch.cat(Ks, dim=0)

        r2t_obj_coords = pixel2cam(r2t_obj_depth[:,0].detach(), Ks.inverse())
        t2r_obj_coords = pixel2cam(t2r_obj_depth[:,0].detach(), Ks.inverse())
        tgt_obj_coords = pixel2cam(tgt_obj_depth[:,0].detach(), Ks.inverse())
        ref_obj_coords = pixel2cam(ref_obj_depth[:,0].detach(), Ks.inverse())

        tr_fwd, tr_bwd = [], []

        for ii in range( sum(num_inst) ) :
            r2t_obj_coord_mean = torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in r2t_obj_coords[ii]])
            t2r_obj_coord_mean = torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in t2r_obj_coords[ii]])
            tgt_obj_coord_mean = torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in tgt_obj_coords[ii]])
            ref_obj_coord_mean = torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in ref_obj_coords[ii]])
            tr_fwd.append( (((r2t_obj_coord_mean-tgt_obj_coord_mean) + (ref_obj_coord_mean-t2r_obj_coord_mean)) / 2).unsqueeze(0) )
            tr_bwd.append( (((t2r_obj_coord_mean-ref_obj_coord_mean) + (tgt_obj_coord_mean-r2t_obj_coord_mean)) / 2).unsqueeze(0) )

        tr_fwd = torch.cat(tr_fwd, dim=0)
        tr_bwd = torch.cat(tr_bwd, dim=0)

        trans_fwd.append(tr_fwd)
        trans_bwd.append(tr_bwd)


    return trans_fwd, trans_bwd



def compute_batch_bg_warping(tgt_img, ref_imgs, tgt_bg_masks, ref_bg_masks, tgt_depth, ref_depths, poses, poses_inv, intrinsics):
    outputs = []
    for ref_img, ref_depth, pose, pose_inv, tgt_bg_mask, ref_bg_mask in zip(ref_imgs, ref_depths, poses, poses_inv, tgt_bg_masks, ref_bg_masks):
        '''
            tgt_img:     ([B, 3, 256, 832])
            ref_img:     ([B, 3, 256, 832])
            tgt_depth:   ([B, 1, 256, 832])
            ref_depth:   ([B, 1, 256, 832])
            pose:        ([B, 6])
            pose_inv:    ([B, 6])
            tgt_bg_mask: ([B, 1, 256, 832])
            ref_bg_mask: ([B, 1, 256, 832])

            bb = 0
            plt.close('all')
            tgt = (tgt_img * 0.5 + 0.5)
            ref = (ref_img * 0.5 + 0.5)
            ea1 = 6; ea2 = 1; ii = 1;
            fig = plt.figure(1, figsize=(9, 13))
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt.detach().cpu().numpy()[bb].transpose(1,2,0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_bg_mask.detach().cpu().numpy()[bb,0]), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref.detach().cpu().numpy()[bb].transpose(1,2,0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_bg_mask.detach().cpu().numpy()[bb,0]), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.tight_layout(), plt.ion(), plt.show()
            
        '''
        ### Outputs:  warped-masked-bg-img,  valid-bg-mask,  valid-bg-proj-depth,  valid-bg-comp-depth ###
        fwd_outputs = compute_bg_warping(tgt_img, ref_img, tgt_bg_mask, tgt_depth, ref_depth, pose, pose_inv, intrinsics)
        bwd_outputs = compute_bg_warping(ref_img, tgt_img, ref_bg_mask, ref_depth, tgt_depth, pose_inv, pose, intrinsics)
        
        outputs.append( [torch.cat([fwd, bwd], dim=0) for fwd, bwd in zip(fwd_outputs, bwd_outputs)] )

    return outputs



def compute_bg_warping(tgt_img, ref_img, bg_mask, tgt_depth, ref_depth, pose, pose_inv, intrinsic):
    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, pose, intrinsic, ref_depth)
    valid_mask = valid_mask * bg_mask

    return ref_img_warped * valid_mask, valid_mask, projected_depth * valid_mask, computed_depth * valid_mask



def compute_batch_obj_warping(tgt_img, ref_imgs, tgt_obj_masks, ref_obj_masks, tgt_depth, ref_depths, ego_poses, ego_poses_inv, obj_poses, obj_poses_inv, intrinsics, num_insts):
    outputs, ovl_obj = [], []

    for ref_img, ref_depth, tgt_obj_mask, ref_obj_mask, ego_pose, ego_pose_inv, obj_pose, obj_pose_inv, num_inst in \
        zip(ref_imgs, ref_depths, tgt_obj_masks, ref_obj_masks, ego_poses, ego_poses_inv, obj_poses, obj_poses_inv, num_insts):
        # pdb.set_trace()

        tgt_Is, ref_Is, tgt_Ds, ref_Ds, ego_Ps, ego_Ps_inv, Ks = [], [], [], [], [], [], []
        for bb, ni in enumerate(num_inst):
            if ni == 0: continue;
            tgt_Is.append( tgt_img[bb].unsqueeze(0).repeat(ni,1,1,1) )
            ref_Is.append( ref_img[bb].unsqueeze(0).repeat(ni,1,1,1) )
            tgt_Ds.append( tgt_depth[bb].unsqueeze(0).repeat(ni,1,1,1) )
            ref_Ds.append( ref_depth[bb].unsqueeze(0).repeat(ni,1,1,1) )
            ego_Ps.append( ego_pose[bb].unsqueeze(0).repeat(ni,1) )
            ego_Ps_inv.append( ego_pose_inv[bb].unsqueeze(0).repeat(ni,1) )
            Ks.append( intrinsics[bb].unsqueeze(0).repeat(ni,1,1) )
        tgt_Is = torch.cat(tgt_Is, dim=0)
        ref_Is = torch.cat(ref_Is, dim=0)
        tgt_Ds = torch.cat(tgt_Ds, dim=0)
        ref_Ds = torch.cat(ref_Ds, dim=0)
        ego_Ps = torch.cat(ego_Ps, dim=0)
        ego_Ps_inv = torch.cat(ego_Ps_inv, dim=0)
        Ks = torch.cat(Ks, dim=0)

        # (rtt_Is, rtt_Ms, prj_Ds, cmp_Ds), ovl_obj
        fwd_outputs, fwd_ovl_obj = compute_obj_warping(ref_Is, ref_obj_mask, tgt_obj_mask, ref_Ds, tgt_Ds, ego_Ps, obj_pose, Ks, num_inst)
        
        bwd_outputs, bwd_ovl_obj = compute_obj_warping(tgt_Is, tgt_obj_mask, ref_obj_mask, tgt_Ds, ref_Ds, ego_Ps_inv, obj_pose_inv, Ks, num_inst)

        outputs.append( [torch.cat([fwd, bwd], dim=0) for fwd, bwd in zip(fwd_outputs, bwd_outputs)] )
        ovl_obj.append( torch.cat([fwd_ovl_obj, bwd_ovl_obj], dim=0) )

    return outputs, ovl_obj



def compute_obj_warping(ref_img, ref_obj_mask, tgt_obj_mask, ref_depth, tgt_depth, ego_pose, obj_pose, intrinsic, num_inst):
    rtt_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, [ego_pose, obj_pose], intrinsic, ref_depth)

    obj_valid_mask = valid_mask * tgt_obj_mask

    rtt_Is, rtt_Ms, prj_Ds, cmp_Ds, ovl_Ms = [], [], [], [], []
    for bb, ni in enumerate(num_inst):
        if ni == 0:
            rtt_Is.append(torch.zeros(1, rtt_img_warped.size(1), rtt_img_warped.size(2), rtt_img_warped.size(3)).cuda())
            rtt_Ms.append(torch.zeros(1, obj_valid_mask.size(1), obj_valid_mask.size(2), obj_valid_mask.size(3)).cuda())
            prj_Ds.append(torch.zeros(1, projected_depth.size(1), projected_depth.size(2), projected_depth.size(3)).cuda())
            cmp_Ds.append(torch.zeros(1, computed_depth.size(1), computed_depth.size(2), computed_depth.size(3)).cuda())
            ovl_Ms.append(torch.zeros(1, obj_valid_mask.size(1), obj_valid_mask.size(2), obj_valid_mask.size(3)).cuda())
            continue;
        rtt_Is.append( (rtt_img_warped*obj_valid_mask)[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True) )
        rtt_Ms.append( obj_valid_mask[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True) )
        prj_Ds.append( (projected_depth*obj_valid_mask)[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True) )
        cmp_Ds.append( (computed_depth*obj_valid_mask)[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True) )
        # ovl_Ms.append( ( 1 - (1-r2t_obj_mask) * (1-tgt_obj_mask) * (1-ref_obj_mask) )[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True).clamp(0,1) )   # ref + r2t + tgt 모두 합해서 마스킹
        ovl_Ms.append( ( 1 - (1-tgt_obj_mask) * (1-ref_obj_mask) )[int(sum(num_inst[:bb])):int(sum(num_inst[:bb])+ni)].sum(dim=0, keepdim=True).clamp(0,1) )   # ref + tgt 모두 합해서 마스킹
    rtt_Is = torch.cat(rtt_Is, dim=0)
    rtt_Ms = torch.cat(rtt_Ms, dim=0)
    prj_Ds = torch.cat(prj_Ds, dim=0)
    cmp_Ds = torch.cat(cmp_Ds, dim=0)
    ovl_Ms = torch.cat(ovl_Ms, dim=0)

    return (rtt_Is, rtt_Ms, prj_Ds, cmp_Ds), ovl_Ms



def compute_reverse_warp_ego(depths, obj_imgs, obj_masks, ego_poses, intrinsics, num_insts):
    '''
        (args)
        depths:     NumSeqs(2) >> B1HW
        ego_poses:  NumSeqs(2) >> B6
        intrinsics: B33

        bb = 0
        plt.close('all')
        aaa = 1/depths[0].detach().cpu()[bb,0]
        bbb = d2f.detach().cpu()[bb,0]
        ccc = rev_d2f.detach().cpu()[bb,0]
        ddd = r_valid.detach().cpu()[bb,0]
        eee = w_valid.detach().cpu()[bb,0]
        fff = v_mask.detach().cpu()[bb,0]
        ggg = 1/w_depth.detach().cpu()[bb,0]
        hhh = 1/w_sc_depth.detach().cpu()[bb,0]
        ea1 = 8; ea2 = 1; ii = 1;
        fig = plt.figure(1, figsize=(7, 13))
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(aaa), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(bbb), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ccc), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ddd), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(eee), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(fff), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ggg), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(hhh), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.tight_layout(), plt.ion(), plt.show()
        
    '''
    flow_reversal_layer = FlowReversal()
    w_depths, w_sc_depths, v_masks, r_flows = [], [], [], []
    w_obj_imgs, w_obj_masks, w_obj_depths, w_obj_sc_depths = [], [], [], []

    for depth, obj_img, obj_mask, ego_pose, num_inst in zip(depths, obj_imgs, obj_masks, ego_poses, num_insts):
        ### 1st step: batch-wise  ###
        d2f, _ = depth2flow(depth, ego_pose, intrinsics)
        rev_d2f, norm = flow_reversal_layer(d2f, d2f)               # torch.Size([4, 2, 256, 832]), torch.Size([4, 2, 256, 832])
        rev_d2f = -rev_d2f
        rev_d2f[norm > 0] = rev_d2f[norm>0]/norm[norm>0].clone()
        r_valid = (norm != 0).float().prod(dim=1, keepdim=True)     # torch.Size([4, 1, 256, 832])
        rev_d2f = rev_d2f * r_valid                                 # torch.Size([4, 2, 256, 832])
        w_depth, w_valid = flow_warp(depth, rev_d2f)                # torch.Size([4, 1, 256, 832]), torch.Size([4, 1, 256, 832])
        v_mask = (w_valid * r_valid).detach()                       # torch.Size([4, 1, 256, 832])
        w_depth = w_depth * v_mask                                  # torch.Size([4, 1, 256, 832])
        w_sc_depth = transform_scale_consistent_depth(w_depth, ego_pose, intrinsics)  * v_mask    # torch.Size([4, 1, 256, 832])
        
        ### 1st step outputs ###
        w_depths.append( w_depth )          # NumRefs(2) >> torch.Size([4, 1, 256, 832])
        w_sc_depths.append( w_sc_depth )    # NumRefs(2) >> torch.Size([4, 1, 256, 832])
        v_masks.append( v_mask )            # NumRefs(2) >> torch.Size([4, 1, 256, 832])
        r_flows.append( rev_d2f )           # NumRefs(2) >> torch.Size([4, 2, 256, 832])
        # pdb.set_trace()

        ### 2nd step: instance-wise ###
        Vs, Fs, Ds, Ts = [], [], [], []
        for bb, ni in enumerate(num_inst):
            if ni == 0: continue;
            Vs.append(v_mask[bb].unsqueeze(0).repeat(ni,1,1,1))
            Fs.append(rev_d2f[bb].unsqueeze(0).repeat(ni,1,1,1))
            Ds.append(w_depth[bb].unsqueeze(0).repeat(ni,1,1,1))
            Ts.append(w_sc_depth[bb].unsqueeze(0).repeat(ni,1,1,1))
        Vs = torch.cat(Vs, dim=0)
        Fs = torch.cat(Fs, dim=0)
        Ds = torch.cat(Ds, dim=0)
        Ts = torch.cat(Ts, dim=0)

        w_obj_img, _ = flow_warp(obj_img, Fs)
        w_obj_mask, _ = flow_warp(obj_mask, Fs)

        ### 2nd step outputs ###
        w_obj_imgs.append( w_obj_img * w_obj_mask.round() * Vs )    # NumRefs(2) >> torch.Size([12, 3, 256, 832])
        w_obj_masks.append( w_obj_mask.round() * Vs )               # NumRefs(2) >> torch.Size([12, 1, 256, 832])
        w_obj_depths.append( Ds * w_obj_mask.round() * Vs)          # NumRefs(2) >> torch.Size([12, 1, 256, 832])
        w_obj_sc_depths.append( Ts * w_obj_mask.round() * Vs)       # NumRefs(2) >> torch.Size([12, 1, 256, 832])
        # pdb.set_trace()

    return w_depths, w_sc_depths, v_masks, r_flows,    w_obj_imgs, w_obj_masks, w_obj_depths, w_obj_sc_depths



def compute_reverse_warp_obj(depths, obj_imgs, obj_masks, obj_poses, intrinsics, num_insts):
    '''
        (args)
        depths:     NumSeqs(2) >> N1HW
        obj_poses:  NumSeqs(2) >> N6
        intrinsics: B33

        bb = 0
        plt.close('all')
        aaa = depths[0].detach().cpu()[bb,0]
        bbb = d2f.detach().cpu()[bb,0]
        ccc = rev_d2f.detach().cpu()[bb,0]
        ddd = r_valid.detach().cpu()[bb,0]
        eee = w_valid.detach().cpu()[bb,0]
        fff = v_mask.detach().cpu()[bb,0]
        ggg = w_depth.detach().cpu()[bb,0]
        hhh = w_sc_depth.detach().cpu()[bb,0]
        iii = obj_img.detach().cpu()[bb,0]
        jjj = w_obj_img.detach().cpu()[bb,0]
        kkk = (w_obj_mask*v_mask).detach().cpu()[bb,0]
        lll = (w_obj_img*w_obj_mask*v_mask).detach().cpu()[bb,0]
        ea1 = 12; ea2 = 1; ii = 1;
        fig = plt.figure(1, figsize=(7, 13))
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(aaa), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(bbb), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ccc), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ddd), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(eee), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(fff), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ggg), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(hhh), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(iii), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(jjj), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(kkk), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(lll), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar();
        fig.tight_layout(), plt.ion(), plt.show()
        
    '''
    flow_reversal_layer = FlowReversal()
    w_obj_imgs, w_obj_masks, w_obj_depths, w_obj_sc_depths = [], [], [], []

    for depth, obj_img, obj_mask, obj_pose, num_inst in zip(depths, obj_imgs, obj_masks, obj_poses, num_insts):
        ### 1st step: instance-wise  ###
        d2f, _ = depth2flow(depth, obj_pose, intrinsics)
        rev_d2f, norm = flow_reversal_layer(d2f, d2f)               # torch.Size([12, 2, 256, 832]), torch.Size([12, 2, 256, 832])
        rev_d2f = -rev_d2f
        rev_d2f[norm > 0] = rev_d2f[norm>0]/norm[norm>0].clone()
        r_valid = (norm != 0).float().prod(dim=1, keepdim=True)     # torch.Size([12, 1, 256, 832])
        rev_d2f = rev_d2f * r_valid                                 # torch.Size([12, 2, 256, 832])
        
        w_depth, w_valid = flow_warp(depth, rev_d2f)                # torch.Size([12, 1, 256, 832]), torch.Size([12, 1, 256, 832])
        v_mask = (w_valid * r_valid).detach()                       # torch.Size([12, 1, 256, 832])
        w_depth = w_depth * v_mask                                  # torch.Size([12, 1, 256, 832])
        w_sc_depth = transform_scale_consistent_depth(w_depth, obj_pose, intrinsics)  * v_mask    # torch.Size([12, 1, 256, 832])
        
        w_obj_img, _ = flow_warp(obj_img, rev_d2f)
        w_obj_mask, _ = flow_warp(obj_mask, rev_d2f)
        
        ### outputs ###
        w_obj_imgs.append( w_obj_img * w_obj_mask.round() * v_mask )        # NumRefs(2) >> torch.Size([12, 3, 256, 832])
        w_obj_masks.append( w_obj_mask.round() * v_mask )                   # NumRefs(2) >> torch.Size([12, 1, 256, 832])
        w_obj_depths.append( w_depth * w_obj_mask.round() * v_mask)         # NumRefs(2) >> torch.Size([12, 1, 256, 832])
        w_obj_sc_depths.append( w_sc_depth * w_obj_mask.round() * v_mask)   # NumRefs(2) >> torch.Size([12, 1, 256, 832])

    return w_obj_imgs, w_obj_masks, w_obj_depths, w_obj_sc_depths
