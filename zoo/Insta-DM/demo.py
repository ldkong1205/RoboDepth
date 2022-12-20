'''
Seokju Lee
PyTorch version 1.4.0, 1.7.0 confirmed

RUN SCRIPT:
./scripts/run_demo.sh

'''

import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import time
import csv
from path import Path
import datetime
import os
import numpy as np
from imageio import imread
from scipy import stats
import itertools

import torch
import torch.backends.cudnn as cudnn

import models
import custom_transforms
from flow_io import flow_read
from demo_utils import compute_batch_bg_warping, compute_batch_obj_warping, compute_obj_translation, compute_reverse_warp_ego, compute_reverse_warp_obj
from rigid_warp import pixel2cam, cam2homo, pose_vec2mat, flow_warp, inverse_warp2
import drawRobotics as dR
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pdb



parser = argparse.ArgumentParser(description='Instance-wise Depth and Motion Learning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', help='path to dataset dir')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispresnet model')
parser.add_argument('--pretrained-ego-pose', dest='pretrained_ego_pose', default=None, metavar='PATH', help='path to pre-trained Ego Pose net model')
parser.add_argument('--pretrained-obj-pose', dest='pretrained_obj_pose', default=None, metavar='PATH', help='path to pre-trained Obj Pose net model')
parser.add_argument('--mni', default=3, type=int, help='maximum number of instances')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--save-fig', action='store_true', help='save figures or not')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SequenceFolder():
    def __init__(self, data_dir, transform, max_num_instances):
        self.transform = transform
        self.max_num_instances = max_num_instances
        
        img_dir = Path(data_dir)
        seg_dir = Path(os.path.join(img_dir.dirname().parent, 'segmentation', img_dir.basename()))
        flo_dir = [Path(os.path.join(img_dir.dirname().parent, 'flow_f', img_dir.basename())), Path(os.path.join(img_dir.dirname().parent, 'flow_b', img_dir.basename()))]

        intrinsics = np.genfromtxt(img_dir/'cam.txt').astype(np.float32).reshape((3, 3))
        imgs = sorted(img_dir.files('*.jpg'))
        flof = sorted(flo_dir[0].files('*.flo'))   # 00: src, 01: tgt
        flob = sorted(flo_dir[1].files('*.flo'))   # 00: tgt, 01: src
        segm = sorted(seg_dir.files('*.npy')) 
        
        sequence_set = []
        for i in range(len(imgs)-1):
            sample = {'intrinsics':intrinsics, 'img0':imgs[i], 'img1':imgs[i+1], 
                      'flof':flof[i], 'flob':flob[i], 'seg0':segm[i], 'seg1':segm[i+1]}   # will be processed when getitem() is called
            sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        img0 = load_as_float(sample['img0'])
        img1 = load_as_float(sample['img1'])

        flof = torch.from_numpy(load_flo_as_float(sample['flof'])).unsqueeze(0)
        flob = torch.from_numpy(load_flo_as_float(sample['flob'])).unsqueeze(0)

        seg0 = torch.from_numpy(load_seg_as_float(sample['seg0']))
        seg1 = torch.from_numpy(load_seg_as_float(sample['seg1']))

        seg0 = seg0[torch.cat([torch.zeros(1).long(), seg0.sum(dim=(1,2)).argsort(descending=True)[:-1]], dim=0)].unsqueeze(0)
        seg1 = seg1[torch.cat([torch.zeros(1).long(), seg1.sum(dim=(1,2)).argsort(descending=True)[:-1]], dim=0)].unsqueeze(0)

        insts0, insts1 = [], []
        fwd_flows, bwd_flows = [], []

        noc_f, noc_b = find_noc_masks(flof, flob)
        seg0w, _ = flow_warp(seg1, flof)
        seg1w, _ = flow_warp(seg0, flob)

        n_inst0 = seg0.shape[1]
        n_inst1 = seg1.shape[1]

        ### Warp seg0 to seg1. Find IoU between seg1w and seg1. Find the maximum corresponded instance in seg1.
        iou_01, ch_01 = inst_iou(seg1w, seg1, valid_mask=noc_b)
        iou_10, ch_10 = inst_iou(seg0w, seg0, valid_mask=noc_f)

        seg0_re = torch.zeros(self.max_num_instances+1, seg0.shape[2], seg0.shape[3])
        seg1_re = torch.zeros(self.max_num_instances+1, seg1.shape[2], seg1.shape[3])
        non_overlap_0 = torch.ones([seg0.shape[2], seg0.shape[3]])
        non_overlap_1 = torch.ones([seg0.shape[2], seg0.shape[3]])

        num_match = 0
        for ch in range(n_inst0):
            condition1 = (ch == ch_10[ch_01[ch]]) and (iou_01[ch] > 0.5) and (iou_10[ch_01[ch]] > 0.5)
            condition2 = ((seg0[0,ch] * non_overlap_0).max() > 0) and ((seg1[0,ch_01[ch]] * non_overlap_1).max() > 0)
            if condition1 and condition2 and (num_match < self.max_num_instances): # matching success!
                num_match += 1
                seg0_re[num_match] = seg0[0,ch] * non_overlap_0
                seg1_re[num_match] = seg1[0,ch_01[ch]] * non_overlap_1
                non_overlap_0 = non_overlap_0 * (1 - seg0_re[num_match])
                non_overlap_1 = non_overlap_1 * (1 - seg1_re[num_match])
        seg0_re[0] = num_match
        seg1_re[0] = num_match

        insts0.append(seg0_re.detach().cpu().numpy().transpose(1,2,0))
        insts1.append(seg1_re.detach().cpu().numpy().transpose(1,2,0))
        fwd_flows.append(flof[0].detach().cpu().numpy().transpose(1,2,0))
        bwd_flows.append(flob[0].detach().cpu().numpy().transpose(1,2,0))

        imgs, segs, intrinsics = self.transform([img0, img1], insts0 + insts1, np.copy(sample['intrinsics']))

        img0 = imgs[0]
        img1 = imgs[1]
        seg0 = segs[0]
        seg1 = segs[1]

        seg0, seg1 = recursive_check_nonzero_inst(seg0, seg1)

        return img0, img1, seg0, seg1, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)



def main():
    print('=> PyTorch version: ' + torch.__version__ + ' || CUDA_VISIBLE_DEVICES: ' + os.environ["CUDA_VISIBLE_DEVICES"])

    global device
    args = parser.parse_args()
    if args.save_fig:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        args.save_path = 'outputs'/Path(args.name)/timestamp
        print('=> will save everything to {}'.format(args.save_path))
        args.save_path.makedirs_p()

    print("=> fetching scenes in '{}'".format(args.data))
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    demo_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    demo_set = SequenceFolder(
        args.data,
        transform=demo_transform,
        max_num_instances=args.mni
    )
    print('=> {} samples found'.format(len(demo_set)))
    demo_loader = torch.utils.data.DataLoader(demo_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet().to(device)
    ego_pose_net = models.EgoPoseNet().to(device)
    obj_pose_net = models.ObjPoseNet().to(device)

    if args.pretrained_ego_pose:
        print("=> using pre-trained weights for EgoPoseNet")
        weights = torch.load(args.pretrained_ego_pose)
        ego_pose_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        ego_pose_net.init_weights()

    if args.pretrained_obj_pose:
        print("=> using pre-trained weights for ObjPoseNet")
        weights = torch.load(args.pretrained_obj_pose)
        obj_pose_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        obj_pose_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    ego_pose_net = torch.nn.DataParallel(ego_pose_net)
    obj_pose_net = torch.nn.DataParallel(obj_pose_net)

    demo_visualize(args, demo_loader, disp_net, ego_pose_net, obj_pose_net)



@torch.no_grad()
def demo_visualize(args, demo_loader, disp_net, ego_pose_net, obj_pose_net):
    global device
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)
    # np.set_printoptions(formatter={'all':lambda x: str(x)})

    # switch to eval mode
    disp_net.eval().to(device)
    ego_pose_net.eval().to(device)
    obj_pose_net.eval().to(device)

    ego_global_mat = np.identity(4)
    ego_global_mats = [ego_global_mat]
    
    objOs, objHs, objXs, objYs, objZs = [], [], [], [], []
    objIDs = []
    colors = ['yellow', 'lightskyblue', 'lime', 'magenta', 'orange', 'coral', 'gold', 'cyan']

    vidx = 0

    for i, (ref_img, tgt_img, ref_seg, tgt_seg, intrinsics, intrinsics_inv) in enumerate(demo_loader):

        ref_img = ref_img.to(device)
        tgt_img = tgt_img.to(device)
        ref_seg = ref_seg.to(device)
        tgt_seg = tgt_seg.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # input instance masking
        ref_bg_mask = 1 - (ref_seg[:,1:].sum(dim=1, keepdim=True)>0).float()
        tgt_bg_mask = 1 - (tgt_seg[:,1:].sum(dim=1, keepdim=True)>0).float()
        ref_bg_img = ref_img * ref_bg_mask * tgt_bg_mask
        tgt_bg_img = tgt_img * ref_bg_mask * tgt_bg_mask
        num_inst = int(ref_seg[:,0,0,0])
        num_insts = [[num_inst], [num_inst]]

        # tracking info
        if len(objIDs) == 0: 
            objIDs.append( np.arange(num_inst).tolist() )
        else:
            # -> ref_seg의 인스턴스들이 tgt_seg_prev의 몇 번째 채널 인스턴스에 매칭되는가?
            p2c_iou, p2c_idx = inst_iou(ref_seg.cpu(), tgt_seg_prev.cpu(), torch.ones(1,1,ref_seg.size(2),ref_seg.size(3)).type_as(ref_seg).cpu())
            p2c_iou = p2c_iou[1:]
            p2c_idx = p2c_idx[1:] - 1
            newColorID = list(set(np.arange(len(colors))) - set(objIDs[-1]))
            newID = []

            for ii, iou in enumerate(p2c_iou):
                if iou > 0.5:
                    newID.append( objIDs[-1][int(p2c_idx[ii])] )
                elif iou != iou:
                    break;
                else:
                    newID.append( newColorID[0] )
                    newColorID = newColorID[1:]
            objIDs.append( newID )

        
        tgt_seg_prev = tgt_seg.clone()
        tgt_seg_prev[0,0] = 0
        objIDs_flatten = list(itertools.chain.from_iterable(objIDs))
        # pdb.set_trace()
        '''
            # plt.close('all')
            ea1 = 4; ea2 = 5; ii = 1;
            fig = plt.figure(99, figsize=(20, 10))
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_seg_prev[0,0].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_seg_prev[0,1].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_seg_prev[0,2].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_seg_prev[0,3].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_seg_prev[0,4].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_seg[0,0].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_seg[0,1].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_seg[0,2].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_seg[0,3].cpu()); plt.colorbar(); 
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_seg[0,4].cpu()); plt.colorbar(); 
            plt.tight_layout(); plt.ion(); plt.show()

        '''

        # compute depth & camera motion
        ref_depth = 1 / disp_net(ref_img)
        tgt_depth = 1 / disp_net(tgt_img)
        ego_pose = ego_pose_net(tgt_bg_img, ref_bg_img)
        ego_pose_inv = ego_pose_net(ref_bg_img, tgt_bg_img)
        # ego_pose = ego_pose_net(tgt_img, ref_img)
        # ego_pose_inv = ego_pose_net(ref_img, tgt_img)

        ego_mat = pose_vec2mat(ego_pose).squeeze(0).cpu().detach().numpy()
        ego_mat = np.concatenate([ego_mat, np.array([0, 0, 0, 1]).reshape(1,4)], axis=0)
        ego_global_mat = ego_global_mat @ ego_mat
        ego_global_mats.append(ego_global_mat)

        ### Batch-wise computing ###    rtt_Is, rtt_Ms, prj_Ds, cmp_Ds  --> (19.11.18) change from fw to iw for bg regions
        ### Outputs:  warped-masked-bg-img,  valid-bg-mask,  valid-bg-proj-depth,  valid-bg-comp-depth ###
        ### NumScales(1) >> NumRefs(2) >> I-M-D-D(4) >> 2B(fwd/bwd)xCxHxW ###
        IMDDs = compute_batch_bg_warping(tgt_img, [ref_img, ref_img], [tgt_bg_mask, tgt_bg_mask], [ref_bg_mask, ref_bg_mask], 
                                         tgt_depth, [ref_depth, ref_depth], [ego_pose, ego_pose], [ego_pose_inv, ego_pose_inv], intrinsics)


        ref_obj_img = ref_img.repeat(num_inst,1,1,1) * ref_seg[0,1:1+num_inst].unsqueeze(1)
        tgt_obj_img = tgt_img.repeat(num_inst,1,1,1) * tgt_seg[0,1:1+num_inst].unsqueeze(1)
        ref_obj_mask = ref_seg[0,1:1+num_inst].unsqueeze(1)
        tgt_obj_mask = tgt_seg[0,1:1+num_inst].unsqueeze(1)
        ref_obj_depth = ref_depth.repeat(num_inst,1,1,1) * ref_seg[0,1:1+num_inst].unsqueeze(1)
        tgt_obj_depth = tgt_depth.repeat(num_inst,1,1,1) * tgt_seg[0,1:1+num_inst].unsqueeze(1)

        _, _, _, _,    r2t_obj_imgs, r2t_obj_masks, _, r2t_obj_sc_depths = \
            compute_reverse_warp_ego([ref_depth, ref_depth], [ref_obj_img, ref_obj_img], [ref_obj_mask, ref_obj_mask], [ego_pose_inv, ego_pose_inv], intrinsics, num_insts)
        _, _, _, _,    t2r_obj_imgs, t2r_obj_masks, _, t2r_obj_sc_depths = \
            compute_reverse_warp_ego([tgt_depth, tgt_depth], [tgt_obj_img, tgt_obj_img], [tgt_obj_mask, tgt_obj_mask], [ego_pose, ego_pose], intrinsics, num_insts)

        obj_pose = obj_pose_net(tgt_obj_img, r2t_obj_imgs[0])
        obj_pose_inv = obj_pose_net(ref_obj_img, t2r_obj_imgs[0])
        obj_pose = torch.cat([obj_pose, torch.zeros_like(obj_pose)], dim=1)
        obj_pose_inv = torch.cat([obj_pose_inv, torch.zeros_like(obj_pose_inv)], dim=1)

        obj_mat = pose_vec2mat(obj_pose).cpu().detach().numpy()
        obj_mat = np.concatenate([obj_mat, np.array([0, 0, 0, 1]).reshape(1,1,4).repeat(obj_pose.size(0),axis=0)], axis=1)
        obj_global_mat = ego_global_mat.reshape(1,4,4).repeat(obj_pose.size(0),axis=0) @ obj_mat

        obj_IMDDs, obj_ovls = compute_batch_obj_warping(tgt_img, [ref_img, ref_img], [tgt_obj_mask, tgt_obj_mask], [ref_obj_mask, ref_obj_mask], tgt_depth, [ref_depth, ref_depth], 
                                                        [ego_pose, ego_pose], [ego_pose_inv, ego_pose_inv], [obj_pose, obj_pose], [obj_pose_inv, obj_pose_inv], intrinsics, num_insts)


        tr_fwd, tr_bwd = compute_obj_translation(r2t_obj_sc_depths, t2r_obj_sc_depths, [tgt_obj_depth, tgt_obj_depth], [ref_obj_depth, ref_obj_depth], num_insts, intrinsics)
        
        rtt_obj_imgs, rtt_obj_masks, rtt_obj_depths, rtt_obj_sc_depths = compute_reverse_warp_obj(r2t_obj_sc_depths, r2t_obj_imgs, r2t_obj_masks, [-obj_pose, -obj_pose], intrinsics.repeat(num_inst,1,1), num_insts)
        # pdb.set_trace()
        '''
            sq = 0; bb = 0; 
            plt.close('all')
            plt.figure(1); plt.imshow(r2t_obj_sc_depths[sq][bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(2); plt.imshow(r2t_sc_depths[sq][0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(3); plt.imshow(rtt_obj_sc_depths[sq][bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(4); plt.imshow(rtt_obj_sc_depth_2[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(5); plt.imshow(rev_d2f[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(6); plt.imshow(d2f[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(7); plt.imshow(norm[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(8); plt.imshow(r2t_obj_masks[sq][bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(9); plt.imshow(rtt_obj_masks[sq][bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(10); plt.imshow(rtt_obj_imgs[sq][bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()

        '''

        _, _, r2t_ego_projected_depth, r2t_ego_computed_depth = inverse_warp2(ref_img, tgt_depth, ego_pose, intrinsics, ref_depth)


        ### KITTI ###
        if 'kitti' in args.data:
            xlim_1 = 0.25; ylim_1 = 0.1;  zlim_1 = 1.2;
            xlim_2 = 0.25; ylim_2 = 0.1;  zlim_2 = 1.2;
            obj_vo_scale = 3.0
            ego_vo_scale = 0.015

        ### CS ###
        if 'cityscapes' in args.data:
            xlim_1 = 0.1;  ylim_1 = 0.1;  zlim_1 = 0.4;
            xlim_2 = 0.12;  ylim_2 = 0.06;  zlim_2 = 0.60;
            obj_vo_scale = 3.0
            ego_vo_scale = 0.005


        ego_init_o = np.array([0,0,0,1]).reshape(4,1)
        ego_init_x = np.array([ego_vo_scale*1,0,0,1]).reshape(4,1)
        ego_init_y = np.array([0,ego_vo_scale*1,0,1]).reshape(4,1)
        ego_init_z = np.array([0,0,ego_vo_scale*1,1]).reshape(4,1)
        egoOs = np.array([mat @ ego_init_o for mat in ego_global_mats])[:,:3,0]
        egoXs = np.array([mat @ ego_init_x for mat in ego_global_mats])[:,:3,0]
        egoYs = np.array([mat @ ego_init_y for mat in ego_global_mats])[:,:3,0]
        egoZs = np.array([mat @ ego_init_z for mat in ego_global_mats])[:,:3,0]

        bbox_y = dict(boxstyle='round', facecolor='yellow', alpha=0.5)
        bbox_c = dict(boxstyle='round', facecolor='coral', alpha=0.5)
        bbox_m = dict(boxstyle='round', facecolor='magenta', alpha=0.5)
        bbox_l = dict(boxstyle='round', facecolor='lime', alpha=0.5)
        bbox_w = dict(boxstyle='round', facecolor='white', alpha=0.5)
        bbox_b = dict(boxstyle='round', facecolor='deepskyblue', alpha=0.5)
        # pdb.set_trace()

        sq = 0; bb = 0;
        r2t_objs_coords = pixel2cam(r2t_obj_sc_depths[0][:,0], intrinsics.inverse().repeat(num_inst,1,1))
        rtt_objs_coords = pixel2cam(rtt_obj_sc_depths[0][:,0], intrinsics.inverse().repeat(num_inst,1,1))
        tgt_objs_coords = pixel2cam(tgt_obj_depth[:,0], intrinsics.inverse().repeat(num_inst,1,1))
        r2t_obj_3d_locs = []
        rtt_obj_3d_locs = []
        tgt_obj_3d_locs = []
        for r2t_obj_coords in r2t_objs_coords: r2t_obj_3d_locs.append(torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in r2t_obj_coords]))
        for rtt_obj_coords in rtt_objs_coords: rtt_obj_3d_locs.append(torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in rtt_obj_coords]))
        for tgt_obj_coords in tgt_objs_coords: tgt_obj_3d_locs.append(torch.cat([coords[coords!=0].mean().unsqueeze(0) for coords in tgt_obj_coords]))
        for obj_loc in tgt_obj_3d_locs: objOs.append( (ego_global_mat @ np.concatenate([obj_loc.detach().cpu().numpy(), np.array([1])]).reshape(4,1))[:3].squeeze() );
        objHs_pred, objHs_comp = [], []
        for ii in range(len(obj_pose_inv)): objHs_pred.append( (ego_global_mat @ np.concatenate([tgt_obj_3d_locs[ii].detach().cpu().numpy(), np.array([1])]).reshape(4,1))[:3].squeeze() + obj_vo_scale*obj_pose_inv[ii].detach().cpu().numpy()[:3] )
        for ii in range(len(obj_pose_inv)): objHs_comp.append( (ego_global_mat @ np.concatenate([tgt_obj_3d_locs[ii].detach().cpu().numpy(), np.array([1])]).reshape(4,1))[:3].squeeze() - obj_vo_scale*tr_fwd[0][ii].detach().cpu().numpy() )
        for pred, comp in zip(objHs_pred, objHs_comp): objHs.append( (pred + comp) / 2 )
        r2t_obj_3d_loc = torch.stack(r2t_obj_3d_locs).unsqueeze(-1).unsqueeze(-1)
        r2t_obj_homo, _ = cam2homo(r2t_obj_3d_loc, intrinsics.repeat(num_inst,1,1), torch.zeros([1,3,1]).cuda())
        r2t_obj_tail = r2t_obj_homo.reshape(num_inst,2).detach().cpu().numpy()
        r2t_obj_trans = -obj_pose[:,:3]
        r2t_obj_trans_gt = -tr_fwd[0]
        r2t_obj_3d_loc_tr = r2t_obj_3d_loc.reshape(num_inst,3) + r2t_obj_trans
        r2t_obj_3d_loc_tr_gt = r2t_obj_3d_loc.reshape(num_inst,3) + r2t_obj_trans_gt
        r2t_obj_homo_tr, _ = cam2homo(r2t_obj_3d_loc_tr.unsqueeze(-1).unsqueeze(-1), intrinsics.repeat(num_inst,1,1), torch.zeros([1,3,1]).cuda())
        r2t_obj_homo_tr_gt, _ = cam2homo(r2t_obj_3d_loc_tr_gt.unsqueeze(-1).unsqueeze(-1), intrinsics.repeat(num_inst,1,1), torch.zeros([1,3,1]).cuda())
        r2t_obj_head = r2t_obj_homo_tr.reshape(num_inst,2).detach().cpu().numpy()
        r2t_obj_head_gt = r2t_obj_homo_tr_gt.reshape(num_inst,2).detach().cpu().numpy()
        arr_scale = 1.5
        tgt = (tgt_img[bb%args.batch_size]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
        tgt_inst = 1 - tgt_bg_mask[bb].repeat(3,1,1).detach().cpu().numpy().transpose(1,2,0)
        tgt_masked = (tgt + 0.2 * tgt_inst).clip(max=1.0)
        ref = (ref_img[bb%args.batch_size]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
        ref_inst = 1 - ref_bg_mask[bb].repeat(3,1,1).detach().cpu().numpy().transpose(1,2,0)
        ref_masked = (ref + 0.2 * ref_inst).clip(max=1.0)
        d_tgt = 1/tgt_depth.detach().cpu()[bb%args.batch_size,0]
        d_ref = 1/ref_depth.detach().cpu()[bb%args.batch_size,0]
        r2t_obj = (r2t_obj_imgs[0].sum(dim=0) * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0) if num_inst != 0 else np.zeros([256,832,3])
        tgt_obj = (tgt_obj_img.sum(dim=0) * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0) if num_inst != 0 else np.zeros([256,832,3])
        i_w_bg = (IMDDs[sq][0] * 0.5 + 0.5)[bb].detach().cpu().numpy().transpose(1,2,0)
        i_w_obj = (obj_IMDDs[sq][0] * 0.5 + 0.5)[bb].detach().cpu().numpy().transpose(1,2,0)
        i_w = ((IMDDs[sq][0] + obj_IMDDs[sq][0]) * 0.5 + 0.5)[bb].detach().cpu().numpy().transpose(1,2,0)
        m_w = obj_IMDDs[sq][1][0].repeat(3,1,1).detach().cpu().numpy().transpose(1,2,0)
        i_w_masked = i_w + 0.2 * m_w
        d_diff = ( ((IMDDs[sq][3] + obj_IMDDs[sq][3]) - (IMDDs[sq][2] + obj_IMDDs[sq][2])).abs() / ((IMDDs[sq][3] + obj_IMDDs[sq][3]) + (IMDDs[sq][2] + obj_IMDDs[sq][2])).abs().clamp(min=1e-3) ).clamp(0,1)[bb,0].detach().cpu()
        d_diff_ego = ( (r2t_ego_projected_depth - r2t_ego_computed_depth).abs() / (r2t_ego_projected_depth + r2t_ego_computed_depth).abs().clamp(min=1e-3) ).clamp(0,1)[bb,0].detach().cpu() * (IMDDs[sq][1] + obj_IMDDs[sq][1])[bb,0].detach().cpu()
        occ = 1.5 * d_diff.unsqueeze(-1).repeat(1,1,3).numpy()
        occ[:,:,2] = 0
        occ[occ<0.1] = 0
        i_w_occ = (i_w_masked + occ).clip(max=1.0)
        tgt_diff = np.abs(i_w-tgt).mean(axis=2)
        th = 5; samp = 20;
        r2t_obj_coords = r2t_objs_coords.sum(dim=0, keepdim=True)
        rtt_obj_coords = rtt_objs_coords.sum(dim=0, keepdim=True)
        tgt_obj_coords = tgt_objs_coords.sum(dim=0, keepdim=True)
        r2t_filt = np.abs(stats.zscore( r2t_obj_coords[bb,2].view(-1)[r2t_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy() )) < th
        rtt_filt = np.abs(stats.zscore( rtt_obj_coords[bb,2].view(-1)[rtt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy() )) < th
        tgt_filt = np.abs(stats.zscore( tgt_obj_coords[bb,2].view(-1)[tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy() )) < th
        npts_r2t = int(r2t_filt.sum())
        npts_rtt = int(rtt_filt.sum())
        npts_tgt = int(tgt_filt.sum())
        X_r2t = r2t_obj_coords[bb,0].view(-1)[r2t_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[r2t_filt][range(0,npts_r2t,samp)]
        Y_r2t = r2t_obj_coords[bb,1].view(-1)[r2t_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[r2t_filt][range(0,npts_r2t,samp)]
        Z_r2t = r2t_obj_coords[bb,2].view(-1)[r2t_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[r2t_filt][range(0,npts_r2t,samp)]
        C_r2t = r2t_obj_imgs[0].sum(dim=0).view(3,-1)[:,r2t_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[:,r2t_filt][:,range(0,npts_r2t,samp)] * 0.5 + 0.5
        C_r2t[0] = 1; C_r2t[1] = 0; C_r2t[2] = 0;
        X_rtt = rtt_obj_coords[bb,0].view(-1)[rtt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[rtt_filt][range(0,npts_rtt,samp)]
        Y_rtt = rtt_obj_coords[bb,1].view(-1)[rtt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[rtt_filt][range(0,npts_rtt,samp)]
        Z_rtt = rtt_obj_coords[bb,2].view(-1)[rtt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[rtt_filt][range(0,npts_rtt,samp)]
        C_rtt = rtt_obj_imgs[0].sum(dim=0).view(3,-1)[:,rtt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[:,rtt_filt][:,range(0,npts_rtt,samp)] * 0.5 + 0.5
        C_rtt[0] = 1; C_rtt[1] = 1; C_rtt[2] = 0;
        X_tgt = tgt_obj_coords[bb,0].view(-1)[tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[tgt_filt][range(0,npts_tgt,samp)]
        Y_tgt = tgt_obj_coords[bb,1].view(-1)[tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[tgt_filt][range(0,npts_tgt,samp)]
        Z_tgt = tgt_obj_coords[bb,2].view(-1)[tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[tgt_filt][range(0,npts_tgt,samp)]
        C_tgt = tgt_obj_img.sum(dim=0).view(3,-1)[:,tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[:,tgt_filt][:,range(0,npts_tgt,samp)] * 0.5 + 0.5
        C_tgt[0] = 0; C_tgt[1] = 0; C_tgt[2] = 1;
        XYZ_global_tgt = np.expand_dims(ego_global_mat, axis=0).repeat(X_tgt.shape[0],axis=0) @ np.expand_dims(np.stack([X_tgt, Y_tgt, Z_tgt, np.ones([X_tgt.shape[0]])]).transpose(1,0), axis=-1)
        C_global_tgt = (tgt_obj_img.sum(dim=0).view(3,-1)[:,tgt_obj_coords[bb].mean(dim=0).view(-1)!=0].detach().cpu().numpy()[:,tgt_filt][:,range(0,npts_tgt,samp)] * 0.5 + 0.5).clip(min=0.0, max=1.0)

        plt.close('all')
        fig = plt.figure(1, figsize=(1920/100, 1080/100), dpi=100)   # figsize=(23, 13)
        gs = GridSpec(nrows=5, ncols=6)
        text_xy = [7, -16]
        text_fd = {'family': 'sans', 'size': 13, 'color': 'black', 'style': 'italic'}
        fig.add_subplot(gs[0, 0:2])
        plt.imshow(ref_masked, vmax=1); plt.text(text_xy[0], text_xy[1], "$I_{t}$", fontdict=text_fd);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        if not args.save_fig: plt.grid(linestyle=':', linewidth=0.4);
        plt.text(55, -29, "Scene: {}, Iter: {}".format(args.data, i), fontsize=6.5);
        plt.text(55, -9, "Model: {}".format(args.pretrained_disp), fontsize=6.5);
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[0, 2:4])
        plt.imshow(d_ref, cmap='turbo', vmax=14); plt.text(text_xy[0], text_xy[1], "$D_{t}$", fontdict=text_fd);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[1, 0:2])
        plt.imshow(tgt_masked, vmax=1); plt.text(text_xy[0], text_xy[1], "$I_{t+1}$", fontdict=text_fd);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[1, 2:4])
        plt.imshow(d_tgt, cmap='turbo', vmax=14); plt.text(text_xy[0], text_xy[1], "$D_{t+1}$", fontdict=text_fd);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[2, 0:2])
        plt.imshow(r2t_obj, vmax=1); plt.text(text_xy[0], text_xy[1], "Ego-warped objects with motion", fontdict=text_fd, size=10);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4)
        plt.text(130, 250, "*ego speed {:0.4f},  6-DoF {}".format(float(ego_pose[0,:3].pow(2).sum().sqrt()), ego_pose[0].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_b, ha='left', va='bottom');
        if num_inst > 0: plt.text(7, 7,  "Obj-1: {:0.4f} {}".format(float(obj_pose[0,:3].pow(2).sum().sqrt()), obj_pose[0][:3].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_m, ha='left', va='top');
        if num_inst > 0 and not args.save_fig: plt.text(330, 7, "#1: {:0.4f} {}".format(float(tr_fwd[0].pow(2).sum(dim=1).sqrt()[0]), tr_fwd[0][0].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_c, ha='left', va='top');
        if num_inst > 1: plt.text(7, 31, "Obj-2: {:0.4f} {}".format(float(obj_pose[1,:3].pow(2).sum().sqrt()), obj_pose[1][:3].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_m, ha='left', va='top');
        if num_inst > 1 and not args.save_fig: plt.text(330, 31, "#2: {:0.4f} {}".format(float(tr_fwd[0].pow(2).sum(dim=1).sqrt()[1]), tr_fwd[0][1].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_c, ha='left', va='top');
        if num_inst > 2: plt.text(7, 55, "Obj-3: {:0.4f} {}".format(float(obj_pose[2,:3].pow(2).sum().sqrt()), obj_pose[2][:3].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_m, ha='left', va='top');
        if num_inst > 2 and not args.save_fig: plt.text(330, 55, "#3: {:0.4f} {}".format(float(tr_fwd[0].pow(2).sum(dim=1).sqrt()[2]), tr_fwd[0][2].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_c, ha='left', va='top');
        if num_inst > 3: plt.text(7, 79, "Obj-4: {:0.4f} {}".format(float(obj_pose[3,:3].pow(2).sum().sqrt()), obj_pose[3][:3].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_m, ha='left', va='top');
        if num_inst > 3 and not args.save_fig: plt.text(330, 79, "#4: {:0.4f} {}".format(float(tr_fwd[0].pow(2).sum(dim=1).sqrt()[3]), tr_fwd[0][3].detach().cpu().numpy().round(4)), fontsize=7, bbox=bbox_c, ha='left', va='top');
        if num_inst > 0 and not args.save_fig: plt.arrow(r2t_obj_tail[0,0], r2t_obj_tail[0,1], arr_scale*(-r2t_obj_tail[0,0]+r2t_obj_head_gt[0,0]), arr_scale*(-r2t_obj_tail[0,1]+r2t_obj_head_gt[0,1]), width=2, head_width=9, head_length=9, color='red', alpha=1); 
        if num_inst > 0: plt.arrow(r2t_obj_tail[0,0], r2t_obj_tail[0,1], arr_scale*(-r2t_obj_tail[0,0]+r2t_obj_head[0,0]), arr_scale*(-r2t_obj_tail[0,1]+r2t_obj_head[0,1]), width=3, head_width=10, head_length=9, color='magenta', alpha=1); 
        if num_inst > 0: plt.text(r2t_obj_tail[0,0]-30, r2t_obj_tail[0,1]+25, "1: {:0.4f}".format(float(obj_pose[0,:3].pow(2).sum().sqrt())), fontsize=7, bbox=bbox_l);
        if num_inst > 1 and not args.save_fig: plt.arrow(r2t_obj_tail[1,0], r2t_obj_tail[1,1], arr_scale*(-r2t_obj_tail[1,0]+r2t_obj_head_gt[1,0]), arr_scale*(-r2t_obj_tail[1,1]+r2t_obj_head_gt[1,1]), width=2, head_width=9, head_length=9, color='red', alpha=1); 
        if num_inst > 1: plt.arrow(r2t_obj_tail[1,0], r2t_obj_tail[1,1], arr_scale*(-r2t_obj_tail[1,0]+r2t_obj_head[1,0]), arr_scale*(-r2t_obj_tail[1,1]+r2t_obj_head[1,1]), width=3, head_width=10, head_length=9, color='magenta', alpha=1); 
        if num_inst > 1: plt.text(r2t_obj_tail[1,0]-30, r2t_obj_tail[1,1]+25, "2: {:0.4f}".format(float(obj_pose[1,:3].pow(2).sum().sqrt())), fontsize=7, bbox=bbox_l);
        if num_inst > 2 and not args.save_fig: plt.arrow(r2t_obj_tail[2,0], r2t_obj_tail[2,1], arr_scale*(-r2t_obj_tail[2,0]+r2t_obj_head_gt[2,0]), arr_scale*(-r2t_obj_tail[2,1]+r2t_obj_head_gt[2,1]), width=2, head_width=9, head_length=9, color='red', alpha=1); 
        if num_inst > 2: plt.arrow(r2t_obj_tail[2,0], r2t_obj_tail[2,1], arr_scale*(-r2t_obj_tail[2,0]+r2t_obj_head[2,0]), arr_scale*(-r2t_obj_tail[2,1]+r2t_obj_head[2,1]), width=3, head_width=10, head_length=9, color='magenta', alpha=1); 
        if num_inst > 2: plt.text(r2t_obj_tail[2,0]-30, r2t_obj_tail[2,1]+25, "3: {:0.4f}".format(float(obj_pose[2,:3].pow(2).sum().sqrt())), fontsize=7, bbox=bbox_l);
        if num_inst > 3 and not args.save_fig: plt.arrow(r2t_obj_tail[3,0], r2t_obj_tail[3,1], arr_scale*(-r2t_obj_tail[3,0]+r2t_obj_head_gt[3,0]), arr_scale*(-r2t_obj_tail[3,1]+r2t_obj_head_gt[3,1]), width=2, head_width=9, head_length=9, color='red', alpha=1); 
        if num_inst > 3: plt.arrow(r2t_obj_tail[3,0], r2t_obj_tail[3,1], arr_scale*(-r2t_obj_tail[3,0]+r2t_obj_head[3,0]), arr_scale*(-r2t_obj_tail[3,1]+r2t_obj_head[3,1]), width=3, head_width=10, head_length=9, color='magenta', alpha=1); 
        if num_inst > 3: plt.text(r2t_obj_tail[3,0]-30, r2t_obj_tail[3,1]+25, "4: {:0.4f}".format(float(obj_pose[3,:3].pow(2).sum().sqrt())), fontsize=7, bbox=bbox_l);
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[3, 0:2])
        plt.imshow(i_w_occ, vmax=1); plt.text(text_xy[0], text_xy[1], "Final synthesis (yellow: dis/occlusion)", fontdict=text_fd, size=10);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);
        fig.add_subplot(gs[4, 0:2])
        plt.imshow(tgt_diff, cmap='bone', vmax=0.5); plt.text(text_xy[0], text_xy[1], "$I_{diff}$", fontdict=text_fd);
        plt.xticks([]) and plt.yticks([]) if args.save_fig else plt.grid(linestyle=':', linewidth=0.4) 
        plt.xlim(0, 832-1); plt.ylim(256-1, 0);

        ### 3d plot 1: cam-coord ###
        ax1 = fig.add_subplot(gs[3:5, 2:4], projection='3d')
        ax1_axfont = {'family': 'sans', 'size': 12, 'weight': 'heavy', 'style': 'italic', 'color': 'gray'}
        ax1_titlefont = {'family': 'sans', 'size': 12, 'color': 'black', 'ha': 'center', 'va': 'bottom', 'linespacing': 2}
        ax1_annotfont = {'family': 'sans', 'size': 8, 'color': 'black', 'ha': 'center', 'va': 'center'}
        ax1.scatter(X_r2t, Y_r2t, Z_r2t, c=C_r2t.transpose(1,0), s=1, alpha=0.4)
        ax1.scatter(X_rtt, Y_rtt, Z_rtt, c=C_rtt.transpose(1,0), s=1, alpha=0.6)
        ax1.scatter(X_tgt, Y_tgt, Z_tgt, c=C_tgt.transpose(1,0), s=1, alpha=0.4)
        ax1.set_xlabel('X', fontdict=ax1_axfont); ax1.set_zlabel('Z', fontdict=ax1_axfont);
        ax1.axes.yaxis.set_ticklabels([])
        ax1.set_xlim(-xlim_1, xlim_1)
        ax1.set_ylim(-ylim_1, ylim_1)
        ax1.set_zlim(0,       zlim_1)
        ax1.text(0, 0, zlim_1*1.20, "[Top-view] Objects in {$t+1$} frame on camera coordinate\n(red: ego-warped $t$→$t+1$, yellow: final-warped $t$→$t+1$, blue: $t+1$)", fontdict=ax1_titlefont)
        if num_inst > 0: ax1.text(-xlim_1/2, 0, zlim_1*1.10, "1: XYZ {}".format(r2t_obj_3d_locs[0].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_c);
        if num_inst > 0: ax1.text(-xlim_1/2, 0, zlim_1*1.05, "1: XYZ {}".format(rtt_obj_3d_locs[0].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_y);
        if num_inst > 0: ax1.text(-xlim_1/2, 0, zlim_1*1.00, "1: XYZ {}".format(tgt_obj_3d_locs[0].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_b);
        if num_inst > 1: ax1.text(+xlim_1/2, 0, zlim_1*1.10, "2: XYZ {}".format(r2t_obj_3d_locs[1].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_c);
        if num_inst > 1: ax1.text(+xlim_1/2, 0, zlim_1*1.05, "2: XYZ {}".format(rtt_obj_3d_locs[1].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_y);
        if num_inst > 1: ax1.text(+xlim_1/2, 0, zlim_1*1.00, "2: XYZ {}".format(tgt_obj_3d_locs[1].detach().cpu().numpy().round(4)), fontdict=ax1_annotfont, bbox=bbox_b);
        ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([2, 1, 2, 1]))
        ax1.view_init(elev=0, azim=-90)

        ### 3d plot 2: world-coord ###
        ax2 = fig.add_subplot(gs[1:5, 4:6], projection='3d')
        ax2_axfont = {'family': 'sans', 'size': 14, 'weight': 'heavy', 'style': 'italic', 'color': 'gray'}
        ax2_titlefont = {'family': 'sans', 'size': 12, 'color': 'black', 'ha': 'center', 'va': 'center'}
        ax2.scatter(XYZ_global_tgt[:,0,0], XYZ_global_tgt[:,1,0], XYZ_global_tgt[:,2,0], c=C_global_tgt.transpose(1,0), s=6, zorder=vidx+1)
        for ii in range(len(egoOs)-1): dR.drawVector(ax2, egoOs[ii], egoOs[ii+1], mutation_scale=1, alpha=0.5, arrowstyle='-', lineStyle=':', lineWidth=1, lineColor='k', zorder=vidx+2)
        for ii in range(len(egoOs)):
            if ii >= len(egoOs) - 1:
                dR.drawPointWithAxis(ax2, egoOs[ii], egoXs[ii]-egoOs[ii], egoYs[ii]-egoOs[ii], egoZs[ii]-egoOs[ii], mutation_scale=3, alpha=1.0, arrowstyle='-', lineWidth=2.0, vectorLength=2, zorder=vidx+4)
                ax2.text(egoOs[ii,0]-xlim_2/4, egoOs[ii,1], egoOs[ii,2], "{:0.4f}".format( np.linalg.norm(egoOs[ii]-egoOs[ii-1]).round(4) ), fontsize=8, bbox=bbox_w, zorder=vidx+5);
            else:
                dR.drawPointWithAxis(ax2, egoOs[ii], egoXs[ii]-egoOs[ii], egoYs[ii]-egoOs[ii], egoZs[ii]-egoOs[ii], mutation_scale=1, alpha=0.4, arrowstyle='-', lineWidth=1.5, zorder=vidx+3)
        for ii in range(len(objOs)):
            if ii >= len(objOs) - len(obj_pose):
                if np.linalg.norm(objOs[ii]-objHs[ii], 2) < 0.1 and np.linalg.norm(objOs[ii]-objHs[ii], 2) > 0.002:
                    dR.drawVector(ax2, objOs[ii], objHs[ii], mutation_scale=20, arrowstyle='fancy', pointEnable=False, lineWidth=0.5, faceColor=colors[objIDs_flatten[ii]], edgeColor='k', zorder=vidx+20);
                    ax2.text(objHs[ii][0]-xlim_2/4, objHs[ii][1], objHs[ii][2], "{:0.4f}".format( (np.linalg.norm(objOs[ii]-objHs[ii])/obj_vo_scale).round(4) ), fontsize=8, bbox=bbox_w, zorder=vidx+30);
            else:
                if np.linalg.norm(objOs[ii]-objHs[ii], 2) < 0.1 and np.linalg.norm(objOs[ii]-objHs[ii], 2) > 0.002:
                    dR.drawVector(ax2, objOs[ii], objHs[ii], mutation_scale=20, alpha=0.3, arrowstyle='fancy', pointEnable=False, lineWidth=0.5, faceColor=colors[objIDs_flatten[ii]], edgeColor='k', zorder=vidx+10);
        ax2.text(-xlim_2*0.9, 0, zlim_2*0.02, "Speed", fontsize=9, style='italic', bbox=bbox_w, zorder=vidx+25);
        ax2.set_xlabel('X', fontdict=ax2_axfont); ax2.set_zlabel('Z', fontdict=ax2_axfont);
        ax2.axes.yaxis.set_ticklabels([])
        ax2.set_xlim(-xlim_2, xlim_2)
        ax2.set_ylim(-ylim_2, ylim_2)
        ax2.set_zlim(0,       zlim_2)
        ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([1.2, 0.6, 2.4, 1]))
        ax2.xaxis._axinfo['juggled'] = (2,0,1)
        ax2.yaxis._axinfo['juggled'] = (2,1,0)
        ax2.zaxis._axinfo['juggled'] = (0,2,1)
        elv = 2; azm = 1;
        if 'cityscapes' in args.data:
            ax2.view_init(elev=-0.01-elv*vidx, azim=-90+azm*vidx)
            # ax2.view_init(elev=-0.01-elv*vidx, azim=-90.01-azm*vidx)
        else:
            if vidx <= 10: ax2.view_init(elev=-0.01-elv*vidx, azim=-90+azm*vidx)                                                # elev: -0~-40, azim: -90~-70
            if 10 < vidx and vidx <= 20: ax2.view_init(elev= -0.01-elv*10 + elv*(vidx-10), azim= -90+azm*10 - azm*(vidx-10))    # elev: -40~-0, azim: -70~-90
            if 20 < vidx and vidx <= 30: ax2.view_init(elev= -0.01 - elv*(vidx-20), azim= -90 - azm*(vidx-20))                  # elev: -0~-40, azim: -90~-110
            if 30 < vidx and vidx <= 40: ax2.view_init(elev= -0.01-elv*10 + elv*(vidx-30), azim= -90-azm*10 + azm*(vidx-30))    # elev: -40~-0, azim: -110~-90
            if 40 < vidx and vidx <= 50: ax2.view_init(elev= -0.01 - elv*(vidx-40), azim= -90 + azm*(vidx-40))                  # elev: -0~-40, azim: -90~-70
            if 50 < vidx and vidx <= 60: ax2.view_init(elev= -0.01-elv*10 + elv*(vidx-50), azim= -90+azm*10 - azm*(vidx-50))    # elev: -40~-0, azim: -70~-90
        ax2.dist = 10 + 0.1*vidx
        ax2_title = fig.add_subplot(gs[4, 4:6])
        ax2_title.axis('off')
        ax2_title.text(0.5, 0.5, '[Top-view] Unified visual odometry on world coordinate', fontdict=ax2_titlefont)
        plt.tight_layout();


        if args.save_fig:
            print('>> Saving image #{:02d}'.format(vidx))
            # plt.savefig('{:}/{:}_{:04d}.png'.format(args.save_path, Path(args.data).basename(), i), dpi=100)
            plt.savefig('{:}/{:04d}.png'.format(args.save_path, vidx), dpi=100)
            plt.close('all')
        else:
            plt.ion(); plt.show();
            print('>> Type \'c\' to continue')
            pdb.set_trace()
            plt.close('all')

        vidx += 1

    return 0



def load_as_float(path):
    return imread(path).astype(np.float32)


def load_flo_as_float(path):
    out = np.array(flow_read(path)).astype(np.float32)    
    return out


def load_seg_as_float(path):
    out = np.load(path).astype(np.float32)
    return out


def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=True)
    return l2_norm


def find_noc_masks(fwd_flow, bwd_flow):
    '''
    fwd_flow: torch.size([1, 2, 256, 832])
    bwd_flow: torch.size([1, 2, 256, 832])
    output: torch.size([1, 1, 256, 832]), torch.size([1, 1, 256, 832])

    input shape of flow_warp(): torch.size([bs, 2, 256, 832])
    '''
    bwd2fwd_flow, _ = flow_warp(bwd_flow, fwd_flow)
    fwd2bwd_flow, _ = flow_warp(fwd_flow, bwd_flow)

    fwd_flow_diff = torch.abs(bwd2fwd_flow + fwd_flow)
    bwd_flow_diff = torch.abs(fwd2bwd_flow + bwd_flow)

    fwd_consist_bound = torch.max(0.05 * L2_norm(fwd_flow), torch.Tensor([3.0]))
    bwd_consist_bound = torch.max(0.05 * L2_norm(bwd_flow), torch.Tensor([3.0]))

    noc_mask_0 = (L2_norm(fwd_flow_diff) < fwd_consist_bound).type(torch.FloatTensor)     # noc_mask_tgt, torch.Size([1, 1, 256, 832]), torch.float32
    noc_mask_1 = (L2_norm(bwd_flow_diff) < bwd_consist_bound).type(torch.FloatTensor)     # noc_mask_src, torch.Size([1, 1, 256, 832]), torch.float32
    # pdb.set_trace()

    return noc_mask_0, noc_mask_1


def inst_iou(seg_src, seg_tgt, valid_mask):
    '''
    -> seg_src의 인스턴스들이 seg_tgt의 몇 번째 채널 인스턴스에 매칭되는가?

    seg_src: torch.Size([1, n_inst, 256, 832])
    seg_tgt:  torch.Size([1, n_inst, 256, 832])
    valid_mask: torch.Size([1, 1, 256, 832])
    '''
    n_inst_src = seg_src.shape[1]
    n_inst_tgt = seg_tgt.shape[1]

    seg_src_m = seg_src * valid_mask.repeat(1,n_inst_src,1,1)
    seg_tgt_m = seg_tgt * valid_mask.repeat(1,n_inst_tgt,1,1)
    # pdb.set_trace()
    '''
    plt.figure(1), plt.imshow(seg_src.sum(dim=0).sum(dim=0)), plt.colorbar(), plt.ion(), plt.show()
    plt.figure(2), plt.imshow(seg_tgt.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
    plt.figure(3), plt.imshow(valid_mask[0,0]),  plt.colorbar(), plt.ion(), plt.show()
    plt.figure(4), plt.imshow(seg_src_m.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
    '''
    for i in range(n_inst_src):
        if i == 0: 
            match_table = torch.from_numpy(np.zeros([1,n_inst_tgt]).astype(np.float32))
            continue;

        overl = (seg_src_m[:,i].unsqueeze(1).repeat(1,n_inst_tgt,1,1) * seg_tgt_m).clamp(min=0,max=1).squeeze(0).sum(1).sum(1)
        union = (seg_src_m[:,i].unsqueeze(1).repeat(1,n_inst_tgt,1,1) + seg_tgt_m).clamp(min=0,max=1).squeeze(0).sum(1).sum(1)

        iou_inst = overl / union
        match_table = torch.cat((match_table, iou_inst.unsqueeze(0)), dim=0)

    iou, inst_idx = torch.max(match_table,dim=1)
    # pdb.set_trace()

    return iou, inst_idx


def recursive_check_nonzero_inst(tgt_inst, ref_inst):
    assert( tgt_inst[0].mean() == ref_inst[0].mean() )
    n_inst =  int(tgt_inst[0].mean())
    for nn in range(n_inst):
        if tgt_inst[nn+1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn+1 == n_inst:
                tgt_inst[nn+1:] = 0
                ref_inst[nn+1:] = 0
            else:
                tgt_inst[nn+1:] = torch.cat([tgt_inst[nn+2:], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0)    # re-ordering
                ref_inst[nn+1:] = torch.cat([ref_inst[nn+2:], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0)    # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
        if ref_inst[nn+1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn+1 == n_inst:
                tgt_inst[nn+1:] = 0
                ref_inst[nn+1:] = 0
            else:
                tgt_inst[nn+1:] = torch.cat([tgt_inst[nn+2:], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0)    # re-ordering
                ref_inst[nn+1:] = torch.cat([ref_inst[nn+2:], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0)    # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
    return tgt_inst, ref_inst



if __name__ == '__main__':
    main()