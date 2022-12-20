'''
Seokju Lee
PyTorch version 1.4.0, 1.7.0 confirmed

RUN SCRIPT:
./scripts/train_resnet_256_kt.sh
./scripts/train_resnet_256_cs.sh

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

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import models
from datasets.sequence_folders import SequenceFolder
import custom_transforms
import custom_transforms_val
from loss_functions import compute_photo_and_geometry_loss, compute_smooth_loss, compute_obj_size_constraint_loss, compute_mof_consistency_loss, compute_errors
from rigid_warp import forward_warp
from logger import TermLogger, AverageMeter
from utils import save_checkpoint, viz_flow
from collections import OrderedDict

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import pdb



parser = argparse.ArgumentParser(description='Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency (KITTI and CityScapes)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-mni', type=int, help='maximum number of instances', default=20)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw, pitch, roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--disp-lr', '--disp-learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate for DispResNet')
parser.add_argument('--ego-lr', '--ego-learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate for EgoPoseNet')
parser.add_argument('--obj-lr', '--obj-learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate for ObjPoseNet')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', default=3, type=int, metavar='N', help='save frequency')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--resnet-pretrained', action='store_true', help='pretrained from resnet model or not')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained DispResNet')
parser.add_argument('--pretrained-ego-pose', dest='pretrained_ego_pose', default=None, metavar='PATH', help='path to pre-trained EgoPoseNet')
parser.add_argument('--pretrained-obj-pose', dest='pretrained_obj_pose', default=None, metavar='PATH', help='path to pre-trained ObjPoseNet')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')

parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=2.0)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=1.0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-o', '--scale-loss-weight', type=float, help='weight for object scale loss', metavar='W', default=0.02)
parser.add_argument('-mc', '--mof-consistency-loss-weight', type=float, help='weight for mof consistency loss', metavar='W', default=0.1)
parser.add_argument('-hp', '--height-loss-weight', type=float, help='weight for height prior loss', metavar='W', default=0.0)
parser.add_argument('-dm', '--depth-loss-weight', type=float, help='weight for depth mean loss', metavar='W', default=0.0)

parser.add_argument('--with-auto-mask', action='store_true', help='with the the mask for stationary points')
parser.add_argument('--with-ssim', action='store_true', help='with ssim or not')
parser.add_argument('--with-mask', action='store_true', help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-only-obj', action='store_true', help='with only obj mask')

parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--debug-mode', action='store_true', help='run codes with debugging mode or not')
parser.add_argument('--no-shuffle', action='store_true', help='feed data without shuffling')
parser.add_argument('--no-input-aug', action='store_true', help='feed data without augmentation')
parser.add_argument('--begin-idx', type=int, default=None, help='beginning index for pre-processed data')



best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_val = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")



def main():
    print('=> PyTorch version: ' + torch.__version__ + ' || CUDA_VISIBLE_DEVICES: ' + os.environ["CUDA_VISIBLE_DEVICES"])

    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    if args.debug_mode:
        args.save_path = 'checkpoints'/Path('debug')/timestamp
    else:
        args.save_path = 'checkpoints'/Path(args.name)/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    tf_writer = SummaryWriter(args.save_path)

    # Data loading
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize_val = custom_transforms_val.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    if args.with_gt:
        valid_transform = custom_transforms_val.Compose([
            custom_transforms_val.ArrayToTensor(), 
            normalize_val
        ])
    else:
        valid_transform = custom_transforms.Compose([
            custom_transforms.ArrayToTensor(), 
            normalize
        ])

    print("=> fetching scenes from '{}'".format(args.data))
    train_set = SequenceFolder(
        root=args.data,
        train=True,
        seed=args.seed,
        shuffle=not(args.no_shuffle),
        max_num_instances=args.mni,
        sequence_length=args.sequence_length,
        transform=train_transform
    )

    # if no GT is available (e.g., Cityscapes), Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            root=args.data,
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            root=args.data,
            train=False,
            seed=args.seed,
            shuffle=not(args.no_shuffle),
            max_num_instances=args.mni,
            sequence_length=args.sequence_length,
            transform=valid_transform,
            proportion=0.1
        )
    print('=> {} samples found in training set || {} samples found in validation set'.format(len(train_set), len(val_set)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=not(args.debug_mode),
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    ego_pose_net = models.EgoPoseNet(18, args.with_pretrain).to(device)
    obj_pose_net = models.ObjPoseNet(18, args.with_pretrain).to(device)

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
        print("=> using pre-trained weights for DispNet")
        weights = torch.load(args.pretrained_disp)
        if args.resnet_pretrained:
            disp_net.load_state_dict(weights, strict=False)
        else:
            disp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    ego_pose_net = torch.nn.DataParallel(ego_pose_net)
    obj_pose_net = torch.nn.DataParallel(obj_pose_net)

    print('=> setting adam solver')

    optim_params = []
    if args.disp_lr != 0:
        optim_params.append({'params': disp_net.module.encoder.parameters(), 'lr': args.disp_lr})
        optim_params.append({'params': disp_net.module.decoder.parameters(), 'lr': args.disp_lr})
        optim_params.append({'params': disp_net.module.obj_height_prior, 'lr': args.disp_lr * 0.1})
    if args.ego_lr != 0:
        optim_params.append({'params': ego_pose_net.parameters(), 'lr': args.ego_lr})
    if args.obj_lr != 0:
        optim_params.append({'params': obj_pose_net.parameters(), 'lr': args.obj_lr})

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        csv_summary = csv.writer(csvfile, delimiter='\t')
        csv_summary.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        csv_full = csv.writer(csvfile, delimiter='\t')
        csv_full.writerow(['photo_loss', 'geometry_loss', 'smooth_loss', 'scale_loss', 'mof_consistency_loss', 'height_loss', 'depth_loss', 'train_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    
    ### validation at the beginning ###
    if not args.debug_mode:
        if args.pretrained_disp:
            logger.reset_valid_bar()
            if args.with_gt:
                errors, error_names = validate_with_gt(args, val_loader, disp_net, 0, logger)
            else:
                errors, error_names = validate_without_gt(args, val_loader, disp_net, ego_pose_net, obj_pose_net, 0, logger)
            for error, name in zip(errors, error_names):
                tf_writer.add_scalar(name, error, 0)
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))
    

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        ### train for one epoch ###
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, ego_pose_net, obj_pose_net, optimizer, args.epoch_size, logger, tf_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        
        ### evaluate on validation set ###
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, ego_pose_net, obj_pose_net, epoch, logger)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            tf_writer.add_scalar(name, error, epoch)
        
        
        tf_writer.add_scalar('training loss', train_loss, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]     # "errors[1]" or "train_loss"
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            epoch,
            args.save_freq,
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': ego_pose_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': obj_pose_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            csv_summary = csv.writer(csvfile, delimiter='\t')
            csv_summary.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()



def train(args, train_loader, disp_net, ego_pose_net, obj_pose_net, optimizer, epoch_size, logger, tf_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)

    w1, w2, w3 = args.photo_loss_weight, args.geometry_consistency_weight, args.smooth_loss_weight
    w4, w5, w6 = args.scale_loss_weight, args.mof_consistency_loss_weight, args.height_loss_weight
    w7 = args.depth_loss_weight

    # switch to train mode
    disp_net.train().to(device)
    ego_pose_net.train().to(device)
    obj_pose_net.train().to(device)

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_insts, ref_insts) in enumerate(train_loader):
        if args.debug_mode and i > 5: break;
        # if i > 5: break;

        log_losses = i > 0 and n_iter % args.print_freq == 0
        
        ### inputs to GPU ###
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        tgt_insts = [img.to(device) for img in tgt_insts]
        ref_insts = [img.to(device) for img in ref_insts]

        ### input instance masking ###
        tgt_bg_masks = [1 - (img[:,1:].sum(dim=1, keepdim=True)>0).float() for img in tgt_insts]
        ref_bg_masks = [1 - (img[:,1:].sum(dim=1, keepdim=True)>0).float() for img in ref_insts]
        tgt_bg_imgs = [tgt_img * tgt_mask * ref_mask for tgt_mask, ref_mask in zip(tgt_bg_masks, ref_bg_masks)]
        ref_bg_imgs = [ref_img * tgt_mask * ref_mask for ref_img, tgt_mask, ref_mask in zip(ref_imgs, tgt_bg_masks, ref_bg_masks)]
        tgt_obj_masks = [1 - mask for mask in tgt_bg_masks]
        ref_obj_masks = [1 - mask for mask in ref_bg_masks]
        num_insts = [tgt_inst[:,0,0,0].int().detach().cpu().numpy().tolist() for tgt_inst in tgt_insts]     # Number of instances for each sequence

        ### object height piror ###
        height_prior = disp_net.module.obj_height_prior
        
        ### compute depth & ego-motion ###
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        ego_poses_fwd, ego_poses_bwd = compute_ego_pose_with_inv(ego_pose_net, tgt_bg_imgs, ref_bg_imgs)    # [ 2 x ([B, 6]) ]

        ### Remove ego-motion effct: transformation with ego-motion ###
        ### NumRefs(2) >> Nx(C+mni)xHxW,  {t-1}->{t} | {t+1}->{t} ###
        r2t_imgs_ego, r2t_insts_ego, r2t_depths_ego, r2t_vals_ego = compute_ego_warp(ref_imgs, ref_insts, ref_depths, ego_poses_bwd, intrinsics)
        ### NumRefs(2) >> Nx(C+mni)xHxW,  {t}->{t-1} | {t}->{t+1} ###
        t2r_imgs_ego, t2r_insts_ego, t2r_depths_ego, t2r_vals_ego = compute_ego_warp([tgt_img, tgt_img], tgt_insts, [tgt_depth, tgt_depth], ego_poses_fwd, intrinsics)
        # pdb.set_trace() # BREAKPOINT-1
        # continue;
        '''
            sq = 0; bb = 0;
            colormap = 'turbo'  # turbo, plasma
            tgt = (tgt_img[bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            ref = (ref_imgs[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            r2t_val = r2t_vals_ego[sq][bb,0].detach().cpu().numpy()
            t2r_val = t2r_vals_ego[sq][bb,0].detach().cpu().numpy()
            r2t = (r2t_imgs_ego[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            t2r = (t2r_imgs_ego[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            r2t_inst = r2t_insts_ego[sq][bb,1:].detach().cpu().numpy().transpose(1,2,0)
            t2r_inst = t2r_insts_ego[sq][bb,1:].detach().cpu().numpy().transpose(1,2,0)
            tgt_d = 1/tgt_depth[bb,0].detach().cpu().numpy()
            ref_d = 1/ref_depths[sq][bb,0].detach().cpu().numpy()
            r2t_d_prev = 1/r2t_depths_ego[0][bb,0].detach().cpu().numpy()
            r2t_d_next = 1/r2t_depths_ego[1][bb,0].detach().cpu().numpy()
            r2t_d_prev_err = (tgt_d - r2t_d_prev)
            r2t_d_next_err = (tgt_d - r2t_d_next)
            plt.close('all');
            ea1 = 4; ea2 = 4; ii = 1;
            fig = plt.figure(1, figsize=(21, 9))
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(ref); plt.colorbar(); plt.text(10, -14, "ref", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(tgt); plt.colorbar(); plt.text(10, -14, "tgt", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_val, vmax=1, vmin=0); plt.colorbar(); plt.text(10, -14, "r2t_val", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_val, vmax=1, vmin=0); plt.colorbar(); plt.text(10, -14, "t2r_val", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t); plt.colorbar(); plt.text(10, -14, "r2t", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r); plt.colorbar(); plt.text(10, -14, "t2r", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(ref_d, cmap=colormap); plt.colorbar(); plt.text(10, -14, "ref_d", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(tgt_d, cmap=colormap); plt.colorbar(); plt.text(10, -14, "tgt_d", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_inst); plt.colorbar(); plt.text(10, -14, "r2t_inst", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(t2r_inst); plt.colorbar(); plt.text(10, -14, "t2r_inst", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_d_prev, cmap=colormap); plt.colorbar(); plt.text(10, -14, "r2t_d_prev", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_d_next, cmap=colormap); plt.colorbar(); plt.text(10, -14, "r2t_d_next", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_d_prev_err, vmax=0.5, vmin=-0.5); plt.colorbar(); plt.text(10, -14, "r2t_d_prev_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1; plt.imshow(r2t_d_next_err, vmax=0.5, vmin=-0.5); plt.colorbar(); plt.text(10, -14, "r2t_d_next_err", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.tight_layout(); plt.ion(); plt.show()
            
        '''

        ### Compute object motion ###
        obj_poses_fwd, obj_poses_bwd = compute_obj_pose_with_inv(obj_pose_net, tgt_img, tgt_insts, r2t_imgs_ego, r2t_insts_ego, ref_imgs, ref_insts, t2r_imgs_ego, t2r_insts_ego, intrinsics, args.mni, num_insts)

        ### Compute composite motion field ###
        tot_mofs_fwd, tot_mofs_bwd = compute_motion_field(tgt_img, ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts)

        ### Compute unified projection loss ###
        loss_1, loss_2, r2t_imgs, t2r_imgs, r2t_flows, t2r_flows, r2t_diffs, t2r_diffs, r2t_vals, t2r_vals = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, tot_mofs_fwd, tot_mofs_bwd, \
                                                                                                                                             args.with_ssim, args.with_mask, args.with_auto_mask, args.padding_mode, args.with_only_obj, \
                                                                                                                                             tgt_obj_masks, ref_obj_masks, r2t_vals_ego, t2r_vals_ego)
        # pdb.set_trace() # BREAKPOINT-2
        # continue;
        ''' ### dpoint ###
            sq = 0; bb = 0;
            colormap = 'turbo'  # turbo, plasma
            mmax = +0.01; mmin = -0.01;
            plt.close('all')
            globals().update(locals())
            ref = (ref_imgs[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            tgt = (tgt_img[bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            r2t_ego = (r2t_imgs_ego[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            t2r_ego = (t2r_imgs_ego[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            r2t = (r2t_imgs[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            t2r = (t2r_imgs[sq][bb]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
            err_tgt = np.abs(tgt - r2t).mean(axis=-1)
            err_ref = np.abs(ref - t2r).mean(axis=-1)
            diff_tgt = r2t_diffs[sq][bb].detach().cpu().numpy().mean(axis=0)
            diff_ref = t2r_diffs[sq][bb].detach().cpu().numpy().mean(axis=0)
            tgtD = (1/tgt_depth[bb,0]).detach().cpu().numpy()
            refD = (1/ref_depths[sq][bb,0]).detach().cpu().numpy()
            tgtMOF_tx = tot_mofs_fwd[sq][bb,0].detach().cpu().numpy()
            tgtMOF_ty = tot_mofs_fwd[sq][bb,1].detach().cpu().numpy()
            tgtMOF_tz = tot_mofs_fwd[sq][bb,2].detach().cpu().numpy()
            refMOF_tx = tot_mofs_bwd[sq][bb,0].detach().cpu().numpy()
            refMOF_ty = tot_mofs_bwd[sq][bb,1].detach().cpu().numpy()
            refMOF_tz = tot_mofs_bwd[sq][bb,2].detach().cpu().numpy()
            tgtMOF_tx_obj = tgtMOF_tx - ego_poses_fwd[sq][bb][0].detach().cpu().numpy()
            tgtMOF_tz_obj = tgtMOF_tz - ego_poses_fwd[sq][bb][2].detach().cpu().numpy()
            refMOF_tx_obj = refMOF_tx - ego_poses_bwd[sq][bb][0].detach().cpu().numpy()
            refMOF_tz_obj = refMOF_tz - ego_poses_bwd[sq][bb][2].detach().cpu().numpy()
            tgtMF_tot = viz_flow(tgtMOF_tx, tgtMOF_tz, scaledown=0.05, output=False)
            refMF_tot = viz_flow(refMOF_tx, refMOF_tz, scaledown=0.05, output=False)
            tgtMF_obj = viz_flow(tgtMOF_tx_obj, tgtMOF_tz_obj, scaledown=0.01, output=False)
            refMF_obj = viz_flow(refMOF_tx_obj, refMOF_tz_obj, scaledown=0.01, output=False)
            tgt_obj_tx = np.array(list(OrderedDict.fromkeys(tgtMOF_tx_obj.reshape(-1)).keys()))
            tgt_obj_tz = np.array(list(OrderedDict.fromkeys(tgtMOF_tz_obj.reshape(-1)).keys()))
            ref_obj_tx = np.array(list(OrderedDict.fromkeys(refMOF_tx_obj.reshape(-1)).keys()))
            ref_obj_tz = np.array(list(OrderedDict.fromkeys(refMOF_tz_obj.reshape(-1)).keys()))
            tgtT_obj = [[tx, tz] for tx, tz in zip(tgt_obj_tx, tgt_obj_tz)]
            refT_obj = [[tx, tz] for tx, tz in zip(ref_obj_tx, ref_obj_tz)]
            plt.close('all');
            ea1 = 6; ea2 = 4; ii = 1;
            fig = plt.figure(1, figsize=(21, 12))   # figsize=(22, 13)
            gs = GridSpec(nrows=6, ncols=4)
            fig.add_subplot(gs[0, 0:2]); ii += 2;
            plt.imshow(ref); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "ref", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[0, 2:4]); ii += 2;
            plt.imshow(tgt); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgt", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[1, 0:2]); ii += 2;
            plt.imshow(t2r_ego); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r_ego", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[1, 2:4]); ii += 2;b
            plt.imshow(r2t_ego); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t_ego", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[2, 0:2]); ii += 2;
            plt.imshow(t2r); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "t2r", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[2, 2:4]); ii += 2;
            plt.imshow(r2t); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "r2t", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[3, 0:2]); ii += 2;
            plt.imshow(refD, cmap=colormap); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "refD", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(gs[3, 2:4]); ii += 2;
            plt.imshow(tgtD, cmap=colormap); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgtD", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgtMF_tot); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4);
            plt.text(10, -14, "tgtMF_tot,  ego (tx, tz): ({:.4f}, {:.4f})".format(ego_poses_fwd[sq][bb][0], ego_poses_fwd[sq][bb][2]), fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(refMF_tot); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4);
            plt.text(10, -14, "refMF_tot,  ego (tx, tz): ({:.4f}, {:.4f})".format(ego_poses_bwd[sq][bb][0], ego_poses_bwd[sq][bb][2]), fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgtMF_obj); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4);
            plt.text(10, -14, "tgtMF_obj:", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            [plt.text(140+(ii-1)*200, -14, "({:.4f}, {:.4f})".format(tx, tz), fontsize=7, bbox=dict(facecolor='None', edgecolor='None')) for ii, (tx, tz) in enumerate(zip(tgt_obj_tx, tgt_obj_tz)) if ii>0]
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(refMF_obj); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4);
            plt.text(10, -14, "refMF_obj:", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            [plt.text(140+(ii-1)*200, -14, "({:.4f}, {:.4f})".format(tx, tz), fontsize=7, bbox=dict(facecolor='None', edgecolor='None')) for ii, (tx, tz) in enumerate(zip(ref_obj_tx, ref_obj_tz)) if ii>0]
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(err_tgt, cmap='bone', vmax=0.5, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "err_tgt", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(err_ref, cmap='bone', vmax=0.5, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "err_ref", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.tight_layout(); plt.ion(); 
            plt.show()


            plt.imshow(tgtMOF_tx, vmax=ego_poses_fwd[sq][bb,0].item()+mmax, vmin=ego_poses_fwd[sq][bb,0].item()+mmin); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgtMOF_tx", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(refMOF_tx, vmax=ego_poses_bwd[sq][bb,0].item()+mmax, vmin=ego_poses_bwd[sq][bb,0].item()+mmin); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "refMOF_tx", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(tgtMOF_ty, vmax=ego_poses_fwd[sq][bb,1].item()+mmax, vmin=ego_poses_fwd[sq][bb,1].item()+mmin); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgtMOF_ty", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(refMOF_ty, vmax=ego_poses_bwd[sq][bb,1].item()+mmax, vmin=ego_poses_bwd[sq][bb,1].item()+mmin); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "refMOF_ty", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(tgtMOF_tz, vmax=ego_poses_fwd[sq][bb,2].item()+mmax*2, vmin=ego_poses_fwd[sq][bb,2].item()+mmin*2); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "tgtMOF_tz", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(refMOF_tz, vmax=ego_poses_bwd[sq][bb,2].item()+mmax*2, vmin=ego_poses_bwd[sq][bb,2].item()+mmin*2); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "refMOF_tz", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            
            plt.imshow(diff_tgt, cmap='bone', vmax=0.5, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "diff_tgt", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            plt.imshow(diff_ref, cmap='bone', vmax=0.5, vmin=0); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.text(10, -14, "diff_ref", fontsize=7, bbox=dict(facecolor='None', edgecolor='None'));
            
        '''

        ### Compute depth smoothness loss ###
        if w3 == 0:
            loss_3 = torch.tensor(.0).cuda()
        else:
            loss_3 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)


        ### Compute object size constraint loss ###
        if w4 == 0:
            loss_4 = torch.tensor(.0).cuda()
        else:
            loss_4 = compute_obj_size_constraint_loss(height_prior, tgt_depth, tgt_insts, ref_depths, ref_insts, intrinsics, args.mni, num_insts)


        ### Compute unified motion consistency loss ###
        if w5 == 0:
            loss_5 = torch.tensor(.0).cuda()
        else:
            loss_5 = compute_mof_consistency_loss(tot_mofs_fwd, tot_mofs_bwd, r2t_flows, t2r_flows, r2t_diffs, t2r_diffs, r2t_vals, t2r_vals, alpha=5, thresh=0.1)


        ### Compute height prior constraint loss ###
        loss_6 = height_prior


        ### Compute depth mean constraint loss ###
        loss_7 = ((1/tgt_depth).mean() + sum([(1/depth).mean() for depth in ref_depths])) / (1 + len(ref_depths))

        
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5 + w6*loss_6 + w7*loss_7
        # pdb.set_trace()
        '''
            loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item()
            w1*loss_1.item(), w2*loss_2.item(), w3*loss_3.item(), w4*loss_4.item(), w5*loss_5.item()
            
            -b 4 -p 1.0 -c 0.5 -s 0.05 -o 0.01 -mc 0.01 -hp 0 -dm 0 -mni 3 \

        '''

        if log_losses:
            tf_writer.add_scalar('photo_loss', loss_1.item(), n_iter)
            tf_writer.add_scalar('geometry_loss', loss_2.item(), n_iter)
            tf_writer.add_scalar('smooth_loss', loss_3.item(), n_iter)
            tf_writer.add_scalar('scale_loss', loss_4.item(), n_iter)
            tf_writer.add_scalar('mof_consistency_loss', loss_5.item(), n_iter)
            tf_writer.add_scalar('height_loss', loss_6.item(), n_iter)
            tf_writer.add_scalar('depth_loss', loss_7.item(), n_iter)
            tf_writer.add_scalar('total_loss', loss.item(), n_iter)

        ### record loss ###
        losses.update(loss.item(), args.batch_size)

        ### compute gradient and do Adam step ###
        if loss > 0:
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()

        ### measure elapsed time ###
        batch_time.update(time.time() - end)
        end = time.time()

        ### fail-safe ###
        if np.isnan(loss.detach().cpu().numpy()): pdb.set_trace()
        if np.isnan(height_prior.detach().cpu().numpy()): pdb.set_trace()
        if np.isnan(tgt_depth.mean().detach().cpu().numpy()): pdb.set_trace()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            csv_full = csv.writer(csvfile, delimiter='\t')
            csv_full.writerow([loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(), loss_6.item(), loss_7.item(), loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]



@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, ego_pose_net, obj_pose_net, epoch, logger):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)

    w1, w2, w3 = args.photo_loss_weight, args.geometry_consistency_weight, args.smooth_loss_weight
    
    # switch to evaluation mode
    disp_net.eval()
    ego_pose_net.eval()
    obj_pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    # for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_insts, ref_insts) in enumerate(val_loader):
        if args.debug_mode and i > 5: break;
        # if i > 5: break;
        
        ### inputs to GPU ###
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        tgt_insts = [img.to(device) for img in tgt_insts]
        ref_insts = [img.to(device) for img in ref_insts]

        ### input instance masking ###
        tgt_bg_masks = [1 - (img[:,1:].sum(dim=1, keepdim=True)>0).float() for img in tgt_insts]
        ref_bg_masks = [1 - (img[:,1:].sum(dim=1, keepdim=True)>0).float() for img in ref_insts]
        tgt_bg_imgs = [tgt_img * tgt_mask * ref_mask for tgt_mask, ref_mask in zip(tgt_bg_masks, ref_bg_masks)]
        ref_bg_imgs = [ref_img * tgt_mask * ref_mask for ref_img, tgt_mask, ref_mask in zip(ref_imgs, tgt_bg_masks, ref_bg_masks)]
        tgt_obj_masks = [1 - mask for mask in tgt_bg_masks]
        ref_obj_masks = [1 - mask for mask in ref_bg_masks]
        num_insts = [tgt_inst[:,0,0,0].int().detach().cpu().numpy().tolist() for tgt_inst in tgt_insts]     # Number of instances for each sequence

        ### compute depth & ego-motion ###
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        ego_poses_fwd, ego_poses_bwd = compute_ego_pose_with_inv(ego_pose_net, tgt_bg_imgs, ref_bg_imgs)    # [ 2 x ([B, 6]) ]

        ### Remove ego-motion effct: transformation with ego-motion ###
        r2t_imgs_ego, r2t_insts_ego, r2t_depths_ego, r2t_vals_ego = compute_ego_warp(ref_imgs, ref_insts, ref_depths, ego_poses_bwd, intrinsics)
        t2r_imgs_ego, t2r_insts_ego, t2r_depths_ego, t2r_vals_ego = compute_ego_warp([tgt_img, tgt_img], tgt_insts, [tgt_depth, tgt_depth], ego_poses_fwd, intrinsics)
        
        ### Compute object motion ###
        obj_poses_fwd, obj_poses_bwd = compute_obj_pose_with_inv(obj_pose_net, tgt_img, tgt_insts, r2t_imgs_ego, r2t_insts_ego, ref_imgs, ref_insts, t2r_imgs_ego, t2r_insts_ego, intrinsics, args.mni, num_insts)

        ### Compute composite motion field ###
        tot_mofs_fwd, tot_mofs_bwd = compute_motion_field(tgt_img, ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts)

        ### Compute unified projection loss ###
        loss_1, loss_2, _, _, _, _, _, _, _, _ = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, tot_mofs_fwd, tot_mofs_bwd, \
                                                                                 args.with_ssim, args.with_mask, args.with_auto_mask, args.padding_mode, args.with_only_obj, \
                                                                                 tgt_obj_masks, ref_obj_masks, r2t_vals_ego, t2r_vals_ego)
        # pdb.set_trace()

        ### Compute depth smoothness loss ###
        loss_3 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Geometry loss', 'Smooth loss']



@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger):
    global device_val
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    errors_fg = AverageMeter(i=len(error_names))
    errors_bg = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    disp_net = disp_net.module.to(device_val)
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, (tgt_img, depth, tgt_inst_sum) in enumerate(val_loader):
        if args.debug_mode and i > 5: break;
        # if i > 5: break;

        tgt_img = tgt_img.to(device_val)    # B, 3, 256, 832
        depth = depth.to(device_val)
        tgt_inst_sum = tgt_inst_sum.to(device_val)

        vmask = (depth > 0).float()
        fg_pixs = vmask * tgt_inst_sum
        bg_pixs = vmask * (1 - tgt_inst_sum)
        fg_ratio = (fg_pixs.sum(dim=1).sum(dim=1) / vmask.sum(dim=1).sum(dim=1)).mean()
        depth_fg = depth * tgt_inst_sum
        depth_bg = depth * (1 - tgt_inst_sum)

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:,0]

        error_all, med_scale = compute_errors(depth, output_depth)
        errors.update(error_all)

        errors_bg.update(compute_errors(depth_bg, output_depth, med_scale)[0])
        if fg_ratio:
            errors_fg.update(compute_errors(depth_fg, output_depth, med_scale)[0])
        # pdb.set_trace()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs_Rel Error {:.4f} ({:.4f})'.format(batch_time, errors.val[1], errors.avg[1]))
    logger.valid_bar.update(len(val_loader))

    return errors.avg, error_names



################################################################################################################################################################################



def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = 1/disp_net(tgt_img)
    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = 1/disp_net(ref_img)
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_ego_pose_with_inv(pose_net, tgt_imgs, ref_imgs):
    poses_fwd = []
    poses_bwd = []
    for tgt_img, ref_img in zip(tgt_imgs, ref_imgs):
        poses_fwd.append(pose_net(tgt_img, ref_img))
        poses_bwd.append(pose_net(ref_img, tgt_img))

    return poses_fwd, poses_bwd


def compute_ego_warp(imgs, insts, depths, poses, intrinsics, is_detach=True):
    '''
        Args:
            imgs:       [[B, 3, 256, 832], [B, 3, 256, 832]]
            insts:      [[B, 3, 256, 832], [B, 3, 256, 832]]
            depths:     [[B, 1, 256, 832], [B, 1, 256, 832]]
            poses:      [[B, 6], [B, 6]]
            intrinsics: [B, 3, 3]
        Returns:
            w_imgs:     [[B, 3, 256, 832], [B, 3, 256, 832]]
            w_vals:     [[B, 1, 256, 832], [B, 1, 256, 832]]
    '''
    w_imgs, w_insts, w_depths, w_vals = [], [], [], []
    for img, inst, depth, pose in zip(imgs, insts, depths, poses):
        img_cat = torch.cat([img, inst[:,1:]], dim=1)
        if is_detach:
            w_img_cat, w_depth, w_val = forward_warp(img_cat, depth.detach(), pose.detach(), intrinsics, upscale=3)
        else:
            w_img_cat, w_depth, w_val = forward_warp(img_cat, depth, pose, intrinsics, upscale=3)
        w_imgs.append( w_img_cat[:,:3] )
        w_insts.append( torch.cat([inst[:,:1], w_img_cat[:,3:].round()], dim=1) )
        w_depths.append( w_depth )
        w_vals.append( w_val )

    return w_imgs, w_insts, w_depths, w_vals


def compute_obj_pose_with_inv(pose_net,   tgtI, tgtMs, r2tIs, r2tMs,   refIs, refMs, t2rIs, t2rMs,   intrinsics, mni, num_insts):
    '''
        Args:
            ------------------------------------------------
            tgtI:  [B, 3, 256, 832]
            tgtMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
            r2tIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
            r2tMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
            ------------------------------------------------
            refIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
            refMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
            t2rIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
            t2rMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
            ------------------------------------------------
            intrinsics: [B, 3, 3]
            num_insts:  [[n1, n2, ...], [n1', n2', ...]]
        Returns:
            "Only translations (tx, ty, tz) are estimated!"
            obj_poses_fwd: [[B, N, 3], [B, N, 3]]
            obj_poses_bwd: [[B, N, 3], [B, N, 3]]
        
        plt.close('all')
        bb = 0
        plt.figure(1); plt.imshow(tgtI.detach().cpu()[bb,0]); plt.colorbar(); plt.ion(); plt.show();

    '''
    bs, _, hh, ww = tgtI.size()

    obj_poses_fwd, obj_poses_bwd = [], []
    
    for tgtM, r2tI, r2tM,   refI, refM, t2rI, t2rM,   num_inst in zip(tgtMs, r2tIs, r2tMs,   refIs, refMs, t2rIs, t2rMs,   num_insts):
        obj_pose_fwd = torch.zeros([bs*mni, 3]).type_as(tgtI)
        obj_pose_bwd = torch.zeros([bs*mni, 3]).type_as(tgtI)

        if sum(num_inst) != 0:
            tgtI_rep = tgtI.repeat_interleave(mni, dim=0)
            tgtM_rep = tgtM[:,1:].reshape(-1,1,hh,ww)
            fwdIdx = (tgtM_rep.mean(dim=[1,2,3])!=0).detach()   # tgt, judge each channel whether instance exists
            tgtO = (tgtI_rep * tgtM_rep)[fwdIdx]

            r2tI_rep = r2tI.repeat_interleave(mni, dim=0)
            r2tM_rep = r2tM[:,1:].reshape(-1,1,hh,ww)
            r2tO = (r2tI_rep * r2tM_rep)[fwdIdx]

            refI_rep = refI.repeat_interleave(mni, dim=0)
            refM_rep = refM[:,1:].reshape(-1,1,hh,ww)
            bwdIdx = (refM_rep.mean(dim=[1,2,3])!=0).detach()   # ref, judge each channel whether instance exists
            refO = (refI_rep * refM_rep)[bwdIdx]

            t2rI_rep = t2rI.repeat_interleave(mni, dim=0)
            t2rM_rep = t2rM[:,1:].reshape(-1,1,hh,ww)
            t2rO = (t2rI_rep * t2rM_rep)[bwdIdx]

            pose_fwd = pose_net(tgtO, r2tO)
            pose_bwd = pose_net(refO, t2rO)
            obj_pose_fwd[fwdIdx] = pose_fwd
            obj_pose_bwd[bwdIdx] = pose_bwd
            # pdb.set_trace()

        obj_poses_fwd.append( obj_pose_fwd.reshape(bs, mni, 3) )
        obj_poses_bwd.append( obj_pose_bwd.reshape(bs, mni, 3) )

    return obj_poses_fwd, obj_poses_bwd


def compute_motion_field(tgt_img, ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts):
    '''
        Args:
            ego_poses_fwd: [torch.Size([B, 6]), torch.Size([B, 6])]
            ego_poses_bwd: [torch.Size([B, 6]), torch.Size([B, 6])]
            obj_poses_fwd: [torch.Size([B, N, 6]), torch.Size([B, N, 6])]
            obj_poses_bwd: [torch.Size([B, N, 6]), torch.Size([B, N, 6])]
            tgt_insts: [torch.Size([B, 1+N, 256, 832]), torch.Size([B, 1+N, 256, 832])]
            ref_insts: [torch.Size([B, 1+N, 256, 832]), torch.Size([B, 1+N, 256, 832])]
        Returns:
            MFs_fwd: [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]
            MFs_bwd: [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]

        plt.close('all')
        bb = 0; su = 2;
        plt.figure(1); plt.imshow(obj_MF_fwd.sum(dim=1, keepdim=False)[bb,su].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()

    '''
    bs, _, hh, ww = tgt_img.size()
    MFs_fwd, MFs_bwd = [], []   # [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]

    for EP_fwd, EP_bwd, OP_fwd, OP_bwd, tgt_inst, ref_inst in zip(ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts):
        if (tgt_inst[:,1:].sum(dim=1)>1).sum() + (ref_inst[:,1:].sum(dim=1)>1).sum(): 
            print("WARNING: overlapped instance region at {}".format(datetime.datetime.now().strftime("%m-%d-%H:%M")))

        MF_fwd = EP_fwd.reshape(bs, 6, 1, 1).repeat(1,1,hh,ww)
        MF_bwd = EP_bwd.reshape(bs, 6, 1, 1).repeat(1,1,hh,ww)

        obj_MF_fwd = tgt_inst[:,1:].unsqueeze(2) * OP_fwd.unsqueeze(-1).unsqueeze(-1)   # [bs, mni, 3, hh, ww]
        obj_MF_bwd = ref_inst[:,1:].unsqueeze(2) * OP_bwd.unsqueeze(-1).unsqueeze(-1)   # [bs, mni, 3, hh, ww]

        MF_fwd[:,:3] += obj_MF_fwd.sum(dim=1, keepdim=False)
        MF_bwd[:,:3] += obj_MF_bwd.sum(dim=1, keepdim=False)

        MFs_fwd.append(MF_fwd)
        MFs_bwd.append(MF_bwd)

    return MFs_fwd, MFs_bwd


def save_image(data, cm, fn, vmin=None, vmax=None):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax)
    plt.savefig(fn, dpi = height) 
    plt.close()


if __name__ == '__main__':
    main()