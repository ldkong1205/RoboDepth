from __future__ import absolute_import, division, print_function
import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
sys.path.append('..')
from mono.datasets.euroc_dataset import FolderDataset
from mono.datasets.kitti_dataset import KITTIOdomDataset
from mono.datasets.utils import readlines,transformation_from_parameters
from mono.model.mono_baseline.pose_encoder import PoseEncoder
from mono.model.mono_baseline.pose_decoder import PoseDecoder
from mono.tools.kitti_evaluation_toolkit import kittiOdomEval


def odo(opt):
    if opt.kitti:
        filenames = readlines("../mono/datasets/splits/odom/test_files_{:02d}.txt".format(opt.sequence_id))

        dataset = KITTIOdomDataset(opt.data_path,
                                   filenames,
                                   opt.height,
                                   opt.width,
                                   [0, 1],
                                   is_train=False,
                                   img_ext='.png',
                                   gt_depth_path=None)
    else:
        dataset = FolderDataset(opt.data_path,
                                None,
                                opt.height,
                                opt.width,
                                [0, 1],
                                is_train=False,
                                img_ext='.png',
                                gt_depth_path=None)

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    pose_encoder = PoseEncoder(18, None, 2)
    pose_decoder = PoseDecoder(pose_encoder.num_ch_enc)

    checkpoint = torch.load(opt.model_path)
    for name, param in pose_encoder.state_dict().items():
        pose_encoder.state_dict()[name].copy_(checkpoint['state_dict']['PoseEncoder.' + name])
    for name, param in pose_decoder.state_dict().items():
        pose_decoder.state_dict()[name].copy_(checkpoint['state_dict']['PoseDecoder.' + name])
    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    global_pose = np.identity(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0,1]], 1)
            axisangle, translation = pose_decoder(pose_encoder(all_color_aug))
            g = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
            backward_transform = g.squeeze().cpu().numpy()#the transformation from frame +1 to frame 0
            global_pose = global_pose @ np.linalg.inv(backward_transform)
            poses.append(global_pose[0:3, :].reshape(1, 12))
    poses = np.concatenate(poses, axis=0)

    if opt.kitti:
        filename = os.path.join(opt.result_dir, "{:02d}_pred.txt".format(opt.sequence_id))
    else:
        filename = os.path.join(opt.result_dir, "fm_ms_euroc_mh04_diff_3.txt")

    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
    if opt.kitti:
        opt.eva_seqs = '{:02d}_pred'.format(opt.sequence_id)
        pose_eval = kittiOdomEval(opt)
        pose_eval.eval(toCameraCoord=False)  # set the value according to the predicted results
    print('saving into ', opt.result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--model_path', default='/media/sconly/24eda5d5-e79b-423b-8dcc-8339a15f3219/weight/fm_depth_odom.pth', help='model save path')
    parser.add_argument('--data_path', default='/media/sconly/24eda5d5-e79b-423b-8dcc-8339a15f3219/data/kitti/Odometry', help='kitti odometry dataset path')
    parser.add_argument('--gt_dir', default='../mono/datasets/gt_pose',help='kitti odometry gt path')
    parser.add_argument('--result_dir', default='/media/sconly/24eda5d5-e79b-423b-8dcc-8339a15f3219/odom/')
    parser.add_argument('--height', default=192)
    parser.add_argument('--width', default=640)
    parser.add_argument('--kitti', default=True, help='whether test on the kitti odometry dataset')
    parser.add_argument('--sequence_id', default=9, help='which kitti odometry sequence for testing')
    opts = parser.parse_args()
    odo(opts)
    print("you can also run 'evo_traj kitti -s *.txt *.txt --ref=*.txt -p --plot_mode=xz' in terminal for visualization")