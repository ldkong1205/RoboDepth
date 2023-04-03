# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join("data", "kitti", "kitti_raw"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="dynadepth")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_quater", "eigen_half", "eigen_three_quater", "eigen_zhou", "eigen_full", "odom", "benchmark", "test"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--num_layers_imu",
                                 type=int,
                                 help="number of resnet layers, for velo and gravity networks",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])
        
        # ABLATION STUDY for DynaDepth
        self.parser.add_argument("--imu_filename",
                                 type=str,
                                 default="matched_oxts.txt",
                                 help="matched_oxts means clean imu data, while we can use other names for noisy oxts (e.g., matched_oxts_gyro/acc/both_1.0/0.1/0.01.txt)")
        self.parser.add_argument("--img_noise_type",
                                 type=str,
                                 default="clean",
                                 choices=["clean", "bcsh", "gg", "mask"],
                                 help="(1) bcsh: brightness/contrast/saturation/hue (2) gg: gamma/gain (3) mask" )
        self.parser.add_argument("--img_noise_brightness",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_noise_contrast",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_noise_saturation",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_noise_hue",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_noise_gamma",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_noise_gain",
                                 type=float,
                                 default=0.5)
        self.parser.add_argument("--img_mask_num",
                                 type=int,
                                 default=1)
        self.parser.add_argument("--img_mask_size",
                                 type=int,
                                 default=50,
                                 help="square and masked on raw image size (inputs[(n,im,-1)] before resize): e.g. 1241x376")
        self.parser.add_argument("--avoid_quat_check",
                                 help="if set, will not check quat.norm to be close to 1 (useful for noisy imu)",
                                 action="store_true")
        
        # IMU options
        self.parser.add_argument("--display_velo_scale",
                                 help="if set, will log the abs diff of velo norm in tensorboard",
                                 action="store_true")
        self.parser.add_argument("--imu_l2_weight",
                                 type=float,
                                 help="l2 loss for imu pre-integrated poses and camera-predicted poses",
                                 default=0)
        self.parser.add_argument("--imu_warp_weight",
                                 type=float,
                                 help="imu warping loss weight",
                                 default=0.5)
        self.parser.add_argument("--imu_consistency_weight",
                                 type=float,
                                 help="imu consistency loss weight",
                                 default=0.01)
        self.parser.add_argument("--trans_scale_factor",
                                 type=float,
                                 help="real baseline: 0.54m, baseline in the code: 0.1m",
                                 default=5.4)
        self.parser.add_argument("--no_grad_imu_consistency",
                                 type=str,
                                 help="disable gradient backpropagation of the specified option",
                                 default="none",
                                 choices=["network", "imu", "none"])
        self.parser.add_argument("--first_save_epoch",
                                 type=int,
                                 help="the first epoch to save models",
                                 default=10)
        self.parser.add_argument("--predict_velo_residue",
                                 help="if set predict velo residual based on predicted translation / dt",
                                 action="store_true")
        self.parser.add_argument("--velo_weight",
                                 type=float,
                                 help="loss weight for velocity_norm",
                                 default=0.001)
        self.parser.add_argument("--gravity_weight",
                                 type=float,
                                 help="loss weight for gravity_norm",
                                 default=0.001)
    
        ## EKF options
        self.parser.add_argument("--use_ekf", 
                                 help="online denoising and integrating IMU data",
                                 action="store_true")
        self.parser.add_argument("--ekf_warming_epochs", 
                                 type=int,
                                 help="when --use_ekf, pretraining epochs before adding ekf",
                                 default=0)
        self.parser.add_argument("--ekf_velo_weight",
                                 type=float,
                                 help="loss weight for L2 norm of ekf_v and v_ck",
                                 default=0)
        self.parser.add_argument("--ekf_gravity_weight",
                                 type=float,
                                 help="loss weight for L2 norm of ekf_g and g_ck",
                                 default=0)

        self.parser.add_argument("--train_init_covar", 
                                 help="train the imu init covar, otherwise fixed",
                                 action="store_true")
        self.parser.add_argument("--train_imu_noise_covar", 
                                 help="train the imu noise covar, otherwise fixed",
                                 action="store_true")
        self.parser.add_argument("--vis_covar_use_fixed", 
                                 help="use fixed vis_covar, otherwise predict by CNN",
                                 action="store_true")
                                 
        self.parser.add_argument("--sample_vis_pose", 
                                 help="Sample the vis_pose for warping from the normal distribution",
                                 action="store_true")
        self.parser.add_argument("--naive_vis_covar", 
                                 help="directly regress vis_covar rather than using 10^(3*tanh(x))",
                                 action="store_true")

        self.parser.add_argument("--resume_imu", 
                                 help="resume from our imu-only checkpoint after --resume_epochs",
                                 action="store_true")
        self.parser.add_argument("--resume_gravity", 
                                 help="resume from our imu+gravity checkpoint after --resume_epochs",
                                 action="store_true")
        self.parser.add_argument("--resume_velo", 
                                 help="resume from our imu+velo+gravity checkpoint after --resume_epochs",
                                 action="store_true")
        self.parser.add_argument("--resume_epochs", 
                                 type=int,
                                 help="pretrain epochs when --resume_imu/gravity/velo",
                                 default=0)


        self.parser.add_argument("--k_imu_clip",
                                 type=int,
                                 help="k_imu_clip x 12 IMU records will be fed into CNN",
                                 default=5)
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode that uses median scaling, otherwise uses the scale-aware evaluation",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--show_scale_hist",
                                 action="store_true")
        self.parser.add_argument("--save_make3d",
                                 action="store_true")
        
        # RoboDepth options
        self.parser.add_argument("--eval_corr_type",
                                 help="name of the corruptions to be evaluated on",
                                 choices=[
                                   'all', 'clean', 
                                   'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform', 'color_quant',
                                   'gaussian_noise', 'impulse_noise', 'shot_noise', 'iso_noise', 'pixelate', 'jpeg_compression'
                                 ],
                                 default=None)

        self.parser.add_argument("--clean",
                                 help="clean results of the model to be evaluated;"
                                   "List of 7 numbers: ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']",
                                 default=None)

        self.parser.add_argument('--seed', type=int, 
                                 help='random seed.',
                                 default=1205)
      
        self.parser.add_argument('--deterministic',
                                 action='store_true',
                                 help='whether to set deterministic options for CUDNN backend.')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
