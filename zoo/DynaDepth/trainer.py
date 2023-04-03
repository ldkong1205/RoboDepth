# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import pdb
import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from ekf import EKFModel
from ekf import proc_vis_covar
from IPython import embed


# torch.autograd.set_detect_anomaly(True)

def compute_imu_pose_with_inv(alpha, R_c, R_cbt_bc, delta_t, gravities, velocities, trans_scale_factor):
    """
    -> Rotation is directly obtained using preintegrated q
    -> Translation is obtained by preintegrated alpha and q
    -> Return:
        -> poses: [from 0 to -1, from 0 to 1]
        -> poses_inv: [from -1 to 0, from 1 to 0]
    """
    # NOTE: T_c denotes [from 0 to -1, from 1 to 0]
    # NOTE: T_c_inv denotes [from -1 to 0, from 0 to 1]
    T_c, T_c_inv = [], []
    for i in [0, 1]:
        rot = R_c[i] # [B, 3, 3]
        dt = delta_t[i].unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
        trans = alpha[i].unsqueeze(-1) + R_c[i] @ R_cbt_bc[i].unsqueeze(-1) - R_cbt_bc[i].unsqueeze(-1) - 0.5 * gravities[i].unsqueeze(-1) * dt * dt + velocities[i].unsqueeze(-1) * dt # [B, 3, 1]
        
        # NOTE: trans is re-scaled by 5.4, but gravities and velocities are still the original scale
        trans = trans / trans_scale_factor
                
        T_mat = torch.cat([rot, trans], dim=2) # [B, 3, 4]
        fill = T_mat.new_zeros([T_mat.shape[0], 1, 4]) # [B, 1, 4]
        fill[:, :, -1] = 1
        T_mat = torch.cat([T_mat, fill], dim=1) # [B, 4, 4]
        T_c.append(T_mat) # [B, 4, 4]
        T_c_inv.append(T_mat.inverse()) # [B, 4, 4]
    
    # NOTE: the indices in poses/poses_inv are different from T_c/T_c_inv
    # -> poses: [from 0 to -1, from 0 to 1]
    # -> poses_inv: [from -1 to 0, from 1 to 0]
    poses = [T_c[0], T_c_inv[1]]
    poses_inv = [T_c_inv[0], T_c[1]]
    return poses, poses_inv


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        
        self.use_imu_warp = self.opt.imu_warp_weight > 0
        self.use_imu_consistency = self.opt.imu_consistency_weight > 0

        self.use_imu_l2 = self.opt.imu_l2_weight > 0
        if self.use_imu_l2:
            # Check we do not use imu_warp/consistency losses
            # assert not self.use_imu_warp
            # assert not self.use_imu_consistency
            self.loss_imu_l2 = torch.nn.MSELoss()

        self.use_imu = self.use_imu_warp or self.use_imu_consistency or self.opt.use_ekf or self.use_imu_l2
        self.compute_imu_warp = self.use_imu_warp or self.use_imu_consistency or self.opt.use_ekf 
        self.ekf_enabled = False

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        assert self.use_pose_net
        assert self.opt.pose_model_type == "separate_resnet"
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
    
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())
        
        if self.opt.velo_weight > 0: assert self.use_imu 
        if self.opt.gravity_weight > 0: assert self.use_imu

        if self.use_imu:
            # Initial EKF module
            if self.opt.use_ekf:
                self.models["ekf_model"] = EKFModel(
                    train_init_covar = self.opt.train_init_covar, 
                    train_imu_noise_covar = self.opt.train_imu_noise_covar,
                    vis_covar_use_fixed = self.opt.vis_covar_use_fixed,
                    trans_scale_factor = self.opt.trans_scale_factor,
                    naive_vis_covar = self.opt.naive_vis_covar
                )
                self.models["ekf_model"].to(self.device)
                self.parameters_to_train += list(self.models["ekf_model"].parameters())
                
            # Initialize velocity networks
            self.models["velo_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers_imu,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            
            self.models["velo_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["velo_encoder"].parameters())
            
            self.models["velo"] = networks.VeloDecoder(
                self.models["velo_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=1)
            
            self.models["velo"].to(self.device)
            self.parameters_to_train += list(self.models["velo"].parameters())
            
            # Initialize gravity networks
            self.models["gravity_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers_imu,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["gravity_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["gravity_encoder"].parameters())
            
            self.models["gravity"] = networks.GravityDecoder(
                self.models["gravity_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=1)
            
            self.models["gravity"].to(self.device)
            self.parameters_to_train += list(self.models["gravity"].parameters())
            
            if self.opt.gravity_weight > 0:
                # From 9.81 to 9.808679801065017 (More exact)
                self.g_enu = torch.Tensor([0, 0, 9.808679801065017])
                self.g_enu = torch.nn.Parameter(self.g_enu, requires_grad=False)
                self.g_enu.to(self.device)

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"
            
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        resume_type = None 
        if self.opt.resume_imu: resume_type = "imu"
        if self.opt.resume_gravity: resume_type = "gravity"
        if self.opt.resume_velo: resume_type = "velo"
        if resume_type is not None:
            assert self.opt.ekf_warming_epochs == 0
            assert self.opt.resume_epochs in [5, 10, 15]
            self.resume_model(resume_type, self.opt.resume_epochs)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        # The corrupted imu data are only used for training 
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, 
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, use_imu=self.use_imu, 
            use_ekf=self.opt.use_ekf, 
            k_imu_clip=self.opt.k_imu_clip, is_train=True, img_ext=img_ext, 
            imu_filename = self.opt.imu_filename, 
            img_noise_type = self.opt.img_noise_type,
            img_noise_brightness = self.opt.img_noise_brightness,
            img_noise_contrast = self.opt.img_noise_contrast,
            img_noise_saturation = self.opt.img_noise_saturation,
            img_noise_hue = self.opt.img_noise_hue,
            img_noise_gamma = self.opt.img_noise_gamma,
            img_noise_gain = self.opt.img_noise_gain,
            img_mask_num = self.opt.img_mask_num,
            img_mask_size = self.opt.img_mask_size,
            avoid_quat_check = self.opt.avoid_quat_check
        )        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        # Always use clean imu data for validation
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, 
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, use_imu=self.use_imu, 
            use_ekf=self.opt.use_ekf, 
            k_imu_clip=self.opt.k_imu_clip, 
            is_train=False, img_ext=img_ext, 
            imu_filename="matched_oxts.txt",
            avoid_quat_check = self.opt.avoid_quat_check)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        
        # num_train_samples are the ones after filtering based on imu
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()


    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            # Pretraining self.opt.ekf_warming_epochs without ekf 
            if self.opt.use_ekf and self.epoch >= self.opt.ekf_warming_epochs:
                self.ekf_enabled = True
            self.run_epoch()
            if (self.epoch >= self.opt.first_save_epoch) and ((self.epoch + 1) % self.opt.save_frequency == 0):
                self.save_model()


    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("========================================")
        print("Training epoch: {}...".format(self.epoch))
        print("=> ekf_enabled: {}".format(self.ekf_enabled))
        self.set_train()

        # print the learning rate of current epoch
        optim_state = self.model_optimizer.state_dict()["param_groups"][0]
        print("=> optim initial_lr: {}".format(optim_state["initial_lr"]))
        print("=> optim lr: {}".format(optim_state["lr"]))
        print("=> lr_scheduler last_lr: {}".format(self.model_lr_scheduler.get_last_lr()))

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["final_loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                # only logging the loss without imu warping losses for comparison
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key in [("preint_imu", -1, 0), ("preint_imu", 0, 1)]:
                for pkey, pipt in inputs[key].items():
                    inputs[key][pkey] = pipt.to(self.device).type(torch.float32)
            else:
                inputs[key] = ipt.to(self.device).type(torch.float32)

        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        
        if self.use_imu:
            self.predict_imu_poses(inputs, outputs, self.opt.use_ekf)

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, self.use_imu_warp, self.use_imu_consistency)
        
        losses["final_loss"] = losses["loss"]
        if self.use_imu_warp:
            losses["final_loss"] += self.opt.imu_warp_weight * losses["loss_imu_warp"]
        if self.use_imu_consistency:
            losses["final_loss"] += self.opt.imu_consistency_weight * losses["loss_imu_consistency"]

        if self.use_imu_l2:
            losses["final_loss"] += self.opt.imu_l2_weight * losses["loss_imu_l2"]

        if self.opt.velo_weight > 0:
            losses["final_loss"] += self.opt.velo_weight * losses["loss_velo"]
        if self.opt.gravity_weight > 0:
            losses["final_loss"] += self.opt.gravity_weight * losses["loss_gravity"]

        if ("ekf_v", 0) in outputs.keys():
            if self.opt.ekf_velo_weight > 0:
                losses["final_loss"] += self.opt.ekf_velo_weight * losses["loss_ekf_velo"]
            if self.opt.ekf_gravity_weight > 0:
                losses["final_loss"] += self.opt.ekf_gravity_weight * losses["loss_ekf_gravity"]

        return outputs, losses
    
    
    def predict_imu_poses(self, inputs, outputs, use_ekf=False):
        """Predict imu poses from imu preintegrations
        -> Perform velocity and gravity prediction inside
        -> outputs may be used if we later decide to try shared_encoder
        """
        alpha, R_c, R_cbt_bc, delta_t, gravities, velocities = [], [], [], [], [], []
        
        ## Use preintegration values without EKF
        # From 0 to -1 / From 1 to 0
        for key in [("preint_imu", -1, 0), ("preint_imu", 0, 1)]:
            alpha.append(inputs[key]["alpha"])
            R_c.append(inputs[key]["R_c"])
            R_cbt_bc.append(inputs[key]["R_cbt_bc"])
            delta_t.append(inputs[key]["delta_t"])
    
        pair_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in [-1, 1]:
             # To maintain ordering we always pass frames in temporal order
             # [-1, 0] / [0, 1] and predict velocity and gravity at -1 / 0
            if f_i < 0:
                pair_inputs = [pair_feats[f_i], pair_feats[0]] # [-1, 0]
            else:
                pair_inputs = [pair_feats[0], pair_feats[f_i]] # [0, 1]
            
            velo_inputs = [self.models["velo_encoder"](torch.cat(pair_inputs, 1))]
            gravity_inputs = [self.models["gravity_encoder"](torch.cat(pair_inputs, 1))]
            
            if self.opt.predict_velo_residue:
                if f_i == -1: 
                    base_velo = outputs[("cam_T_cam", 0, -1)][:, :3, 3] # from 0 to -1
                if f_i == 1:
                    base_velo = outputs[("cam_T_cam_inv", 0, 1)][:, :3, 3] # from 1 to 0
                velocity = base_velo + self.models["velo"](velo_inputs)
            else:
                velocity = self.models["velo"](velo_inputs)
                
            gravity = self.models["gravity"](gravity_inputs)
            
            velocities.append(velocity)
            gravities.append(gravity)
            if f_i == -1:
                outputs[("velocity", -1)] = velocity 
                outputs[("gravity", -1)] = gravity 
            if f_i == 1:
                outputs[("velocity", 0)] = velocity 
                outputs[("gravity", 0)] = gravity
        
        # Preintegrated poses (Note translation is at the /=5.4 scale w.r.t. 0.1m baseline )
        # poses:     [from 0 to -1, from 0 to 1]
        # poses_inv: [from -1 to 0, from 1 to 0]
        poses, poses_inv = compute_imu_pose_with_inv(alpha, R_c, R_cbt_bc, delta_t, gravities, velocities, self.opt.trans_scale_factor)
                
        ## EKF pipeline, from 0 to -1, and from 1 to 0
        # NOTE: We do EKF at the original translation scale rather than /= 5.4!
        # -> EKF propagation to obtain IMU error states
        # -> EKF update to fuse IMU and vision pose predictions
        if use_ekf and self.ekf_enabled:
            dts_full = [inputs[("preint_imu", -1, 0)]["dts_full"], 
                        inputs[("preint_imu", 0, 1)]["dts_full"]]
            wa_xyz_full = [inputs[("preint_imu", -1, 0)]["wa_xyz_full"], 
                           inputs[("preint_imu", 0, 1)]["wa_xyz_full"]]
            R_ckbt_full = [inputs[("preint_imu", -1, 0)]["R_ckbt_full"], 
                           inputs[("preint_imu", 0, 1)]["R_ckbt_full"]]
            H0_full = [inputs[("preint_imu", -1, 0)]["J_l_inv_neg_R_cb"], 
                       inputs[("preint_imu", 0, 1)]["J_l_inv_neg_R_cb"]]
            H1_full = [inputs[("preint_imu", -1, 0)]["R_cb_p_bc_wedge_neg"], 
                       inputs[("preint_imu", 0, 1)]["R_cb_p_bc_wedge_neg"]]
            
            ## NOTE: All translations in EKF are at original scale, rather than /=5.4!!
            #    * Thus we need to *= 5.4 here to get the originally scaled translation
            # preintegrated IMU poses from c_k+1 to c_k: [from 0 to -1, from 1 to 0]
            preimu_rot_full = [inputs[("preint_imu", -1, 0)]["phi_c"], 
                               inputs[("preint_imu", 0, 1)]["phi_c"]]
            preimu_trans_full = [poses[0][:, 0:3, 3] * self.opt.trans_scale_factor, 
                                 poses_inv[1][:, 0:3, 3] * self.opt.trans_scale_factor] 
            
            ## NOTE: CNN predicted translations and std have scale up to a baseline of 0.1m
            #    * Thus we need to account for the scale to meet EKF orignal scale requirment
            #    * We do this in ekf.py since the std here is not the real std!
            # camera predicted poses from c_k+1 to c_k: [from 0 to -1, from 1 to 0]
            vis_rot_full = [outputs[("axisangle", 0, -1)][:, 0].squeeze(1),
                            outputs[("axisangle", 0, 1)][:, 0].squeeze(1)]
            vis_trans_full = [outputs[("translation", 0, -1)][:, 0].squeeze(1),
                              outputs[("translation", 0, 1)][:, 0].squeeze(1)]
            vis_rot_std_full = [outputs[("std_axisangle", 0, -1)][:, 0].squeeze(1),
                                  outputs[("std_axisangle", 0, 1)][:, 0].squeeze(1)]
            vis_trans_std_full = [outputs[("std_translation", 0, -1)][:, 0].squeeze(1),
                                    outputs[("std_translation", 0, 1)][:, 0].squeeze(1)]
            
            # ekf_phi/t_c: [from 0 to -1, from 1 to 0]
            # ekf_v/g_ck: expressed at [frame -1, frame 0]
            ekf_phi_c, ekf_t_c, ekf_v_ck, ekf_g_ck, vis_covar, imu_error_covar = self.models["ekf_model"].forward(
                    dts_full = dts_full, 
                    wa_xyz_full = wa_xyz_full, 
                    R_ckbt_full = R_ckbt_full, 
                    velocities_full = velocities, 
                    gravities_full = gravities, 
                    H0_full = H0_full, 
                    H1_full = H1_full, 
                    preimu_rot_full = preimu_rot_full, 
                    preimu_trans_full = preimu_trans_full,
                    vis_rot_full = vis_rot_full,
                    vis_trans_full = vis_trans_full,
                    vis_rot_std_full = vis_rot_std_full,
                    vis_trans_std_full = vis_trans_std_full
                )

            ## EKF operates on the original scale -> Need to /=5.4 w.r.t. 0.1m baseline
            ekf_t_c[0] /= self.opt.trans_scale_factor
            ekf_t_c[1] /= self.opt.trans_scale_factor
        
            # ("imu_T", 0, -1): from 0 to -1; ("imu_T", 0, 1):  from 0 to 1
            outputs[("imu_T", 0, -1)] = transformation_from_parameters(
                        ekf_phi_c[0].unsqueeze(1), ekf_t_c[0].unsqueeze(1), invert=False)
            outputs[("imu_T", 0, 1)] = transformation_from_parameters(
                        ekf_phi_c[1].unsqueeze(1), ekf_t_c[1].unsqueeze(1), invert=True)
            outputs[("ekf_v", -1)], outputs[("ekf_v", 0)] = ekf_v_ck[0], ekf_v_ck[1]
            outputs[("ekf_g", -1)], outputs[("ekf_g", 0)] = ekf_g_ck[0], ekf_g_ck[1]
            outputs["vis_covar"] = vis_covar.mean()
            outputs["vis_covar_abs"] = vis_covar.abs().mean()
            outputs["imu_error_covar"] = imu_error_covar.mean()
            outputs["imu_error_covar_abs"] = imu_error_covar.abs().mean()
        else:
            # ("imu_T", 0, -1): from 0 to -1; ("imu_T", 0, 1):  from 0 to 1
            outputs[("imu_T", 0, -1)], outputs[("imu_T", 0, 1)] = poses[0], poses[1]
            # outputs[("ekf_v", -1)], outputs[("ekf_v", 0)] = None, None
            # outputs[("ekf_g", -1)], outputs[("ekf_g", 0)] = None, None
            
        
        
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    # [B, 2, 1, 3] & [B, 2, 1, 3]
                    # axisangle is so3 (log mapping) of R (SO3)
                    # NOTE: we reserve the pose w.r.t. monodepth2!
                    #   * ("(covar)_axisangle"/"translation", 0, -1) from 0 to -1
                    #   * ("(covar)_axisangle"/"translation", 0, 1) from 1 to 0
                    axisangle, translation, std_axisangle, std_translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs["std_axisangle", 0, f_i] = std_axisangle 
                    outputs["std_translation", 0, f_i] = std_translation
                    
                    ## Get sampled_axisangle/translation from gaussian distribution
                    sampled_axisangle = axisangle[:, 0] # [B, 1, 3]
                    sampled_translation = translation[:, 0] # [B, 1, 3]
                    if self.opt.sample_vis_pose and self.ekf_enabled and self.opt.use_ekf and not self.opt.vis_covar_use_fixed:
                        vis_std = torch.cat([std_axisangle[:, 0], std_translation[:, 0]], dim=2).squeeze(1) # [B, 6]
                        vis_covar = proc_vis_covar(self.models["ekf_model"].get_par(), 
                                                   vis_std, 
                                                   vis_covar_use_fixed = False, 
                                                   return_diag = True, 
                                                   naive_vis_covar = self.opt.naive_vis_covar
                                                )
                        proc_std_axisangle = vis_covar[:, 0:3].sqrt().unsqueeze(1) # [B, 1, 3]
                        proc_std_translation = vis_covar[:, 3:6].sqrt().unsqueeze(1) # [B, 1, 3]
                        sampled_axisangle += proc_std_axisangle * torch.randn_like(sampled_axisangle)
                        sampled_translation += proc_std_translation * torch.randn_like(sampled_translation)
                        
                    # Invert the matrix if the frame id is positive rather than negative in monodepth2!
                    #   * ("cam_T_cam", 0, -1) from 0 to -1
                    #   * ("cam_T_cam", 0, 1) from 0 to 1
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        sampled_axisangle, sampled_translation, invert=(f_i > 0))
                    
                    #   * ("cam_T_cam_inv", 0, -1) from -1 to 0
                    #   * ("cam_T_cam_inv", 0, 1) from 1 to 0
                    if self.opt.predict_velo_residue:
                        outputs[("cam_T_cam_inv", 0, f_i)] = transformation_from_parameters(
                            sampled_axisangle, sampled_translation, invert=(f_i < 0))

        return outputs


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    # Fixed at 0.1 (A scale factor 5.4 is applied for translation)
                    T = inputs["stereo_T"] 
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    raise NotImplementedError("Logic not check in our primary motion-monodepth2")
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], invert=frame_id > 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                
                if self.compute_imu_warp and frame_id != "s":
                    T = outputs[("imu_T", 0, frame_id)]
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                    
                    outputs[("sample_imu", frame_id, scale)] = pix_coords
                    outputs[("color_imu", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample_imu", frame_id, scale)],
                        padding_mode="border")
                    

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    
    def compute_losses(self, inputs, outputs, use_imu_warp, use_imu_consistency):
        """Compute the reprojection and smoothness losses for a minibatch
        Return: 
            losses: dict, and the keys:
                -> "loss", "loss/{}".format(scale)
                -> if use_imu_warp: "loss_imu_warp", "loss_imu_warp/{}".format(scale)
                -> if use_imu_consistency: "loss_imu_consistency", "loss_imu_consistency/{}".format(scale)
        """
        losses = {}
        total_loss = 0
        if use_imu_warp:
            total_loss_imu_warp = 0
        if use_imu_consistency:
            total_loss_imu_consistency = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            if use_imu_warp:
                loss_imu_warp = 0
                imu_reprojection_losses = []
            
            if use_imu_consistency:
                loss_imu_consistency = 0
                imu_consistency_losses = []
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                
                if use_imu_warp and frame_id in [-1, 1]:
                    pred_imu = outputs[("color_imu", frame_id, scale)]
                    imu_reprojection_losses.append(self.compute_reprojection_loss(pred_imu, target))
                
                # Sen: NOTE: we don't need mask for pred_imu and pred!
                #   -> They can be wrong but it is ok if they are both wrong in the same way!
                if use_imu_consistency and frame_id in [-1, 1]:
                    pred_imu = outputs[("color_imu", frame_id, scale)]
                    if self.opt.no_grad_imu_consistency == "none":
                        imu_consistency_losses.append(self.compute_reprojection_loss(pred_imu, pred))
                    if self.opt.no_grad_imu_consistency == "network":
                        imu_consistency_losses.append(self.compute_reprojection_loss(pred_imu, pred.detach()))
                    if self.opt.no_grad_imu_consistency == "imu":
                        imu_consistency_losses.append(self.compute_reprojection_loss(pred_imu.detach(), pred))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            if use_imu_warp:
                imu_reprojection_losses = torch.cat(imu_reprojection_losses, 1)
            if use_imu_consistency:
                imu_consistency_losses = torch.cat(imu_consistency_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                if use_imu_warp:
                    imu_reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                if use_imu_warp:
                    imu_reprojection_loss = imu_reprojection_losses.mean(1, keepdim=True)
                if use_imu_consistency:
                    imu_consistency_loss = imu_consistency_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                if use_imu_warp:
                    imu_reprojection_loss = imu_reprojection_losses
                if use_imu_consistency:
                    imu_consistency_loss = imu_consistency_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                if use_imu_warp:
                    combined_imu = torch.cat((identity_reprojection_loss, imu_reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
                if use_imu_warp:
                    combined_imu = imu_reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
                if use_imu_warp:
                    to_optimise_imu = combined_imu
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                if use_imu_warp:
                    to_optimise_imu, idxs_imu = torch.min(combined_imu, dim=1)
            
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            if use_imu_warp:
                loss_imu_warp += to_optimise_imu.mean()
                
            # NOTE: we don't need mask for pred_imu and pred!
            if use_imu_consistency:
                loss_imu_consistency += imu_consistency_loss.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            if use_imu_warp:
                total_loss_imu_warp += loss_imu_warp 
                losses["loss_imu_warp/{}".format(scale)] = loss_imu_warp
            if use_imu_consistency:
                total_loss_imu_consistency += loss_imu_consistency
                losses["loss_imu_consistency/{}".format(scale)] = loss_imu_consistency

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        if use_imu_warp:
            total_loss_imu_warp /= self.num_scales
            losses["loss_imu_warp"] = total_loss_imu_warp
        if use_imu_consistency:
            total_loss_imu_consistency /= self.num_scales
            losses["loss_imu_consistency"] = total_loss_imu_consistency

        if self.use_imu_l2:
            losses["loss_imu_l2"] = 0
            for idx in [-1, 1]:
                losses["loss_imu_l2"] += self.loss_imu_l2(outputs[("cam_T_cam", 0, idx)], outputs[("imu_T", 0, idx)])
        

        if self.opt.display_velo_scale:
            # Only used for logging, not training
            losses["velo_norm_diff"] = 0
            losses["velo_norm_ori"] = 0
            for idx in [-1, 0]:
                losses["velo_norm_diff"] += (torch.norm(outputs[("velocity", idx)], dim=1) - inputs[("preint_imu", idx, idx+1)]["v_norm"]).abs().mean()
                losses["velo_norm_ori"] += inputs[("preint_imu", idx, idx+1)]["v_norm"].abs().mean()
            losses["velo_norm_diff"] /= 2.
            losses["velo_norm_ori"] /= 2.
        
        if self.opt.velo_weight > 0:
            losses["loss_velo"] = 0
            for idx in [-1, 0]:
                losses["loss_velo"] += (torch.norm(outputs[("velocity", idx)], dim=1) - inputs[("preint_imu", idx, idx+1)]["v_norm"]).abs().mean()
        
        if self.opt.gravity_weight > 0:
            losses["loss_gravity"] = 0
            for idx in [-1, 0]:
                losses["loss_gravity"] += (torch.norm(outputs[("gravity", idx)], dim=1) - torch.norm(self.g_enu)).abs().mean()
        
        ## EKF updated velocity and gravity L2 norm losses
        if ("ekf_v", 0) in outputs.keys():
            if self.opt.ekf_velo_weight > 0:
                losses["loss_ekf_velo"] = 0
                for idx in [-1, 0]:
                    losses["loss_ekf_velo"] += (torch.norm(outputs[("velocity", idx)] - outputs[("ekf_v", idx)])).abs().mean()
            
            if self.opt.ekf_gravity_weight > 0:
                losses["loss_ekf_gravity"] = 0
                for idx in [-1, 0]:
                    losses["loss_ekf_gravity"] += (torch.norm(outputs[("gravity", idx)] - outputs[("ekf_g", idx)])).abs().mean()
            
            # Only used for logging
            for k in ["vis_covar", "vis_covar_abs", "imu_error_covar", "imu_error_covar_abs"]:
                losses[k] = outputs[k]
            
        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

            
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

        
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                    
                    if self.compute_imu_warp and s == 0 and frame_id in [-1, 1]:
                        writer.add_image(
                            "color_imu_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_imu", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

                    
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
        
        src_path = os.path.join(self.log_path, "models/src")
        if os.path.exists(src_path):
            shutil.rmtree(src_path)
        os.mkdir(src_path)
        os.system("cp *.py {}".format(src_path))
        

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    

    def resume_model(self, resume_type, resume_epochs):
        """Load model(s) from disk
        NOTE: Only load model weights, but not optimizer states!

        Input
            resume_type: STR
                "imu": imu-only model
                "gravity": imu+gravity model
                "velo": imu+gravity+velo model
            resume_epochs: INT
        """
        RESUME_FOLDERS ={
            "imu": "ckp/resume_imu",
            "gravity": "ckp/resume_gravity",
            "velo": "ckp/resume_velo"
        }

        # epoch - 1 since the epochs are indexed from 0
        resume_folder = os.path.join(RESUME_FOLDERS[resume_type], "weights_{}".format(resume_epochs-1))

        print("=====================================")
        print("=> Resuming model weights from {}".format(resume_folder))

        assert os.path.isdir(resume_folder),  "Cannot find folder {}".format(resume_folder)

        for n in self.models.keys():
            path = os.path.join(resume_folder, "{}.pth".format(n))
            if os.path.isfile(path):
                print("Loading {} weights...".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            else:
                print("* Warning: Cannot find {} weights...".format(n))

        print("=> Resuming finished!")
        print("=====================================")
        
