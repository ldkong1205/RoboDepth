# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import pdb
import random
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from liegroups.numpy import SO3 as SO3_np
from collections import Counter


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
        
def quat_mul(q0, q1):
    """Multiply two quaternions in np.array
    -> Input: numpy arrays
    -> Output: liegroups.numpy.SO3 object
    """
    w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1 
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1 
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    q0q1 = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])
    return SO3_np.from_quaternion(q0q1)


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_imu=False,
                 use_ekf=False,
                 k_imu_clip=5,
                 is_train=False,
                 img_ext=".png",
                 imu_filename="matched_oxts.txt",
                 img_noise_type="clean",
                 img_noise_brightness=0.5,
                 img_noise_contrast=0.5,
                 img_noise_saturation=0.5,
                 img_noise_hue=0.5,
                 img_noise_gamma=0.5,
                 img_noise_gain=0.5,
                 img_mask_num=1,
                 img_mask_size=50,
                 avoid_quat_check=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.imu_filename = imu_filename
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.use_imu = use_imu
        self.use_ekf = use_ekf 
        self.k_imu_clip = k_imu_clip # previous 12 x k IMU records will be fed into CNN
        self.img_noise_type = img_noise_type
        self.avoid_quat_check = avoid_quat_check
        
        # We only support 0, -1, 1 (, s) when using imu
        if self.use_imu:
            assert -1 in self.frame_idxs 
            assert 1 in self.frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            
        
        # "bcsh": brightness/contrast/saturation/hue noises, "gg": gamma/gain noises
        self.img_noiser = None
        if self.img_noise_type == "bcsh":
            self.img_noiser = transforms.ColorJitter(
                brightness = img_noise_brightness,
                contrast = img_noise_contrast,
                saturation = img_noise_saturation,
                hue = img_noise_hue 
            )
        
        if self.img_noise_type == "gg":
            self.gamma_range = img_noise_gamma # e.g. 0.5
            self.gain_range = img_noise_gain   # e.g. 0.5
        
        if self.img_noise_type == "mask":
            self.mask_num = img_mask_num
            self.mask_size = img_mask_size
            self.mask = Image.new("RGB", (self.mask_size, self.mask_size), (0, 0, 0))
            
            
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        
        # preloading imu data
        if self.use_imu:
            self.seqs, self.scenes = self.parse_seqs()
            self.T_IMU_CAM = self.get_imu_camera_T()
            self.preint_imus, self.raw_imus = self.preintegrate_imu()

            print("==============================")
            print("=> Num of filenames before filtering based on IMU nan: {}".format(len(self.filenames)))
            self.filenames = self.filter_files()
            print("=> Num of filenames After filtering based on IMU nan: {}".format(len(self.filenames)))
            # self.imus = self.load_imu()


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        
        Sen -> Now we can apply different augmentation to each image to introduce image noises 
        """
        for k in list(inputs):
            # frame = inputs[k]
            if "color" in k:
                n, im, i = k
                if self.img_noise_type == "bcsh":
                    inputs[(n, im, -1)] = self.img_noiser(inputs[(n, im, -1)])
                
                if self.img_noise_type == "gg":
                    aug_gamma = random.uniform(1 - self.gamma_range, 1 + self.gamma_range)
                    aug_gain = random.uniform(1 - self.gain_range, 1 + self.gain_range)
                    inputs[(n, im, -1)] = transforms.functional.adjust_gamma(inputs[(n, im, -1)], gamma=aug_gamma, gain=aug_gain)
                
                if self.img_noise_type == "mask":
                    w, h = inputs[(n, im, -1)].size
                    for _ in range(self.mask_num):
                        wshift = random.randint(0, w - self.mask_size)
                        hshift = random.randint(0, h - self.mask_size)
                        inputs[(n, im, -1)].paste(self.mask, (wshift, hshift))
                    
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])                    

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


    def __len__(self):
        return len(self.filenames)


    def filter_files(self):
        """Filter out self.filenames that don't have preint_imu for [0, -1, 1]
        """
        filenames = []
        for filename in self.filenames:
            line = filename.split()
            assert len(line) == 3
            folder = line[0]
            frame_index = int(line[1])
            side = line[2]
            assert side in ['l', 'r']
            cam_idx = "02" if side == 'l' else "03"
            # Check -1 to 0 and 0 to 1
            # e.g. "2011_09_26/2011_09_26_drive_0022_sync/02/0-1"
            if self.preint_imus["{}/{}/{}-{}".format(folder, cam_idx, frame_index-1, frame_index)] == "nan":
                continue  
            if self.preint_imus["{}/{}/{}-{}".format(folder, cam_idx, frame_index, frame_index+1)] == "nan":
                continue 
            filenames.append(filename)
        return filenames
            

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            
            curr_width = self.width // (2 ** scale)
            curr_height = self.height // (2 ** scale)
            K[0, :] *= curr_width
            K[1, :] *= curr_height
            
            # Sen: Now adjust K while keeping pose unchanged
            if do_flip:
                K[0, 2] = curr_width - K[0, 2]

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            
            # NOTE: Now change K rather than pose!
            # baseline_sign = -1 if do_flip else 1
            baseline_sign = 1
            
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        
        ## Get IMU preintegrations and now only support [0, -1, 1 (, 's')]
        if self.use_imu:
            assert self.frame_idxs[:3] == [0, -1, 1]
            assert side in ['l', 'r']
            cam_idx = "02" if side == 'l' else "03"
            
            # From 0 to -1 and from 1 to 0 (NOTE: the direction may be different from network parts here)
            # keys: "wa_xyz", "imu_time",
            #   "delta_t", "alpha", "beta", 
            #   "R_c", "R_c_inv", "R_cbt_bc", "v_norm"
            #   "R_ckbt" (list of all R_{c_kb_t} where t ranges from t_k to t_k+1)
            inputs[("preint_imu", -1, 0)] = copy.deepcopy(self.preint_imus["{}/{}/{}-{}".format(folder, cam_idx, frame_index-1, frame_index)])
            inputs[("preint_imu", 0, 1)] = copy.deepcopy(self.preint_imus["{}/{}/{}-{}".format(folder, cam_idx, frame_index, frame_index+1)])
            
            # For do_flip: Now change K rather than pose!
            for imu_pair in [("preint_imu", -1, 0), ("preint_imu", 0, 1)]:
                for key in inputs[imu_pair].keys():
                    inputs[imu_pair][key] = torch.from_numpy(np.array(inputs[imu_pair][key]))
        
        ## Get 12 x k IMU records for predicting IMU noise parameters
        # NOTE: Now for each interval we pad to 12 (fixed) to allow batch operation
        # NOTE: Now we also feed time intervals to the denoising network!
        if self.use_ekf:
            assert self.frame_idxs[:3] == [0, -1, 1]
            assert side in ['l', 'r']
            cam_idx = "02" if side == 'l' else "03"
            def get_key(folder, cam_idx, frame_index, offset):
                # print("=> processing ({}, {})".format(frame_index+offset, frame_index+offset+1))
                return "{}/{}/{}-{}".format(folder, cam_idx, frame_index+offset, frame_index+offset+1)

            imu_list = []
            imu_time_list = []
            for i in range(self.k_imu_clip):   
                tmp_key = get_key(folder, cam_idx, frame_index, i - self.k_imu_clip + 1)
                if tmp_key in self.raw_imus.keys():
                    tmp_imu = copy.deepcopy(self.raw_imus[tmp_key])
                    if tmp_imu in ["nan", "none"]:
                        imu_list.append(np.zeros((12, 6)))
                        imu_time_list.append(np.zeros((12)))
                    else:
                        wa_xyz_ = tmp_imu["wa_xyz"] # [12, 6] or [11, 6]
                        imu_time_ = tmp_imu["imu_time"] # [12] or [11]
                        if wa_xyz_.shape[0] == 11:
                            wa_xyz_ = np.concatenate([wa_xyz_, np.expand_dims(wa_xyz_[-1, :], 0)], 0)
                            imu_time_ = np.concatenate([imu_time_, np.expand_dims(imu_time_[-1], 0)], 0)
                        imu_list.append(wa_xyz_)
                        imu_time_list.append(imu_time_)
                else:
                    imu_list.append(np.zeros((12, 6)))
                    imu_time_list.append(np.zeros((12)))
                    
            # deal with the overlapped imu records at current end and next beginning
            imus = np.concatenate(imu_list, axis=0)
            imu_times = np.concatenate(imu_time_list, axis=0)
            assert imus.shape[0] == imu_times.shape[0]
            assert imus.shape[0] == self.k_imu_clip * 12
            
            inputs["imus"] = torch.from_numpy(imus)
            
            # imu_times starts from 0. s.t. to be fed into CNN
            imu_times[imu_times > 0] -= imu_times[0]
            inputs["imu_times"] = torch.from_numpy(imu_times)
        
        return inputs
    

    def proc_imu(self, imu, imu_time):
        def get_second(t):
            t = [float(x) for x in t.split(' ')[1].split(':')] 
            return t[0] * 3600 + t[1] * 60 + t[2]
        
        # imu: np.array(12, 30), dtype=float64
        # imu_time: np.array(12, ), dtype=float64
        imu = np.array([x.strip().split(' ') for x in imu.split('|')], dtype=float)
        imu_time = np.array([get_second(x.strip()) for x in imu_time.split('|')])
        assert imu.shape[0] == imu_time.shape[0]
        
        # calculate the preintegration terms
        w_xyz, w_flu = imu[:, 17:20], imu[:, 20:23]
        a_xyz, a_flu = imu[:, 11:14], imu[:, 14:17]
        v_ne, v_flu = imu[:, 6:8], imu[:, 8:11] # Used for velocity magnitude
        pos_acc, vel_acc = imu[:, 23], imu[:, 24]
        
        # w_xyz: [11, 3], a_xyz: [11, 3], wa_xyz: [11, 6]
        wa_xyz = np.concatenate([w_xyz, a_xyz], axis=1)
        
        delta_t, alpha, beta = 0, 0, 0
        q = np.identity(3) if self.avoid_quat_check else SO3_np.identity()
        linearized_ba, linearized_bg = 0, 0
        
        # R_b: list of all preintegrated rotations R_bk_bt, t=t_k, t_k + 1, ..., t_k+1
        # dts: list of all delta_t, length is len(R_b) - 1
        R_b = []
        dts = []
        if self.avoid_quat_check:
            R_b.append(q)
        else:
            R_b.append(q.mat)
        
        for idx in range(imu.shape[0]):
            if idx == 0: continue  
            dt = imu_time[idx] - imu_time[idx-1] # unit: s

            acc_0 = q @ (a_xyz[idx-1] - linearized_ba) if self.avoid_quat_check else q.mat @ (a_xyz[idx-1] - linearized_ba) 
            gyr = 0.5 * (w_xyz[idx-1] + w_xyz[idx]) - linearized_bg 
            
            # NOTE: There are two ways to compute delta_q
            if self.avoid_quat_check:
                q = q @ SO3_np.exp(dt * gyr).mat
            else:
                q = quat_mul(q.to_quaternion(), np.array([1, gyr[0] * dt / 2, gyr[1] * dt / 2, gyr[2] * dt / 2]))
            
            acc_1 = q @ (a_xyz[idx] - linearized_ba) if self. avoid_quat_check else q.mat @ (a_xyz[idx] - linearized_ba)
            acc = 0.5 * (acc_0 + acc_1)
            alpha += acc * dt * dt 
            beta += acc * dt
            delta_t += dt
            
            dts.append(dt) # unit: s
            if self.avoid_quat_check:
                R_b.append(q)
            else:
                R_b.append(q.mat)

        # NOTE: We should only take the v_norm at the beginning, not accumulated!
        v_norm = np.linalg.norm(v_flu[0])
        
        return wa_xyz, imu, imu_time, delta_t, alpha, beta, R_b, v_norm, dts
    
    
    def preintegrate_imu(self):
        """Returns the IMU preintegration results from the dataset as a dictionary.
        NOTE: In this stage we save all imu preintegration in self.seqs
        """
        # key: e.g. "2011_09_26/2011_09_26_drive_0022_sync/02/0-1"
        preint_imus, raw_imus = dict(), dict()
        print("Preintegrating IMU data (Save raw wa_xyz also)...")
        for seq in tqdm(self.seqs):
            scene = seq.split('/')[0]
            # NOTE: Now we can use imu_filename other than "matched_oxts.txt" to allow corrupted imus
            file_imu = os.path.join(self.data_path, seq, "matched_oxts/{}".format(self.imu_filename))
            file_imu_time = os.path.join(self.data_path, seq, "matched_oxts/matched_timestamps.txt")
            imus, imus_time = [], [] 
            with open(file_imu, 'r') as f:
                print("=> Read IMU file: {}".format(file_imu))
                for line in f.readlines():
                    imus.append(line.strip())
            with open(file_imu_time, 'r') as f:
                for line in f.readlines():
                    imus_time.append(line.strip())
            
            for idx, (imu_, imu_time_) in enumerate(zip(imus, imus_time)):
                if imu_ == "nan" or len(imu_.split('|')) not in [11, 12]:
                    for cam_idx in ["02", "03"]:
                        key = "{}/{}/{}-{}".format(seq, cam_idx, idx, idx+1)
                        preint_imus[key] = "nan"
                        raw_imus[key] = "nan"
                    continue                
                
                # wa_xyz: [11/12, 6], imu: [11/12, 30], imu_time: [11/12], delta_t: 0.1 (scalar)
                # alpha: [3], beta: [3], R_b: [3, 3], v_norm: scalar, dts: [10/11]
                wa_xyz, imu, imu_time, delta_t, alpha, beta, R_b, v_norm, dts = self.proc_imu(imu_, imu_time_)
                
                # # NOTE: Now we last pad imu_time to length-12 always -> Later use: Check the last two times: If equal, remove the last
                # # wa_xyz: [12, 6], imu: [12, 30], imu_time: [12]
                # assert len(imu_time) in [11, 12]
                # if len(imu_time) == 11:
                #     imu_time = np.concatenate([imu_time, np.array([imu_time[-1]])])
                #     wa_xyz = np.concatenate([wa_xyz, np.expand_dims(wa_xyz[-1], 0)])
                #     imu = np.concatenate([imu, np.expand_dims(imu[-1], 0)])
                
                for cam_idx in ["02", "03"]:
                    T_key = "{}/{}".format(scene, cam_idx)
                    R_cb, t_cb = self.T_IMU_CAM[T_key]["R_cb"], self.T_IMU_CAM[T_key]["t_cb"]
                    R_bc, t_bc = self.T_IMU_CAM[T_key]["R_bc"], self.T_IMU_CAM[T_key]["t_bc"]
                    # 'alpha': [], # R_cb @ \alpha_{b_kb_{k+1}}
                    # 'beta': [], # R_cb @ \beta_{b_kb_{k+1}}
                    # 'R_c': [], # R_cb @ R_{b_kb_{k+1}} @ R_bc
                    # 'R_c_inv': [], # np.linalg.inv(R_c)
                    # 'R_cbt_bc': [], # R_cb @ t_bc
                    # 'v_norm': [], # np.linalg.norm(v_flu)
                    alpha = R_cb @ alpha 
                    beta = R_cb @ beta 
                    R_c = R_cb @ R_b[-1] @ R_bc 
                    R_c_inv = np.linalg.inv(R_c)
                    R_cbt_bc = R_cb @ t_bc
                    key = "{}/{}/{}-{}".format(seq, cam_idx, idx, idx+1)
                    preint_imus[key] = {
                        "delta_t": delta_t, 
                        "alpha": alpha, "beta": beta, 
                        "R_c": R_c, "R_c_inv": R_c_inv, 
                        "R_cbt_bc": R_cbt_bc, "v_norm": v_norm,
                        "R_cb": R_cb, "R_bc": R_bc, "t_bc": t_bc,
                        
                    }
                    raw_imus[key] = {
                        "wa_xyz": wa_xyz, # [11, 6]
                        "imu_time": imu_time,
                    }
                    
                    ## Used for the derivative of H in EKF update
                    # phi_c: the log (so3) of R_c 
                    # J_l_inv_neg_R_cb: J_l(-phi_c)^{-1} @ R_cb
                    # R_cb_p_bc_wedge_neg = -R_ckbk+1 @ p_bc_wedge
                    phi_c = SO3_np(R_c).log()
                    J_l_inv_neg_R_cb = SO3_np.inv_left_jacobian(-phi_c) @ R_cb
                    R_cb_p_bc_wedge_neg = - R_cb @ R_b[-1] @ SO3_np.wedge(t_bc)
                    preint_imus[key]["phi_c"] = phi_c
                    preint_imus[key]["J_l_inv_neg_R_cb"] = J_l_inv_neg_R_cb
                    preint_imus[key]["R_cb_p_bc_wedge_neg"] = R_cb_p_bc_wedge_neg
                    
                    
                    ## Used for F and G in EKF propagation
                    # R_ckbt: [], # list of R_cb @ R_{b_kb_t}} for all t from t_k to t_k+1
                    dts_full = np.array(dts)
                    wa_xyz_full = wa_xyz
                    R_ckbt_full = np.array([R_cb @ tmpR for tmpR in R_b])
                    
                    # For batch ops, the IMU length should be the same. But now 11 or 12
                    # Pad no-motion to the end to fix len at 12, but save the true length in "imu_len"
                    imu_len = R_ckbt_full.shape[0]
                    assert imu_len == wa_xyz.shape[0]
                    assert imu_len in [11, 12]
                    
                    if imu_len == 11:
                        dts_full = np.concatenate([dts_full, np.zeros(1)], axis=0)
                        wa_xyz_full = np.concatenate([wa_xyz_full, np.zeros((1, 6))], axis=0)
                        R_ckbt_full = np.concatenate([R_ckbt_full, np.expand_dims(R_ckbt_full[-1], axis=0)], axis=0) 
                    
                    preint_imus[key]["dts_full"] = np.array(dts_full)  # [11, 6]
                    preint_imus[key]["R_ckbt_full"] = np.array(R_ckbt_full) # [12, 3, 3]
                    preint_imus[key]["wa_xyz_full"] = np.array(wa_xyz_full) # [12, 6]
                    preint_imus[key]["imu_len"] = imu_len # 11 or 12  
                    
        return preint_imus, raw_imus


    def get_imu_camera_T(self):
        """Get the fixed transformation from IMU to camera for each sequence
        Return
            T_IMU_CAM: dict
                e.g. key: "2011_09_26/02"
                e.g. val: dict with key: "R_cb", "t_cb", "R_bc", t_bc" (numpy arrays) 
        """
        T_IMU_CAM = dict()
        # e.g. "2011_09_26"
        for scene in self.scenes:
            # Read IMU_2_CAM extrinsics
            file_imu2velo = os.path.join(self.data_path, scene, "calib_imu_to_velo.txt")
            file_velo2cam = os.path.join(self.data_path, scene, "calib_velo_to_cam.txt")
            file_cam2cam  = os.path.join(self.data_path, scene, "calib_cam_to_cam.txt")
            
            imu2velo = read_calib_file(file_imu2velo)
            velo2cam = read_calib_file(file_velo2cam)
            cam2cam = read_calib_file(file_cam2cam)
            
            velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
            imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
            cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))
            
            # NOTE: imu2cam_0 here is the transformation from imu to cam0
            # -> We still need one more step from cam0 to cam2/cam3
            # -> See Lee Clement's pykitti
            imu2cam_0 = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
            
            for cam_idx in ["02", "03"]:
                P_rect = np.reshape(cam2cam['P_rect_{}'.format(cam_idx)], (3, 4))            
                T_cam = np.eye(4) # The rectified extrinsics from cam0 to camN (02/03)
                T_cam[0, 3] = P_rect[0, 3] / P_rect[0, 0]
                T_cb = T_cam @ imu2cam_0  # From imu to cam2/3
                T_bc = np.linalg.inv(T_cb) # From cam2/3 to imu
                R_cb, t_cb = T_cb[0:3, 0:3], T_cb[0:3, 3]
                R_bc, t_bc = T_bc[0:3, 0:3], T_bc[0:3, 3]
                key = "{}/{}".format(scene, cam_idx)
                T_IMU_CAM[key] = {
                    "R_cb": R_cb, "t_cb": t_cb,
                    "R_bc": R_bc, "t_bc": t_bc
                }
        return T_IMU_CAM
            
    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def parse_seqs(self):
        raise NotImplementedError
    
    def load_imu(self):
        raise NotImplementedError
    
    

