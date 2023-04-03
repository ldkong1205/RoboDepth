"""
EKF propagation and update functions
* ekf_propagate()
* ekf_update()
"""
from __future__ import absolute_import, division, print_function

import pdb, os, time, sys
import numpy as np 
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import matmul as mm
from liegroups.torch import skew3_b, exp_SO3_b


class EKFParams:
    def __init__(self):
        self.init_covar_diag_sqrt = np.array([0, 0, 0, 0, 0, 0,  # C, r
                                              1e-2, 1e-2, 1e-2,  # v
                                              1e-4, 1e-4, 1e-4,  # g
                                              1e-8, 1e-8, 1e-8,  # bw
                                              1e-1, 1e-1, 1e-1])  # ba
        self.init_covar_diag_eps = 1e-12
        
        # self.exclude_resume_weights = ["imu_noise_covar_weights", "init_covar_diag_sqrt"]
        
        self.imu_noise_covar_diag = np.array([1e-7,  # w
                                              1e-7,  # bw
                                              1e-2,  # a
                                              1e-3])  # ba
        self.imu_noise_covar_beta = 4
        self.imu_noise_covar_gamma = 1

        
        self.vis_fixed_covar = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
        
        self.vis_covar_init_guess = 1e1
        self.vis_covar_beta = 3
        self.vis_covar_gamma = 1
        
        # account for trans_scale_factor
        trans_scale_factor_2 = 5.4 * 5.4
        self.vis_fixed_covar /= trans_scale_factor_2
        self.vis_covar_init_guess /= trans_scale_factor_2
        
        # error scale for covar loss, not really used,
        # but must be 1.0 for self.gaussian_pdf_loss = False
        self.vis_covar_scale = 1.0


def proc_vis_covar(par, vis_std, vis_covar_use_fixed, return_diag, naive_vis_covar):
    """[Processing visual measurement covariances]

    Args:
        par ([dict]): [the parameters used for defining vis_std]
        vis_std ([torch.Tensor]): [(2B, 6)] 
            (1) naive_vis_covar is True: The standard error of visual measurements
            (2) naive_vis_covar is False: The metric used in 10**(3*tanh(x))
        vis_covar_use_fixed ([bool]): [Whether use predefined or CNN predicted vis_std/covar]

    Returns:
        [vis_covar]: [The processed visual covariances]
    """
    vis_covar_scale = torch.ones(6, device=vis_std.device)
    vis_covar_scale[0:3] = vis_covar_scale[0:3] * par.vis_covar_scale
    if vis_covar_use_fixed:
        vis_covar_diag = torch.tensor(par.vis_fixed_covar, dtype=torch.float32, device=vis_std.device)
        vis_covar_diag = vis_covar_diag * vis_covar_scale
        vis_covar_diag = vis_covar_diag.repeat(vis_std.shape[0], 1)
    elif naive_vis_covar:
        vis_covar_diag = par.vis_covar_init_guess * (vis_std ** 2)
    else:
        vis_covar_diag = 10 ** (par.vis_covar_beta * torch.tanh(par.vis_covar_gamma * vis_std))
        vis_covar_diag = par.vis_covar_init_guess * vis_covar_diag
    
    vis_covar_diag = vis_covar_diag / vis_covar_scale.view(1, 6)
    if return_diag:
        return vis_covar_diag
    vis_covar = torch.diag_embed(vis_covar_diag)
    return vis_covar
        

class EKFModel(torch.nn.Module):
    """[summary]
    """
    def __init__(self, train_init_covar, train_imu_noise_covar, vis_covar_use_fixed, trans_scale_factor, naive_vis_covar):
        """Pre-define noise covariances etc. 
        """
        super(EKFModel, self).__init__()
        self.par = EKFParams()
        self.vis_covar_use_fixed = vis_covar_use_fixed
        self.trans_scale_factor = trans_scale_factor
        self.naive_vis_covar = naive_vis_covar
        
        ## IMU initial covariance
        self.init_covar_diag_sqrt = torch.nn.Parameter(torch.tensor(self.par.init_covar_diag_sqrt, dtype=torch.float32))
        if train_init_covar:
            self.init_covar_diag_sqrt.requires_grad = True
        else:
            self.init_covar_diag_sqrt.requires_grad = False
        
        ## IMU noise covariance
        self.imu_noise_covar_weights = torch.nn.Linear(1, 4, bias=False)
        if train_imu_noise_covar:
            for p in self.imu_noise_covar_weights.parameters():
                p.requires_grad = True
            self.imu_noise_covar_weights.weight.data /= 10
        else:
            for p in self.imu_noise_covar_weights.parameters():
                p.requires_grad = False
            self.imu_noise_covar_weights.weight.data.zero_()
    
    
    def get_par(self):
        return self.par    
    
    
    def get_imu_noise_covar(self):
        covar = 10 ** (self.par.imu_noise_covar_beta * torch.tanh(self.par.imu_noise_covar_gamma * self.imu_noise_covar_weights(
                torch.ones(1, device=self.imu_noise_covar_weights.weight.device))))

        imu_noise_covar_diag = torch.tensor(self.par.imu_noise_covar_diag, dtype=torch.float32,device=self.imu_noise_covar_weights.weight.device).repeat_interleave(3) 
        imu_noise_covar_diag = imu_noise_covar_diag * torch.stack([
                                            covar[0], covar[0], covar[0],
                                            covar[1], covar[1], covar[1],
                                            covar[2], covar[2], covar[2],
                                            covar[3], covar[3], covar[3]])
        
        return torch.diag(imu_noise_covar_diag)
            
    
    def force_symmetrical(self, M):
        M_upper = torch.triu(M)
        return M_upper + M_upper.transpose(-2, -1) * \
               (1 - torch.eye(M_upper.size(-2), M_upper.size(-1), device=M.device).repeat(M_upper.size(0), 1, 1))

    
    def propagate(self, 
                  dts, 
                  wa_xyzs, 
                  R_ckbts, 
                  g_k,
                  bw_k,
                  ba_k,
                  imu_noise_covar,
                  imu_erorr_covar):
        """EKF Propagation
        """
        batch_size = dts.shape[0]
        assert dts.shape[1] + 1 == wa_xyzs.shape[1] == R_ckbts.shape[1]
        for idx in range(dts.shape[1]):
            dt = dts[:, idx]
            gyro_meas = wa_xyzs[:, idx, :3]
            accel_meas = wa_xyzs[:, idx, 3:]
            R_ckbt = R_ckbts[:, idx, :, :]
            R_ckbt_transpose = R_ckbt.transpose(-2, -1)
            
            dt2 = dt * dt
            w = gyro_meas - bw_k
            w_skewed = skew3_b(w.unsqueeze(-1))
            a = accel_meas - mm(R_ckbt_transpose, g_k.unsqueeze(-1)).squeeze(-1) - ba_k
            I3 = torch.eye(3, 3, device=dts.device).repeat(batch_size, 1, 1)
            exp_int_w = exp_SO3_b((dt.unsqueeze(-1) * w).unsqueeze(-1))
            exp_int_w_transpose = exp_int_w.transpose(-2, -1)
            
            # propagate uncertainty, 2nd order
            F = torch.zeros(batch_size, 18, 18, device=dts.device)
            F[:, 0:3, 0:3] = -w_skewed
            F[:, 0:3, 12:15] = -I3 
            F[:, 3:6, 6:9] = I3 
            F[:, 6:9, 0:3] = -mm(R_ckbt, skew3_b(mm(R_ckbt_transpose, g_k.unsqueeze(-1)) + a.unsqueeze(-1)))
            F[:, 6:9, 9:12] = -I3
            F[:, 6:9, 15:18] = -R_ckbt
            
            G = torch.zeros(batch_size, 18, 12, device=dts.device)
            G[:, 0:3, 0:3] = -I3
            G[:, 6:9, 6:9] = -R_ckbt
            G[:, 12:15, 3:6] = I3 
            G[:, 15:18, 9:12] = I3 
            
            # dt, dt2 from [8] to [8, 1, 1]
            dt = dt.unsqueeze(-1).unsqueeze(-1)
            dt2 = dt2.unsqueeze(-1).unsqueeze(-1)
            
            Phi = torch.eye(18, 18, device=dts.device).repeat(batch_size, 1, 1)
            Phi += F * dt + 0.5 * mm(F, F) * dt2
            
            # This part is approx -> Can be removed
            Phi[:, 0:3, 0:3] = exp_int_w_transpose
            Q = mm(mm(mm(mm(Phi, G), imu_noise_covar.repeat(batch_size, 1, 1)),
                  G.transpose(-2, -1)), Phi.transpose(-2, -1)) * dt
            imu_erorr_covar = mm(mm(Phi, imu_erorr_covar), Phi.transpose(-2, -1)) + Q
            imu_erorr_covar = self.force_symmetrical(imu_erorr_covar)
            
        return imu_erorr_covar
        
        
    def update(self, 
               preimu_rot, 
               preimu_trans,
               imu_error_covar,
               vis_rot, 
               vis_trans,
               vis_covar,
               H0, H1,
               v_ck, g_ck):
        """EKF update
        """
        # preimu_rot and vis_rot are phi_c (so3 of R)
        residual_rot = vis_rot - preimu_rot
        residual_trans = vis_trans - preimu_trans
        residual = torch.cat([residual_rot, residual_trans], dim=1)
        
        batch_size = vis_rot.shape[0]
        I3 = torch.eye(3, 3, device=vis_rot.device).repeat(batch_size, 1, 1)
        H = torch.zeros(batch_size, 6, 18, device=vis_rot.device)
        H[:, 0:3, 0:3] = H0 
        H[:, 3:6, 0:3] = H1 
        H[:, 3:6, 3:6] = I3 
        
        H_transpose = H.transpose(-2, -1)
        
        S = mm(mm(H, imu_error_covar), H_transpose) + vis_covar 
        K = mm(mm(imu_error_covar, H_transpose), S.inverse()) # [B, 18, 6]
        
        est_error = mm(K, residual.unsqueeze(-1))
        
        # I18 = torch.eye(18, 18, device=vis_rot.device).repeat(batch_size, 1, 1)
        # est_covar = mm(I18 - mm(K, H), imu_error_covar)
        
        phi_ckbkp1_error = est_error[:, 0:3]
        p_ckbkp1_error = est_error[:, 3:6]
        v_ck_error = est_error[:, 6:9]
        g_ck_error = est_error[:, 9:12]
        # bw_bt_error = est_error[:, 12:15]
        # ba_bt_error = est_error[:, 15:18]
        
        ekf_phi_c = preimu_rot + mm(H0, phi_ckbkp1_error).squeeze(-1)
        ekf_t_c = preimu_trans + mm(H1, phi_ckbkp1_error).squeeze(-1) + p_ckbkp1_error.squeeze(-1)
        ekf_v_ck = v_ck + v_ck_error.squeeze(-1)
        ekf_g_ck = g_ck + g_ck_error.squeeze(-1)
        
        return ekf_phi_c, ekf_t_c, ekf_v_ck, ekf_g_ck
        
    
    def forward(self, 
                dts_full, 
                wa_xyz_full, 
                R_ckbt_full, 
                velocities_full, 
                gravities_full, 
                H0_full,
                H1_full,
                preimu_rot_full,
                preimu_trans_full, 
                vis_rot_full,
                vis_trans_full,
                vis_rot_std_full,
                vis_trans_std_full):
        """EKF propagation and update 
        
        Args: All list has length 2: from 0 to -1, from 1 to 0 (Ending with _full), [(Size of each element),..]
            dts: Raw delta_time data, [(B, 11), (B, 11)]
            wa_xyz: Raw wa_xyz data, [(B, 12, 6), (B, 12, 6)]
            R_ckbt: Preintegrated R_ckbt, [(B, 12, 3, 3), (B, 12, 3, 3)]
            velocities: CNN predicted velocities, [(B, 3), (B, 3)]
            gravities: CNN predicted gravities at frame -1 and 0, [(B, 3), (B, 3)]
            H0: H[0:3, 0:3] in EKF update, [(B, 3, 3), (B, 3, 3)]
            H1: H[3:6, 0:3] in EKF update, [(B, 3, 3), (B, 3, 3)],
            preimu_rot: IMU preintegrated rotations (phi), [(B, 3), (B, 3)]
            preimu_trans: IMU preintegrated translations, [(B, 3), (B, 3)]
            vis_rot: Camera predicted rotations (phi), [(B, 3), (B, 3)]
            vis_trans: Camera predicted translations, [(B, 3), (B, 3)]
            vis_rot_std: Camera predicted rotation std (phi), [(B, 3), (B, 3)]
            vis_trans_std: Camera predicted translation std, [(B, 3), (B, 3)]
        """
        dts = torch.cat(dts_full, dim=0) # (2B, 11)
        wa_xyzs = torch.cat(wa_xyz_full, dim=0) # (2B, 12, 6)
        R_ckbts = torch.cat(R_ckbt_full, dim=0) # (2B, 12, 3, 3)
        v_k = torch.cat(velocities_full, dim=0) # (2B, 3)
        g_k = torch.cat(gravities_full, dim=0) # (2B, 3)
        H0 = torch.cat(H0_full, dim=0) # (2B, 3, 3)
        H1 = torch.cat(H1_full, dim=0) # (2B, 3, 3)
        preimu_rot = torch.cat(preimu_rot_full, dim=0) # (2B, 3)
        preimu_trans = torch.cat(preimu_trans_full, dim=0) # (2B, 3)
        vis_rot = torch.cat(vis_rot_full, dim=0) # (2B, 3)
        vis_trans = torch.cat(vis_trans_full, dim=0) # (2B, 3)
        vis_rot_std = torch.cat(vis_rot_std_full, dim=0) # (2B, 3)
        vis_trans_std = torch.cat(vis_trans_std_full, dim=0) # (2B, 3)
        vis_std = torch.cat([vis_rot_std, vis_trans_std], dim=1) # (2B, 6)
        
        ## Process vis_meas and vis_covar predicted from camera images using CNN
        vis_covar = proc_vis_covar(self.par, vis_std, vis_covar_use_fixed=self.vis_covar_use_fixed, return_diag=False, naive_vis_covar=self.naive_vis_covar)
        
        ## NOTE: All translations in EKF are at original scale, rather than /=5.4!!
        # k * N(mean, cov) = N(k * mean, k^2 * cov)
        vis_trans = vis_trans * self.trans_scale_factor # Account for scale factor
        vis_covar = vis_covar * (self.trans_scale_factor ** 2)
        
        ## Initialize the initialial covariances and biases# ??? Set the covar of R, p to zero using U ??? (In deep_ekf_vio)
        # -> Need to specify imu_noise_covar and prev_covar here!!!
        ba_k = 0.
        bw_k = 0.
        imu_noise_covar = self.get_imu_noise_covar()
        
        batch_size = dts.shape[0]
        
        prev_covar = torch.diag(self.init_covar_diag_sqrt * self.init_covar_diag_sqrt + self.par.init_covar_diag_eps).repeat(batch_size, 1, 1)
        
        U = torch.diag(torch.tensor([0.] * 6 + [1.] * 12, device=dts.device)).repeat(batch_size, 1, 1)
        
        imu_erorr_covar = torch.matmul(torch.matmul(U, prev_covar), U.transpose(-2, -1))
        
        # EKF propagation
        imu_error_covar = self.propagate(
                dts = dts, 
                wa_xyzs = wa_xyzs, 
                R_ckbts = R_ckbts, 
                g_k = g_k, 
                bw_k = bw_k, ba_k = ba_k,
                imu_noise_covar = imu_noise_covar,
                imu_erorr_covar = imu_erorr_covar
            )
        
        # EKF update
        ekf_phi_c_full, ekf_t_c_full, ekf_v_ck_full, ekf_g_ck_full = self.update(
                preimu_rot = preimu_rot, 
                preimu_trans = preimu_trans, 
                imu_error_covar = imu_error_covar,
                vis_rot = vis_rot, 
                vis_trans = vis_trans,
                vis_covar = vis_covar,
                H0 = H0, H1 = H1,
                v_ck = v_k, g_ck = g_k
            )

        # Separate *_full into [from 0 to -1, from 1 to 0]
        assert batch_size % 2 == 0
        half_size = int(batch_size / 2)
        ekf_phi_c = [ekf_phi_c_full[ : half_size], 
                     ekf_phi_c_full[half_size : ]]
        ekf_t_c = [ekf_t_c_full[ : half_size], 
                   ekf_t_c_full[half_size : ]]
        ekf_v_ck = [ekf_v_ck_full[ : half_size],
                    ekf_v_ck_full[half_size : ]]
        ekf_g_ck = [ekf_g_ck_full[ : half_size],
                    ekf_g_ck_full[half_size : ]]
        
        return ekf_phi_c, ekf_t_c, ekf_v_ck, ekf_g_ck, vis_covar, imu_error_covar

    
    def get_vis_covar(self, vis_rot_std_full, vis_trans_std_full):
        """Get the vis_covar only for displaying
        """
        vis_rot_std = torch.cat(vis_rot_std_full, dim=0) # (2B, 3)
        vis_trans_std = torch.cat(vis_trans_std_full, dim=0) # (2B, 3)
        vis_std = torch.cat([vis_rot_std, vis_trans_std], dim=1) # (2B, 6)
        
        ## Process vis_meas and vis_covar predicted from camera images using CNN
        vis_covar = proc_vis_covar(self.par, vis_std, vis_covar_use_fixed=self.vis_covar_use_fixed, return_diag=False, naive_vis_covar=self.naive_vis_covar)
        
        ## NOTE: All translations in EKF are at original scale, rather than /=5.4!!
        # k * N(mean, cov) = N(k * mean, k^2 * cov)
        vis_covar = vis_covar * (self.trans_scale_factor ** 2)
        return vis_covar





