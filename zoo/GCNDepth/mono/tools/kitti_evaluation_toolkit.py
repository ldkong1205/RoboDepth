from __future__ import absolute_import, division, print_function

import glob
import argparse
import sys, os.path
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
sys.path.append('.')
from mono.tools import transformations as tr
from mono.tools.pose_evaluation_utils import *
from mono.tools.trajectory import align_trajectory


class kittiOdomEval():
    def __init__(self, config):
        assert os.path.exists(config.gt_dir), "Error of ground_truth pose path!"
        gt_files = glob.glob(config.gt_dir + '/*.txt')
        gt_files = [os.path.split(f)[1] for f in gt_files]
        self.seqs_with_gt = [os.path.splitext(f)[0] for f in gt_files]

        self.lengths = [100,200,300,400,500,600,700,800]
        self.num_lengths = len(self.lengths)
        self.gt_dir     = config.gt_dir
        self.result_dir = config.result_dir
        self.eval_seqs  = []
        
        # evalute all files in the folder
        if config.eva_seqs == '*':
            if not os.path.exists(self.result_dir):
                print('File path error!')
                exit()
            if os.path.exists(self.result_dir + '/all_stats.txt'): 
                os.remove(self.result_dir + '/all_stats.txt')
            files = glob.glob(self.result_dir + '/*.txt')
            assert files, "There is not trajectory files in: {}".format(self.result_dir)
            for f in files:
                dirname, basename = os.path.split(f)
                file_name = os.path.splitext(basename)[0]
                self.eval_seqs.append(str(file_name))
        else:
            seqs = config.eva_seqs.split(',')
            self.eval_seqs = [str(s) for s in seqs]

        self.eval_seqs = [s[:-5] for s in self.eval_seqs]    # xxxx_pred => xxxx

        # # Ref: https://github.com/MichaelGrupp/evo/wiki/Plotting
        # os.system("evo_config set plot_seaborn_style whitegrid \
        #             plot_linewidth 1.0 \
        #             plot_fontfamily sans-serif \
        #             plot_fontscale 1.0 \
        #             plot_figsize 10 10 \
        #             plot_export_format pdf")

    def toCameraCoord(self, pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0,   0,   1,  0],
                          [-1,  0,   0,  0],
                          [0,  -1,   0,  0],
                          [0,   0,   0,  1]])
        inv_R_C2L = np.linalg.inv(R_C2L)            
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot 

    def loadPoses(self, file_name, toCameraCoord):
        '''
            Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            if toCameraCoord:
                poses[frame_idx] = self.toCameraCoord(P)
            else:
                poses[frame_idx] = P

        return poses

    def loadPoseSe3(self, tra):

        poses = {}

        for cnt, p in enumerate(tra.poses_se3):

            poses[cnt] = p

        return poses

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0,3] - P2[0,3]
            dy = P1[1,3] - P2[1,3]
            dz = P1[2,3] - P2[2,3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))    
        self.distance = dist[-1]
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d,1.0),-1.0))

    def translationError(self, pose_error):
        dx = pose_error[0,3]
        dy = pose_error[1,3]
        dz = pose_error[2,3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        self.max_speed = 0
        # pre-compute distances (from ground truth as reference)
        dist = self.trajectoryDistances(poses_gt)
        # every second, kitti data 10Hz
        self.step_size = 10
        # for all start positions do
        # for first_frame in range(9, len(poses_gt), self.step_size):
        for first_frame in range(0, len(poses_gt), self.step_size):
            # for all segment lengths do
            for i in range(self.num_lengths):
                # current length
                len_ = self.lengths[i]
                # compute last frame of the segment
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                # Continue if sequence not long enough
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # compute rotational and translational errors, relative pose error (RPE)
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # compute speed 
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1*num_frames)   # 10Hz
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name,'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err 

    def plot_xyz(self, seq, poses_ref, poses_pred, plot_path_dir):
        
        def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
            """
                plot a path/trajectory based on xyz coordinates into an axis
                :param axarr: an axis array (for x, y & z) e.g. from 'fig, axarr = plt.subplots(3)'
                :param traj: trajectory
                :param style: matplotlib line style
                :param color: matplotlib color
                :param label: label (for legend)
                :param alpha: alpha value for transparency
            """
            x = range(0, len(positions_xyz))
            xlabel = "index"
            ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
            # plt.title('PRY')
            for i in range(0, 3):
                axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('XYZ')           

        fig, axarr = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))  
        
        pred_xyz = np.array([p[:3, 3] for _,p in poses_pred.items()])
        traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)
        if poses_ref:
            ref_xyz = np.array([p[:3, 3] for _,p in poses_ref.items()])
            traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)
      
        name = "{}_xyz".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)       
        # plt.show()
        pdf.close()

    def plot_rpy(self, seq, poses_ref, poses_pred, plot_path_dir, axes='szxy'):
        
        def traj_rpy(axarr, orientations_euler, style='-', color='black', title="", label="", alpha=1.0):
            """
            plot a path/trajectory's Euler RPY angles into an axis
            :param axarr: an axis array (for R, P & Y) e.g. from 'fig, axarr = plt.subplots(3)'
            :param traj: trajectory
            :param style: matplotlib line style
            :param color: matplotlib color
            :param label: label (for legend)
            :param alpha: alpha value for transparency
            """
            x = range(0, len(orientations_euler))
            xlabel = "index"
            ylabels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
            # plt.title('PRY')
            for i in range(0, 3):
                axarr[i].plot(x, np.rad2deg(orientations_euler[:, i]), style,
                            color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('PRY')           

        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))

        pred_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_pred.items()])
        traj_rpy(axarr_rpy, pred_rpy, '-', 'b', title='RPY', label='Ours', alpha=1.0)
        if poses_ref:
            ref_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _,p in poses_ref.items()])
            traj_rpy(axarr_rpy, ref_rpy, '-', 'r', label='GT', alpha=1.0)

        name = "{}_rpy".format(seq)
        plt.savefig(plot_path_dir +  "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + name + ".pdf")        
        fig_rpy.tight_layout()
        pdf.savefig(fig_rpy)       
        # plt.show()
        pdf.close()

    def plotPath_2D_3(self, seq, poses_gt, poses_result, plot_path_dir):
        '''
            plot path in XY, XZ and YZ plane
        '''
        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        ### get the value
        if poses_gt: 
            poses_gt = [(k,poses_gt[k]) for k in sorted(poses_gt.keys())]
            x_gt = np.asarray([pose[0,3] for _,pose in poses_gt])
            y_gt = np.asarray([pose[1,3] for _,pose in poses_gt])
            z_gt = np.asarray([pose[2,3] for _,pose in poses_gt])
        poses_result = [(k,poses_result[k]) for k in sorted(poses_result.keys())]
        x_pred = np.asarray([pose[0,3] for _,pose in poses_result])
        y_pred = np.asarray([pose[1,3] for _,pose in poses_result])
        z_pred = np.asarray([pose[2,3] for _,pose in poses_result])
        
        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        plt.subplot(1,3,1)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
                            for lims, mean_ in ((xlim, xmean),
                                                (ylim, ymean))
                            for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,2)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,3)
        ax = plt.gca()
        if poses_gt: plt.plot(y_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(y_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        png_title = "{}_path".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        # plt.show()
        plt.close()

    def plotPath_3D(self, seq, poses_gt, poses_result, plot_path_dir):
        """
            plot the path in 3D space
        """
        from mpl_toolkits.mplot3d import Axes3D

        start_point = [[0], [0], [0]]
        fontsize_ = 8
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        poses_dict = {}      
        poses_dict["Ours"] = poses_result
        if poses_gt:
            poses_dict["Ground Truth"] = poses_gt

        fig = plt.figure(figsize=(8,8), dpi=110)
        ax = fig.gca(projection='3d')

        for key,_ in poses_dict.items():
            plane_point = []
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                plane_point.append([pose[0,3], pose[2,3], pose[1,3]])
            plane_point = np.asarray(plane_point)
            style = style_pred if key == 'Ours' else style_gt
            plt.plot(plane_point[:,0], plane_point[:,1], plane_point[:,2], style, label=key)  
        plt.plot(start_point[0], start_point[1], start_point[2], style_O, label='Start Point')

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)
        plot_radius = max([abs(lim - mean_)
                        for lims, mean_ in ((xlim, xmean),
                                            (ylim, ymean),
                                            (zlim, zmean))
                        for lim in lims])
        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

        ax.legend()
        # plt.legend(loc="upper right", prop={'size':fontsize_}) 
        ax.set_xlabel('x (m)', fontsize=fontsize_)
        ax.set_ylabel('z (m)', fontsize=fontsize_)
        ax.set_zlabel('y (m)', fontsize=fontsize_)
        ax.view_init(elev=20., azim=-35)

        png_title = "{}_path_3D".format(seq)
        plt.savefig(plot_path_dir+"/"+png_title+".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
        fig.tight_layout()
        pdf.savefig(fig)  
        # plt.show()
        plt.close()

    def plotError_segment(self, seq, avg_segment_errs, plot_error_dir):
        '''
            avg_segment_errs: dict [100: err, 200: err...]
        '''
        fontsize_ = 15
        plot_y_t = []
        plot_y_r = []
        plot_x = []
        for idx, value in avg_segment_errs.items():
            if value == []:
                continue
            plot_x.append(idx)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Translation Error (%)',fontsize=fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Path Length (m)',fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize=fontsize_)
        png_title = "{}_error_seg".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    def plotError_speed(self, seq, avg_speed_errs, plot_error_dir):
        '''
            avg_speed_errs: dict [s1: err, s2: err...]
        '''
        fontsize_ = 15
        plot_y_t = []
        plot_y_r = []
        plot_x = []
        for idx, value in avg_speed_errs.items():
            if value == []:
                continue
            plot_x.append(idx * 3.6)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1]/np.pi * 180)
        
        fig = plt.figure(figsize=(15,6), dpi=100)
        plt.subplot(1,2,1)        
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_t)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Translation Error (%)',fontsize = fontsize_)

        plt.subplot(1,2,2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_r)*(1+0.1)])
        plt.xlabel('Speed (km/h)',fontsize = fontsize_)
        plt.ylabel('Rotation Error (deg/m)',fontsize = fontsize_)
        png_title = "{}_error_speed".format(seq)
        plt.savefig(plot_error_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    def computeSegmentErr(self, seq_errs):
        '''
            This function calculates average errors for different segment.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_  = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def computeSpeedErr(self, seq_errs):
        '''
            This function calculates average errors for different speed.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for s in range(2, 25, 2):
            segment_errs[s] = []

        # Get errors
        for err in seq_errs:
            speed = err[4]
            t_err = err[2]
            r_err = err[1]
            for key in segment_errs.keys():
                if np.abs(speed - key) < 2.0:
                    segment_errs[key].append([t_err, r_err])

        # Compute average
        for key in segment_errs.keys():
            if segment_errs[key] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[key])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[key])[:,1])
                avg_segment_errs[key] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[key] = []
        return avg_segment_errs

    def call_evo_traj(self, pred_file, save_file, gt_file=None, plot_plane='xy'):
        command = ''
        if os.path.exists(save_file): os.remove(save_file)
        
        if gt_file != None:
            command = ("evo_traj kitti %s --ref=%s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, gt_file, plot_plane, save_file)
        else:
            command = ("evo_traj kitti %s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, plot_plane, save_file)
        os.system(command)

    def eval(self, toCameraCoord):
        '''
            to_camera_coord: whether the predicted pose needs to be convert to camera coordinate
        '''
        eval_dir = self.result_dir
        if not os.path.exists(eval_dir): os.makedirs(eval_dir)

        total_err = []
        ave_errs = {}       
        for seq in self.eval_seqs:
            eva_seq_dir = os.path.join(eval_dir, '{}_eval'.format(seq))
            pred_file_name = self.result_dir + '/{}_pred.txt'.format(seq)
            # pred_file_name = self.result_dir + '/{}.txt'.format(seq)
            gt_file_name   = self.gt_dir + '/{}.txt'.format(seq)
            save_file_name = eva_seq_dir + '/{}.pdf'.format(seq)
            assert os.path.exists(pred_file_name), "File path error: {}".format(pred_file_name)

            tra_pred = read_kitti_poses_file(pred_file_name)
            tra_gt = read_kitti_poses_file(gt_file_name)

            tra_corrected = align_trajectory(tra_pred, tra_gt, correct_only_scale=True)


            poses_result = self.loadPoseSe3(tra_corrected)

            poses_gt = self.loadPoseSe3(tra_gt)


            # ----------------------------------------------------------------------
            # load pose
            # if seq in self.seqs_with_gt:
            #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=gt_file_name)
            # else:
            #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=None)
            #     continue

            poses_result_back = self.loadPoses(pred_file_name, toCameraCoord=toCameraCoord)

            if not os.path.exists(eva_seq_dir): os.makedirs(eva_seq_dir) 

            if seq not in self.seqs_with_gt:
                self.calcSequenceErrors(poses_result, poses_result)
                print ("\nSequence: " + str(seq))
                print ('Distance (m): %d' % self.distance)
                print ('Max speed (km/h): %d' % (self.max_speed*3.6))
                self.plot_rpy(seq, None, poses_result, eva_seq_dir)
                self.plot_xyz(seq, None, poses_result, eva_seq_dir)
                self.plotPath_3D(seq, None, poses_result, eva_seq_dir)
                self.plotPath_2D_3(seq, None, poses_result, eva_seq_dir)
                continue
          
            poses_gt_back = self.loadPoses(gt_file_name, toCameraCoord=False)

            # ----------------------------------------------------------------------
            # compute sequence errors
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, eva_seq_dir + '/{}_error.txt'.format(seq))

            total_err += seq_err

            # ----------------------------------------------------------------------
            # Compute segment errors
            avg_segment_errs = self.computeSegmentErr(seq_err)
            avg_speed_errs = self.computeSpeedErr(seq_err)

            # ----------------------------------------------------------------------
            # compute overall error
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
            print ("\nSequence: " + str(seq))
            print ('Distance (m): %d' % self.distance)
            print ('Max speed (km/h): %d' % (self.max_speed*3.6))
            print ("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
            print ("Average sequence rotational error (deg/m): {0:.4f}\n".format(ave_r_err/np.pi * 180))
            with open(eva_seq_dir + '/%s_stats.txt' % seq, 'w') as f:
                f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
                f.writelines('Average sequence rotation error (deg/m):  {0:.4f}'.format(ave_r_err/np.pi * 180))
            ave_errs[seq] = [ave_t_err, ave_r_err]

            # ----------------------------------------------------------------------
            # Ploting
            self.plot_rpy(seq, poses_gt, poses_result, eva_seq_dir)
            self.plot_xyz(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_3D(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotPath_2D_3(seq, poses_gt, poses_result, eva_seq_dir)
            self.plotError_segment(seq, avg_segment_errs, eva_seq_dir)
            self.plotError_speed(seq, avg_speed_errs, eva_seq_dir)

            plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--gt_dir', type=str, default='./ground_truth_pose', help='Directory path of the ground truth odometry')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory path of storing the odometry results')
    parser.add_argument('--eva_seqs', type=str, default='09_pred,10_pred', help='The sequences to be evaluated')
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to convert the pose to camera coordinate')
    args = parser.parse_args()
    pose_eval = kittiOdomEval(args)
    pose_eval.eval(toCameraCoord=args.toCameraCoord)   # set the value according to the predicted results