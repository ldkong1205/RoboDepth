import numpy as np
import cv2
import os
import sys

sys.path.append('../../../')
import dataloader.file_io.get_path as gp

disp_names = ['disp_noc_0', 'disp_noc_1', 'disp_occ_0', 'disp_occ_1']
depth_names = ['depth_noc_0', 'depth_noc_1', 'depth_occ_0', 'depth_occ_1']
line_numbers = [20, 28, 20, 28]
for disp_name, depth_name, line_number in zip(disp_names, depth_names, line_numbers):
    path_getter = gp.GetPath()
    data_path = path_getter.get_data_path()
    data_path = os.path.join(data_path, 'kitti_2015', 'training')
    disp_path = os.path.join(data_path, disp_name)
    depth_path = os.path.join(data_path, depth_name)
    if not os.path.isdir(depth_path):
        os.makedirs(depth_path)
    calib_path = os.path.join(data_path, 'calib_cam_to_cam')
    file_list_im = os.listdir(disp_path)
    file_list_cam = os.listdir(calib_path)
    for image_file, cam_file in zip(file_list_im, file_list_cam):
        im_file = os.path.join(disp_path, image_file)
        cam_file = os.path.join(calib_path, cam_file)
        disp = cv2.imread(im_file, -1).astype(np.float)/256.
        cam_matrix = open(cam_file).readlines()[:line_number][-1][6:].split()
        foc_length = (float(cam_matrix[0]) + float(cam_matrix[4]))/2.0
        depth = 0.54*foc_length/(disp + 0.00000000001)
        depth[disp == 0] = 0
        depth = (depth*256).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_path, image_file), depth)
