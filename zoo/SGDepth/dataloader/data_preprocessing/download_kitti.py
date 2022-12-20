import os
import sys
import wget
import zipfile
import pandas as pd
import shutil
import numpy as np
import glob
import cv2

import dataloader.file_io.get_path as gp
import dataloader.file_io.dir_lister as dl
from kitti_utils import pcl_to_depth_map


def download_kitti_all(kitti_folder='kitti_download'):
    """ This pathon-script downloads all KITTI folders and aranges them in a
    coherent data structure which can respectively be used by the other data
    scripts. It is recommended to keep the standard name KITTI. Note that the
    path is determined automatically inside of file_io/get_path.py

    parameters:
    - kitti_folder: Name of the folder in which the dataset should be downloaded
                    This is no path but just a name. the path ios determined by
                    get_path.py

    """

    # Download the standard KITTI Raw data

    path_getter = gp.GetPath()
    dataset_folder_path = path_getter.get_data_path()
    assert os.path.isdir(dataset_folder_path), 'Path to dataset folder does not exist'

    kitti_path = os.path.join(dataset_folder_path, kitti_folder)
    kitti_raw_data = pd.read_csv('kitti_archives_to_download.txt',
                                 header=None, delimiter=' ')[0].values
    kitti_path_raw = os.path.join(kitti_path, 'Raw_data')
    if not os.path.isdir(kitti_path_raw):
        os.makedirs(kitti_path_raw)
    for url in kitti_raw_data:
        folder = os.path.split(url)[1]
        folder = os.path.join(kitti_path_raw, folder)
        folder = folder[:-4]
        wget.download(url, out=kitti_path_raw)
        unzipper = zipfile.ZipFile(folder + '.zip', 'r')
        unzipper.extractall(kitti_path_raw)
        unzipper.close()
        os.remove(folder + '.zip')

    kitti_dirs_days = os.listdir(kitti_path_raw)

    # Get ground truth depths

    kitti_path_depth_annotated = os.path.join(kitti_path, 'Depth_improved')
    if not os.path.isdir(kitti_path_depth_annotated):
        os.makedirs(kitti_path_depth_annotated)
    url_depth_annotated = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip'
    wget.download(url_depth_annotated, out=kitti_path_depth_annotated)
    depth_zipped = os.path.join(kitti_path_depth_annotated, os.path.split(url_depth_annotated)[1])
    unzipper = zipfile.ZipFile(depth_zipped, 'r')
    unzipper.extractall(kitti_path_depth_annotated)
    unzipper.close()
    os.remove(depth_zipped)

    trainval_folder = os.listdir(kitti_path_depth_annotated)
    kitti_drives_list = []
    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        kitti_drives_list.extend([os.path.join(sub_folder, i) for i in os.listdir(sub_folder)])

    for sub_folder in kitti_dirs_days:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)
        for drive in kitti_drives_list:
            if os.path.split(sub_folder)[1] in drive:
                shutil.move(drive, sub_folder)

    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_annotated, sub_folder)
        shutil.rmtree(sub_folder)

    # Get sparse depths

    kitti_path_depth_sparse = os.path.join(kitti_path, 'Depth_projected')
    if not os.path.isdir(kitti_path_depth_sparse):
        os.makedirs(kitti_path_depth_sparse)
    url_depth_sparse = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip'
    wget.download(url_depth_sparse, out=kitti_path_depth_sparse)
    depth_zipped = os.path.join(kitti_path_depth_sparse, os.path.split(url_depth_sparse)[1])
    unzipper = zipfile.ZipFile(depth_zipped,  'r')
    unzipper.extractall(kitti_path_depth_sparse)
    unzipper.close()
    os.remove(depth_zipped)

    trainval_folder = os.listdir(kitti_path_depth_sparse)
    kitti_drives_list = []
    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_sparse, sub_folder)
        kitti_drives_list.extend([os.path.join(sub_folder, i) for i in os.listdir(sub_folder)])

    for sub_folder in kitti_dirs_days:
        sub_folder = os.path.join(kitti_path_depth_sparse, sub_folder)
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)
        for drive in kitti_drives_list:
            if os.path.split(sub_folder)[1] in drive:
                shutil.move(drive, sub_folder)
    
    for sub_folder in trainval_folder:
        sub_folder = os.path.join(kitti_path_depth_sparse, sub_folder)
        shutil.rmtree(sub_folder)

    # download test_files and integrate them into the folder structure

    url_depth_testset = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip'
    wget.download(url_depth_testset, out=kitti_path)
    depth_zipped = os.path.join(kitti_path, os.path.split(url_depth_testset)[1])
    unzipper = zipfile.ZipFile(depth_zipped, 'r')
    unzipper.extractall(kitti_path)
    unzipper.close()
    os.remove(depth_zipped)

    init_depth_completion_folder = os.path.join(kitti_path, 'depth_selection',
                                                'test_depth_completion_anonymous', 'image')
    target_depth_completion_folder = os.path.join(kitti_path_raw, 'test_depth_completion', 'image_02')
    if not os.path.isdir(target_depth_completion_folder):
        os.makedirs(target_depth_completion_folder)
    shutil.move(init_depth_completion_folder, target_depth_completion_folder)
    os.rename(os.path.join(target_depth_completion_folder, os.path.split(init_depth_completion_folder)[1]),
              os.path.join(target_depth_completion_folder, 'data'))
    
    init_depth_completion_folder = os.path.join(kitti_path, 'depth_selection',
                                                'test_depth_completion_anonymous', 'intrinsics')
    target_depth_completion_folder = os.path.join(kitti_path_raw, 'test_depth_completion')
    shutil.move(init_depth_completion_folder, target_depth_completion_folder) 

    init_depth_completion_folder = os.path.join(kitti_path, 'depth_selection',
                                                'test_depth_completion_anonymous', 'velodyne_raw')
    target_depth_completion_folder = os.path.join(kitti_path_depth_sparse, 'test_depth_completion', 'image_02')
    if not os.path.isdir(target_depth_completion_folder):
        os.makedirs(target_depth_completion_folder)
    shutil.move(init_depth_completion_folder, target_depth_completion_folder)
    os.rename(os.path.join(target_depth_completion_folder, os.path.split(init_depth_completion_folder)[1]),
              os.path.join(target_depth_completion_folder, 'data'))

    init_depth_prediction_folder = os.path.join(kitti_path, 'depth_selection',
                                                'test_depth_prediction_anonymous', 'image')
    target_depth_prediction_folder = os.path.join(kitti_path_raw, 'test_depth_prediction', 'image_02')
    if not os.path.isdir(target_depth_prediction_folder):
        os.makedirs(target_depth_prediction_folder)
    shutil.move(init_depth_prediction_folder, target_depth_prediction_folder)
    os.rename(os.path.join(target_depth_prediction_folder, os.path.split(init_depth_prediction_folder)[1]),
              os.path.join(target_depth_prediction_folder, 'data'))

    init_depth_prediction_folder = os.path.join(kitti_path, 'depth_selection',
                                                'test_depth_prediction_anonymous', 'intrinsics')
    target_depth_prediction_folder = os.path.join(kitti_path_raw, 'test_depth_prediction')
    shutil.move(init_depth_prediction_folder, target_depth_prediction_folder)

    shutil.rmtree(os.path.join(kitti_path, 'depth_selection'))


def adjust_improvedgt_folders(kitti_folder = 'kitti_download'):
    """ This function adjust the format of the improved ground truth folder structure
    to the structure of the KITTI raw data and afterward removes the old directories.
    It is taken care that only the directories from the Download are worked on so that
    the procedure does not work on directories which it is not supposed to"""

    path_getter = gp.GetPath()
    dataset_folder_path = path_getter.get_data_path()
    gt_path = os.path.join(dataset_folder_path, kitti_folder)
    gt_path = os.path.join(gt_path, 'Depth_improved')
    assert os.path.isdir(gt_path), 'Path to data does not exist'
    folders = dl.DirLister.get_directories(gt_path)
    folders = dl.DirLister.include_dirs_by_name(folders, 'proj_depth')
    for f in folders:
        ground_path, camera = os.path.split(f)
        ground_path = os.path.split(ground_path)[0]
        ground_path = os.path.split(ground_path)[0]
        target_path = os.path.join(ground_path, camera, 'data')
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        else:
            continue
        for filepath in glob.glob(os.path.join(f, '*')):
            # Move each file to destination Directory
            shutil.move(filepath, target_path)
        print(target_path)

    for f in folders:
        remove_path = os.path.split(f)[0]
        remove_path = os.path.split(remove_path)[0]
        print(remove_path)
        shutil.rmtree(remove_path, ignore_errors=True)


def adjust_projectedvelodyne_folders(kitti_folder='kitti_download'):
    """ This function adjust the format of the sparse ground truth folder structure
    to the structure of the KITTI raw data and afterward removes the old directories.
    It is taken care that only the directories from the Download are worked on so that
    the procedure does not work on directories which it is not supposed to"""

    path_getter = gp.GetPath()
    dataset_folder_path = path_getter.get_data_path()
    gt_path = os.path.join(dataset_folder_path, kitti_folder)
    gt_path = os.path.join(gt_path, 'Depth_projected')
    assert os.path.isdir(gt_path), 'Path to data does not exist'
    folders = dl.DirLister.get_directories(gt_path)
    folders = dl.DirLister.include_dirs_by_name(folders, 'proj_depth')
    for f in folders:
        ground_path, camera = os.path.split(f)
        ground_path = os.path.split(ground_path)[0]
        ground_path = os.path.split(ground_path)[0]
        target_path = os.path.join(ground_path, camera, 'data')
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        else:
            continue
        for filepath in glob.glob(os.path.join(f, '*')):
            # Move each file to destination Directory
            shutil.move(filepath, target_path)
        print(target_path)

    for f in folders:
        remove_path = os.path.split(f)[0]
        remove_path = os.path.split(remove_path)[0]
        print(remove_path)
        shutil.rmtree(remove_path, ignore_errors=True)


def generate_depth_from_velo(kitti_folder='kitti_download'):
    """ This function generates the depth maps that correspond to the
    single point clouds of the raw LiDAR scans"""

    path_getter = gp.GetPath()
    dataset_folder_path = path_getter.get_data_path()
    gt_path = os.path.join(dataset_folder_path, kitti_folder)
    depth_path = os.path.join(gt_path, 'Depth')
    gt_path = os.path.join(gt_path, 'Raw_data')
    assert os.path.isdir(gt_path), 'Path to data does not exist'
    folders = dl.DirLister.get_directories(gt_path)
    folders = dl.DirLister.include_dirs_by_name(folders, 'velodyne_points')
    for f in folders:
        base_dir = os.path.split(f)[0]
        base_dir = os.path.split(base_dir)[0]
        calib_dir = os.path.split(base_dir)[0]
        image_dir_2 = os.path.join(base_dir, 'image_02', 'data')
        image_dir_3 = os.path.join(base_dir, 'image_03', 'data')
        day, drive = os.path.split(base_dir)
        day = os.path.split(day)[1]
        depth_dir_2 = os.path.join(depth_path, day, drive, 'image_02', 'data')
        depth_dir_3 = os.path.join(depth_path, day, drive, 'image_03', 'data')
        if not os.path.isdir(depth_dir_2):
            os.makedirs(depth_dir_2)
        if not os.path.isdir(depth_dir_3):
            os.makedirs(depth_dir_3)

        for file in glob.glob(os.path.join(f, '*')):
            filename = os.path.split(file)[1]
            filename_img = filename[:-3] + 'png'
            im_size_2 = cv2.imread(os.path.join(image_dir_2, filename_img)).shape[:2]
            im_size_3 = cv2.imread(os.path.join(image_dir_3, filename_img)).shape[:2]
            depth_2 = pcl_to_depth_map(calib_dir, file, im_size_2, 2)
            depth_3 = pcl_to_depth_map(calib_dir, file, im_size_3, 3)
            depth_2 = (depth_2 * 256).astype(np.uint16)
            depth_3 = (depth_3 * 256).astype(np.uint16)

            cv2.imwrite(os.path.join(depth_dir_2, filename_img), depth_2)
            cv2.imwrite(os.path.join(depth_dir_3, filename_img), depth_3)
        print(f)


if __name__ == '__main__':
    kitti_folder = 'kitti_download'
    download_kitti_all(kitti_folder)
    adjust_improvedgt_folders(kitti_folder)
    adjust_projectedvelodyne_folders(kitti_folder)
    generate_depth_from_velo(kitti_folder)