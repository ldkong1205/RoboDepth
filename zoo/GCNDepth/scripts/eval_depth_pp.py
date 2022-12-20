from __future__ import absolute_import, division, print_function
import cv2
import sys
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
sys.path.append('..')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines, compute_errors
from mono.datasets.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
STEREO_SCALE_FACTOR = 36
MIN_DEPTH=1e-3
MAX_DEPTH=80

def batch_post_process_disparity(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(MODEL_PATH, CFG_PATH, GT_PATH):
    filenames = readlines("../mono/datasets/splits/exp/val_files.txt")
    cfg = Config.fromfile(CFG_PATH)

    dataset = KITTIRAWDataset(cfg.data['in_path'],
                          filenames,
                          cfg.data['height'],
                          cfg.data['width'],
                          [0],
                          is_train=False,
                          gt_depth_path=None)

    dataloader = DataLoader(dataset,
                        2,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True,
                        drop_last=True)

    cfg.model['imgs_per_gpu'] = 2
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.eval()

    pred_disps = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            print(batch_idx)
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            outputs = model(inputs)

            disp = outputs[("disp", 0, 0)]
            # N = pred_disp.shape[0] // 2
            # pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
    pred_disps = np.concatenate(pred_disps)

    gt_depths = np.load(GT_PATH, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if cfg.data['stereo_scale']:
        print('using baseline')
    else:
        print('using mean scaling')

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))

        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)

        if cfg.data['stereo_scale']:
            ratio = STEREO_SCALE_FACTOR

        pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    mean_errors = np.array(errors).mean(0)
    print("Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    CFG_PATH = '../config/cfg_kitti_fm.py'#path to cfg file
    GT_PATH = '/media/sconly/harddisk/data/kitti/kitti_raw/rawdata/gt_depths.npz'#path to kitti gt depth
    MODEL_PATH = '/media/sconly/harddisk/weight/epoch_20.pth'#path to model weights
    evaluate(MODEL_PATH, CFG_PATH, GT_PATH)
