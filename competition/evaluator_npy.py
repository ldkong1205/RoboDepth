from __future__ import absolute_import, division, print_function

import argparse
import os
import cv2
import numpy as np

from skimage.transform import resize

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


# splits_dir = os.path.join(os.path.dirname(__file__), "splits")
splits_dir = 'splits'

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def get_args():
    parser = argparse.ArgumentParser(description='RoboDepth Competition ICRA 2023')

    parser.add_argument("--data_path", type=str,
                        help="path to the training data")
    parser.add_argument("--eval_stereo", action="store_true",
                        help="if set evaluates in stereo mode")
    parser.add_argument("--eval_mono", action="store_true",
                        help="if set evaluates in mono mode")
    parser.add_argument("--disable_median_scaling", action="store_true",
                        help="if set disables median scaling in evaluation")
    parser.add_argument("--pred_depth_scale_factor", type=float, default=1,
                        help="if set multiplies predictions by this number")
    parser.add_argument("--pred_to_eval", type=str,
                        help="optional path to a .npy disparities file to evaluate")
    parser.add_argument("--eval_split", type=str, default="eigen", choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                        help="which split to run eval on")
    parser.add_argument("--save_pred_disps", action="store_true",
                        help="if set saves predicted disparities")
    parser.add_argument("--no_eval", action="store_true",
                        help="if set disables evaluation")
    parser.add_argument("--eval_eigen_to_benchmark", action="store_true",
                        help="if set assume we are loading eigen results from npy but we want to evaluate using the new benchmark.")
    parser.add_argument("--eval_out_dir", type=str,
                        help="if set will output the disparities to this folder")
                        
    parser.add_argument("--load_weights_folder", type=str,
                        help="name of model to load")
    parser.add_argument("--models_to_load", nargs="+", type=str, default=["encoder", "depth", "pose_encoder", "pose"],
                        help="models to load")
    parser.add_argument("--num_layers", type=int, default=18, choices=[18, 34, 50, 101, 152],
                        help="number of resnet layers")

    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="minimum depth")
    parser.add_argument("--max_depth", type=float, default=100.0,
                        help="maximum depth")

    parser.add_argument("--height", type=int, default=192,
                        help="input image height")
    parser.add_argument("--width", type=int, default=640,
                        help="input image width")

    return parser.parse_args()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # predefined
    opt.eval_mono = True
    opt.num_workers = 0

    opt.pred_to_eval = 'disp.npy'


    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


    print("-> Loading predictions from {}".format(opt.pred_to_eval))
    pred_disps = np.load(opt.pred_to_eval)

    gt_path = os.path.join("robodepth_gt.npy")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    gt_depths = np.load(gt_path, allow_pickle=True)
    
    assert len(pred_disps) == len(gt_depths)

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        # pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_disp = resize(pred_disp, (gt_height, gt_width))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([
                0.40810811 * gt_height, 0.99189189 * gt_height,
                0.03594771 * gt_width,  0.96405229 * gt_width
            ]).astype(np.int32)
            
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    evaluate(get_args())
