from __future__ import absolute_import, division, print_function

import os
import random
import time
import logging
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from utils_kittic import get_scores_monodepth2_mono
from utils_kittic import logger_summary


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = 'splits'
# splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


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


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt, corruption, severity, logger, info):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80


    # clear cache
    encoder_dict, dataset, dataloader  = None, None, None



    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(
            opt.data_path, filenames,
            encoder_dict['height'], encoder_dict['width'],
            [0], 4, is_train=False, img_ext='.png',
            eval_corr_type=(corruption, severity),
            on_the_fly=False,
        )
        dataloader = DataLoader(
            dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False
        )

        encoder = networks.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
        encoder.num_ch_enc = [64, 128, 216, 288, 288]  # = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder()

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, -1)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True,allow_pickle=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
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

    results_edit=open('results.txt',mode='a')
    results_edit.write("\n " + 'model_name: %s '%(opt.load_weights_folder))
    results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    results_edit.close()
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    log_abs_rel, log_sq_rel, log_rmse, log_rmse_log, log_a1, log_a2, log_a3 = mean_errors.tolist()

    if corruption:
        info[corruption][severity] = {
            'abs_rel' : round(log_abs_rel, 3),
            'sq_rel'  : round(log_sq_rel, 3),
            'rmse'    : round(log_rmse, 3),
            'rmse_log': round(log_rmse_log, 3),
            'a1'      : round(log_a1, 3),
            'a2'      : round(log_a2, 3),
            'a3'      : round(log_a3, 3),
        }
        logger.info(info[corruption][severity])
    else:
        info['clean'] = {
            'abs_rel' : round(log_abs_rel, 3),
            'sq_rel'  : round(log_sq_rel, 3),
            'rmse'    : round(log_rmse, 3),
            'rmse_log': round(log_rmse_log, 3),
            'a1'      : round(log_a1, 3),
            'a2'      : round(log_a2, 3),
            'a3'      : round(log_a3, 3),
        }
        logger.info(info['clean'])


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    options = MonodepthOptions()

    opt = options.parse()
    # evaluate(options.parse())

    # set random seed
    if opt.seed is not None:
        print(f'Set random seed to {opt.seed}, deterministic: '
                    f'{opt.deterministic}')
        set_random_seed(opt.seed, deterministic=opt.deterministic)

    # set logger
    log_dir = './logs'
    if not(os.path.exists(log_dir)):
        os.makedirs(log_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, f'{timestamp}.log'))
    logger = logging.getLogger()

    logger.info("RoboDepth Benchmark")
    logger.info("-"*100+'\n')
    logger.info("Evaluating weights loaded from: '{}' ...".format(opt.load_weights_folder))

    print("seed: '{}'.".format(opt.seed))
    logger.info("seed: '{}'.".format(opt.seed))

    # define corruptions
    corruptions = [
        'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform', 'color_quant',
        'gaussian_noise', 'impulse_noise', 'shot_noise', 'iso_noise', 'pixelate', 'jpeg_compression',
    ]

    logger.info("Evaluating '{}' corruptions, including:".format(len(corruptions)))
    for i, j in enumerate(corruptions):
        logger.info("[{}] - {}".format(i+1, j))

    logger.info("-"*100+'\n')

    # set info
    info = {
        'clean': dict(),
        'brightness': dict(), 'dark': dict(), 'fog': dict(), 'frost': dict(), 'snow': dict(), 'contrast': dict(),
        'defocus_blur': dict(), 'glass_blur': dict(), 'motion_blur': dict(), 'zoom_blur': dict(), 'elastic_transform': dict(), 'color_quant': dict(),
        'gaussian_noise': dict(), 'impulse_noise': dict(), 'shot_noise': dict(), 'iso_noise': dict(), 'pixelate': dict(), 'jpeg_compression': dict(),
        'avgs': dict(),
    }

    # set summary
    summary = {
        'printlogs': dict(),
        'avgs_DEE1': dict(),
        'avgs_DEE2': dict(),
        'avgs_DEE3': dict(),
    }

    # load baseline results
    # if opt.eval_mono and not opt.eval_stereo:
    baseline = get_scores_monodepth2_mono()  # (MonoDepth2, mono, 640x192)
    logger.info("Loading baseline scores (MonoDepth2, mono, 640x192) ...")
    

    # STEP 1: evaluation for clean
    logger.info("Evaluating clean ...")
    evaluate(opt=opt, corruption=None, severity=None, logger=logger, info=info)

    # calculate scores
    info['clean']['DEE1'] = round(info['clean']['abs_rel'] - info['clean']['a1']  + 1, 3)        # abs_rel - a1 + 1
    info['clean']['DEE2'] = round((info['clean']['abs_rel'] - info['clean']['a1']  + 1) / 2, 3)  # 0.5 * (abs_rel - a1 + 1)
    info['clean']['DEE3'] = round(info['clean']['abs_rel'] / info['clean']['a1'] , 3)            # abs_rel / a1

    # print results
    logger.info("Successful! Paste the following into the experiment log page:")

    printlog_clean = \
"""\n
### Clean
| $\\text{{Abs Rel}}$ | $\\text{{Sq Rel}}$ | $\\text{{RMSE}}$ | $\\text{{RMSE log}}$ | $\\delta < 1.25$ | $\\delta < 1.25^2$ | $\\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |

- **Summary:** $\\text{{DEE}}_1=$ {:.3f}, $\\text{{DEE}}_2=$ {:.3f}, $\\text{{DEE}}_3=$ {:.3f}
""".format(
    info['clean']['abs_rel'], info['clean']['sq_rel'], info['clean']['rmse'], info['clean']['rmse_log'], info['clean']['a1'], info['clean']['a2'], info['clean']['a3'],
    info['clean']['DEE1'], info['clean']['DEE2'], info['clean']['DEE3']
)
    logger.info(printlog_clean)
    logger.info("-"*100+'\n')



    # STEP 2: evaluations for corruptions
    for corruption in corruptions:

        logger.info("Evaluating corruption '{}' ...".format(corruption))

        for severity in range(1, 6):
            logger.info("Level '{}' ...".format(severity))
            evaluate(opt=opt, corruption=corruption, severity=severity, logger=logger, info=info)

        # calculate scores
        l1, l2, l3, l4, l5 = info[corruption][1], info[corruption][2], info[corruption][3], info[corruption][4], info[corruption][5]

        avg_abs_rel  = sum([l1['abs_rel'],  l2['abs_rel'],  l3['abs_rel'],  l4['abs_rel'],  l5['abs_rel']])  / 5
        avg_sq_rel   = sum([l1['sq_rel'],   l2['sq_rel'],   l3['sq_rel'],   l4['sq_rel'],   l5['sq_rel']])   / 5
        avg_rmse     = sum([l1['rmse'],     l2['rmse'],     l3['rmse'],     l4['rmse'],     l5['rmse']])     / 5
        avg_rmse_log = sum([l1['rmse_log'], l2['rmse_log'], l3['rmse_log'], l4['rmse_log'], l5['rmse_log']]) / 5
        avg_a1       = sum([l1['a1'],       l2['a1'],       l3['a1'],       l4['a1'],       l5['a1']])       / 5
        avg_a2       = sum([l1['a2'],       l2['a2'],       l3['a2'],       l4['a2'],       l5['a2']])       / 5
        avg_a3       = sum([l1['a3'],       l2['a3'],       l3['a3'],       l4['a3'],       l5['a3']])       / 5

        info['avgs'][corruption] = {
            'avg_abs_rel' : round(avg_abs_rel, 3),
            'avg_sq_rel'  : round(avg_sq_rel, 3),
            'avg_rmse'    : round(avg_rmse, 3),
            'avg_rmse_log': round(avg_rmse_log, 3),
            'avg_a1'      : round(avg_a1, 3),
            'avg_a2'      : round(avg_a2, 3),
            'avg_a3'      : round(avg_a3, 3),
            'DEE1'        : round(avg_abs_rel - avg_a1 + 1, 3),        # abs_rel - a1 + 1
            'DEE2'        : round((avg_abs_rel - avg_a1 + 1) / 2, 3),  # 0.5 * (abs_rel - a1 + 1)
            'DEE3'        : round(avg_abs_rel / avg_a1, 3),            # abs_rel / a1
        }

        # print results
        logger.info("Successful! Paste the following into the experiment log page:")

        printlog = \
"""\n
### {}
| Level | $\\text{{Abs Rel}}$ | $\\text{{Sq Rel}}$ | $\\text{{RMSE}}$ | $\\text{{RMSE log}}$ | $\\delta < 1.25$ | $\\delta < 1.25^2$ | $\\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
|   2   | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
|   3   | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
|   4   | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
|   5   | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
|  avg  | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |

- **Summary:** $\\text{{DEE}}_1=$ {:.3f}, $\\text{{DEE}}_2=$ {:.3f}, $\\text{{DEE}}_3=$ {:.3f}
""".format(
    corruption.capitalize(),
    l1['abs_rel'], l1['sq_rel'], l1['rmse'], l1['rmse_log'], l1['a1'], l1['a2'], l1['a3'],
    l2['abs_rel'], l2['sq_rel'], l2['rmse'], l2['rmse_log'], l2['a1'], l2['a2'], l2['a3'],
    l3['abs_rel'], l3['sq_rel'], l3['rmse'], l3['rmse_log'], l3['a1'], l3['a2'], l3['a3'],
    l4['abs_rel'], l4['sq_rel'], l4['rmse'], l4['rmse_log'], l4['a1'], l4['a2'], l4['a3'],
    l5['abs_rel'], l5['sq_rel'], l5['rmse'], l5['rmse_log'], l5['a1'], l5['a2'], l5['a3'],
    info['avgs'][corruption]['avg_abs_rel'],
    info['avgs'][corruption]['avg_sq_rel'],
    info['avgs'][corruption]['avg_rmse'],
    info['avgs'][corruption]['avg_rmse_log'],
    info['avgs'][corruption]['avg_a1'],
    info['avgs'][corruption]['avg_a2'],
    info['avgs'][corruption]['avg_a3'],
    info['avgs'][corruption]['DEE1'],
    info['avgs'][corruption]['DEE2'],
    info['avgs'][corruption]['DEE3'],
)

        logger.info(printlog)
        logger.info("-"*100)

        summary['printlogs'][corruption] = printlog
        summary['avgs_DEE1'][corruption] = info['avgs'][corruption]['DEE1']
        summary['avgs_DEE2'][corruption] = info['avgs'][corruption]['DEE2']
        summary['avgs_DEE3'][corruption] = info['avgs'][corruption]['DEE3']



    # STEP 3: summarize evaluations
    logger.info("Finished evaluation!")
    logger.info("Evaluating weights were loaded from: '{}'.".format(opt.load_weights_folder))
    logger.info("A total number of '{}' corruptions were evaluated.".format(len(corruptions)))

    logger.info("Paste the summary into the experiment log page:")
    printsummary = logger_summary(info, summary, baseline)
    logger.info(printsummary)

    printsummary_ = \
"""

{clean}

{bright}

{dark}

{fog}

{frost}

{snow}

{contrast}

{defocus}

{glass}

{motion}

{zoom}

{elastic}

{quant}

{gaussian}

{impulse}

{shot}

{iso}

{pixelate}

{jpeg}

""".format(
    clean     = printlog_clean,                             # 00
    bright    = summary['printlogs']['brightness'],         # 01
    dark      = summary['printlogs']['dark'],               # 02
    fog       = summary['printlogs']['fog'],                # 03
    frost     = summary['printlogs']['frost'],              # 04
    snow      = summary['printlogs']['snow'],               # 05
    contrast  = summary['printlogs']['contrast'],           # 06
    defocus   = summary['printlogs']['defocus_blur'],       # 07
    glass     = summary['printlogs']['glass_blur'],         # 08
    motion    = summary['printlogs']['motion_blur'],        # 09
    zoom      = summary['printlogs']['zoom_blur'],          # 10
    elastic   = summary['printlogs']['elastic_transform'],  # 11
    quant     = summary['printlogs']['color_quant'],        # 12
    gaussian  = summary['printlogs']['gaussian_noise'],     # 13
    impulse   = summary['printlogs']['impulse_noise'],      # 14
    shot      = summary['printlogs']['shot_noise'],         # 15
    iso       = summary['printlogs']['iso_noise'],          # 16
    pixelate  = summary['printlogs']['pixelate'],           # 17
    jpeg      = summary['printlogs']['jpeg_compression'],   # 18
)
    
    logger.info(printsummary_)

    print()
