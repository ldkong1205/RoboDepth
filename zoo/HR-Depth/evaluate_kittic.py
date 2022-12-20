from __future__ import absolute_import, division, print_function

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import random
import time
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import HRDepthOptions
import datasets
import networks

from utils_kittic import get_scores_monodepth2_mono, get_scores_monodepth2_stereo, get_scores_monodepth2_mono_stereo
from utils_kittic import logger_summary


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = 'splits'
# splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
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


def evaluate(opt, corruption, severity, logger, info):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    # clear cache
    encoder_dict, dataset, dataloader  = None, None, None

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    print("Width: {}, Height: {}".format(encoder_dict['width'], encoder_dict['height']))

    # dataset
    dataset = datasets.KITTIRAWDataset(
        data_path=opt.data_path, filenames=filenames,
        height=encoder_dict['height'], width=encoder_dict['width'],  # [192, 640]
        frame_idxs=[0], num_scales=4, 
        is_train=False, img_ext='.png',
        eval_corr_type=(corruption, severity),
        on_the_fly=False,
    )
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # encoder
    if opt.Lite_HR_Depth:
        encoder = networks.MobileEncoder(pretrained=None)
    elif opt.HR_Depth:
        encoder = networks.ResnetEncoder(18, False)
    else:
        assert False," Please choose HR-Depth or Lite-HR-Depth "
    
    # decoder
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, mobile_encoder=opt.Lite_HR_Depth)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, -1)].to(device)  # [16, 3, 192, 640]
            # input_color = data[("color", 0, 0)].to(device)

            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disparity", "Scale0")], 0.1, 100.0)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # Apply the mask proposed by Eigen
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
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

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

    # load configs
    options = HRDepthOptions()
    opt = options.parse()

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
        'avgs_DES1': dict(),
        'avgs_DES2': dict(),
        'avgs_DES3': dict(),
    }

    # load baseline results
    # if opt.eval_mono and not opt.eval_stereo:
    baseline = get_scores_monodepth2_mono()  # (MonoDepth2, mono, 640x192)
    logger.info("Loading baseline scores (MonoDepth2, mono, 640x192) ...")
    # elif not opt.eval_mono and opt.eval_stereo:
    #     baseline = get_scores_monodepth2_mono_stereo()  # (MonoDepth2, stereo, 640x192)
    #     logger.info("Loading baseline scores (MonoDepth2, stereo, 640x192) ...")



    # STEP 1: evaluation for clean
    logger.info("Evaluating clean ...")
    evaluate(opt=opt, corruption=None, severity=None, logger=logger, info=info)

    # calculate scores
    info['clean']['DES1'] = round(info['clean']['abs_rel'] - info['clean']['a1']  + 1, 3)        # abs_rel - a1 + 1
    info['clean']['DES2'] = round((info['clean']['abs_rel'] - info['clean']['a1']  + 1) / 2, 3)  # 0.5 * (abs_rel - a1 + 1)
    info['clean']['DES3'] = round(info['clean']['abs_rel'] / info['clean']['a1'] , 3)            # abs_rel / a1

    # print results
    logger.info("Successful! Paste the following into the experiment log page:")

    printlog_clean = \
"""\n
### Clean
| $\\text{{Abs Rel}}$ | $\\text{{Sq Rel}}$ | $\\text{{RMSE}}$ | $\\text{{RMSE log}}$ | $\\delta < 1.25$ | $\\delta < 1.25^2$ | $\\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |

- **Summary:** $\\text{{DES}}_1=$ {:.3f}, $\\text{{DES}}_2=$ {:.3f}, $\\text{{DES}}_3=$ {:.3f}
""".format(
    info['clean']['abs_rel'], info['clean']['sq_rel'], info['clean']['rmse'], info['clean']['rmse_log'], info['clean']['a1'], info['clean']['a2'], info['clean']['a3'],
    info['clean']['DES1'], info['clean']['DES2'], info['clean']['DES3']
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
            'DES1'        : round(avg_abs_rel - avg_a1 + 1, 3),        # abs_rel - a1 + 1
            'DES2'        : round((avg_abs_rel - avg_a1 + 1) / 2, 3),  # 0.5 * (abs_rel - a1 + 1)
            'DES3'        : round(avg_abs_rel / avg_a1, 3),            # abs_rel / a1
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

- **Summary:** $\\text{{DES}}_1=$ {:.3f}, $\\text{{DES}}_2=$ {:.3f}, $\\text{{DES}}_3=$ {:.3f}
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
    info['avgs'][corruption]['DES1'],
    info['avgs'][corruption]['DES2'],
    info['avgs'][corruption]['DES3'],
)

        logger.info(printlog)
        logger.info("-"*100)

        summary['printlogs'][corruption] = printlog
        summary['avgs_DES1'][corruption] = info['avgs'][corruption]['DES1']
        summary['avgs_DES2'][corruption] = info['avgs'][corruption]['DES2']
        summary['avgs_DES3'][corruption] = info['avgs'][corruption]['DES3']



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


