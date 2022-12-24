import argparse
import os
import random

import numpy as np
import torch
from torch.utils import data
from dataset import RoboDepthDataset

# corruptions: weather & lighting
from utils import create_brightness, create_dark, create_fog, create_frost, create_snow, create_contrast

# corruptions: sensor & movement
from utils import create_defocus_blur, create_glass_blur, create_motion_blur, create_zoom_blur, create_elastic, create_color_quant

# corruptions: data & processing
from utils import create_gaussian_noise, create_impulse_noise, create_shot_noise, create_iso_noise, create_pixelate, create_jpeg

# copy clean
from utils import copy_clean


def get_args():
    parser = argparse.ArgumentParser(description='Create RoboDepth Corruptions')
    # general configurations
    parser.add_argument('--image_list', type=str,
                        help="the file path to the image list.", default="splits/eigen.txt")
    parser.add_argument('--H', type=int, 
                        help='height for the image.', default=192)
    parser.add_argument('--W', type=int, 
                        help='width for the image.', default=640)          
    parser.add_argument('--save_path', type=str,
                        help="the file path for saving corrputed images.", default="data_test/data")
    parser.add_argument('--seed', type=int, 
                        help='random seed.', default=1005)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    # corruption configurations
    parser.add_argument('--if_brightness', type=bool,
                        help="whether to crate 'brightness' corruptions.")
    parser.add_argument('--if_dark', type=bool,
                        help="whether to crate 'dark' corruptions.")
    parser.add_argument('--if_fog', action='store_true',
                        help="whether to crate 'fog' corruptions.")
    parser.add_argument('--if_frost', action='store_true',
                        help="whether to crate 'frost' corruptions.")
    parser.add_argument('--if_snow', action='store_true',
                        help="whether to crate 'snow' corruptions.")
    parser.add_argument('--if_contrast', action='store_true',
                        help="whether to crate 'contrast' corruptions.")
    parser.add_argument('--if_defocus_blur', action='store_true',
                        help="whether to crate 'defocus_blur' corruptions.")
    parser.add_argument('--if_glass_blur', action='store_true',
                        help="whether to crate 'glass_blur' corruptions.")
    parser.add_argument('--if_motion_blur', action='store_true',
                        help="whether to crate 'motion_blur' corruptions.")
    parser.add_argument('--if_zoom_blur', action='store_true',
                        help="whether to crate 'zoom_blur' corruptions.")
    parser.add_argument('--if_elastic', action='store_true',
                        help="whether to crate 'elastic' corruptions.")
    parser.add_argument('--if_color_quant', action='store_true',
                        help="whether to crate 'color_quant' corruptions.")
    parser.add_argument('--if_gaussian_noise', action='store_true',
                        help="whether to crate 'gaussian_noise' corruptions.")
    parser.add_argument('--if_impulse_noise', action='store_true',
                        help="whether to crate 'impulse_noise' corruptions.")
    parser.add_argument('--if_shot_noise', action='store_true',
                        help="whether to crate 'shot_noise' corruptions.")
    parser.add_argument('--if_iso_noise', action='store_true',
                        help="whether to crate 'iso_noise' corruptions.")
    parser.add_argument('--if_pixelate', action='store_true',
                        help="whether to crate 'pixelate' corruptions.")
    parser.add_argument('--if_jpeg', action='store_true',
                        help="whether to crate 'jpeg_compression' corruptions.")
    parser.add_argument('--if_copy_clean', action='store_true',
                        help="whether to copy 'clean' images.")
    parser.add_argument('--severity_levels', type=list,
                        help="severity levels to be applied.", default=[])
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = get_args()

    # set random seeds
    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)



if __name__ == '__main__':
    main()
