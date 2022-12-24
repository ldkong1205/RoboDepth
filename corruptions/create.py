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
                        help='random seed.', default=42)
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
    # corruption severity levels
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

    if args.if_brightness:
        print("Creating 'brightness' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_brightness(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'brightness')))

    if args.if_dark:
        print("Creating 'dark' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_dark(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'dark')))

    if args.if_fog:
        print("Creating 'fog' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_fog(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'fog')))

    if args.if_frost:
        print("Creating 'frost' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_frost(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'frost')))

    if args.if_snow:
        print("Creating 'snow' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_snow(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'snow')))

    if args.if_contrast:
        print("Creating 'contrast' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_contrast(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'contrast')))

    if args.if_defocus_blur:
        print("Creating 'defocus_blur' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_defocus_blur(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'defocus_blur')))

    if args.if_glass_blur:
        print("Creating 'glass_blur' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_glass_blur(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'glass_blur')))

    if args.if_motion_blur:
        print("Creating 'motion_blur' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_motion_blur(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'motion_blur')))

    if args.if_zoom_blur:
        print("Creating 'zoom_blur' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_zoom_blur(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'zoom_blur')))

    if args.if_elastic:
        print("Creating 'elastic' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_elastic(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'elastic')))

    if args.if_color_quant:
        print("Creating 'color_quant' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_color_quant(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'color_quant')))

    if args.if_gaussian_noise:
        print("Creating 'gaussian_noise' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_gaussian_noise(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'gaussian_noise')))

    if args.if_impulse_noise:
        print("Creating 'impulse_noise' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_impulse_noise(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'impulse_noise')))

    if args.if_shot_noise:
        print("Creating 'shot_noise' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_shot_noise(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'shot_noise')))

    if args.if_iso_noise:
        print("Creating 'iso_noise' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_iso_noise(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'iso_noise')))

    if args.if_pixelate:
        print("Creating 'pixelate' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_pixelate(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'pixelate')))

    if args.if_jpeg:
        print("Creating 'jpeg_compression' corruptions ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        create_jpeg(
            dataloader,
            save_path=args.save_path,
            severity_levels=args.severity_levels,
        )
        print("Successful! The corrupted images are save to: '{}'.\n".format(os.path.join(args.save_path, 'jpeg_compression')))

    if args.if_copy_clean:
        print("Copying clean set ...")
        dataset = RoboDepthDataset(image_list=args.image_list, H=args.H, W=args.W)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        copy_clean(
            dataloader,
            save_path=args.save_path,
        )
        print("Successful! The clean images are save to: '{}'.\n".format(os.path.join(args.save_path, 'clean')))


if __name__ == '__main__':
    main()
