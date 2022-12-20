from __future__ import division

import argparse
from mmcv import Config
from mmcv.runner import load_checkpoint

from mono.datasets.get_dataset import get_dataset
from mono.apis import (train_mono,
                       init_dist,
                       get_root_logger,
                       set_random_seed)
from mono.model.registry import MONO
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default='/home/user/Documents/code/fm_depth/config/cfg_kitti_fm_joint.py',
                        help='train config file path')
    parser.add_argument('--work_dir',
                        default='/media/user/harddisk/weight/fmdepth',
                        help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--gpus',
                        default='0',
                        type=str,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')
    parser.add_argument('--seed',
                        type=int,
                        default=1024,
                        help='random seed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args.config)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = [int(_) for _ in args.gpus.split(',')]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    print('cfg is ', cfg)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model_name = cfg.model['name']
    model = MONO.module_dict[model_name](cfg.model)

    if cfg.resume_from is not None:
        load_checkpoint(model, cfg.resume_from, map_location='cpu')
    elif cfg.finetune is not None:
        print('loading from', cfg.finetune)
        checkpoint = torch.load(cfg.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    train_dataset = get_dataset(cfg.data, training=True)
    if cfg.validate:
        val_dataset = get_dataset(cfg.data, training=False)
    else:
        val_dataset = None

    train_mono(model,
               train_dataset,
               val_dataset,
               cfg,
               distributed=distributed,
               validate=cfg.validate,
               logger=logger)


if __name__ == '__main__':
    main()