from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class HRDepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="HR-Depth options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 #  default=os.path.join(file_dir, "kitti_data")
                                 default='kitti_data')
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # ABLATION options
        self.parser.add_argument("--HR_Depth",
                                 help="if set, uses HR Depth network",
                                 action="store_true")
        self.parser.add_argument("--Lite_HR_Depth",
                                 help="if set, uses lite hr depth network",
                                 action="store_true")

        # EVALUATION options
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)

        # RoboDepth options
        self.parser.add_argument('--seed', type=int, 
                                 help='random seed.',
                                 default=1205)
      
        self.parser.add_argument('--deterministic',
                                 action='store_true',
                                 help='whether to set deterministic options for CUDNN backend.')

    def parse(self):
        options = self.parser.parse_args()
        return options
