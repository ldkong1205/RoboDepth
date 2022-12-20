#!/usr/bin/env python3

# Python standard library
import os

# Public libraries
import numpy as np
import torch

# Local imports
from arguments import PoseEvaluationArguments
from harness import Harness


class PoseEvaluator(Harness):
    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.pose_validation_fixed_scaling
        self.val_num_log_images = opt.eval_num_images

    def evaluate(self):
        print('Evaluate pose predictions:', flush=True)

        scores = self._run_pose_validation()

        for domain in scores:
            print(f'  - Results for domain {domain}:')

            metrics = scores[domain].get_scores()

            print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(metrics['mean'], metrics['std']))

        self._log_gpu_memory()


if __name__ == "__main__":
    opt = PoseEvaluationArguments().parse()

    if opt.model_load is None:
        raise Exception('You must use --model-load to select a model state directory to run evaluation on')

    if opt.sys_best_effort_determinism:
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    evaluator = PoseEvaluator(opt)
    evaluator.evaluate()
