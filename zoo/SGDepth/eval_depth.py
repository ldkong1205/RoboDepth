#!/usr/bin/env python3

# Python standard library
import os

# Public libraries
import numpy as np
import torch
from torchvision.utils import save_image

# Local imports
import colors
from arguments import DepthEvaluationArguments
from harness import Harness


class DepthEvaluator(Harness):
    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.depth_validation_fixed_scaling
        self.ratio_on_validation = opt.depth_ratio_on_validation
        self.val_num_log_images = opt.eval_num_images

    def evaluate(self):
        print('Evaluate depth predictions:', flush=True)

        scores, ratios, images = self._run_depth_validation(self.val_num_log_images)

        for domain in scores:
            print(f'  - Results for domain {domain}:')

            if len(ratios[domain]) > 0:
                ratios_np = np.array(ratios[domain])
                if self.ratio_on_validation:
                    dataset_split_pos = int(len(ratios_np)/4)
                else:
                    dataset_split_pos = int(len(ratios_np))
                ratio_median = np.median(ratios_np[:dataset_split_pos])
                ratio_norm_std = np.std(ratios_np[:dataset_split_pos] / ratio_median)

                print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(ratio_median, ratio_norm_std))

            metrics = scores[domain].get_scores()

            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(metrics['abs_rel'], metrics['sq_rel'],
                                             metrics['rmse'], metrics['rmse_log'],
                                             metrics['delta1'], metrics['delta2'],
                                             metrics['delta3']) + "\\\\")

        for domain in images:
            domain_dir = os.path.join(self.log_path, 'eval_images', domain)
            os.makedirs(domain_dir, exist_ok=True)

            for i, (color_gt, depth_gt, depth_pred) in enumerate(images[domain]):
                image_path = os.path.join(domain_dir, f'img_{i}.png')

                logged_images = (
                    color_gt,
                    colors.depth_norm_image(depth_pred),
                    colors.depth_norm_image(depth_gt),
                )

                save_image(
                    torch.cat(logged_images, 2).clamp(0, 1),
                    image_path
                )

        self._log_gpu_memory()


if __name__ == "__main__":
    
    opt = DepthEvaluationArguments().parse()

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

    evaluator = DepthEvaluator(opt)
    evaluator.evaluate()
