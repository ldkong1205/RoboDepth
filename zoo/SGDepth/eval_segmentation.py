#!/usr/bin/env python3

# Python standard library
import os

# Public libraries
import torch
from torchvision.utils import save_image

# Local imports
import colors
from arguments import SegmentationEvaluationArguments
from harness import Harness


class SegmentationEvaluator(Harness):
    def _init_resampler(self, opt):
        pass

    def _init_validation(self, opt):
        self.val_num_log_images = opt.eval_num_images
        self.eval_name = opt.model_name

    def evaluate(self):
        print('Evaluate segmentation predictions:', flush=True)

        scores, images = self._run_segmentation_validation(
            self.val_num_log_images
        )

        for domain in scores:
            print('eval_name    | domain               |     miou | accuracy')

            metrics = scores[domain].get_scores()

            miou = metrics['meaniou']
            acc = metrics['meanacc']

            print(f'{self.eval_name:12} | {domain:20}  | {miou:8.3f} | {acc:8.3f}', flush=True)

        for domain in images:
            domain_dir = os.path.join(self.log_path, 'eval_images', domain)
            os.makedirs(domain_dir, exist_ok=True)

            for i, (color_gt, seg_gt, seg_pred) in enumerate(images[domain]):
                image_path = os.path.join(domain_dir, f'img_{i}.png')

                logged_images = (
                    color_gt,
                    colors.seg_idx_image(seg_pred),
                    colors.seg_idx_image(seg_gt),
                )

                save_image(
                    torch.cat(logged_images, 2).clamp(0, 1),
                    image_path
                )

        self._log_gpu_memory()

        return scores


if __name__ == "__main__":
    opt = SegmentationEvaluationArguments().parse()

    if opt.model_load is None:
        raise Exception('You must use --model-load to select a model state directory to run evaluation on')

    if opt.sys_best_effort_determinism:
        import random
        import numpy as np

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    evaluator = SegmentationEvaluator(opt)
    evaluator.evaluate()
