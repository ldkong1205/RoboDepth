#!/usr/bin/env python3

# Python standard library
import os

# Public libraries
import torch
import tensorboardX as tensorboard

# Local imports
import colors
import loaders, loaders.segmentation, loaders.depth

from arguments import TrainingArguments
from timer import Timer
from harness import Harness
from losses import DepthLosses, SegLosses
from dc_masking import DCMasking


class Trainer(Harness):
    def _init_losses(self, opt):
        self.depth_losses = DepthLosses(
            self.device,
            opt.depth_disable_automasking,
            opt.depth_avg_reprojection,
            opt.depth_disparity_smoothness,
        )

        self.seg_losses = SegLosses(self.device)

        self.masking_enable = opt.masking_enable
        self.masking_from_epoch = opt.masking_from_epoch
        self.mask_calculator = DCMasking(opt.masking_from_epoch, opt.train_num_epochs, opt.moving_mask_percent,
                                         opt.masking_linear_increase)

    def _init_logging(self, opt):
        self.print_frequency = opt.train_print_frequency
        self.tb_frequency = opt.train_tb_frequency
        self.checkpoint_frequency = opt.train_checkpoint_frequency

    def _init_tensorboard(self, opt):
        self.writers = dict(
            (mode, tensorboard.SummaryWriter(os.path.join(self.log_path, mode)))
            for mode in ('train', 'validation', 'images')
        )

    def _init_train_loaders(self, opt):
        print('Loading training dataset metadata:', flush=True)

        # Directly call the loader setup functions from loaders/depth.py
        # and loaders/segmentation.py that are passed in via --loaders_depth
        # and --loaders_segmentation.
        # If you read this while researching the cause of an Exception make sure
        # there is a function defined in loaders/*.py that exactly matches the
        # name you specified on the commandline.
        # This design was choosen as it can be quite tricky to get all the transforms
        # and loader configurations quite right for each dataset, so training/validation
        # should fail when a dataset is selected that we don't know yet how to handle.

        depth_train_loaders = list(
            getattr(loaders.depth, loader_name)(
                resize_height=opt.depth_resize_height,
                resize_width=opt.depth_resize_width,
                crop_height=opt.depth_crop_height,
                crop_width=opt.depth_crop_width,
                batch_size=opt.depth_training_batch_size,
                num_workers=opt.sys_num_workers,
            )
            for loader_name in opt.depth_training_loaders.split(',') if (loader_name != '')
        )

        segmentation_train_loaders = list(
            getattr(loaders.segmentation, loader_name)(
                resize_height=opt.segmentation_resize_height,
                resize_width=opt.segmentation_resize_width,
                crop_height=opt.segmentation_crop_height,
                crop_width=opt.segmentation_crop_width,
                batch_size=opt.segmentation_training_batch_size,
                num_workers=opt.sys_num_workers
            )
            for loader_name in opt.segmentation_training_loaders.split(',') if (loader_name != '')
        )

        self.train_loaders = loaders.FixedLengthLoaderList(
            depth_train_loaders + segmentation_train_loaders,
            opt.train_batches_per_epoch
        )

    def _init_training(self, opt):
        self.num_epochs = opt.train_num_epochs

    def _flush_logging(self):
        print('', end='', flush=True)

        for writer in self.writers.values():
            writer.flush()

    def _log_depth(self, domain_name, batch_idx, inputs, outputs, losses):
        with torch.no_grad():
            depth = outputs['depth', 0, 0].cpu()

            # Multiple times each epoch ...
            if (batch_idx % self.tb_frequency) == 0:
                # ... log the averaged loss to tensorboard
                self.writers['train'].add_scalar(
                    f"{domain_name}_loss", losses["loss_depth"].cpu(), self.state.step
                )

                # ... log the depth distribution parameters to make sure they look healty
                self.writers['train'].add_scalar(
                    f"{domain_name}_depth_mean", depth.mean(), self.state.step
                )

                self.writers['train'].add_scalar(
                    f"{domain_name}_depth_std", depth.std(), self.state.step
                )

            # A few times each epoch ...
            if (batch_idx % self.print_frequency) == 0:
                print(f"  - {domain_name} losses at epoch {self.state.epoch} (batch {batch_idx}):")

                # ... log the reprojection + smoothness loss
                loss = losses["loss_depth"].cpu()
                print(f"    - avg {loss:.4f}")

            # Once at the start of each epoch ...
            if batch_idx == 0:
                from_prev = outputs['color', -1, 0].cpu() if (('color', -1, 0) in outputs) else 0
                from_next = outputs['color', 1, 0].cpu() if (('color', 1, 0) in outputs) else 0
                target = inputs['color', 0, 0].cpu() if (('color', 0, 0) in inputs) else 0
                surface_normal = outputs['normals_pointcloud', 0].cpu() if (('normals_pointcloud', 0) in outputs) else 0

                # ... log the depth prediction, target and reprojected images
                logged_images = (
                    colors.depth_norm_image(depth),
                    colors.surface_normal_image(surface_normal),
                    target, from_prev, from_next,
                )

                self.writers['images'].add_images(
                    f"{domain_name}_images",
                    torch.cat(logged_images, 2).clamp(0,1),
                    self.state.step
                )

    def _log_seg(self, domain_name, batch_idx, inputs, outputs, losses):
        with torch.no_grad():
            # Multiple times each epoch ...
            if (batch_idx % self.tb_frequency) == 0:
                # ... log the segmentation loss to tensorboard
                self.writers['train'].add_scalar(
                    f"{domain_name}_loss", losses["loss_seg"].cpu(), self.state.step
                )

            # A few times each epoch ...
            if (batch_idx % self.print_frequency) == 0:
                print(f"  - {domain_name} losses at epoch {self.state.epoch} (batch {batch_idx}):")

                # ... log the cross entropy loss
                loss_seg = losses["loss_seg"].cpu()
                print(f"    - cross_entropy: {loss_seg:.4f}")

            # Once at the start of each epoch ...
            if batch_idx == 0:
                seg = outputs['segmentation_logits', 0].softmax(1).cpu()
                gt = inputs['segmentation', 0, 0][:, 0, :, :].cpu().long()
                src = inputs['color', 0, 0].cpu()

                logged_images = (
                    colors.seg_prob_image(seg),
                    colors.seg_idx_image(gt),
                    src
                )

                self.writers['images'].add_images(
                    f"{domain_name}_images",
                    torch.cat(logged_images, 2),
                    self.state.step
                )

    def _process_batch_depth(self, dataset, output, output_masked, batch_idx, domain_name):
        if ('disp', 0) not in output:
            return 0

        # Process depth output, mask outputs are added to output_masked
        predictions_depth = self.resample.warp_images(dataset, output, output_masked)
        output.update(predictions_depth)

        if output_masked is not None:
            self.mask_calculator.compute_moving_mask(output_masked)

        losses_depth = self.depth_losses.compute_losses(dataset, output, output_masked)

        self._log_depth(domain_name, batch_idx, dataset, output, losses_depth)

        return losses_depth["loss_depth"]

    def _process_batch_seg(self, dataset, output, batch_idx, domain_name):
        if ('segmentation_logits', 0) not in output:
            return 0

        losses_seg = self.seg_losses.seg_losses(dataset, output)

        self._log_seg(domain_name, batch_idx, dataset, output, losses_seg)

        return losses_seg["loss_seg"]

    # def _process_batch_domain(self, dataset, output, batch_idx, domain_name):
    #     if ('domain_logits', 0) not in output:
    #         return 0
    #
    #     losses_domain = self.domain_losses.domain_losses(dataset, output)
    #
    #     self._log_domain(domain_name, batch_idx, dataset, output, losses_domain)
    #
    #     return losses_domain["loss_domain"]

    def _run_epoch(self):
        print(f"Epoch {self.state.epoch}:")

        self.mask_calculator.clear_iou_log()

        with self.state.model_manager.get_train() as model:
            timer = Timer()

            timer.enter('loading')
            for batch_idx, batch in enumerate(self.train_loaders):

                # Apply gradient scaling depending on which strategy was chosen and log to tensorboard
                gs_depth, gs_seg = model.get_gradient_scales()
                model.set_gradient_scales(gs_depth, gs_seg)

                timer.enter(f"optimizer")
                self.state.optimizer.zero_grad()

                timer.enter(f"transfer")
                batch = self._batch_to_device(batch)

                timer.enter('forward')

                # Compute the additional segmentation masks if masking is enabled
                if self.masking_enable and self.masking_from_epoch <= self.state.epoch:
                    with torch.no_grad(), self.state.model_manager.get_eval() as model_eval:
                        outputs_masked = self.mask_calculator.compute_segmentation_frames(batch, model_eval)
                else:
                    outputs_masked = tuple(None for i in range(len(batch)))

                outputs = model(batch)

                loss_depth = 0
                loss_seg = 0

                for dataset, output, output_masked in zip(batch, outputs, outputs_masked):
                    domain_name = dataset['domain'][0]

                    # Calculate loss for the depth prediction
                    loss_depth += self._process_batch_depth(dataset, output, output_masked, batch_idx, domain_name)

                    # Calculate loss for the segmentation prediction
                    loss_seg += self._process_batch_seg(dataset, output, batch_idx, domain_name)

                timer.enter(f"optimizer")

                loss = loss_depth + loss_seg
                loss.backward()

                self.state.optimizer.step()  # performs a single optimization step

                if (batch_idx % self.print_frequency) == 0:
                    print('  - Breakdown of time spent this epoch:')
                    for category, t in timer.items():
                        print(f'    - {category}: {t:.3f}', flush=True)

                self.state.step += 1

                timer.enter('loading')

        self.mask_calculator.calculate_iou_threshold(current_epoch=self.state.epoch)
        self.state.lr_scheduler.step()

    def _run_validation(self):
        print(f'Validation scores for epoch {self.state.epoch}:')

        depth_scores, depth_ratios, _ = self._run_depth_validation()

        segmentation_scores, _ = self._run_segmentation_validation()

        for domain, score in depth_scores.items():
            metrics = score.get_scores()

            print(f'  - {domain}:')

            for metric in sorted(metrics):
                value = metrics[metric]

                print(f'    - {metric}: {value:.4f}')

                self.writers['validation'].add_scalar(
                    f"{domain}_{metric}", value, self.state.step
                )

        for domain, ratios in depth_ratios.items():
            if len(ratios) > 0:
                ratios_tch = torch.tensor(ratios)
                ratio_median = ratios_tch.median()
                ratio_norm_std = (ratios_tch / ratio_median).std()

                print(f'    - ratio_median: {ratio_median:.4f}')
                print(f'    - ratio_norm_std: {ratio_norm_std:.4f}')

                self.writers['validation'].add_scalar(
                    f"{domain}_ratio_median", ratio_median, self.state.step
                )

                self.writers['validation'].add_scalar(
                    f"{domain}_ratio_norm_std", ratio_norm_std, self.state.step
                )

        for domain, score in segmentation_scores.items():
            metrics = score.get_scores()

            print(f'  - {domain}:')

            for metric in sorted(metrics):
                value = metrics[metric]

                if metric in ('iou', 'acc', 'prec'):
                    # ignore non-scalars
                    continue

                print(f'    - {metric}: {value:.4f}')

                self.writers['validation'].add_scalar(
                    f"{domain}_{metric}", value, self.state.step
                )

    def train(self):
        while self.state.epoch < self.num_epochs:
            self._run_epoch()
            self._run_validation()
            self._flush_logging()

            self.state.epoch += 1

            # Save after save frequency
            if (self.state.epoch % self.checkpoint_frequency) == 0:
                self.state.store_checkpoint()

        # Save at end of training
        self.state.store_checkpoint()

        print('Completed without errors', flush=True)
        self._log_gpu_memory()


if __name__ == "__main__":
    opt = TrainingArguments().parse()

    if opt.sys_best_effort_determinism:
        import numpy as np
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    trainer = Trainer(opt)
    trainer.train()
