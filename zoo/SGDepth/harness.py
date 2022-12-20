#!/usr/bin/env python3

# Python standard library
import json
import pickle
import os

# Public libraries
import numpy as np
import torch
import torch.nn.functional as functional

# IfN libraries
import dataloader.file_io.get_path as get_path
from dataloader.eval.metrics import DepthRunningScore, SegmentationRunningScore, PoseRunningScore

# Local imports
import loaders, loaders.segmentation, loaders.depth, loaders.pose, loaders.fns

from state_manager import StateManager
from perspective_resample import PerspectiveResampler


class Harness(object):
    def __init__(self, opt):
        print('Starting initialization', flush=True)

        self._init_device(opt)
        self._init_resampler(opt)
        self._init_losses(opt)
        self._init_log_dir(opt)
        self._init_logging(opt)
        self._init_tensorboard(opt)
        self._init_state(opt)
        self._init_train_loaders(opt)
        self._init_training(opt)
        self._init_validation_loaders(opt)
        self._init_validation(opt)
        self._save_opts(opt)

        print('Summary:')
        print(f'  - Model name: {opt.model_name}')
        print(f'  - Logging directory: {self.log_path}')
        print(f'  - Using device: {self._pretty_device_name()}')

    def _init_device(self, opt):
        cpu = not torch.cuda.is_available()
        cpu = cpu or opt.sys_cpu

        self.device = torch.device("cpu" if cpu else "cuda")

    def _init_resampler(self, opt):
        if hasattr(opt, 'depth_min_sampling_res'):
            self.resample = PerspectiveResampler(opt.model_depth_max, opt.model_depth_min, opt.depth_min_sampling_res)
        else:
            self.resample = PerspectiveResampler(opt.model_depth_max, opt.model_depth_min)

    def _init_losses(self, opt):
        pass

    def _init_log_dir(self, opt):
        path_getter = get_path.GetPath()
        log_base = path_getter.get_checkpoint_path()

        self.log_path = os.path.join(log_base, opt.experiment_class, opt.model_name)

        os.makedirs(self.log_path, exist_ok=True)

    def _init_logging(self, opt):
        pass

    def _init_tensorboard(self, opt):
        pass

    def _init_state(self, opt):
        self.state = StateManager(
            opt.experiment_class, opt.model_name, self.device, opt.model_split_pos, opt.model_num_layers,
            opt.train_depth_grad_scale, opt.train_segmentation_grad_scale,
            opt.train_weights_init, opt.model_depth_resolutions, opt.model_num_layers_pose,
            opt.train_learning_rate, opt.train_weight_decay, opt.train_scheduler_step_size
        )
        if opt.model_load is not None:
            self.state.load(opt.model_load, opt.model_disable_lr_loading)

    def _init_train_loaders(self, opt):
        pass

    def _init_training(self, opt):
        pass

    def _init_validation_loaders(self, opt):
        print('Loading validation dataset metadata:', flush=True)

        if hasattr(opt, 'depth_validation_loaders'):
            self.depth_validation_loader = loaders.ChainedLoaderList(
                getattr(loaders.depth, loader_name)(
                    img_height=opt.depth_validation_resize_height,
                    img_width=opt.depth_validation_resize_width,
                    batch_size=opt.depth_validation_batch_size,
                    num_workers=opt.sys_num_workers
                )
                for loader_name in opt.depth_validation_loaders.split(',') if (loader_name != '')
            )

        if hasattr(opt, 'pose_validation_loaders'):
            self.pose_validation_loader = loaders.ChainedLoaderList(
                getattr(loaders.pose, loader_name)(
                    img_height=opt.pose_validation_resize_height,
                    img_width=opt.pose_validation_resize_width,
                    batch_size=opt.pose_validation_batch_size,
                    num_workers=opt.sys_num_workers
                )
                for loader_name in opt.pose_validation_loaders.split(',') if (loader_name != '')
            )

        if hasattr(opt, 'segmentation_validation_loaders'):
            self.segmentation_validation_loader = loaders.ChainedLoaderList(
                getattr(loaders.segmentation, loader_name)(
                    resize_height=opt.segmentation_validation_resize_height,
                    resize_width=opt.segmentation_validation_resize_width,
                    batch_size=opt.segmentation_validation_batch_size,
                    num_workers=opt.sys_num_workers
                )
                for loader_name in opt.segmentation_validation_loaders.split(',') if (loader_name != '')
            )

    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.depth_validation_fixed_scaling

    def _pretty_device_name(self):
        dev_type = self.device.type

        dev_idx = (
            f',{self.device.index}'
            if (self.device.index is not None)
            else ''
        )

        dev_cname = (
            f' ({torch.cuda.get_device_name(self.device)})'
            if (dev_type == 'cuda')
            else ''
        )

        return f'{dev_type}{dev_idx}{dev_cname}'

    def _log_gpu_memory(self):
        if self.device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(self.device)

            print('Maximum MB of GPU memory used:')
            print(str(max_mem/(1024**2)))

    def _save_opts(self, opt):
        opt_path = os.path.join(self.log_path, 'opt.json')

        with open(opt_path, 'w') as fd:
            json.dump(vars(opt), fd, indent=2)

    def _batch_to_device(self, batch_cpu):
        batch_gpu = list()

        for dataset_cpu in batch_cpu:
            dataset_gpu = dict()

            for k, ipt in dataset_cpu.items():
                if isinstance(ipt, torch.Tensor):
                    dataset_gpu[k] = ipt.to(self.device)

                else:
                    dataset_gpu[k] = ipt

            batch_gpu.append(dataset_gpu)

        return tuple(batch_gpu)

    def _validate_batch_depth(self, model, batch, score, ratios, images):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        im_scores = list()

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)

        colors_gt = batch[0]['color', 0, -1]
        depths_gt = batch[0]['depth', 0, 0][:, 0]

        disps_pred = outputs[0]["disp", 0]
        disps_scaled_pred = self.resample.scale_disp(disps_pred)
        disps_scaled_pred = disps_scaled_pred.cpu()[:, 0]

        # Process each image from the batch separately
        for i in range(depths_gt.shape[0]):
            # If you are here due to an exception, make sure that your loader uses
            # AddKeyValue('domain', domain_name), AddKeyValue('validation_mask', mask_fn)
            # and AddKeyValue('validation_clamp', clamp_fn) to add these keys to each input sample.
            # There is no sensible default, that works for all datasets,
            # so you have have to define one on a per-set basis.
            domain = batch[0]['domain'][i]
            mask_fn = loaders.fns.get(batch[0]['validation_mask'][i])
            clamp_fn = loaders.fns.get(batch[0]['validation_clamp'][i])

            color_gt = colors_gt[i].unsqueeze(0)
            depth_gt = depths_gt[i].unsqueeze(0)
            disp_scaled_pred = disps_scaled_pred[i].unsqueeze(0)

            img_height = depth_gt.shape[1]
            img_width = depth_gt.shape[2]
            disp_scaled_pred = functional.interpolate(
                disp_scaled_pred.unsqueeze(1),
                (img_height, img_width),
                align_corners=False,
                mode='bilinear'
            ).squeeze(1)
            depth_pred = 1 / disp_scaled_pred

            images.append((color_gt, depth_gt, depth_pred))

            # Datasets/splits define their own masking rules
            # delegate masking to functions defined in the loader
            mask = mask_fn(depth_gt)
            depth_pred = depth_pred[mask]
            depth_gt = depth_gt[mask]

            if self.fixed_depth_scaling != 0:
                ratio = self.fixed_depth_scaling

            else:
                median_gt = np.median(depth_gt.numpy())
                median_pred = np.median(depth_pred.numpy())

                ratio = (median_gt / median_pred).item()

            ratios.append(ratio)
            depth_pred *= ratio

            # Datasets/splits define their own prediction clamping rules
            # delegate clamping to functions defined in the loader
            depth_pred = clamp_fn(depth_pred)

            score.update(
                depth_gt.numpy(),
                depth_pred.numpy()
            )

        return im_scores

    def _validate_batch_segmentation(self, model, batch, score, images):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        im_scores = list()

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)  # forward the data through the network

        colors_gt = batch[0]['color', 0, -1]
        segs_gt = batch[0]['segmentation', 0, 0].squeeze(1).long() # shape [1,1024,2048]
        segs_pred = outputs[0]['segmentation_logits', 0] # shape [1,20,192,640] one for every class
        segs_pred = functional.interpolate(segs_pred, segs_gt[0, :, :].shape, mode='nearest') # upscale predictions

        for i in range(segs_pred.shape[0]):
            color_gt = colors_gt[i].unsqueeze(0)
            seg_gt = segs_gt[i].unsqueeze(0)
            seg_pred = segs_pred[i].unsqueeze(0)

            images.append((color_gt, seg_gt, seg_pred.argmax(1).cpu()))

            seg_pred = seg_pred.exp().cpu() # exp preds and shift to CPU
            seg_pred = seg_pred.numpy() # transform preds to np array
            seg_pred = seg_pred.argmax(1) # get the highest score for classes per pixel
            seg_gt = seg_gt.numpy() # transform gt to np array

            score.update(seg_gt, seg_pred)

        return im_scores

    def _validate_batch_pose(self, model, batch, score):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)
        poses_pred = outputs[0][("cam_T_cam", 0, 1)]
        poses_gt = batch[0][('poses', 0, -1)]

        for i in range(poses_pred.shape[0]):
            pose_gt = poses_gt[i].unsqueeze(0).cpu().numpy()
            pose_pred = poses_pred[i].squeeze(0).cpu().numpy()
            score.update(pose_gt, pose_pred)

    def _validate_batch_joint(self, model, batch, depth_score, depth_ratios, depth_images,
                              seg_score, seg_images, seg_perturbations,
                              seg_im_scores, depth_im_scores):

        # apply a perturbation onto the input image
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=255)
        batch, seg_perturbation = self.attack_model.perturb(batch, model, loss_fn)
        seg_perturbations.append(seg_perturbation)

        # pass the evaluation to the single evaluation routines
        with torch.no_grad():
            seg_im_scores.extend(
                self._validate_batch_segmentation(model, batch, seg_score, seg_images)
            )
            depth_im_scores.extend(
                self._validate_batch_depth(model, batch, depth_score, depth_ratios, depth_images)
            )

    def _run_depth_validation(self, images_to_keep=0):
        scores = dict()
        ratios = dict()
        images = dict()

        with torch.no_grad(), self.state.model_manager.get_eval() as model:
            for batch in self.depth_validation_loader:
                domain = batch[0]['domain'][0]

                if domain not in scores:
                    scores[domain] = DepthRunningScore()
                    ratios[domain] = list()
                    images[domain] = list()

                _ = self._validate_batch_depth(model, batch, scores[domain], ratios[domain], images[domain])

                images[domain] = images[domain][:images_to_keep]

        return scores, ratios, images

    def _run_pose_validation(self):
        scores = dict()

        with torch.no_grad(), self.state.model_manager.get_eval() as model:
            for batch in self.pose_validation_loader:

                domain = batch[0]['domain'][0]

                if domain not in scores:
                    scores[domain] = PoseRunningScore()

                self._validate_batch_pose(model, batch, scores[domain])

        return scores

    def _run_segmentation_validation(self, images_to_keep=0):
        scores = dict()
        images = dict()

        # torch.no_grad() = disable gradient calculation
        with torch.no_grad(), self.state.model_manager.get_eval() as model:
            for batch in self.segmentation_validation_loader:
                domain = batch[0]['domain'][0]
                num_classes = batch[0]['num_classes'][0].item()

                if domain not in scores:
                    scores[domain] = SegmentationRunningScore(num_classes)
                    images[domain] = list()

                _ = self._validate_batch_segmentation(model, batch, scores[domain], images[domain])

                images[domain] = images[domain][:images_to_keep]

        return scores, images
