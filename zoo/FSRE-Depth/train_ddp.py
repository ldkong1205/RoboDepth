import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DistributedSampler

from datasets.kitti_dataset import KittiDataset
from trainer_parallel import TrainerParallel
from utils import *
from utils.depth_utils import compute_depth_errors
from utils.seg_utils import decode_seg_map


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class Trainer:
    def __init__(self, options):
        self.opt = options
        now = datetime.now().strftime("%m-%d-%Y")
        self.log_path = os.path.join(self.opt.log_dir, now, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.trainer = TrainerParallel(options)
        self.model_optimizer = optim.Adam(self.trainer.parameters_to_train, self.opt.learning_rate)
        self.epoch = 0
        self.step = 0
        self.is_best = False
        self.best_loss = 1e9

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)

        # data
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))

        train_dataset = KittiDataset(
            height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, filenames=train_filenames, data_path=self.opt.data_path, is_train=True,
            num_scales=len(self.opt.scales))

        val_filenames = readlines(fpath.format("val"))
        val_dataset = KittiDataset(
            height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, filenames=val_filenames, data_path=self.opt.data_path, is_train=False,
            num_scales=len(self.opt.scales))
        if self.opt.local_rank == 0:
            self.writers = {}
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        torch.cuda.set_device(self.opt.local_rank)
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        print("WORLD SIZE: ", self.world_size)
        self.trainer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.trainer)
        self.trainer = self.trainer.cuda()
        print(f'PROC {self.opt.local_rank}: LOAD MODEL ON GPU {next(self.trainer.parameters()).device} ')
        self.trainer = DDP(self.trainer, device_ids=[self.opt.local_rank],
                           output_device=self.opt.local_rank, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            num_workers=self.opt.batch_size, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.batch_size, pin_memory=True, drop_last=True, sampler=val_sampler)
        print(f'PROC {self.opt.local_rank}: SET TRAIN LOADER. SIZE {len(self.train_loader)}')
        print(f'PROC {self.opt.local_rank}: SET VAL LOADER. SIZE {len(self.val_loader)}')

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size // self.world_size * self.opt.num_epochs
        if self.opt.local_rank == 0:
            self.save_opts()

    def lr_decay(self):
        print('decay learning rate')
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] *= self.opt.decay_rate

    def set_train(self):
        """Convert all models to training mode
        """
        self.trainer.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.trainer.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            if self.epoch in self.opt.lr_decay:
                self.lr_decay()
            self.run_epoch()
            self.val()

            if self.opt.local_rank == 0:
                self.save_model()
            dist.barrier()

            if isinstance(self.trainer, DDP):
                self.trainer.module.epoch += 1
            else:
                self.trainer.epoch += 1

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        data_loading_time = 0
        gpu_time = 0
        before_op_time = time.time()

        if self.opt.local_rank == 0:
            train_loss = None

        for batch_idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()

            losses, outputs = self.trainer(inputs)
            losses['loss'] = losses['loss'].mean()

            if self.opt.split != 'test':
                self.model_optimizer.zero_grad()
                losses["loss"].backward()

                self.model_optimizer.step()

            duration = time.time() - before_op_time
            gpu_time += duration

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % 250 == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            # late_phase = self.step % (len(self.train_loader) // self.opt.batch_size // 2) == 0

            for loss_type in losses:
                losses[loss_type] = reduce_tensor(losses[loss_type].data, self.world_size)

            if self.opt.local_rank == 0:
                if early_phase or late_phase:
                    self.log_time(batch_idx, duration, losses, data_loading_time, gpu_time)
                    self.log("train", inputs, outputs, {})
                    data_loading_time = 0
                    gpu_time = 0
            self.step += 1
            before_op_time = time.time()

            if self.opt.local_rank == 0:
                if train_loss is None:
                    train_loss = {loss_type: float(losses[loss_type].data.mean()) for loss_type in losses}
                else:
                    for loss_type in losses:
                        train_loss[loss_type] += float(losses[loss_type].data.mean())

        if self.opt.local_rank == 0:
            for key in train_loss:
                train_loss[key] /= len(self.train_loader)
            self.log("train", inputs, outputs, train_loss)

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        if self.opt.local_rank == 0:
            val_loss = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()
                losses, outputs = self.trainer(inputs)

                for loss_type in losses:
                    losses[loss_type] = reduce_tensor(losses[loss_type].data, self.world_size)

                if self.opt.local_rank == 0:
                    if val_loss is None:
                        val_loss = {loss_type: float(losses[loss_type].data.mean()) for loss_type in losses}
                    else:
                        for loss_type in val_loss:
                            val_loss[loss_type] += float(losses[loss_type].data.mean())

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, val_loss)

            if self.opt.local_rank == 0:
                for key in val_loss:
                    val_loss[key] /= len(self.val_loader)
                self.log("val", inputs, outputs, val_loss)
                if val_loss['loss'] < self.best_loss:
                    self.is_best = True
                    self.best_loss = val_loss['loss']
                else:
                    self.is_best = False

        del inputs, outputs, losses

        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            if metric in losses:
                losses[metric] += depth_errors[i].data.cpu()
            else:
                losses[metric] = depth_errors[i].data.cpu()

    def log_time(self, batch_idx, duration, losses, data_time, gpu_time):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {} | CPU/GPU time: {:0.1f}s/{:0.1f}s"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses['loss'].data.cpu().mean(),
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  data_time, gpu_time))
        print(str({item: round(float(losses[item].data.cpu().mean()), 6) for item in losses}))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        s = 0
        for j in range(min(3, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j], self.step)

                if frame_id != 0:

                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j], self.step)


                else:
                    disp = outputs[("disp", s)].data
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(disp[j]), self.step)

                    if self.opt.semantic_distil:
                        if s == 0:
                            writer.add_image("semantic_target_{}".format(j),
                                             decode_seg_map(inputs[("seg", 0, 0)][j].data),
                                             self.step)
                            logits = outputs[("seg_logits", 0)].argmax(dim=1, keepdim=True)
                            writer.add_image("semantic_pred_{}".format(j),
                                             decode_seg_map(logits[j].data),
                                             self.step)

                    if self.opt.sgt:
                        layer = min(self.opt.sgt_layers)
                        boundary_region = outputs[("boundary", layer)]
                        non_boundary_region = outputs[("non_boundary", layer)]
                        writer.add_image("boundary_{}/{}".format(layer, j), boundary_region[j], self.step)
                        writer.add_image("non_boundary_{}/{}".format(layer, j), non_boundary_region[j], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if self.is_best:
            save_folder += '_best'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if isinstance(self.trainer, DDP):
            models = self.trainer.module.models
        else:
            models = self.trainer.models

        for model_name, model in models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            temp = {key.replace('module.', ''): to_save[key] for key in to_save}
            to_save = temp

            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['epoch'] = self.epoch
                to_save['step'] = self.step
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.trainer.models:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.trainer.models[n].state_dict()
            pretrained_dict = torch.load(path)
            if n == 'encoder':
                self.epoch = pretrained_dict['epoch'] + 1
                if isinstance(self.trainer, DDP):
                    self.trainer.module.epoch = self.epoch
                else:
                    self.trainer.epoch = self.epoch

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.trainer.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            for state in self.model_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == '__main__':
    from options import Options

    options = Options()
    opts = options.parse()

    if __name__ == "__main__":
        trainer = Trainer(opts)
        trainer.train()
