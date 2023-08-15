# Adapted from https://github.com/weiyithu/SurroundDepth

from __future__ import absolute_import, division, print_function

import os
import cv2
import time
import numpy as np
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import DataLoader
from mmcv.utils import ProgressBar

from layers import disp_to_depth, post_process_inv_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = indices[self.rank:self.total_size:self.num_replicas]
    
        return iter(indices)


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def my_collate(batch):
    batch_new = {}
    keys_list = list(batch[0].keys())
    special_key_list = ['id', 'match_spatial']

    for key in keys_list: 
        if key not in special_key_list:
            batch_new[key] = [item[key] for item in batch]
            batch_new[key] = torch.cat(batch_new[key], axis=0)
        else:
            batch_new[key] = []
            for item in batch:
                for value in item[key]:
                    batch_new[key].append(value)

    return batch_new


class LogPrint:
    def __init__(self, log_path) -> None:
        self.log_path = log_path
        if not os.path.exists(log_path): os.makedirs(log_path)
        
    def __call__(self, string):
        print(string)
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(string + '\n')


def infer(opt):
    """Evaluates a pretrained model using a specified test set
    """

    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank, world_size = get_dist_info()
    
    if rank == 0:
        log = LogPrint(opt.log_dir)

    errors = {}
    eval_types = ['scale-ambiguous', 'scale-aware']
    for eval_type in eval_types:
        errors[eval_type] = {}

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        if local_rank == 0:
            log("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        # device
        device = torch.device("cuda", local_rank)

        encoder_dict = torch.load(encoder_path, map_location=device)
        
        dataset = datasets.NuscDataset(opt, opt.height, opt.width,
                              opt.frame_ids, 4, is_train=False)
        
        val_sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, collate_fn=my_collate, num_workers=opt.num_workers, sampler=val_sampler, pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        if local_rank == 0:
            log("-> Computing predictions with size {}x{}".format(
                encoder_dict['width'], encoder_dict['height']))

        ratios_median = []
        if rank == 0:
            progress_bar = ProgressBar(len(dataloader.dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        with torch.no_grad():
            for data in dataloader:
                
                input_color = data[("color", 0, 0)].cuda()
                gt_depths = data["depth"].cpu().numpy()
                
                camera_ids = data["id"]
                
                features = encoder(input_color)
                output = depth_decoder(features)

                pred_disps_tensor, pred_depths = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

                input_color_flip = torch.flip(input_color, [3])
                features_flip = encoder(input_color_flip)
                output_flip = depth_decoder(features_flip)

                pred_disps_flip_tensor, pred_depths_flip = disp_to_depth(output_flip[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disps_flip = post_process_inv_depth(pred_disps_tensor, pred_disps_flip_tensor)
                pred_disps = pred_disps_flip.cpu()[:, 0].numpy()

                for i in range(pred_disps.shape[0]):
                    camera_id = camera_ids[i]
                    if camera_id not in list(errors['scale-aware']):
                        errors['scale-aware'][camera_id] = []
                        errors['scale-ambiguous'][camera_id] = []

                    gt_depth = gt_depths[i]
                    gt_height, gt_width = gt_depth.shape[:2]
    
                    pred_disp = pred_disps[i]
                    pred_depth = 1 / pred_disp                   
                    pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

                    mask = np.logical_and(gt_depth > opt.min_depth, gt_depth < opt.max_depth)
                    
     
                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]
                    
                    
                    ratio_median = np.median(gt_depth) / np.median(pred_depth)
                    ratios_median.append(ratio_median)
                    pred_depth_median = pred_depth.copy()*ratio_median
        
                    pred_depth_median[pred_depth_median < opt.min_depth] = opt.min_depth
                    pred_depth_median[pred_depth_median > opt.max_depth] = opt.max_depth
        
                    errors['scale-ambiguous'][camera_id].append(compute_errors(gt_depth, pred_depth_median))
                    
                    
                    pred_depth[pred_depth < opt.min_depth] = opt.min_depth
                    pred_depth[pred_depth > opt.max_depth] = opt.max_depth
        
                    errors['scale-aware'][camera_id].append(compute_errors(gt_depth, pred_depth))
                
                if rank == 0:
                    for _ in range(opt.batch_size * world_size):
                        progress_bar.update()
    
        for eval_type in eval_types:
            for camera_id in errors[eval_type].keys():
                errors[eval_type][camera_id] = np.array(errors[eval_type][camera_id])

        folder = os.path.join(opt.log_dir, 'eval')
        savepath = os.path.join(folder, '{}.pkl'.format(local_rank))
        if not os.path.exists(folder): os.makedirs(folder)
        with open(savepath, 'wb') as f:
            pickle.dump(errors, f)
        
        if local_rank == 0:
            log('median: {}'.format(np.array(ratios_median).mean()))


def evaluation(opt):
    
    rank, world_size = get_dist_info()

    if rank == 0: 
        log = LogPrint(opt.log_dir)
        log("-> Evaluating ")

        errors = {}
        eval_types = ['scale-ambiguous', 'scale-aware']
        for eval_type in eval_types:
            errors[eval_type] = {}
        
        for i in range(world_size):
            while not os.path.exists(os.path.join(opt.log_dir, 'eval', '{}.pkl'.format(i))):
                time.sleep(1)
            time.sleep(5)
            with open(os.path.join(opt.log_dir, 'eval', '{}.pkl'.format(i)), 'rb') as f:
                errors_i = pickle.load(f)
                for eval_type in eval_types:
                    for camera_id in errors_i[eval_type].keys():
                        if camera_id not in errors[eval_type].keys():
                            errors[eval_type][camera_id] = []

                        errors[eval_type][camera_id].append(errors_i[eval_type][camera_id])

        
        num_sum = 0
        for eval_type in eval_types:
            for camera_id in errors[eval_type].keys():
                errors[eval_type][camera_id] = np.concatenate(errors[eval_type][camera_id], axis=0)
            
                if eval_type == 'scale-aware':
                    num_sum += errors[eval_type][camera_id].shape[0]

                errors[eval_type][camera_id] = errors[eval_type][camera_id].mean(0)

        # assert num_sum == 6019 * 6
        os.system('rm {}/*'.format(os.path.join(opt.log_dir, 'eval')))

        for eval_type in eval_types:
            log("{} evaluation:".format(eval_type))
            mean_errors_sum = 0
            for camera_id in errors[eval_type].keys():
                mean_errors_sum += errors[eval_type][camera_id]
            mean_errors_sum /= len(errors[eval_type].keys())
            errors[eval_type]['all'] = mean_errors_sum

            for camera_id in errors[eval_type].keys():
                mean_errors = errors[eval_type][camera_id]
                log(camera_id)
                log(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                log(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


if __name__ == "__main__":
    options = MonodepthOptions()
    # TODO: the inference script is slow
    # even use 4 x A100 GPU
    infer(options.parse())
    evaluation(options.parse())
