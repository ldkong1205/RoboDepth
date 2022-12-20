import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.cma import CMA
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.resnet_encoder import ResnetEncoder
from networks.seg_decoder import SegDecoder
from utils.depth_utils import BackprojectDepth, Project3D, disp_to_depth, SSIM, get_smooth_loss, \
    transformation_from_parameters


class TrainerParallel(nn.Module):
    def __init__(self, options):
        super(TrainerParallel, self).__init__()
        self.opt = options
        self.epoch = 0

        self.models = nn.ModuleDict({
            'encoder': ResnetEncoder(num_layers=self.opt.num_layers, pretrained=self.opt.pretrained,
                                     ),
            'pose_encoder': ResnetEncoder(num_layers=18, num_input_images=2,
                                          pretrained=self.opt.pretrained),
        })

        self.models.update({
            'pose': PoseDecoder(self.models['pose_encoder'].num_ch_enc)
        })

        if not self.opt.no_cma:
            self.models.update({
                'decoder': CMA(self.models['encoder'].num_ch_enc, opt=self.opt)
            })

        else:
            self.models.update({
                'depth': DepthDecoder(self.models['encoder'].num_ch_enc,
                                      scales=self.opt.scales, opt=self.opt),
            })

            if self.opt.semantic_distil is not None:
                self.models['seg'] = SegDecoder(self.models['encoder'].num_ch_enc, scales=[0])

        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)

        self.ssim = SSIM()

        self.parameters_to_train = []
        for model in self.models:
            self.parameters_to_train += list(self.models[model].parameters())
        self.loss_functions = {}
        self.masking_functions = []

        self.loss_functions = {self.compute_reprojection: self.opt.reprojection}

        if self.opt.disparity_smoothness:
            self.loss_functions[self.compute_smoothness] = self.opt.disparity_smoothness

        if self.opt.semantic_distil:
            self.loss_functions[self.compute_semantic_distil] = self.opt.semantic_distil

        if self.opt.sgt:
            self.loss_functions[self.compute_sgt_loss] = self.opt.sgt

    def forward(self, inputs):
        losses = {}
        loss = 0
        outputs = self.compute_outputs(inputs)

        for loss_function, loss_weight in self.loss_functions.items():
            loss_type = loss_function.__name__
            losses[loss_type] = loss_function(inputs, outputs) * loss_weight

        for loss_type, value in losses.items():
            to_optim = value.mean()
            loss += to_optim

        losses["loss"] = loss
        for key, value in outputs.items():
            if key != 'loss':
                outputs[key] = value.data
        return losses, outputs

    def compute_outputs(self, inputs):
        outputs = {}
        features = {}
        center = inputs[("color_aug", 0, 0)]

        features[0] = self.models["encoder"](center)
        for frame_id in self.opt.frame_ids[1:]:
            color_aug = inputs[("color_aug", frame_id, 0)]

            if frame_id == 1:
                pose_inputs = torch.cat([center, color_aug], dim=1)
            elif frame_id == -1:
                pose_inputs = torch.cat([color_aug, center], dim=1)
            else:
                raise Exception("invalid frame_ids")

            if pose_inputs.shape[3] > 640:
                pose_inputs = F.interpolate(pose_inputs, size=(192, 640), mode='bilinear')
            pose_features = self.models['pose_encoder'](pose_inputs)
            axisangle, translation = self.models['pose']([pose_features])
            outputs[("axisangle", frame_id)] = axisangle
            T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=frame_id < 0)
            outputs[("T", frame_id)] = T

        if not self.opt.no_cma:
            disp, seg = self.models['decoder'](features[0])
            outputs.update(disp)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs[("disp", s)], (self.opt.height, self.opt.width), mode='bilinear', align_corners=False)
                else:
                    disp = outputs[("disp", s)]
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, s)] = depth
            outputs.update(seg)
        else:
            if self.opt.semantic_distil is not None:
                seg = self.models["seg"](features[0])
                outputs.update(seg)

            outputs.update(self.models["depth"](features[0]))
            _, depth = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs[("disp", s)], (self.opt.height, self.opt.width), mode='bilinear',
                                         align_corners=False)
                else:
                    disp = outputs[("disp", s)]
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, s)] = depth

        for frame_id in self.opt.frame_ids[1:]:
            for s in self.opt.scales:
                cam_points = self.backproject_depth(outputs[("depth", 0, s)], inputs[("inv_K")])
                pix_coords, next_depth = self.project_3d(cam_points, inputs[("K")], outputs[("T", frame_id)])
                outputs[("sample", frame_id, s)] = pix_coords
                outputs[("next_depth", frame_id, s)] = next_depth

        return outputs

    def compute_affinity(self, feature, kernel_size):
        pad = kernel_size // 2
        feature = F.normalize(feature, dim=1)
        unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        similarity = (feature * unfolded).sum(dim=1, keepdim=True)
        eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
        return affinity

    def compute_sgt_loss(self, inputs, outputs):

        assert len(self.opt.sgt_layers) == len(self.opt.sgt_kernel_size)
        seg_target = inputs[("seg", 0, 0)]
        _, _, h, w = seg_target.shape
        total_loss = 0

        for s, kernel_size in zip(self.opt.sgt_layers, self.opt.sgt_kernel_size):
            pad = kernel_size // 2
            h = self.opt.height // 2 ** s
            w = self.opt.width // 2 ** s
            seg = F.interpolate(seg_target, size=(h, w), mode='nearest')
            center = seg
            padded = F.pad(center, [pad] * 4, value=-1)
            aggregated_label = torch.zeros(*(center.shape + (kernel_size, kernel_size))).to(center.device)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    shifted = padded[:, :, 0 + i: h + i, 0 + j:w + j]
                    label = center == shifted
                    aggregated_label[:, :, :, :, i, j] = label
            aggregated_label = aggregated_label.float()
            pos_idx = (aggregated_label == 1).float()
            neg_idx = (aggregated_label == 0).float()
            pos_idx_num = pos_idx.sum(dim=-1).sum(dim=-1)
            neg_idx_num = neg_idx.sum(dim=-1).sum(dim=-1)

            boundary_region = (pos_idx_num >= kernel_size - 1) & (
                    neg_idx_num >= kernel_size - 1)
            non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0)

            if s == min(self.opt.sgt_layers):
                outputs[('boundary', s)] = boundary_region.data
                outputs[('non_boundary', s)] = non_boundary_region.data

            feature = outputs[('d_feature', s)]
            affinity = self.compute_affinity(feature, kernel_size=kernel_size)
            pos_dist = (pos_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                       pos_idx.sum(dim=-1).sum(dim=-1)[
                           boundary_region]
            neg_dist = (neg_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                       neg_idx.sum(dim=-1).sum(dim=-1)[
                           boundary_region]
            zeros = torch.zeros(pos_dist.shape).to(pos_dist.device)
            loss = torch.max(zeros, pos_dist - neg_dist + self.opt.sgt_margin)

            total_loss += loss.mean() / (2 ** s)

        return total_loss

    def compute_reprojection(self, inputs, outputs):

        total_losses = 0
        target = inputs[("color", 0, 0)]
        for s in self.opt.scales:
            losses = []
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                outputs[("color", frame_id, s)] = pred = F.grid_sample(inputs[("color", frame_id, 0)],
                                                                       outputs[("sample", frame_id, s)],
                                                                       padding_mode="border", align_corners=True
                                                                       )
                reprojection_loss = self.reprojection_loss(pred, target)
                outputs[("reprojection_loss", frame_id)] = reprojection_loss

                losses.append(reprojection_loss)

                pred = inputs[("color", frame_id, 0)]
                identity_reprojection_losses.append(self.reprojection_loss(pred, target))

            losses = torch.cat(losses, dim=1)

            # Apply automask in Monodepth2
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, dim=1)
            identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape).cuda(
                device=target.device) * 0.00001
            combined = torch.cat([losses, identity_reprojection_losses], dim=1)
            to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)

            total_losses += to_optimise.mean() / (2 ** s)

        return total_losses

    def compute_semantic_distil(self, inputs, outputs):

        total_loss = 0
        # for s in self.opt.scales:

        scales = [0]

        for s in scales:
            seg_target = inputs[("seg", 0, s)].long().squeeze(1)
            seg_pred = outputs[("seg_logits", s)]
            weights = seg_target.sum(1, keepdim=True).float()
            ignore_mask = (weights == 0)
            weights[ignore_mask] = 1
            seg_loss = F.cross_entropy(seg_pred, seg_target, reduction='none')
            total_loss += seg_loss.mean() / (2 ** s)

        return total_loss

    def compute_smoothness(self, inputs, outputs):
        total_loss = 0
        for s in self.opt.scales:
            disp = outputs[("disp", s)]
            color = inputs[("color", 0, s)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            total_loss += smooth_loss / (2 ** s)

        return total_loss

    def reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return loss
