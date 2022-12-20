from copy import deepcopy
import numpy as np

import torch

from dataloader.eval.metrics import SegmentationRunningScore


class DCMasking(object):
    def __init__(self, masking_from_epoch, num_epochs, moving_mask_percent, masking_linear_increase):
        self.masking_from_epoch = masking_from_epoch
        self.num_epochs = num_epochs
        self.moving_mask_percent = moving_mask_percent
        self.masking_linear_increase = masking_linear_increase

        self.segmentation_input_key = ('color_aug', 0, 0)
        self.logits_key = ('segmentation_logits', 0)

        self.metric_model_moving = SegmentationRunningScore(2)

        self.iou_thresh = dict()
        self.iou_thresh['non_moving'] = 0.0
        self.iou_thresh['moving'] = 0.0

        self.iou_log = dict()
        self.iou_log['non_moving'] = list()
        self.iou_log['moving'] = list()

    def _moving_class_criterion(self, segmentation):
        # TODO this is valid for the Cityscapes class definitions and has to be adapted for other datasets
        #  to be more generic
        mask = (segmentation > 10) & (segmentation < 100)
        return mask

    def compute_segmentation_frames(self, batch, model):
        batch_masking = deepcopy(batch)

        # get the depth indices
        batch_indices = tuple([idx_batch for idx_batch, sub_batch in enumerate(batch_masking)
                               if any('depth' in purpose_tuple
                                      for purpose_tuple in sub_batch['purposes'])
                               ])

        # get the depth images
        batch_masking = tuple([sub_batch for sub_batch in batch_masking
                               if any('depth' in purpose_tuple
                                      for purpose_tuple in sub_batch['purposes'])
                               ])

        # replace the purpose to segmentation
        for idx1, sub_batch in enumerate(batch_masking):
            for idx2, purpose_tuple in enumerate(sub_batch['purposes']):
                batch_masking[idx1]['purposes'][idx2] = tuple([purpose.replace('depth', 'segmentation')
                                                               for purpose in purpose_tuple])

        # generate the correct keys and outputs
        input_image_keys = [key for key in batch_masking[0].keys() if 'color_aug' in key]
        output_segmentation_keys = [('segmentation', key[1], key[2]) for key in input_image_keys]
        outputs_masked = list(dict() for i in range(len(batch)))

        # pass all depth image frames through the network to get the segmentation outputs
        for in_key, out_key in zip(input_image_keys, output_segmentation_keys):
            wanted_keys = ['domain', 'purposes', 'domain_idx', in_key]
            batch_masking_key = deepcopy(batch_masking)
            batch_masking_key = tuple([{key: sub_batch[key] for key in sub_batch.keys()
                                        if key in wanted_keys}
                                       for sub_batch in batch_masking_key])
            for idx1 in range(len(batch_masking_key)):
                batch_masking_key[idx1][self.segmentation_input_key] = \
                    batch_masking_key[idx1][in_key].clone()
                if in_key != self.segmentation_input_key:
                    del batch_masking_key[idx1][in_key]

            outputs_masked_key = model(batch_masking_key)
            cur_idx_outputs = 0
            for idx_batch in range(len(outputs_masked)):
                if idx_batch in batch_indices:
                    outputs_masked[idx_batch][out_key] = outputs_masked_key[cur_idx_outputs][self.logits_key].argmax(1)
                    cur_idx_outputs += 1
                else:
                    outputs_masked[idx_batch] = None

        outputs_masked = tuple(outputs_masked)
        return outputs_masked

    def compute_moving_mask(self, output_masked):
        """Compute moving mask and iou
                """
        segmentation = output_masked[("segmentation", 0, 0)]
        # Create empty mask
        moving_mask_combined = torch.zeros(segmentation.shape).to(segmentation.device)
        # Create binary mask moving in t = 0,  movable object = 1, non_movable = 0

        # Create binary masks (moving / non-moving)
        moving_mask = dict()
        moving_mask[0] = self._moving_class_criterion(segmentation).float()
        for key in output_masked.keys():
            if key[0] == "segmentation_warped":
                moving_mask[key[1]] = self._moving_class_criterion(output_masked[("segmentation_warped", key[1], 0)])

        # Calculate IoU for each frame separately
        for i in range(moving_mask[0].shape[0]):

            # Average score over frames
            for frame_id in moving_mask.keys():
                if frame_id == 0:
                    continue
                # For binary class
                self.metric_model_moving.update(
                    np.array(moving_mask[frame_id][i].cpu()), np.array(moving_mask[0][i].cpu()))

            scores = self.metric_model_moving.get_scores()

            if not np.isnan(scores['iou'][0]):
                self.iou_log['non_moving'].append(scores['iou'][0])
            if not np.isnan(scores['iou'][1]):
                self.iou_log['moving'].append(scores['iou'][1])
                # Calculate Mask if scores of moving objects is not NaN
                # mask every moving class, were the iou is smaller than threshold
                if scores['iou'][1] < self.iou_thresh['moving']:
                    # Add moving mask of t = 0
                    moving_mask_combined[i] += self._moving_class_criterion(segmentation[i]).float()
                    # Add moving mask of segmentation mask of t!=0 warped to t=0
                    for frame_id in moving_mask.keys():
                        if frame_id == 0:
                            continue
                        moving_mask_combined[i] += self._moving_class_criterion(
                            output_masked[("segmentation_warped", frame_id, 0)][i]).float()
                        # mask moving in t != 0
            self.metric_model_moving.reset()
        # movable object = 0, non_movable = 1
        output_masked['moving_mask'] = (moving_mask_combined < 1).float().detach()

    def clear_iou_log(self):
        self.iou_log = dict()
        self.iou_log['non_moving'] = list()
        self.iou_log['moving'] = list()

    def calculate_iou_threshold(self, current_epoch):
        if self.masking_from_epoch <= current_epoch:
            self.iou_thresh = dict()
            if self.masking_linear_increase:
                percentage = 1 - (1 / (self.num_epochs - 1 - self.masking_from_epoch) * (
                            current_epoch + 1 - self.masking_from_epoch))  # Mask 100 % to 0 %
            else:
                percentage = self.moving_mask_percent
            try:
                self.iou_thresh['non_moving'] = np.percentile(self.iou_log['non_moving'], (100 * percentage)).item()
            except Exception as e:
                self.iou_thresh['non_moving'] = 0.0
                print('Error calculating percentile of non_moving')
                print(e)
            try:
                self.iou_thresh['moving'] = np.percentile(self.iou_log['moving'], (100 * percentage)).item()
            except Exception as e:
                self.iou_thresh['moving'] = 0.0
                print('Error calculating percentile of moving')
                print(e)
