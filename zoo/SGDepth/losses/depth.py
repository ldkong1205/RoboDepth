import torch
import torch.nn.functional as functional

import losses as trn_losses


class DepthLosses(object):
    def __init__(self, device, disable_automasking=False, avg_reprojection=False, disparity_smoothness=0):
        self.automasking = not disable_automasking
        self.avg_reprojection = avg_reprojection
        self.disparity_smoothness = disparity_smoothness
        self.scaling_direction = "up"
        self.masked_supervision = True

        # noinspection PyUnresolvedReferences
        self.ssim = trn_losses.SSIM().to(device)
        self.smoothness = trn_losses.SmoothnessLoss()

    def _combined_reprojection_loss(self, pred, target):
        """Computes reprojection losses between a batch of predicted and target images
        """

        # Calculate the per-color difference and the mean over all colors
        l1 = (pred - target).abs().mean(1, True)

        ssim = self.ssim(pred, target).mean(1, True)

        reprojection_loss = 0.85 * ssim + 0.15 * l1

        return reprojection_loss

    def _reprojection_losses(self, inputs, outputs, outputs_masked):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        frame_ids = tuple(frozenset(k[1] for k in outputs if k[0] == 'color'))
        resolutions = tuple(frozenset(k[2] for k in outputs if k[0] == 'color'))

        losses = dict()

        color = inputs["color", 0, 0]
        target = inputs["color", 0, 0]

        # Compute reprojection losses for the unwarped input images
        identity_reprojection_loss = tuple(
            self._combined_reprojection_loss(inputs["color", frame_id, 0], target)
            for frame_id in frame_ids
        )
        identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)

        if self.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_loss.mean(1, keepdim=True)

        for resolution in resolutions:
            # Compute reprojection losses (prev frame to cur and next frame to cur)
            reprojection_loss = tuple(
                self._combined_reprojection_loss(outputs["color", frame_id, resolution], target)
                for frame_id in frame_ids
            )
            reprojection_loss = torch.cat(reprojection_loss, 1)

            # If avg_reprojection is disabled and automasking is enabled
            # there will be four "loss  images" stacked in the end and
            # the per-pixel minimum will be selected for optimization.
            # Cases where this is relevant are, for example, image borders,
            # where information is missing, or areas occluded in one of the
            # input images but not all of them.
            # If avg_reprojection is enabled the number of images to select
            # the minimum loss from is reduced by average-combining them.
            if self.avg_reprojection:
                reprojection_loss = reprojection_loss.mean(1, keepdim=True)

            # Pixels that are equal in the (unwarped) source image
            # and target image (e.g. no motion) are not that helpful
            # and can be masked out.
            if self.automasking:
                reprojection_loss = torch.cat(
                    (identity_reprojection_loss, reprojection_loss), 1
                )
                # Select the per-pixel minimum loss from
                # (prev_unwarped, next_unwarped, prev_unwarped, prev_warped).
                # Pixels where the unwarped input images are selected
                # act as gradient black holes, as nothing is backpropagated
                # into the network.
                reprojection_loss, idxs = torch.min(reprojection_loss, dim=1)

            # Segmentation moving mask to mask DC objects
            if outputs_masked is not None:
                moving_mask = outputs_masked['moving_mask']
                reprojection_loss = reprojection_loss * moving_mask

            loss = reprojection_loss.mean()

            if self.disparity_smoothness != 0:
                disp = outputs["disp", resolution]

                ref_color = functional.interpolate(
                    color, disp.shape[2:], mode='bilinear', align_corners=False
                )

                mean_disp = disp.mean((2, 3), True)
                norm_disp = disp / (mean_disp + 1e-7)

                disp_smth_loss = self.smoothness(norm_disp, ref_color)
                disp_smth_loss = self.disparity_smoothness * disp_smth_loss / (2 ** resolution)

                losses[f'disp_smth_loss/{resolution}'] = disp_smth_loss

                loss += disp_smth_loss

            losses[f'loss/{resolution}'] = loss

        losses['loss_depth_reprojection'] = sum(
            losses[f'loss/{resolution}']
            for resolution in resolutions
        ) / len(resolutions)

        return losses

    def compute_losses(self, inputs, outputs, outputs_masked):
        losses = self._reprojection_losses(inputs, outputs, outputs_masked)
        losses['loss_depth'] = losses['loss_depth_reprojection']

        return losses
