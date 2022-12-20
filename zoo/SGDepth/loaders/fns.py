import torch

VAL_MIN_DEPTH = 1e-3
VAL_MAX_DEPTH = 80


def _validation_mask_kitti_zhou(depth_gt):
    # Select only points that are not too far or too near to be useful
    dist_mask = (depth_gt > VAL_MIN_DEPTH) & (depth_gt < VAL_MAX_DEPTH)

    # Mask out points that lie outside the
    # area of the image that contains usually
    # useful pixels.
    img_height = dist_mask.shape[1]
    img_width = dist_mask.shape[2]

    crop_top = int(0.40810811 * img_height)
    crop_bot = int(0.99189189 * img_height)
    crop_lft = int(0.03594771 * img_width)
    crop_rth = int(0.96405229 * img_width)

    crop_mask = torch.zeros_like(dist_mask)
    crop_mask[:, crop_top:crop_bot, crop_lft:crop_rth] = True

    # Combine the two masks from above
    # noinspection PyTypeChecker
    mask = dist_mask & crop_mask

    return mask


def _validation_mask_kitti_kitti(depth_gt):
    mask = depth_gt > 0

    return mask


def _validation_mask_cityscapes(depth_gt):

    dist_mask = (depth_gt > VAL_MIN_DEPTH) & (depth_gt < VAL_MAX_DEPTH)
    # Mask out points that lie outside the
    # area of the image that contains usually
    # useful pixels.
    img_height = dist_mask.shape[1]
    img_width = dist_mask.shape[2]

    crop_top = int(0.1 * img_height)
    crop_bot = int(0.7 * img_height)
    crop_lft = int(0.1 * img_width)
    crop_rth = int(0.9 * img_width)

    crop_mask = torch.zeros_like(dist_mask)
    crop_mask[:, crop_top:crop_bot, crop_lft:crop_rth] = True

    # Combine the two masks from above
    # noinspection PyTypeChecker
    mask = dist_mask & crop_mask

    return mask


def _validation_clamp_kitti(depth_pred):
    depth_pred = depth_pred.clamp(VAL_MIN_DEPTH, VAL_MAX_DEPTH)

    return depth_pred


def _validation_clamp_cityscapes(depth_pred):
    depth_pred = depth_pred.clamp(VAL_MIN_DEPTH, VAL_MAX_DEPTH)

    return depth_pred


def get(name):
    return globals()[f'_{name}']
