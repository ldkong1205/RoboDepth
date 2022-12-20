import torch
import torch.nn.functional as functional

from .cityscapes import COLOR_SCHEME_CITYSCAPES
from .plasma import COLOR_SCHEME_PLASMA
from .tango import COLOR_SCHEME_TANGO

def seg_prob_image(probs):
    """Takes a torch tensor of shape (N, C, H, W) containing a map
    of cityscapes class probabilities C as input and generate a
    color image of shape (N, C, H, W) from it.
    """

    # Choose the number of categories according
    # to the dimesion of the input Tensor
    colors = COLOR_SCHEME_CITYSCAPES[:probs.shape[1]]

    # Make the category channel the last dimesion (N, W, H, C),
    # matrix multiply so that the color channel is the last
    # dimesion and restore the shape array to (N, C, H, W).
    image = (probs.transpose(1, -1) @ colors).transpose(-1, 1)

    return image

def domain_prob_image(probs):
    """Takes a torch tensor of shape (N, C, H, W) containing a map
    of domain probabilities C as input and generate a
    color image of shape (N, C, H, W) from it.
    """

    # Choose the number of categories according
    # to the dimesion of the input Tensor
    colors = COLOR_SCHEME_TANGO[:probs.shape[1]]

    # Make the category channel the last dimesion (N, W, H, C),
    # matrix multiply so that the color channel is the last
    # dimesion and restore the shape array to (N, C, H, W).
    image = (probs.transpose(1, -1) @ colors).transpose(-1, 1)

    return image

def seg_idx_image(idxs):
    """Takes a torch tensor of shape (N, H, W) containing a map
    of cityscapes train ids as input and generate a color image
    of shape (N, C, H, W) from it.
    """

    # Take the dimesionality from (N, H, W) to (N, C, H, W)
    # and make the tensor invariant over the C dimension
    idxs = idxs.unsqueeze(1)
    idxs = idxs.expand(-1, 3, -1, -1)

    h, w = idxs.shape[2:]

    # Extend the dimesionality of the color scheme from
    # (IDX, C) to (IDX, C, H, W) and make it invariant over
    # the last two dimensions.
    color = COLOR_SCHEME_CITYSCAPES.unsqueeze(2).unsqueeze(3)
    color = color.expand(-1, -1, h, w)

    image = torch.gather(color, 0, idxs)

    return image

def _depth_to_percentile_normalized_disp(depth):
    """This performs the same steps as normalize_depth_for_display
    from the SfMLearner repository, given the default options.
    This treads every image in the batch separately.
    """

    disp = 1 / (depth + 1e-6)

    disp_sorted, _ = disp.flatten(1).sort(1)
    idx = disp_sorted.shape[1] * 95 // 100
    batch_percentiles = disp_sorted[:,idx].view(-1, 1, 1, 1)

    disp_norm = disp / (batch_percentiles + 1e-6)

    return disp_norm

def depth_norm_image(depth):
    """Takes a torch tensor of shape (N, H, W) containing depth
    as input and outputs normalized depth images colored with the
    matplotlib plasma color scheme.
    """

    # Perform the kind-of-industry-standard
    # normalization for image generation
    disp = _depth_to_percentile_normalized_disp(depth)

    # We generate two indexing maps into the colors
    # tensor and a map to interpolate between these
    # two indexed colors.
    # First scale the disp tensor from [0, 1) to [0, num_colors).
    # Then take the dimesionality from (N, H, W) to (N, C, H, W)
    # and make it invariant over the C dimension
    num_colors = COLOR_SCHEME_PLASMA.shape[0]
    idx = disp * num_colors
    idx = idx.expand(-1, 3, -1, -1)

    h, w = idx.shape[2:]

    # Extend the dimesionality of the color scheme from
    # (IDX, C) to (IDX, C, H, W) and make it invariant over
    # the last two dimensions.
    colors = COLOR_SCHEME_PLASMA.unsqueeze(2).unsqueeze(3)
    colors = colors.expand(-1, -1, h, w)

    # Values in idx are somewhere between two color indices.
    # First generate an image based on the lower indices
    idx_low = idx.floor().long().clamp(0, num_colors - 1)
    img_low = torch.gather(colors, 0, idx_low)

    # Then generate an image based on the upper indices
    idx_high = (idx_low + 1).clamp(0, num_colors - 1)
    img_high = torch.gather(colors, 0, idx_high)

    # Then interpolate between these two
    sel_rel = (idx - idx_low.float()).clamp(0, 1)
    img = img_low + sel_rel * (img_high - img_low)

    return img

def surface_normal_image(surface_normal):
    surface_normal = surface_normal.permute(0, 3, 1, 2)
    surface_normal = functional.pad(surface_normal, (0, 1, 0, 1), 'replicate')
    surface_normal = (surface_normal + 1)  / 2

    return surface_normal
