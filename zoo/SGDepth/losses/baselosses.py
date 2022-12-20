"""Collection of losses with corresponding functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, disp, img):
        """ adapted from https://github.com/nianticlabs/monodepth2
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """adapted from https://github.com/nianticlabs/monodepth2
    Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, bg_replace=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.bg_replace = bg_replace  # background replacement index
        assert isinstance(ignore_index, int), "Ignore_index has to be of type int"
        self.loss = torch.nn.NLLLoss(weight, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        if self.bg_replace is None:
            if self.ignore_index is not None:
                targets[targets == 255] = self.ignore_index
        else:
            assert isinstance(self.bg_replace, int), "bg_pelace has to be of type int"
            targets[targets == 255] = self.bg_replace

        return self.loss(nn.functional.log_softmax(outputs, dim=1), targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, ignore_background=False, train_id_0=0, reduction='mean',
                 device=None, norm=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight has to be None or type torch.tensor"
        assert isinstance(ignore_index, int), "ignore_index has to be of type int"
        assert isinstance(ignore_background, bool), "ignore_background has to be of type bool"
        # With train_id_0 != 0, a slice of outputs can be passed to this loss even if train_ids > #classes. train_id_0
        # will make sure that train_ids < #classes is valid (iff class train_ids are consecutive!).
        assert isinstance(train_id_0, int), "train_id_0 has to be of type int"
        assert reduction in ('mean', 'sum', 'none'), "reduction only supports 'mean' (default), 'sum' and 'none'"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background
        self.train_id_0 = train_id_0
        self.reduction = reduction
        self.device = device
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_index, reduction='none')
        self.norm = norm

    def forward(self, outputs, targets):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1],\
            "either provide weights for all classes or none"

        # Cast class trainIDs of targets into [0, #classes -1] by subtracting smallest trainID, background is taken
        # care of afterwards.
        targets = torch.add(targets, -self.train_id_0)
        bg = 255 - self.train_id_0

        # Background treatment
        if self.ignore_background:
            targets[targets == bg] = self.ignore_index
        else:
            targets[targets == bg] = outputs.shape[1] - 1

        # Calculate unreduced loss
        loss = self.loss(outputs, targets)

        # Apply reduction manually since NLLLoss shows non-deterministic behaviour on GPU
        if self.reduction == 'mean':
            denom = 0
            if self.weight is not None:
                for i in range(outputs.shape[1]):
                    denom += torch.sum((targets == i).int()) * self.weight[i]
            else:
                if self.ignore_background:
                    denom = torch.numel(targets) - int(torch.sum((targets == self.ignore_index).int()))
                else:
                    denom = torch.numel(targets)
            # To avoid an exploding loss when all targets are background
            if denom == 0:
                denom = 1e-15

            if self.norm == 'BWH':
                denom = torch.numel(targets)

            return torch.sum(loss) / denom
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, ignore_background=False, train_id_0=0, reduction='mean',
                 device=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight has to be None or type torch.tensor"
        assert isinstance(ignore_index, int), "ignore_index has to be of type int"
        assert isinstance(ignore_background, bool), "ignore_background has to be of type bool"
        # With train_id_0 != 0, a slice of outputs can be passed to this loss even if train_ids > #classes. train_id_0
        # will make sure that train_ids < #classes is valid (iff class train_ids are consecutive!).
        assert isinstance(train_id_0, int), "train_id_0 has to be of type int"
        assert reduction in ('mean', 'sum', 'none'), "reduction only supports 'mean' (default), 'sum' and 'none'"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background
        self.train_id_0 = train_id_0
        self.reduction = reduction
        self.device = device
        self.loss = torch.nn.BCEWithLogitsLoss(weight, reduction='none')

    def forward(self, outputs, targets):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"

        targets = torch.add(targets, -self.train_id_0)

        denom = torch.numel(targets)
        if denom == 0:
            denom = 1e-15

        mask_pos = torch.zeros(outputs.shape)
        for i in range(outputs.shape[1]):
            mask_pos_ = mask_pos[:, i, ...]
            mask_pos_[targets == i] = 1
        mask_pos = mask_pos.to(self.device)

        loss = self.loss(outputs, mask_pos)
        return torch.sum(loss) / denom


class FocalLoss(nn.Module):
    def __init__(self, ignore_index=-100, ignore_background=False, train_id_0=0, reduction='mean', focus=0, device=None):
        super().__init__()

        assert isinstance(ignore_index, int), "ignore_index has to be of type int"
        assert isinstance(ignore_background, bool), "ignore_background has to be of type bool"
        # With train_id_0 != 0, a slice of outputs can be passed to this loss even if train_ids > #classes. train_id_0
        # will make sure that train_ids < #classes is valid (iff class train_ids are consecutive!).
        assert isinstance(train_id_0, int), "train_id_0 has to be of type int"
        assert reduction in ('mean', 'sum', 'none'), "reduction only supports 'mean' (default), 'sum' and 'none'"
        assert isinstance(focus, int), "focus has to be of type int"
        assert focus >= 0, "focus has to be >= 0"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.ignore_index = ignore_index
        self.ignore_background = ignore_background
        self.train_id_0 = train_id_0
        self.reduction = reduction
        self.focus = focus
        self.device = device
        self.loss = nn.NLLLoss(weight=None, ignore_index=ignore_index, reduction='none')

    def forward(self, outputs, targets):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"

        # Cast class trainIDs of targets into [0, #classes -1] by subtracting smallest trainID, background is taken
        # care of afterwards.
        targets = torch.add(targets, -self.train_id_0)
        bg = 255 - self.train_id_0

        # Background treatment
        if self.ignore_background:
            targets[targets == bg] = self.ignore_index
        else:
            targets[targets == bg] = outputs.shape[1] - 1

        scale_down = torch.pow(1 - F.softmax(outputs, dim=1), self.focus)

        mask = torch.zeros(outputs.shape).to(self.device)

        for i in range(outputs.shape[1]):
            mask_t = mask[:, i, ...]
            mask_t[targets == i] = 1

        scale_down = torch.sum(mask * scale_down, dim=1)

        # Calculate unreduced loss
        loss = scale_down * self.loss(F.log_softmax(outputs, dim=1), targets)

        # Apply reduction manually since NLLLoss shows non-deterministic behaviour on GPU
        if self.reduction == 'mean':
            return torch.sum(loss) / torch.sum(scale_down)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class KnowledgeDistillationCELoss(nn.Module):
    def __init__(self, weight=None, temp=1, device=None, norm=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight may only be None or type torch.tensor"
        assert isinstance(temp, int), "temp has to be of type int, default is 2"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.temp = temp
        self.device = device
        self.norm = norm

    def forward(self, outputs, targets, targets_new=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1],\
            "either provide weights for all classes or none"

        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...
        denom_corr = 0
        if targets_new is not None:
            mask = torch.ones(targets_new.shape).to(self.device)
            mask[targets_new != 255] = 0
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = - targets * outputs

        # Apply mean reduction, weights make only sense if there is more than one class!
        if self.weight is not None:
            loss = torch.sum(torch.sum(loss, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            denom = torch.sum(torch.sum(targets, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            loss = loss / denom
        else:
            if self.norm == 'BWH':
                denom_corr = 0
            loss = torch.sum(loss) / (torch.numel(loss[:, 0, ...]) - denom_corr)

        return self.temp**2 * loss  # Gradients are scaled down by 1 / T if not corrected

class KnowledgeDistillationCELossWithGradientScaling(nn.Module):
    def __init__(self, temp=1, device=None, gs=None, norm=False):
        """Initialises the loss

                :param temp: temperature of the knowledge distillation loss, reduces to CE-loss for t = 1
                :param device: torch device used during training
                :param gs: defines the strength of the scaling
                :param norm: defines how the loss is normalized

        """

        super().__init__()
        assert isinstance(temp, int), "temp has to be of type int, default is 1"
        assert isinstance(device, torch.device), "device has to be of type torch.device"
        assert gs > 0, "gs has to be > 0"
        assert isinstance(norm, bool), "norm has to be of type bool"

        self.temp = temp
        self.device = device
        self.gs = gs
        self.norm = norm

    def forward(self, outputs, targets, targets_new=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"

        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...

        # here the weights are calculated as described in the paper, just remove the weights from the calculation as
        # in KnowledgeDistillationCELoss
        denom_corr = 0
        ln2 = torch.log(torch.tensor([2.0]).to(self.device))  # basis change
        entropy = -torch.sum(targets * torch.log(targets), dim=1, keepdim=True) / ln2
        weights = entropy * self.gs + 1

        # calculate the mask from the new targets, so that only the regions without labels are considered
        if targets_new is not None:
            mask = torch.ones(targets_new.shape).to(self.device)
            mask[targets_new != 255] = 0
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            weights = mask * weights
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = - weights * torch.sum(targets * outputs, dim=1, keepdim=True)

        # Apply mean reduction
        if self.norm:
            denom = torch.sum(weights)
        else:
            denom = torch.numel(loss[:, 0, ...]) - denom_corr
        loss = torch.sum(loss) / denom

        return self.temp**2 * loss  # Gradients are scaled down by 1 / T if not corrected

class KnowledgeDistillationCELossUmbertoOldWithBackground(nn.Module):
    def __init__(self, weight=None, temp=1, device=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight may only be None or type torch.tensor"
        assert isinstance(temp, int), "temp has to be of type int, default is 2"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.temp = temp
        self.device = device

    def forward(self, outputs, targets, targets_new=None, nco=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1],\
            "either provide weights for all classes or none"
        assert isinstance(nco, int) and nco > 0, "nco (num classed old) has to be type int and > 0"

        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...
        denom_corr = 0
        if targets_new is not None:
            mask = torch.ones(targets_new.shape).to(self.device)
            mask[((targets_new != 255).int() + (targets_new >= nco).int()) == 2] = 0
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = - targets * outputs

        # Apply mean reduction, weights make only sense if there is more than one class!
        if self.weight is not None:
            loss = torch.sum(torch.sum(loss, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            denom = torch.sum(torch.sum(targets, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            loss = loss / denom
        else:
            loss = torch.sum(loss) / (torch.numel(loss[:, 0, ...]) - denom_corr)

        return self.temp * loss  # Gradients are scaled down by 1 / T if not corrected

class KnowledgeDistillationCELossUmbertoOld(nn.Module):
    def __init__(self, weight=None, temp=1, device=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight may only be None or type torch.tensor"
        assert isinstance(temp, int), "temp has to be of type int, default is 2"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.temp = temp
        self.device = device

    def forward(self, outputs, targets, targets_new=None, nco=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1],\
            "either provide weights for all classes or none"
        assert isinstance(nco, int) and nco > 0, "nco (num classed old) has to be type int and > 0"

        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...
        denom_corr = 0
        if targets_new is not None:
            mask = torch.zeros(targets_new.shape).to(self.device)
            mask[targets_new < nco] = 1
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = - targets * outputs

        # Apply mean reduction, weights make only sense if there is more than one class!
        if self.weight is not None:
            loss = torch.sum(torch.sum(loss, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            denom = torch.sum(torch.sum(targets, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            loss = loss / denom
        else:
            loss = torch.sum(loss) / (torch.numel(loss[:, 0, ...]) - denom_corr)

        return self.temp * loss  # Gradients are scaled down by 1 / T if not corrected

class KnowledgeDistillationCELossUmbertoNew(nn.Module):
    def __init__(self, weight=None, temp=1, device=None):
        super().__init__()

        assert weight is None or torch.is_tensor(weight), "weight may only be None or type torch.tensor"
        assert isinstance(temp, int), "temp has to be of type int, default is 2"
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.weight = weight
        self.temp = temp
        self.device = device

    def forward(self, outputs, targets, targets_new=None, nco=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"
        assert self.weight is None or self.weight.shape[0] == outputs.shape[1],\
            "either provide weights for all classes or none"
        assert isinstance(nco, int) and nco > 0, "nco (num classed old) has to be type int and > 0"

        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...
        denom_corr = 0
        if targets_new is not None:
            mask = torch.zeros(targets_new.shape).to(self.device)
            mask[((targets_new != 255).int() + (targets_new >= nco).int()) == 2] = 1
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = - targets * outputs

        # Apply mean reduction, weights make only sense if there is more than one class!
        if self.weight is not None:
            loss = torch.sum(torch.sum(loss, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            denom = torch.sum(torch.sum(targets, dim=(0,) + tuple(range(2, loss.dim()))) * self.weight)
            loss = loss / denom
        else:
            loss = torch.sum(loss) / (torch.numel(loss[:, 0, ...]) - denom_corr)

        return self.temp * loss  # Gradients are scaled down by 1 / T if not corrected


class BackgroundLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.device = device
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"

        # Create uniformly distributed targets for background area
        targets = targets.reshape(tuple([targets.shape[0], 1, *targets.shape[1:]]))
        mask = torch.zeros(targets.shape, dtype=torch.float).to(self.device)
        mask[targets == 255] = 1
        denom = torch.sum(mask, dtype=torch.int).float()
        mask = mask.expand_as(outputs)
        targets = torch.ones(outputs.shape, dtype=torch.float).to(self.device) / outputs.shape[1]

        # Calculate unreduced loss
        loss = mask * self.loss(outputs, targets)

        # Apply mean reduction, weights make only sense if there is more than one class!
        return torch.sum(loss) / denom


class BackgroundCELoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        assert isinstance(device, torch.device), "device has to be of type torch.device"

        self.device = device

    def forward(self, outputs, targets):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape[0] == targets.shape[0], "'minibatch' of outputs and targets has to agree"
        assert outputs.shape[2:] == targets.shape[1:], "'d1, d2, ..., dk' of outputs and targets have to agree"

        # Create uniformly distributed targets for background area
        targets = targets.reshape(tuple([targets.shape[0], 1, *targets.shape[1:]]))
        mask = torch.zeros(targets.shape, dtype=torch.float).to(self.device)
        mask[targets == 255] = 1
        denom = torch.sum(mask, dtype=torch.int).float()
        mask = mask / outputs.shape[1]
        mask = mask.expand_as(outputs)

        # Calculate unreduced loss
        loss = - mask * outputs

        # Apply mean reduction, weights make only sense if there is more than one class!
        return torch.sum(loss) / denom


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu == self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min] < self.thresh else sorteds[self.n_min]
            labels[picks > thresh] = self.ignore_lb
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss
