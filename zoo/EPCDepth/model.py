import os
import torch
import random
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import progressbar
import numpy as np

from network.rsu_decoder import RSUDecoder
from network.depth_decoder import DepthDecoder
from network.encoder import resnet_encoder
from dataset.kitti_dataset import KittiDataset
import tools


class Model:
    def __init__(self, args):
        self.args = args

        if args.vis:
            return

        self.model = {}
        self.device = torch.device("cpu" if self.args.no_cuda or not torch.cuda.is_available() else "cuda")

        self.model["encoder"] = resnet_encoder(num_layers=self.args.num_layers, num_inputs=1,
                                               pretrained=self.args.pretrained).to(self.device)

        if self.args.use_full_scale:
            self.model["depth_decoder"] = RSUDecoder(num_output_channels=1, use_encoder_disp=True,
                                                     encoder_layer_channels=self.model["encoder"].layer_channels).to(self.device)
        else:
            self.model["depth_decoder"] = DepthDecoder(num_output_channels=1,
                                                       encoder_layer_channels=self.model["encoder"].layer_channels).to(self.device)

        val_dataset = KittiDataset(data_path=self.args.data_path, img_height=self.args.img_height, img_width=self.args.img_width,
                                   train=False, split=self.args.split, test=self.args.val)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers, pin_memory=True)

        if self.args.val:
            return

        train_dataset = KittiDataset(data_path=self.args.data_path, img_height=self.args.img_height, img_width=self.args.img_width,
                                     train=True, split=self.args.split, use_depth_hint=self.args.use_depth_hint)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=True)

        parameters_to_train = list(self.model["encoder"].parameters()) + list(self.model["depth_decoder"].parameters())

        self.optimizer = torch.optim.Adam(parameters_to_train, self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.scheduler_step_size, 0.1)

        self.ssim = tools.SSIM()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.args.logs_dir, current_time)
        self.writer = SummaryWriter(log_dir=log_dir, comment="Record network info")

        self.save_dir = os.path.join(self.args.models_dir, current_time)
        os.makedirs(self.save_dir, exist_ok=True)

    def main(self):

        if self.args.vis:
            self.visualization()
            return

        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            if not self.args.val:
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict((checkpoint["lr_scheduler"]))
            for m_name, _ in self.model.items():
                if m_name in checkpoint:
                    self.model[m_name].load_state_dict(checkpoint[m_name])
                else:
                    print("There is no weight in checkpoint for model {}".format(m_name))

        if self.args.val:
            with torch.no_grad():
                self.validate()
            return

        for epoch in range(self.args.start_epoch, self.args.epochs):

            train_loss = self.train_epoch(epoch)
            self.writer.add_scalar("Train Losses", train_loss, epoch)

            with torch.no_grad():
                val_errors = self.validate()
            self.writer.add_scalar("abs_rel", val_errors[0], epoch)
            self.writer.add_scalar("sq_rel", val_errors[1], epoch)
            self.writer.add_scalar("rmse", val_errors[2], epoch)
            self.writer.add_scalar("rmse_log", val_errors[3], epoch)
            self.writer.add_scalar("a1", val_errors[4], epoch)
            self.writer.add_scalar("a2", val_errors[5], epoch)
            self.writer.add_scalar("a3", val_errors[6], epoch)

            save_filename = os.path.join(self.save_dir, "checkpoint_epoch{}.pth.tar".format(epoch))
            model_state = {
                "epoch": epoch,
                "abs_rel": val_errors[0],
                "sq_rel": val_errors[1],
                "rmse": val_errors[2],
                "rmse_log": val_errors[3],
                "a1": val_errors[4],
                "a2": val_errors[5],
                "a3": val_errors[6],
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
            for m_name, m in self.model.items():
                model_state[m_name] = m.state_dict()
            torch.save(model_state, save_filename)

            self.lr_scheduler.step()
            torch.cuda.empty_cache()

    def validate(self):
        for m in self.model.values():
            m.eval()

        pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                    progressbar.Timer(), ",", progressbar.Variable('abs_rel', width=1), ",", progressbar.Variable('sq_rel', width=1), ",",
                    progressbar.Variable('rmse', width=1, precision=4), ",", progressbar.Variable('rmse_log', width=1), ",",
                    progressbar.Variable('a1', width=1), ",", progressbar.Variable('a2', width=1), ",", progressbar.Variable('a3', width=1)]
        pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.val_loader), prefix="Val:").start()

        all_disps = []
        depth_errors_meter = tools.AverageMeter()
        for batch, data in enumerate(self.val_loader):
            for key, ipt in data.items():
                data[key] = ipt.to(self.device, non_blocking=True)

            ipt = data["curr"]
            if self.args.post_process:
                # Post-processed results require each image to have two forward passes
                ipt = torch.cat((ipt, torch.flip(ipt, [3])), 0)

            pred_disps = self.model["depth_decoder"](self.model["encoder"](ipt))
            if self.args.output_scale != -1:
                pred_disps, _ = tools.disp_to_depth(pred_disps[self.args.output_scale], self.args.min_depth, self.args.max_depth)
                pred_disps = pred_disps.data.cpu()[:, 0].numpy()
            else:
                mean_disps = 0
                for i in range(3):
                    tmp, _ = tools.disp_to_depth(pred_disps[i], self.args.min_depth, self.args.max_depth)
                    tmp = F.interpolate(tmp, [self.args.img_height, self.args.img_width], mode="bilinear", align_corners=False)
                    tmp = tmp.data.cpu()[:, 0].numpy()
                    mean_disps = mean_disps + tmp
                mean_disps = mean_disps / 3
                pred_disps = mean_disps

            if self.args.post_process:
                N = pred_disps.shape[0] // 2
                pred_disps = tools.post_process_disparity(pred_disps[:N], pred_disps[N:, :, ::-1])

            all_disps.append(pred_disps)

            depth_gts = data["depth_gt"].data.cpu().numpy()
            depth_errors = tools.compute_depth_errors(depth_gts, pred_disps, self.args.val_split, False)

            depth_errors_meter.update(depth_errors, data["curr"].size(0))

            pbar.update(batch, abs_rel=depth_errors_meter.avg[0],
                        sq_rel=depth_errors_meter.avg[1],
                        rmse=depth_errors_meter.avg[2],
                        rmse_log=depth_errors_meter.avg[3],
                        a1=depth_errors_meter.avg[4],
                        a2=depth_errors_meter.avg[5],
                        a3=depth_errors_meter.avg[6])

        pbar.finish()

        all_disps = np.concatenate(all_disps)
        if self.args.val:
            np.save(os.path.join(self.args.disps_path, "disparities"), all_disps)

        return depth_errors_meter.avg

    def train_epoch(self, epoch):
        for m in self.model.values():
            m.train()

        pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                    progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                    progressbar.Variable('Loss')]
        pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.train_loader),
                                       prefix="Epoch {}/{}: ".format(epoch, self.args.epochs)).start()

        losses = tools.AverageMeter()

        for batch, data in enumerate(self.train_loader):
            for key, ipt in data.items():
                data[key] = ipt.to(self.device, non_blocking=True)

            if self.args.use_data_graft:
                data = self.data_graft(data)

            loss, predicts = self.train_step(data)
            if self.args.use_spp_distillate:
                spp_loss = self.spp_distillate(data, predicts)
                loss += self.args.spp_loss * spp_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(val=loss.data, n=data["curr"].size(0))
            pbar.update(batch, LR=self.optimizer.state_dict()['param_groups'][0]['lr'],
                        Loss="{losses.val:.3f}|{losses.avg:.3f}".format(losses=losses))

        pbar.finish()

        return losses.avg

    def spp_distillate(self, data, predicts):
        with torch.no_grad():
            disp_best = None
            decoder_disp_best = None
            reproj_loss_min = None
            for scale, disp in enumerate(predicts["disparity"]):
                reproj_loss = self.compute_reprojection_loss(predicts["warp_from_other_side"][scale], data["curr"])
                if scale == 0:
                    disp_best = disp
                    reproj_loss_min = reproj_loss
                elif scale == 5:
                    decoder_disp_best = disp_best.clone()
                    disp_best = disp
                    reproj_loss_min = reproj_loss
                else:
                    disp_best = torch.where(reproj_loss < reproj_loss_min, disp, disp_best)
                    reproj_loss_min, _ = torch.cat([reproj_loss, reproj_loss_min], dim=1).min(dim=1, keepdim=True)

            if decoder_disp_best is not None:
                decoder_disp_best = decoder_disp_best.detach()
                encoder_disp_best = disp_best.detach()
            else:
                decoder_disp_best = disp_best.detach()

        pp_loss = []
        for scale, disp in enumerate(predicts["disparity"]):
            disp_best = decoder_disp_best if scale < 5 else encoder_disp_best
            pp_loss.append(torch.log(torch.abs(disp_best - disp) + 1).mean())
        return torch.stack(pp_loss).mean()

    def train_step(self, data):
        predicts = {}
        features = self.model["encoder"](data["curr_color_aug"])
        predicts["disparity"] = self.model["depth_decoder"](features)
        predicts["depth"] = []
        for i in range(len(predicts["disparity"])):
            predicts["disparity"][i] = F.interpolate(predicts["disparity"][i], [self.args.img_height, self.args.img_width], mode="bilinear",
                                                     align_corners=False)
            _, depth = tools.disp_to_depth(predicts["disparity"][i], self.args.min_depth, self.args.max_depth)
            predicts["depth"].append(depth)

        warp_img = self.get_warp_img(data, predicts)
        predicts.update(warp_img)
        loss = self.compute_loss(data, predicts)
        return loss, predicts

    def data_graft(self, data):
        rand_w = random.randint(0, 4) / 5
        b, c, h, w = data["curr"].shape
        if int(rand_w * h) == 0:
            return data
        l_num = data["side"][data["side"] == 2].shape[0]
        r_num = data["side"][data["side"] == 3].shape[0]
        l_graft_idx = torch.randperm(l_num).to(self.device)
        r_graft_idx = torch.randperm(r_num).to(self.device)
        graft_h = int(rand_w * h)
        flip = random.random()
        for name in data:
            if "curr" in name or "other_side" in name or name == "depth_hint":
                data[name][data["side"] == 2, :, graft_h:] = data[name][data["side"] == 2].clone()[l_graft_idx, :, graft_h:]
                data[name][data["side"] == 3, :, graft_h:] = data[name][data["side"] == 3].clone()[r_graft_idx, :, graft_h:]
                if flip < 0.5:
                    d = data[name].clone()
                    data[name][:, :, :-graft_h] = d[:, :, graft_h:]
                    data[name][:, :, -graft_h:] = d[:, :, :graft_h]
        return data

    def get_warp_img(self, data, predicts):
        warp_img = {}

        K = data["K"]
        T = data["stereo_T"]

        if self.args.use_depth_hint:
            D = data["depth_hint"]
            warp_img["warp_from_hint"] = tools.generate_warp_image(data["other_side"], K, T, D)

        warp_img["warp_from_other_side"] = []
        for D in predicts["depth"]:
            warp_img["warp_from_other_side"].append(tools.generate_warp_image(data["other_side"], K, T, D))

        return warp_img

    def compute_loss(self, data, predicts):
        target = data["curr"]
        losses = []

        proxy_supervised = None
        proxy_supervised_loss = None
        if self.args.use_depth_hint:
            depth_hint_reproj_loss = self.compute_reprojection_loss(predicts["warp_from_hint"], target)
            depth_hint_reproj_loss += 1000 * (data["depth_hint"] <= 0).float()
            if proxy_supervised_loss is None:
                proxy_supervised_loss = depth_hint_reproj_loss
                proxy_supervised = data["depth_hint"]

        for scale in range(len(predicts["disparity"])):
            scale_losses = []

            reprojection_loss = self.compute_reprojection_loss(predicts["warp_from_other_side"][scale], target)
            all_reprojection_loss = reprojection_loss

            if not self.args.disable_automasking:
                identity_reprojection_loss = self.compute_reprojection_loss(data["other_side"], target)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                all_reprojection_loss = torch.cat((all_reprojection_loss, identity_reprojection_loss), dim=1)

            loss1 = self.compute_loss_with_proxy_supervised(all_reprojection_loss, proxy_supervised_loss, proxy_supervised,
                                                            predicts, scale, data)
            scale_losses.append(loss1)

            if self.args.disparity_smoothness != 0:
                mean_disp = predicts["disparity"][scale].mean(2, True).mean(3, True)
                norm_disp = predicts["disparity"][scale] / (mean_disp + 1e-7)
                smooth_loss = self.args.disparity_smoothness * self.compute_smooth_loss(norm_disp, target) / (2 ** scale)
                scale_losses.append(smooth_loss)

            losses.append(torch.sum(torch.stack(scale_losses)))

        return torch.mean(torch.stack(losses))

    def compute_loss_with_proxy_supervised(self, all_reprojection_loss, proxy_supervised_loss, proxy_supervised, predicts, scale, data):
        if proxy_supervised_loss is not None:
            all_reprojection_loss = torch.cat((all_reprojection_loss, proxy_supervised_loss), dim=1)
        idxs = torch.argmin(all_reprojection_loss, dim=1, keepdim=True)
        if self.args.disable_automasking:
            reproj_loss_mask = torch.ones_like(all_reprojection_loss[:, [0]])
            proxy_supervised_mask = (idxs == 1).float()  # will be zero if proxy_supervised_loss is None
        else:
            reproj_loss_mask = (idxs != 1).float()
            proxy_supervised_mask = (idxs == 2).float()

        reproj_loss = all_reprojection_loss[:, [0]] * reproj_loss_mask
        reproj_loss = reproj_loss.sum() / (reproj_loss_mask.sum() + 1e-7)
        reproj_loss_with_proxy_supervised = reproj_loss

        if proxy_supervised_loss is not None:
            proxy_supervised_loss = self.compute_proxy_supervised_loss(predicts["depth"][scale], proxy_supervised, proxy_supervised_mask)
            reproj_loss_with_proxy_supervised += proxy_supervised_loss

        return reproj_loss_with_proxy_supervised

    @staticmethod
    def compute_proxy_supervised_loss(pred, target, loss_mask):
        loss = torch.log(torch.abs(target - pred) + 1)
        loss = loss * loss_mask
        loss = loss.sum() / (loss_mask.sum() + 1e-7)
        return loss

    @staticmethod
    def compute_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        img = F.interpolate(img, disp.shape[2:], mode="bilinear", align_corners=False)

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()

        return smooth_loss

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def visualization(self):
        import matplotlib as mpl
        import matplotlib.cm as cm
        from PIL import Image

        assert self.args.disps_path is not None, "Your disparity save path is None!"

        save_dir = os.path.join(os.path.dirname(self.args.disps_path), "disps_vis")
        os.makedirs(save_dir, exist_ok=True)
        disps = np.load(self.args.disps_path)
        for idx, pred_disp in enumerate(disps):
            vmax = np.percentile(pred_disp, 95)
            normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(pred_disp)[:, :, :3] * 255).astype(np.uint8)
            im = Image.fromarray(colormapped_im)
            im.save(os.path.join(save_dir, "disp{}.png".format(idx)))

        print("Successfully visualize {} disparity maps to {}".format(disps.shape[0], save_dir))
