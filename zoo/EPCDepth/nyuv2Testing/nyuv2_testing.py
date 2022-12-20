import torch
from torch.utils.data import DataLoader
import numpy as np
import progressbar
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
import os
import argparse

from nyuv2Testing.nyu_dataset import NYUTestDataset
from network.rsu_decoder import RSUDecoder
from network.encoder import resnet_encoder
import tools

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="path to the data", default="./NYU-Depth-V2")
parser.add_argument("--resume", type=str, help="path of models to resume", metavar="PATH")
parser.add_argument("--save_dir", type=str, help="path to save results", metavar="PATH", default="./NYU-Depth-V2")
parser.add_argument("--img_height", type=int, help="input image height", default=288)
parser.add_argument("--img_width", type=int, help="input image width", default=384)
parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
parser.add_argument("--max_depth", type=float, help="maximum depth", default=10.0)
parser.add_argument("--num_layers", type=int, help="maximum depth", default=50)
parser.add_argument("--post_process", action="store_true")

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    encoder = resnet_encoder(num_layers=args.num_layers, num_inputs=1, pretrained=False).to(device)
    depth_decoder = RSUDecoder(num_output_channels=1, use_encoder_disp=True, encoder_layer_channels=encoder.layer_channels).to(device)

    val_dataset = NYUTestDataset(data_path=args.data_path, height=args.img_height, width=args.img_width)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    checkpoint = torch.load(args.resume, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    depth_decoder.load_state_dict(checkpoint["depth_decoder"])

    encoder.eval()
    depth_decoder.eval()

    pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.Variable('abs_rel', width=1), ",", progressbar.Variable('sq_rel', width=1), ",",
                progressbar.Variable('rmse', width=1, precision=4), ",", progressbar.Variable('rmse_log', width=1), ",",
                progressbar.Variable('a1', width=1), ",", progressbar.Variable('a2', width=1), ",", progressbar.Variable('a3', width=1)]
    pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(val_loader), prefix="Val:").start()

    depth_errors_meter = tools.AverageMeter()
    for batch, data in enumerate(val_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device, non_blocking=True)

        ipt = data[0]
        if args.post_process:
            # Post-processed results require each image to have two forward passes
            ipt = torch.cat((ipt, torch.flip(ipt, [3])), 0)

        pred_disps = depth_decoder(encoder(ipt))
        pred_disps, _ = tools.disp_to_depth(pred_disps[0], args.min_depth, args.max_depth)
        pred_disps = pred_disps.data.cpu()[:, 0].numpy()  # (b, h, w)

        if args.post_process:
            N = pred_disps.shape[0] // 2
            pred_disps = tools.post_process_disparity(pred_disps[:N], pred_disps[N:, :, ::-1])

        pred_disp = pred_disps[0]
        vmax = np.percentile(pred_disp, 95)
        normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(pred_disp)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        os.makedirs(args.save_dir, exist_ok=True)
        im.save(os.path.join(args.save_dir, "EPCDepth", "disp{}.png".format(batch)))

        pred_disp = 1 / data[1][0, 0].data.cpu().numpy()
        vmax = np.percentile(pred_disp, 95)
        normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(pred_disp)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        im.save(os.path.join(args.save_dir, "ground-truth", "gt{}.png".format(batch)))

        gt_depth = data[1][0, 0].data.cpu().numpy()  # (h2, w2)
        pred_depth = 1 / pred_disps[0]  # (h, w)
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        depth_errors = tools.compute_multi_errors(gt_depth, pred_depth)

        depth_errors_meter.update(depth_errors, 1)

        pbar.update(batch, abs_rel=depth_errors_meter.avg[0],
                    sq_rel=depth_errors_meter.avg[1],
                    rmse=depth_errors_meter.avg[2],
                    rmse_log=depth_errors_meter.avg[3],
                    a1=depth_errors_meter.avg[4],
                    a2=depth_errors_meter.avg[5],
                    a3=depth_errors_meter.avg[6])

    pbar.finish()
