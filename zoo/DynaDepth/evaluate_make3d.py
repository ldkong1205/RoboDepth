from turtle import color
from layers import disp_to_depth
import networks
import cv2
import os 
import pdb
import torch
import scipy.misc
import shutil
import matplotlib as mpl
import matplotlib.cm as cm

from scipy import io
import numpy as np
from options import MonodepthOptions
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed


splits_dir = os.path.join(os.path.dirname(__file__), "splits")
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# Models which were trained were scaled by 5.4 to ease the training
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
TRANS_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

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


def read_make3d():
    """Read Images and Depths from Make3d test dataset

    Returns:
        images: list of centre-cropped images (2x1 ratio)
        depths_gt_cropped: list of cropped depth maps
    """
    main_path = 'data/make3d'
    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()

    depths_gt = []
    images = []
    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = int(1704 / 2)
    depth_new_height = 21
    for filename in test_filenames:
        mat = io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)),verify_compressed_data_integrity=False)
        
        depths_gt.append(mat["Position3DGrid"][:,:,3])
        
        image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
        image = image[ int((2272 - color_new_height)/2):int((2272 + color_new_height)/2),:,:]
        images.append(image[:,:,::-1])
        
    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = map(lambda x: x[int((55 - 21)/2):int((55 + 21)/2),:], depths_gt)

    depths_gt_cropped = list(depths_gt_cropped)

    return images, depths_gt_cropped


def colorize_depth(value):
    # The style used in SharinGAN
    cmapper = cm.get_cmap(cm.get_cmap('plasma'))
    value = cmapper(value, bytes=True) # (nxmx4)
    value = Image.fromarray(value).convert("RGB")
    return value
    

def colorize_disp(value):
    # The style used in monodepth2
    vmax = np.percentile(value, 95)
    normalizer = mpl.colors.Normalize(vmin=value.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(value)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colormapped_im)
    
    
def evaluate(opt):
    """Evaluates a pretrained model on the 134 test images of Make3D
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 70
    
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)
            
    print("=> Loading weights from {}".format(opt.load_weights_folder))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))
    
    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []
    color_raw = []
    
    # Read centre-cropped images and gt_depths from make3d
    images, depths_gt_cropped = read_make3d()

    print("==============================================")
    print("=> Evaluating the 134 test images of Make3D...")
    print("=> Computing predictions with size 512x256")
    
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()
    
    if opt.eval_mono:
        print("   Mono evaluation - using median scaling")
    else:
        print("   Scale-aware evaluation - "
              "disabling median scaling, scaling by {}".format(TRANS_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = TRANS_SCALE_FACTOR
    
    errors = []
    ratios = []
    with torch.no_grad():
        for i in range(len(images)):
            input_color = images[i]
            
            # Process input_color in the same way as in KittiRawDataset
            # (1) resize (2) to_sensor
            # Need to nomarlize because in encoder: x = (input_image - 0.45) / 0.225
            input_color = cv2.resize(input_color/255.0, (512, 256), interpolation=cv2.INTER_NEAREST)
            color_raw.append(input_color)

            
            # input_color: [1, 3, H, W]
            input_color = torch.tensor(input_color, dtype = torch.float).cuda().permute(2,0,1)[None,:,:,:]

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                            
            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps)
    
    if opt.save_make3d:
        out_folder = "make3d_vis/{}".format(opt.load_weights_folder.split('/')[1])
        if os.path.isdir(out_folder): 
            shutil.rmtree(out_folder)
        os.mkdir(out_folder)
            
    for i in range(len(images)):
        depth_gt = depths_gt_cropped[i]
        depth_pred = 1 / pred_disps[i]
        depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        
        mask = np.logical_and(depth_gt > 0, depth_gt < MAX_DEPTH)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        
        if opt.save_make3d:
            out_depth = 1 / pred_disps[i] 
            out_depth *= opt.pred_depth_scale_factor
        
        depth_pred *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(depth_gt) / np.median(depth_pred)
            ratios.append(ratio)
            depth_pred *= ratio
            if opt.save_make3d:
                out_depth *= ratio
        
        depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH
        
        if opt.save_make3d:
            out_disp = 1 / out_depth 
            out_depth /= MAX_DEPTH
            t0 = (color_raw[i] * 255).astype(np.uint8)
            t1 = np.array(colorize_disp(out_disp))
            t2 = np.array(colorize_depth(out_depth))
            combine = np.concatenate([t0, t1, t2], axis=0)  # [256x2, 512, 3]
            Image.fromarray(combine).save("{}/combine_{:06d}.png".format(out_folder, i))
            
        
        errors.append(compute_errors(depth_gt, depth_pred))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
