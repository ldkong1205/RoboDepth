from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
from cv2 import imwrite
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_folder',type = str,
                        help='the folder name of model')
    parser.add_argument('--model_name',type = str)
    '''parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                                "mono+stereo_no_pt_640x192",
                                "mono_1024x320",
                                "stereo_1024x320",
                                "mono+stereo_1024x320"])'''
    parser.add_argument('--ext', type=str,
                            help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                            help='if set, disables CUDA',
                            action='store_true')
    return parser.parse_args()

def test_simple(args):
    """Function to predict for a single image or folder of images
        """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
            #device = torch.device("cuda")
            device = "cuda"
    else:
            device = "cpu"

    model_path = os.path.join(args.model_folder, args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    para_sum_encoder = sum(p.numel() for p in encoder.parameters())
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())
    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum = para_sum_decoder + para_sum_encoder
    print("encoder has {} parameters".format(para_sum_encoder))
    print("depth_decoder has {} parameters".format(para_sum_decoder))
    print("encoder and depth_ decoder have  total {} parameters".format(para_sum))
    
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format('png')))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            image_name =  args.model_folder[39:-8] + args.model_name[7:] +  args.image_path[7:-4] 
            rgb = transforms.ToTensor()(input_image)
            
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_rgb = input_image
            input_r = pil.fromarray(np.uint8(input_rgb))
            
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            
            rgb1 = rgb.permute(1,2,0).detach().cpu().numpy() * 255
            
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            #disp_resized = disp
            # just like Featdepth
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, depth_resized = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            print(disp_resized_np.shape)
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join('results',"{}_disp.jpeg".format(image_name))
            
            # concatenate both vertically
            #image = np.concatenate([rgb1, im], 0)
            # save a grey scale map for point cloud viz
            #depth_resized = depth_resized.squeeze().cpu().numpy()
            scaled_disp = (50 / scaled_disp).squeeze().cpu().numpy()
            #scaled_disp = scaled_disp.squeeze().cpu().numpy()
            im_grey = pil.fromarray(np.uint8((scaled_disp * 255)),'L')
            name_grey_depth = os.path.join('results',"{}_grey_disp.png".format(image_name))
            name_corped_rgb = os.path.join('results',"rgb.png")
            im_grey.save(name_grey_depth) 
            input_r.save(name_corped_rgb)
            #just save a single depth
            im.save(name_dest_im)

            #save a concatenated iamge for depth and rgb
            #imwrite(name_dest_im,image[:,:,::-1])
            
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
