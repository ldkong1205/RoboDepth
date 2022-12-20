'''
seokju Lee

'''

import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
import datetime
from tqdm import tqdm
import os,sys

from matplotlib import pyplot as plt
import pdb

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img, img

@torch.no_grad()
def main():
    args = parser.parse_args()

    print("=> Tested at {}".format(datetime.datetime.now().strftime("%m-%d-%H:%M")))
    
    print('=> Load dispnet model from {}'.format(args.pretrained_dispnet))

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    import models

    disp_net = models.DispResNet().to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    with open(args.dataset_list, 'r') as f:
        test_files = list(f.read().splitlines())
    print('=> {} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    for j in tqdm(range(len(test_files))):
        tgt_img, ori_img = load_tensor_image(dataset_dir + test_files[j], args)
        pred_disp = disp_net(tgt_img).cpu().numpy()[0,0]
        # pdb.set_trace()
        '''
            fig = plt.figure(9, figsize=(8, 10))
            fig.add_subplot(2,1,1)
            plt.imshow(ori_img.transpose(1,2,0)/255, vmin=0, vmax=1), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.add_subplot(2,1,2)
            plt.imshow(pred_disp), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar()
            fig.tight_layout(), plt.ion(), plt.show()
        '''

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1/pred_disp
    
    np.save(output_dir/'predictions.npy', predictions)


if __name__ == '__main__':
    main()