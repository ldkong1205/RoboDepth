'''
Seokju Lee

'''

import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import cv2

from matplotlib import pyplot as plt
import pdb

def crawl_folders(folders_list):
        imgs = []
        depth = []
        segs = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            imgs.extend(current_imgs)
            for img in current_imgs:
                # Fetch depth file
                dd = img.dirname()/(img.name[:-4] + '.npy')
                assert(dd.isfile()), "depth file {} not found".format(str(dd))
                depth.append(dd)
                # Fetch segmentation file
                ss = folder.dirname().parent/'segmentation'/folder.basename()/(img.name[:-4] + '.npy')
                assert(ss.isfile()), "segmentation file {} not found".format(str(ss))
                segs.append(ss)
            
        return imgs, depth, segs


def load_as_float(path):
    return imread(path).astype(np.float32)


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/'image'/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth, self.segs = crawl_folders(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img = load_as_float(self.imgs[index])   # H x W x 3
        depth = np.load(self.depth[index]).astype(np.float32)   # H x W
        seg = torch.from_numpy(np.load(self.segs[index]).astype(np.float32))    # N x H X W

        # # Re-ordering segmentation by each mask size
        # seg_sort = torch.cat([torch.zeros(1).long(), seg.sum(dim=(1,2)).argsort(descending=True)[:-1]], dim=0)
        # seg = seg[seg_sort]

        # Sum segmentation for every mask
        seg = seg.sum(dim=0, keepdim=False).clamp(min=0.0, max=1.0)     # H x W

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth, seg

    def __len__(self):
        return len(self.imgs)
