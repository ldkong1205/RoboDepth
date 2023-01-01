from __future__ import absolute_import, division, print_function

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    def __init__(
        self,
        data_path: str,
        filenames: list,
        height: int = 192,
        width: int = 640,
        frame_idxs: list = [0],
        num_scales: int = 4,
        is_train: bool = False,
        img_ext: str = '.png',
    ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].split()
        path = line[0]

        input_color = self.get_color(path)
        input_color = np.asarray(input_color)
        inputs[("color", 0, -1)] = self.to_tensor(input_color)

        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        return inputs

    def get_color(self, path):
        raise NotImplementedError

