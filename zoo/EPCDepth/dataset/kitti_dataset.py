import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import skimage.transform
from torchvision import transforms
from torch.utils.data import Dataset
from read_depth import generate_depth_map


class RandomFlipImage(object):
    def __init__(self, is_flip):
        self.is_flip = is_flip

    def __call__(self, x):
        if not self.is_flip:
            return x

        out = x.transpose(Image.FLIP_LEFT_RIGHT)
        return out


class RandomColorAugImage(object):
    def __init__(self, is_color_aug, color_aug):
        self.color_aug = color_aug
        self.is_color_aug = is_color_aug

    def __call__(self, x):
        if not self.is_color_aug:
            return x

        out = self.color_aug(x)
        return out


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def opencv_loader(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class KittiDataset(Dataset):
    def __init__(self, data_path, img_height, img_width, train, split="eigen_full", depth_hint_path=None, use_depth_hint=False, test=False):
        super(KittiDataset, self).__init__()
        self.data_path = data_path
        self.train = train
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.depth_gt_size = (375, 1242)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.use_depth_hint = use_depth_hint
        if self.use_depth_hint:
            if depth_hint_path is None:
                depth_hint_path = os.path.join(self.data_path, "depth_hints")
            self.depth_hint_path = depth_hint_path

        K = np.array([[0.58, 0, 0.5, 0],
                      [0, 1.92, 0.5, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        K[0, :] *= self.img_width
        K[1, :] *= self.img_height
        self.K = K

        filenames_file = os.path.join(os.path.dirname(__file__), "filenames", split)
        if self.train:
            filenames_file = os.path.join(filenames_file, "train_files.txt")
        elif not test:
            filenames_file = os.path.join(filenames_file, "val_files.txt")
        else:
            filenames_file = os.path.join(filenames_file, "test_files.txt")

        self.filenames = pd.read_table(filenames_file, names=["folder", "frame_idx", "side"], sep=' ')

    def get_img(self, folder, frame_idx, side, loader=pil_loader):
        img_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), "{:010d}{}".format(frame_idx, ".jpg"))
        img = loader(img_path)
        return img

    def get_depth_hint(self, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(self.depth_hint_path, folder, "image_0{}".format(self.side_map[side]),
                                       "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path)[0]  # (h, w)
        if is_flip:
            depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  # (1, h, w)

        return depth_hint

    def get_depth_gt(self, folder, frame_index, side, is_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        assert os.path.isfile(velo_filename), "Frame {} in {} don't have ground truth".format(frame_index, folder)

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side], True)

        depth_gt = skimage.transform.resize(
            depth_gt, self.depth_gt_size, order=0, preserve_range=True, mode='constant')

        if is_flip:
            depth_gt = np.fliplr(depth_gt)
        # depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt.astype(np.float32))  # (h, w)

        return depth_gt

    def transform(self, x, is_flip, is_color_aug, color_aug):
        # will be [0,1]
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width), interpolation=Image.ANTIALIAS),
            RandomFlipImage(is_flip=is_flip),
            RandomColorAugImage(is_color_aug=is_color_aug, color_aug=color_aug),
            transforms.ToTensor(),
        ])
        return transform(x)

    def __getitem__(self, index):
        """
        :param index:
        :return: this func need return img shape (b, c, h, w) or (c, h, w)
        """
        is_flip = random.random() > 0.5 and self.train
        is_color_aug = random.random() > 0.5 and self.train
        color_aug = transforms.ColorJitter.get_params(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))

        data = {}

        folder, frame_idx, side = self.filenames.loc[index]
        data["curr"] = self.transform(self.get_img(folder, frame_idx, side), is_flip, False, color_aug)
        data["id"] = index
        if self.train:
            data["side"] = self.side_map[side]
            data["curr_color_aug"] = self.transform(self.get_img(folder, frame_idx, side), is_flip, is_color_aug, color_aug)

            other_side = {"r": "l", "l": "r"}[side]
            data["other_side"] = self.transform(self.get_img(folder, frame_idx, other_side), is_flip, False, color_aug)
            data["other_side_color_aug"] = self.transform(self.get_img(folder, frame_idx, other_side), is_flip, is_color_aug, color_aug)

            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if is_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            data["stereo_T"] = torch.from_numpy(stereo_T)

            if self.use_depth_hint:
                data["depth_hint"] = self.get_depth_hint(folder, frame_idx, side, is_flip)

            data["K"] = torch.from_numpy(self.K)

        else:
            data["depth_gt"] = self.get_depth_gt(folder, frame_idx, side, is_flip)

        return data

    def __len__(self):
        return self.filenames.shape[0]

    def getfilename(self):
        return self.filenames
