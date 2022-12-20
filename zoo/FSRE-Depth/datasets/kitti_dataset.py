import random
from copy import deepcopy

import PIL.Image as pil
import skimage.transform
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from datasets.kitti_utils import *
from utils.seg_utils import labels


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode == 'P':
        return Image.open(path)
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class KittiDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 height,
                 width,
                 frame_idxs,
                 filenames,
                 data_path,
                 num_scales,
                 is_train,
                 img_ext='.png',
                 ):
        super(KittiDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.num_scales = num_scales

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.resize_img = {}
        self.resize_seg = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize_img[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.ANTIALIAS)

            self.resize_seg[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.BILINEAR)
        if is_train:
            self.load_depth = False

        else:
            self.load_depth = self.check_depth()

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        # self.class_dict = self.get_classes(self.filenames)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose seg_networks receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_img[i](inputs[(n, im, -1)])

                del inputs[(n, im, -1)]

            if "seg" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_seg[i](inputs[(n, im, -1)])

                del inputs[(n, im, -1)]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "seg" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.tensor(np.array(f)).float().unsqueeze(0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) in [3, 4, 2]:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        if side is None:
            if do_color_aug:
                color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)
            self.preprocess(inputs, color_aug)
            return inputs

        inputs[("seg", 0, -1)] = self.get_seg_map(folder, frame_index, side, do_flip)

        K = deepcopy(self.K)
        if do_flip:
            K[0, 2] = 1 - K[0, 2]

        K[0, :3] *= self.width
        K[1, :3] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        im_path = self.get_image_path(folder, frame_index, side)

        color = self.loader(im_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        if side is not None:
            image_path = os.path.join(
                self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        else:
            image_path = os.path.join(
                self.data_path, folder, "{}".format(f_str))
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_seg_map(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        path = path.replace('Kitti', 'Kitti/segmentation')
        path = path.replace('/data', '')

        seg = self.loader(path, mode='P')
        seg_copy = np.array(seg.copy())

        for k in np.unique(seg):
            seg_copy[seg_copy == k] = labels[k].trainId
        seg = Image.fromarray(seg_copy, mode='P')

        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg
