
import os
from .utils import readlines, sec_to_hm_str


def get_dataset(cfg, training=True):
    dataset_name = cfg['name']
    if dataset_name == 'kitti':
        from .kitti_dataset import KITTIRAWDataset as dataset
    elif dataset_name == 'kitti_odom':
        from .kitti_dataset import KITTIOdomDataset as dataset
    elif dataset_name == 'cityscape':
        from .cityscape_dataset import CityscapeDataset as dataset
    elif dataset_name == 'folder':
        from .folder_dataset import FolderDataset as dataset
    elif dataset_name == 'eth3d':
        from .eth3d_dataset import FolderDataset as dataset
    elif dataset_name == 'euroc':
        from .euroc_dataset import FolderDataset as dataset

    fpath = os.path.join(os.path.dirname(__file__), "splits", cfg.split, "{}_files.txt")
    filenames = readlines(fpath.format("train")) if training else readlines(fpath.format('val'))
    img_ext = '.png' if cfg.png == True else '.jpg'

    dataset = dataset(cfg.in_path,
                      filenames,
                      cfg.height,
                      cfg.width,
                      cfg.frame_ids if training else [0],
                      is_train=training,
                      img_ext=img_ext,
                      gt_depth_path=cfg.gt_depth_path)
    return dataset