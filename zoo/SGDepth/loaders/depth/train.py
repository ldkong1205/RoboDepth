from torch.utils.data import DataLoader, ConcatDataset

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_zhou_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_zhou_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'mono',
        'split': 'zhou_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) train split for depth training", flush=True)

    return loader


def kitti_kitti_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_kitti_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'mono',
        'split': 'kitti_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color',),
                'keys_to_video': ('color',)}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (kitti_split) train set for depth training", flush=True)

    return loader


def kitti_odom09_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_odom09_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'stereo',
        'split': 'odom09_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (odom09_split) train split for depth training", flush=True)

    return loader


def kitti_benchmark_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads image sequences for depth training from the
    kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color', 'depth', 'camera_intrinsics', 'K')
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0, fraction=0.5),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_benchmark_train_depth'),
        tf.AddKeyValue('purposes', ('depth', 'domain')),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'video',
        'stereo_mode': 'stereo',
        'split': 'benchmark_split',
        'video_frames': (0, -1, 1),
        'disable_const_items': False
    }

    cfg_left = {'keys_to_load': ('color', ),
                'keys_to_video': ('color', )}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (benchmark_split) train split for depth training",
          flush=True)

    return loader
