from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_zhou_validation(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_zhou_val_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_zhou'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='zhou_split',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) validation set for depth validation",
          flush=True)

    return loader


def kitti_zhou_test(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth evaluation from the kitti test set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_zhou_test_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_zhou'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='zhou_split',
        trainvaltest_split='test',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (zhou_split) test set for depth evaluation", flush=True)

    return loader


def kitti_kitti_validation(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_kitti_val_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_kitti'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='kitti_split',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (kitti_split) validation set for depth validation",
          flush=True)

    return loader


def kitti_2015_train(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth evaluation from the kitti_2015 training set (but for evaluation).
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_2015_train_depth'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_kitti'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti_2015',
        trainvaltest_split='train',
        video_mode='mono',
        stereo_mode='mono',
        keys_to_load=('color', 'depth'),
        data_transforms=transforms,
        video_frames=(0, ),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti_2015 test set for depth evaluation", flush=True)

    return loader


