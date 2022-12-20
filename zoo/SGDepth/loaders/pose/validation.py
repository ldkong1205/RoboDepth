from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_odom09_validation(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_odom09_val_pose'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='odom09_split',
        trainvaltest_split='test',
        video_mode='video',
        stereo_mode='mono',
        keys_to_load=('color', 'poses'),
        keys_to_video=('color', ),
        data_transforms=transforms,
        video_frames=(0, -1, 1),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (odom09 split) validation set for pose validation",
          flush=True)

    return loader


def kitti_odom10_validation(img_height, img_width, batch_size, num_workers):
    """A loader that loads images and depth ground truth for
    depth validation from the kitti validation set.
    """

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize(
            (img_height, img_width),
            image_types=('color', )
        ),
        tf.CreateColoraug(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_odom10_val_pose'),
        tf.AddKeyValue('purposes', ('depth', )),
    ]

    dataset = StandardDataset(
        dataset='kitti',
        split='odom10_split',
        trainvaltest_split='test',
        video_mode='video',
        stereo_mode='mono',
        keys_to_load=('color', 'poses'),
        keys_to_video=('color', ),
        data_transforms=transforms,
        video_frames=(0, -1, 1),
        disable_const_items=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti (odom10 split) validation set for pose validation",
          flush=True)

    return loader
