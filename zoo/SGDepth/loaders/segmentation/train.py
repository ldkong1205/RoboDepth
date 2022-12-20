from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
from dataloader.definitions.labels_file import labels_cityscape_seg
import dataloader.pt_data_loader.mytransforms as tf


def cityscapes_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    cityscapes training set.
    """

    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize((resize_height, resize_width)),
        tf.RandomRescale(1.5),
        tf.RandomCrop((crop_height, crop_width)),
        tf.ConvertSegmentation(),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'cityscapes_train_seg'),
        tf.AddKeyValue('purposes', ('segmentation', 'domain')),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset_name = 'cityscapes'

    dataset = StandardDataset(
        dataset=dataset_name,
        trainvaltest_split='train',
        video_mode='mono',
        stereo_mode='mono',
        labels_mode='fromid',
        disable_const_items=True,
        labels=labels,
        keys_to_load=('color', 'segmentation'),
        data_transforms=transforms,
        video_frames=(0,)
    )

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the cityscapes train set for segmentation training", flush=True)

    return loader
