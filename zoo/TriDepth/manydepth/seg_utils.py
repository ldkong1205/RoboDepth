import torch
from collections import namedtuple

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


def decode_seg_map(seg_map, num_classes=19):
    input_shape = seg_map.shape
    seg_map = seg_map.flatten()
    empty = torch.stack([torch.zeros(seg_map.shape) for _ in range(3)])

    for i in torch.unique(seg_map).data.cpu():
        i = int(i)
        seg_map = seg_map.flatten()

        mask = seg_map == i
        for c in range(3):
            empty[c][mask] = palette[3 * i + c]
    # empty = empty.view(3, *input_shape[1:])
    empty = empty.view(3, *input_shape)
    return empty.byte()


Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])
labels = [
    # name  id  trainId  category catId  hasInstances ignoreInEval color
    Label('unlabeled',              0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle',            1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border',   2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi',             3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static',                 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic',                5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground',                 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road',                   7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk',               8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking',                9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track',             10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building',               11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall',                   12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence',                  13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail',             14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge',                 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel',                 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole',                   17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup',              18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light',          19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign',           20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation',             21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain',                22, 1, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky',                    23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person',                 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider',                  25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car',                    26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck',                  27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus',                    28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan',                29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer',                30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train',                  31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle',             32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle',                33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate',          -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


train_id_to_rel_depth = {
    5: 1,   # pole

    13: 7,   # car
    14: 7,   # truck
    7: 8,    # traffic sign
    # 6: 9,    # traffic light
    8: 10,   # vegetation (tree actually, )

    4: 21,   # fence
    3: 22,   # wall
    2: 23,   # building

    9: 255,   # terrain (grassland, )
    1: 255,   # sidewalk
    0: 255,   # road
    10: 255,  # sky
}


def is_abs_background_clz(trainId):
    return trainId in [
        0, 1, 9, 10,
    ]

def is_abs_foreground_clz(trainId):
    return trainId in [
        # 2,  # building
        # 3,  # wall
        # 4,  # fence
        5,  # pole
        6,  # traffic light
        7,  # traffic sign
        # 8,  # vegetation
        # 11,  # person
        # 12,  # rider
        # 13,  # car
        # 14,  # truck
        # 15,  # bus
        # 16,  # train
        # 17,  # motorcycle
        # 18,  # bicycle
    ]


