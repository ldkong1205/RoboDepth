import torch
from dataloader.definitions.labels_file import labels_cityscape_seg

# Extract the Cityscapes color scheme
TRID_TO_LABEL = labels_cityscape_seg.gettrainid2label()

COLOR_SCHEME_CITYSCAPES = torch.tensor(
    tuple(
        TRID_TO_LABEL[tid].color if (tid in TRID_TO_LABEL) else (0, 0, 0)
        for tid in range(256)
    )
).float() / 255
