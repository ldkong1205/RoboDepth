# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from functools import partial
from PIL import Image  # using pillow-simd for increased speed

from .cityscapes_dataset import CityscapesEvalDataset

def pil_loader_foggy(path, replace_dict: str = {}, beta: float = None):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    for key, value in replace_dict.items():
        path = path.replace(key, value)
    path = path[:-4] + f"_foggy_beta_{beta}.png"
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CityscapesFoggyEvalDataset(CityscapesEvalDataset):
    """Cityscapes evaluation dataset - here we are loading the raw, original images rather than
    preprocessed triplets, and so cropping needs to be done inside get_color.
    """
    RAW_HEIGHT = 1024
    RAW_WIDTH = 2048

    def __init__(self, *args, **kwargs):
        super(CityscapesFoggyEvalDataset, self).__init__(*args, **kwargs)
        
        replace_dict = {"/cpfs01/shared/llmit/llmit_hdd/xieshaoyuan/CityScapes/leftImg8bit": "/cpfs01/shared/llmit/llmit_hdd/xieshaoyuan/Foggy-CityScapes/leftImg8bit_foggy"}
        beta = self.opt.beta
        self.loader = partial(pil_loader_foggy, replace_dict=replace_dict, beta=beta)