import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RoboDepthDataset(Dataset):
    
    def __init__(self, image_list, H, W):
        self.image_list = image_list
        with open(self.image_list, "r") as f:
            self.total_images = f.read().splitlines()
        self.H = H
        self.W = W
    
    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        image_loc = self.total_images[idx]
        image = Image.open(image_loc).convert("RGB")  # [1242, 375]
        image = image.resize((self.W, self.H), Image.ANTIALIAS)  # [640, 192]
        image = np.asarray(image)
        return (image, self.total_images[idx])

