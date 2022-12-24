import os
import re
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

from imagecorruptions import corrupt
               

def create_brightness(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Brightness'.
    """
    corruption = 'brightness'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_dark(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Dark'.
    """
    corruption = 'dark'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = low_light(image, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = corrupted
            im.save(im_path)
    return 


def create_fog(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Fog'.
    """
    corruption = 'fog'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_frost(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Frost'.
    """
    corruption = 'frost'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_snow(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Snow'.
    """
    corruption = 'snow'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_contrast(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Contrast'.
    """
    corruption = 'contrast'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_defocus_blur(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Defocus Blur'.
    """
    corruption = 'defocus_blur'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_glass_blur(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Glass Blur'.
    """
    corruption = 'glass_blur'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_motion_blur(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Motion Blur'.
    """
    corruption = 'motion_blur'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_zoom_blur(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Zoom Blur'.
    """
    corruption = 'zoom_blur'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_elastic(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Elastic Transform'.
    """
    corruption = 'elastic_transform'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_color_quant(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Color Quantization'.
    """
    corruption = 'color_quant'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        image = transforms.ToPILImage()(image)
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = color_quant(image, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = corrupted
            im.save(im_path)
    return 


def create_gaussian_noise(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Gaussian Noise'.
    """
    corruption = 'gaussian_noise'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_impulse_noise(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Impulse Noise'.
    """
    corruption = 'impulse_noise'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_shot_noise(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Shot Noise'.
    """
    corruption = 'shot_noise'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_iso_noise(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'ISO Noise'.
    """
    corruption = 'iso_noise'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = iso_noise(image, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = corrupted
            im.save(im_path)
    return 


def create_pixelate(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'Pixelate'.
    """
    corruption = 'pixelate'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def create_jpeg(loader, save_path=None, severity_levels=[]):
    """
    Create corruptions: 'JPEG Compression'.
    """
    corruption = 'jpeg_compression'
    if len(severity_levels) == 0:
        severity_levels = [1, 2, 3, 4, 5]
    
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        for severity in severity_levels:
            folder_path = os.path.join(save_path, corruption, str(severity))
            if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            im_path = os.path.join(folder_path, name[0])
            if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
            im = transforms.ToPILImage()(corrupted)
            im.save(im_path)
    return 


def copy_clean(loader, save_path=None, severity_levels=[]):
    """
    Copy clean images.
    """
    clean = 'clean'
    folder_path = os.path.join(save_path, clean)
    if not (os.path.isdir(folder_path)): os.makedirs(folder_path)
    for _, batch in enumerate(loader):
        image, name = batch
        image = image.squeeze().numpy()
        im_path = os.path.join(folder_path, name[0])
        if not (os.path.isdir(im_path[:-14])): os.makedirs(im_path[:-14])
        im = transforms.ToPILImage()(image)
        im.save(im_path)
    return 


def low_light(x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity-1]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity-1)
    return x_scaled


def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return Image.fromarray(np.uint8(x))


def color_quant(x, severity):
    bits = 5 - severity + 1
    x = PIL.ImageOps.posterize(x, bits)
    return x


def iso_noise(x, severity):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity-1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return Image.fromarray(np.uint8(x))


def read_pfm(path):
    """
    Read pfm file.
        Args:
            path (str): path to file
        Returns:
            tuple: (data, scale)
    """
    with open(path, "rb") as file:
        color, width, height, scale, endian = None, None, None, None, None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF": color = True
        elif header.decode("ascii") == "Pf": color = False
        else: raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match: width, height = list(map(int, dim_match.groups()))
        else: raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale
