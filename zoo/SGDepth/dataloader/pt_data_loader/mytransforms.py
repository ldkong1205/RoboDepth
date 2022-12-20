# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""this module provides transform operations which can be used
to augment the image. Desirable transformations might be stuff like
rotation, scaling, totensor, randomcrop, horizontal flip"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np
import torch
import PIL.Image as pil
from PIL import ImageFilter
import cv2
import random
import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_fun

IMAGENAMES = ['color', 'segmentation', 'depth', 'flow']
NUMERICNAMES = ['camera_intrinsics', 'poses', 'velocity', 'timestamp']


class LoadRGB(object):
    """ Loads the RGB image by converting the array to a PIL image """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'color' in name:
                if sample[key].dtype == 'uint16':
                    sample[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2RGB).astype(np.float32) / 256.
                    sample[key] = sample[key].astype(np.uint8)
                elif sample[key].dtype == 'uint8':
                    sample[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2RGB)
                else:
                    sample[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2RGB)
                sample[key] = pil.fromarray(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class LoadSegmentation(object):
    """ Creates PIL Image from the dataset segmentation (numpy array) """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'segmentation' in name:
                sample[key] = pil.fromarray(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__

    def inverse(self, sample):
        """
        performs the exact inverse operation to the sample as the normal call
        """
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'segmentation' in name:
                sample[key] = np.array(sample[key])
        return sample


class ConvertSegmentation(object):
    """ Convert the labels to their train IDs """

    def __init__(self, labels=None, labels_mode=None, original=False):
        """ Creates a ConvertSegmentation object

        :param labels: labels from the definitions file
        :param labels_mode: mode by which to read the labels file. Can be 'fromid', 'fromtrainid' or 'fromrgb' or
            'fromid_third_channel' depending on the dataset format
        :param original: if 'True', the original image will be converted. Otherwise, the scaled image
            will be converted (recommended as it's much faster). If no scaled image is used / existent, the argument has
            to be set 'True' manually!
            Default: 'False'
        """
        self.original = bool(original)
        self._set_mode(labels=labels, labels_mode=labels_mode)

    def _set_mode(self, labels, labels_mode):
        if labels_mode not in ('fromid', 'fromtrainid', 'fromrgb', 'fromid_third_channel', None):
            raise NotImplementedError('Mode is not implemented, please choose one of the following:' +
                                      ' ("fromid", "fromtrainid", "fromrgb", "fromid_third_channel")')
        self.mode = labels_mode

        if self.mode is not None and labels is not None:
            self.train_ids = np.array(tuple(l.trainId for l in labels))
            self.colors = np.array(tuple(l.color for l in labels))
            self.ids = np.array(tuple(l.id for l in labels))

    def set_mode(self, labels, labels_mode):
        self._set_mode(labels=labels, labels_mode=labels_mode)

    def _filter_keys(self, sample):
        for key in tuple(sample):
            if not isinstance(key, tuple):
                continue

            if len(key) != 3:
                continue

            name, _, res = key

            if self.original and (res != -1):
                continue

            if not self.original and (res == -1):
                continue

            if not name.startswith('segmentation'):
                continue

            yield key

    def _from_rgb(self, img):
        if (img.ndim != 3) or (img.shape[2] != 3):
            raise ValueError('Mode "fromrgb" expects the input image to have shape (H, W, 3)')

        # Using BGR color instead of RGB is a good idea
        # (said no-one ever but opencv devs).
        img = img[...,::-1]

        # Take the image from dimension (H, W, 3) to (H, W, 1, 3).
        # Substract the colors array of dimension (N_CLASSES, 3).
        # Take the absolute of the resulting (H, W, N_CLASSES, 3) array.
        # Sum over the color channel, resulting in a (H, W, N_CLASSES) array.
        # The array will contain zeros in the N_CLASSES axis whenever the class
        # color matches the pixel color.
        return np.sum(
            np.abs(np.expand_dims(img, 2) - self.colors),
            3
        )

    def _from_id(self, img):
        if img.ndim != 2:
            raise ValueError('Mode "fromid" expects the input image to have shape (H, W)')

        # Subtract the (N_CLASSES)-shaped ids array from the (H, W)-shaped
        # image array, resulting in a (H, W, N_CLASSES)-shaped array and take the absolute.
        # The array will contain zeros in the N_CLASSES axis whenever
        # the class id matches the value in img.
        return np.abs(np.expand_dims(img, 2) - self.ids)

    def _from_id_third_channel(self, img):
        return self._from_id(img[..., 2])

    def __call__(self, sample):
        if (self.mode is None) or (self.mode == 'fromtrainid'):
            return sample

        for key in self._filter_keys(sample):
            img = sample.pop(key)
            img = np.array(img)

            if self.mode == 'fromrgb':
                error = self._from_rgb(img)

            elif self.mode == 'fromid':
                error = self._from_id(img)

            elif self.mode == 'fromid_third_channel':
                error = self._from_id_third_channel(img)

            else:
                raise ValueError('Unkown label mode')

            # Select the indices to the N_CLASSES axis that have the smallest color/id error.
            # Replace the indices with train_ids.
            idx = np.argmin(error, 2)
            train_ids = self.train_ids[idx].astype(np.uint8)
            sample[key] = pil.fromarray(train_ids)

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class LoadDepth(object):
    """ Creates PIL Image from the dataset depth (numpy array) """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'depth' in name:
                sample[key] = pil.fromarray(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__

    def inverse(self, sample):
        """
        performs the exact inverse operation to the sample as the normal call
        """
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'depth' in name:
                sample[key] = np.array(sample[key])
        return sample


class ConvertDepth(object):
    """ Converts the depth image to depth in meters """

    def __init__(self, depth_mode=None):
        self.depth_mode = depth_mode

    def set_mode(self, depth_mode):
        self.depth_mode = depth_mode

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'depth' in name:
                sample[key] = np.array(sample[key]).astype(np.float)
                if self.depth_mode == 'uint_16':
                    sample[key] = sample[key] / 256.
                elif self.depth_mode == 'uint_16_subtract_one':   # This mode is specifically tailored to fit Cityscapes
                    sample[key][sample[key] > 1.0] = 0.209313 * 2262.52/(
                            (np.array(sample[key][sample[key] > 1.0]).astype(np.float) - 1.0) / 256.)
                elif self.depth_mode == 'normalized_100':
                    sample[key] = sample[key] / 100.
                elif self.depth_mode == '3_channel_normalized_100':
                    sample[key] = sample[key] / 100.
                    sample[key] = sample[key][:, :, 0].reshape(sample[key].shape[0], sample[key].shape[1])
                else:
                    raise Exception('Unknown Depth Mode')
                sample[key] = pil.fromarray(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__

    def inverse(self, sample):
        """ Performs the exact inverse operation to the sample as the normal call """

        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'depth' in name:
                if self.depth_mode == 'uint_16':
                    sample[key] = np.array(sample[key]).astype(np.float) * 256.
                elif self.depth_mode == 'uint_16_subtract_one':
                    raise NotImplementedError
                elif self.depth_mode == 'normalized_100':
                    sample[key] = np.array(sample[key]).astype(np.float) * 100.
                elif self.depth_mode == '3_channel_normalized_100':
                    mono_image = sample[key]
                    sample[key] = np.zeros((mono_image.shape[0], mono_image.shape[1], 3))
                    for i in range(3):
                        sample[key][:, :, i] = mono_image
                    sample[key] = np.array(sample[key]).astype(np.float) * 100.
        return sample


class LoadFlow(object):
    """ Creates PIL Image from the dataset optical flow image (numpy array) """

    def __init__(self, validation_mode):
        self.validation_mode = validation_mode

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            # In validation mode, keep the data type as a numpy array. Otherwise, convert to uint8 PIL image.
            if 'flow' in name:
                sample[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2RGB)
                if not self.validation_mode:
                    sample[key][:, :, 0:2] = sample[key][:, :, 0:2].astype(np.float32) / 256.
                    sample[key] = pil.fromarray(sample[key].astype(np.uint8))
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class ConvertFlow(object):
    """ Converts the flow image.

    This transform must be executed after all image-altering transforms, e.g. rotating, scaling and cropping since
    the converted flow data will not be a pillow image anymore.
    """

    def __init__(self, flow_mode=None, validation_mode=None):
        self.flow_mode = flow_mode
        self.validation_mode = validation_mode

    def set_mode(self, flow_mode=None, validation_mode=None):
        assert (flow_mode is not None or validation_mode is not None), 'Please enter a flow_mode or a validation_mode'
        if flow_mode is not None:
            self.flow_mode = flow_mode
        if validation_mode is not None:
            self.validation_mode = validation_mode

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'flow' in name:
                # Convert the flow to a range -512 ... 512 (1st and 2nd channel)
                sample[key] = np.array(sample[key]).astype(np.float)
                if not self.validation_mode:
                    sample[key][:, :, 0:2] = sample[key][:, :, 0:2] * 256
                if self.flow_mode == 'kitti':
                    sample[key][:, :, 0:2] = (sample[key][:, :, 0:2] - 2**15) / 64
                else:
                    raise Exception('Unknown Flow Mode')
                # Set flow values for invalid pixels to 0 (e.g. for padded pixels at the corners)
                sample[key][:, :, 0] = sample[key][:, :, 2] * sample[key][:, :, 0]
                sample[key][:, :, 1] = sample[key][:, :, 2] * sample[key][:, :, 1]
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class LoadNumerics(object):
    """ Loads the numeric values, which are not images and are not subject to image pre-processing """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in NUMERICNAMES):
                sample[key] = np.array(sample[key]).astype(np.float)
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class ExchangeStereo(object):
    """ Exchanges the roles of the left and the right image """

    def __init__(self):
        pass

    def _should_flip(self):
        return True

    def __call__(self, sample):
        if not self._should_flip():
            return sample

        new_sample = {}

        for key in sample:
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                new_sample[key] = sample[key]
                continue

            if any(item in name for item in IMAGENAMES):
                if 'right' in name:
                    new_key = (key[0][:-6], key[1], key[2])
                else:
                    new_key = (key[0] + '_right', key[1], key[2])

                new_sample[new_key] = sample[key]

            else:
                new_sample[key] = sample[key]

        if 'stereo_T' in new_sample:
            new_sample['stereo_T'][0, 3] *= -1

        return new_sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomExchangeStereo(ExchangeStereo):
    """ Randomly exchanges the roles of the left and the right image """

    def _should_flip(self):
        return random.uniform(0, 1) < 0.5


class RemoveRightStereo(object):
    """ Removes right stereo images """

    def __init__(self):
        pass

    def __call__(self, sample):
        for key in list(sample.keys()):
            if isinstance(key, tuple) and len(key) == 3:
                if 'right' in key[0]:
                    del sample[key]

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomHorizontalFlip(object):
    """ Randomly flips the image horizontally """

    def __init__(self):
        pass

    def __call__(self, sample):
        is_flip = random.uniform(0, 1) < 0.5
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and is_flip:
                sample[key] = transforms_fun.hflip(sample[key])
        if is_flip and 'stereo_T' in list(sample.keys()):
            sample['stereo_T'][0, 3] *= -1
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomVerticalFlip(object):
    """ Randomly flips the image vertically """

    def __init__(self):
        pass

    def __call__(self, sample):
        is_flip = random.uniform(0, 1) < 0.5
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and is_flip:
                sample[key] = transforms_fun.vflip(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class CreateScaledImage(object):
    """ Creates the scaled image objects inside the dictionary, no parameters """

    def __init__(self, keep_originals=True):
        self.keep_originals = keep_originals

    def __call__(self, sample):
        new_sample = dict()

        if self.keep_originals:
            new_sample.update(sample)

        for key in sample:
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
                frame = key[1]

                new_sample[(name, frame, 0)] = sample[key]

            elif isinstance(key, tuple) and len(key) == 2:
                name = key[0]

                new_sample[(name, 0)] = sample[key]

            else:
                new_sample[key] = sample[key]

        return new_sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomRotate(object):
    """ Rotates the image randomly in a specified angular range

    This transform will rotate the image in a given range. Afterwards, a center crop is performed to ensure that
    there is no boundary with invalid values. The crop size is always determined beased on the largest possible
    rotation.
    """

    def __init__(self, rotation, fraction=1.0):
        """ Creates a RandomRotate object

        :param rotation: defines the rotation angle in degrees. Can be float or 2-tuple with lower and upper boundary
        :param fraction: defines which fraction of images is supposed to be randomly rotated
        """
        assert isinstance(rotation, (float, tuple)), 'rotation has to be a float or a tuple'
        assert isinstance(fraction, float), 'fraction has to be a float'
        assert fraction >= 0 and fraction <= 1, 'fraction has to be between 0 and 1'
        if isinstance(rotation, float):
            self.rotation = (-rotation, rotation)
        else:
            self.rotation = rotation
        self.fraction = fraction

    def __call__(self, sample):
        is_rotate = random.uniform(0, 1) < self.fraction
        run_rotation = random.uniform(self.rotation[0], self.rotation[1])
        max_rotation = max(abs(self.rotation[0]), abs(self.rotation[1]))
        print(max_rotation)
        im_shape = sample[('color', 0, 0)].size
        crop_shape = self._getCropSize(im_shape[0], im_shape[1], max_rotation)
        print("Im-Shape: {}, Crop-Shape: {}".format(im_shape, crop_shape))
        cropper = CenterCrop(crop_shape)
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3 and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and is_rotate:
                if 'color' in name or ('depth' in name and 'processed' in name):
                    sample[key] = transforms_fun.affine(sample[key], angle=run_rotation, translate=(0, 0),
                                                        scale=1.0, shear=0, resample=pil.BILINEAR)
                elif 'segmentation' in name or 'depth' in name or 'flow' in name:
                    sample[key] = transforms_fun.affine(sample[key], angle=run_rotation, translate=(0, 0),
                                                        scale=1.0, shear=0, resample=pil.NEAREST)
        if is_rotate:
            sample = cropper(sample)
        return sample

    def _getCropSize(self, w, h, angle):
        """
        Given a rectangle of size w x h that has been rotated by 'angle' (in degrees), computes the width and height of
        the largest possible axis-aligned rectangle (maximal area) within the rotated rectangle.

        This solution is taken from the StackOverflow user coproc: https://stackoverflow.com/a/16778797
        """
        angle = angle * math.pi / 180
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
        return hr, wr

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomTranslate(object):
    """ Translates the image randomly in a given range.

    This transform will translate the image in a given range. Afterwards, a center crop is performed to ensure that
    there is no boundary with invalid values. The crop size is always determined beased on the largest possible
    translation.
    """

    def __init__(self, translation, fraction=1.0):
        """ Creates a RandomTranslate object

        :param translation: defines how many pixels the image is randomly translated. Can be a positive integer or a
            tuple for x and y direction
        :param fraction: defines which fraction of images is supposed to be randomly translated
        """
        assert isinstance(translation, (int, tuple))
        assert isinstance(fraction, float), 'fraction has to be a float'
        assert fraction >= 0 and fraction <= 1, 'fraction has to be between 0 and 1'
        if isinstance(translation, int):
            assert translation >= 0, 'translation can only be positive'
            self.translation = (translation, translation)
        else:
            assert translation[0] >= 0 and translation[1] >= 0, 'translation can only be positive'
            self.translation = translation
        self.fraction = fraction

    def __call__(self, sample):
        is_trans = random.uniform(0, 1) < self.fraction
        run_translate = (random.randint(-self.translation[0], self.translation[0]),
                         random.randint(-self.translation[1], self.translation[1]))
        im_shape = sample[('color', 0, 0)].size
        crop_shape = (im_shape[1] - 2*self.translation[1], im_shape[0] - 2*self.translation[0])
        cropper = CenterCrop(crop_shape)
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3 and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and is_trans:
                sample[key] = transforms_fun.affine(sample[key], angle=0, translate=run_translate, scale=1.0, shear=0)
        if is_trans:
            sample = cropper(sample)
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomRescale(object):
    """ Rescales the image randomly """

    def __init__(self, scale, fraction=1.0):
        """ Creates a RandomRescale object

        :param scale: defines how many pixels the image is randomly rescaled
        :param fraction: defines which fraction of images is supposed to be randomly rescaled
        """
        assert isinstance(scale, (float, tuple))
        assert isinstance(fraction, float), 'fraction has to be a float'
        assert fraction >= 0 and fraction <= 1, 'fraction has to be between 0 and 1'
        if isinstance(scale, float):
            self.scale = (1.0, scale)
        else:
            self.scale = scale
        self.fraction = fraction

    def __call__(self, sample):
        is_rescale = random.uniform(0, 1) < self.fraction
        if len(self.scale) == 2:
            run_scale = random.uniform(self.scale[0], self.scale[1])
        else:
            tuple_pos = random.randint(0, len(self.scale) - 1)
            run_scale = self.scale[tuple_pos]
        native_im_shape = sample[('color', 0, 0)].size
        output_size = (int(native_im_shape[1] // run_scale), int(native_im_shape[0] // run_scale))
        resize_interp = transforms.Resize(output_size, interpolation=pil.BILINEAR)
        resize_nearest = transforms.Resize(output_size, interpolation=pil.NEAREST)
        for key in sample.keys():
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and is_rescale:
                if 'color' in name or ('depth' in name and 'processed' in name):
                    sample[key] = resize_interp(sample[key])
                elif 'segmentation' in name or 'depth' in name or 'flow' in name:
                    sample[key] = resize_nearest(sample[key])

            if 'depth' in name:
                sample[key] = pil.fromarray(np.asarray(sample[key])/run_scale)
            elif 'camera_intrinsics' in name or 'K' in name:
                K = sample[key]
                K[0, :] = K[0, :] / run_scale
                K[1, :] = K[1, :] / run_scale
                sample[key] = K
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class Resize(object):
    """ Rescales the image in a sample to a given size """

    def __init__(self, output_size, image_types=None, exceptions=None, aspect_ratio=False, multiple_of=1):
        """ Creates a Resize object.

        :param output_size: output size of the image, it is resized always from native resolution. Can be an integer to
            specify the new scale in y direction or a tuple for x and y direction
        :param image_types: image types which are supposed to be resized
        :param exceptions: image types which are not to be resized
        :param aspect_ratio: preserve the aspect ratio when resizing (only if output_size is of type tuple). The bigger
            scale factor is always chosen, i.e. if scale_x = output_x / native_x = 0.8 and scale_y = 0.7, the
            y-direction will be resized with a scale factor of 0.8 too.
        :param multiple_of: if aspect_ratio is True, the dimension having the smaller scale factor will be additionally
            shrunk to being a multiple of multiple_of, e. g. multiple_of = 16, then the y-dimension of the aspect_ratio
            example will be a multiple of 16 and maximally 0.8 * native_y.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            assert output_size > 0, 'output_size must be > 0'
        else:
            assert output_size[0] > 0 and output_size[1] > 0, 'output_size must be > 0'
        assert isinstance(aspect_ratio, bool), "aspect ratio must be of type bool"
        assert isinstance(multiple_of, int) and multiple_of > 0, "multiple_of must be type int and > 0"
        self.output_size = output_size
        self.image_types = image_types
        self.exceptions = exceptions
        self.aspect_ratio = aspect_ratio
        self.mof = multiple_of

    def __call__(self, sample):
        native_im_shape = sample[('color', 0, 0)].size
        output_size = self.output_size

        if self.aspect_ratio and isinstance(self.output_size, tuple):
            scale_0 = self.output_size[0] / native_im_shape[1]
            scale_1 = self.output_size[1] / native_im_shape[0]
            output_size = self.get_new_dim(native_im_shape, max(scale_0, scale_1))

        resize_interp = transforms.Resize(output_size, interpolation=pil.BILINEAR)
        resize_nearest = transforms.Resize(output_size, interpolation=pil.NEAREST)

        for key in list(sample.keys()):
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue

            if self.image_types is not None and not any(item in name for item in self.image_types):
                continue
            if self.exceptions is not None and any(item in name for item in self.exceptions):
                continue

            if 'color' in name or ('depth' in name and 'processed' in name):
                new_image = resize_interp(sample[key])
            elif 'segmentation' in name or 'depth' in name or 'flow' in name:
                new_image = resize_nearest(sample[key])
            elif 'camera_intrinsics' in name or 'K' in name:
                K = sample[key].copy()
                K[0, :] *= self.output_size[1] / native_im_shape[0]
                K[1, :] *= self.output_size[0] / native_im_shape[1]
                new_image = K

            sample[key] = new_image
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__

    def get_new_dim(self, native, scale):
        dim_0 = int(np.ceil(scale * native[1]) / self.mof) * self.mof
        dim_1 = int(np.ceil(scale * native[0]) / self.mof) * self.mof
        return dim_0, dim_1


class MultiResize(object):
    """ Rescales the image in a sample to a given size """

    def __init__(self, scales, image_types=['color', 'camera_intrinsics', 'K'], exceptions=None):
        """ Creates a MultiResize object.

        :param scales: scales at which to resize, 0 would be the scale of output size, all others are power of 2
        :param image_types: image types which are supposed to be resized
        :param exceptions: image types which are not to be resized
        """
        self.scales = scales
        self.image_types = image_types
        self.exceptions = exceptions

    def __call__(self, sample):
        native_im_shape = np.array(sample[('color', 0, 0)].size)
        native_im_shape = np.roll(native_im_shape, 1)
        for key in list(sample.keys()):
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue

            if self.image_types is not None and not any(item in name for item in self.image_types):
                continue
            if self.exceptions is not None and any(item in name for item in self.exceptions):
                continue

            for scale in self.scales:
                scale_factor = 2 ** scale
                if 'color' in name or ('depth' in name and 'processed' in name):
                    new_image = transforms_fun.resize(sample[key], native_im_shape//scale_factor,
                                                      interpolation=pil.BILINEAR)
                elif 'segmentation' in name or 'depth' in name or 'flow' in name:
                    new_image = transforms_fun.resize(sample[key], native_im_shape//scale_factor,
                                                      interpolation=pil.NEAREST)
                elif 'camera_intrinsics' in name or 'K' in name:
                    K = sample[key].copy()
                    K[0, :] = K[0, :] / scale_factor
                    K[1, :] = K[1, :] / scale_factor
                    new_image = K

                new_key = list(key)
                new_key[-1] = scale
                new_key = tuple(new_key)
                sample.update({new_key: new_image})
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RandomCrop(object):
    """ Crop randomly the image in a sample. All images must have same dimension! If the parameter pad_if_needed is set
    to True, a padding is performed at the sides for which the crop sizes exceeds the image size.
    """

    def __init__(self, output_size, pad_if_needed=False):
        """ Creates a RandomCrop object

        :param output_size: Desired output size. If int, square crop is made.
        :type output_size: tuple or int
        :param pad_if_needed: If true, the image is padded at the sides to match the desired output size
        :type pad_if_needed: boolean
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.pad_if_needed = pad_if_needed

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and key[-1] == 0:
                w, h = sample[key].size[:2]
                break
        new_h, new_w = self.output_size
        side_padding = False
        if self.pad_if_needed:
            side_padding_size_h = max(0, new_h-h)
            h = h + 2 * side_padding_size_h
            side_padding_size_w = max(0, new_w-w)
            w = w + 2 * side_padding_size_w
            side_padding_size = (side_padding_size_w, side_padding_size_h)
            if side_padding_size != (0,0):
                side_padding = True
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        for key in sample.keys():
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES):
                if side_padding:
                    sample[key] = transforms_fun.pad(sample[key], padding=side_padding_size, fill=0)
                sample[key] = transforms_fun.crop(sample[key], top, left, new_h, new_w)
            elif 'camera_intrinsics' in name or 'K' in name:
                K = sample[key]
                K[0, 2] = new_w / 2.0
                K[1, 2] = new_h / 2.0
                sample[key] = K

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class CenterCrop(object):
    """Center crop the image in a sample. All images must have same dimension! If the size of the crops
    exceeds the image size, an automatic padding is performed at the borders for the color, depth and
    segmentation images with the padding value 0.
    """

    def __init__(self, output_size, pad_if_needed=False):
        """ Creates a CenterCrop object.

        :param output_size: Desired output size. If int, square crop is made.
        :type output_size: tuple or int
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.pad_if_needed = pad_if_needed

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and key[-1] == 0:
                w, h = sample[key].size[:2]
                break
        new_h, new_w = self.output_size
        centercrop = transforms.CenterCrop((new_h, new_w))
        for key in sample.keys():
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES):
                sample[key] = centercrop(sample[key])
            elif 'camera_intrinsics' in name or 'K' in name:
                K = sample[key]

                K[0, 2] = new_w / 2.0
                K[1, 2] = new_h / 2.0
                sample[key] = K

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class SidesCrop(object):
    """ Crops the image at the sides, with the specified parameters. """

    def __init__(self, hw, tl):
        """ Creates a SidesCrop object

        :param hw: 2-tuple with height and width of the image
        :param tl: 2-tuple with pixels at the top and left to crop
        """
        assert isinstance(hw, tuple)
        assert isinstance(tl, tuple)
        assert hw[0] > 0 and hw[1] > 0, 'image dimensions hw must be >0'
        self.top = tl[0]
        self.left = tl[1]
        self.height = hw[0]
        self.width = hw[1]

    def __call__(self, sample):
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES) and key[-1] == 0:
                w, h = sample[key].size[:2]
                break
        new_w, new_h = self.width, self.height
        for key in sample.keys():
            if isinstance(key, tuple) and key[-1] == 0:
                name = key[0]
            else:
                continue
            if any(item in name for item in IMAGENAMES):
                sample[key] = transforms_fun.crop(sample[key], 0, 0, self.height, self.width)
            elif 'camera_intrinsics' in name or 'K' in name:
                K = sample[key]

                K[0, 0] *= new_w / w
                K[1, 1] *= new_h / h

                K[0, 2] = new_w / 2.0
                K[1, 2] = new_h / 2.0

                sample[key] = K

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class CreateColoraug(object):
    """ Creates the color_aug object from the color images """

    def __init__(self, scales=[0], new_element=True):
        """ Creates a CreateColoraug object.

        :param scales: scales at which the color augmentation is applied
        :param new_element: defines if a new element is created in addition to the normal color image
        :type new_element: boolean
        """
        self.new_element = new_element
        self.scales = scales

    def __call__(self, sample):
        for key in list(sample.keys()):
            if isinstance(key, tuple):
                name = key[0]
                scale = key[-1]
                if scale not in self.scales:
                    continue
            else:
                continue

            if 'color' in name:
                sample.update({(name + '_aug', key[1], scale): sample[key]})
                if self.new_element is False:
                    del sample[key]

            if 'K' in name:
                K = sample[key]
                inv_K = np.linalg.pinv(K)

                sample[('inv_K', scale)] = inv_K

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class ColorJitter(object):
    """ Adjust the Brightness, Saturation, Contrast, Hue and gamma values of the input image. Needs a color_aug object.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0, fraction=1.0):
        """ Creates a ColorJitter object.

        :param brightness: adjust between 1-brightnes and 1+brightness
        :param contrast: adjust between 1-contrast and 1+contrast
        :param saturation: adjust between 1- saturation and 1+ saturation
        :param hue: adjust between -hue and +hue
        :param gamma: adjust the gamme between w and 1+gamma
        :param fraction: fraction at which probability the color-jitter is applied to the images
        """
        assert isinstance(fraction, float), 'fraction has to be a float'
        assert fraction >= 0 and fraction <= 1, 'fraction has to be between 0 and 1'
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.fraction = fraction

    def __call__(self, sample):
        run_brightness = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        run_contrast = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        run_saturation = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        run_hue = random.uniform(max(-0.5, -self.hue), min(self.hue, 0.5))
        run_gamma = random.uniform(1, 1 + self.gamma)

        is_color_jitter = random.uniform(0.0, 1.0) < self.fraction

        for key in list(sample.keys()):
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'color' in name and 'aug' in name:
                image = sample[key]
                if is_color_jitter:
                    image = transforms_fun.adjust_brightness(image, run_brightness)
                    image = transforms_fun.adjust_contrast(image, run_contrast)
                    image = transforms_fun.adjust_saturation(image, run_saturation)
                    image = transforms_fun.adjust_hue(image, run_hue)
                    image = transforms_fun.adjust_gamma(image, run_gamma)

                sample[key] = image
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class GaussianBlurr(object):
    """ Performs a gaussian blurr on the image. Needs a color_aug object """

    def __init__(self, fraction=1.0, max_rad = 1.0):
        """ Creates a GaussianBlurr object.

        :param fraction: fraction at which probability the color-jitter is applied to the images
        :param max_rad: max. blurr radius
        """
        assert isinstance(fraction, float), 'fraction has to be a float'
        assert fraction >= 0 and fraction <= 1, 'fraction has to be between 0 and 1'
        self.fraction = fraction
        self.max_rad = max_rad

    def __call__(self, sample):
        blurr_radius = random.uniform(0, self.max_rad)
        is_blurr = random.uniform(0.0, 1.0) < self.fraction
        for key in list(sample.keys()):
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'color' in name and 'aug' in name:
                image = sample[key]
                if is_blurr:
                    image = image.filter(ImageFilter.GaussianBlur(radius=blurr_radius))
                sample[key] = image
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RemoveOriginals(object):
    """ Remove the images at native scale to save loading time """
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in list(sample.keys()):
            if not isinstance(key, tuple):
                continue

            if key[-1] == -1:
                del sample[key]

        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __init__(self):
        pass

    def __call__(self, sample):
        torch_dict = {}
        transformer = transforms.ToTensor()
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                torch_dict.update({key: torch.from_numpy(sample[key])})
                continue

            if 'segmentation' in name or 'depth' in name or 'flow' in name:
                if len(np.array(sample[key]).shape) == 2:
                    sample[key] = np.expand_dims(np.array(sample[key]), 2)
                sample[key] = np.transpose(np.array(sample[key]), (2, 0, 1)).astype(np.float32)
                torch_dict.update({key: torch.from_numpy(sample[key])})
            elif 'color' in name:
                torch_dict.update({key: transformer(sample[key])})
            elif any(item in name for item in NUMERICNAMES):
                torch_dict.update({key: torch.from_numpy(sample[key])})
            else:
                RuntimeError
        return torch_dict

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class Relabel(object):
    """ Relabels the segmentation masks (throws sometimes errors, if not done!) """

    def __init__(self, old_label, new_label):
        self.olabel = old_label
        self.nlabel = new_label

    def __call__(self, sample):
        torch_dict = sample.copy()
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                torch_dict.update({key: sample[key]})
                continue

            if 'segmentation' in name:
                if len(np.array(sample[key]).shape) == 2:
                    sample[key] = np.expand_dims(np.array(sample[key]), 2)
                sample[key][sample[key] == self.olabel] = self.nlabel
                torch_dict.update({key: sample[key]})
            else:
                RuntimeError
        return torch_dict


class OneHotEncoding(object):
    """ Relabels the segmentation masks (throws sometimes errors, if not done!) """

    def __init__(self, num_classes=20):
        self.num_classes = num_classes

    def __call__(self, sample):
        torch_dict = sample.copy()
        for key in sample.keys():
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                torch_dict.update({key: sample[key]})
                continue

            if 'segmentation' in name:
                sample[key] = torch.eye(self.num_classes)[sample[key].long()].squeeze().permute(2,0,1) # one_hot encoding
                torch_dict.update({key: sample[key]})
            else:
                RuntimeError
        return torch_dict


class NormalizeZeroMean(object):
    """ Zero means normalization of the image to a certain range, standard values
    are for pretraining on Imagenet for torchvision models. Needs a color_aug object.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """ Creates a NormalizeZeroMean object

        :param mean: mean of the three channels
        :param std: standard deviation of the three channels
        """
        assert isinstance(mean, tuple) and len(mean) == 3, 'mean has to be a 3-tuple'
        assert isinstance(std, tuple) and len(std) == 3, 'std has to be a 3-tuple'
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        for key in list(sample.keys()):
            if isinstance(key, tuple) and len(key) == 3:
                name = key[0]
            else:
                continue
            if 'color' in name and 'aug' in name:
                sample[key] = normalize(sample[key])
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class AdjustKeys(object):
    """ Adjust the dictionary keys of the dataset to the needs of the model """
    def __init__(self, model):
        self.model = model

    def __call__(self, sample):
        new_sample = {}
        if self.model == 'monodepth2':
            for key in list(sample.keys()):
                if not isinstance(key, tuple):
                    new_key = key
                elif key[0] == 'color_right' and key[1] == 0:
                    new_key = ('color', 's', key[2])
                elif key[0] == 'color_right_aug' and key[1] == 0:
                    new_key = ('color_aug', 's', key[2])
                else:
                    new_key = key
                new_sample.update({new_key: sample[key]})
        return new_sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class RemapKeys(object):
    def __init__(self, remap):
        self.remap = remap

    def __call__(self, sample):
        new_sample = dict()

        for k, v in sample.items():
            new_sample[self.remap[k] if (k in self.remap) else k] = v

        return new_sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__


class AddKeyValue(object):
    """ Add a constant key/value pair """

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __call__(self, sample):
        sample[self.key] = self.value
        return sample

    def __eq__(self, other):
        return type(self).__name__ == other.__name__
