from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        if self.no_normalize:
            sample['left'] = torch.from_numpy(left)
        else:
            sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))

        if self.no_normalize:
            sample['right'] = torch.from_numpy(right)
        else:
            sample['right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]

        # pad zero when crop size is larger than original image size
        if self.img_height > ori_height or self.img_width > ori_width:

            # can be used for only pad one side
            top_pad = max(self.img_height - ori_height, 0)
            right_pad = max(self.img_width - ori_width, 0)

            # try edge padding
            sample['left'] = np.lib.pad(sample['left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='edge')
            sample['right'] = np.lib.pad(sample['right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='edge')

            if 'disp' in sample.keys():
                sample['disp'] = np.lib.pad(sample['disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

            # update image resolution
            ori_height, ori_width = sample['left'].shape[:2]

        assert self.img_height <= ori_height and self.img_width <= ori_width

        # Training: random crop
        self.offset_x = np.random.randint(ori_width - self.img_width + 1)

        start_height = 0
        assert ori_height - start_height >= self.img_height

        self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

        sample['left'] = self.crop_img(sample['left'])
        sample['right'] = self.crop_img(sample['right'])
        if 'disp' in sample.keys():
            sample['disp'] = self.crop_img(sample['disp'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))

            sample['disp'] = np.copy(np.flipud(sample['disp']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                contrast_factor = np.random.uniform(0.8, 1.2)

            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['right'] = F.adjust_gamma(sample['right'], gamma)

        return sample


class RandomBrightness(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left'] = F.adjust_brightness(sample['left'], brightness)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                brightness = np.random.uniform(0.5, 2.0)

            sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                hue = np.random.uniform(-0.1, 0.1)

            sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_saturation(sample['left'], saturation)

            if self.asymmetric_color_aug and np.random.random() < 0.5:
                saturation = np.random.uniform(0.8, 1.2)

            sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __init__(self,
                 asymmetric_color_aug=True,
                 ):

        self.asymmetric_color_aug = asymmetric_color_aug

    def __call__(self, sample):
        transforms = [RandomContrast(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomGamma(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomBrightness(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomHue(asymmetric_color_aug=self.asymmetric_color_aug),
                      RandomSaturation(asymmetric_color_aug=self.asymmetric_color_aug)]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample


class RandomScale(object):
    def __init__(self,
                 min_scale=-0.4,
                 max_scale=0.4,
                 crop_width=512,
                 nearest_interp=False,  # for sparse gt
                 ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_width = crop_width
        self.nearest_interp = nearest_interp

    def __call__(self, sample):
        if np.random.rand() < 0.5:
            h, w = sample['disp'].shape

            scale_x = 2 ** np.random.uniform(self.min_scale, self.max_scale)

            scale_x = np.clip(scale_x, self.crop_width / float(w), None)

            # only random scale x axis
            sample['left'] = cv2.resize(sample['left'], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)
            sample['right'] = cv2.resize(sample['right'], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)

            sample['disp'] = cv2.resize(
                sample['disp'], None, fx=scale_x, fy=1.,
                interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
            ) * scale_x

            if 'pseudo_disp' in sample and sample['pseudo_disp'] is not None:
                sample['pseudo_disp'] = cv2.resize(sample['pseudo_disp'], None, fx=scale_x, fy=1.,
                                                   interpolation=cv2.INTER_LINEAR) * scale_x

        return sample


class Resize(object):
    def __init__(self,
                 scale_x=1,
                 scale_y=1,
                 nearest_interp=True,  # for sparse gt
                 ):
        """
        Resize low-resolution data to high-res for mixed dataset training
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.nearest_interp = nearest_interp

    def __call__(self, sample):
        scale_x = self.scale_x
        scale_y = self.scale_y

        sample['left'] = cv2.resize(sample['left'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        sample['right'] = cv2.resize(sample['right'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        sample['disp'] = cv2.resize(
            sample['disp'], None, fx=scale_x, fy=scale_y,
            interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
        ) * scale_x

        return sample


class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample = ToPILImage()(sample)

            # only supported in higher version pytorch
            # default output channels is 1
            sample['left'] = F.rgb_to_grayscale(sample['left'], num_output_channels=3)
            sample['right'] = F.rgb_to_grayscale(sample['right'], num_output_channels=3)

            sample = ToNumpyArray()(sample)

        return sample


class RandomRotateShiftRight(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            angle, pixel = 0.1, 2
            px = np.random.uniform(-pixel, pixel)
            ag = np.random.uniform(-angle, angle)

            right_img = sample['right']

            image_center = (
                np.random.uniform(0, right_img.shape[0]),
                np.random.uniform(0, right_img.shape[1])
            )

            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

            sample['right'] = right_img

        return sample


class RandomOcclusion(object):
    def __init__(self, p=0.5,
                 occlusion_mask_zero=False):
        self.p = p
        self.occlusion_mask_zero = occlusion_mask_zero

    def __call__(self, sample):
        bounds = [50, 100]
        if np.random.random() < self.p:
            img2 = sample['right']
            ht, wd = img2.shape[:2]

            if self.occlusion_mask_zero:
                mean_color = 0
            else:
                mean_color = np.mean(img2.reshape(-1, 3), axis=0)

            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])
            img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

            sample['right'] = img2

        return sample
