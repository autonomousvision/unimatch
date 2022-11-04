import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms.functional as F
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value)

            if isinstance(value, list):  # multi-frame target images
                sample[key] = [torch.from_numpy(v) for v in value]

        sample['img_ref'] = sample['img_ref'].permute((2, 0, 1)) / 255.  # [3, H, W]
        sample['img_tgt'] = sample['img_tgt'].permute((2, 0, 1)) / 255.  # [3, H, W]

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        norm_keys = ['img_ref', 'img_tgt']

        assert isinstance(sample['img_ref'], torch.Tensor)
        assert sample['img_ref'].size(0) == 3  # [3, H, W]

        for key in norm_keys:
            # multi-frame inference
            if key == 'img_tgt' and isinstance(sample['img_tgt'], list):
                for i in range(len(sample['img_tgt'])):
                    # Images have converted to tensor, with shape [C, H, W]
                    tgt = sample['img_tgt'][i]
                    for t, m, s in zip(tgt, self.mean, self.std):
                        t.sub_(m).div_(s)
                    sample['img_tgt'][i] = tgt
            else:
                # Images have converted to tensor, with shape [C, H, W]
                for t, m, s in zip(sample[key], self.mean, self.std):
                    t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        crop_h, crop_w = self.crop_size

        ori_h, ori_w = sample['img_ref'].shape[:2]

        out_intrinsics = sample['intrinsics'].copy()

        offset_y = np.random.randint(ori_h - crop_h + 1)
        offset_x = np.random.randint(ori_w - crop_w + 1)

        for key in ['img_ref', 'img_tgt', 'depth']:
            sample[key] = sample[key][offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]

        # valid mask for sparse data
        if 'valid' in sample:
            sample['valid'] = sample['valid'][offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]

        out_intrinsics[0, 2] -= offset_x
        out_intrinsics[1, 2] -= offset_y

        sample['intrinsics'] = out_intrinsics

        return sample


class RandomColor(object):
    def __init__(self, asymmetric=True):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        transforms = [RandomContrast(asymmetric=self.asymmetric),
                      RandomGamma(asymmetric=self.asymmetric),
                      RandomBrightness(asymmetric=self.asymmetric),
                      RandomHue(asymmetric=self.asymmetric),
                      RandomSaturation(asymmetric=self.asymmetric)]

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


class RandomResize(object):
    def __init__(self, min_size, min_scale=-0.2, max_scale=0.2):
        # min_size bigger than crop_size
        self.min_size = min_size

        self.min_scale = min_scale
        self.max_scale = max_scale

        self.stretch_prob = 0.4
        self.max_stretch = 0.2

    def __call__(self, sample):
        if np.random.random() < 0.5:
            min_h, min_w = self.min_size
            ori_h, ori_w = sample['img_ref'].shape[:2]

            min_scale = np.maximum(min_h / float(ori_h), min_w / float(ori_w), dtype=np.float32)

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale

            if np.random.random() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None).astype(np.float32)
            scale_y = np.clip(scale_y, min_scale, None).astype(np.float32)

            # Resize
            sample['img_ref'] = cv2.resize(sample['img_ref'], None, fx=scale_x, fy=scale_y,
                                           interpolation=cv2.INTER_LINEAR)
            sample['img_tgt'] = cv2.resize(sample['img_tgt'], None, fx=scale_x, fy=scale_y,
                                           interpolation=cv2.INTER_LINEAR)

            if 'depth' in sample:
                sample['depth'] = cv2.resize(sample['depth'], None, fx=scale_x, fy=scale_y,
                                             interpolation=cv2.INTER_LINEAR)

            if 'valid' in sample:
                sample['valid'] = cv2.resize(sample['valid'], None, fx=scale_x, fy=scale_y,
                                             interpolation=cv2.INTER_LINEAR)
                sample['valid'] = (sample['valid'] > 0.99).astype(np.float32)

            out_intrinsics = sample['intrinsics'].copy()
            out_intrinsics[0] = out_intrinsics[0] * scale_x
            out_intrinsics[1] = out_intrinsics[1] * scale_y
            sample['intrinsics'] = out_intrinsics

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['img_ref'] = Image.fromarray(sample['img_ref'].astype('uint8'))
        sample['img_tgt'] = Image.fromarray(sample['img_tgt'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['img_ref'] = np.array(sample['img_ref']).astype(np.float32)
        sample['img_tgt'] = np.array(sample['img_tgt']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __init__(self, asymmetric=False):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['img_ref'] = F.adjust_contrast(sample['img_ref'], contrast_factor)

            if self.asymmetric and np.random.random() < 0.2:
                contrast_factor = np.random.uniform(0.8, 1.2)

            sample['img_tgt'] = F.adjust_contrast(sample['img_tgt'], contrast_factor)

        return sample


class RandomGamma(object):
    def __init__(self, asymmetric=False):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['img_ref'] = F.adjust_gamma(sample['img_ref'], gamma)

            if self.asymmetric and np.random.random() < 0.2:
                gamma = np.random.uniform(0.7, 1.5)

            sample['img_tgt'] = F.adjust_gamma(sample['img_tgt'], gamma)

        return sample


class RandomBrightness(object):
    def __init__(self, asymmetric=False):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['img_ref'] = F.adjust_brightness(sample['img_ref'], brightness)

            if self.asymmetric and np.random.random() < 0.2:
                brightness = np.random.uniform(0.5, 2.0)

            sample['img_tgt'] = F.adjust_brightness(sample['img_tgt'], brightness)

        return sample


class RandomHue(object):
    def __init__(self, asymmetric=False):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['img_ref'] = F.adjust_hue(sample['img_ref'], hue)

            if self.asymmetric and np.random.random() < 0.2:
                hue = np.random.uniform(-0.1, 0.1)

            sample['img_tgt'] = F.adjust_hue(sample['img_tgt'], hue)

        return sample


class RandomSaturation(object):
    def __init__(self, asymmetric=False):
        self.asymmetric = asymmetric

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample['img_ref'] = F.adjust_saturation(sample['img_ref'], saturation)

            if self.asymmetric and np.random.random() < 0.2:
                saturation = np.random.uniform(0.8, 1.2)

            sample['img_tgt'] = F.adjust_saturation(sample['img_tgt'], saturation)

        return sample
