import numpy as np
from torch.utils.data import Dataset

import os
from glob import glob
from PIL import Image

from utils.file_io import read_img


class ScannetDataset(Dataset):
    def __init__(self,
                 data_dir='datasets/Scannet',
                 mode='train',
                 transforms=None,
                 return_filename=False,
                 ):

        self.data_dir = data_dir
        self.transforms = transforms
        self.return_filename = return_filename

        assert mode in ['train', 'test', 'demo']

        self.mode = mode

        self.samples = []

        # following BA-Net's splits
        dir_path = os.path.dirname(os.path.realpath(__file__))
        split_file = 'scannet_banet_' + mode + '_pairs.txt'

        split_file = os.path.join(dir_path, split_file)

        with open(split_file) as f:
            pairs = f.readlines()

        pairs = [pair.rstrip() for pair in pairs]

        for i in range(len(pairs)):
            scene, img_ref_id, img_tgt_id = pairs[i].split(' ')
            key = 'scannet_' + mode + '_' + scene + '_' + img_ref_id

            scene = os.path.join(data_dir, mode, scene)

            intrinsics = os.path.join(scene, 'intrinsic', 'intrinsic_depth.txt')

            img_ref = os.path.join(scene, 'color', img_ref_id + '.jpg')
            img_tgt = os.path.join(scene, 'color', img_tgt_id + '.jpg')

            depth = os.path.join(scene, 'depth', img_ref_id + '.png')
            pose_ref = os.path.join(scene, 'pose', img_ref_id + '.txt')
            pose_tgt = os.path.join(scene, 'pose', img_tgt_id + '.txt')

            if not os.path.isfile(img_ref) or not os.path.isfile(img_tgt) or not os.path.isfile(depth) or \
                    not os.path.isfile(pose_ref) or not os.path.isfile(pose_tgt):
                continue

            sample = (img_ref, img_tgt, pose_ref, pose_tgt, depth, intrinsics, key)

            self.samples.append(sample)

    def __getitem__(self, i):
        img_ref, img_tgt, pose_ref, pose_tgt, depth, intrinsics, key = self.samples[i]

        img_ref_filename, img_tgt_filename = img_ref, img_tgt

        img_ref = self._read_image(img_ref)
        img_tgt = self._read_image(img_tgt)
        depth = self._read_depth(depth)
        valid = (depth > 0).astype(np.float32)  # invalid depth is 0

        # pose: camera to world
        pose_ref = np.loadtxt(pose_ref, delimiter=' ').astype(np.float32).reshape((4, 4))
        pose_tgt = np.loadtxt(pose_tgt, delimiter=' ').astype(np.float32).reshape((4, 4))

        # relative pose
        pose = np.linalg.inv(pose_tgt) @ pose_ref

        intrinsics = np.loadtxt(intrinsics).astype(np.float32).reshape((4, 4))[:3, :3]  # [3, 3]

        sample = {
            'img_ref': img_ref,
            'img_tgt': img_tgt,
            'intrinsics': intrinsics,
            'pose': pose,
            'depth': depth,
            'valid': valid,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_filename:
            return img_ref_filename, img_tgt_filename, sample

        return sample

    def __len__(self):

        return len(self.samples)

    def _read_image(self, filename):
        img = Image.open(filename).resize((640, 480))  # resize to depth shape
        img = np.array(img).astype(np.float32)

        return img

    def _read_depth(self, filename):
        depth = np.array(Image.open(filename)).astype(np.float32) / 1000.

        return depth

    def __rmul__(self, v):
        self.samples = v * self.samples

        return self


class DemonDataset(Dataset):
    def __init__(self,
                 data_dir='datasets/Demon',
                 mode='train',
                 transforms=None,
                 sequence_length=2,
                 ):

        if 'test' in mode:
            data_dir = os.path.join(data_dir, 'test')
        else:
            data_dir = os.path.join(data_dir, 'train')

        self.data_dir = data_dir
        self.transforms = transforms

        assert sequence_length == 2  # only support two input views currently

        self.samples = []

        scenes = [os.path.join(data_dir, scene_dir) for scene_dir in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(os.path.join(data_dir, scene_dir))) and mode in scene_dir]

        demi_length = sequence_length // 2

        for scene in scenes:
            intrinsics = np.genfromtxt(os.path.join(scene, 'cam.txt')).astype(np.float32).reshape((3, 3))  # [3, 3]
            poses = np.genfromtxt(os.path.join(scene, 'poses.txt')).astype(np.float32)
            imgs = sorted(glob(os.path.join(scene, '*.jpg')))
            if len(imgs) < sequence_length:
                continue
            for i in range(len(imgs)):
                if i < demi_length:
                    shifts = list(range(0, sequence_length))
                    shifts.pop(i)
                elif i >= len(imgs) - demi_length:
                    shifts = list(range(len(imgs) - sequence_length, len(imgs)))
                    shifts.pop(i - len(imgs))
                else:
                    shifts = list(range(i - demi_length, i + (sequence_length + 1) // 2))
                    shifts.pop(demi_length)

                img_ref = imgs[i]
                depth = os.path.join(os.path.dirname(img_ref), os.path.basename(img_ref)[:-4] + '.npy')
                pose_ref = np.concatenate((poses[i, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)  # [4, 4]

                assert len(shifts) < 2  # only support two input images currently

                for j in shifts:
                    img_tgt = imgs[j]
                    pose_tgt = np.concatenate((poses[j, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                    pose = (pose_tgt @ np.linalg.inv(pose_ref)).astype(np.float32)  # [4, 4]

                    sample = (img_ref, img_tgt, pose, depth, intrinsics)

                    self.samples.append(sample)

    def __getitem__(self, i):
        img_ref, img_tgt, pose, depth, intrinsics = self.samples[i]

        img_ref = read_img(img_ref)
        img_tgt = read_img(img_tgt)
        depth = np.load(depth)
        valid = (depth > 0).astype(np.float32)  # invalid depth is 0

        sample = {
            'img_ref': img_ref,
            'img_tgt': img_tgt,
            'intrinsics': intrinsics,
            'pose': pose,
            'depth': depth,
            'valid': valid,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):

        return len(self.samples)
