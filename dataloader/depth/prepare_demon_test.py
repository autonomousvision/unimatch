import os
import sys
import subprocess

from joblib import Parallel, delayed
import numpy as np
import imageio

imageio.plugins.freeimage.download()
from imageio.plugins import freeimage
import h5py
from lz4.block import decompress
import scipy.misc

import cv2

from path import Path

path = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def dump_example(dataset_name):
    print("Converting {:}.h5 ...".format(dataset_name))
    file = h5py.File(os.path.join(path, "testdata", "{:}.h5".format(dataset_name)), "r")

    for (seq_idx, seq_name) in enumerate(file):
        if dataset_name == 'scenes11_test':
            scale = 0.4
        else:
            scale = 1

        print("Processing sequence {:d}/{:d}".format(seq_idx, len(file)))
        dump_dir = os.path.join(path, 'test', dataset_name + "_" + "{:05d}".format(seq_idx))
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        dump_dir = Path(dump_dir)
        sequence = file[seq_name]["frames"]["t0"]
        poses = []
        for (f_idx, f_name) in enumerate(sequence):
            frame = sequence[f_name]
            for dt_type in frame:
                dataset = frame[dt_type]
                img = dataset[...]
                if dt_type == "camera":
                    if f_idx == 0:
                        intrinsics = np.array([[img[0], 0, img[3]], [0, img[1], img[4]], [0, 0, 1]])
                    pose = np.array(
                        [[img[5], img[8], img[11], img[14] * scale], [img[6], img[9], img[12], img[15] * scale],
                         [img[7], img[10], img[13], img[16] * scale]])
                    poses.append(pose.tolist())
                elif dt_type == "depth":
                    dimension = dataset.attrs["extents"]
                    depth = np.array(np.frombuffer(decompress(img.tobytes(), dimension[0] * dimension[1] * 2),
                                                   dtype=np.float16)).astype(np.float32)
                    depth = depth.reshape(dimension[0], dimension[1]) * scale

                    dump_depth_file = dump_dir / '{:04d}.npy'.format(f_idx)
                    np.save(dump_depth_file, depth)
                elif dt_type == "image":
                    img = imageio.imread(img.tobytes())
                    dump_img_file = dump_dir / '{:04d}.jpg'.format(f_idx)
                    imageio.imsave(dump_img_file, img)

        dump_cam_file = dump_dir / 'cam.txt'
        np.savetxt(dump_cam_file, intrinsics)
        poses_file = dump_dir / 'poses.txt'
        np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

        if len(dump_dir.files('*.jpg')) < 2:
            dump_dir.rmtree()


def preparedata():
    num_threads = 1
    SUB_DATASET_NAMES = (["rgbd_test", "scenes11_test", "sun3d_test"])

    dump_root = os.path.join(path, 'test')
    if not os.path.isdir(dump_root):
        os.mkdir(dump_root)

    if num_threads == 1:
        for scene in SUB_DATASET_NAMES:
            dump_example(scene)
    else:
        Parallel(n_jobs=num_threads)(delayed(dump_example)(scene) for scene in SUB_DATASET_NAMES)

    dump_root = Path(dump_root)
    subdirs = dump_root.dirs()
    subdirs = [subdir.basename() for subdir in subdirs]
    subdirs = sorted(subdirs)
    with open(dump_root / 'test.txt', 'w') as tf:
        for subdir in subdirs:
            tf.write('{}\n'.format(subdir))

    print("Finished Converting Data.")


if __name__ == "__main__":
    preparedata()
