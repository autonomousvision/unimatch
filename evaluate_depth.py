import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from glob import glob

from dataloader.depth import augmentation as transforms
from dataloader.depth.datasets import ScannetDataset, DemonDataset
from loss.depth_loss import compute_errors
from utils.utils import InputPadder
from utils.visualization import viz_depth_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def validate_scannet(model,
                     padding_factor=16,
                     inference_size=None,
                     attn_type='swin',
                     attn_splits_list=None,
                     prop_radius_list=None,
                     num_reg_refine=1,
                     num_depth_candidates=64,
                     count_time=False,
                     eval_min_depth=0.5,
                     eval_max_depth=10,
                     min_depth=0.5,
                     max_depth=10,
                     save_vis_depth=False,
                     save_dir=None,
                     ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = ScannetDataset(transforms=val_transform,
                                 mode='test',
                                 )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    error_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    error_sum = [0.] * len(error_names)

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    if save_vis_depth:
        assert save_dir is not None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    for i, sample in enumerate(val_dataset):
        if i % 500 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        img_ref = sample['img_ref'].to(device).unsqueeze(0)  # [1, 3, H, W]
        img_tgt = sample['img_tgt'].to(device).unsqueeze(0)  # [1, 3, H, W]
        intrinsics = sample['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]
        pose = sample['pose'].to(device).unsqueeze(0)  # [1, 4, 4]
        gt_depth = sample['depth'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(img_ref.shape, padding_factor=padding_factor, mode='kitti')
            img_ref, img_tgt = padder.pad(img_ref, img_tgt)
        else:
            ori_size = img_ref.shape[-2:]
            img_ref = F.interpolate(img_ref, size=inference_size, mode='bilinear',
                                    align_corners=True)
            img_tgt = F.interpolate(img_tgt, size=inference_size, mode='bilinear',
                                    align_corners=True)

        mask = (gt_depth > eval_min_depth) & (gt_depth < eval_max_depth)

        # only evaluate on valid gt data
        mask = mask & (sample['valid'].to(device) > 0.5)

        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_depth = model(img_ref, img_tgt,
                               attn_type=attn_type,
                               attn_splits_list=attn_splits_list,
                               prop_radius_list=prop_radius_list,
                               num_reg_refine=num_reg_refine,
                               intrinsics=intrinsics,
                               pose=pose,
                               min_depth=1. / max_depth,
                               max_depth=1. / min_depth,
                               num_depth_candidates=num_depth_candidates,
                               task='depth',
                               )['flow_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_depth = padder.unpad(pred_depth)[0]  # [H, W]
        else:
            # resize back
            pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                       align_corners=True).squeeze(1)[0]  # [H, W]

            # NOTE: no scale depth magnitude when resize

        if save_vis_depth:
            filename = os.path.join(save_dir, '%04d_depth_pred.png' % valid_samples)
            viz_inv_depth = viz_depth_tensor(1. / pred_depth.cpu(),
                                             return_numpy=True,
                                             colormap='plasma')  # [H, W, 3] uint8
            Image.fromarray(viz_inv_depth).save(filename)

        gt_depth = gt_depth.cpu().numpy()
        pred_depth = pred_depth.cpu().numpy()
        mask = mask.cpu().numpy()

        metrics = list(compute_errors(gt_depth[mask], pred_depth[mask]))

        error_sum = [error_sum[i] + metrics[i] for i in range(len(error_sum))]

    error_mean = [error / num_samples for error in error_sum]
    results = dict(zip(error_names, error_mean))

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_demon(model,
                   padding_factor=16,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   num_depth_candidates=64,
                   count_time=False,
                   eval_min_depth=0.5,
                   eval_max_depth=10,
                   min_depth=0.5,
                   max_depth=10,
                   save_vis_depth=False,
                   save_dir=None,
                   demon_split='rgbd',
                   debug=False,
                   ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = DemonDataset(transforms=val_transform,
                               mode=demon_split + '_test',
                               )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    error_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    error_sum = [0.] * len(error_names)

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    if save_vis_depth:
        assert save_dir is not None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 500 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        img_ref = sample['img_ref'].to(device).unsqueeze(0)  # [1, 3, H, W]
        img_tgt = sample['img_tgt'].to(device).unsqueeze(0)  # [1, 3, H, W]
        intrinsics = sample['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]
        pose = sample['pose'].to(device).unsqueeze(0)  # [1, 4, 4]
        gt_depth = sample['depth'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(img_ref.shape, padding_factor=padding_factor, mode='kitti')
            img_ref, img_tgt = padder.pad(img_ref, img_tgt)
        else:
            ori_size = img_ref.shape[-2:]
            img_ref = F.interpolate(img_ref, size=inference_size, mode='bilinear',
                                    align_corners=True)
            img_tgt = F.interpolate(img_tgt, size=inference_size, mode='bilinear',
                                    align_corners=True)

        mask = (gt_depth > eval_min_depth) & (gt_depth < eval_max_depth)

        # only evaluate on valid gt data
        mask = mask & (sample['valid'].to(device) > 0.5)

        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_depth = model(img_ref, img_tgt,
                               attn_type=attn_type,
                               attn_splits_list=attn_splits_list,
                               prop_radius_list=prop_radius_list,
                               num_reg_refine=num_reg_refine,
                               intrinsics=intrinsics,
                               pose=pose,
                               min_depth=1. / max_depth,
                               max_depth=1. / min_depth,
                               num_depth_candidates=num_depth_candidates,
                               task='depth',
                               )['flow_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_depth = padder.unpad(pred_depth)[0]  # [H, W]
        else:
            # resize back
            pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                       align_corners=True).squeeze(1)[0]  # [H, W]

        if save_vis_depth:
            filename = os.path.join(save_dir, '%04d.png' % valid_samples)
            viz_inv_depth = viz_depth_tensor(1. / pred_depth.cpu(),
                                             return_numpy=True,
                                             colormap='plasma')  # [H, W, 3] uint8
            Image.fromarray(viz_inv_depth).save(filename)

        gt_depth = gt_depth.cpu().numpy()
        pred_depth = pred_depth.cpu().numpy()
        mask = mask.cpu().numpy()

        metrics = list(compute_errors(gt_depth[mask], pred_depth[mask]))

        error_sum = [error_sum[i] + metrics[i] for i in range(len(error_sum))]

    error_mean = [error / num_samples for error in error_sum]
    results = dict(zip(error_names, error_mean))

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def inference_depth(model,
                    inference_dir=None,
                    output_path='output',
                    padding_factor=16,
                    inference_size=None,
                    attn_type='swin',
                    attn_splits_list=None,
                    prop_radius_list=None,
                    num_reg_refine=1,
                    num_depth_candidates=64,
                    min_depth=0.5,
                    max_depth=10,
                    depth_from_argmax=False,
                    pred_bidir_depth=False,
                    ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    valid_samples = 0

    fixed_inference_size = inference_size

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # assume scannet dataset file structure
    imgs = sorted(glob(os.path.join(inference_dir, 'color', '*.jpg')) +
                  glob(os.path.join(inference_dir, 'color', '*.png')))
    poses = sorted(glob(os.path.join(inference_dir, 'pose', '*.txt')))

    intrinsics_file = glob(os.path.join(inference_dir, 'intrinsic', '*.txt'))[0]

    assert len(imgs) == len(poses)

    num_samples = len(imgs)

    for i in range(len(imgs) - 1):
        if i % 50 == 0:
            print('=> Predicting %d/%d' % (i, num_samples))

        img_ref = np.array(Image.open(imgs[i]).convert('RGB')).astype(np.float32)
        img_tgt = np.array(Image.open(imgs[i + 1]).convert('RGB')).astype(np.float32)

        intrinsics = np.loadtxt(intrinsics_file).astype(np.float32).reshape((4, 4))[:3, :3]  # [3, 3]

        pose_ref = np.loadtxt(poses[i], delimiter=' ').astype(np.float32).reshape((4, 4))
        pose_tgt = np.loadtxt(poses[i + 1], delimiter=' ').astype(np.float32).reshape((4, 4))
        # relative pose
        pose = np.linalg.inv(pose_tgt) @ pose_ref

        sample = {'img_ref': img_ref,
                  'img_tgt': img_tgt,
                  'intrinsics': intrinsics,
                  'pose': pose,
                  }
        sample = val_transform(sample)

        img_ref = sample['img_ref'].to(device).unsqueeze(0)  # [1, 3, H, W]
        img_tgt = sample['img_tgt'].to(device).unsqueeze(0)  # [1, 3, H, W]
        intrinsics = sample['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]
        pose = sample['pose'].to(device).unsqueeze(0)  # [1, 4, 4]

        nearest_size = [int(np.ceil(img_ref.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(img_ref.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = img_ref.shape[-2:]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            img_ref = F.interpolate(img_ref, size=inference_size, mode='bilinear',
                                    align_corners=True)
            img_tgt = F.interpolate(img_tgt, size=inference_size, mode='bilinear',
                                    align_corners=True)

        valid_samples += 1

        with torch.no_grad():
            pred_depth = model(img_ref, img_tgt,
                               attn_type=attn_type,
                               attn_splits_list=attn_splits_list,
                               prop_radius_list=prop_radius_list,
                               num_reg_refine=num_reg_refine,
                               intrinsics=intrinsics,
                               pose=pose,
                               min_depth=1. / max_depth,
                               max_depth=1. / min_depth,
                               num_depth_candidates=num_depth_candidates,
                               pred_bidir_depth=pred_bidir_depth,
                               depth_from_argmax=depth_from_argmax,
                               task='depth',
                               )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # resize back
            pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                       align_corners=True).squeeze(1)  # [1, H, W]

        pr_depth = pred_depth[0]

        filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '.png')
        viz_inv_depth = viz_depth_tensor(1. / pr_depth.cpu(),
                                         return_numpy=True)  # [H, W, 3] uint8
        Image.fromarray(viz_inv_depth).save(filename)

        if pred_bidir_depth:
            assert pred_depth.size(0) == 2

            pr_depth_bwd = pred_depth[1]

            filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '_bwd.png')
            viz_inv_depth = viz_depth_tensor(1. / pr_depth_bwd.cpu(),
                                             return_numpy=True)  # [H, W, 3] uint8
            Image.fromarray(viz_inv_depth).save(filename)

    print('Done!')
