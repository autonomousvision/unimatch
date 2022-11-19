import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import os
import skimage.io
import cv2
from PIL import Image
from glob import glob

from loss.stereo_metric import d1_metric, thres_metric
from dataloader.stereo.datasets import (FlyingThings3D, KITTI15,
                                        ETH3DStereo, MiddleburyEval3)
from dataloader.stereo import transforms
from utils.utils import InputPadder

from utils.file_io import write_pfm
from utils.visualization import vis_disparity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='output',
                            padding_factor=16,
                            attn_splits_list=False,
                            corr_radius_list=False,
                            prop_radius_list=False,
                            attn_type=None,
                            num_reg_refine=1,
                            inference_size=None,
                            ):
    """ create submission for the KITTI leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = KITTI15(mode='testing', transform=val_transform)

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in enumerate(test_dataset):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        pred_disp = model(left, right,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          num_reg_refine=num_reg_refine,
                          task='stereo',
                          )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, left_name)

        skimage.io.imsave(save_name, (pred_disp.cpu().numpy() * 256.).astype(np.uint16))


@torch.no_grad()
def create_eth3d_submission(model,
                            output_path='output',
                            padding_factor=16,
                            attn_type=None,
                            attn_splits_list=False,
                            corr_radius_list=False,
                            prop_radius_list=False,
                            num_reg_refine=1,
                            inference_size=None,
                            submission_mode='train',
                            save_vis_disp=False,
                            ):
    """ create submission for the eth3d stereo leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = ETH3DStereo(mode=submission_mode,
                               transform=val_transform,
                               save_filename=True
                               )

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fixed_inference_size = inference_size

    for i, sample in enumerate(test_dataset):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        nearest_size = [int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(left.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = left.shape[-2:]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(left, right,
                      attn_type=attn_type,
                      attn_splits_list=attn_splits_list,
                      corr_radius_list=corr_radius_list,
                      prop_radius_list=prop_radius_list,
                      num_reg_refine=num_reg_refine,
                      task='stereo',
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(left, right,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          num_reg_refine=num_reg_refine,
                          task='stereo',
                          )['flow_preds'][-1]  # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))

        if save_vis_disp:
            save_name = os.path.join(output_path, filename + '.png')
            disp = vis_disparity(pred_disp.cpu().numpy())
            cv2.imwrite(save_name, disp)
        else:
            save_disp_name = os.path.join(output_path, filename + '.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(output_path, filename + '.txt')
            with open(save_runtime_name, 'w') as f:
                f.write('runtime ' + str(inference_time))


@torch.no_grad()
def create_middlebury_submission(model,
                                 output_path='output',
                                 padding_factor=16,
                                 attn_type=None,
                                 attn_splits_list=False,
                                 corr_radius_list=False,
                                 prop_radius_list=False,
                                 num_reg_refine=1,
                                 inference_size=None,
                                 submission_mode='train',
                                 save_vis_disp=False,
                                 ):
    """ create submission for the Middlebury leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = MiddleburyEval3(mode=submission_mode,
                                   resolution='F',
                                   transform=val_transform,
                                   save_filename=True,
                                   )

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in enumerate(test_dataset):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(left, right,
                      attn_type=attn_type,
                      attn_splits_list=attn_splits_list,
                      corr_radius_list=corr_radius_list,
                      prop_radius_list=prop_radius_list,
                      num_reg_refine=num_reg_refine,
                      task='stereo',
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(left, right,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          task='stereo',
                          )['flow_preds'][-1]  # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))  # works for both windows and linux

        if save_vis_disp:
            save_name = os.path.join(output_path, filename + '.png')
            disp = vis_disparity(pred_disp.cpu().numpy())
            cv2.imwrite(save_name, disp)
        else:
            save_disp_dir = os.path.join(output_path, filename)
            os.makedirs(save_disp_dir, exist_ok=True)

            save_disp_name = os.path.join(save_disp_dir, 'disp0GMStereo.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(save_disp_dir, 'timeGMStereo.txt')
            with open(save_runtime_name, 'w') as f:
                f.write(str(inference_time))


@torch.no_grad()
def validate_things(model,
                    max_disp=400,
                    padding_factor=16,
                    inference_size=None,
                    attn_type=None,
                    num_iters_per_scale=None,
                    attn_splits_list=None,
                    corr_radius_list=None,
                    prop_radius_list=None,
                    num_reg_refine=1,
                    ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = FlyingThings3D(mode='TEST', transform=val_transform)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 1000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              num_iters_per_scale=num_iters_per_scale,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)

        val_epe += epe.item()
        val_d1 += d1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples

    print('Validation things EPE: %.3f, D1: %.4f' % (
        mean_epe, mean_d1))

    results['things_epe'] = mean_epe
    results['things_d1'] = mean_d1

    return results


@torch.no_grad()
def validate_kitti15(model,
                     padding_factor=16,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     num_reg_refine=1,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = KITTI15(transform=val_transform,
                          )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples
    mean_thres3 = val_thres3 / valid_samples

    print('Validation KITTI15 EPE: %.3f, D1: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres3))

    results['kitti15_epe'] = mean_epe
    results['kitti15_d1'] = mean_d1
    results['kitti15_3px'] = mean_thres3

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_eth3d(model,
                   padding_factor=16,
                   inference_size=None,
                   attn_type=None,
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = ETH3DStereo(transform=val_transform,
                              )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_thres1 = val_thres1 / valid_samples

    print('Validation ETH3D EPE: %.3f, 1px: %.4f' % (
        mean_epe, mean_thres1))

    results['eth3d_epe'] = mean_epe
    results['eth3d_1px'] = mean_thres1

    return results


@torch.no_grad()
def validate_middlebury(model,
                        padding_factor=16,
                        inference_size=None,
                        attn_type=None,
                        attn_splits_list=None,
                        corr_radius_list=None,
                        prop_radius_list=None,
                        num_reg_refine=1,
                        resolution='H',
                        ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MiddleburyEval3(transform=val_transform,
                                  resolution=resolution,
                                  )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres2 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres2 += thres2.item()

    mean_epe = val_epe / valid_samples
    mean_thres2 = val_thres2 / valid_samples

    print('Validation Middlebury EPE: %.3f, 2px: %.4f' % (
        mean_epe, mean_thres2))

    results['middlebury_epe'] = mean_epe
    results['middlebury_2px'] = mean_thres2

    return results


@torch.no_grad()
def inference_stereo(model,
                     inference_dir=None,
                     inference_dir_left=None,
                     inference_dir_right=None,
                     output_path='output',
                     padding_factor=16,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     num_reg_refine=1,
                     pred_bidir_disp=False,
                     pred_right_disp=False,
                     save_pfm_disp=False,
                     ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    assert inference_dir or (inference_dir_left and inference_dir_right)

    if inference_dir is not None:
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))

        left_filenames = filenames[::2]
        right_filenames = filenames[1::2]

    else:
        left_filenames = sorted(glob(inference_dir_left + '/*.png') + glob(inference_dir_left + '/*.jpg'))
        right_filenames = sorted(glob(inference_dir_right + '/*.png') + glob(inference_dir_right + '/*.jpg'))

    assert len(left_filenames) == len(right_filenames)

    num_samples = len(left_filenames)
    print('%d test samples found' % num_samples)

    fixed_inference_size = inference_size

    for i in range(num_samples):

        if (i + 1) % 50 == 0:
            print('predicting %d/%d' % (i + 1, num_samples))

        left_name = left_filenames[i]
        right_name = right_filenames[i]

        left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)
        sample = {'left': left, 'right': right}

        sample = val_transform(sample)

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

        nearest_size = [int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(left.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = left.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        with torch.no_grad():
            if pred_bidir_disp:
                new_left, new_right = hflip(right), hflip(left)
                left = torch.cat((left, new_left), dim=0)
                right = torch.cat((right, new_right), dim=0)

            if pred_right_disp:
                left, right = hflip(right), hflip(left)

            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)  # [1, H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, os.path.basename(left_name)[:-4] + '_disp.png')

        if pred_right_disp:
            pred_disp = hflip(pred_disp)

        disp = pred_disp[0].cpu().numpy()

        if save_pfm_disp:
            save_name_pfm = save_name[:-4] + '.pfm'
            write_pfm(save_name_pfm, disp)

        disp = vis_disparity(disp)
        cv2.imwrite(save_name, disp)

        if pred_bidir_disp:
            assert pred_disp.size(0) == 2  # [2, H, W]
            save_name = os.path.join(output_path, os.path.basename(left_name)[:-4] + '_disp_right.png')

            # flip back
            disp = hflip(pred_disp[1]).cpu().numpy()

            if save_pfm_disp:
                save_name_pfm = save_name[:-4] + '.pfm'
                write_pfm(save_name_pfm, disp)

            disp = vis_disparity(disp)
            cv2.imwrite(save_name, disp)

    print('Done!')
