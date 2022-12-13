from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloader.flow.datasets import FlyingChairs, FlyingThings3D, MpiSintel, KITTI
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile, flow_to_image
import imageio

from utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from unimatch.geometry import forward_backward_consistency_check
from utils.file_io import extract_video


@torch.no_grad()
def create_sintel_submission(model,
                             output_path='sintel_submission',
                             padding_factor=8,
                             save_vis_flow=False,
                             no_save_flo=False,
                             attn_type='swin',
                             attn_splits_list=False,
                             corr_radius_list=False,
                             prop_radius_list=False,
                             num_reg_refine=1,
                             inference_size=None,
                             ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = MpiSintel(split='test', aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if inference_size is not None:
                assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
                ori_size = image1.shape[-2:]
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                       align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                       align_corners=True)
            else:
                padder = InputPadder(image1.shape, padding_factor=padding_factor)
                image1, image2 = padder.pad(image1, image2)

            results_dict = model(image1, image2,
                                 attn_type=attn_type,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 num_reg_refine=num_reg_refine,
                                 task='flow',
                                 )

            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

            # resize back
            if inference_size is not None:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()
            else:
                flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)

            if save_vis_flow:
                vis_flow_file = output_file.replace('.flo', '.png')
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='kitti_submission',
                            padding_factor=8,
                            save_vis_flow=False,
                            attn_type='swin',
                            attn_splits_list=False,
                            corr_radius_list=False,
                            prop_radius_list=False,
                            num_reg_refine=1,
                            inference_size=None,
                            ):
    """ Create submission for the KITTI leaderboard """
    model.eval()
    test_dataset = KITTI(split='testing', aug_params=None)

    print('%d test samples' % len(test_dataset))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]

        image1 = image1[None].cuda()  # [1, 3, H, W]
        image2 = image2[None].cuda()

        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)
        else:
            padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             )

        flow_pr = results_dict['flow_preds'][-1]

        # resize back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()
        else:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model,
                    with_speed_metric=False,
                    attn_type='swin',
                    attn_splits_list=None,
                    corr_radius_list=None,
                    prop_radius_list=None,
                    num_reg_refine=1,
                    ):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    if with_speed_metric:
        s0_10_list = []
        s10_40_list = []
        s40plus_list = []

    val_dataset = FlyingChairs(split='validation')

    print('Number of validation image pairs: %d' % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if with_speed_metric:
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_mask = (flow_gt_speed < 10)
            if valid_mask.max() > 0:
                s0_10_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
            if valid_mask.max() > 0:
                s10_40_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed > 40)
            if valid_mask.max() > 0:
                s40plus_list.append(epe[valid_mask].cpu().numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)
    px3 = np.mean(epe_all > 3)
    px5 = np.mean(epe_all > 5)
    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    results['chairs_epe'] = epe
    results['chairs_1px'] = px1
    results['chairs_3px'] = px3
    results['chairs_5px'] = px5

    if with_speed_metric:
        s0_10 = np.mean(np.concatenate(s0_10_list))
        s10_40 = np.mean(np.concatenate(s10_40_list))
        s40plus = np.mean(np.concatenate(s40plus_list))

        print("Validation Chairs s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
            s0_10,
            s10_40,
            s40plus))

        results['chairs_s0_10'] = s0_10
        results['chairs_s10_40'] = s10_40
        results['chairs_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_things(model,
                    padding_factor=8,
                    with_speed_metric=False,
                    max_val_flow=400,
                    val_things_clean_only=True,
                    attn_type='swin',
                    attn_splits_list=False,
                    corr_radius_list=False,
                    prop_radius_list=False,
                    num_reg_refine=1,
                    ):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        if val_things_clean_only:
            if dstype == 'frames_finalpass':
                continue

        val_dataset = FlyingThings3D(dstype=dstype, test_set=True, validate_subset=True,
                                     )
        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            results_dict = model(image1, image2,
                                 attn_type=attn_type,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 num_reg_refine=num_reg_refine,
                                 task='flow',
                                 )
            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

            if with_speed_metric:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_list = np.mean(np.concatenate(epe_list))

        epe = np.mean(epe_list)

        if dstype == 'frames_cleanpass':
            dstype = 'things_clean'
        if dstype == 'frames_finalpass':
            dstype = 'things_final'

        print("Validation Things test set (%s) EPE: %.3f" % (dstype, epe))
        results[dstype + '_epe'] = epe

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print("Validation Things test (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                dstype, s0_10,
                s10_40,
                s40plus))

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_sintel(model,
                    count_time=False,
                    padding_factor=8,
                    with_speed_metric=False,
                    evaluate_matched_unmatched=False,
                    attn_type='swin',
                    attn_splits_list=False,
                    corr_radius_list=False,
                    prop_radius_list=False,
                    num_reg_refine=1,
                    ):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    if count_time:
        total_time = 0
        num_runs = 100

    for dstype in ['clean', 'final']:
        val_dataset = MpiSintel(split='training', dstype=dstype,
                                load_occlusion=evaluate_matched_unmatched,
                                )

        print('Number of validation image pairs: %d' % len(val_dataset))
        epe_list = []

        if evaluate_matched_unmatched:
            matched_epe_list = []
            unmatched_epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            if evaluate_matched_unmatched:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(flow_gt.unsqueeze(0)).squeeze(0)  # [H, W]

            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            if count_time and val_id >= 5:  # 5 warmup
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            results_dict = model(image1, image2,
                                 attn_type=attn_type,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 num_reg_refine=num_reg_refine,
                                 task='flow',
                                 )

            flow_pr = results_dict['flow_preds'][-1]

            if count_time and val_id >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if val_id >= num_runs + 4:
                    break

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if evaluate_matched_unmatched:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

            if with_speed_metric:
                flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
                valid_mask = (flow_gt_speed < 10)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)

        dstype_ori = dstype

        print("Validation Sintel (%s) EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (dstype_ori, epe, px1, px3, px5))

        dstype = 'sintel_' + dstype

        results[dstype + '_epe'] = np.mean(epe_list)
        results[dstype + '_1px'] = px1
        results[dstype + '_3px'] = px3
        results[dstype + '_5px'] = px5

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print("Validation Sintel (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                dstype_ori, s0_10,
                s10_40,
                s40plus))

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

        if count_time:
            print('Time: %.6fs' % (total_time / num_runs))
            break  # only the clean pass when counting time

        if evaluate_matched_unmatched:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))

            print('Validatation Sintel (%s) matched epe: %.3f, unmatched epe: %.3f' % (
                dstype_ori, matched_epe, unmatched_epe))

            results[dstype + '_matched'] = matched_epe
            results[dstype + '_unmatched'] = unmatched_epe

    return results


@torch.no_grad()
def validate_kitti(model,
                   padding_factor=8,
                   with_speed_metric=False,
                   average_over_pixels=True,
                   attn_type='swin',
                   attn_splits_list=False,
                   corr_radius_list=False,
                   prop_radius_list=False,
                   num_reg_refine=1,
                   debug=False,
                   ):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = KITTI(split='training')
    print('Number of validation image pairs: %d' % len(val_dataset))

    out_list, epe_list = [], []
    results = {}

    if with_speed_metric:
        if average_over_pixels:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []
        else:
            s0_10_epe_sum = 0
            s0_10_valid_samples = 0
            s10_40_epe_sum = 0
            s10_40_valid_samples = 0
            s40plus_epe_sum = 0
            s40plus_valid_samples = 0

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             )

        # useful when using parallel branches
        flow_pr = results_dict['flow_preds'][-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        if with_speed_metric:
            # flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            flow_gt_speed = mag

            if average_over_pixels:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

            else:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s0_10_valid_samples += 1

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s10_40_valid_samples += 1

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s40plus_valid_samples += 1

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        if average_over_pixels:
            epe_list.append(epe[val].cpu().numpy())
        else:
            epe_list.append(epe[val].mean().item())

        out_list.append(out[val].cpu().numpy())

        if debug:
            if val_id > 10:
                break

    if average_over_pixels:
        epe_list = np.concatenate(epe_list)
    else:
        epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI EPE: %.3f, F1-all: %.3f" % (epe, f1))
    results['kitti_epe'] = epe
    results['kitti_f1'] = f1

    if with_speed_metric:
        if average_over_pixels:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))
        else:
            s0_10 = s0_10_epe_sum / s0_10_valid_samples
            s10_40 = s10_40_epe_sum / s10_40_valid_samples
            s40plus = s40plus_epe_sum / s40plus_valid_samples

        print("Validation KITTI s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
            s0_10,
            s10_40,
            s40plus))

        results['kitti_s0_10'] = s0_10
        results['kitti_s10_40'] = s10_40
        results['kitti_s40+'] = s40plus

    return results


@torch.no_grad()
def inference_flow(model,
                   inference_dir,
                   inference_video=None,
                   output_path='output',
                   padding_factor=8,
                   inference_size=None,
                   save_flo_flow=False,  # save raw flow prediction as .flo
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   save_video=False,
                   concat_flow_img=False,
                   ):
    """ Inference on a directory or a video """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if save_video:
        assert inference_video is not None

    fixed_inference_size = inference_size
    transpose_img = False

    if inference_video is not None:
        filenames, fps = extract_video(inference_video)  # list of [H, W, 3]
    else:
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    print('%d images found' % len(filenames))

    vis_flow_preds = []
    ori_imgs = []

    for test_id in range(0, len(filenames) - 1):
        if (test_id + 1) % 50 == 0:
            print('predicting %d/%d' % (test_id + 1, len(filenames)))

        if inference_video is not None:
            image1 = filenames[test_id]
            image2 = filenames[test_id + 1]
        else:
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        if concat_flow_img:
            ori_imgs.append(image1)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             pred_bidir_flow=pred_bidir_flow,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        if inference_video is not None:
            output_file = os.path.join(output_path, '%04d_flow.png' % test_id)
        else:
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow.png')

        if inference_video is not None and save_video:
            vis_flow_preds.append(flow_to_image(flow))
        else:
            # save vis flow
            save_vis_flow_tofile(flow, output_file)

        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]
            flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            if inference_video is not None:
                output_file = os.path.join(output_path, '%04d_flow_bwd.png' % test_id)
            else:
                output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow_bwd.png')

            # save vis flow
            save_vis_flow_tofile(flow_bwd, output_file)

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                fwd_occ, bwd_occ = forward_backward_consistency_check(flow_pr[:1], flow_pr[1:])  # [1, H, W] float

                if inference_video is not None:
                    fwd_occ_file = os.path.join(output_path, '%04d_occ_fwd.png' % test_id)
                    bwd_occ_file = os.path.join(output_path, '%04d_occ_bwd.png' % test_id)
                else:
                    fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_fwd.png')
                    bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_bwd.png')

                Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
                Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)

        if save_flo_flow:
            if inference_video is not None:
                output_file = os.path.join(output_path, '%04d_pred.flo' % test_id)
            else:
                output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred.flo')
            frame_utils.writeFlow(output_file, flow)
            if pred_bidir_flow:
                if inference_video is not None:
                    output_file_bwd = os.path.join(output_path, '%04d_pred_bwd.flo' % test_id)
                else:
                    output_file_bwd = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred_bwd.flo')
                frame_utils.writeFlow(output_file_bwd, flow_bwd)

    if save_video:
        suffix = '_flow_img.mp4' if concat_flow_img else '_flow.mp4'
        output_file = os.path.join(output_path, os.path.basename(inference_video)[:-4] + suffix)

        if concat_flow_img:
            results = []
            assert len(ori_imgs) == len(vis_flow_preds)

            concat_axis = 0 if ori_imgs[0].shape[0] < ori_imgs[0].shape[1] else 1
            for img, flow in zip(ori_imgs, vis_flow_preds):
                concat = np.concatenate((img, flow), axis=concat_axis)
                results.append(concat)
        else:
            results = vis_flow_preds

        imageio.mimwrite(output_file, results, fps=fps, quality=8)

    print('Done!')
