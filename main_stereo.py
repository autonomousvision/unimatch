import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataloader.stereo.datasets import build_dataset
from utils import misc

from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.visualization import disp_error_img, save_images
from loss.stereo_metric import d1_metric
from evaluate_stereo import (validate_things, validate_kitti15, validate_eth3d,
                             validate_middlebury, create_kitti_submission,
                             create_eth3d_submission,
                             create_middlebury_submission,
                             inference_stereo,
                             )
from unimatch.unimatch import UniMatch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='sceneflow', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['kitti15'], type=str, nargs='+')
    parser.add_argument('--max_disp', default=400, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=288, type=int)
    parser.add_argument('--img_width', default=512, type=int)
    parser.add_argument('--padding_factor', default=16, type=int)

    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=326, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')
    parser.add_argument('--resume_exclude_upsampler', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='stereo', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='F', choices=['Q', 'H', 'F'])

    # submission
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--eth_submission_mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--middlebury_submission_mode', default='training', type=str, choices=['training', 'test'])
    parser.add_argument('--output_path', default='output', type=str)

    # log
    parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=100000, type=int)

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_bidir_disp', action='store_true',
                        help='predict both left and right disparities')
    parser.add_argument('--pred_right_disp', action='store_true',
                        help='predict right disparity')
    parser.add_argument('--save_pfm_disp', action='store_true',
                        help='save predicted disparity as .pfm format')

    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):
    print_info = not args.eval and not args.submission and args.inference_dir is None and \
                 args.inference_dir_left is None and args.inference_dir_right is None

    if print_info and args.local_rank == 0:
        print(args)

        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.output_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    if print_info:
        print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    if print_info:
        print('=> Number of trainable parameters: %d' % num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0

    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)

        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']

        if print_info:
            print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    if args.submission:
        if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
            create_kitti_submission(model_without_ddp,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    inference_size=args.inference_size,
                                    )

        if 'eth3d' in args.val_dataset:
            create_eth3d_submission(model_without_ddp,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    inference_size=args.inference_size,
                                    submission_mode=args.eth_submission_mode,
                                    save_vis_disp=args.save_vis_disp,
                                    )

        if 'middlebury' in args.val_dataset:
            create_middlebury_submission(model_without_ddp,
                                         output_path=args.output_path,
                                         padding_factor=args.padding_factor,
                                         attn_type=args.attn_type,
                                         attn_splits_list=args.attn_splits_list,
                                         corr_radius_list=args.corr_radius_list,
                                         prop_radius_list=args.prop_radius_list,
                                         num_reg_refine=args.num_reg_refine,
                                         inference_size=args.inference_size,
                                         submission_mode=args.middlebury_submission_mode,
                                         save_vis_disp=args.save_vis_disp,
                                         )

        return

    if args.eval:
        val_results = {}

        if 'things' in args.val_dataset:
            results_dict = validate_things(model_without_ddp,
                                           max_disp=args.max_disp,
                                           padding_factor=args.padding_factor,
                                           inference_size=args.inference_size,
                                           attn_type=args.attn_type,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           num_reg_refine=args.num_reg_refine,
                                           )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
            results_dict = validate_kitti15(model_without_ddp,
                                            padding_factor=args.padding_factor,
                                            inference_size=args.inference_size,
                                            attn_type=args.attn_type,
                                            attn_splits_list=args.attn_splits_list,
                                            corr_radius_list=args.corr_radius_list,
                                            prop_radius_list=args.prop_radius_list,
                                            num_reg_refine=args.num_reg_refine,
                                            count_time=args.count_time,
                                            debug=args.debug,
                                            )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'eth3d' in args.val_dataset:
            results_dict = validate_eth3d(model_without_ddp,
                                          padding_factor=args.padding_factor,
                                          inference_size=args.inference_size,
                                          attn_type=args.attn_type,
                                          attn_splits_list=args.attn_splits_list,
                                          corr_radius_list=args.corr_radius_list,
                                          prop_radius_list=args.prop_radius_list,
                                          num_reg_refine=args.num_reg_refine,
                                          )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'middlebury' in args.val_dataset:
            results_dict = validate_middlebury(model_without_ddp,
                                               padding_factor=args.padding_factor,
                                               inference_size=args.inference_size,
                                               attn_type=args.attn_type,
                                               attn_splits_list=args.attn_splits_list,
                                               corr_radius_list=args.corr_radius_list,
                                               prop_radius_list=args.prop_radius_list,
                                               num_reg_refine=args.num_reg_refine,
                                               resolution=args.middlebury_resolution,
                                               )

            if args.local_rank == 0:
                val_results.update(results_dict)

        return

    if args.inference_dir or (args.inference_dir_left and args.inference_dir_right):
        inference_stereo(model_without_ddp,
                         inference_dir=args.inference_dir,
                         inference_dir_left=args.inference_dir_left,
                         inference_dir_right=args.inference_dir_right,
                         output_path=args.output_path,
                         padding_factor=args.padding_factor,
                         inference_size=args.inference_size,
                         attn_type=args.attn_type,
                         attn_splits_list=args.attn_splits_list,
                         corr_radius_list=args.corr_radius_list,
                         prop_radius_list=args.prop_radius_list,
                         num_reg_refine=args.num_reg_refine,
                         pred_bidir_disp=args.pred_bidir_disp,
                         pred_right_disp=args.pred_right_disp,
                         save_pfm_disp=args.save_pfm_disp,
                         )

        return

    train_data = build_dataset(args)

    print('=> {} training samples found in the training set'.format(len(train_data)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler,
                              )

    last_epoch = start_step if args.resume and not args.no_resume_optimizer else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)

    total_steps = start_step
    epoch = start_epoch
    print('=> Start training...')

    while total_steps < args.num_steps:
        model.train()

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.local_rank == 0:
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], total_steps + 1)

        for i, sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]

            mask = (gt_disp > 0) & (gt_disp < args.max_disp)

            if not mask.any():
                continue

            pred_disps = model(left, right,
                               attn_type=args.attn_type,
                               attn_splits_list=args.attn_splits_list,
                               corr_radius_list=args.corr_radius_list,
                               prop_radius_list=args.prop_radius_list,
                               num_reg_refine=args.num_reg_refine,
                               task='stereo',
                               )['flow_preds']

            disp_loss = 0
            all_loss = []

            # loss weights
            loss_weights = [0.9 ** (len(pred_disps) - 1 - power) for power in
                            range(len(pred_disps))]

            for k in range(len(pred_disps)):
                pred_disp = pred_disps[k]
                weight = loss_weights[k]

                curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                             reduction='mean')
                disp_loss += weight * curr_loss
                all_loss.append(curr_loss)

            total_loss = disp_loss

            # more efficient zero_grad
            for param in model.parameters():
                param.grad = None

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            lr_scheduler.step()

            total_steps += 1

            if total_steps % args.summary_freq == 0 and args.local_rank == 0:
                img_summary = dict()
                img_summary['left'] = left
                img_summary['right'] = right
                img_summary['gt_disp'] = gt_disp

                img_summary['pred_disp'] = pred_disps[-1]

                pred_disp = pred_disps[-1]

                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)

                save_images(summary_writer, 'train', img_summary, total_steps)

                epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                print('step: %06d \t epe: %.3f' % (total_steps, epe.item()))

                summary_writer.add_scalar('train/epe', epe.item(), total_steps)
                summary_writer.add_scalar('train/disp_loss', disp_loss.item(), total_steps)
                summary_writer.add_scalar('train/total_loss', total_loss.item(), total_steps)

                # save all losses
                for s in range(len(all_loss)):
                    save_name = 'train/loss' + str(len(all_loss) - s - 1)
                    save_value = all_loss[s]
                    summary_writer.add_scalar(save_name, save_value, total_steps)

                d1 = d1_metric(pred_disp, gt_disp, mask)
                summary_writer.add_scalar('train/d1', d1.item(), total_steps)

            # always save the latest model for resuming training
            if args.local_rank == 0 and total_steps % args.save_latest_ckpt_freq == 0:
                # Save lastest checkpoint after each epoch
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                save_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                }

                torch.save(save_dict, checkpoint_path)

            # save checkpoint of specific epoch
            if args.local_rank == 0 and total_steps % args.save_ckpt_freq == 0:
                print('Save checkpoint at step: %d' % total_steps)
                checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)

                save_dict = {
                    'model': model_without_ddp.state_dict(),
                }

                torch.save(save_dict, checkpoint_path)

            # validation
            if total_steps % args.val_freq == 0:
                val_results = {}

                if 'things' in args.val_dataset:
                    results_dict = validate_things(model_without_ddp,
                                                   max_disp=args.max_disp,
                                                   padding_factor=args.padding_factor,
                                                   inference_size=args.inference_size,
                                                   attn_type=args.attn_type,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   num_reg_refine=args.num_reg_refine,
                                                   )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
                    results_dict = validate_kitti15(model_without_ddp,
                                                    padding_factor=args.padding_factor,
                                                    inference_size=args.inference_size,
                                                    attn_type=args.attn_type,
                                                    attn_splits_list=args.attn_splits_list,
                                                    corr_radius_list=args.corr_radius_list,
                                                    prop_radius_list=args.prop_radius_list,
                                                    num_reg_refine=args.num_reg_refine,
                                                    count_time=args.count_time,
                                                    )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'eth3d' in args.val_dataset:
                    results_dict = validate_eth3d(model_without_ddp,
                                                  padding_factor=args.padding_factor,
                                                  inference_size=args.inference_size,
                                                  attn_type=args.attn_type,
                                                  attn_splits_list=args.attn_splits_list,
                                                  corr_radius_list=args.corr_radius_list,
                                                  prop_radius_list=args.prop_radius_list,
                                                  num_reg_refine=args.num_reg_refine,
                                                  )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'middlebury' in args.val_dataset:
                    results_dict = validate_middlebury(model_without_ddp,
                                                       padding_factor=args.padding_factor,
                                                       inference_size=args.inference_size,
                                                       attn_type=args.attn_type,
                                                       attn_splits_list=args.attn_splits_list,
                                                       corr_radius_list=args.corr_radius_list,
                                                       prop_radius_list=args.prop_radius_list,
                                                       num_reg_refine=args.num_reg_refine,
                                                       resolution=args.middlebury_resolution,
                                                       )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if args.local_rank == 0:
                    # save to tensorboard
                    for key in val_results:
                        tag = key.split('_')[0]
                        tag = tag + '/' + key
                        summary_writer.add_scalar(tag, val_results[key], total_steps)

                    # save validation results to file
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)
                        # order of metrics
                        metrics = ['things_epe', 'things_d1',
                                   'kitti15_epe', 'kitti15_d1', 'kitti15_3px',
                                   'eth3d_epe', 'eth3d_1px',
                                   'middlebury_epe', 'middlebury_2px',
                                   ]

                        eval_metrics = []
                        for metric in metrics:
                            if metric in val_results.keys():
                                eval_metrics.append(metric)

                        metrics_values = [val_results[metric] for metric in eval_metrics]

                        num_metrics = len(eval_metrics)

                        f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                        f.write(("| {:20.4f} " * num_metrics).format(*metrics_values))

                        f.write('\n\n')

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
