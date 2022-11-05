import argparse
import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from unimatch.unimatch import UniMatch
from dataloader.depth.datasets import DemonDataset, ScannetDataset
from dataloader.depth import augmentation
from loss.depth_loss import depth_loss_func, depth_grad_loss_func
from evaluate_depth import validate_scannet, validate_demon, inference_depth
from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--dataset', default='scannet', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['scannet'], type=str, nargs='+',
                        help='validation datasets')
    parser.add_argument('--image_size', default=[480, 640], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--demon_split', default='rgbd', type=str)
    parser.add_argument('--eval_min_depth', default=0.5, type=float)
    parser.add_argument('--eval_max_depth', default=10, type=float)
    parser.add_argument('--save_vis_depth', action='store_true')
    parser.add_argument('--count_time', action='store_true')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=5000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=100000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='depth', type=str)
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
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--min_depth', default=0.5, type=float,
                        help='min depth for plane-sweep stereo')
    parser.add_argument('--max_depth', default=10, type=float,
                        help='max depth for plane-sweep stereo')
    parser.add_argument('--num_depth_candidates', default=64, type=int)
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # loss
    parser.add_argument('--depth_loss_weight', default=20, type=float)
    parser.add_argument('--depth_grad_loss_weight', default=20, type=float)

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--depth_from_argmax', action='store_true')
    parser.add_argument('--pred_bidir_depth', action='store_true')

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):
    print_info = not args.eval and args.inference_dir is None
    if args.local_rank == 0 and print_info:
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)

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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_info:
        print('Number of params:', num_params)
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
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

    # evaluation
    if args.eval:
        val_results = {}

        if 'scannet' in args.val_dataset:
            results_dict = validate_scannet(model_without_ddp,
                                            attn_type=args.attn_type,
                                            attn_splits_list=args.attn_splits_list,
                                            prop_radius_list=args.prop_radius_list,
                                            num_reg_refine=args.num_reg_refine,
                                            num_depth_candidates=args.num_depth_candidates,
                                            count_time=args.count_time,
                                            eval_min_depth=args.eval_min_depth,
                                            eval_max_depth=args.eval_max_depth,
                                            min_depth=args.min_depth,
                                            max_depth=args.max_depth,
                                            save_vis_depth=args.save_vis_depth,
                                            save_dir=args.output_path,
                                            )

            val_results.update(results_dict)

            results_str = "\t".join("{}: {:.4f}".format(k, v) for k, v in results_dict.items())
            print(results_str)

        if 'demon' in args.val_dataset:
            results_dict = validate_demon(model_without_ddp,
                                          padding_factor=args.padding_factor,
                                          inference_size=args.inference_size,
                                          attn_type=args.attn_type,
                                          attn_splits_list=args.attn_splits_list,
                                          prop_radius_list=args.prop_radius_list,
                                          num_reg_refine=args.num_reg_refine,
                                          num_depth_candidates=args.num_depth_candidates,
                                          count_time=args.count_time,
                                          eval_min_depth=args.eval_min_depth,
                                          eval_max_depth=args.eval_max_depth,
                                          min_depth=args.min_depth,
                                          max_depth=args.max_depth,
                                          save_vis_depth=args.save_vis_depth,
                                          save_dir=args.output_path,
                                          demon_split=args.demon_split,
                                          debug=args.debug,
                                          )

            val_results.update(results_dict)

            results_str = "\t".join("{}: {:.4f}".format(k, v) for k, v in results_dict.items())
            print(results_str)

        return

    if args.inference_dir:
        inference_depth(model_without_ddp,
                        inference_dir=args.inference_dir,
                        output_path=args.output_path,
                        padding_factor=args.padding_factor,
                        inference_size=args.inference_size,
                        attn_type=args.attn_type,
                        attn_splits_list=args.attn_splits_list,
                        prop_radius_list=args.prop_radius_list,
                        num_depth_candidates=args.num_depth_candidates,
                        num_reg_refine=args.num_reg_refine,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        depth_from_argmax=args.depth_from_argmax,
                        pred_bidir_depth=args.pred_bidir_depth,
                        )

        return

    # build dataset
    train_transform = augmentation.Compose([
        augmentation.RandomColor(),
        augmentation.RandomResize(min_size=args.image_size),
        augmentation.RandomCrop(crop_size=args.image_size),
        augmentation.ToTensor(),
        augmentation.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if args.dataset == 'scannet':
        train_set = ScannetDataset(transforms=train_transform,
                                   mode='train',
                                   )

    elif args.dataset == 'demon':
        train_set = DemonDataset(mode='train',
                                 transforms=train_transform,
                                 )
    else:
        raise NotImplementedError

    # multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    last_epoch = start_step if args.resume and not args.no_resume_optimizer else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 10,
                                                       pct_start=0.05,
                                                       cycle_momentum=False,
                                                       anneal_strategy='cos',
                                                       last_epoch=last_epoch,
                                                       )

    if args.local_rank == 0 and not args.eval:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step,
                        img_mean=IMAGENET_MEAN,
                        img_std=IMAGENET_STD,
                        )

    total_steps = start_step
    epoch = start_epoch
    print('Start training')

    while total_steps < args.num_steps:
        model.train()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img_ref = sample['img_ref'].to(device)
            img_tgt = sample['img_tgt'].to(device)
            intrinsics = sample['intrinsics'].to(device)
            pose = sample['pose'].to(device)  # relative pose, [B, 4, 4]
            gt_depth = sample['depth'].to(device)

            valid_mask = (gt_depth >= args.min_depth) & (gt_depth <= args.max_depth) & \
                         (gt_depth == gt_depth)

            if 'valid' in sample:
                valid_mask = valid_mask * sample['valid'].to(device)  # [B, H, W]

            results_dict = model(img_ref,
                                 img_tgt,
                                 attn_type=args.attn_type,
                                 attn_splits_list=args.attn_splits_list,
                                 prop_radius_list=args.prop_radius_list,
                                 num_reg_refine=args.num_reg_refine,
                                 intrinsics=intrinsics,
                                 pose=pose,
                                 min_depth=1. / args.max_depth,
                                 max_depth=1. / args.min_depth,
                                 num_depth_candidates=args.num_depth_candidates,
                                 task='depth',
                                 )

            depth_preds = results_dict['flow_preds']

            loss = 0

            metrics = {}

            if args.depth_loss_weight > 0:
                depth_loss = depth_loss_func(depth_preds, gt_depth, valid_mask,
                                             gamma=0.9,
                                             )

                loss = loss + args.depth_loss_weight * depth_loss

                # no valid pixel
                if not isinstance(depth_loss, float):
                    metrics.update({'depth_loss': depth_loss.item()})

            if args.depth_grad_loss_weight > 0:
                depth_grad_loss = depth_grad_loss_func(depth_preds, gt_depth,
                                                       valid_mask,
                                                       gamma=0.9)

                loss = loss + args.depth_grad_loss_weight * depth_grad_loss

                # no valid pixel
                if not isinstance(depth_grad_loss, float):
                    metrics.update({'depth_grad_loss': depth_grad_loss.item()})

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics, is_depth=True)

            if args.local_rank == 0:
                logger.add_image_summary(img_ref[0], img_tgt[0],
                                         is_depth=True)

                # visualize inverse depth
                if args.depth_loss_weight > 0:
                    # fill invalid values in gt depth
                    gt_depth_vis = gt_depth[0]

                    if 'valid' in sample:  # sparse gt
                        # inverse is very small
                        gt_depth_vis[valid_mask[0] < 0.5] = 9999999

                    depth_vis = 1. / depth_preds[-1][0]

                    logger.add_depth_summary(depth_vis, 1. / gt_depth_vis)

            total_steps += 1

            if args.local_rank == 0:
                if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                    print('Save checkpoint at step: %d' % total_steps)
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)

                    save_dict = {
                        'model': model_without_ddp.state_dict()
                    }

                    torch.save(save_dict, checkpoint_path)

                if total_steps % args.save_latest_ckpt_freq == 0:
                    # save lastest checkpoint
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                    print('Save latest checkpoint')
                    save_dict = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }

                    torch.save(save_dict, checkpoint_path)

            if total_steps % args.val_freq == 0:
                print('Start validation')

                val_results = {}

                if 'scannet' in args.val_dataset:
                    results_dict = validate_scannet(model_without_ddp,
                                                    padding_factor=args.padding_factor,
                                                    inference_size=args.inference_size,
                                                    attn_type=args.attn_type,
                                                    attn_splits_list=args.attn_splits_list,
                                                    prop_radius_list=args.prop_radius_list,
                                                    num_reg_refine=args.num_reg_refine,
                                                    num_depth_candidates=args.num_depth_candidates,
                                                    count_time=args.count_time,
                                                    eval_min_depth=args.eval_min_depth,
                                                    eval_max_depth=args.eval_max_depth,
                                                    min_depth=args.min_depth,
                                                    max_depth=args.max_depth,
                                                    save_vis_depth=args.save_vis_depth,
                                                    save_dir=args.output_path,
                                                    )

                    print('evaluation results on scannet:')
                    results_str = "\t".join("{}: {:.4f}".format(k, v) for k, v in results_dict.items())
                    print(results_str)

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'demon' in args.val_dataset:
                    results_dict = validate_demon(model_without_ddp,
                                                  padding_factor=args.padding_factor,
                                                  inference_size=args.inference_size,
                                                  attn_type=args.attn_type,
                                                  attn_splits_list=args.attn_splits_list,
                                                  prop_radius_list=args.prop_radius_list,
                                                  num_reg_refine=args.num_reg_refine,
                                                  num_depth_candidates=args.num_depth_candidates,
                                                  count_time=args.count_time,
                                                  eval_min_depth=args.eval_min_depth,
                                                  eval_max_depth=args.eval_max_depth,
                                                  min_depth=args.min_depth,
                                                  max_depth=args.max_depth,
                                                  save_vis_depth=args.save_vis_depth,
                                                  save_dir=args.output_path,
                                                  demon_split=args.demon_split,
                                                  )

                    print('evaluation results on demon %s:' % args.demon_split)
                    results_str = "\t".join("{}: {:.4f}".format(k, v) for k, v in results_dict.items())
                    print(results_str)

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                # save to tensorboard
                for key in val_results:
                    summary_writer.add_scalar(key, val_results[key], total_steps)

                # save validation results to file
                val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                with open(val_file, 'a') as f:
                    f.write('step: %06d\n' % total_steps)
                    # order of metrics
                    metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3',
                               ]

                    eval_metrics = []
                    for metric in metrics:
                        if metric in val_results.keys():
                            eval_metrics.append(metric)

                    metrics_values = [val_results[metric] for metric in eval_metrics]

                    num_metrics = len(eval_metrics)

                    # save as markdown format
                    f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                    f.write(("| {:20.4f} " * num_metrics).format(*metrics_values))

                    f.write('\n\n')

                model.train()

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
