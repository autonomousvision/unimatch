#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)
# with additional 3 local regression refinements

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
NUM_GPUS=8

# sceneflow
# resume gmstereo scale2 model, which is trained from flow things model
CHECKPOINT_DIR=checkpoints_stereo/sceneflow-gmstereo-scale2-regrefine3-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth \
--no_resume_optimizer \
--stage sceneflow \
--lr 4e-4 \
--batch_size 16 \
--val_dataset things kitti15 \
--img_height 384 \
--img_width 768 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# vkitti2
CHECKPOINT_DIR=checkpoints_stereo/vkitti2-gmstereo-scale2-regrefine3-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/sceneflow-gmstereo-scale2-regrefine3-resumeflowthings/step_100000.pth \
--no_resume_optimizer \
--stage vkitti2 \
--val_dataset kitti15 \
--lr 4e-4 \
--batch_size 16 \
--img_height 320 \
--img_width 832 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 5000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 30000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# kitti, this is our final model for kitti submission
CHECKPOINT_DIR=checkpoints_stereo/kitti-gmstereo-scale2-regrefine3-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/vkitti2-gmstereo-scale2-regrefine3-resumeflowthings/step_030000.pth \
--no_resume_optimizer \
--stage kitti15mix \
--val_dataset kitti15 \
--lr 4e-4 \
--batch_size 16 \
--img_height 352 \
--img_width 1216 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 2000 \
--save_ckpt_freq 2000 \
--save_latest_ckpt_freq 1000 \
--num_steps 10000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# middlebury, train on 480x640 first
CHECKPOINT_DIR=checkpoints_stereo/middlebury-gmstereo-scale2-regrefine3-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/sceneflow-gmstereo-scale2-regrefine3-resumeflowthings/step_100000.pth \
--no_resume_optimizer \
--stage middlebury \
--val_dataset middlebury \
--inference_size 768 1024 \
--lr 4e-4 \
--batch_size 16 \
--img_height 480 \
--img_width 640 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# middlebury, finetune on 768x1024 resolution, max disparity range 600 in loss
# this is our final model for middlebury submission
CHECKPOINT_DIR=checkpoints_stereo/middlebury-gmstereo-scale2-regrefine3-resumeflowthings-fthighres && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/middlebury-gmstereo-scale2-regrefine3-resumeflowthings/step_100000.pth \
--no_resume_optimizer \
--max_disp 600 \
--stage middlebury_ft \
--val_dataset middlebury \
--inference_size 1536 2048 \
--lr 4e-4 \
--batch_size 8 \
--img_height 768 \
--img_width 1024 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 5000 \
--save_ckpt_freq 10000 \
--save_latest_ckpt_freq 1000 \
--num_steps 50000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# eth3d
CHECKPOINT_DIR=checkpoints_stereo/eth3d-gmstereo-scale2-regrefine3-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/sceneflow-gmstereo-scale2-regrefine3-resumeflowthings/step_100000.pth \
--no_resume_optimizer \
--stage eth3d \
--val_dataset eth3d \
--lr 4e-4 \
--batch_size 24 \
--img_height 416 \
--img_width 640 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# eth3d, finetune, this is our final model for eth3d submission
CHECKPOINT_DIR=checkpoints_stereo/eth3d-gmstereo-scale2-regrefine3-resumeflowthings-ft && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_stereo/eth3d-gmstereo-scale2-regrefine3-resumeflowthings/step_100000.pth \
--no_resume_optimizer \
--stage eth3d_ft \
--val_dataset eth3d \
--lr 4e-4 \
--batch_size 24 \
--img_height 416 \
--img_width 640 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--summary_freq 100 \
--val_freq 3000 \
--save_ckpt_freq 3000 \
--save_latest_ckpt_freq 1000 \
--num_steps 30000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log























