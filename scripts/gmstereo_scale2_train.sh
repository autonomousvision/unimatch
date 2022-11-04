#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
NUM_GPUS=8

# sceneflow
# resume flow things model
CHECKPOINT_DIR=checkpoints_stereo/sceneflow-gmstereo-scale2-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmflow-scale2-things-36579974.pth \
--no_resume_optimizer \
--stage sceneflow \
--batch_size 32 \
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
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log





