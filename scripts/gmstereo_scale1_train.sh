#!/usr/bin/env bash

# basic GMStereo without any refinement (1/8 feature only)

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
NUM_GPUS=8

# sceneflow (our final model is trained for 100K steps, for ablation, we train for 50K)
# resume flow things model (our ablations are trained from random init)
CHECKPOINT_DIR=checkpoints_stereo/sceneflow-gmstereo-scale1-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmflow-scale1-things-e9887eda.pth \
--no_resume_optimizer \
--stage sceneflow \
--batch_size 64 \
--val_dataset things kitti15 \
--img_height 384 \
--img_width 768 \
--padding_factor 16 \
--upsample_factor 8 \
--attn_type self_swin2d_cross_1d \
--summary_freq 1000 \
--val_freq 10000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log





