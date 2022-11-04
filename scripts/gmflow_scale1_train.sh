#!/usr/bin/env bash

# basic GMFlow without any refinement (1/8 feature only)

# number of gpus for training, please set according to your hardware
# can be trained on 4x 16GB V100 or 2x 32GB V100 or 2x 40GB A100 gpus
NUM_GPUS=4

# chairs
CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage chairs \
--batch_size 16 \
--val_dataset chairs sintel kitti \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# things (our final model is trained for 800K iterations, for ablation study, you can train for 200K)
CHECKPOINT_DIR=checkpoints_flow/things-gmflow-scale1 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints_flow/chairs-gmflow-scale1/step_100000.pth \
--stage things \
--batch_size 8 \
--val_dataset things sintel kitti \
--lr 2e-4 \
--image_size 384 768 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 40000 \
--save_ckpt_freq 50000 \
--num_steps 800000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# a final note: if your training is terminated unexpectedly, you can resume from the latest checkpoint
# an example: resume chairs training
# CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/chairs-gmflow-scale1/checkpoint_latest.pth \
# --stage chairs \
# --batch_size 16 \
# --val_dataset chairs sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --with_speed_metric \
# --val_freq 10000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


