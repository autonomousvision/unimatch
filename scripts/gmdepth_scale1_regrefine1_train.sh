#!/usr/bin/env bash

# GMDepth (1/8 feature only), with additional 1 local regression refinement at 1/8 resolution

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
NUM_GPUS=8


# scannet
CHECKPOINT_DIR=checkpoints_depth/scannet-gmdepth-scale1-regrefine1-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_depth.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth \
--no_resume_optimizer \
--dataset scannet \
--val_dataset scannet \
--image_size 480 640 \
--batch_size 64 \
--lr 4e-4 \
--reg_refine \
--num_reg_refine 1 \
--summary_freq 100 \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# demon
CHECKPOINT_DIR=checkpoints_depth/demon-gmdepth-scale1-regrefine1-resumeflowthings && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_depth.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmdepth-scale1-resumeflowthings-demon-a2fe127b.pth \
--no_resume_optimizer \
--dataset demon \
--val_dataset demon \
--demon_split rgbd \
--image_size 448 576 \
--batch_size 64 \
--lr 4e-4 \
--reg_refine \
--num_reg_refine 1 \
--summary_freq 100 \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


