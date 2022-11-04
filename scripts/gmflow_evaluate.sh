#!/usr/bin/env bash


# gmflow-scale1
CUDA_VISIBLE_DEVICES=0 python main_flow.py \
--eval \
--resume pretrained/gmflow-scale1-things-e9887eda.pth \
--val_dataset sintel \
--with_speed_metric


# gmflow-scale2
CUDA_VISIBLE_DEVICES=0 python main_flow.py \
--eval \
--resume pretrained/gmflow-scale2-things-36579974.pth \
--val_dataset kitti \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--with_speed_metric


# gmflow-scale2-regrefine6
CUDA_VISIBLE_DEVICES=0 python main_flow.py \
--eval \
--resume pretrained/gmflow-scale2-regrefine6-things-776ed612.pth \
--val_dataset kitti \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 \
--with_speed_metric




