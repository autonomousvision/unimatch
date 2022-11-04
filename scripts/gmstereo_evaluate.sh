#!/usr/bin/env bash


# gmstereo-scale1
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--eval \
--resume pretrained/gmstereo-scale1-resumeflowthings-sceneflow-16e38788.pth \
--val_dataset kitti15


# gmstereo-scale2
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--eval \
--resume pretrained/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth \
--val_dataset kitti15 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1


# gmstereo-scale2-regrefine3
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--eval \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth \
--val_dataset kitti15 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3


